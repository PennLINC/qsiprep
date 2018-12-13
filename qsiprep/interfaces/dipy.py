#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import nibabel as nb
import numpy as np
import os

from tempfile import TemporaryDirectory
from time import time
from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject
)
from nipype.interfaces import ants
from nipype.interfaces.ants.registration import RegistrationInputSpec
from .gradients import concatenate_bvecs, concatenate_bvals, GradientRotation
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel
from ..utils.brainsuite_shore import BrainSuiteShoreModel

LOGGER = logging.getLogger('nipype.interface')


class MedianOtsuInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="b0 template image")
    num_pass = traits.Int(4, usedefault=True, desc='Number of pass of the median filter')
    median_radius = traits.Int(4, usedefault=True,
                               desc='Radius (in voxels) of the applied median filter')
    dilate = traits.Int(6, usedefault=True, desc='Voxels to dilate after masking')


class MedianOtsuOutputSpec(TraitedSpec):
    masked_input = File(exists=True, desc='Masked version of the input image')
    out_mask = File(exists=True, desc='Median-Otsu mask of the input image')


class MedianOtsu(SimpleInterface):
    input_spec = MedianOtsuInputSpec
    output_spec = MedianOtsuOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file

        b0_img = nb.load(in_file)
        b0_data = b0_img.get_fdata()

        masked_data, data_mask = median_otsu(b0_data,
                                             median_radius=self.inputs.median_radius,
                                             numpass=self.inputs.num_pass,
                                             autocrop=False,
                                             dilate=self.inputs.dilate)

        self._results['out_mask'] = fname_presuffix(
            in_file, suffix='_mask', newpath=runtime.cwd)

        self._results['masked_input'] = fname_presuffix(
            in_file, suffix='_brain_masked', newpath=runtime.cwd)

        masked_img = nb.Nifti1Image(masked_data, b0_img.affine, b0_img.header)
        masked_img.to_filename(self._results['masked_input'])

        mask_img = nb.Nifti1Image(data_mask.astype('f8'), b0_img.affine, b0_img.header)
        mask_img.to_filename(self._results['out_mask'])

        return runtime


class HistEQInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to equalize')
    mask_file = File(exists=True, mandatory=True, desc='Mask image')


class HistEQOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='equalized file')


class HistEQ(SimpleInterface):
    input_spec = HistEQInputSpec
    output_spec = HistEQOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file

        uneq_img = nb.load(in_file)
        uneq_data = uneq_img.get_data()

        mask = nb.load(self.inputs.mask_file)
        bool_mask = mask.get_data() > 0
        data_voxels = uneq_data[bool_mask]

        # Do a clip on 2 to 98th percentile
        bottom_2, top_98 = np.percentile(data_voxels, np.array([1, 99]), axis=None)
        clipped_b0 = np.clip(data_voxels, 0, top_98)
        eq_data = histeq(clipped_b0, num_bins=512)
        output = np.zeros_like(mask.get_data())
        output[bool_mask] = eq_data
        eq_img = nb.Nifti1Image(output, uneq_img.affine, uneq_img.header)
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_equalized', newpath=runtime.cwd)
        eq_img.to_filename(self._results['out_file'])
        return runtime


class IdealSignalRegistrationInputSpec(RegistrationInputSpec):
    model_name = traits.Enum("3dSHORE", "MAPMRI", desc='model to generate ideal signal')
    moving_image = InputMultiObject(File(exists=True), desc='list of 3d dwis')
    # fixed_image_masks
    bvals = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    bvecs = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    save_cmd = traits.Bool(True, usedefault=True,
                           desc='write a log of command lines that were applied')
    b0_indices = traits.List(mandatory=True)
    initial_transforms = InputMultiObject(File(exists=True))
    b0_template = File(exists=True)
    mask_image = File(exists=True)
    fixed_image = File(mandatory=False)


class IdealSignalRegistrationOutputSpec(TraitedSpec):
    confounds = File(exists=True)
    transforms = OutputMultiObject(File(exists=True))
    log_cmdline = File(desc='a list of command lines used to apply transforms')
    rotated_bvecs = OutputMultiObject(File(exists=True))
    corrected_images = OutputMultiObject(File(exists=True))

class IdealSignalRegistration(SimpleInterface):
    input_spec = IdealSignalRegistrationInputSpec
    output_spec = IdealSignalRegistrationOutputSpec

    def _run_interface(self, runtime):
        b0_indices = self.inputs.b0_indices
        dwi_files = self.inputs.moving_image
        num_dwis = len(dwi_files)
        # Load the data
        if type(self.inputs.bvecs) is np.ndarray:
            bvecs = self.inputs.bvecs
        else:
            bvecs = concatenate_bvecs(self.inputs.bvecs)
        if type(self.inputs.bvals) is np.ndarray:
            bvals = self.inputs.bvals
        else:
            bvals = concatenate_bvals(self.inputs.bvals)
        mask_img = nb.load(self.inputs.mask_image)
        mask_array = mask_img.get_data() > 0

        images = [nb.load(img) for img in dwi_files]

        # MODEL PART ######################################################
        # Prepare a b0 template
        b0_images = [img for img in images if img not in b0_indices]
        b0_data = np.stack([img.get_data() for img in b0_images], -1)
        b0_mean = b0_data.mean(3)

        # Prepate non-b0 images
        target_indices = [idx for idx in range(num_dwis) if idx not in b0_indices]
        non_b0_images = [images[idx] for idx in target_indices]
        non_b0_image_paths = [dwi_files[idx] for idx in target_indices]
        model_bvecs = np.row_stack([np.zeros(3)] + [bvecs[n] for n in target_indices])
        model_bvals = np.array([0.] + [bvals[n] for n in target_indices])
        model_data = np.stack([b0_mean] + [img.get_data() for img in non_b0_images], -1)

        gtab = gradient_table(bvals=model_bvals, bvecs=model_bvecs)
        if self.inputs.model_name == "MAPMRI":
            model = MapmriModel(gtab)
            LOGGER.info('Fitting MAPMRI model')
            model_fit = model.fit(model_data, mask=mask_array)
            LOGGER.info('Predicting signal')
            predicted_signal = model_fit.fitted_signal()
        elif self.inputs.model_name == "3dSHORE":
            model = BrainSuiteShoreModel(gtab, regularization="L2")
            LOGGER.info('Fitting 3dSHORE model')
            model_fit = model.fit(model_data, mask=mask_array)
            LOGGER.info('Predicting signal')
            predicted_signal = model_fit.fitted_signal()
        else:
            raise Exception("%s model not supported", self.inputs.model_name)
        # Get the predicted signal out

        target_files = []
        for fitnum in range(len(target_indices)):
            output_fname = os.path.join(
                runtime.cwd, 'ideal_signal_%03d.nii.gz' % target_indices[fitnum])
            fit_data = predicted_signal[..., fitnum+1]
            nb.Nifti1Image(fit_data, mask_img.affine, mask_img.header).to_filename(output_fname)
            target_files.append(output_fname)

        # REGISTRATION PART ######################################################
        # Get all inputs from the Registration object
        ifargs = self.inputs.get()

        # Get number of parallel jobs
        num_threads = ifargs.pop('num_threads')
        save_cmd = ifargs.pop('save_cmd')

        # Remove certain keys
        for key in ['environ', 'ignore_exception', 'print_out_composite_warp_file',
                    'terminal_output', 'output_warped_image', 'moving_image', 'model_name',
                    'dwi_files', 'b0_indices', 'b0_template', 'bvals', 'bvecs', 'mask_image',
                    'initial_transforms', 'interpolation', 'fixed_image',
                    't1_2_mni_forward_transform', 'copy_dtype']:
            ifargs.pop(key, None)

        # Get a temp folder ready
        tmp_folder = TemporaryDirectory(prefix='tmp-', dir=runtime.cwd)

        # Inputs are ready to run in parallel
        if num_threads < 1:
            num_threads = None

        if num_threads == 1:
            out_files = [_reg_to_ideal((
                in_file, in_xfm, ifargs, i, runtime.cwd))
                for i, (in_file, in_xfm) in enumerate(zip(non_b0_image_paths, target_files))
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                out_files = list(pool.map(_reg_to_ideal, [
                    (in_file, in_xfm, ifargs, i, runtime.cwd)
                    for i, (in_file, in_xfm) in enumerate(zip(non_b0_image_paths, target_files))]
                ))
        tmp_folder.cleanup()

        # Store the warped images
        resampled_images = [el[0] for el in out_files]
        self._results['corrected_images'] = resampled_images

        # Add back in the b0s, return the full transform list
        recombined_transforms = [''] * num_dwis
        for b0_index in self.inputs.b0_indices:
            recombined_transforms[b0_index] = self.inputs.initial_transforms[b0_index]
        dwi_transforms = [el[1] for el in out_files]
        for target_index, target_transform in zip(target_indices, dwi_transforms):
            recombined_transforms[target_index] = target_transform
        self._results['transforms'] = recombined_transforms

        rotator = GradientRotation(affine_transforms=recombined_transforms,
                                   bvec_files=self.inputs.bvecs,
                                   bval_files=self.inputs.bvals)
        self._results['rotated_bvecs'] = rotator.run().results['bvecs']

        if save_cmd:
            self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
            with open(self._results['log_cmdline'], 'w') as cmdfile:
                print('\n-------\n'.join([el[2] for el in out_files]), file=cmdfile)

        return runtime


def _reg_to_ideal(args):
    """Create a composite transform from inputs."""
    in_file, in_target, ifargs, index, newpath = args
    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)

    xfm = ants.Registration(
        moving_image=in_file, fixed_image=in_target, output_warped_image=out_file,
        interpolation='LanczosWindowedSinc', **ifargs)
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    run = xfm.run()
    runtime = run.runtime
    LOGGER.info(runtime.cmdline)

    # Force floating point precision
    nii = nb.load(out_file)
    nii.set_data_dtype(np.dtype('float32'))
    nii.to_filename(out_file)

    return (out_file, run.outputs.forward_transforms, runtime.cmdline)
