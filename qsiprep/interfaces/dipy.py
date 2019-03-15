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
import os.path as op

from tempfile import TemporaryDirectory
from time import time
from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu
from dipy.direction import peak_directions
from dipy.core.sphere import HemiSphere

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject, isdefined
)
from nipype.interfaces import ants
from nipype.interfaces.ants.registration import RegistrationInputSpec
from .gradients import concatenate_bvecs, concatenate_bvals, GradientRotation
from dipy.core.gradients import gradient_table
from dipy.reconst import mapmri
from dipy.core.geometry import cart2sphere
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis

LOGGER = logging.getLogger('nipype.interface')
from .converters import get_dsi_studio_ODF_geometry, amplitudes_to_fibgz, amplitudes_to_sh_mif
import subprocess


def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    print(out)
    print(err)

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


class DipyReconInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    local_bvec_file = File(exists=True)
    big_delta = traits.Float()
    little_delta = traits.Float()
    b0_threshold = traits.CFloat(50, usedefault=True)
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)

class DipyReconOutputSpec(TraitedSpec):
    fibgz = File()
    fod_sh_mif = File()


class DipyReconInterface(SimpleInterface):
    input_spec = DipyReconInputSpec
    output_spec = DipyReconOutputSpec

    def _get_gtab(self):
        little_delta = self.inputs.little_delta if isdefined(self.inputs.little_delta) else None
        big_delta = self.inputs.big_delta if isdefined(self.inputs.big_delta) else None
        gtab = gradient_table(bvals=np.loadtxt(self.inputs.bval_file),
                              bvecs=np.loadtxt(self.inputs.bvec_file).T,
                              b0_threshold=self.inputs.b0_threshold,
                              big_delta=big_delta,
                              small_delta=little_delta)
        return gtab

    def _get_mask(self, amplitudes_img, gtab):
        if not isdefined(self.inputs.mask_file):
            dwi_data = amplitudes_img.get_data()
            LOGGER.warning("Creating an Otsu mask, check that the whole brain is covered.")
            _, mask_array = median_otsu(dwi_data[..., gtab.b0s_mask], 3, 2)
            # Needed for synthetic data
            mask_array = mask_array * (dwi_data.sum(3) > 0)
            mask_img = nb.Nifti1Image(mask_array.astype(np.float32), amplitudes_img.affine,
                                      amplitudes_img.header)
        else:
            mask_img = nb.load(self.inputs.mask_file)
            mask_array = mask_img.get_data() > 0
        return mask_img, mask_array

    def _save_scalar(self, data, suffix, runtime, ref_img):
        output_fname = fname_presuffix(self.inputs.dwi_file, suffix=suffix,
                                       newpath=runtime.cwd)
        nb.Nifti1Image(data, ref_img.affine, ref_img.header).to_filename(output_fname)
        return output_fname

    def _write_external_formats(self, runtime, fit_obj, mask_img, suffix):

        if not (self.inputs.write_fibgz or self.inputs.write_mif):
            return

        # Convert to amplitudes for other software
        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        hs = HemiSphere(x=x, y=y, z=z)
        odf_amplitudes = nb.Nifti1Image(fit_obj.odf(hs), mask_img.affine, mask_img.header)

        if self.inputs.write_fibgz:
            output_fib_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+".fib",
                                              newpath=runtime.cwd, use_ext=False)
            LOGGER.info("Writing DSI Studio fib file %s", output_fib_file)
            amplitudes_to_fibgz(odf_amplitudes, verts, faces, output_fib_file, mask_img,
                                num_fibers=5)
            self._results['fibgz'] = output_fib_file

        if self.inputs.write_mif:
            output_mif_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+".mif",
                                              newpath=runtime.cwd, use_ext=False)
            LOGGER.info("Writing sh mif file %s", output_mif_file)
            amplitudes_to_sh_mif(odf_amplitudes, verts, output_mif_file, runtime.cwd)
            self._results['fod_sh_mif'] = output_mif_file


class MAPMRIInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    laplacian_regularization = traits.Bool(True, usedefault=True)
    laplacian_weighting = traits.Either(traits.Str("GCV"),
                                        traits.Float(0.2),
                                        usedefault=True)
    positivity_constraint = traits.Bool(False, usedefault=True)
    pos_grid = traits.Int(15, usedefault=True)
    pos_radius = traits.Either(traits.Str('adaptive'), traits.Int(),
                               usedefault=True)
    anisotropic_scaling = traits.Bool(True, usedefault=True)
    eigenvalue_threshold = traits.Float(1e-04, usedefault=True)
    bval_threshold = traits.Float()
    dti_scale_estimation = traits.Bool(True, usedefault=True)
    static_diffusivity = traits.Float(0.7e-3, usedefault=True)
    cvxpy_solver = traits.Str()


class MAPMRIOutputSpec(DipyReconOutputSpec):
    rtop = File()
    lapnorm = File()
    msd = File()
    qiv = File()
    rtap = File()
    rtpp = File()
    ng = File()
    perng = File()
    parng = File()
    mapmri_coeffs = File()


class MAPMRIReconstruction(DipyReconInterface):
    input_spec = MAPMRIInputSpec
    output_spec = MAPMRIOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        data = dwi_img.get_fdata(dtype=np.float32)
        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        if self.inputs.laplacian_regularization and \
           self.inputs.positivity_constraint:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=True,
                laplacian_weighting=self.inputs.laplacian_weighting,
                positivity_constraint=True,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling)

        elif self.inputs.positivity_constraint:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=False,
                positivity_constraint=True,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling)

        elif self.inputs.laplacian_regularization:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=True,
                laplacian_weighting=self.inputs.laplacian_weighting,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling)

        else:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=False,
                positivity_constraint=False,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling)

        LOGGER.info("Fitting MAPMRI Model.")
        mapfit_aniso = map_model_aniso.fit(data, mask=mask_array)
        rtop = mapfit_aniso.rtop()
        self._results['rtop'] = self._save_scalar(rtop, "_rtop", runtime, dwi_img)

        ll = mapfit_aniso.norm_of_laplacian_signal()
        self._results['lapnorm'] = self._save_scalar(ll, "_lapnorm", runtime, dwi_img)

        m = mapfit_aniso.msd()
        self._results['msd'] = self._save_scalar(m, "_msd", runtime, dwi_img)

        q = mapfit_aniso.qiv()
        self._results['qiv'] = self._save_scalar(q, "_qiv", runtime, dwi_img)

        rtap = mapfit_aniso.rtap()
        self._results['rtap'] = self._save_scalar(q, "_rtap", runtime, dwi_img)

        rtpp = mapfit_aniso.rtpp()
        self._results['rtpp'] = self._save_scalar(q, "_rtpp", runtime, dwi_img)

        coeffs = mapfit_aniso.mapmri_coeff
        self._results['mapmri_coeffs'] = self._save_scalar(coeffs, "_mapcoeffs", runtime, dwi_img)
        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, mapfit_aniso, mask_img, "_MAPMRI")

        return runtime


class BrainSuiteShoreReconstructionInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    zeta = traits.Float(700)
    tau = traits.Float(4 * np.pi**2, usedefault=True)
    regularization = traits.Enum("L2", "L1", usedefault=True)
    # For L2
    lambdaN = traits.Float(1e-8)
    lambdaL = traits.Float(1e-8)
    # For L1
    regularization_weighting = traits.Str("CV", usedefault=True)
    l1_positive_constraint = traits.Bool(False, usedefault=True)
    l1_cv = traits.Either(None, traits.Str())
    l1_maxiter = traits.Int(1000, usedefault=True)
    l1_verbose = traits.Bool(False, usedefault=True)
    l1_alpha = traits.Float(1.0, usedefault=True)
    # For EAP
    pos_grid = traits.Int(11, usedefault=True)
    pos_radius = traits.Float(20e-03, usedefault=True)


class BrainSuiteShoreReconstructionOutputSpec(DipyReconOutputSpec):
    shore_coeffs = File()
    rtop = File()


class BrainSuiteShoreReconstruction(DipyReconInterface):
    input_spec = BrainSuiteShoreReconstructionInputSpec
    output_spec = BrainSuiteShoreReconstructionOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        b0s_mask = gtab.b0s_mask
        dwis_mask = np.logical_not(b0s_mask)
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype=np.float32)
        b0_images = dwi_data[..., b0s_mask]
        b0_mean = b0_images.mean(3)
        dwi_images = dwi_data[..., dwis_mask]

        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        final_bvals = np.concatenate([np.array([0]), gtab.bvals[dwis_mask]])
        final_bvecs = np.row_stack([np.array([0., 0., 0.]), gtab.bvecs[dwis_mask]])
        final_data = np.concatenate([b0_mean[..., np.newaxis], dwi_images], 3)
        final_grads = gradient_table(bvals=final_bvals, bvecs=final_bvecs, b0_threshold=5)

        # Cleanup
        del dwi_images
        del b0_images
        del dwi_data

        bss_model = BrainSuiteShoreModel(
            final_grads,
            regularization=self.inputs.regularization,
            radial_order=self.inputs.radial_order,
            zeta=self.inputs.zeta,
            tau=self.inputs.tau,
            # For L2
            lambdaN=self.inputs.lambdaN,
            lambdaL=self.inputs.lambdaL,
            # For L1
            regularization_weighting=self.inputs.regularization_weighting,
            l1_positive_constraint=traits.Bool(False, usedefault=True),
            l1_cv=self.inputs.l1_cv,
            l1_maxiter=self.inputs.l1_maxiter,
            l1_verbose=self.inputs.l1_verbose,
            l1_alpha=self.inputs.l1_alpha,
            # For EAP
            pos_grid=self.inputs.pos_grid
            )
        bss_fit = bss_model.fit(final_data, mask=mask_array)
        rtop = bss_fit.rtop_signal()
        coeffs = bss_fit.shore_coeff

        coeffs_file = fname_presuffix(self.inputs.dwi_file, suffix="_shore_coeff",
                                      newpath=runtime.cwd)
        rtop_file = fname_presuffix(self.inputs.dwi_file, suffix="_rtop",
                                    newpath=runtime.cwd)
        nb.Nifti1Image(coeffs, dwi_img.affine, dwi_img.header).to_filename(coeffs_file)
        nb.Nifti1Image(rtop, dwi_img.affine, dwi_img.header).to_filename(rtop_file)
        self._results['shore_coeffs'] = coeffs_file
        self._results['rtop'] = rtop_file

        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, bss_fit, mask_img, "_BS3dSHORE")
        return runtime


class IdealSignalRegistrationInputSpec(RegistrationInputSpec):
    model_name = traits.Enum("3dSHORE", "MAPMRI", desc='model to generate ideal signal')
    moving_image = InputMultiObject(File(exists=True), desc='list of raw 3d dwis')
    last_iter_images = InputMultiObject(File(exists=True), desc='list of just-corrected dwis')
    # fixed_image_masks
    bvals = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    bvecs = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    save_cmd = traits.Bool(True, usedefault=True,
                           desc='write a log of command lines that were applied')
    b0_indices = traits.List(mandatory=True)
    initial_transforms = InputMultiObject(File(exists=True),
                                          desc='all b0-based initial transforms')
    mask_image = File(exists=True)
    fixed_image = File(mandatory=False)


class IdealSignalRegistrationOutputSpec(TraitedSpec):
    transforms = OutputMultiObject(File(exists=True))
    log_cmdline = File(desc='a list of command lines used to apply transforms')
    rotated_bvecs = OutputMultiObject(File(exists=True))
    corrected_images = OutputMultiObject(File(exists=True))
    ideal_images = OutputMultiObject(File(exists=True))


class IdealSignalRegistration(SimpleInterface):
    input_spec = IdealSignalRegistrationInputSpec
    output_spec = IdealSignalRegistrationOutputSpec

    def _run_interface(self, runtime):
        b0_indices = self.inputs.b0_indices
        raw_dwi_files = self.inputs.moving_image
        last_iter_images = self.inputs.last_iter_images
        num_dwis = len(raw_dwi_files)
        # Load the data
        if type(self.inputs.bvecs) is np.ndarray:
            bvecs = self.inputs.bvecs
        else:
            bvecs = concatenate_bvecs(self.inputs.bvecs)
        if type(self.inputs.bvals) is np.ndarray:
            bvals = self.inputs.bvals
        else:
            bvals = concatenate_bvals(self.inputs.bvals, None)
        mask_img = nb.load(self.inputs.mask_image)
        mask_array = mask_img.get_data() > 0

        # images = [nb.load(img) for img in raw_dwi_files]

        # MODEL PART ######################################################
        # Prepare a b0 template
        b0_image_paths = [img for img_num, img in enumerate(last_iter_images)
                          if img_num in b0_indices]
        b0_mean = np.stack([nb.load(img).get_data() for img in b0_image_paths], -1).mean(3)

        # Prepate non-b0 images
        target_indices = [idx for idx in range(num_dwis) if idx not in b0_indices]
        non_b0_image_paths = [last_iter_images[idx] for idx in target_indices]
        model_bvecs = np.row_stack([np.zeros(3)] + [bvecs[n] for n in target_indices])
        model_bvals = np.array([0.] + [bvals[n] for n in target_indices])
        model_data = np.stack([b0_mean] +
                              [nb.load(img).get_data().astype(np.float32) for img in
                              non_b0_image_paths], -1)  # 64bit precision is too much here

        gtab = gradient_table(bvals=model_bvals, bvecs=model_bvecs)
        predicted_signal = estimate_signal_targets(self.inputs.model_name, gtab, model_data,
                                                   mask_array)
        target_files = []
        for fitnum, fit_data in zip(target_indices, predicted_signal):
            output_fname = os.path.join(
                runtime.cwd, 'ideal_signal_%03d.nii.gz' % fitnum)
            nb.Nifti1Image(fit_data, mask_img.affine, mask_img.header).to_filename(output_fname)
            target_files.append(output_fname)
        b0_mean_file = os.path.join(runtime.cwd, 'b0_average.nii.gz')
        nb.Nifti1Image(b0_mean, mask_img.affine, mask_img.header).to_filename(b0_mean_file)
        ideal_images = [''] * num_dwis
        for b0_index in self.inputs.b0_indices:
            ideal_images[b0_index] = b0_mean_file
        for target_index, target_file in zip(target_indices, target_files):
            ideal_images[target_index] = target_file
        self._results['ideal_images'] = ideal_images

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
                    'initial_transforms', 'interpolation', 'fixed_image', 'last_iter_images',
                    'output_transform_prefix', 't1_2_mni_forward_transform', 'copy_dtype']:
            ifargs.pop(key, None)

        # Get a temp folder ready
        tmp_folder = TemporaryDirectory(prefix='tmp-', dir=runtime.cwd)

        # Inputs are ready to run in parallel
        if num_threads < 1:
            num_threads = None

        non_b0_raw_image_paths = [raw_dwi_files[idx] for idx in target_indices]
        if num_threads == 1:
            out_files = [_reg_to_ideal((
                in_file, in_target, index, ifargs, runtime.cwd))
                for in_file, in_target, index in zip(non_b0_raw_image_paths, target_files,
                                                     target_indices)
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                out_files = list(pool.map(_reg_to_ideal, [
                    (in_file, in_target, index, ifargs, runtime.cwd)
                    for in_file, in_target, index in zip(non_b0_raw_image_paths, target_files,
                                                         target_indices)]
                ))
        tmp_folder.cleanup()

        # Store the warped images
        _resampled_images = [el[0] for el in out_files]

        # Add back in the b0s, return the full transform list and warped images
        recombined_transforms = [''] * num_dwis
        resampled_images = [''] * num_dwis
        for b0_index, b0_image in zip(self.inputs.b0_indices, b0_image_paths):
            recombined_transforms[b0_index] = self.inputs.initial_transforms[b0_index]
            resampled_images[b0_index] = b0_image
        dwi_transforms = [el[1] for el in out_files]
        for target_index, target_transform, _resampled_image in zip(target_indices,
                                                                    dwi_transforms,
                                                                    _resampled_images):
            recombined_transforms[target_index] = target_transform if type(target_transform) \
                is str else target_transform[0]
            resampled_images[target_index] = _resampled_image
        self._results['transforms'] = recombined_transforms
        self._results['corrected_images'] = resampled_images

        rotator = GradientRotation(affine_transforms=recombined_transforms,
                                   bvec_files=self.inputs.bvecs,
                                   bval_files=self.inputs.bvals)
        self._results['rotated_bvecs'] = rotator.run().outputs.bvecs

        if save_cmd:
            self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
            with open(self._results['log_cmdline'], 'w') as cmdfile:
                print('\n-------\n'.join([el[2] for el in out_files]), file=cmdfile)

        return runtime


def _reg_to_ideal(args):
    """Create a composite transform from inputs."""
    in_file, in_target, index, ifargs, newpath = args
    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)
    out_transform = fname_presuffix(in_file, suffix='_xform-%05d_Affine.mat' % index,
                                    newpath=newpath, use_ext=False)

    xfm = ants.Registration(
        moving_image=in_file, fixed_image=in_target, output_warped_image=out_file,
        interpolation='BSpline', output_transform_prefix=out_transform,
        **ifargs)
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    run = xfm.run()
    runtime = run.runtime
    LOGGER.info(runtime.cmdline)

    # Force floating point precision
    nii = nb.load(out_file, mmap=False)
    nii.set_data_dtype(np.dtype('float32'))
    nii.to_filename(out_file)

    return (out_file, run.outputs.forward_transforms, runtime.cmdline)


def estimate_signal_targets(model_name, gtab, model_data, mask_array, cutoff=50):
    def get_model(grads):
        if model_name == "MAPMRI":
            return MapmriModel(grads)
        elif model_name == "3dSHORE":
            return BrainSuiteShoreModel(grads, regularization="L2")

    full_bvals = gtab.bvals
    full_bvecs = gtab.bvecs

    num_grads = gtab.gradients.shape[0]
    LOGGER.info('Fitting 3dSHORE model')
    full_model = get_model(gtab)
    scaled_gradients = full_bvals[:, np.newaxis] * full_bvecs
    phi = brainsuite_shore_basis(full_model.radial_order, full_model.zeta, gtab, full_model.tau)
    predictions = []
    for loo in range(1, num_grads):
        # The model is symmetric, so remove any samples that are too close or the opposite
        distances = np.sqrt(np.sum((scaled_gradients - scaled_gradients[loo]) ** 2, axis=1))
        distances_flip = np.sqrt(np.sum((-scaled_gradients + scaled_gradients[loo]) ** 2, axis=1))
        training_mask = (distances > cutoff) * (distances_flip > cutoff)

        # Extract training images and gradients
        training_bvals = full_bvals[training_mask]
        training_bvecs = full_bvecs[training_mask]
        training_gtab = gradient_table(bvals=training_bvals, bvecs=training_bvecs)
        training_data = model_data[..., training_mask]

        LOGGER.info("\tleaving out %d: training with %d samples", loo, training_mask.sum())
        iter_model = get_model(training_gtab)
        iter_fit = iter_model.fit(training_data, mask=mask_array)

        shore_array = iter_fit._shore_coef[mask_array]
        this_dir = phi[loo]
        output_data = np.zeros(mask_array.shape)
        output_data[mask_array] = np.dot(shore_array, this_dir)
        predictions.append(output_data)
    return predictions
