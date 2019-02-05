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
from dipy.core.geometry import cart2sphere
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis

LOGGER = logging.getLogger('nipype.interface')
from .converters import get_dsi_studio_ODF_geometry
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


class BrainSuiteShoreReconstructionInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    local_bvec_file = File(exists=True)
    dwi_file = File(exists=True, mandatory=True)
    radial_order = traits.Int(6, usedefault=True)
    zeta = traits.Float(700)
    tau = traits.Float(4 * np.pi**2)
    regularization = traits.Enum("L1", "L2")
    big_delta = traits.Float()
    little_delta = traits.Float()
    b0_threshold = traits.CFloat(50)
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
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)


class BrainSuiteShoreReconstructionOutputSpec(TraitedSpec):
    shore_coeffs = File()
    rtop = File()


class BrainSuiteShoreReconstruction(SimpleInterface):
    input_spec = BrainSuiteShoreReconstructionInputSpec
    output_spec = BrainSuiteShoreReconstructionOutputSpec

    def _run_interface(self,runtime):
        gtab = gradient_table(bvals=np.loadtxt(self.inputs.bval_file),
                              bvecs=np.loatxt(self.inputs.bvec_file).T,
                              b0_threshold=self.inputs.b0_threshold)
        b0s_mask = gtab.b0s_mask
        dwis_mask = np.logical_not(b0s_mask)
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype=np.float32)
        b0_images = dwi_data[..., b0s_mask]
        dwi_images = dwi_data[..., dwis_mask]

        mask_img = nb.load(self.inputs.mask_file)
        mask_array = mask_img.get_data() > 0

        b0_mean =  b0_images.mean(3)
        final_bvals = np.concatenate([np.array([0]), gtab.bvals[dwis_mask]])
        final_bvecs = np.row_stack([np.array([0., 0., 0.]), gtab.bvecs[dwis_mask]])
        final_data = np.concatenate([b0_mean[...,np.newaxis], dwi_images], 3)
        final_grads = gradient_table(bvals=final_bvals, bvecs=final_bvecs, b0_threshold=5)

        # Cleanup
        del dwi_images
        del b0_images
        del dwi_data

        bss_model = BrainSuiteShoreModel(final_grads, regularization="L2")
        bss_fit = bss_model.fit(final_data, mask=mask_array)
        rtop = bss_fit.rtop_signal()
        coeffs = bss_fit.shore_coeff

        coeffs_file = fname_presuffix(self.inputs_dwi_file, suffix="_shore_coeff",
                                      newpath=runtime.cwd)
        rtop_file = fname_presuffix(self.inputs_dwi_file, suffix="_rtop",
                                      newpath=runtime.cwd)
        nb.Nifti1Image(coeffs, dwi_img.affine, dwi_img.header).to_filename(coeffs_file)
        nb.Nifti1Image(rtop, dwi_img.affine, dwi_img.header).to_filename(rtop_file)
        self._results['shore_coeffs'] = coeffs_file
        self._results['rtop'] = rtop_file

        if not (self.inputs.write_fibgz or self.inputs.write_mif):
            return runtime

        # Convert to amplitudes for other software
        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        hs = HemiSphere(x=x, y=y, z=z)
        odf_amplitudes = bss_fit.odf(hs)
        if self.inputs.write_fibgz:
            output_fib_file = fname_presuffix(self.inputs_dwi_file, suffix="_3dSHORE.fib",
                                              newpath=runtime.cwd, use_ext=False)
            fit_to_fibgz(odf_amplitudes, hs, verts, faces, output_fib_file)
        if self.inputs.write_mif:
            output_mif_file = fname_presuffix(self.inputs_dwi_file, suffix="_3dSHORE.mif",
                                              newpath=runtime.cwd, use_ext=False)
            fit_to_mif()

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


def fit_to_mif(odf_amplitudes, dwi_img, sphere, output_mif_file, runtime):
    x, y, z = sphere.vertices.T
    _, theta, phi = cart2sphere(-x, -y, z)
    dirs_txt = op.join(runtime.cwd, "ras+directions.txt")
    tmp_nii = op.join(runtime.cwd, "amps.nii")
    nb.Nifti1Image(odf_amplitudes, dwi_img.affine, dwi_img.header).to_filename(tmp_nii)
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))
    popen_run(["amp2sh", "-force", "-directions", dirs_txt, tmp_nii, output_mif_file])
    os.remove(tmp_nii)


def fit_to_fibgz(odf_amplitudes, sphere, verts, faces, output_fib_file, n_fibers=5):
    """Write a dipy fit object to a DSI Studio fib file.
    """
    ampl_mask = np.abs(odf_amplitudes.sum(3)) > 1e-6
    flat_mask = ampl_mask.flatten(order="F")
    odf_array = odf_amplitudes.reshape(-1, odf_amplitudes.shape[3], order="F")
    masked_odfs = odf_array[flat_mask, :]
    del odf_array
    z0 = masked_odfs.max()
    masked_odfs = masked_odfs/z0
    n_odfs = masked_odfs.shape[0]
    peak_indices = np.zeros((n_odfs, n_fibers))
    peak_vals = np.zeros((n_odfs, n_fibers))

    dsi_mat = {}
    # Create matfile that can be read by dsi Studio
    dsi_mat['dimension'] = np.array(amplitudes_img.shape[:3])
    dsi_mat['voxel_size'] = np.array(amplitudes_img.header.get_zooms()[:3])
    n_voxels = int(np.prod(dsi_mat['dimension']))
    #LOGGER.info("Detecting Peaks")
    for odfnum in tqdm(range(masked_odfs.shape[0])):
        dirs, vals, indices = peak_directions(masked_odfs[odfnum], hs)
        for dirnum, (val, idx) in enumerate(zip(vals, indices)):
            if dirnum == n_fibers:
                break
            peak_indices[odfnum, dirnum] = idx
            peak_vals[odfnum, dirnum] = val

    for nfib in range(n_fibers):
        # fill in the "fa" values
        fa_n = np.zeros(n_voxels)
        fa_n[flat_mask] = peak_vals[:, nfib]
        dsi_mat['fa%d' % nfib] = fa_n.astype(np.float32)

        # Fill in the index values
        index_n = np.zeros(n_voxels)
        index_n[flat_mask] = peak_indices[:, nfib]
        dsi_mat['index%d' % nfib] = index_n.astype(np.int16)

    # Add in the ODFs
    num_odf_matrices = n_odfs // ODF_COLS
    split_indices = (np.arange(num_odf_matrices) + 1) * ODF_COLS
    odf_splits = np.array_split(masked_odfs, split_indices, axis=0)
    for splitnum, odfs in enumerate(odf_splits):
        dsi_mat['odf%d' % splitnum] = odfs.T.astype(np.float32)

    dsi_mat['odf_vertices'] = verts.T
    dsi_mat['odf_faces'] = faces.T
    dsi_mat['z0'] = np.array([1.])
    savemat(output_fib_file, dsi_mat, format='4', appendmat=False)


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
