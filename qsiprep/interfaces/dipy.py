#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import subprocess
import shutil
from pkg_resources import resource_filename as pkgr
import nibabel as nb
import numpy as np
from dipy.io.utils import nifti1_symmat
from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu
from dipy.core.sphere import HemiSphere
from dipy.core.gradients import gradient_table
from dipy.reconst import mapmri, dti
from nilearn.image import load_img
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, isdefined
)
from .patch2self import patch2self
from .denoise import (SeriesPreprocReport, SeriesPreprocReportInputSpec,
                      SeriesPreprocReportOutputSpec)
from .converters import get_dsi_studio_ODF_geometry, amplitudes_to_fibgz, amplitudes_to_sh_mif
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from ..interfaces.mrtrix import _convert_fsl_to_mrtrix

LOGGER = logging.getLogger('nipype.interface')
TAU_DEFAULT = 1. / (4 * np.pi**2)


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

        mask_img = nb.Nifti1Image(data_mask.astype('f8'), b0_img.affine,
                                  b0_img.header)
        mask_img.to_filename(self._results['out_mask'])

        return runtime


class Patch2SelfInputSpec(SeriesPreprocReportInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc="4D diffusion MRI data file")
    patch_radius = traits.Either(traits.Int(0), traits.Str('auto'),
                                 desc='patch radius in voxels.')
    bval_file = File(exists=True, mandatory=True,
                     desc="bval file containing b-values")
    model = traits.Str('ridge', usedefault=True,
                       desc='Regression model for Patch2Self')
    alpha = traits.Float(1., usedefault=True,
                         desc='Regularization parameter for Ridge and Lasso')
    b0_threshold = traits.Float(50., usedefault=True,
                                desc='Threshold to segregate b0s')
    mask = File(desc='mask image (unused)')
    b0_denoising = traits.Bool(True, usedefault=True,
                               desc='denoise the b=0 images too')
    clip_negative_vals = traits.Bool(False, usedefault=True,
                                     desc='Sets negative values after denoising to 0')
    shift_intensity = traits.Bool(True, usedefault=True,
                                  desc='Shifts the distribution of intensities per '
                                  'volume to give non-negative values')
    out_report = File('patch2self_report.svg', usedefault=True,
                      desc='filename for the visual report')


class Patch2SelfOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(exists=True,
                    desc='Denoised version of the input image')
    noise_image = File(exists=True,
                       desc='Residuals depicting suppressed noise')


class Patch2Self(SeriesPreprocReport, SimpleInterface):
    input_spec = Patch2SelfInputSpec
    output_spec = Patch2SelfOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file
        bval_file = self.inputs.bval_file
        denoised_file = fname_presuffix(
            in_file, suffix='_denoised_patch2self', newpath=runtime.cwd)
        noise_file = fname_presuffix(
            in_file, suffix='_denoised_residuals_patch2self',
            newpath=runtime.cwd)
        noisy_img = nb.load(in_file)
        noisy_arr = noisy_img.get_fdata()
        bvals = np.loadtxt(bval_file)

        # Determine the patch radius
        num_non_b0 = (bvals > self.inputs.b0_threshold).sum()
        very_few_directions = num_non_b0 < 20
        few_directions = num_non_b0 < 50
        if self.inputs.patch_radius == 'auto':
            if very_few_directions:
                patch_radius = [3, 3, 3]
            elif few_directions:
                patch_radius = [1, 1, 1]
            else:
                patch_radius = [0, 0, 0]
        else:
            patch_radius = [self.inputs.patch_radius] * 3
            if self.inputs.patch_radius > 3 and not very_few_directions:
                LOGGER.info("a very large patch radius is not necessary when more than "
                            "20 gradient directions have been sampled.")
            elif self.inputs.patch_radius > 1 and not few_directions:
                LOGGER.info("a large patch radius is not necessary when more than "
                            "50 gradient directions have been sampled.")
            elif self.inputs.patch_radius == 0 and few_directions:
                LOGGER.warning("When < 50 gradient directions are available, it is "
                               "recommended to increase patch_radius to > 0.")

        denoised_arr, noise_residuals = \
            patch2self(noisy_arr, bvals,
                       model=self.inputs.model,
                       alpha=self.inputs.alpha,
                       patch_radius=patch_radius,
                       b0_threshold=self.inputs.b0_threshold,
                       verbose=True,
                       b0_denoising=self.inputs.b0_denoising,
                       clip_negative_vals=self.inputs.clip_negative_vals,
                       shift_intensity=self.inputs.shift_intensity
                       )

        # Back to nifti
        denoised_img = nb.Nifti1Image(denoised_arr, noisy_img.affine,
                                      noisy_img.header)
        p2s_residuals = nb.Nifti1Image(noise_residuals, noisy_img.affine,
                                       noisy_img.header)
        denoised_img.to_filename(denoised_file)
        p2s_residuals.to_filename(noise_file)
        self._results['out_file'] = denoised_file
        self._results['noise_image'] = noise_file
        return runtime

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        noise_name = outputs['noise_image']
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


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
        bottom_2, top_98 = np.percentile(data_voxels, np.array([1, 99]),
                                         axis=None)
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
    big_delta = traits.Either(None, traits.Float(), usedefault=True)
    little_delta = traits.Either(None, traits.Float(), usedefault=True)
    b0_threshold = traits.CFloat(50, usedefault=True)
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)
    # To extrapolate
    extrapolate_scheme = traits.Enum('HCP', 'ABCD')


class DipyReconOutputSpec(TraitedSpec):
    fibgz = File()
    fod_sh_mif = File()
    extrapolated_dwi = File()
    extrapolated_bvals = File()
    extrapolated_bvecs = File()
    extrapolated_b = File()
    odf_amplitudes = File()
    odf_directions = File()


class DipyReconInterface(SimpleInterface):
    input_spec = DipyReconInputSpec
    output_spec = DipyReconOutputSpec

    def _get_gtab(self, external_bvals=None, external_bvecs=None):
        little_delta = self.inputs.little_delta if isdefined(self.inputs.little_delta) else None
        big_delta = self.inputs.big_delta if isdefined(self.inputs.big_delta) else None
        bval_file = self.inputs.bval_file if external_bvals is None else external_bvals
        bvec_file = self.inputs.bvec_file if external_bvecs is None else external_bvecs
        gtab = gradient_table(bvals=np.loadtxt(bval_file),
                              bvecs=np.loadtxt(bvec_file).T,
                              b0_threshold=self.inputs.b0_threshold,
                              big_delta=big_delta,
                              small_delta=little_delta)
        return gtab

    def _get_mask(self, amplitudes_img, gtab):
        if not isdefined(self.inputs.mask_file):
            dwi_data = amplitudes_img.get_data()
            LOGGER.warning("Creating an Otsu mask, check that the whole brain is covered.")
            _, mask_array = median_otsu(dwi_data,
                                        vol_idx=gtab.b0s_mask,
                                        median_radius=3,
                                        numpass=2)

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
        output_amps_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+"_amp.nii.gz",
                                           newpath=runtime.cwd, use_ext=False)
        output_dirs_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+"_dirs.npy",
                                           newpath=runtime.cwd, use_ext=False)
        odf_amplitudes.to_filename(output_amps_file)
        np.save(output_dirs_file, verts[:hemisphere])
        self._results['odf_amplitudes'] = output_amps_file
        self._results['odf_directions'] = output_dirs_file

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

    def _extrapolate_scheme(self, scheme_name, runtime, fit_obj, mask_img):
        if scheme_name not in ("ABCD", "HCP"):
            return
        output_dwi_file = fname_presuffix(self.inputs.dwi_file,
                                          suffix=scheme_name,
                                          newpath=runtime.cwd, use_ext=True)
        output_bval_file = fname_presuffix(self.inputs.dwi_file,
                                           suffix='{}.bval'.format(scheme_name),
                                           newpath=runtime.cwd, use_ext=False)
        output_bvec_file = fname_presuffix(self.inputs.dwi_file,
                                           suffix='{}.bvec'.format(scheme_name),
                                           newpath=runtime.cwd, use_ext=False)
        output_b_file = fname_presuffix(self.inputs.dwi_file,
                                        suffix='{}.b'.format(scheme_name),
                                        newpath=runtime.cwd, use_ext=False)
        # Copy in the bval and bvecs
        bval_file = pkgr('qsiprep', 'data/schemes/{}.bval'.format(scheme_name))
        bvec_file = pkgr('qsiprep', 'data/schemes/{}.bvec'.format(scheme_name))
        shutil.copyfile(bval_file, output_bval_file)
        shutil.copyfile(bvec_file, output_bvec_file)
        self._results['extrapolated_bvecs'] = bvec_file
        self._results['extrapolated_bvals'] = bval_file
        _convert_fsl_to_mrtrix(bval_file, bvec_file, output_b_file)
        self._results['extrapolated_b'] = output_b_file

        gtab_to_predict = self._get_gtab(external_bvals=bval_file, external_bvecs=bvec_file)
        new_data = fit_obj.predict(gtab_to_predict, S0=1)
        nb.Nifti1Image(new_data, mask_img.affine, mask_img.header).to_filename(output_dwi_file)
        self._results['extrapolated_dwi'] = output_dwi_file


class MAPMRIInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    laplacian_regularization = traits.Bool(True, usedefault=True)
    laplacian_weighting = traits.Either(traits.Str("GCV"),
                                        traits.Float(0.2),
                                        usedefault=True)
    positivity_constraint = traits.Bool(False, usedefault=True)
    pos_grid = traits.Int(15, usedefault=True)
    pos_radius = traits.Either(traits.Str('adaptive'), traits.Int(),
                               default='adaptive', usedefault=True)
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
        weighting = "GCV" if self.inputs.laplacian_weighting == "GCV" else \
            self.inputs.laplacian_weighting

        if self.inputs.laplacian_regularization and \
           self.inputs.positivity_constraint:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=True,
                laplacian_weighting=weighting,
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
                laplacian_weighting=weighting,
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
        self._results['rtap'] = self._save_scalar(rtap, "_rtap", runtime, dwi_img)

        rtpp = mapfit_aniso.rtpp()
        self._results['rtpp'] = self._save_scalar(rtpp, "_rtpp", runtime, dwi_img)

        coeffs = mapfit_aniso.mapmri_coeff
        self._results['mapmri_coeffs'] = self._save_scalar(coeffs, "_mapcoeffs", runtime, dwi_img)

        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, mapfit_aniso, mask_img, "_MAPMRI")

        return runtime


class BrainSuiteShoreReconstructionInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    zeta = traits.Float(700, usedefault=True)
    tau = traits.Float(TAU_DEFAULT, usedefault=True)
    regularization = traits.Enum("L2", "L1", usedefault=True)
    # For L2
    lambdaN = traits.Float(1e-8, usedefault=True)
    lambdaL = traits.Float(1e-8, usedefault=True)
    # For L1
    regularization_weighting = traits.Str("CV", usedefault=True)
    l1_positive_constraint = traits.Bool(False, usedefault=True)
    l1_cv = traits.Either(traits.Int(3), None, usedefault=True)
    l1_maxiter = traits.Int(1000, usedefault=True)
    l1_verbose = traits.Bool(False, usedefault=True)
    l1_alpha = traits.Float(1.0, usedefault=True)
    # For EAP
    pos_grid = traits.Int(11, usedefault=True)
    pos_radius = traits.Float(20e-03, usedefault=True)


class BrainSuiteShoreReconstructionOutputSpec(DipyReconOutputSpec):
    shore_coeffs_image = File()
    rtop_image = File()
    alpha_image = File()
    r2_image = File()
    cnr_image = File()
    regularization_image = File()


class BrainSuiteShoreReconstruction(DipyReconInterface):
    input_spec = BrainSuiteShoreReconstructionInputSpec
    output_spec = BrainSuiteShoreReconstructionOutputSpec

    def _extrapolate_scheme(self, scheme_name, runtime, fit_obj, mask_array, mask_img):
        if scheme_name not in ("ABCD", "HCP"):
            return
        output_dwi_file = fname_presuffix(self.inputs.dwi_file,
                                          suffix=scheme_name,
                                          newpath=runtime.cwd, use_ext=True)
        output_bval_file = fname_presuffix(self.inputs.dwi_file,
                                           suffix='{}.bval'.format(scheme_name),
                                           newpath=runtime.cwd, use_ext=False)
        output_bvec_file = fname_presuffix(self.inputs.dwi_file,
                                           suffix='{}.bvec'.format(scheme_name),
                                           newpath=runtime.cwd, use_ext=False)
        output_b_file = fname_presuffix(self.inputs.dwi_file,
                                        suffix='{}.b'.format(scheme_name),
                                        newpath=runtime.cwd, use_ext=False)
        # Copy in the bval and bvecs
        bval_file = pkgr('qsiprep', 'data/schemes/{}.bval'.format(scheme_name))
        bvec_file = pkgr('qsiprep', 'data/schemes/{}.bvec'.format(scheme_name))
        shutil.copyfile(bval_file, output_bval_file)
        shutil.copyfile(bvec_file, output_bvec_file)
        self._results['extrapolated_bvecs'] = bvec_file
        self._results['extrapolated_bvals'] = bval_file
        _convert_fsl_to_mrtrix(bval_file, bvec_file, output_b_file)
        self._results['extrapolated_b'] = output_b_file

        prediction_gtab = self._get_gtab(external_bvals=bval_file, external_bvecs=bvec_file)
        prediction_shore = brainsuite_shore_basis(fit_obj.model.radial_order, fit_obj.model.zeta,
                                                  prediction_gtab, fit_obj.model.tau)

        shore_array = fit_obj._shore_coef[mask_array]
        output_data = np.zeros(mask_array.shape + (len(prediction_gtab.bvals),))
        output_data[mask_array] = np.dot(shore_array, prediction_shore.T)

        nb.Nifti1Image(output_data, mask_img.affine, mask_img.header).to_filename(output_dwi_file)
        self._results['extrapolated_dwi'] = output_dwi_file

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
        final_grads = gradient_table(bvals=final_bvals,
                                     bvecs=final_bvecs,
                                     b0_threshold=50,
                                     big_delta=self.inputs.big_delta,
                                     small_delta=self.inputs.little_delta)

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
            l1_positive_constraint=self.inputs.l1_positive_constraint,
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
        alphas_file = fname_presuffix(self.inputs.dwi_file, suffix="_alpha",
                                      newpath=runtime.cwd)
        r2_file = fname_presuffix(self.inputs.dwi_file, suffix="_r2",
                                  newpath=runtime.cwd)
        cnr_file = fname_presuffix(self.inputs.dwi_file, suffix="_cnr",
                                   newpath=runtime.cwd)
        regl_file = fname_presuffix(self.inputs.dwi_file, suffix="_regularization",
                                    newpath=runtime.cwd)

        nb.Nifti1Image(coeffs, dwi_img.affine, dwi_img.header).to_filename(coeffs_file)
        nb.Nifti1Image(rtop, dwi_img.affine, dwi_img.header).to_filename(rtop_file)
        nb.Nifti1Image(bss_fit.regularization, dwi_img.affine, dwi_img.header
                       ).to_filename(regl_file)
        nb.Nifti1Image(bss_fit.r2, dwi_img.affine, dwi_img.header
                       ).to_filename(r2_file)
        nb.Nifti1Image(bss_fit.cnr, dwi_img.affine, dwi_img.header
                       ).to_filename(cnr_file)
        nb.Nifti1Image(bss_fit.alpha, dwi_img.affine, dwi_img.header
                       ).to_filename(alphas_file)
        self._results['shore_coeffs_image'] = coeffs_file
        self._results['rtop_image'] = rtop_file
        self._results['alpha_image'] = alphas_file
        self._results['r2_image'] = r2_file
        self._results['cnr_image'] = cnr_file
        self._results['regularization_image'] = regl_file

        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, bss_fit, mask_img, "_BS3dSHORE")
        # Make HARDIs if desired
        extrapolate = self.inputs.extrapolate_scheme
        if isdefined(extrapolate):
            self._extrapolate_scheme(extrapolate, runtime, bss_fit, mask_array, mask_img)
        return runtime


class TensorReconstructionInputSpec(DipyReconInputSpec):
    pass


class TensorReconstructionOutputSpec(DipyReconOutputSpec):
    color_fa_image = File()
    fa_image = File()
    md_image = File()
    rd_image = File()
    ad_image = File()
    cnr_image = File()


class TensorReconstruction(DipyReconInterface):
    input_spec = TensorReconstructionInputSpec
    output_spec = TensorReconstructionOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype=np.float32)
        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        # Fit it
        tenmodel = dti.TensorModel(gtab)
        ten_fit = tenmodel.fit(dwi_data, mask_array)
        lower_triangular = ten_fit.lower_triangular()
        tensor_img = nifti1_symmat(lower_triangular, dwi_img.affine)
        output_tensor_file = fname_presuffix(self.inputs.dwi_file,
                                             suffix='tensor',
                                             newpath=runtime.cwd, use_ext=True)
        tensor_img.to_filename(output_tensor_file)

        # FA MD RD and AD
        for metric in ["fa", "md", "rd", "ad", "color_fa"]:
            data = getattr(ten_fit, metric).astype("float32")
            out_name = fname_presuffix(self.inputs.dwi_file,
                                       suffix=metric,
                                       newpath=runtime.cwd, use_ext=True)
            nb.Nifti1Image(data, dwi_img.affine).to_filename(out_name)
            self._results[metric + "_image"] = out_name

        return runtime
