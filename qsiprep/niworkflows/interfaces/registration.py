# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ReportCapableInterfaces for registration tools

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from distutils.version import LooseVersion

import nibabel as nb
import numpy as np
from nilearn import image as nli
from nilearn.image import index_img
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface)
from nipype.interfaces.mixins import reporting
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl, ants, afni

from .. import NIWORKFLOWS_LOG
from . import report_base as nrc
from .mni import (
    RobustMNINormalizationInputSpec,
    RobustMNINormalization
)
from .fixes import (FixHeaderApplyTransforms as ApplyTransforms,
                    FixHeaderRegistration as Registration)


class RobustMNINormalizationInputSpecRPT(
        nrc.SVGReportCapableInputSpec, RobustMNINormalizationInputSpec):
    pass


class RobustMNINormalizationOutputSpecRPT(
        reporting.ReportCapableOutputSpec, ants.registration.RegistrationOutputSpec):
    pass


class RobustMNINormalizationRPT(
        nrc.RegistrationRC, RobustMNINormalization):
    input_spec = RobustMNINormalizationInputSpecRPT
    output_spec = RobustMNINormalizationOutputSpecRPT

    def _post_run_hook(self, runtime):
        # We need to dig into the internal ants.Registration interface
        self._fixed_image = self._get_ants_args()['fixed_image']
        if isinstance(self._fixed_image, (list, tuple)):
            self._fixed_image = self._fixed_image[0]  # get first item if list

        if self._get_ants_args().get('fixed_image_mask') is not None:
            self._fixed_image_mask = self._get_ants_args().get('fixed_image_mask')
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)

        return super(RobustMNINormalizationRPT, self)._post_run_hook(runtime)


class ANTSRegistrationInputSpecRPT(nrc.SVGReportCapableInputSpec,
                                   ants.registration.RegistrationInputSpec):
    pass


class ANTSRegistrationOutputSpecRPT(reporting.ReportCapableOutputSpec,
                                    ants.registration.RegistrationOutputSpec):
    pass


class ANTSRegistrationRPT(nrc.RegistrationRC, Registration):
    input_spec = ANTSRegistrationInputSpecRPT
    output_spec = ANTSRegistrationOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.fixed_image[0]
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)

        return super(ANTSRegistrationRPT, self)._post_run_hook(runtime)


class ANTSApplyTransformsInputSpecRPT(nrc.SVGReportCapableInputSpec,
                                      ants.resampling.ApplyTransformsInputSpec):
    pass


class ANTSApplyTransformsOutputSpecRPT(reporting.ReportCapableOutputSpec,
                                       ants.resampling.ApplyTransformsOutputSpec):
    pass


class ANTSApplyTransformsRPT(nrc.RegistrationRC, ApplyTransforms):
    input_spec = ANTSApplyTransformsInputSpecRPT
    output_spec = ANTSApplyTransformsOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference_image
        self._moving_image = self.aggregate_outputs(runtime=runtime).output_image
        NIWORKFLOWS_LOG.info('Report - setting fixed (%s) and moving (%s) images',
                             self._fixed_image, self._moving_image)

        return super(ANTSApplyTransformsRPT, self)._post_run_hook(runtime)


class ApplyTOPUPInputSpecRPT(nrc.SVGReportCapableInputSpec,
                             fsl.epi.ApplyTOPUPInputSpec):
    wm_seg = File(argstr='-wmseg %s',
                  desc='reference white matter segmentation mask')


class ApplyTOPUPOutputSpecRPT(reporting.ReportCapableOutputSpec,
                              fsl.epi.ApplyTOPUPOutputSpec):
    pass


class ApplyTOPUPRPT(nrc.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = ApplyTOPUPInputSpecRPT
    output_spec = ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = index_img(self.aggregate_outputs(runtime=runtime).out_corrected, 0)
        self._moving_image = index_img(self.inputs.in_files[0], 0)
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info('Report - setting corrected (%s) and warped (%s) images',
                             self._fixed_image, self._moving_image)

        return super(ApplyTOPUPRPT, self)._post_run_hook(runtime)


class FUGUEInputSpecRPT(nrc.SVGReportCapableInputSpec,
                        fsl.preprocess.FUGUEInputSpec):
    wm_seg = File(argstr='-wmseg %s',
                  desc='reference white matter segmentation mask')


class FUGUEOutputSpecRPT(reporting.ReportCapableOutputSpec,
                         fsl.preprocess.FUGUEOutputSpec):
    pass


class FUGUERPT(nrc.RegistrationRC, fsl.FUGUE):
    input_spec = FUGUEInputSpecRPT
    output_spec = FUGUEOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.aggregate_outputs(runtime=runtime).unwarped_file
        self._moving_image = self.inputs.in_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting corrected (%s) and warped (%s) images',
            self._fixed_image, self._moving_image)

        return super(FUGUERPT, self)._post_run_hook(runtime)


class FLIRTInputSpecRPT(nrc.SVGReportCapableInputSpec,
                        fsl.preprocess.FLIRTInputSpec):
    pass


class FLIRTOutputSpecRPT(reporting.ReportCapableOutputSpec,
                         fsl.preprocess.FLIRTOutputSpec):
    pass


class FLIRTRPT(nrc.RegistrationRC, fsl.FLIRT):
    input_spec = FLIRTInputSpecRPT
    output_spec = FLIRTOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference
        self._moving_image = self.aggregate_outputs(runtime=runtime).out_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)

        return super(FLIRTRPT, self)._post_run_hook(runtime)


class ApplyXFMInputSpecRPT(nrc.SVGReportCapableInputSpec,
                           fsl.preprocess.ApplyXFMInputSpec):
    pass


class ApplyXFMRPT(FLIRTRPT, fsl.ApplyXFM):
    input_spec = ApplyXFMInputSpecRPT
    output_spec = FLIRTOutputSpecRPT


if LooseVersion("0.0.0") < fs.Info.looseversion() < LooseVersion("6.0.0"):
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec
else:
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec6


class BBRegisterInputSpecRPT(nrc.SVGReportCapableInputSpec,
                             _BBRegisterInputSpec):
    # Adds default=True, usedefault=True
    out_lta_file = traits.Either(traits.Bool, File, default=True, usedefault=True,
                                 argstr="--lta %s", min_ver='5.2.0',
                                 desc="write the transformation matrix in LTA format")


class BBRegisterOutputSpecRPT(reporting.ReportCapableOutputSpec,
                              fs.preprocess.BBRegisterOutputSpec):
    pass


class BBRegisterRPT(nrc.RegistrationRC, fs.BBRegister):
    input_spec = BBRegisterInputSpecRPT
    output_spec = BBRegisterOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = os.path.join(self.inputs.subjects_dir,
                               self.inputs.subject_id, 'mri')
        target_file = os.path.join(mri_dir, 'brainmask.mgz')

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp='nearest')
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        self._contour = os.path.join(mri_dir, 'ribbon.mgz')
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)

        return super(BBRegisterRPT, self)._post_run_hook(runtime)


class MRICoregInputSpecRPT(nrc.SVGReportCapableInputSpec,
                           fs.registration.MRICoregInputSpec):
    pass


class MRICoregOutputSpecRPT(reporting.ReportCapableOutputSpec,
                            fs.registration.MRICoregOutputSpec):
    pass


class MRICoregRPT(nrc.RegistrationRC, fs.MRICoreg):
    input_spec = MRICoregInputSpecRPT
    output_spec = MRICoregOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = None
        if isdefined(self.inputs.subject_id):
            mri_dir = os.path.join(self.inputs.subjects_dir,
                                   self.inputs.subject_id, 'mri')

        if isdefined(self.inputs.reference_file):
            target_file = self.inputs.reference_file
        else:
            target_file = os.path.join(mri_dir, 'brainmask.mgz')

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp='nearest')
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        if mri_dir is not None:
            self._contour = os.path.join(mri_dir, 'ribbon.mgz')
        NIWORKFLOWS_LOG.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image, self._moving_image)

        return super(MRICoregRPT, self)._post_run_hook(runtime)


class SimpleBeforeAfterInputSpecRPT(nrc.SVGReportCapableInputSpec):
    before = File(exists=True, mandatory=True, desc='file before')
    after = File(exists=True, mandatory=True, desc='file after')
    wm_seg = File(desc='reference white matter segmentation mask')


class SimpleBeforeAfterRPT(nrc.RegistrationRC, nrc.ReportingInterface):
    input_spec = SimpleBeforeAfterInputSpecRPT

    _fixed_image_label = "after"
    _moving_image_label = "before"

    def _post_run_hook(self, runtime):
        """ there is not inner interface to run """
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting before (%s) and after (%s) images',
            self._fixed_image, self._moving_image)

        return super(SimpleBeforeAfterRPT, self)._post_run_hook(runtime)


class ResampleBeforeAfterInputSpecRPT(SimpleBeforeAfterInputSpecRPT):
    base = traits.Enum('before', 'after', usedefault=True, mandatory=True)


class ResampleBeforeAfterRPT(SimpleBeforeAfterRPT):
    input_spec = ResampleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        if self.inputs.base == 'before':
            resampled_after = nli.resample_to_img(self._fixed_image, self._moving_image)
            fname = fname_presuffix(self._fixed_image, suffix='_resampled', newpath=runtime.cwd)
            resampled_after.to_filename(fname)
            self._fixed_image = fname
        else:
            resampled_before = nli.resample_to_img(self._moving_image, self._fixed_image)
            fname = fname_presuffix(self._moving_image, suffix='_resampled', newpath=runtime.cwd)
            resampled_before.to_filename(fname)
            self._moving_image = fname
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        NIWORKFLOWS_LOG.info(
            'Report - setting before (%s) and after (%s) images',
            self._fixed_image, self._moving_image)

        runtime = super(ResampleBeforeAfterRPT, self)._post_run_hook(runtime)
        NIWORKFLOWS_LOG.info('Successfully created report (%s)', self._out_report)
        os.unlink(fname)

        return runtime


class EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")
    sbref_file = File(exists=True, desc="Single band reference image")
    mc_method = traits.Enum(
        "AFNI", "FSL", usedefault=True,
        desc="Which software to use to perform motion correction")


class EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(desc="Number of detected non-steady "
                                           "state volumes in the beginning of "
                                           "the input file")


class EstimateReferenceImage(SimpleInterface):
    """
    Given an 4D EPI file estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If detected uses
    T1 saturated volumes (non-steady state) otherwise a median of
    of a subset of motion corrected volumes is used.
    """
    input_spec = EstimateReferenceImageInputSpec
    output_spec = EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):
        in_nii = nb.load(self.inputs.in_file)
        n_volumes_to_discard = _get_vols_to_discard(in_nii)

        self._results["n_volumes_to_discard"] = n_volumes_to_discard

        if isdefined(self.inputs.sbref_file):
            self._results['ref_image'] = self.inputs.sbref_file
            return runtime

        # Slicing may induce inconsistencies with shape-dependent values in extensions.
        # For now, remove all. If this turns out to be a mistake, we can select extensions
        # that don't break pipeline stages.
        in_nii.header.extensions.clear()

        out_ref_fname = os.path.join(runtime.cwd, "ref_image.nii.gz")

        if n_volumes_to_discard == 0:
            if in_nii.shape[-1] > 40:
                slice_fname = os.path.join(runtime.cwd, "slice.nii.gz")
                nb.Nifti1Image(in_nii.dataobj[:, :, :, 20:40], in_nii.affine,
                               in_nii.header).to_filename(slice_fname)
            else:
                slice_fname = self.inputs.in_file

            if self.inputs.mc_method == "AFNI":
                res = afni.Volreg(in_file=slice_fname, args='-Fourier -twopass',
                                  zpad=4, outputtype='NIFTI_GZ').run()
            elif self.inputs.mc_method == "FSL":
                res = fsl.MCFLIRT(in_file=slice_fname,
                                  ref_vol=0, interpolation='sinc').run()
            mc_slice_nii = nb.load(res.outputs.out_file)

            median_image_data = np.median(mc_slice_nii.get_data(), axis=3)
        else:
            median_image_data = np.median(
                in_nii.dataobj[:, :, :, :n_volumes_to_discard], axis=3)

        nb.Nifti1Image(median_image_data, in_nii.affine,
                       in_nii.header).to_filename(out_ref_fname)

        self._results["ref_image"] = out_ref_fname

        return runtime


def _get_vols_to_discard(img):
    from nipype.algorithms.confounds import is_outlier
    data_slice = img.dataobj[:, :, :, :50]
    global_signal = data_slice.mean(axis=0).mean(axis=0).mean(axis=0)
    return is_outlier(global_signal)
