#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Prepare files for TOPUP and eddy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import os.path as op
from pkg_resources import resource_filename as pkgr_fn
import numpy as np
import nibabel as nb
from nilearn.image import index_img
from nipype import logging
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiObject, SimpleInterface,
    isdefined)
from .fmap import get_topup_inputs_from, eddy_inputs_from_dwi_files
LOGGER = logging.getLogger('nipype.interface')


class GatherEddyInputsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b0_threshold = traits.CInt(100, usedefault=True)
    original_files = InputMultiObject(File(exists=True))
    epi_fmaps = InputMultiObject(File(exists=True),
                                 desc='files from fmaps/ for distortion correction')
    topup_max_b0s_per_spec = traits.CInt(3, usedefault=True)
    topup_requested = traits.Bool(False, usedefault=True)


class GatherEddyInputsOutputSpec(TraitedSpec):
    topup_datain = File(exists=True)
    topup_imain = File(exists=True)
    topup_config = traits.Str()
    pre_topup_image = File(exists=True)
    eddy_acqp = File(exists=True)
    eddy_indices = File(exists=True)
    forward_transforms = traits.List()
    forward_warps = traits.List()
    topup_report = traits.Str(desc="description of where data came from")


class GatherEddyInputs(SimpleInterface):
    """Manually prepare inputs for TOPUP and eddy.

    **Inputs**
        rpe_b0: str
            path to a file (3D or 4D) containing b=0 images with the reverse PE direction
        dwi_file: str
            path to a 4d DWI nifti file
        bval_file: str
            path to the bval file
        bvec_file: str
            path to the bvec file
    """
    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):

        # Gather inputs for TOPUP
        topup_prefix = op.join(runtime.cwd, "topup_")
        topup_datain_file, topup_imain_file, topup_text = get_topup_inputs_from(
            dwi_file=self.inputs.dwi_file,
            bval_file=self.inputs.bval_file,
            b0_threshold=self.inputs.b0_threshold,
            topup_prefix=topup_prefix,
            bids_origin_files=self.inputs.original_files,
            epi_fmaps=self.inputs.epi_fmaps,
            max_per_spec=self.inputs.topup_max_b0s_per_spec,
            topup_requested=self.inputs.topup_requested)
        self._results['topup_datain'] = topup_datain_file
        self._results['topup_imain'] = topup_imain_file
        self._results['topup_report'] = topup_text

        # If there are an odd number of slices, use b02b0_1.cnf
        example_b0 = nb.load(self.inputs.dwi_file)
        self._results['topup_config'] = 'b02b0.cnf'
        if 1 in (example_b0.shape[0] % 2, example_b0.shape[1] % 2, example_b0.shape[2] % 2):
            LOGGER.warning(
                "Using slower b02b0_1.cnf because an axis has an odd number of slices")
            self._results['topup_config'] = pkgr_fn('qsiprep.data', 'b02b0_1.cnf')

        # For the apply topup report:
        pre_topup_image = index_img(topup_imain_file, 0)
        pre_topup_image_file = topup_prefix + "pre_image.nii.gz"
        pre_topup_image.to_filename(pre_topup_image_file)
        self._results['pre_topup_image'] = pre_topup_image_file

        # Gather inputs for eddy
        eddy_prefix = op.join(runtime.cwd, "eddy_")
        acqp_file, index_file = eddy_inputs_from_dwi_files(self.inputs.original_files,
                                                           eddy_prefix)
        self._results['eddy_acqp'] = acqp_file
        self._results['eddy_indices'] = index_file

        # these have already had HMC, SDC applied
        self._results['forward_transforms'] = []
        self._results['forward_warps'] = []
        return runtime


class ExtendedEddyOutputSpec(fsl.epi.EddyOutputSpec):
    shell_PE_translation_parameters = File(
        exists=True,
        desc=('the translation along the PE-direction between the different shells'))
    outlier_map = File(
        exists=True, desc='All numbers are either 0, meaning that scan-slice '
        'is not an outliers, or 1 meaning that it is.')
    outlier_n_stdev_map = File(
        exists=True, desc='how many standard deviations off the mean difference '
        'between observation and prediction is.')
    outlier_n_sqr_stdev_map = File(
        exists=True, desc='how many standard deviations off the square root of the '
        'mean squared difference between observation and prediction is.')
    outlier_free_data = File(
        exists=True, desc=' the original data given by --imain not corrected for '
        'susceptibility or EC-induced distortions or subject movement, but with '
        'outlier slices replaced by the Gaussian Process predictions.')


class ExtendedEddy(fsl.Eddy):
    output_spec = ExtendedEddyOutputSpec

    _num_threads = 1

    def __init__(self, **inputs):
        super(ExtendedEddy, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not isdefined(self.inputs.num_threads):
            self.inputs.num_threads = self._num_threads
        else:
            self._num_threads_update()
        self.inputs.on_trait_change(self._use_cuda, 'use_cuda')
        if isdefined(self.inputs.use_cuda):
            self._use_cuda()

    def _num_threads_update(self):
        self._num_threads = self.inputs.num_threads
        if not isdefined(self.inputs.num_threads):
            if 'OMP_NUM_THREADS' in self.inputs.environ:
                del self.inputs.environ['OMP_NUM_THREADS']
        else:
            self.inputs.environ['OMP_NUM_THREADS'] = str(
                self.inputs.num_threads)

    def _use_cuda(self):
        self._cmd = 'eddy_cuda9.1' if self.inputs.use_cuda else 'eddy_openmp'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected'] = os.path.abspath(
            '%s.nii.gz' % self.inputs.out_base)
        outputs['out_parameter'] = os.path.abspath(
            '%s.eddy_parameters' % self.inputs.out_base)

        # File generation might depend on the version of EDDY
        out_rotated_bvecs = os.path.abspath(
            '%s.eddy_rotated_bvecs' % self.inputs.out_base)
        out_movement_rms = os.path.abspath(
            '%s.eddy_movement_rms' % self.inputs.out_base)
        out_restricted_movement_rms = os.path.abspath(
            '%s.eddy_restricted_movement_rms' % self.inputs.out_base)
        out_shell_alignment_parameters = os.path.abspath(
            '%s.eddy_post_eddy_shell_alignment_parameters' % self.inputs.out_base)
        shell_PE_translation_parameters = op.abspath(
            '%s.eddy_post_eddy_shell_PE_translation_parameters' % self.inputs.out_base)
        out_outlier_report = os.path.abspath(
            '%s.eddy_outlier_report' % self.inputs.out_base)
        outlier_map = op.abspath(
            '%s.eddy_outlier_map' % self.inputs.out_base)
        outlier_n_stdev_map = op.abspath(
            '%s.eddy_outlier_n_stdev_map' % self.inputs.out_base)
        outlier_n_sqr_stdev_map = op.abspath(
            '%s.eddy_outlier_n_sqr_stdev_map' % self.inputs.out_base)

        if isdefined(self.inputs.cnr_maps) and self.inputs.cnr_maps:
            out_cnr_maps = os.path.abspath(
                '%s.eddy_cnr_maps.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_cnr_maps):
                outputs['out_cnr_maps'] = out_cnr_maps
        if isdefined(self.inputs.residuals) and self.inputs.residuals:
            out_residuals = os.path.abspath(
                '%s.eddy_residuals.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_residuals):
                outputs['out_residuals'] = out_residuals

        if os.path.exists(out_rotated_bvecs):
            outputs['out_rotated_bvecs'] = out_rotated_bvecs
        if os.path.exists(out_movement_rms):
            outputs['out_movement_rms'] = out_movement_rms
        if os.path.exists(out_restricted_movement_rms):
            outputs['out_restricted_movement_rms'] = \
                out_restricted_movement_rms
        if os.path.exists(out_shell_alignment_parameters):
            outputs['out_shell_alignment_parameters'] = \
                out_shell_alignment_parameters
        if os.path.exists(out_outlier_report):
            outputs['out_outlier_report'] = out_outlier_report

        if op.exists(shell_PE_translation_parameters):
            outputs['shell_PE_translation_parameters'] = shell_PE_translation_parameters
        if op.exists(outlier_map):
            outputs['outlier_map'] = outlier_map
        if op.exists(outlier_n_stdev_map):
            outputs['outlier_n_stdev_map'] = outlier_n_stdev_map
        if op.exists(outlier_n_sqr_stdev_map):
            outputs['outlier_n_sqr_stdev_map'] = outlier_n_sqr_stdev_map

        return outputs

    def _format_arg(self, name, spec, value):
        if name == 'field':
            pth, fname, _ = split_filename(value)
            return spec.argstr % op.join(pth, fname)
        return super(ExtendedEddy, self)._format_arg(name, spec, value)


class Eddy2SPMMotionInputSpec(BaseInterfaceInputSpec):
    eddy_motion = File(exists=True)


class Eddy2SPMMotionOututSpec(TraitedSpec):
    spm_motion_file = File(exists=True)


class Eddy2SPMMotion(SimpleInterface):
    input_spec = Eddy2SPMMotionInputSpec
    output_spec = Eddy2SPMMotionOututSpec

    def _run_interface(self, runtime):
        # Load the eddy motion params File
        eddy_motion = np.loadtxt(self.inputs.eddy_motion)
        spm_motion = eddy_motion[:, :6]
        spm_motion_file = fname_presuffix(self.inputs.eddy_motion, suffix="spm_rp.txt",
                                          use_ext=False, newpath=runtime.cwd)
        np.savetxt(spm_motion_file, spm_motion)
        self._results['spm_motion_file'] = spm_motion_file

        return runtime
