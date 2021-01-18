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
from .fmap import eddy_inputs_from_dwi_files
from .epi_fmap import get_best_b0_topup_inputs_from
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
    topup_first = File(exists=True)
    topup_config = traits.Str()
    pre_topup_image = File(exists=True)
    eddy_acqp = File(exists=True)
    eddy_first = File(exists=True)
    b0_csv = File(exists=True)
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
        topup_datain_file, topup_imain_file, topup_text, b0_csv, topup0, eddy0 = \
            get_best_b0_topup_inputs_from(
                dwi_file=self.inputs.dwi_file,
                bval_file=self.inputs.bval_file,
                b0_threshold=self.inputs.b0_threshold,
                cwd=runtime.cwd,
                bids_origin_files=self.inputs.original_files,
                epi_fmaps=self.inputs.epi_fmaps,
                max_per_spec=self.inputs.topup_max_b0s_per_spec,
                topup_requested=self.inputs.topup_requested)
        self._results['topup_datain'] = topup_datain_file
        self._results['topup_imain'] = topup_imain_file
        self._results['topup_report'] = topup_text
        self._results['b0_csv'] = b0_csv
        self._results['topup_first'] = topup0
        self._results['eddy_first'] = eddy0

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


def boilerplate_from_eddy_config(eddy_config, fieldmap_type):
    """Write boilerplate text based on an eddy config dict.
    """
    ext_eddy = ExtendedEddy(**eddy_config)
    desc = [
        "FSL (version %s)'s eddy was used for head motion correction and "
        "Eddy current correction [@anderssoneddy]." % ext_eddy.version
    ]

    # Basic eddy setup
    desc.append(
        "Eddy was configured with a $q$-space smoothing factor "
        "of %d," % ext_eddy.inputs.fudge_factor)
    desc.append("a total of %d iterations," % ext_eddy.inputs.niter)
    desc.append("and %d voxels used to estimate hyperparameters." %
                ext_eddy.inputs.nvoxhp)

    # Specify flm/slm model types
    slm = "was" if ext_eddy.inputs.slm == "none" else \
        "and a %s second level model were" % ext_eddy.inputs.slm
    desc.append("A %s first level model %s used to characterize Eddy current-"
                "related spatial distortion." % (ext_eddy.inputs.flm, slm))

    # fwhm of pre-conditioning filter
    if isdefined(ext_eddy.inputs.fwhm):
        desc.append("A filter with fwhm=%04f was used to pre-condition the "
                    "data before using it to estimate distortions."
                    % ext_eddy.inputs.fwhm)

    # force shelled scheme
    if isdefined(ext_eddy.inputs.is_shelled) and ext_eddy.inputs.is_shelled:
        desc.append("$q$-space coordinates were forcefully assigned to shells.")

    if isdefined(ext_eddy.inputs.fep) and ext_eddy.inputs.fep:
        desc.append("Any empty planes detected in images were filled.")

    # did you sep_offs_mov?
    if isdefined(ext_eddy.inputs.dont_sep_offs_move) and \
            ext_eddy.inputs.dont_sep_offs_move:
        desc.append("No attempt was made to separate field offset from "
                    "subject movement.")
    else:
        desc.append("Field offset was attempted to be separated from "
                    "subject movement.")

    # did you peas?
    if isdefined(ext_eddy.inputs.dont_peas) and ext_eddy.inputs.dont_peas:
        desc.append("No alignment of shells was performed post-eddy.")
    else:
        desc.append("Shells were aligned post-eddy.")

    # repol settings
    if isdefined(ext_eddy.inputs.repol) and ext_eddy.inputs.repol:
        desc.append("Eddy's outlier replacement was run [@eddyrepol].")

        ol_group = {
            "sw": "slice", "gw": "multi-band group",
            "both": "both slice and multi-band group",
            traits.Undefined: "slice"}[ext_eddy.inputs.outlier_type]
        nvox = ext_eddy.inputs.outlier_nstd if \
            isdefined(ext_eddy.inputs.outlier_nstd) else 250
        desc.append("Data were grouped by %s, only including values from "
                    "slices determined to contain at least %d intracerebral "
                    "voxels." % (ol_group, nvox))
        mbf = ext_eddy.inputs.multiband_factor if \
            isdefined(ext_eddy.inputs.multiband_factor) else 1
        mb_off = ext_eddy.inputs.multiband_offset if \
            isdefined(ext_eddy.inputs.multiband_factor) else 0
        if mbf > 1 and "multi-band group" in ol_group:
            offs_txt = "was"
            if mb_off != 0:
                offs_txt = {-1: "bottom", 1: "top"}
                offs_txt = "and slices removed from the %s of the volume were" % offs_txt
            desc.append("A multi-band accelleration factor of %d "
                        "%s assumed." % (mbf, offs_txt))

        # The threshold for outliers
        std_threshold = ext_eddy.inputs.outlier_nstd if \
            isdefined(ext_eddy.inputs.outlier_nstd) else 4
        ssq = " sum-of-squares" if isdefined(ext_eddy.inputs.outlier_sqr) \
            and ext_eddy.inputs.outlier_sqr else ""
        pos = " (positively or negatively%s)" % ssq if isdefined(ext_eddy.inputs.outlier_pos) \
            and ext_eddy.inputs.outlier_pos else ""
        desc.append("Groups deviating by more than %d standard deviations%s from the prediction "
                    "had their data replaced with imputed values."
                    % (std_threshold, pos))

    # slice-to-vol
    if isdefined(ext_eddy.inputs.mporder) and  \
            ext_eddy.inputs.mporder > 0:
        niter = ext_eddy.inputs.slice2vol_niter if \
            isdefined(ext_eddy.inputs.slice2vol_niter) else 5
        lam = ext_eddy.inputs.slice2vol_lambda if \
            isdefined(ext_eddy.inputs.slice2vol_lambda) else 1
        s2v_interp = ext_eddy.inputs.slice2vol_interp if \
            isdefined(ext_eddy.inputs.slice2vol_interp) else "trilinear"
        desc.append("Slice-to-volume correction was estimated with "
                    "temporal order %d, %d iterations, %s interpolation "
                    "and lambda=%.3f [@eddys2v]."
                    % (ext_eddy.inputs.mporder, niter, s2v_interp, lam))

    # TOPUP
    if fieldmap_type in ("rpe_series", "epi"):
        desc.append("Data was collected with reversed phase-encode blips, resulting "
                    "in pairs of images with distortions going in opposite directions.")
        if fieldmap_type == "epi":
            desc.append("Here, b=0 reference images with reversed "
                        "phase encoding directions were used "
                        "along with an equal number of b=0 images extracted "
                        "from the DWI scans.")
        else:
            desc.append("Here, multiple DWI series were acquired with opposite phase encoding "
                        "directions, so b=0 images were extracted from each.")
        desc.append("From these pairs the susceptibility-induced off-resonance field was "
                    "estimated using a method similar to that described in [@topup]. "
                    "The fieldmaps were ultimately incorporated into the "
                    "Eddy current and head motion correction interpolation.")

    # move by susceptibility
    if isdefined(ext_eddy.inputs.estimate_move_by_susceptibility) and \
            ext_eddy.inputs.estimate_move_by_susceptibility:
        mbs_niter = ext_eddy.inputs.mbs_niter if \
            isdefined(ext_eddy.inputs.mbs_niter) else 10
        mbs_lambda = ext_eddy.inputs.mbs_mbs_lambda if \
            isdefined(ext_eddy.inputs.mbs_lambda) else 10
        mbs_ksp = ext_eddy.inputs.mbs_ksp if \
            isdefined(ext_eddy.inputs.mbs_ksp) else 10
        desc.append("Dynamic susceptibility distortion correction was "
                    "applied with %d iterations, lambda=%.2f and spline "
                    "knot-spacing of %.2fmm [@eddysus]."
                    % (mbs_niter, mbs_lambda, mbs_ksp))

    # Format the interpolation
    lsr_ref = ' [@fsllsr]' if ext_eddy.inputs.method == 'lsr' else ''
    desc.append("Final interpolation was performed using the `%s` method%s.\n\n" % (
        ext_eddy.inputs.method, lsr_ref))

    return " ".join(desc)
