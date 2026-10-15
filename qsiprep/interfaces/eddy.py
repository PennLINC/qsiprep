# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Prepare files for TOPUP and eddy."""

import json
import os
import os.path as op
import re

import nibabel as nb
import numpy as np
from nilearn.image import index_img
from nipype import logging
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix, split_filename

from .. import config
from ..data import load as load_data
from ..utils.versions import eddy_version, tortoise_version, versioned
from .epi_fmap import (
    add_synthetic_b0_to_topup_inputs,
    eddy_inputs_from_dwi_files,
    get_best_b0_topup_inputs_from,
    synb0_topup_config,
)

LOGGER = logging.getLogger('nipype.interface')

_EDDY_CUDA_RE = re.compile(r'^eddy_cuda(\d+)\.(\d+)$')


def _find_eddy_cuda(default='eddy_cuda10.2'):
    """Locate the installed FSL eddy CUDA binary (e.g. ``eddy_cuda11.0``).

    FSL ships version-specific binaries whose name encodes the CUDA version
    (``eddy_cuda11.0``). Older code hardcoded ``eddy_cuda10.2``, which current
    FSL builds no longer provide. Scan the directories on ``PATH`` for any
    ``eddy_cuda<major>.<minor>`` executable and return the newest one. Some
    packagings (e.g. the pixi-based qsiprep image) instead ship a single,
    *unversioned* ``eddy_cuda``; use it as a fallback when no versioned binary is
    present, so ``--gpu eddy`` works there too.

    Parameters
    ----------
    default : str, optional
        Name returned when no ``eddy_cuda`` binary is found, so the downstream
        missing-dependency check reports a recognizable command.

    Returns
    -------
    str
        Basename of the selected eddy CUDA binary, or ``default`` if none found.
    """
    found = {}
    plain = None
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        if not directory or not os.path.isdir(directory):
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            match = _EDDY_CUDA_RE.match(entry)
            if match is None and entry != 'eddy_cuda':
                continue
            full_path = os.path.join(directory, entry)
            if not os.path.isfile(full_path) or not os.access(full_path, os.X_OK):
                continue
            if match is None:
                # Unversioned eddy_cuda: keep the first one on PATH.
                if plain is None:
                    plain = entry
                continue
            version = (int(match.group(1)), int(match.group(2)))
            # Keep the first match on PATH for each basename.
            found.setdefault(entry, version)

    if found:
        # A version-suffixed binary is preferred over a plain one, newest first.
        if len(found) > 1:
            LOGGER.warning(
                'Multiple eddy_cuda binaries found on PATH (%s); using the newest.',
                ', '.join(sorted(found)),
            )
        return max(found, key=found.get)

    if plain is not None:
        return plain

    LOGGER.warning('No eddy_cuda binary found on PATH; falling back to %s', default)
    return default


class GatherEddyInputsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b0_threshold = traits.CInt(100, usedefault=True)
    original_files = InputMultiObject(File(exists=True))
    epi_fmaps = InputMultiObject(
        File(exists=True), desc='files from fmaps/ for distortion correction'
    )
    topup_max_b0s_per_spec = traits.CInt(1, usedefault=True)
    topup_requested = traits.Bool(False, usedefault=True)
    synb0_requested = traits.Bool(
        False,
        usedefault=True,
        desc='a synthetic distortion-free b=0 will join the TOPUP inputs '
        'downstream as its own zero-readout distortion group, so a single '
        'measured distortion group is enough to run TOPUP',
    )
    raw_image_sdc = traits.Bool(True, usedefault=True)
    num_threads = traits.CInt(
        1, usedefault=True, desc='CPU cores for the TORTOISE SelectBestB0 b=0 scoring'
    )
    eddy_config = File(exists=True, mandatory=True)
    json_file = File(exists=True)
    sidecars = traits.Dict(
        desc='per-file PE/readout metadata from the grouping model, to spare sidecar reads'
    )


class GatherEddyInputsOutputSpec(TraitedSpec):
    topup_datain = File(exists=True)
    topup_imain = File(exists=True)
    topup_first = File(exists=True)
    topup_config = traits.Str()
    pre_topup_image = File(exists=True)
    eddy_acqp = File(exists=True)
    eddy_first = File(exists=True)
    b0_tsv = File(exists=True)
    eddy_indices = File(exists=True)
    forward_transforms = traits.List()
    forward_warps = traits.List()
    topup_report = traits.Str(desc='description of where data came from')
    json_file = File(exists=True)
    multiband_factor = traits.Int()


class GatherEddyInputs(SimpleInterface):
    """Manually prepare inputs for TOPUP and eddy.

    Inputs
    ------
    rpe_b0 : str
        path to a file (3D or 4D) containing b=0 images with the reverse PE direction
    dwi_file : str
        path to a 4d DWI nifti file
    bval_file : str
        path to the bval file
    bvec_file : str
        path to the bvec file
    """

    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):
        sidecars = self.inputs.sidecars if isdefined(self.inputs.sidecars) else None

        # Gather inputs for TOPUP
        topup_prefix = op.join(runtime.cwd, 'topup_')
        topup_datain_file, topup_imain_file, topup_text, b0_tsv, topup0, eddy0 = (
            get_best_b0_topup_inputs_from(
                dwi_file=self.inputs.dwi_file,
                bval_file=self.inputs.bval_file,
                b0_threshold=self.inputs.b0_threshold,
                cwd=runtime.cwd,
                bids_origin_files=self.inputs.original_files,
                epi_fmaps=self.inputs.epi_fmaps,
                max_per_spec=self.inputs.topup_max_b0s_per_spec,
                topup_requested=self.inputs.topup_requested,
                raw_image_sdc=self.inputs.raw_image_sdc,
                sidecars=sidecars,
                num_threads=self.inputs.num_threads,
                synb0_requested=self.inputs.synb0_requested,
            )
        )
        self._results['topup_datain'] = topup_datain_file
        self._results['topup_imain'] = topup_imain_file
        self._results['topup_report'] = topup_text
        self._results['b0_tsv'] = b0_tsv
        self._results['topup_first'] = topup0
        self._results['eddy_first'] = eddy0

        # If there are an odd number of slices, use b02b0_1.cnf
        example_b0 = nb.load(self.inputs.dwi_file)
        self._results['topup_config'] = 'b02b0.cnf'
        if 1 in (example_b0.shape[0] % 2, example_b0.shape[1] % 2, example_b0.shape[2] % 2):
            LOGGER.warning('Using slower b02b0_1.cnf because an axis has an odd number of slices')
            self._results['topup_config'] = str(load_data('b02b0_1.cnf'))

        # For the apply topup report:
        pre_topup_image = index_img(topup_imain_file, 0)
        pre_topup_image_file = topup_prefix + 'pre_image.nii.gz'
        pre_topup_image.to_filename(pre_topup_image_file)
        self._results['pre_topup_image'] = pre_topup_image_file

        # Gather inputs for eddy
        eddy_prefix = op.join(runtime.cwd, 'eddy_')
        acqp_file, index_file = eddy_inputs_from_dwi_files(
            self.inputs.original_files, eddy_prefix, sidecars=sidecars
        )
        self._results['eddy_acqp'] = acqp_file
        self._results['eddy_indices'] = index_file

        # these have already had HMC, SDC applied
        self._results['forward_transforms'] = []
        self._results['forward_warps'] = []

        # Based on the eddy config, determine whether to send a json argument
        with open(self.inputs.eddy_config) as eddy_cfg_f:
            eddy_config = json.load(eddy_cfg_f)
            # json file is only allowed if mporder is defined
            if 'mporder' in eddy_config:
                self._results['json_file'] = self.inputs.json_file

        return runtime


class Synb0TopupInputsInputSpec(BaseInterfaceInputSpec):
    topup_datain = File(
        exists=True, mandatory=True, desc='datain rows describing the real b=0 volumes'
    )
    topup_imain = File(
        exists=True, mandatory=True, desc='4D stack of the selected real b=0 volumes'
    )
    synthetic_b0 = File(
        exists=True,
        mandatory=True,
        desc='synthetic distortion-free b=0 on the b=0 reference grid',
    )
    smoothing_sigma_mm = traits.Float(
        1.15,
        usedefault=True,
        desc='Gaussian sigma (mm) applied to the real b=0 volumes so their '
        'smoothness matches the U-Net output',
    )


class Synb0TopupInputsOutputSpec(TraitedSpec):
    topup_datain = File(exists=True)
    topup_imain = File(exists=True)
    topup_config = traits.Str(desc='TOPUP config matching the imain dimensions')


class Synb0TopupInputs(SimpleInterface):
    """Join a synthetic distortion-free b=0 to the TOPUP inputs.

    The synthetic volume becomes the last imain volume with a zero-readout
    datain row, and the real b=0 volumes are slightly smoothed to match its
    smoothness (see
    :func:`~qsiprep.interfaces.epi_fmap.add_synthetic_b0_to_topup_inputs`).
    The TOPUP config is chosen for the imain dimensions (see
    :func:`~qsiprep.interfaces.epi_fmap.synb0_topup_config`).
    """

    input_spec = Synb0TopupInputsInputSpec
    output_spec = Synb0TopupInputsOutputSpec

    def _run_interface(self, runtime):
        datain, imain = add_synthetic_b0_to_topup_inputs(
            topup_datain=self.inputs.topup_datain,
            topup_imain=self.inputs.topup_imain,
            synthetic_b0=self.inputs.synthetic_b0,
            cwd=runtime.cwd,
            smoothing_sigma_mm=self.inputs.smoothing_sigma_mm,
        )
        self._results['topup_datain'] = datain
        self._results['topup_imain'] = imain
        self._results['topup_config'] = synb0_topup_config(nb.load(imain).shape)
        return runtime


class ExtendedEddyInputSpec(fsl.epi.EddyInputSpec):
    num_threads = traits.Int(1, usedefault=True, argstr='--nthr=%d')


class ExtendedEddyOutputSpec(fsl.epi.EddyOutputSpec):
    shell_PE_translation_parameters = File(
        exists=True, desc=('the translation along the PE-direction between the different shells')
    )
    outlier_map = File(
        exists=True,
        desc='All numbers are either 0, meaning that scan-slice '
        'is not an outliers, or 1 meaning that it is.',
    )
    outlier_n_stdev_map = File(
        exists=True,
        desc='how many standard deviations off the mean difference '
        'between observation and prediction is.',
    )
    outlier_n_sqr_stdev_map = File(
        exists=True,
        desc='how many standard deviations off the square root of the '
        'mean squared difference between observation and prediction is.',
    )
    outlier_free_data = File(
        exists=True,
        desc=' the original data given by --imain not corrected for '
        'susceptibility or EC-induced distortions or subject movement, but with '
        'outlier slices replaced by the Gaussian Process predictions.',
    )


class ExtendedEddy(fsl.Eddy):
    input_spec = ExtendedEddyInputSpec
    output_spec = ExtendedEddyOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self.inputs.on_trait_change(self._use_cuda, 'use_cuda')
        if isdefined(self.inputs.use_cuda):
            self._use_cuda()

    def _use_cuda(self):
        self._cmd = _find_eddy_cuda() if self.inputs.use_cuda else 'eddy_cpu'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected'] = os.path.abspath(f'{self.inputs.out_base}.nii.gz')
        outputs['out_parameter'] = os.path.abspath(f'{self.inputs.out_base}.eddy_parameters')

        # File generation might depend on the version of EDDY
        out_rotated_bvecs = os.path.abspath(f'{self.inputs.out_base}.eddy_rotated_bvecs')
        out_movement_rms = os.path.abspath(f'{self.inputs.out_base}.eddy_movement_rms')
        out_restricted_movement_rms = os.path.abspath(
            f'{self.inputs.out_base}.eddy_restricted_movement_rms'
        )
        out_shell_alignment_parameters = os.path.abspath(
            f'{self.inputs.out_base}.eddy_post_eddy_shell_alignment_parameters'
        )
        shell_PE_translation_parameters = op.abspath(
            f'{self.inputs.out_base}.eddy_post_eddy_shell_PE_translation_parameters'
        )
        out_outlier_report = os.path.abspath(f'{self.inputs.out_base}.eddy_outlier_report')
        outlier_map = op.abspath(f'{self.inputs.out_base}.eddy_outlier_map')
        outlier_n_stdev_map = op.abspath(f'{self.inputs.out_base}.eddy_outlier_n_stdev_map')
        outlier_n_sqr_stdev_map = op.abspath(
            f'{self.inputs.out_base}.eddy_outlier_n_sqr_stdev_map'
        )

        if isdefined(self.inputs.cnr_maps) and self.inputs.cnr_maps:
            out_cnr_maps = os.path.abspath(f'{self.inputs.out_base}.eddy_cnr_maps.nii.gz')
            if os.path.exists(out_cnr_maps):
                outputs['out_cnr_maps'] = out_cnr_maps
        if isdefined(self.inputs.residuals) and self.inputs.residuals:
            out_residuals = os.path.abspath(f'{self.inputs.out_base}.eddy_residuals.nii.gz')
            if os.path.exists(out_residuals):
                outputs['out_residuals'] = out_residuals

        if os.path.exists(out_rotated_bvecs):
            outputs['out_rotated_bvecs'] = out_rotated_bvecs
        if os.path.exists(out_movement_rms):
            outputs['out_movement_rms'] = out_movement_rms
        if os.path.exists(out_restricted_movement_rms):
            outputs['out_restricted_movement_rms'] = out_restricted_movement_rms
        if os.path.exists(out_shell_alignment_parameters):
            outputs['out_shell_alignment_parameters'] = out_shell_alignment_parameters
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
        if name == 'json':
            if isdefined(self.inputs.mporder):
                return spec.argstr % value
            return ''
        return super()._format_arg(name, spec, value)


def _fsl_to_ras_axis_flip(ref_file):
    """Return the per-axis ±1 that converts FSL rigid params to RAS+ for ``ref_file``.

    FSL reports motion in its radiological voxel frame; the RAS direction of each
    axis follows the sign of the affine diagonal, with FSL flipping x for a
    neurological (positive-determinant) image. Applied to both translation and
    (in-frame Euler) rotation components. Assumes an axis-aligned affine, which
    holds for essentially all DWI acquisitions.
    """
    aff = nb.load(ref_file).affine[:3, :3]
    flip = np.sign(np.diag(aff))
    if np.linalg.det(aff) > 0:
        flip[0] *= -1
    return flip


class Eddy2SPMMotionInputSpec(BaseInterfaceInputSpec):
    eddy_motion = File(exists=True)
    ref_file = File(exists=True, desc='reference image defining the FSL<->RAS axis convention')


class Eddy2SPMMotionOututSpec(TraitedSpec):
    spm_motion_file = File(exists=True)
    eddy_ec_file = File(exists=True)


class Eddy2SPMMotion(SimpleInterface):
    input_spec = Eddy2SPMMotionInputSpec
    output_spec = Eddy2SPMMotionOututSpec

    def _run_interface(self, runtime):
        # eddy_parameters columns: 0-5 are rigid motion (3 translation mm, 3 rotation rad) in
        # FSL's radiological frame; the remaining columns are the eddy-current field
        # coefficients (10 for --flm=quadratic: ~3 linear x/y/z + ~6 quadratic + 1 spare).
        eddy_motion = np.loadtxt(self.inputs.eddy_motion)
        if eddy_motion.ndim == 1:
            eddy_motion = eddy_motion[np.newaxis, :]

        # Rigid motion, converted FSL -> RAS+ so it matches the SHORELine/DIFFPREP export.
        spm_motion = eddy_motion[:, :6].astype(float)
        if isdefined(self.inputs.ref_file):
            flip = _fsl_to_ras_axis_flip(self.inputs.ref_file)
            spm_motion[:, :3] *= flip
            spm_motion[:, 3:6] *= flip
        spm_motion_file = fname_presuffix(
            self.inputs.eddy_motion, suffix='spm_rp.txt', use_ext=False, newpath=runtime.cwd
        )
        np.savetxt(spm_motion_file, spm_motion)
        self._results['spm_motion_file'] = spm_motion_file

        # Eddy-current field coefficients -> headed TSV confounds columns (eddy_ec_NN).
        ec = np.atleast_2d(eddy_motion[:, 6:])
        ec_file = fname_presuffix(
            self.inputs.eddy_motion, suffix='eddy_ec.tsv', use_ext=False, newpath=runtime.cwd
        )
        header = '\t'.join(f'eddy_ec_{i:02d}' for i in range(ec.shape[1]))
        np.savetxt(ec_file, ec, delimiter='\t', header=header, comments='')
        self._results['eddy_ec_file'] = ec_file

        return runtime


def boilerplate_from_eddy_config(eddy_config, fieldmap_type, pepolar_method, gre_in_eddy=False):
    """Write boilerplate text based on an eddy config dict.

    ``gre_in_eddy`` says a GRE fieldmap was passed to eddy, which then applies
    the susceptibility correction in its own resampling, as it does TOPUP's field.
    """
    run_topup = 'topup' in pepolar_method.lower()
    ext_eddy = ExtendedEddy(**eddy_config)
    inputs = ext_eddy.inputs
    version = eddy_version()
    eddy = f'`eddy` from *FSL* (version {version})' if version else '`eddy` from *FSL*'
    if inputs.slm == 'none':
        model = f'a {inputs.flm} eddy-current model'
    else:
        model = f'a {inputs.flm} first-level and {inputs.slm} second-level eddy-current model'
    desc = [
        f'Head motion and eddy-current distortions were corrected with {eddy} '
        f'[@anderssoneddy], using {model}, a *q*-space smoothing factor of '
        f'{inputs.fudge_factor:g}, {inputs.niter} iterations and {inputs.nvoxhp} voxels to '
        'estimate the hyperparameters.'
    ]

    if isdefined(inputs.fwhm):
        desc.append(
            f'The data were smoothed (FWHM {inputs.fwhm} mm) before the distortions were '
            'estimated.'
        )
    if isdefined(inputs.is_shelled) and inputs.is_shelled:
        desc.append('The data were declared shelled, so `eddy` did not check the sampling.')
    if isdefined(inputs.fep) and inputs.fep:
        desc.append('Empty planes in the images were filled.')
    if isdefined(inputs.dont_sep_offs_move) and inputs.dont_sep_offs_move:
        desc.append('The field offset was not separated from subject movement.')
    else:
        desc.append('The field offset was modeled separately from subject movement.')
    if isdefined(inputs.dont_peas) and inputs.dont_peas:
        desc.append('Shells were not aligned to one another after correction.')
    else:
        desc.append('Shells were aligned to one another after correction.')

    if isdefined(inputs.repol) and inputs.repol:
        group = {
            'sw': 'slice',
            'gw': 'multiband slice group',
            'both': 'slice and multiband slice group',
            traits.Undefined: 'slice',
        }[inputs.outlier_type]
        nvox = inputs.outlier_nvox if isdefined(inputs.outlier_nvox) else 250
        nstd = inputs.outlier_nstd if isdefined(inputs.outlier_nstd) else 4
        direction = (
            ' in either direction' if isdefined(inputs.outlier_pos) and inputs.outlier_pos else ''
        )
        squares = (
            ' (using sums of squared differences)'
            if isdefined(inputs.outlier_sqr) and inputs.outlier_sqr
            else ''
        )
        desc.append(
            f'Outliers, detected per {group}, were replaced with the `eddy` prediction '
            f'[@eddyrepol]: data deviating from the prediction by more than {nstd} standard '
            f'deviations{direction}{squares} were treated as outliers, and only slices with at '
            f'least {nvox} brain voxels were assessed.'
        )
        mbf = inputs.multiband_factor if isdefined(inputs.multiband_factor) else 1
        mb_off = inputs.multiband_offset if isdefined(inputs.multiband_offset) else 0
        if mbf > 1 and 'multiband' in group:
            removed = ''
            if mb_off != 0:
                end = {-1: 'bottom', 1: 'top'}[mb_off]
                removed = f', with slices removed from the {end} of the volume'
            desc.append(f'A multiband factor of {mbf} was assumed{removed}.')

    if isdefined(inputs.mporder) and inputs.mporder > 0:
        niter = inputs.slice2vol_niter if isdefined(inputs.slice2vol_niter) else 5
        lam = inputs.slice2vol_lambda if isdefined(inputs.slice2vol_lambda) else 1
        interp = inputs.slice2vol_interp if isdefined(inputs.slice2vol_interp) else 'trilinear'
        desc.append(
            f'Within-volume (slice-to-volume) motion was modeled with a temporal order of '
            f'{inputs.mporder}, {niter} iterations, {interp} interpolation and '
            f'$\\lambda$ = {lam:g} [@eddys2v].'
        )

    if (
        isdefined(inputs.estimate_move_by_susceptibility)
        and inputs.estimate_move_by_susceptibility
    ):
        mbs_niter = inputs.mbs_niter if isdefined(inputs.mbs_niter) else 10
        mbs_lambda = inputs.mbs_lambda if isdefined(inputs.mbs_lambda) else 10
        mbs_ksp = inputs.mbs_ksp if isdefined(inputs.mbs_ksp) else 10
        desc.append(
            'Changes in the susceptibility field caused by head movement were modeled with '
            f'{mbs_niter} iterations, $\\lambda$ = {mbs_lambda:g} and a {mbs_ksp:g} mm spline '
            'knot spacing [@eddysus].'
        )

    # DRBUDDI is described in its own workflow
    if run_topup:
        desc.append(topup_boilerplate(fieldmap_type, pepolar_method))

    # Whether eddy's own resampling Jacobian-modulated the corrections it applied.
    # This is independent of --ignore jacobian, which controls only the modulation
    # QSIPrep itself applies -- see eddy_modulates_distortion.
    from ..utils.eddy_config import eddy_modulates_distortion

    corrections = (
        'eddy-current and susceptibility distortion corrections'
        if run_topup or gre_in_eddy
        else 'eddy-current correction'
    )
    method = f'`{inputs.method}`' + (' [@fsllsr]' if inputs.method == 'lsr' else '')
    if eddy_modulates_distortion(eddy_config):
        desc.append(
            f'`eddy` resampled the data with its {method} method, which Jacobian-modulates the '
            f'{corrections}.'
        )
    else:
        desc.append(
            f'`eddy` resampled the data with its {method} method, which does not '
            f'Jacobian-modulate the {corrections}; QSIPrep cannot add this modulation after '
            '`eddy` has resampled the data.'
        )
    return ' '.join(desc) + '\n\n'


def topup_boilerplate(fieldmap_type, pepolar_method):
    """Write boilerplate text based on fieldmaps."""
    if fieldmap_type not in ('rpe_series', 'epi', 'synb0'):
        return ''

    applied = (
        'The estimated field was passed to `eddy`, which applied it in the same resampling as '
        'the head motion and eddy-current corrections.'
    )
    if fieldmap_type == 'synb0':
        # An un-forced SyNb0 never overrides a usable fieldmap, so reaching it
        # with fieldmaps or RPE series in the dataset means the anatomical
        # reference was forced or the fieldmaps were ignored - the boilerplate
        # must not claim none were available in those cases.
        ignored = {'fieldmaps', 'pepolar-dwis'} & set(config.workflow.ignore or [])
        if config.workflow.force_sdc_anat_reference or ignored:
            reason = (
                'The acquired field map data were not used for susceptibility distortion '
                'correction; instead, '
            )
        else:
            reason = 'No field map was available for these series, so '
        second_pass = (
            ' The *b*=0 images were then unwarped with that field (Jacobian-modulated), a new '
            'synthetic *b*=0 image was generated from them, and TOPUP was run again on the '
            'acquired *b*=0 images paired with the new synthetic one.'
            if config.workflow.two_pass_sdc
            else ''
        )
        return (
            f'\n\n{reason}the susceptibility-induced field was estimated with TOPUP [@topup] '
            'using a synthetic distortion-free *b*=0 image [@synb0disco]: the selected *b*=0 '
            'images were smoothed ($\\sigma$ = 1.15 mm) to match the synthetic image, which '
            f'entered TOPUP as an additional volume with zero total readout time.{second_pass} '
            f'{applied}'
        )

    if fieldmap_type == 'epi':
        source = (
            '*b*=0 images of the DWI series and of the field map acquired with the reverse '
            'phase-encoding polarity'
        )
    else:
        source = '*b*=0 images of DWI series acquired with opposite phase-encoding polarities'
    stage = (
        'In the first of two distortion-correction stages, the'
        if 'drbuddi' in pepolar_method.lower()
        else 'The'
    )
    tortoise = versioned('*TORTOISE*', tortoise_version())
    return (
        f'\n\n{stage} susceptibility-induced off-resonance field was estimated with TOPUP '
        f'[@topup] from {source}. The most mutually consistent *b*=0 images were chosen with '
        f'`SelectBestB0` from {tortoise} [@tortoisev4], which rigidly aligns the candidates '
        'from each distortion group before comparing them with a windowed correlation. '
        f'{applied}'
    )
