"""
Wrappers for the TORTOISE programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import logging
import os
import os.path as op
import shutil
import subprocess

import nibabel as nb
import nilearn.image as nim
import numpy as np
import pandas as pd
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.viz.utils import compose_view, cuts_from_bbox

from ..viz.utils import plot_denoise
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
)
from .epi_fmap import get_best_b0_topup_inputs_from, safe_get_3d_image
from .fmap import get_distortion_grouping
from .gradients import write_concatenated_fsl_gradients
from .images import split_bvals_bvecs, to_lps

LOGGER = logging.getLogger('nipype.interface')

# --sloppy working-grid resolution for DRBUDDI/EPIREG, in mm. TORTOISE's own
# rule upsamples to <=1mm regardless of the acquisition, which dominates
# runtime and memory; smoke tests do not need sub-voxel registration.
SLOPPY_EPI_WORKING_RES = 2.5

SLOPPY_DRBUDDI = (
    '--DRBUDDI_stage '
    r'\[learning_rate=\{0.4\},cfs=\{4:2:1\},field_smoothing=\{9:0\},'
    r'metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] '
)


def sloppy_epi_working_res():
    """DRBUDDI/EPIREG working-grid kwargs for ``--sloppy``, else empty.

    Returned as kwargs so a stock (unpatched) TORTOISE is unaffected in normal
    runs -- the flag is only emitted when ``--sloppy`` is set.
    """
    from .. import config

    if config.execution.sloppy:
        return {'epi_working_res': SLOPPY_EPI_WORKING_RES}
    return {}


class TORTOISEInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(desc='number of OMP threads')


class TORTOISECommandLine(CommandLine):
    """Support for TORTOISE commands that utilize OpenMP
    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.

    Subclasses that ship a CUDA build set ``_cuda_cmd`` and expose a ``use_cuda``
    input; see :meth:`_use_cuda`.
    """

    input_spec = TORTOISEInputSpec
    _num_threads = None
    # Name of the CUDA build of ``_cmd``, when one exists. None disables GPU
    # switching for this interface.
    _cuda_cmd = None

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not self._num_threads:
            self._num_threads = os.environ.get('OMP_NUM_THREADS', None)
            if not self._num_threads:
                self._num_threads = os.environ.get('NSLOTS', None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()
        if self._cuda_cmd is not None:
            self.inputs.on_trait_change(self._use_cuda, 'use_cuda')
            self._use_cuda()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})

    def _use_cuda(self):
        """Select the CPU or CUDA build of this TORTOISE tool.

        The CUDA build is not numerically identical to the CPU one, so switching
        changes results, not just runtime.
        """
        want_cuda = isdefined(self.inputs.use_cuda) and self.inputs.use_cuda
        # type(self)._cmd is the CPU name; self._cmd below is an instance attr.
        self._cmd = self._cuda_cmd if want_cuda else type(self)._cmd

    def run(self, **inputs):
        if 'num_threads' in inputs:
            self.inputs.num_threads = inputs['num_threads']
        self._num_threads_update()
        return super().run(**inputs)


_USE_CUDA_TRAIT_DESC = (
    'Run the CUDA build of this TORTOISE tool (<cmd>_cuda) instead of the CPU '
    'build. The GPU must be exposed to the container (docker --gpus all / '
    'apptainer --nv). Results are NOT identical to the CPU build.'
)


class _GatherDRBUDDIInputsInputSpec(TORTOISEInputSpec):
    dwi_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    bval_files = traits.Either(InputMultiObject(File(exists=True)), File(exists=True))
    bvec_files = traits.Either(InputMultiObject(File(exists=True)), File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    b0_threshold = traits.CInt(100, usedefault=True)
    epi_fmaps = InputMultiObject(
        File(exists=True), desc='files from fmaps/ for distortion correction'
    )
    raw_image_sdc = traits.Bool(True, usedefault=True)
    fieldmap_type = traits.Enum('epi', 'rpe_series', mandatory=True)
    dwi_series_pedir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-', mandatory=True)


class _GatherDRBUDDIInputsOutputSpec(TraitedSpec):
    blip_up_image = File(exists=True)
    blip_up_bmat = File(exists=True)
    blip_up_json = File(exists=True)
    blip_down_image = File(exists=True)
    blip_down_bmat = File(exists=True)
    blip_assignments = traits.List()
    report = traits.Str()


class GatherDRBUDDIInputs(SimpleInterface):
    input_spec = _GatherDRBUDDIInputsInputSpec
    output_spec = _GatherDRBUDDIInputsOutputSpec

    def _run_interface(self, runtime):
        # Write the metadata
        up_json = op.join(runtime.cwd, 'blip_up.json')
        with open(up_json, 'w') as up_jsonf:
            up_jsonf.write(f'{{"PhaseEncodingDirection": "{self.inputs.dwi_series_pedir}"}}\n')
        self._results['blip_up_json'] = up_json

        # Coerce the bvals and bvecs into lists of files
        if isinstance(self.inputs.bval_files, list) and len(self.inputs.bval_files) == 1:
            bval_files, bvec_files = split_bvals_bvecs(
                self.inputs.bval_files[0],
                self.inputs.bvec_files[0],
                deoblique=False,
                img_files=self.inputs.dwi_files,
                working_dir=runtime.cwd,
            )
        else:
            bval_files, bvec_files = self.inputs.bval_files, self.inputs.bvec_files

        if self.inputs.fieldmap_type == 'rpe_series':
            (
                self._results['blip_assignments'],
                self._results['blip_up_image'],
                self._results['blip_up_bmat'],
                self._results['blip_down_image'],
                self._results['blip_down_bmat'],
            ) = split_into_up_and_down_niis(
                dwi_files=self.inputs.dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                original_images=self.inputs.original_files,
                prefix=op.join(runtime.cwd, 'drbuddi'),
                make_bmat=True,
            )

        elif self.inputs.fieldmap_type == 'epi':
            # Use the same function that was used to get images for TOPUP, but get the images
            # directly from the CSV
            _, _, _, b0_tsv, _, _ = get_best_b0_topup_inputs_from(
                dwi_file=self.inputs.dwi_files,
                bval_file=bval_files,
                b0_threshold=self.inputs.b0_threshold,
                cwd=runtime.cwd,
                bids_origin_files=self.inputs.original_files,
                epi_fmaps=self.inputs.epi_fmaps,
                max_per_spec=True,
                raw_image_sdc=self.inputs.raw_image_sdc,
            )

            b0s_df = pd.read_table(b0_tsv)
            selected_images = b0s_df[b0s_df.selected_for_sdc].reset_index(drop=True)
            up_row = selected_images.loc[0]
            down_row = selected_images.loc[1]
            up_img = to_lps(safe_get_3d_image(up_row.bids_origin_file, up_row.original_volume))
            up_img.set_data_dtype('float32')
            down_img = to_lps(
                safe_get_3d_image(down_row.bids_origin_file, down_row.original_volume)
            )
            down_img.set_data_dtype('float32')

            # Save the images
            blip_up_nii = op.join(runtime.cwd, 'blip_up_b0.nii')
            blip_down_nii = op.join(runtime.cwd, 'blip_down_b0.nii')
            up_img.to_filename(blip_up_nii)
            down_img.to_filename(blip_down_nii)
            self._results['blip_up_image'] = blip_up_nii
            self._results['blip_down_image'] = blip_down_nii
            self._results['blip_assignments'] = split_into_up_and_down_niis(
                dwi_files=self.inputs.dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                original_images=self.inputs.original_files,
                prefix=op.join(runtime.cwd, 'drbuddi'),
                make_bmat=False,
                assignments_only=True,
            )
            self._results['blip_up_bmat'] = write_dummy_bmtxt(blip_up_nii)
            self._results['blip_down_bmat'] = write_dummy_bmtxt(blip_down_nii)

        return runtime


def write_dummy_bmtxt(nii_file):
    new_fname = fname_presuffix(nii_file, suffix='.bmtxt', use_ext=False)
    img = nim.load_img(nii_file)
    nvols = 1 if img.ndim < 4 else img.ndim.shape[3]
    with open(new_fname, 'w') as bmtxt_f:
        bmtxt_f.write('\n'.join(['0 0 0 0 0 0'] * nvols) + '\n')
    return new_fname


class _DRBUDDIInputSpec(TORTOISEInputSpec):
    num_threads = traits.Int(
        desc='number of OMP threads',
        argstr='--ncores %d',
        help='Number of cores to use in the CPU version. The default is 50% of system cores.',
        nohash=True,
    )
    blip_up_image = File(
        exists=True,
        help='Full path to the input UP NIFTI file to be corrected.',
        argstr='-u %s',
        mandatory=True,
        copyfile=True,
    )
    blip_up_bmat = File(
        exists=True,
        help='Full path to the input UP NIFTI bmtxt file.',
        mandatory=False,
        copyfile=True,
    )
    blip_up_json = File(
        exists=True,
        help='Phase encoding information will be read from this',
        argstr='--up_json %s',
        mandatory=True,
        copyfile=True,
    )
    blip_down_image = File(
        exists=True,
        help='Full path to the input DOWN NIFTI file to be corrected.',
        argstr='-d %s',
        mandatory=True,
        copyfile=True,
    )
    blip_down_bmat = File(
        exists=True,
        help='Full path to the input DOWN NIFTI bmtxt file.',
        mandatory=False,
        copyfile=True,
    )
    structural_image = InputMultiObject(
        File(exists=True, copyfile=False),
        argstr='-s %s',
        help="Path(s) to anatomical image files. Can provide more than one. NO T1W's!!",
    )
    fieldmap_type = traits.Enum('epi', 'rpe_series', mandatory=True)
    blip_assignments = traits.List()
    tensor_fit_bval_max = traits.Int(
        0,
        argstr='--DRBUDDI_DWI_bval_tensor_fitting %d',
        desc="Up to which b-value should be used for DRBUDDI's tensor fitting. "
        'Default: 0 , meaning use all b-values',
    )
    disable_initial_rigid = traits.Bool(
        False,
        argstr='--DRBUDDI_disable_initial_rigid %d',
        desc='DRBUDDI performs an initial registration between the up and down data.'
        'This registration starts with rigid, followed by a quick diffeomorphic '
        'and finalized by another rigid. This parameter, when set to 1 disables '
        'all these registrations. Default: False',
    )
    start_with_diffeomorphic_for_rigid_reg = traits.Bool(
        False,
        argstr='--DRBUDDI_start_with_diffeomorphic_for_rigid_reg',
        desc='DRBUDDI performs an initial registration between the up and down data. '
        'This registration starts with rigid, followed by a quick diffeomorphic '
        'and finalized by another rigid. This parameter, when set to 1 disables '
        'the very initial rigid registration and starts with the quick diffemorphic. '
        'This is helpful with VERY DISTORTED data, for which the initial rigid '
        'registration is problematic. Default: False',
    )
    estimate_learning_rate_per_iteration = traits.Bool(
        False,
        argstr='--DRBUDDI_estimate_LR_per_iteration %d',
        desc='Flat to estimate learning rate at every iteration. '
        'Makes DRBUDDI slower but better results. Default: False',
    )
    sloppy = traits.Bool(
        False, argstr=SLOPPY_DRBUDDI, desc='use underpowered (sloppy) registration for speed'
    )
    # Opt-in: have TORTOISE synthesize a single tensor-fittable shell for
    # DRBUDDI's [b0, FA] registration target instead of fitting a tensor to the
    # q-space grid directly. Requires a patched TORTOISE exposing the flag.
    synth_shell_bval = traits.Float(
        argstr='--DRBUDDI_synth_shell_bval %g',
        desc='b-value of the shell synthesized for the DRBUDDI target (0 = off)',
    )
    synth_shell_ndirs = traits.Int(
        argstr='--DRBUDDI_synth_shell_ndirs %d',
        desc='number of directions in the synthesized shell (default 30)',
    )
    epi_working_res = traits.Float(
        argstr='--epi_working_res %g',
        desc=(
            'Resolution (mm) of the internal registration grid used by DRBUDDI '
            'and EPIREG. TORTOISE otherwise derives it from the structural (or '
            'the DWI when no structural is given) divided by 1.3 until the mean '
            'is <= 1mm, which upsamples well past the acquired resolution: 1.7mm '
            'data becomes a 0.77mm grid, 9.5x the memory for no evident gain. '
            'Requires a patched TORTOISE that exposes --epi_working_res.'
        ),
    )
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr='--disable_itk_threads')
    use_cuda = traits.Bool(False, usedefault=True, desc=_USE_CUDA_TRAIT_DESC)


class _DRBUDDIOutputSpec(TraitedSpec):
    # Direct outputs from DRBUDDI
    undistorted_reference = File(exists=True)
    bdown_to_bup_rigid_trans_h5 = File(exists=True)
    blip_down_b0 = File(exists=True)
    blip_down_b0_corrected = File(exists=True)
    blip_down_b0_corrected_jac = File(exists=True)
    blip_down_b0_quad = File(exists=True)
    blip_up_b0 = File(exists=True)
    blip_up_b0_corrected = File(exists=True)
    blip_up_b0_corrected_jac = File(exists=True)
    blip_up_b0_quad = File(exists=True)
    deformation_finv = File(exists=True)
    deformation_minv = File(exists=True)
    blip_up_FA = File(exists=True)
    blip_down_FA = File(exists=True)
    structural_image = File(exists=True)


class DRBUDDI(TORTOISECommandLine):
    input_spec = _DRBUDDIInputSpec
    output_spec = _DRBUDDIOutputSpec
    _cmd = 'DRBUDDI'
    _cuda_cmd = 'DRBUDDI_cuda'

    def _format_arg(self, name, spec, value):
        """Trick to get blip_down_bmat symlinked without an arg"""
        if name in ('blip_down_bmat', 'blip_up_bmat'):
            return ''
        if name == 'structural_image':
            return '-s ' + ' '.join(value)
        return super()._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['undistorted_reference'] = op.abspath('b0_corrected_final.nii')
        outputs['blip_down_b0'] = op.abspath('blip_down_b0.nii')
        outputs['blip_down_b0_corrected'] = op.abspath('blip_down_b0_corrected.nii')
        outputs['blip_down_b0_corrected_jac'] = op.abspath('blip_down_b0_corrected_JAC.nii')
        outputs['blip_down_b0_quad'] = op.abspath('blip_down_b0_quad.nii')
        outputs['blip_up_b0'] = op.abspath('blip_up_b0.nii')
        outputs['blip_up_b0_corrected'] = op.abspath('blip_up_b0_corrected.nii')
        outputs['blip_up_b0_corrected_jac'] = op.abspath('blip_up_b0_corrected_JAC.nii')
        outputs['blip_up_b0_quad'] = op.abspath('blip_up_b0_quad.nii')
        outputs['deformation_finv'] = op.abspath('deformation_FINV.nii.gz')
        outputs['deformation_minv'] = op.abspath('deformation_MINV.nii.gz')

        # There will be an hdf5 transform file if there is an initial rigid
        if not self.inputs.disable_initial_rigid:
            outputs['bdown_to_bup_rigid_trans_h5'] = op.abspath('bdown_to_bup_rigidtrans.hdf5')

        # There will be FA images created if two DWI series were used as inputs
        if self.inputs.fieldmap_type == 'rpe_series':
            outputs['blip_up_FA'] = op.abspath('blip_up_FA.nii')
            outputs['blip_down_FA'] = op.abspath('blip_down_FA.nii')

        # If there was a T2w
        if self.inputs.structural_image:
            outputs['structural_image'] = op.abspath('structural_used.nii')
        return outputs


class _DRBUDDIAggregateOutputsInputSpec(TORTOISEInputSpec):
    blip_assignments = traits.List()
    undistorted_reference = File(exists=True)
    bdown_to_bup_rigid_trans_h5 = File(exists=True)
    undistorted_reference = File(exists=True)
    blip_down_b0 = File(exists=True)
    blip_down_b0_corrected = File(exists=True)
    blip_down_b0_corrected_jac = File(exists=True)
    blip_down_b0_quad = File(exists=True)
    blip_up_b0 = File(exists=True)
    blip_up_b0_corrected = File(exists=True)
    blip_up_b0_corrected_jac = File(exists=True)
    blip_up_b0_quad = File(exists=True)
    deformation_finv = File(exists=True, desc='blip up to b0_corrected')
    deformation_minv = File(exists=True)
    blip_up_FA = File(exists=True)
    blip_down_FA = File(exists=True)
    fieldmap_type = traits.Enum('epi', 'rpe_series', mandatory=True)
    structural_image = File(exists=True)
    wm_seg = File(exists=True, desc='White matter segmentation image')


class _DRBUDDIAggregateOutputsOutputSpec(TraitedSpec):
    # Aggregated outputs for convenience
    sdc_warps = OutputMultiObject(File(exists=True))
    sdc_scaling_images = OutputMultiObject(File(exists=True))
    # Fieldmap outputs for the reports
    up_fa_corrected_image = File(exists=True)
    down_fa_corrected_image = File(exists=True)
    # The best image for coregistration to the corrected DWI
    b0_ref = File(exists=True)


class DRBUDDIAggregateOutputs(SimpleInterface):
    input_spec = _DRBUDDIAggregateOutputsInputSpec
    output_spec = _DRBUDDIAggregateOutputsOutputSpec

    def _run_interface(self, runtime):
        # If the structural image has been used, return that as the b0ref, otherwise
        # it's the b0_corrected_final
        self._results['b0_ref'] = (
            self.inputs.structural_image
            if isdefined(self.inputs.structural_image)
            else self.inputs.undistorted_reference
        )

        # there may be 2 transforms for the blip down data. If so, compose them
        if isdefined(self.inputs.bdown_to_bup_rigid_trans_h5):
            # combine the rigid with displacement
            down_warp = op.join(runtime.cwd, 'blip_down_composite.nii.gz')
            xfm = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_down_b0,
                transforms=[self.inputs.deformation_minv, self.inputs.bdown_to_bup_rigid_trans_h5],
                reference_image=self.inputs.undistorted_reference,
                output_image=down_warp,
                print_out_composite_warp_file=True,
                interpolation='LanczosWindowedSinc',
            )
            xfm.terminal_output = 'allatonce'
            xfm.resource_monitor = False
            _ = xfm.run()
        else:
            down_warp = self.inputs.deformation_minv

        # Calculate the scaling images
        scaling_blip_up_file = op.join(runtime.cwd, 'blip_up_scale.nii.gz')
        scaling_blip_down_file = op.join(runtime.cwd, 'blip_down_scale.nii.gz')
        scaling_blip_up_img = nim.math_img(
            'a/b', a=self.inputs.undistorted_reference, b=self.inputs.blip_up_b0_corrected
        )
        scaling_blip_up_img.to_filename(scaling_blip_up_file)
        scaling_blip_down_img = nim.math_img(
            'a/b', a=self.inputs.undistorted_reference, b=self.inputs.blip_down_b0_corrected
        )
        scaling_blip_down_img.to_filename(scaling_blip_down_file)

        self._results['sdc_warps'] = [
            self.inputs.deformation_finv if blip_dir == 'up' else down_warp
            for blip_dir in self.inputs.blip_assignments
        ]
        self._results['sdc_scaling_images'] = [
            scaling_blip_up_file if blip_dir == 'up' else scaling_blip_down_file
            for blip_dir in self.inputs.blip_assignments
        ]

        if self.inputs.fieldmap_type == 'rpe_series':
            fa_up_warped = fname_presuffix(
                self.inputs.blip_up_FA, newpath=runtime.cwd, suffix='_corrected'
            )
            xfm_fa_up = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_up_FA,
                transforms=[self.inputs.deformation_finv],
                reference_image=self.inputs.undistorted_reference,
                output_image=fa_up_warped,
                interpolation='NearestNeighbor',
            )
            xfm_fa_up.terminal_output = 'allatonce'
            xfm_fa_up.resource_monitor = False
            xfm_fa_up.run()

            fa_down_warped = fname_presuffix(
                self.inputs.blip_down_FA, newpath=runtime.cwd, suffix='_corrected'
            )
            xfm_fa_down = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_down_FA,
                transforms=[self.inputs.deformation_minv, self.inputs.bdown_to_bup_rigid_trans_h5],
                reference_image=self.inputs.undistorted_reference,
                output_image=fa_down_warped,
                interpolation='NearestNeighbor',
            )
            xfm_fa_down.terminal_output = 'allatonce'
            xfm_fa_down.resource_monitor = False
            xfm_fa_down.run()
            self._results['up_fa_corrected_image'] = fa_up_warped
            self._results['down_fa_corrected_image'] = fa_down_warped

        return runtime


class _GibbsInputSpec(TORTOISEInputSpec, SeriesPreprocReportInputSpec):
    """Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75)
    phase_encoding_dir nsh minW(optional) maxW(optional)"""

    in_file = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    out_file = traits.File(
        argstr='%s',
        position=1,
        name_source='in_file',
        name_template='%s_unrung.nii',
        use_extension=False,
    )
    kspace_coverage = traits.Float(mandatory=True, position=2, argstr='%.4f')
    phase_encoding_dir = traits.Enum(
        0, 1, mandatory=True, argstr='%d', position=3, desc='0: horizontal, 1:vertical'
    )
    nsh = traits.Int(argstr='%d', position=4)
    min_w = traits.Int()
    mask = File()
    num_threads = traits.Int(1, usedefault=True, nohash=True)


class _GibbsOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(exists=True)


class Gibbs(SeriesPreprocReport, TORTOISECommandLine):
    input_spec = _GibbsInputSpec
    output_spec = _GibbsOutputSpec
    _cmd = 'Gibbs'

    def _get_plotting_images(self):
        input_dwi = nim.load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = nim.load_img(ref_name)
        return input_dwi, denoised_nii, None

    def _generate_report(self):
        """Generate a reportlet."""
        LOGGER.info('Generating denoising visual report')

        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
        image_intensities = np.array([img.mean() for img in image_data.T])
        lowb_index = int(np.argmax(image_intensities))
        highb_index = int(np.argmin(image_intensities))

        # Original images
        orig_lowb_nii = input_dwi.slicer[..., lowb_index]
        orig_highb_nii = input_dwi.slicer[..., highb_index]

        # Denoised images
        denoised_lowb_nii = denoised_nii.slicer[..., lowb_index]
        denoised_highb_nii = denoised_nii.slicer[..., highb_index]

        # Find spatial extent of the image
        contour_nii = mask_nii = None
        if isdefined(self.inputs.mask):
            contour_nii = nim.load_img(self.inputs.mask)
        else:
            mask_nii = nim.threshold_img(denoised_lowb_nii, 50)
        cuts = cuts_from_bbox(contour_nii or mask_nii, cuts=self._n_cuts)

        diff_lowb_nii = nb.Nifti1Image(
            orig_lowb_nii.get_fdata() - denoised_lowb_nii.get_fdata(),
            affine=denoised_lowb_nii.affine,
        )
        diff_highb_nii = nb.Nifti1Image(
            orig_highb_nii.get_fdata() - denoised_highb_nii.get_fdata(),
            affine=denoised_highb_nii.affine,
        )

        # Call composer
        compose_view(
            plot_denoise(
                denoised_lowb_nii,
                denoised_highb_nii,
                'moving-image',
                estimate_brightness=True,
                cuts=cuts,
                label='De-Gibbs',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            plot_denoise(
                diff_lowb_nii,
                diff_highb_nii,
                'fixed-image',
                estimate_brightness=True,
                cuts=cuts,
                label='Estimated Ringing',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            out_file=self._out_report,
        )

        self._calculate_nmse(input_dwi, denoised_nii)


class _TORTOISEConvertInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True, copyfile=True)
    bvec_file = File(exists=True, mandatory=True, copyfile=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)


class _TORTOISEConvertOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    mask_file = File(exists=True)
    bmtxt_file = File(exists=True)


class TORTOISEConvert(SimpleInterface):
    input_spec = _TORTOISEConvertInputSpec
    output_spec = _TORTOISEConvertOutputSpec

    def _run_interface(self, runtime):
        """Convert gzipped niftis and bval/bvec into TORTOISE format.

        TORTOISEProcess requires the ``.bmtxt`` file to sit in the same
        directory as the ``.nii`` and share the same basename (it locates the
        bmtxt from the nii by extension swap).
        """
        dwi_file = fname_presuffix(
            self.inputs.dwi_file, newpath=runtime.cwd, use_ext=False, suffix='.nii'
        )
        dwi_img = nim.load_img(self.inputs.dwi_file, dtype='float32')
        dwi_img.set_data_dtype('float32')
        dwi_img.to_filename(dwi_file)
        src_bmtxt = make_bmat_file(self.inputs.bval_file, self.inputs.bvec_file)
        # Co-locate the bmtxt with the renamed DWI and give it a matching stem
        # so TORTOISEProcess can pair them.
        bmtxt_file = op.splitext(dwi_file)[0] + '.bmtxt'
        if op.abspath(src_bmtxt) != op.abspath(bmtxt_file):
            shutil.copyfile(src_bmtxt, bmtxt_file)

        if isdefined(self.inputs.mask_file):
            mask_file = fname_presuffix(
                self.inputs.mask_file, newpath=runtime.cwd, use_ext=False, suffix='.nii'
            )
            mask_img = nim.load_img(self.inputs.mask_file, dtype='float32')
            mask_img.set_data_dtype('float32')
            mask_img.to_filename(mask_file)
            self._results['mask_file'] = mask_file

        self._results['dwi_file'] = dwi_file
        self._results['bmtxt_file'] = bmtxt_file

        return runtime


def split_into_up_and_down_niis(
    dwi_files,
    bval_files,
    bvec_files,
    original_images,
    prefix,
    make_bmat=True,
    assignments_only=False,
):
    """Takes the concatenated output from pre_hmc_wf and split it into "up" and "down"
    decompressed nifti files with float32 datatypes."""
    group_names, group_assignments = get_distortion_grouping(original_images)

    if not len(set(group_names)) == 2 and not assignments_only:
        raise Exception('DRBUDDI requires exactly one blip up and one blip down')

    up_images = []
    up_bvals = []
    up_bvecs = []
    up_prefix = prefix + '_up_dwi'
    up_dwi_file = up_prefix + '.nii'
    up_bmat_file = up_prefix + '.bmtxt'
    down_images = []
    down_bvals = []
    down_bvecs = []
    down_prefix = prefix + '_down_dwi'
    down_dwi_file = down_prefix + '.nii'
    down_bmat_file = down_prefix + '.bmtxt'

    # We know up is first because we concatenated them ourselves
    up_group_name = group_assignments[0]
    blip_assignments = []
    for dwi_file, bval_file, bvec_file, distortion_group in zip(
        dwi_files, bval_files, bvec_files, group_assignments, strict=False
    ):
        if distortion_group == up_group_name:
            up_images.append(dwi_file)
            up_bvals.append(bval_file)
            up_bvecs.append(bvec_file)
            blip_assignments.append('up')
        else:
            down_images.append(dwi_file)
            down_bvals.append(bval_file)
            down_bvecs.append(bvec_file)
            blip_assignments.append('down')

    if assignments_only:
        return blip_assignments

    # Write the 4d up image
    up_4d = nim.concat_imgs(up_images, dtype='float32', auto_resample=False)
    up_4d.set_data_dtype('float32')
    up_4d.to_filename(up_dwi_file)
    up_bval_file, up_bvec_file = write_concatenated_fsl_gradients(up_bvals, up_bvecs, up_prefix)

    # Write the 4d down image
    down_4d = nim.concat_imgs(down_images, dtype='float32', auto_resample=False)
    down_4d.set_data_dtype('float32')
    down_4d.to_filename(down_dwi_file)
    down_bval_file, down_bvec_file = write_concatenated_fsl_gradients(
        down_bvals, down_bvecs, down_prefix
    )

    # Send back FSL-style gradients
    if not make_bmat:
        return (
            blip_assignments,
            up_dwi_file,
            up_bval_file,
            up_bvec_file,
            down_dwi_file,
            down_bval_file,
            down_bvec_file,
        )

    # Convert to bmatrix text file
    make_bmat_file(up_bval_file, up_bvec_file)
    make_bmat_file(down_bval_file, down_bvec_file)

    return blip_assignments, up_dwi_file, up_bmat_file, down_dwi_file, down_bmat_file


def make_bmat_file(bvals, bvecs):
    pout = subprocess.run(['FSLBVecsToTORTOISEBmatrix', op.abspath(bvals), op.abspath(bvecs)])
    print(pout)
    return bvals.replace('bval', 'bmtxt')


def bmtxt_to_fsl(bmtxt_file, working_dir=None):
    """Convert a TORTOISE 6-col bmtxt file to FSL bval/bvec text files.

    Wraps the upstream ``TORTOISEBmatrixToFSLBVecs`` binary, which writes
    sibling files with extensions ``.bvecs`` and ``.bvals``. Returns
    ``(bval_path, bvec_path)``.
    """
    bmtxt_abs = op.abspath(bmtxt_file)
    if working_dir is not None:
        # Copy into the working directory so outputs land there
        dst = op.join(working_dir, op.basename(bmtxt_abs))
        if dst != bmtxt_abs:
            shutil.copyfile(bmtxt_abs, dst)
        bmtxt_abs = dst
    subprocess.run(['TORTOISEBmatrixToFSLBVecs', bmtxt_abs], check=True)
    base = bmtxt_abs[: -len('.bmtxt')]
    return base + '.bvals', base + '.bvecs'


def generate_diffprep_boilerplate(correction_mode):
    """Methods boilerplate describing the DIFFPREP HMC backend.

    ``correction_mode`` comes from ``--diffprep-config`` (default
    ``quadratic``), so the transform model has to be named from it rather than
    assumed.
    """
    mode_desc = {
        'motion': ('rigid head motion only', 'rigid-body transform'),
        'quadratic': (
            'rigid head motion together with quadratic eddy currents',
            '24-parameter Okan-quadratic transform',
        ),
        'cubic': (
            'rigid head motion together with cubic eddy currents',
            'cubic transform',
        ),
    }
    corrects, transform = mode_desc[correction_mode]
    return (
        '\n\nHead motion correction was performed with DIFFPREP '
        '[@diffprep], part of the TORTOISE [@tortoisev4] software package, '
        f'in {correction_mode} mode (correcting {corrects}). DIFFPREP fits a '
        'SHORE/MAPMRI signal model to the data and iteratively registers each '
        f"volume to a model-predicted target using TORTOISE's {transform}. "
        'The corrected volumes and motion-rotated bmatrix were then passed to '
        'the rest of the pipeline.\n\n'
    )


def generate_drbuddi_boilerplate(fieldmap_type, t2w_sdc, with_topup=False):
    """Generate boilerplate that describes how DRBUDDI is being used."""

    desc = ['\n\nDRBUDDI [@drbuddi], part of the TORTOISE [@tortoisev4] software package,']
    if not with_topup:
        # Until now there will have been no description of the SDC procedure.
        # Add extra details about the input data.
        desc.append(
            'was used to perform susceptibility distortion correction. '
            'Data was collected with reversed phase-encode blips, resulting '
            'in pairs of images with distortions going in opposite directions.'
        )
    else:
        desc += ['was used to perform a second stage of distortion correction.']

    # Describe what's going on
    if fieldmap_type == 'epi':
        desc.append(
            'DRBUDDI used b=0 reference images with reversed phase encoding directions to estimate'
        )
    else:
        desc.append(
            'DRBUDDI used multiple motion-corrected DWI series acquired '
            'with opposite phase encoding '
            'directions. A b=0 image **and** the Fractional Anisotropy '
            'images from both phase encoding diesctions were used together in '
            'a multi-modal registration to estimate'
        )
    desc.append('the susceptibility-induced off-resonance field.')

    if t2w_sdc:
        desc.append('A T2-weighted image was included in the multimodal registration.')
    desc.append(
        'Signal intensity was adjusted '
        'in the final interpolated images using a method similar to LSR.\n\n'
    )
    return ' '.join(desc)


class _DIFFPREPInputSpec(TORTOISEInputSpec):
    # TORTOISEProcess sizes its thread pool from the HOST core count (ignoring
    # cgroups/affinity) and calls omp_set_num_threads() itself, so neither
    # OMP_NUM_THREADS nor its percent setting can bound it under a scheduler;
    # --ncores is an absolute count read by the patched TORTOISE.
    ncores = traits.Int(
        argstr='--ncores %d',
        desc='absolute number of CPU cores TORTOISE may use',
    )
    # TORTOISE treats the GPU as one more worker rather than offloading the
    # series (npass = ceil(Nvols / (NGPUs*ratio + ncores - NGPUs))); the right
    # ratio is a property of the machine. Left unset, TORTOISE keeps its default.
    gpu_cpu_ratio = traits.Int(
        argstr='--gpu_cpu_ratio %d',
        desc='volumes the GPU processes per pass, against one per CPU thread',
    )
    dwi_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        argstr='-u %s',
        desc='Uncompressed 4D NIfTI passed to TORTOISEProcess as the UP image. '
        'A sibling ``.bmtxt`` and ``.json`` must be present.',
    )
    bmtxt_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='Sibling TORTOISE 6-column bmatrix file. Found by TORTOISEProcess '
        "from the .nii's basename; not passed as its own argstr.",
    )
    json_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='Sibling BIDS JSON sidecar with PhaseEncodingDirection. Found by '
        "TORTOISEProcess from the .nii's basename; not passed as its own argstr.",
    )
    correction_mode = traits.Enum(
        'motion',
        'quadratic',
        'cubic',
        argstr='-c %s',
        mandatory=True,
        desc='Motion & eddy correction mode forwarded to TORTOISE. '
        '"motion" = rigid only; "quadratic" = rigid + quadratic eddy '
        '(recommended); "cubic" = rigid + cubic eddy.',
    )
    b0_id = traits.Int(
        -1,
        usedefault=True,
        argstr='--b0_id %d',
        desc='Index of b=0 volume to use as the registration target. '
        '-1 (default) lets TORTOISE pick the best one.',
    )
    is_human_brain = traits.Bool(
        True,
        usedefault=True,
        argstr='--is_human_brain %d',
        desc='Whether the data is an in-vivo human brain. Enables iterative '
        'SHORE-prediction refinement.',
    )
    rot_eddy_center = traits.Enum(
        'isocenter',
        'center_voxel',
        'center_slice',
        usedefault=True,
        argstr='--rot_eddy_center %s',
        desc='Rotation and eddy-currents center.',
    )
    niter = traits.Int(
        argstr='--niter %d',
        desc='Number of iterations for the high-b / SHORE-prediction refinement '
        'loop. Zero disables all iterative correction, which is much faster. '
        'Left unset by default so TORTOISE picks its own default.',
    )
    extra_args = traits.List(
        traits.Str(),
        usedefault=True,
        desc='Additional flags appended verbatim to the TORTOISEProcess command. '
        'Use to access TORTOISE knobs not surfaced as first-class fields '
        '(e.g. ["--big_delta", "0.030"]).',
    )
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr='--disable_itk_threads')
    use_cuda = traits.Bool(False, usedefault=True, desc=_USE_CUDA_TRAIT_DESC)
    epi_mode = traits.Enum(
        'off',
        'T2Wreg',
        usedefault=True,
        desc='TORTOISE --epi mode. "off": no in-TORTOISE SDC (default). '
        '"T2Wreg": structural EPI correction using --structural_image.',
    )
    structural_image = File(
        exists=True,
        desc='T2w structural image for --epi T2Wreg (must NOT be a T1w). '
        'Required when epi_mode == "T2Wreg".',
    )


class _DIFFPREPOutputSpec(TraitedSpec):
    corrected_dwi_file = File(
        exists=True, desc='Motion + eddy corrected 4D DWI, in the input grid.'
    )
    corrected_bmtxt_file = File(exists=True, desc='Motion-rotated 6-col bmatrix.')
    transformations_file = File(
        exists=True,
        desc='Per-volume 24-parameter Okan-quadratic transforms as written by DIFFPREP.',
    )
    # epi_mode == 'T2Wreg' only. The EPI stage's displacement field is emitted as
    # a transform rather than baked into the image, so qsiprep can compose it with
    # HMC and coregistration and resample once -- the same contract DRBUDDI has.
    sdc_warp = File(desc='EPI (susceptibility) displacement field, in the DWI world frame.')
    b0_up_image = File(desc='b=0 before the EPI correction (for the SDC report).')
    b0_up_corrected_image = File(desc='b=0 after the EPI correction (for the SDC report).')
    structural_image = File(desc='The T2w TORTOISE actually registered against.')


class DIFFPREP(TORTOISECommandLine):
    """TORTOISE V4 DIFFPREP motion + eddy-current correction.

    Runs the ``TORTOISEProcess`` binary starting from the Import step (so the
    proc.nii/proc.bmtxt/proc.json working files get staged) with all stages
    other than motion+eddy (and optionally the EPI/``T2Wreg`` stage) disabled
    via their own flags. See ``_base_step_flags`` for why ``--step import``
    (not ``--step motioneddy``) is required. Emits the per-volume corrected
    DWI, rotated bmatrix, and 24-parameter transform file.
    """

    input_spec = _DIFFPREPInputSpec
    output_spec = _DIFFPREPOutputSpec
    _cmd = 'TORTOISEProcess'
    _cuda_cmd = 'TORTOISEProcess_cuda'

    # Hardcoded flags that put TORTOISE into "DIFFPREP-only" mode. qsiprep
    # already runs its own denoising / gibbs / SDC stages, so we explicitly
    # turn each of them off -- but we still START from the Import step.
    #
    # NOTE: ``--epi`` is appended separately in ``_parse_inputs`` so it can
    # switch between ``off`` (SDC handled downstream by qsiprep) and ``T2Wreg``
    # (structural EPI correction performed here and baked into the output).
    _base_step_flags = (
        '--step import '
        '--denoising off '
        '--gibbs 0 '
        '--drift off '
        '--do_QC 0 '
        '--remove_temp 0 '
        '--s2v 0 '
        '--repol 0'
    )

    def _format_arg(self, name, spec, value):
        # bmtxt_file and json_file are passed via sibling-stem lookup, not argstr
        if name in ('bmtxt_file', 'json_file'):
            return ''
        return super()._format_arg(name, spec, value)

    def _parse_inputs(self, skip=None):
        skip = list(skip or [])
        # These are handled manually below (no argstr / sibling-file lookup).
        skip += ['epi_mode', 'structural_image', 'extra_args']
        parsed = super()._parse_inputs(skip=skip)
        parsed.append(self._base_step_flags)

        # EPI (susceptibility) stage.
        if self.inputs.epi_mode == 'T2Wreg':
            if not isdefined(self.inputs.structural_image):
                raise ValueError('epi_mode="T2Wreg" requires a structural_image (T2w).')
            parsed.append(f'--epi T2Wreg -s {op.abspath(self.inputs.structural_image)}')
        else:
            parsed.append('--epi off')

        if isdefined(self.inputs.extra_args) and self.inputs.extra_args:
            parsed.append(' '.join(self.inputs.extra_args))
        return parsed

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # TORTOISE stages its working files into a temp-proc subfolder of the
        # input's directory and runs DIFFPREP on a "_proc" copy there. With
        # ``--step import`` the import step creates
        #   <dir>/<stem>_temp_proc/<stem>_proc.nii (+ .bmtxt, .json)
        # and DIFFPREP's motion+eddy outputs land alongside as
        #   <dir>/<stem>_temp_proc/<stem>_proc_moteddy.{nii,bmtxt}
        #   <dir>/<stem>_temp_proc/<stem>_proc_moteddy_transformations.txt
        # nipype copies the input into the node cwd (copyfile=True), so resolve
        # against the cwd copy when present.
        nii_path = op.abspath(self.inputs.dwi_file)
        cwd_nii = op.join(os.getcwd(), op.basename(nii_path))
        base_nii = cwd_nii if op.exists(cwd_nii) else nii_path
        stem = op.basename(base_nii).removesuffix('.nii')
        temp_proc = op.join(op.dirname(base_nii), stem + '_temp_proc')
        proc_base = op.join(temp_proc, stem + '_proc')

        # The per-volume transforms are the motion+eddy parameters themselves, so
        # they always come from that step regardless of what follows.
        outputs['transformations_file'] = proc_base + '_moteddy_transformations.txt'

        # Motion+eddy output, in the input's own grid. This is what qsiprep wants
        # in EVERY case -- including T2Wreg, see below.
        outputs['corrected_dwi_file'] = proc_base + '_moteddy.nii'
        outputs['corrected_bmtxt_file'] = proc_base + '_moteddy.bmtxt'

        if self.inputs.epi_mode == 'T2Wreg':
            # Deliberately NOT the ``*_TORTOISE_final.nii`` FINALDATA image:
            # passing ``-s`` also runs StructuralAlignment + FinalData, which
            # would pull coregistration and ACPC alignment forward into HMC.
            # Emit the EPI stage's displacement field instead, so qsiprep
            # composes it with HMC and coregistration and resamples once -- the
            # same contract as the standalone DRBUDDI path. There is no way to
            # stop TORTOISE before StructuralAlignment (``--step`` only sets the
            # *start* step); those stages still run and their output is unused.
            sdc_warp = op.join(temp_proc, 'deformation_FINV.nii.gz')
            if not op.exists(sdc_warp):
                raise FileNotFoundError(
                    f'epi_mode="T2Wreg": expected the EPI displacement field {sdc_warp}, '
                    'found none.'
                )
            outputs['sdc_warp'] = sdc_warp
            # Named outputs so nipype's remove_unnecessary_outputs keeps them --
            # they are the before/after pair the SDC report needs.
            for key, fname in (
                ('b0_up_image', 'blip_up_b0.nii'),
                ('b0_up_corrected_image', 'blip_up_b0_corrected.nii'),
                ('structural_image', 'structural_used.nii'),
            ):
                candidate = op.join(temp_proc, fname)
                if op.exists(candidate):
                    outputs[key] = candidate
        return outputs


def _read_okan_transformations(path):
    """Read a ``_moteddy_transformations.txt`` file and return 24-element
    parameter vectors. Accepts both the VNL bracketed ``[a, b, ...]`` form and
    plain whitespace-separated rows."""
    rows = []
    with open(path) as fobj:
        for line in fobj:
            line = line.strip().strip('[]')
            if not line:
                continue
            vals = [float(x) for x in line.replace(',', ' ').split()]
            if len(vals) < 24:
                raise ValueError(f'expected 24 columns per row in {path}, got {len(vals)}')
            rows.append(np.asarray(vals[:24], dtype=float))
    if not rows:
        raise ValueError(f'no transform rows found in {path}')
    return rows


class _DIFFPREPMotionParamsInputSpec(BaseInterfaceInputSpec):
    transformations_file = File(
        exists=True,
        mandatory=True,
        desc='DIFFPREP _moteddy_transformations.txt file with 24 columns per volume.',
    )


class _DIFFPREPMotionParamsOutputSpec(TraitedSpec):
    spm_motion_file = File(exists=True)


class DIFFPREPMotionParams(SimpleInterface):
    """Extract a 6-column SPM-style motion-parameters file from DIFFPREP's
    24-parameter transform file.

    The output columns are the leading 6 parameters of TORTOISE's
    ``OkanQuadraticTransform`` in SPM realignment-parameter order
    (translation_x/y/z in mm of LPS physical coordinate, rotation_x/y/z as
    Euler angles in radians). The remaining 18 Okan parameters encode the
    eddy-current polynomial + rotation/eddy centre and are intentionally
    dropped -- they are not rigid head motion. Units match the eddy and
    SHORELine SPM motion files (translation mm, rotation radians).
    """

    input_spec = _DIFFPREPMotionParamsInputSpec
    output_spec = _DIFFPREPMotionParamsOutputSpec

    def _run_interface(self, runtime):
        rows = _read_okan_transformations(self.inputs.transformations_file)
        params = np.asarray(rows, dtype=float)
        spm_motion = params[:, :6]
        spm_motion_file = fname_presuffix(
            self.inputs.transformations_file,
            suffix='_spm_rp.txt',
            use_ext=False,
            newpath=runtime.cwd,
        )
        np.savetxt(spm_motion_file, spm_motion)
        self._results['spm_motion_file'] = spm_motion_file
        return runtime


class _DIFFPREPSplitOutputsInputSpec(BaseInterfaceInputSpec):
    corrected_dwi_file = File(exists=True, mandatory=True)
    corrected_bmtxt_file = File(exists=True, mandatory=True)
    b0_threshold = traits.CInt(100, usedefault=True)


class _DIFFPREPSplitOutputsOutputSpec(TraitedSpec):
    dwi_files = OutputMultiObject(File(exists=True))
    bvec_files = OutputMultiObject(File(exists=True))
    bval_files = OutputMultiObject(File(exists=True))
    b0_indices = traits.List(traits.Int())
    forward_transforms = traits.List(File(exists=True))


class DIFFPREPSplitOutputs(SimpleInterface):
    """Split TORTOISE's corrected 4D DWI + 6-col bmatrix into per-volume files
    that match the qsiprep contract: per-volume dwi/bval/bvec triples plus a
    list of identity ITK transforms (DIFFPREP has already baked the motion+eddy
    correction into the volumes, so downstream apply-transform nodes must be
    no-ops)."""

    input_spec = _DIFFPREPSplitOutputsInputSpec
    output_spec = _DIFFPREPSplitOutputsOutputSpec

    _identity_itk = (
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: MatrixOffsetTransformBase_double_3_3\n'
        'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n'
        'FixedParameters: 0 0 0\n'
    )

    def _run_interface(self, runtime):
        dwi_img = nb.load(self.inputs.corrected_dwi_file)
        nvols = 1 if dwi_img.ndim < 4 else dwi_img.shape[3]

        # Convert TORTOISE bmtxt -> FSL bvals/bvecs
        bval_path, bvec_path = bmtxt_to_fsl(self.inputs.corrected_bmtxt_file, runtime.cwd)

        # Split the 4D image into per-volume niftis under runtime.cwd
        dwi_data = np.asanyarray(dwi_img.dataobj)
        if dwi_img.ndim < 4:
            dwi_data = dwi_data[..., np.newaxis]
        per_vol_dwis = []
        base = op.join(runtime.cwd, 'diffprep_vol')
        for vol_idx in range(nvols):
            vol_img = nb.Nifti1Image(
                dwi_data[..., vol_idx].astype('float32'),
                dwi_img.affine,
                dwi_img.header,
            )
            vol_img.set_data_dtype('float32')
            vol_path = f'{base}_{vol_idx:04d}.nii.gz'
            vol_img.to_filename(vol_path)
            per_vol_dwis.append(vol_path)

        # deoblique=True routes the gradients through the same mrtrix-based RAS+
        # conversion ``SplitDWIsFSL(deoblique_bvecs=True)`` uses for the eddy
        # backend: TORTOISEBmatrixToFSLBVecs emits FSL-convention (voxel-frame)
        # bvecs, which are orientation- and obliquity-dependent, and the
        # conversion removes that dependence when the output grid is oblique.
        per_vol_bvals, per_vol_bvecs = split_bvals_bvecs(
            bval_path,
            bvec_path,
            deoblique=True,
            img_files=per_vol_dwis,
            working_dir=runtime.cwd,
        )

        # Build b0 indices list from the bvals
        bvals_arr = np.loadtxt(bval_path).reshape(-1)
        b0_indices = [int(i) for i, b in enumerate(bvals_arr) if b < self.inputs.b0_threshold]

        # Per-volume identity ITK affines (TORTOISE has already baked the
        # correction into the volume images themselves).
        forward_transforms = []
        for vol_idx in range(nvols):
            xfm_path = op.join(runtime.cwd, f'diffprep_identity_{vol_idx:04d}.txt')
            with open(xfm_path, 'w') as fobj:
                fobj.write(self._identity_itk)
            forward_transforms.append(xfm_path)

        self._results['dwi_files'] = per_vol_dwis
        self._results['bvec_files'] = per_vol_bvecs
        self._results['bval_files'] = per_vol_bvals
        self._results['b0_indices'] = b0_indices
        self._results['forward_transforms'] = forward_transforms
        return runtime


def _distortion_group_assignments(original_files):
    """Return a per-volume group label in ``{1, 2}`` for a reverse-PE series.

    Wraps :func:`get_distortion_grouping` (which numbers groups by the order
    their acqp first appears while iterating the *sorted* unique originals) and
    re-labels so that **the group of the first volume is always ``1``**. This is
    the same "up is first" convention :func:`split_into_up_and_down_niis` uses
    inside ``GatherDRBUDDIInputs``, so the diffprep split and DRBUDDI's re-split
    agree on which volumes are the "+"/"up" side.
    """
    _, raw_groups = get_distortion_grouping(original_files)
    n_groups = len(set(raw_groups))
    if n_groups != 2:
        raise ValueError(
            'rpe_series DIFFPREP requires exactly two phase-encoding groups; '
            f'found {n_groups} in the original files.'
        )
    first = raw_groups[0]
    return [1 if grp == first else 2 for grp in raw_groups]


class _SplitDWIsByDistortionGroupInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(
        exists=True,
        mandatory=True,
        desc='Merged 4D DWI from pre_hmc (already denoised + b0-harmonized across '
        'the two PE directions). This is the file DIFFPREP must NOT see as a whole.',
    )
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    original_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='One original BIDS file per volume; the distortion group of each '
        'volume is read from these (same partition GatherDRBUDDIInputs uses).',
    )
    pe_axis = traits.Enum(
        'i',
        'j',
        'k',
        mandatory=True,
        desc='Phase-encoding axis (no sign). The first-appearing group is labelled '
        "'+' and the second '-'. DIFFPREP is sign-agnostic on the axis "
        "(DIFFPREP.cxx maps both 'j' and 'j-' to \"vertical\"), so this sign only "
        'documents provenance; it does not change the eddy model.',
    )
    b0_threshold = traits.CInt(100, usedefault=True)


class _SplitDWIsByDistortionGroupOutputSpec(TraitedSpec):
    group1_dwi_file = File(exists=True)
    group1_bval_file = File(exists=True)
    group1_bvec_file = File(exists=True)
    group1_pe_dir = traits.Str()
    group2_dwi_file = File(exists=True)
    group2_bval_file = File(exists=True)
    group2_bvec_file = File(exists=True)
    group2_pe_dir = traits.Str()
    group_assignments = traits.List(
        traits.Int(), desc='Per-volume group id (1 or 2) in merged order.'
    )


class SplitDWIsByDistortionGroup(SimpleInterface):
    """Re-split an already-merged reverse-PE DWI series back into its two
    phase-encoding groups so DIFFPREP can be run once per direction.

    DIFFPREP models a single phase axis against a single b=0 reference for the
    whole file (``TORTOISEProcess`` runs DIFFPREP once per PE direction, see
    ``TORTOISE.cxx`` ``for(int PE=0;PE<2;PE++)``). qsiprep concatenates the two
    PE directions into one 4D file for FSL eddy in ``pre_hmc``; feeding that
    concatenation to a single DIFFPREP run silently mis-corrects, so we undo the
    merge here (keeping pre_hmc's per-PE denoising + b0 harmonization) and emit
    one single-PE series per group. The merge is intentionally left in place --
    DRBUDDI re-splits the *recombined* series itself and relies on the shared
    b0-intensity scale the merge established.
    """

    input_spec = _SplitDWIsByDistortionGroupInputSpec
    output_spec = _SplitDWIsByDistortionGroupOutputSpec

    def _run_interface(self, runtime):
        assignments = _distortion_group_assignments(self.inputs.original_files)

        dwi_img = nb.load(self.inputs.dwi_file)
        nvols = 1 if dwi_img.ndim < 4 else dwi_img.shape[3]
        if nvols != len(assignments):
            raise ValueError(
                f'{self.inputs.dwi_file} has {nvols} volumes but '
                f'{len(assignments)} original files were supplied.'
            )
        dwi_data = np.asanyarray(dwi_img.dataobj)
        if dwi_img.ndim < 4:
            dwi_data = dwi_data[..., np.newaxis]

        bvals = np.loadtxt(self.inputs.bval_file).reshape(-1)
        bvecs = np.loadtxt(self.inputs.bvec_file)
        # FSL stores bvecs as 3 x N; transpose if a stray N x 3 slips through.
        if bvecs.ndim == 2 and bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
            bvecs = bvecs.T
        bvecs = np.atleast_2d(bvecs)

        self._results['group_assignments'] = assignments
        for group_id, sign in ((1, ''), (2, '-')):
            idx = [i for i, grp in enumerate(assignments) if grp == group_id]
            prefix = op.join(runtime.cwd, f'group{group_id}')

            grp_dwi = op.abspath(prefix + '_dwi.nii.gz')
            grp_img = nb.Nifti1Image(
                dwi_data[..., idx].astype('float32'), dwi_img.affine, dwi_img.header
            )
            grp_img.set_data_dtype('float32')
            grp_img.to_filename(grp_dwi)

            grp_bval = op.abspath(prefix + '.bval')
            grp_bvec = op.abspath(prefix + '.bvec')
            np.savetxt(grp_bval, bvals[idx].reshape(1, -1), fmt='%g')
            np.savetxt(grp_bvec, bvecs[:, idx], fmt='%.8f')

            # Warn when this side lacks a robust b=0 pool. TORTOISE's
            # select_best_b0 (--b0_id -1) silently degrades to "use the first
            # b=0" when there are too few, which yields a poor registration
            # reference for the whole side.
            grp_bvals = bvals[idx]
            n_b0ish = int(np.sum(np.abs(grp_bvals - grp_bvals.min()) <= 30))
            if n_b0ish < 4:
                LOGGER.warning(
                    'rpe_series DIFFPREP group %d has only %d b=0-like volumes '
                    '(within 30 s/mm^2 of its minimum b-value). TORTOISE may fall '
                    'back to a single b=0 reference for this phase-encoding '
                    'direction, degrading its motion/eddy correction.',
                    group_id,
                    n_b0ish,
                )

            self._results[f'group{group_id}_dwi_file'] = grp_dwi
            self._results[f'group{group_id}_bval_file'] = grp_bval
            self._results[f'group{group_id}_bvec_file'] = grp_bvec
            self._results[f'group{group_id}_pe_dir'] = self.inputs.pe_axis + sign

        return runtime


class _ConcatenateDIFFPREPGroupsInputSpec(BaseInterfaceInputSpec):
    group1_dwi_file = File(exists=True, mandatory=True)
    group1_bmtxt_file = File(exists=True, mandatory=True)
    group1_transformations_file = File(exists=True, mandatory=True)
    group2_dwi_file = File(exists=True, mandatory=True)
    group2_bmtxt_file = File(exists=True, mandatory=True)
    group2_transformations_file = File(exists=True, mandatory=True)
    group_assignments = traits.List(
        traits.Int(),
        mandatory=True,
        desc='Per-volume group id (1 or 2) in the original merged order, as '
        'produced by SplitDWIsByDistortionGroup.',
    )


class _ConcatenateDIFFPREPGroupsOutputSpec(TraitedSpec):
    corrected_dwi_file = File(exists=True)
    corrected_bmtxt_file = File(exists=True)
    transformations_file = File(exists=True)


class ConcatenateDIFFPREPGroups(SimpleInterface):
    """Recombine the two per-PE-direction DIFFPREP outputs into one series in
    the original (merged) volume order.

    Each group was corrected by its own DIFFPREP run, so the two corrected 4D
    files live in **different corrected spaces** -- that is expected and correct;
    reconciling those two spaces is exactly DRBUDDI's job. This node only
    restores the per-volume order (so ``original_files`` still lines up with the
    recombined series that DRBUDDI re-splits) and stitches the rotated bmatrix
    and per-volume transforms to match. The emitted triple is a drop-in for a
    single DIFFPREP node's ``corrected_dwi_file`` / ``corrected_bmtxt_file`` /
    ``transformations_file``, so every downstream node is unchanged.
    """

    input_spec = _ConcatenateDIFFPREPGroupsInputSpec
    output_spec = _ConcatenateDIFFPREPGroupsOutputSpec

    def _run_interface(self, runtime):
        assignments = self.inputs.group_assignments

        img1 = nb.load(self.inputs.group1_dwi_file)
        img2 = nb.load(self.inputs.group2_dwi_file)
        data1 = np.asanyarray(img1.dataobj)
        data2 = np.asanyarray(img2.dataobj)
        if data1.ndim < 4:
            data1 = data1[..., np.newaxis]
        if data2.ndim < 4:
            data2 = data2[..., np.newaxis]
        if data1.shape[:3] != data2.shape[:3]:
            raise ValueError(
                'The two per-direction DIFFPREP outputs are on different voxel '
                f'grids ({data1.shape[:3]} vs {data2.shape[:3]}); they must share '
                'a grid to be recombined into one series.'
            )

        bmat1 = np.atleast_2d(np.loadtxt(self.inputs.group1_bmtxt_file))
        bmat2 = np.atleast_2d(np.loadtxt(self.inputs.group2_bmtxt_file))
        xf1 = _read_okan_transformations(self.inputs.group1_transformations_file)
        xf2 = _read_okan_transformations(self.inputs.group2_transformations_file)

        counts = {
            1: (data1.shape[3], bmat1.shape[0], len(xf1)),
            2: (data2.shape[3], bmat2.shape[0], len(xf2)),
        }
        for group_id, (n_dwi, n_bmat, n_xf) in counts.items():
            n_expected = assignments.count(group_id)
            if not n_dwi == n_bmat == n_xf == n_expected:
                raise ValueError(
                    f'group {group_id} DIFFPREP outputs disagree on volume count: '
                    f'dwi={n_dwi}, bmtxt={n_bmat}, transforms={n_xf}, '
                    f'assignments={n_expected}.'
                )

        # Walk the original order, pulling the next available volume/row from
        # whichever group each position belongs to. Handles interleaved groups,
        # not just contiguous plus-then-minus blocks.
        sources = {
            1: {'data': data1, 'bmat': bmat1, 'xf': xf1},
            2: {'data': data2, 'bmat': bmat2, 'xf': xf2},
        }
        cursors = {1: 0, 2: 0}
        out_vols = []
        out_bmat = []
        out_xf = []
        for group_id in assignments:
            src = sources[group_id]
            k = cursors[group_id]
            out_vols.append(src['data'][..., k])
            out_bmat.append(src['bmat'][k])
            out_xf.append(src['xf'][k])
            cursors[group_id] = k + 1

        corrected_dwi = op.join(runtime.cwd, 'diffprep_rpe_corrected.nii.gz')
        combined = np.stack(out_vols, axis=-1).astype('float32')
        combined_img = nb.Nifti1Image(combined, img1.affine, img1.header)
        combined_img.set_data_dtype('float32')
        combined_img.to_filename(corrected_dwi)

        corrected_bmtxt = op.join(runtime.cwd, 'diffprep_rpe_corrected.bmtxt')
        np.savetxt(corrected_bmtxt, np.vstack(out_bmat), fmt='%.10g')

        transforms_file = op.join(runtime.cwd, 'diffprep_rpe_transformations.txt')
        np.savetxt(transforms_file, np.vstack(out_xf), fmt='%.10g')

        self._results['corrected_dwi_file'] = corrected_dwi
        self._results['corrected_bmtxt_file'] = corrected_bmtxt
        self._results['transformations_file'] = transforms_file
        return runtime


def _tortoise_heuristic_deltas(bmtxt_file):
    """Replicate TORTOISE's internal small/big-delta heuristic.

    ``EstimateMAPMRI`` derives diffusion timings from the maximum b-value when
    they are not supplied (``small_delta = (b_max / gyro^2 / G^2 / 2 * 1e6)^(1/3)
    * 1000`` with ``G = 2 * 40 mT/m`` and ``big_delta = 3 * small_delta``, in ms).
    ``SynthesizeDWIsFromMAPMRI`` has no such fallback, so we compute the deltas
    once and pass the identical values to both tools. The b-value of each volume
    is the trace of its 6-column b-matrix row (columns 0, 3, 5)."""
    bmat = np.loadtxt(bmtxt_file)
    if bmat.ndim == 1:
        bmat = bmat[np.newaxis, :]
    bvals = bmat[:, 0] + bmat[:, 3] + bmat[:, 5]
    max_bval = float(np.max(bvals))

    gyro = 267.51532e6
    grad_strength = 2.0 * 40e-3  # TORTOISE assumes 2 * 40 mT/m
    temp = max_bval / gyro / gyro / grad_strength / grad_strength / 2.0 * 1e6
    small_delta = temp ** (1.0 / 3.0) * 1000.0  # ms, matching TORTOISE
    big_delta = small_delta * 3.0
    return small_delta, big_delta


class _SynthesizeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='corrected 4D DWI (TORTOISE output)')
    bmtxt_file = File(exists=True, mandatory=True, desc='corrected 6-col TORTOISE bmtxt')
    mask_file = File(exists=True, mandatory=True, desc='brain mask (any grid; resampled to DWI)')
    map_order = traits.Int(4, usedefault=True, desc='MAPMRI order for the QC model fit')
    num_threads = traits.Int(desc='OMP threads for the TORTOISE estimators')


class _SynthesizeDWIsOutputSpec(TraitedSpec):
    synth_dwi_file = File(exists=True, desc='4D model-synthesized DWI at the corrected gradients')
    per_volume_synth = OutputMultiObject(File(exists=True), desc='per-volume synthesized images')
    qc_mask = File(exists=True, desc='brain mask resampled onto the corrected DWI grid')


class SynthesizeDWIs(SimpleInterface):
    """Fit a MAPMRI model to a corrected DWI and synthesize an "ideal" volume
    at every measured gradient, for slice-wise QC.

    Runs ``EstimateTensor`` -> ``EstimateMAPMRI`` -> ``SynthesizeDWIsFromMAPMRI``
    entirely within the node's working directory, copying the DWI and its
    ``.bmtxt`` sibling into ``runtime.cwd`` with a fixed stem so output
    discovery is unambiguous."""

    input_spec = _SynthesizeDWIsInputSpec
    output_spec = _SynthesizeDWIsOutputSpec

    def _run_interface(self, runtime):
        cwd = runtime.cwd
        stem = 'diffprep_corrected'

        # TORTOISE wants uncompressed float32 niftis and finds the b-matrix as
        # a sibling file with the same stem (<stem>.bmtxt).
        nii_path = op.join(cwd, stem + '.nii')
        dwi_img = nim.load_img(self.inputs.dwi_file, dtype='float32')
        dwi_img.set_data_dtype('float32')
        dwi_img.to_filename(nii_path)

        bmtxt_path = op.join(cwd, stem + '.bmtxt')
        shutil.copyfile(self.inputs.bmtxt_file, bmtxt_path)

        # Resample the mask onto the corrected DWI grid so it is valid both as
        # a TORTOISE --mask and as the SliceQC mask (which indexes the data by
        # slice and therefore requires an identical 3D shape/affine).
        ref_3d = nim.index_img(dwi_img, 0)
        mask_img = nim.resample_to_img(self.inputs.mask_file, ref_3d, interpolation='nearest')
        mask_img.set_data_dtype('float32')
        mask_path = op.join(cwd, stem + '_mask.nii')
        mask_img.to_filename(mask_path)

        # Identical, self-consistent deltas for fit and synthesis.
        small_delta, big_delta = _tortoise_heuristic_deltas(bmtxt_path)

        env = dict(os.environ)
        num_threads = self.inputs.num_threads if isdefined(self.inputs.num_threads) else None
        if not num_threads:
            num_threads = int(env.get('OMP_NUM_THREADS', 1) or 1)
        env['OMP_NUM_THREADS'] = str(num_threads)

        def _run(cmd):
            LOGGER.info('SynthesizeDWIs: %s', ' '.join(map(str, cmd)))
            subprocess.run(cmd, check=True, cwd=cwd, env=env)

        # 1. Tensor (provides DT + A0 for the MAPMRI fit and the synthesizer).
        dt_path = op.join(cwd, stem + '_L1_DT.nii')
        am_path = op.join(cwd, stem + '_L1_AM.nii')
        _run(['EstimateTensor', '--input', nii_path, '--mask', mask_path])

        # 2. MAPMRI coefficients, deltas pinned explicitly.
        coeffs_path = op.join(cwd, stem + '_mapmri.nii')
        _run([
            'EstimateMAPMRI',
            '--input', nii_path,
            '--mask', mask_path,
            '--dti', dt_path,
            '--A0', am_path,
            '--map_order', str(self.inputs.map_order),
            '--small_delta', f'{small_delta:.7f}',
            '--big_delta', f'{big_delta:.7f}',
        ])  # fmt:skip

        # 3. Synthesize an ideal DWI at the corrected gradient table. The tool
        #    is positional and writes <coeffs-stem>_synth.nii next to coeffs.
        _run([
            'SynthesizeDWIsFromMAPMRI',
            coeffs_path,
            dt_path,
            am_path,
            f'{small_delta:.7f}',
            f'{big_delta:.7f}',
            bmtxt_path,
        ])  # fmt:skip
        synth_path = coeffs_path[: -len('.nii')] + '_synth.nii'
        if not op.exists(synth_path):
            raise FileNotFoundError(f'SynthesizeDWIsFromMAPMRI did not produce {synth_path}')

        # Split into per-volume images in native (corrected-bmtxt) order so the
        # i-th synthesized volume pairs with the i-th corrected volume that
        # DIFFPREPSplitOutputs emits.
        synth_img = nb.load(synth_path)
        synth_data = np.asanyarray(synth_img.dataobj)
        if synth_data.ndim < 4:
            synth_data = synth_data[..., np.newaxis]
        per_vol = []
        for vol_idx in range(synth_data.shape[3]):
            vol_img = nb.Nifti1Image(
                synth_data[..., vol_idx].astype('float32'),
                synth_img.affine,
                synth_img.header,
            )
            vol_img.set_data_dtype('float32')
            vol_path = op.join(cwd, f'synth_vol_{vol_idx:04d}.nii.gz')
            vol_img.to_filename(vol_path)
            per_vol.append(vol_path)

        self._results['synth_dwi_file'] = synth_path
        self._results['per_volume_synth'] = per_vol
        self._results['qc_mask'] = mask_path
        return runtime
