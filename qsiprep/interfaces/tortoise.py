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

SLOPPY_DRBUDDI = (
    '--DRBUDDI_stage '
    r'\[learning_rate=\{0.4\},cfs=\{4:2:1\},field_smoothing=\{9:0\},'
    r'metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] '
)


class TORTOISEInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(desc='number of OMP threads')


class TORTOISECommandLine(CommandLine):
    """Support for TORTOISE commands that utilize OpenMP
    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.
    """

    input_spec = TORTOISEInputSpec
    _num_threads = None

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

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})

    def run(self, **inputs):
        if 'num_threads' in inputs:
            self.inputs.num_threads = inputs['num_threads']
        self._num_threads_update()
        return super().run(**inputs)


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
    nvols = 1 if img.ndim < 4 else img.shape[3]
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
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr='--disable_itk_threads')


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
    # mask_file is OPTIONAL. The downstream TORTOISEProcess (DIFFPREP) binary
    # does NOT consume an externally-supplied mask — its `DPCreateMask()`
    # method (DIFFPREP.cxx:2338-2358) reads `b0_mask_img` from the
    # registration settings file, and if that's empty (the default in
    # qsiprep, which doesn't populate it) it auto-generates a mask via
    # `create_mask(b0_img, noise_img)` on the b0 itself. The implementation
    # of TORTOISEConvert._run_interface already guards on `isdefined()` for
    # this input, but a mistaken `mandatory=True` here used to short-circuit
    # the run before that guard could trigger. Relaxed to optional so the
    # auto-mask path is reachable; behaviour is unchanged when a mask is
    # actually wired in.
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
        directory as the ``.nii`` and share the same basename (it locates
        the bmtxt from the nii by extension swap). The DWI gets renamed
        here to ``runtime.cwd/<input-dwi-stem>.nii`` — and its stem
        typically carries pre-HMC suffixes like ``_LPS_denoised_mrdegibbs``
        added by the upstream pre_hmc_wf. The bmtxt, however, was being
        written by ``make_bmat_file`` *next to the bval file* with the
        bval's stem (which lacks those pre-HMC suffixes — bvals don't go
        through denoise/degibbs). The mismatch caused TORTOISEProcess to
        fail with:

            Either the .bmtxt file or the .bvecs/.bvals file should be
            present in the same folder as the up data file and should
            have the same basename.

        Fix: after ``make_bmat_file`` writes the bmtxt next to the bval,
        copy it to runtime.cwd with the DWI's basename so the pair is
        co-located and same-stemmed.
        """
        dwi_file = fname_presuffix(
            self.inputs.dwi_file, newpath=runtime.cwd, use_ext=False, suffix='.nii'
        )
        dwi_img = nim.load_img(self.inputs.dwi_file, dtype='float32')
        dwi_img.set_data_dtype('float32')
        dwi_img.to_filename(dwi_file)
        src_bmtxt = make_bmat_file(self.inputs.bval_file, self.inputs.bvec_file)
        # Co-locate the bmtxt with the renamed DWI and give it a matching
        # stem so TORTOISEProcess can pair them.
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
            import shutil

            shutil.copyfile(bmtxt_abs, dst)
        bmtxt_abs = dst
    subprocess.run(['TORTOISEBmatrixToFSLBVecs', bmtxt_abs], check=True)
    base = bmtxt_abs[: -len('.bmtxt')]
    return base + '.bvals', base + '.bvecs'


# ---------------------------------------------------------------------------
# DIFFPREP (TORTOISE) HMC backend
# ---------------------------------------------------------------------------


class _DIFFPREPInputSpec(TORTOISEInputSpec):
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
    extra_args = traits.List(
        traits.Str(),
        usedefault=True,
        desc='Additional flags appended verbatim to the TORTOISEProcess command. '
        'Use to access TORTOISE knobs not surfaced as first-class fields '
        '(e.g. ["--big_delta", "0.030"]).',
    )
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr='--disable_itk_threads')


class _DIFFPREPOutputSpec(TraitedSpec):
    corrected_dwi_file = File(exists=True, desc='Motion + eddy corrected 4D DWI (uncompressed).')
    corrected_bmtxt_file = File(exists=True, desc='Motion-rotated 6-col bmatrix.')
    transformations_file = File(
        exists=True,
        desc='Per-volume 24-parameter Okan-quadratic transforms as written by DIFFPREP.',
    )


class DIFFPREP(TORTOISECommandLine):
    """TORTOISE V4 DIFFPREP motion + eddy-current correction.

    Runs the ``TORTOISEProcess`` binary starting from the Import step (so the
    proc.nii/proc.bmtxt/proc.json working files get staged) with all stages
    other than motion+eddy disabled via their own flags (``--step import
    --denoising off --gibbs 0 --drift off --epi off --repol 0 --s2v 0``).
    See ``_step_flags`` for why ``--step import`` (not ``--step motioneddy``)
    is required. Emits the per-volume corrected DWI, rotated bmatrix, and
    24-parameter transform file.
    """

    input_spec = _DIFFPREPInputSpec
    output_spec = _DIFFPREPOutputSpec
    _cmd = 'TORTOISEProcess'

    # Hardcoded flags that put TORTOISE into "DIFFPREP-only" mode. qsiprep
    # already runs its own denoising / gibbs / SDC stages, so we explicitly
    # turn each of them off — but we still START from the Import step.
    #
    # IMPORTANT: `--step` sets TORTOISE's *start* step, not "the only step to
    # run". The Import step (TORTOISE::CheckAndCopyInputData) is what stages
    # the working files TORTOISE needs — proc.nii / proc.bmtxt / and crucially
    # proc.json — into the temp_proc folder. TORTOISE::Process() then reads
    # proc.json *unconditionally* (TORTOISE.cxx ~line 531). Using
    # `--step motioneddy` skips Import (the gate at TORTOISE.cxx:514 is an
    # exact `== STEPS::Import` match), so proc.json is never created and the
    # subsequent read aborts with an nlohmann "unexpected end of input"
    # parse error. Starting from `import` runs the full chain, but the
    # intervening stages are no-ops given the flags below:
    #   * `--denoising off` → DenoiseData only estimates noise; it does NOT
    #     modify the DWI (TORTOISE.cxx:1394 branch), so no double-denoising.
    #   * `--gibbs 0`       → GibbsUnringData early-returns (guarded by
    #     `gibbs_option`, TORTOISE.cxx:1284), so no double-unringing.
    #   * `--drift off`, `--epi off`, `--repol 0`, `--s2v 0` similarly gate
    #     their respective stages off.
    # MotionEddy (DIFFPREP) and everything after it run identically to the
    # old `--step motioneddy` setting, so this only ADDS the (required)
    # Import plus two verified no-op stages.
    _step_flags = (
        '--step import '
        '--denoising off '
        '--gibbs 0 '
        '--drift off '
        '--epi off '
        '--do_QC 0 '
        '--remove_temp 0 '
        '--s2v 0 '
        '--repol 0'
    )

    def _format_arg(self, name, spec, value):
        # bmtxt_file and json_file are passed via sibling-stem lookup, not argstr
        if name in ('bmtxt_file', 'json_file'):
            return ''
        # extra_args gets joined verbatim
        if name == 'extra_args':
            return ' '.join(value) if value else ''
        return super()._format_arg(name, spec, value)

    def _parse_inputs(self, skip=None):
        parsed = super()._parse_inputs(skip=skip)
        # Append the hardcoded "DIFFPREP-only" flags to the end of the command.
        parsed.append(self._step_flags)
        return parsed

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # TORTOISE stages all its working files into a temp-proc subfolder
        # of the input's directory and runs DIFFPREP on a "_proc" copy there.
        # With `--step import` (see _step_flags), the import step creates
        #   <dir>/<stem>_temp_proc/<stem>_proc.nii   (+ .bmtxt, .json)
        # and DIFFPREP's motion+eddy outputs land alongside it as
        #   <dir>/<stem>_temp_proc/<stem>_proc_moteddy.{nii,bmtxt}
        #   <dir>/<stem>_temp_proc/<stem>_proc_moteddy_transformations.txt
        # NOT as <dir>/<stem>_moteddy.* in the node cwd (which is what an
        # earlier version of this method incorrectly assumed). TORTOISE's
        # *_TORTOISE_final.* output does land in the cwd, but it has no
        # transformations sidecar, so we anchor all three outputs on the
        # consistent _proc_moteddy triple from the same DIFFPREP step.
        #
        # The temp-proc folder is named from the `-u` input TORTOISE was
        # given: nipype copies the input into the node cwd (copyfile=True),
        # so resolve against the cwd copy when present.
        nii_path = op.abspath(self.inputs.dwi_file)
        cwd_nii = op.join(os.getcwd(), op.basename(nii_path))
        base_nii = cwd_nii if op.exists(cwd_nii) else nii_path
        stem = op.basename(base_nii)
        if stem.endswith('.nii'):
            stem = stem[: -len('.nii')]
        temp_proc = op.join(op.dirname(base_nii), stem + '_temp_proc')
        proc_base = op.join(temp_proc, stem + '_proc')
        outputs['corrected_dwi_file'] = proc_base + '_moteddy.nii'
        outputs['corrected_bmtxt_file'] = proc_base + '_moteddy.bmtxt'
        outputs['transformations_file'] = proc_base + '_moteddy_transformations.txt'
        return outputs


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
    ``OkanQuadraticTransform`` (see ``itkOkanQuadraticTransform.hxx``
    ``SetParameters``), in SPM realignment-parameter order::

        translation_x  translation_y  translation_z   rotation_x  rotation_y  rotation_z
        <-------- mm (LPS physical) -------->          <------ radians ------>

    parameters[0:3] are added directly to the LPS physical coordinate (so
    millimetres); parameters[3:6] are the Euler angles fed to the cos/sin
    rotation matrix (so radians). The remaining 18 Okan parameters encode the
    eddy-current polynomial + rotation/eddy centre and are intentionally
    dropped here — they are not rigid head motion.

    Units match the eddy (``Eddy2SPMMotion``) and SHORELine
    (``CombineMotions``) SPM motion files: translation mm, rotation radians.
    Note the rotation *parameterisation* differs across backends — eddy and
    DIFFPREP report Euler angles, SHORELine reports an axis-angle rotation
    vector — so the rotation columns are only directly comparable between
    backends in the small-angle regime; translation is exactly comparable.
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


# ---------------------------------------------------------------------------
# OkanQuadraticTransform decomposition
#
# DIFFPREP estimates a 24-parameter OkanQuadraticTransform per volume. The
# transform applies a rigid motion + scaling to all three axes, then overwrites
# the *phase-encoding* axis with a polynomial in the rigid-transformed
# coordinates. That structure is exactly:
#
#     OkanTransform = rigid(motion)  ∘  voxel_shift_map(eddy, along PE only)
#
# Both pieces are nitransforms-native: an Affine + a DenseFieldTransform whose
# data has zeros on the two non-PE components. Composed in order (affine first,
# then warp), they reproduce the C++ OkanQuadraticTransform::TransformPoint
# output exactly. See itkOkanQuadraticTransform.hxx in TORTOISEV4.
#
# All math happens in TORTOISE's "DP" coordinate frame, then is re-expressed in
# image-physical coords (LPS). See DIFFPREP.cxx::ChangeImageHeaderToDP.
# ---------------------------------------------------------------------------


_PHASE_AXIS_FROM_BIDS = {'i': 0, 'j': 1, 'k': 2}


def _dp_offset(ref_affine, ref_shape, rot_eddy_center):
    """Physical-space offset such that ``c_DP = D^T (c_image - offset)``.

    Matches the three modes of ``DIFFPREP::ChangeImageHeaderToDP``.
    """
    Dspac = ref_affine[:3, :3]
    spacing = np.linalg.norm(Dspac, axis=0)
    origin = ref_affine[:3, 3]

    if rot_eddy_center == 'isocenter':
        return np.zeros(3)
    if rot_eddy_center == 'center_voxel':
        center_idx = (np.array(ref_shape, dtype=float) - 1.0) / 2.0
        return Dspac @ center_idx + origin
    if rot_eddy_center == 'center_slice':
        center_idx = (np.array(ref_shape, dtype=float) - 1.0) / 2.0
        center_point = Dspac @ center_idx + origin
        out = np.zeros(3)
        out[2] = center_point[2]
        return out
    raise ValueError(f'unknown rot_eddy_center: {rot_eddy_center!r}')


def _build_rotation(angle_x, angle_y, angle_z):
    """Z · Y · X Euler rotation matrix, matching OkanQuadraticTransform::ComputeMatrix."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _phase_axis_dp(phase_encoding_direction):
    """Map a BIDS PhaseEncodingDirection ('i', 'j', 'k' with optional '-') to
    a 0/1/2 DP voxel-axis index. Accepts an int already in [0, 1, 2]."""
    if isinstance(phase_encoding_direction, int):
        if phase_encoding_direction not in (0, 1, 2):
            raise ValueError(f'phase axis must be in 0/1/2: {phase_encoding_direction!r}')
        return phase_encoding_direction
    return _PHASE_AXIS_FROM_BIDS[phase_encoding_direction.rstrip('-')]


def okan_decompose(
    params24,
    ref_image,
    phase_encoding_direction='j',
    rot_eddy_center='isocenter',
    do_cubic=None,
):
    """Split a 24-parameter Okan-quadratic transform into a 4x4 affine plus a
    dense displacement field, both in image-physical (LPS) coordinates.

    Parameters
    ----------
    params24 : array-like, shape (24,)
        One row from DIFFPREP's ``*_moteddy_transformations.txt``.
    ref_image : nibabel image
        Reference grid (size + spacing + direction + origin) for the field.
    phase_encoding_direction : str or int
        BIDS ``PhaseEncodingDirection`` (``'i'``/``'j'``/``'k'`` with optional
        ``'-'``) or the equivalent integer voxel axis index.
    rot_eddy_center : str
        ``'isocenter'`` (default), ``'center_voxel'`` or ``'center_slice'``.
    do_cubic : bool, optional
        If ``None``, inferred from whether any of params[14:21] are non-zero.

    Returns
    -------
    affine_4x4 : ndarray, shape (4, 4)
        Rigid (+ scaling) part as a homogeneous matrix in image-physical
        coordinates. Maps output (b0_template) coordinates to input
        (per-volume) coordinates, matching ITK / ANTs convention.
    disp_field : ndarray, shape (X, Y, Z, 3)
        Dense displacement field. Non-zero entries lie only along the
        image-physical direction of the DP phase axis (i.e., the
        corresponding column of the reference direction matrix).

    Notes
    -----
    The composition ``A · x + disp(x)`` reproduces ``OkanTransform(x)``
    exactly (up to floating-point rounding) for every voxel x.
    """
    params = np.asarray(params24, dtype=float).reshape(-1)
    if params.size != 24:
        raise ValueError(f'expected 24 parameters, got {params.size}')

    phase_axis = _phase_axis_dp(phase_encoding_direction)
    if do_cubic is None:
        do_cubic = bool(np.any(np.abs(params[14:21]) > 0))

    ref_affine = np.asarray(ref_image.affine, dtype=float)
    ref_shape = tuple(ref_image.shape[:3])
    Dspac = ref_affine[:3, :3]
    spacing = np.linalg.norm(Dspac, axis=0)
    direction = Dspac / spacing[None, :]
    origin = ref_affine[:3, 3]

    # Image <-> DP coord transforms. c_DP = D^T (c_image - offset).
    offset = _dp_offset(ref_affine, ref_shape, rot_eddy_center)
    M_to_dp = np.eye(4)
    M_to_dp[:3, :3] = direction.T
    M_to_dp[:3, 3] = -direction.T @ offset
    M_from_dp = np.eye(4)
    M_from_dp[:3, :3] = direction
    M_from_dp[:3, 3] = offset

    # Rigid part of the Okan transform in DP coords:
    #   y = R (c_DP - center) + T + center
    center_okan = params[21:24]
    T_okan = params[0:3]
    R_okan = _build_rotation(params[3], params[4], params[5])
    A_rigid_dp = np.eye(4)
    A_rigid_dp[:3, :3] = R_okan
    A_rigid_dp[:3, 3] = T_okan + center_okan - R_okan @ center_okan

    # Compose into image-physical coords: A_image = M_from_dp · A_rigid_dp · M_to_dp
    A_image = M_from_dp @ A_rigid_dp @ M_to_dp

    # Dense VSM along the DP phase axis.
    indices = np.indices(ref_shape, dtype=float).reshape(3, -1)  # (3, N)
    c_image = (Dspac @ indices).T + origin  # (N, 3)
    c_dp = (M_to_dp[:3, :3] @ c_image.T).T + M_to_dp[:3, 3]  # (N, 3) in DP

    # p = R (c_DP - center) + T  (the same `p` the C++ polynomial sees)
    p = (R_okan @ (c_dp - center_okan).T).T + T_okan
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]

    new_phase = (
        params[6] * px + params[7] * py + params[8] * pz
        + params[9] * px * py + params[10] * px * pz + params[11] * py * pz
        + params[12] * (px * px - py * py)
        + params[13] * (2.0 * pz * pz - px * px - py * py)
    )
    if do_cubic:
        new_phase += (
            params[14] * px * py * pz
            + params[15] * pz * (px * px - py * py)
            + params[16] * px * (4.0 * pz * pz - px * px - py * py)
            + params[17] * py * (4.0 * pz * pz - px * px - py * py)
            + params[18] * px * (px * px - 3.0 * py * py)
            + params[19] * py * (3.0 * px * px - py * py)
            + params[20] * pz * (2.0 * pz * pz - 3.0 * px * px - 3.0 * py * py)
        )

    # The eddy "shift" along the DP phase axis: how much the polynomial
    # diverges from the rigid output on that axis.
    shift_scalar = new_phase - p[:, phase_axis]

    # Express as a 3-component physical displacement in image coords. The DP
    # phase axis points along D[:, phase_axis] in image-physical-space.
    disp_dir = direction[:, phase_axis]
    disp_field = (shift_scalar[:, None] * disp_dir[None, :]).reshape(
        *ref_shape, 3
    )

    return A_image, disp_field


# ---------------------------------------------------------------------------
# nipype wrapper around okan_decompose
# ---------------------------------------------------------------------------


_ITK_AFFINE_TEMPLATE = (
    '#Insight Transform File V1.0\n'
    '#Transform 0\n'
    'Transform: MatrixOffsetTransformBase_double_3_3\n'
    'Parameters: {p}\n'
    'FixedParameters: 0 0 0\n'
)


def write_itk_affine(matrix_4x4, out_path):
    """Write a 4x4 image-physical-space affine as an ITK ``.txt`` transform
    file. Fixed center is the world origin; rotation matrix is stored row-major
    followed by the translation."""
    M = matrix_4x4[:3, :3]
    t = matrix_4x4[:3, 3]
    params = list(M.flatten()) + list(t)
    with open(out_path, 'w') as fobj:
        fobj.write(_ITK_AFFINE_TEMPLATE.format(p=' '.join(f'{v:.10g}' for v in params)))
    return out_path


def write_itk_warp(disp_field, ref_affine, out_path):
    """Write a (X, Y, Z, 3) displacement field as an ITK/ANTs-compatible 5D
    NIfTI (shape ``(X, Y, Z, 1, 3)``, intent code ``NIFTI_INTENT_VECTOR``)."""
    # ITK uses a singleton 4th axis between spatial dims and the vector dim
    field = np.asarray(disp_field, dtype='<f4')
    field = field.reshape(*field.shape[:3], 1, 3)
    img = nb.Nifti1Image(field, ref_affine)
    img.header.set_intent('vector')
    img.to_filename(out_path)
    return out_path


class _DIFFPREPDecomposeTransformsInputSpec(BaseInterfaceInputSpec):
    transformations_file = File(
        exists=True,
        mandatory=True,
        desc="DIFFPREP's _moteddy_transformations.txt (24 cols per volume).",
    )
    reference_image = File(
        exists=True,
        mandatory=True,
        desc='3D reference image defining the grid for the per-volume warp '
        'fields (typically the corrected b=0 average).',
    )
    phase_encoding_direction = traits.Enum(
        'i', 'i-', 'j', 'j-', 'k', 'k-',
        mandatory=True,
        desc='BIDS PhaseEncodingDirection.',
    )
    rot_eddy_center = traits.Enum(
        'isocenter', 'center_voxel', 'center_slice',
        usedefault=True,
        desc="TORTOISE's --rot_eddy_center setting (must match what was used "
        'when transformations_file was produced).',
    )
    do_cubic = traits.Bool(
        desc='Whether cubic params (14-20) should be evaluated. Inferred from '
        'the data when not set.',
    )


class _DIFFPREPDecomposeTransformsOutputSpec(TraitedSpec):
    affine_files = OutputMultiObject(File(exists=True))
    warp_files = OutputMultiObject(File(exists=True))


class DIFFPREPDecomposeTransforms(SimpleInterface):
    """Decompose each row of DIFFPREP's transformations.txt into an ITK affine
    file + a dense displacement field NIfTI, both in image-physical (LPS)
    coordinates.

    The two together reproduce the original ``OkanQuadraticTransform`` when
    applied in the order ``[affine, warp]`` — i.e., they slot directly into
    qsiprep's downstream ``apply_hmc_transforms`` (or any nitransforms /
    ANTs composition) for single-shot resampling.
    """

    input_spec = _DIFFPREPDecomposeTransformsInputSpec
    output_spec = _DIFFPREPDecomposeTransformsOutputSpec

    def _run_interface(self, runtime):
        params_per_vol = _read_okan_transformations(self.inputs.transformations_file)
        ref_img = nb.load(self.inputs.reference_image)
        if ref_img.ndim > 3:
            # Squeeze any 4th singleton or pick the first volume for a 4D series
            if ref_img.shape[3] == 1:
                ref_img = ref_img.slicer[..., 0]
            else:
                ref_img = ref_img.slicer[..., 0]
        do_cubic = self.inputs.do_cubic if isdefined(self.inputs.do_cubic) else None

        affine_files = []
        warp_files = []
        for vol_idx, params in enumerate(params_per_vol):
            A, disp = okan_decompose(
                params,
                ref_img,
                phase_encoding_direction=self.inputs.phase_encoding_direction,
                rot_eddy_center=self.inputs.rot_eddy_center,
                do_cubic=do_cubic,
            )
            affine_path = op.join(runtime.cwd, f'diffprep_affine_{vol_idx:04d}.txt')
            warp_path = op.join(runtime.cwd, f'diffprep_warp_{vol_idx:04d}.nii.gz')
            write_itk_affine(A, affine_path)
            write_itk_warp(disp, ref_img.affine, warp_path)
            affine_files.append(affine_path)
            warp_files.append(warp_path)

        self._results['affine_files'] = affine_files
        self._results['warp_files'] = warp_files
        return runtime


def _read_okan_transformations(path):
    """Read a ``_moteddy_transformations.txt`` file and yield 24-element
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
                raise ValueError(
                    f'expected 24 columns per row in {path}, got {len(vals)}'
                )
            rows.append(np.asarray(vals[:24], dtype=float))
    if not rows:
        raise ValueError(f'no transform rows found in {path}')
    return rows


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
    that match the qsiprep contract: per-volume dwi/bval/bvec triples plus
    a list of identity ITK transforms (DIFFPREP has already baked the
    motion+eddy correction into the volumes, so downstream apply-transform
    nodes must be no-ops)."""

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
        from .images import split_bvals_bvecs

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

        # Split the FSL gradients into per-volume txt files
        per_vol_bvals, per_vol_bvecs = split_bvals_bvecs(
            bval_path,
            bvec_path,
            deoblique=False,
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


def write_diffprep_json(json_file, phase_encoding_direction, working_dir=None):
    """Write a minimal BIDS sidecar JSON next to a DWI nifti so that
    TORTOISEProcess can read PhaseEncodingDirection from it. Returns the
    path of the file written."""
    import json

    target = op.join(working_dir, op.basename(json_file)) if working_dir else json_file
    payload = {'PhaseEncodingDirection': phase_encoding_direction}
    with open(target, 'w') as fobj:
        json.dump(payload, fobj)
    return target


def generate_diffprep_boilerplate(correction_mode):
    """Methods boilerplate describing the DIFFPREP HMC backend."""

    mode_desc = {
        'motion': 'rigid head motion only',
        'quadratic': 'rigid head motion together with quadratic eddy currents',
        'cubic': 'rigid head motion together with cubic eddy currents',
    }[correction_mode]
    return (
        f'\n\nHead motion correction was performed with DIFFPREP '
        f'[@diffprep], part of the TORTOISE [@tortoisev4] software package, '
        f'in {correction_mode} mode (correcting {mode_desc}). DIFFPREP fits a '
        'SHORE/MAPMRI signal model to the data and iteratively registers each '
        'volume to a model-predicted target using TORTOISE\'s 24-parameter '
        'Okan-quadratic transform. The corrected volumes and motion-rotated '
        'bmatrix were then passed to the rest of the pipeline.\n\n'
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


# ---------------------------------------------------------------------------
# Model-based "ideal" DWI synthesis for slice-wise QC (carpet plot)
# ---------------------------------------------------------------------------
#
# FSL eddy emits a per-slice outlier map directly, and SHORELine builds
# model-predicted ("noise-free") volumes from its 3dSHORE fit; both feed
# qsiprep's SliceQC -> DMRISummary carpet plot. TORTOISE DIFFPREP does neither
# of those by default -- it only persists model-synthesized volumes when REPOL
# outlier replacement is enabled, which we keep OFF for the non-shelled CS-DSI
# schemes (shell binning degenerates on continuous-q data). To get the same
# observed-vs-model carpet plot for the DIFFPREP backend, we fit a MAPMRI model
# to the *corrected* DWI with TORTOISE's own CLI tools and synthesize an ideal
# volume at every corrected gradient, then hand the (observed, ideal) pair to
# the same SliceQC node SHORELine uses.


def _tortoise_heuristic_deltas(bmtxt_file):
    """Replicate TORTOISE's internal small/big-delta heuristic.

    ``EstimateMAPMRI`` derives diffusion timings from the maximum b-value when
    they are not supplied on the command line (``estimate_mapmri_main.cxx``:
    ``small_delta = (b_max / gyro^2 / G^2 / 2 * 1e6)^(1/3) * 1000``; ``G`` is
    taken as ``2 * 40 mT/m`` and ``big_delta = 3 * small_delta``, in ms).

    ``SynthesizeDWIsFromMAPMRI`` has **no** such fallback -- it consumes the
    deltas as mandatory positional arguments. For the carpet-plot QC we only
    need the fit and the synthesis to share an identical ``tdiff`` and q-space
    scaling so the synthesized signal reproduces the *model's* prediction of
    the observed signal (deviations then localize genuine slice corruption).
    Computing the deltas here once and passing the identical values to both
    ``EstimateMAPMRI`` and ``SynthesizeDWIsFromMAPMRI`` guarantees that
    consistency and exactly matches what TORTOISE would have chosen on its own.

    The b-value of each volume is the trace of its 6-column b-matrix row
    (columns 0, 3, 5 -> Bxx, Byy, Bzz).
    """
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
    entirely within the node's working directory. Unlike the qsirecon TORTOISE
    recon interfaces -- which rely on the input NIfTI living in the node cwd so
    that TORTOISE's "write outputs next to the input" convention lines up with
    a basename-relative ``_list_outputs`` -- this interface copies the DWI and
    its ``.bmtxt`` sibling into ``runtime.cwd`` with a fixed stem, so output
    discovery is unambiguous regardless of where the upstream files lived.
    """

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
        mask_img = nim.resample_to_img(
            self.inputs.mask_file, ref_3d, interpolation='nearest'
        )
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
        ])

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
        ])
        synth_path = coeffs_path[: -len('.nii')] + '_synth.nii'
        if not op.exists(synth_path):
            raise FileNotFoundError(
                f'SynthesizeDWIsFromMAPMRI did not produce {synth_path}'
            )

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
