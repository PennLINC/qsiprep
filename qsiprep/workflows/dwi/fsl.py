"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_dwi_preproc_wf

"""

import json
from pkg_resources import resource_filename as pkgr_fn
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl, afni, ants

from ...interfaces.eddy import GatherEddyInputs, ExtendedEddy, Eddy2SPMMotion
from ...interfaces.dwi_merge import MergeDWIs
from ...interfaces.images import SplitDWIs, ConformDwi
from ...interfaces.fmap import B0RPEFieldmap
from ...engine import Workflow

# dwi workflows
from .util import init_enhance_and_skullstrip_dwi_wf
from ..fieldmap.base import init_sdc_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(scan_groups,
                    b0_threshold,
                    impute_slice_threshold,
                    fmap_demean,
                    fmap_bspline,
                    eddy_config,
                    mem_gb=3,
                    omp_nthreads=1,
                    dwi_metadata=None,
                    slice_quality='outlier_n_sqr_stdev_map',
                    sloppy=False,
                    name="fsl_hmc_wf"):
    """
    This workflow controls the dwi preprocessing stages using FSL tools.

    I couldn't get this to work reliably unless everything was oriented in LAS+ before going to
    TOPUP and eddy. For this reason, if TOPUP is going to be used (for an epi fieldmap or an
    RPE series) or there is no fieldmap correction, operations occurring before eddy are done in
    LAS+. The fieldcoefs are applied during eddy's run and the corrected series comes out.
    This is finally converted to LPS+ and sent to the rest of the pipeline.

    If a GRE fieldmap is available, a fieldmap in Hz is calculated and sent to eddy.

    Finally, if SyN is chosen, it is applied to the LPS+ converted, eddy-resampled data.


    **Parameters**

        scan_groups: dict
            dictionary with fieldmaps and warp space information for the dwis
        impute_slice_threshold: float
            threshold for a slice to be replaced with imputed values. Overrides the
            parameter in ``eddy_config`` if set to a number > 0.
        do_topup: bool
            Should topup be performed before eddy? requires an rpe series or an
            rpe_b0.
        eddy_config: str
            Path to a JSON file containing settings for the call to ``eddy``.


    **Inputs**

        dwi_files: list
            List of single-volume files across all DWI series
        b0_indices: list
            Indexes into ``dwi_files`` that correspond to b=0 volumes
        bvecs: list
            List of paths to single-line bvec files
        bvals: list
            List of paths to single-line bval files
        b0_images: list
            List of single b=0 volumes
        original_files: list
            List of the files from which each DWI volume came from.


    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'b0_indices', 'bvec_files', 'bval_files', 'b0_images',
                    'original_files', 't1_brain', 't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "b0_template_mask", "pre_sdc_template",
                    "hmc_optimization_data", "sdc_method", 'slice_quality', 'motion_params',
                    "cnr_map", "bvec_files_to_transform", "dwi_files_to_transform", "b0_indices",
                    "to_dwi_ref_affines", "to_dwi_ref_warps", "rpe_b0_info"]),
        name='outputnode')

    workflow = Workflow(name=name)
    gather_inputs = pe.Node(GatherEddyInputs(), name="gather_inputs")
    if eddy_config is None:
        # load from the defaults
        eddy_cfg_file = pkgr_fn('qsiprep.data', 'eddy_params.json')
    else:
        eddy_cfg_file = eddy_config

    with open(eddy_cfg_file, "r") as f:
        eddy_args = json.load(f)

    # These should be in LAS+
    dwi_merge = pe.Node(MergeDWIs(), name="dwi_merge")
    eddy = pe.Node(ExtendedEddy(**eddy_args), name="eddy")
    spm_motion = pe.Node(Eddy2SPMMotion(), name="spm_motion")
    # Convert eddy outputs back to LPS+, split them
    pre_topup_lps = pe.Node(ConformDwi(orientation="LPS"), name='pre_topup_lps')
    pre_topup_enhance = init_enhance_and_skullstrip_dwi_wf(name='pre_topup_enhance')
    back_to_lps = pe.Node(ConformDwi(orientation="LPS"), name='back_to_lps')
    split_eddy_lps = pe.Node(SplitDWIs(b0_threshold=b0_threshold), name="split_eddy_lps")
    mean_b0_lps = pe.Node(ants.AverageImages(dimension=3, normalize=True), name='mean_b0_lps')
    lps_b0_enhance = init_enhance_and_skullstrip_dwi_wf(name='lps_b0_enhance')

    workflow.connect([
        # These images and gradients should be in LAS+
        (inputnode, gather_inputs, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('b0_indices', 'b0_indices'),
            ('b0_images', 'b0_images'),
            ('original_files', 'original_files')]),
        # Re-concatenate
        (inputnode, dwi_merge, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('original_files', 'bids_dwi_files')]),
        (gather_inputs, eddy, [
            ('eddy_indices', 'in_index'),
            ('eddy_acqp', 'in_acqp')]),
        (dwi_merge, eddy, [
            ('out_dwi', 'in_file'),
            ('out_bval', 'in_bval'),
            ('out_bvec', 'in_bvec')]),
        (gather_inputs, pre_topup_lps, [
            ('pre_topup_image', 'dwi_file')]),
        (gather_inputs, outputnode, [('forward_transforms', 'to_dwi_ref_affines')]),
        (pre_topup_lps, pre_topup_enhance, [
            ('dwi_file', 'inputnode.in_file')]),
        (pre_topup_enhance, outputnode, [
            ('outputnode.bias_corrected_file', 'pre_sdc_template')]),
        (eddy, back_to_lps, [
            ('out_corrected', 'dwi_file'),
            ('out_rotated_bvecs', 'bvec_file')]),
        (dwi_merge, back_to_lps, [('out_bval', 'bval_file')]),
        (back_to_lps, split_eddy_lps, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (dwi_merge, outputnode, [
            ('original_images', 'original_files')]),
        (split_eddy_lps, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform')]),
        (split_eddy_lps, mean_b0_lps, [('b0_images', 'images')]),
        (mean_b0_lps, lps_b0_enhance, [('output_average_image', 'inputnode.in_file')]),
        (eddy, outputnode, [
            (slice_quality, 'slice_quality'),
            ('out_cnr_maps', 'cnr_map'),
            (slice_quality, 'hmc_optimization_data')]),
        (eddy, spm_motion, [('out_parameter', 'eddy_motion')]),
        (spm_motion, outputnode, [('spm_motion_file', 'motion_params')])
    ])

    # Fieldmap correction to be done in LAS+: TOPUP for rpe series or epi fieldmap
    # If a topupref is provided, use it for TOPUP
    rpe_b0 = None
    fieldmap_type = scan_groups['fieldmap_info']['suffix']
    if fieldmap_type == 'epi':
        rpe_b0 = scan_groups['fieldmap_info']['epi']
    elif fieldmap_type == 'rpe_series':
        rpe_b0 = scan_groups['fieldmap_info']['rpe_series'][0]
    using_topup = rpe_b0 is not None

    if using_topup:
        outputnode.inputs.sdc_method = "TOPUP"
        # Whether an rpe series (from dwi/) or an epi fmap (in fmap/) extract just the
        # b=0s for topup
        prepare_rpe_b0 = pe.Node(
            B0RPEFieldmap(b0_file=rpe_b0, orientation='LAS', output_3d_images=False),
            name="prepare_rpe_b0")
        topup = pe.Node(fsl.TOPUP(out_field="fieldmap_HZ.nii.gz"), name="topup")
        # Enhance and skullstrip the TOPUP output to get a mask for eddy
        unwarped_mean = pe.Node(afni.TStat(outputtype='NIFTI_GZ'), name='unwarped_mean')
        unwarped_enhance = init_enhance_and_skullstrip_dwi_wf(name='unwarped_enhance')

        workflow.connect([
            (prepare_rpe_b0, outputnode, [('fmap_info', 'inputnode.rpe_b0_info')]),
            (prepare_rpe_b0, gather_inputs, [('fmap_file', 'rpe_b0')]),
            (gather_inputs, topup, [
                ('topup_datain', 'encoding_file'),
                ('topup_imain', 'in_file'),
                ('topup_config', 'config')]),
            (gather_inputs, outputnode, [('forward_warps', 'to_dwi_ref_warps')]),
            (topup, unwarped_mean, [('out_corrected', 'in_file')]),
            (unwarped_mean, unwarped_enhance, [('out_file', 'inputnode.in_file')]),
            (unwarped_enhance, eddy, [('outputnode.mask_file', 'in_mask')]),
            (topup, eddy, [
                ('out_field', 'field')]),
            (lps_b0_enhance, outputnode, [
                ('outputnode.bias_corrected_file', 'b0_template'),
                ('outputnode.mask_file', 'b0_template_mask')]),
            ])
        return workflow

    # Enhance and skullstrip the TOPUP output to get a mask for eddy
    distorted_enhance = init_enhance_and_skullstrip_dwi_wf(name='distorted_enhance')
    workflow.connect([
        # Use the distorted mask for eddy
        (gather_inputs, distorted_enhance, [('pre_topup_image', 'inputnode.in_file')]),
        (distorted_enhance, eddy, [('outputnode.mask_file', 'in_mask')]),
    ])
    if fieldmap_type in ('fieldmap', 'phasediff', 'phase', 'syn'):

        outputnode.inputs.sdc_method = fieldmap_type
        b0_sdc_wf = init_sdc_wf(
            scan_groups['fieldmap_info'], dwi_metadata, omp_nthreads=omp_nthreads,
            fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)

        workflow.connect([
            # Calculate distortion correction on eddy-corrected data
            (lps_b0_enhance, b0_sdc_wf, [
                ('outputnode.bias_corrected_file', 'inputnode.b0_ref'),
                ('outputnode.skull_stripped_file', 'inputnode.b0_ref_brain'),
                ('outputnode.mask_file', 'inputnode.b0_mask')]),
            (inputnode, b0_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform',
                 'inputnode.t1_2_mni_reverse_transform')]),

            # These deformations will be applied later, use the unwarped image now
            (b0_sdc_wf, outputnode, [
                ('outputnode.out_warp', 'to_dwi_ref_warps'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.b0_ref', 'b0_template'),
                ('outputnode.b0_mask', 'b0_template_mask')])])

    else:
        outputnode.inputs.sdc_method = "None"
        workflow.connect([
            (lps_b0_enhance, outputnode, [
                ('outputnode.skull_stripped_file', 'b0_template'),
                ('outputnode.mask_file', 'b0_template_mask')]),
            ])

    return workflow
