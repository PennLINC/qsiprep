"""
Implementing the FSL preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_hmc_wf

"""

import json
from pkg_resources import resource_filename as pkgr_fn
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

from ...interfaces.eddy import (GatherEddyInputs, ExtendedEddy, Eddy2SPMMotion,
                                boilerplate_from_eddy_config)
from ...interfaces.images import SplitDWIs, ConformDwi, IntraModalMerge
from ...interfaces.reports import TopupSummary
from ...interfaces.nilearn import EnhanceB0
from ...interfaces import DerivativesDataSink
from ...engine import Workflow

# dwi workflows
from .util import init_dwi_reference_wf
from ..fieldmap.base import init_sdc_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(scan_groups,
                    source_file,
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

    If a GRE fieldmap is available, the correction is applied to eddy's outputs after they have
    been converted back to LPS+.

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

        dwi_file: str
            DWI series
        bvec_file: str
            bvec file
        bval_file: str
            bval file
        b0_indices: list
            Indexes into ``dwi_files`` that correspond to b=0 volumes
        b0_images: list
            List of single b=0 volumes
        original_files: list
            List of the files from which each DWI volume came.
        t1_brain: str
            Skull stripped T1w image
        t1_mask: str
            mask for t1_brain

    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_file', 'bvec_file', 'bval_file', 'b0_indices', 'b0_images',
                    'original_files', 't1_brain', 't1_mask', 't1_seg',
                    't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "b0_template_mask", "pre_sdc_template", "bval_files",
                    "hmc_optimization_data", "sdc_method", 'slice_quality', 'motion_params',
                    "cnr_map", "bvec_files_to_transform", "dwi_files_to_transform", "b0_indices",
                    "to_dwi_ref_affines", "to_dwi_ref_warps", "rpe_b0_info"]),
        name='outputnode')

    workflow = Workflow(name=name)
    gather_inputs = pe.Node(
        GatherEddyInputs(b0_threshold=b0_threshold), name="gather_inputs")
    if eddy_config is None:
        # load from the defaults
        eddy_cfg_file = pkgr_fn('qsiprep.data', 'eddy_params.json')
    else:
        eddy_cfg_file = eddy_config

    with open(eddy_cfg_file, "r") as f:
        eddy_args = json.load(f)
    enhance_pre_sdc = pe.Node(EnhanceB0(), name='enhance_pre_sdc')

    # Run in parallel if possible
    LOGGER.info("Using %d threads in eddy", omp_nthreads)
    eddy_args["num_threads"] = omp_nthreads
    pre_eddy_b0_ref_wf = init_dwi_reference_wf(register_t1=True, source_file=source_file,
                                               name='pre_eddy_b0_ref_wf', gen_report=False)
    eddy = pe.Node(ExtendedEddy(**eddy_args), name="eddy")
    spm_motion = pe.Node(Eddy2SPMMotion(), name="spm_motion")

    # Convert eddy outputs back to LPS+, split them
    back_to_lps = pe.Node(ConformDwi(orientation="LPS"), name='back_to_lps')
    cnr_lps = pe.Node(ConformDwi(orientation="LPS"), name='cnr_lps')
    split_eddy_lps = pe.Node(SplitDWIs(b0_threshold=b0_threshold, deoblique_bvecs=True),
                             name="split_eddy_lps")

    # Convert the b=0 template from pre_eddy_b0_ref to LPS+
    b0_ref_to_lps = pe.Node(ConformDwi(orientation="LPS"), name='b0_ref_to_lps')
    b0_ref_mask_to_lps = pe.Node(ConformDwi(orientation="LPS"), name='b0_ref_mask_to_lps')
    b0_ref_brain_to_lps = pe.Node(ConformDwi(orientation="LPS"), name='b0_ref_brain_to_lps')

    workflow.connect([
        # These images and gradients should be in LAS+
        (inputnode, gather_inputs, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('original_files', 'original_files')]),
        (inputnode, pre_eddy_b0_ref_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg')]),
        # Convert distorted ref to LPS+
        (pre_eddy_b0_ref_wf, b0_ref_to_lps, [
            ('outputnode.ref_image', 'dwi_file')]),
        (pre_eddy_b0_ref_wf, b0_ref_mask_to_lps, [
            ('outputnode.dwi_mask', 'dwi_file')]),
        (pre_eddy_b0_ref_wf, b0_ref_brain_to_lps, [
            ('outputnode.ref_image_brain', 'dwi_file')]),
        (gather_inputs, eddy, [
            ('eddy_indices', 'in_index'),
            ('eddy_acqp', 'in_acqp')]),
        (inputnode, eddy, [
            ('dwi_file', 'in_file'),
            ('bval_file', 'in_bval'),
            ('bvec_file', 'in_bvec')]),
        (pre_eddy_b0_ref_wf, eddy, [('outputnode.dwi_mask', 'in_mask')]),
        (gather_inputs, outputnode, [
            ('forward_transforms', 'to_dwi_ref_affines')]),
        (gather_inputs, enhance_pre_sdc, [
            ('pre_topup_image', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [
            ('enhanced_file', 'pre_sdc_template')]),
        (eddy, back_to_lps, [
            ('out_corrected', 'dwi_file'),
            ('out_rotated_bvecs', 'bvec_file')]),
        (inputnode, back_to_lps, [('bval_file', 'bval_file')]),
        (back_to_lps, split_eddy_lps, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (inputnode, outputnode, [
            ('original_files', 'original_files')]),
        (split_eddy_lps, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices')]),
        (eddy, cnr_lps, [('out_cnr_maps', 'dwi_file')]),
        (cnr_lps, outputnode, [('dwi_file', 'cnr_map')]),
        (eddy, outputnode, [
            (slice_quality, 'slice_quality'),
            (slice_quality, 'hmc_optimization_data')]),
        (eddy, spm_motion, [('out_parameter', 'eddy_motion')]),
        (b0_ref_mask_to_lps, outputnode, [('dwi_file', 'b0_template_mask')]),
        (spm_motion, outputnode, [('spm_motion_file', 'motion_params')])
    ])

    # Fieldmap correction to be done in LAS+: TOPUP for rpe series or epi fieldmap
    # If a topupref is provided, use it for TOPUP
    fieldmap_type = scan_groups['fieldmap_info']['suffix'] or ''
    workflow.__desc__ = boilerplate_from_eddy_config(eddy_args, fieldmap_type)
    if fieldmap_type in ('epi', 'rpe_series'):
        # If there are EPI fieldmaps in fmaps/, make sure they get to TOPUP. It will always use
        # b=0 images from the DWI series regardless
        gather_inputs.inputs.topup_requested = True
        if 'epi' in scan_groups['fieldmap_info']:
            gather_inputs.inputs.epi_fmaps = scan_groups['fieldmap_info']['epi']
        outputnode.inputs.sdc_method = "TOPUP"
        topup = pe.Node(fsl.TOPUP(out_field="fieldmap_HZ.nii.gz", scale=1), name="topup")
        topup_summary = pe.Node(TopupSummary(), name='topup_summary')
        ds_report_topupsummary = pe.Node(
            DerivativesDataSink(suffix='topupsummary', source_file=source_file),
            name='ds_report_topupsummary',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_topupcsv = pe.Node(
            DerivativesDataSink(suffix='topupcsv', source_file=source_file),
            name='ds_topupcsv',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        # Enhance and skullstrip the TOPUP output to get a mask for eddy
        unwarped_mean = pe.Node(IntraModalMerge(hmc=False, to_lps=False), name='unwarped_mean')
        # Register the first volume of topup imain to the first volume of the merged dwi
        topup_to_eddy_reg = pe.Node(fsl.FLIRT(dof=6, output_type="NIFTI_GZ"),
                                    name="topup_to_eddy_reg")
        workflow.connect([
            # There will be no SDC warps, they are applied by eddy
            (gather_inputs, outputnode, [('forward_warps', 'to_dwi_ref_warps')]),
            (gather_inputs, topup, [
                ('topup_datain', 'encoding_file'),
                ('topup_imain', 'in_file'),
                ('topup_config', 'config')]),
            (topup, eddy, [
                ('out_field', 'field')]),
            (gather_inputs, topup_to_eddy_reg, [
                ('topup_first', 'in_file'),
                ('eddy_first', 'reference')]),
            (gather_inputs, ds_topupcsv, [('b0_csv', 'in_file')]),
            (topup_to_eddy_reg, eddy, [('out_matrix_file', 'field_mat')]),
            # Use corrected images from TOPUP to make a mask for eddy
            (topup, unwarped_mean, [('out_corrected', 'in_files')]),
            (unwarped_mean, pre_eddy_b0_ref_wf, [('out_avg', 'inputnode.b0_template')]),
            (b0_ref_to_lps, outputnode, [('dwi_file', 'b0_template')]),
            # Save reports
            (gather_inputs, topup_summary, [('topup_report', 'summary')]),
            (topup_summary, ds_report_topupsummary, [('out_report', 'in_file')]),
        ])

        return workflow

    # The topup inputs will only have one PE direction,
    # so they can be used to make a b=0 reference to mask for eddy
    distorted_merge = pe.Node(
        IntraModalMerge(hmc=True, to_lps=False), name='distorted_merge')
    workflow.connect([
        # Use the distorted mask for eddy
        (gather_inputs, distorted_merge, [('topup_imain', 'in_files')]),
        (distorted_merge, pre_eddy_b0_ref_wf, [('out_avg', 'inputnode.b0_template')])])

    if fieldmap_type in ('fieldmap', 'syn') or fieldmap_type.startswith("phase"):

        outputnode.inputs.sdc_method = fieldmap_type
        b0_sdc_wf = init_sdc_wf(
            scan_groups['fieldmap_info'], dwi_metadata, omp_nthreads=omp_nthreads,
            fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)

        workflow.connect([
            # Send to SDC workflow
            (b0_ref_to_lps, b0_sdc_wf, [
                ('dwi_file', 'inputnode.b0_ref')]),
            (b0_ref_brain_to_lps, b0_sdc_wf, [('dwi_file', 'inputnode.b0_ref_brain')]),
            (b0_ref_mask_to_lps, b0_sdc_wf, [('dwi_file', 'inputnode.b0_mask')]),
            (inputnode, b0_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform',
                 'inputnode.t1_2_mni_reverse_transform')]),
            # These deformations will be applied later, use the unwarped image now
            (b0_sdc_wf, outputnode, [
                ('outputnode.out_warp', 'to_dwi_ref_warps'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.b0_ref', 'b0_template')])])

    else:
        outputnode.inputs.sdc_method = "None"
        workflow.connect([
            (b0_ref_to_lps, outputnode, [
                ('dwi_file', 'b0_template')])])
    return workflow
