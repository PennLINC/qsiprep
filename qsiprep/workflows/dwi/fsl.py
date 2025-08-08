"""
Implementing the FSL preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_hmc_wf

"""

import json
import os

from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgr_fn

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.eddy import (
    Eddy2SPMMotion,
    ExtendedEddy,
    GatherEddyInputs,
    boilerplate_from_eddy_config,
)
from ...interfaces.fmap import ParallelTOPUP
from ...interfaces.gradients import ExtractB0s
from ...interfaces.images import ConformDwi, IntraModalMerge, SplitDWIsFSL
from ...interfaces.nilearn import EnhanceB0
from ...interfaces.reports import TopupSummary
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf

# dwi workflows
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_fsl_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    dwi_metadata=None,
    slice_quality='outlier_n_sqr_stdev_map',
    name='fsl_hmc_wf',
):
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
        pepolar_method : str
            Either 'DRBUDDI', 'TOPUP' or 'DRBUDDI+TOPUP'. The method for SDC when EPI
            fieldmaps are used.
        eddy_config: str
            Path to a JSON file containing settings for the call to ``eddy``.


    **Inputs**

        dwi_file: str
            DWI series. Possibly concatenated, denoised, etc
        bvec_file: str
            bvec file
        bval_file: str
            bval file
        json_file: str
            path to sidecar json file for dwi_file
        b0_indices: list
            Indexes into ``dwi_files`` that correspond to b=0 volumes
        b0_images: list
            List of single b=0 volumes
        original_files: list
            List of the files from which each DWI volume came. One per original file
        t1_brain: str
            Skull stripped T1w image
        t1_mask: str
            mask for t1_brain

    """
    # Check for FSL binary
    fsl_check = os.environ.get('FSL_BUILD')
    if fsl_check == 'no_fsl':
        raise Exception(
            """Container in use does not have FSL. To use this workflow,
            please download the qsiprep container with FSL installed."""
        )
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'b0_indices',
                'b0_images',
                'original_files',
                't1_brain',
                't1_mask',
                't1_seg',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pre_sdc_template',
                'bval_files',
                'hmc_optimization_data',
                'sdc_method',
                'slice_quality',
                'motion_params',
                'cnr_map',
                'bvec_files_to_transform',
                'dwi_files_to_transform',
                'b0_indices',
                'to_dwi_ref_affines',
                'to_dwi_ref_warps',
                'rpe_b0_info',
                'sdc_scaling_images',
                # From SDC
                'fieldmap_type',
                'fieldmap_hz',
                'b0_up_image',
                'b0_up_corrected_image',
                'b0_down_image',
                'b0_down_corrected_image',
                'up_fa_image',
                'up_fa_corrected_image',
                'down_fa_image',
                'down_fa_corrected_image',
                't2w_image',
                # Not from SDC, but from the eddy-resampled data
                'b0_template',
                'b0_template_mask',
            ]
        ),
        name='outputnode',
    )

    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads
    if config.workflow.eddy_config is None:
        # load from the defaults
        eddy_cfg_file = pkgr_fn('qsiprep.data', 'eddy_params.json')
    else:
        eddy_cfg_file = config.workflow.eddy_config

    with open(eddy_cfg_file) as f:
        eddy_args = json.load(f)

    gather_inputs = pe.Node(
        GatherEddyInputs(
            b0_threshold=config.workflow.b0_threshold,
            raw_image_sdc=False,
            eddy_config=eddy_cfg_file,
        ),
        name='gather_inputs',
    )
    enhance_pre_sdc = pe.Node(EnhanceB0(), name='enhance_pre_sdc')

    # Run in parallel if possible
    if eddy_args['use_cuda']:
        eddy_args['num_threads'] = 1
        config.loggers.workflow.info('Using CUDA and %d threads in eddy', eddy_args['num_threads'])
    else:
        eddy_args['num_threads'] = omp_nthreads
        config.loggers.workflow.info('Using %d threads in eddy', eddy_args['num_threads'])
    pre_eddy_b0_ref_wf = init_dwi_reference_wf(
        source_file=source_file,
        name='pre_eddy_b0_ref_wf',
        gen_report=False,
    )
    eddy = pe.Node(ExtendedEddy(**eddy_args), name='eddy')
    spm_motion = pe.Node(Eddy2SPMMotion(), name='spm_motion')

    # Convert eddy outputs back to LPS+, split them
    back_to_lps = pe.Node(ConformDwi(orientation='LPS'), name='back_to_lps')
    cnr_lps = pe.Node(ConformDwi(orientation='LPS'), name='cnr_lps')
    split_eddy_lps = pe.Node(
        SplitDWIsFSL(b0_threshold=config.workflow.b0_threshold, deoblique_bvecs=True),
        name='split_eddy_lps',
    )

    extract_b0_series = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='extract_b0_series'
    )
    b0_ref_for_coreg = init_dwi_reference_wf(
        gen_report=False,
        desc='b0_for_coreg',
        name='b0_ref_for_coreg',
        source_file=source_file,
    )

    workflow.connect([
        # These images and gradients should be in LAS+
        (inputnode, gather_inputs, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('json_file', 'json_file'),
            ('original_files', 'original_files'),
        ]),
        (inputnode, pre_eddy_b0_ref_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_mask', 'inputnode.t1_mask'),
        ]),
        (gather_inputs, eddy, [
            ('eddy_indices', 'in_index'),
            ('eddy_acqp', 'in_acqp'),
            ('json_file', 'json'),
            ('multiband_factor', 'multiband_factor'),
        ]),
        (inputnode, eddy, [
            ('dwi_file', 'in_file'),
            ('bval_file', 'in_bval'),
            ('bvec_file', 'in_bvec'),
        ]),
        (gather_inputs, outputnode, [('forward_transforms', 'to_dwi_ref_affines')]),
        (gather_inputs, enhance_pre_sdc, [('pre_topup_image', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [('enhanced_file', 'pre_sdc_template')]),
        (eddy, back_to_lps, [
            ('out_corrected', 'dwi_file'),
            ('out_rotated_bvecs', 'bvec_file'),
        ]),
        (inputnode, back_to_lps, [('bval_file', 'bval_file')]),
        (back_to_lps, split_eddy_lps, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
        (inputnode, outputnode, [('original_files', 'original_files')]),
        (split_eddy_lps, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices'),
        ]),
        (eddy, cnr_lps, [('out_cnr_maps', 'dwi_file')]),
        (cnr_lps, outputnode, [('dwi_file', 'cnr_map')]),
        (eddy, outputnode, [
            (slice_quality, 'slice_quality'),
            (slice_quality, 'hmc_optimization_data'),
        ]),
        (eddy, spm_motion, [('out_parameter', 'eddy_motion')]),
        (spm_motion, outputnode, [('spm_motion_file', 'motion_params')]),
        # Create a b=0 reference from Eddy's output
        (back_to_lps, extract_b0_series, [
            ('dwi_file', 'dwi_series'),
            ('bval_file', 'bval_file'),
        ]),
        (extract_b0_series, b0_ref_for_coreg, [('b0_average', 'inputnode.b0_template')]),
        (b0_ref_for_coreg, outputnode, [('outputnode.dwi_mask', 'b0_template_mask')]),
    ])  # fmt:skip

    # Fieldmap correction to be done in LAS+: TOPUP for rpe series or epi fieldmap
    # If a topupref is provided, use it for TOPUP
    fieldmap_type = scan_groups['fieldmap_info']['suffix'] or ''
    workflow.__desc__ = boilerplate_from_eddy_config(
        eddy_args, fieldmap_type, config.workflow.pepolar_method
    )

    # Are we running TOPUP?
    if (
        fieldmap_type in ('epi', 'rpe_series')
        and 'topup' in config.workflow.pepolar_method.lower()
    ):
        # If there are EPI fieldmaps in fmaps/, make sure they get to TOPUP. It will always use
        # b=0 images from the DWI series regardless
        gather_inputs.inputs.topup_requested = True
        if 'epi' in scan_groups['fieldmap_info']:
            gather_inputs.inputs.epi_fmaps = scan_groups['fieldmap_info']['epi']

        outputnode.inputs.sdc_method = 'TOPUP'
        topup = pe.Node(
            ParallelTOPUP(out_field='fieldmap_HZ.nii.gz', scale=1, nthreads=omp_nthreads),
            name='topup',
            n_procs=omp_nthreads,
        )
        topup_summary = pe.Node(TopupSummary(), name='topup_summary')
        ds_report_topupsummary = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='topupsummary',
                suffix='dwi',
                source_file=source_file,
            ),
            name='ds_report_topupsummary',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_pepolar_qc_tsv = pe.Node(
            DerivativesDataSink(
                desc='pepolar',
                suffix='qc',
                extension='tsv',
                source_file=source_file,
            ),
            name='ds_pepolar_qc_tsv',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        # Enhance and skullstrip the TOPUP output to get a mask for eddy
        unwarped_mean = pe.Node(IntraModalMerge(hmc=False, to_lps=False), name='unwarped_mean')
        # Register the first volume of topup imain to the first volume of the merged dwi
        topup_to_eddy_reg = pe.Node(
            fsl.FLIRT(dof=6, output_type='NIFTI_GZ'), name='topup_to_eddy_reg'
        )
        transform_mask_to_eddy = pe.Node(
            fsl.ApplyXFM(apply_xfm=True, interp='nearestneighbour', output_type='NIFTI_GZ'),
            name='transform_mask_to_eddy',
        )
        transform_fmap_to_eddy = pe.Node(
            fsl.ApplyXFM(apply_xfm=True, interp='nearestneighbour', output_type='NIFTI_GZ'),
            name='transform_fmap_to_eddy',
        )

        workflow.connect([
            (gather_inputs, topup, [
                ('topup_datain', 'encoding_file'),
                ('topup_imain', 'in_file'),
                ('topup_config', 'config'),
            ]),
            (gather_inputs, ds_pepolar_qc_tsv, [('b0_tsv', 'in_file')]),
            (topup, eddy, [('out_field', 'field')]),
            (gather_inputs, topup_to_eddy_reg, [
                ('topup_first', 'in_file'),
                ('eddy_first', 'reference'),
            ]),
            (topup_to_eddy_reg, eddy, [('out_matrix_file', 'field_mat')]),

            # Use corrected images from TOPUP to make a mask for eddy
            (topup, unwarped_mean, [('out_corrected', 'in_files')]),
            (unwarped_mean, pre_eddy_b0_ref_wf, [('out_avg', 'inputnode.b0_template')]),

            # Ensure that the mask is aligned with eddy's first image
            (pre_eddy_b0_ref_wf, transform_mask_to_eddy, [('outputnode.dwi_mask', 'in_file')]),
            (topup, transform_fmap_to_eddy, [('out_field', 'in_file')]),
            (topup_to_eddy_reg, transform_mask_to_eddy, [('out_matrix_file', 'in_matrix_file')]),
            (topup_to_eddy_reg, transform_fmap_to_eddy, [('out_matrix_file', 'in_matrix_file')]),
            (gather_inputs, transform_mask_to_eddy, [('eddy_first', 'reference')]),
            (gather_inputs, transform_fmap_to_eddy, [('eddy_first', 'reference')]),
            (transform_mask_to_eddy, eddy, [('out_file', 'in_mask')]),
            (transform_fmap_to_eddy, outputnode, [('out_file', 'fieldmap_hz')]),

            # Save reports
            (gather_inputs, topup_summary, [('topup_report', 'summary')]),
            (topup_summary, ds_report_topupsummary, [('out_report', 'in_file')]),
        ])  # fmt:skip

        if 'drbuddi' not in config.workflow.pepolar_method.lower():
            config.loggers.workflow.info('Using single-stage SDC, TOPUP-only')
            workflow.connect([
                # There will be no SDC warps, they are applied by eddy
                (gather_inputs, outputnode, [('forward_warps', 'to_dwi_ref_warps')]),
                (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
            ])  # fmt:skip
    else:
        # If we're not using TOPUP we need to make a mask for eddy based on the
        # distorted brain shapes
        distorted_merge = pe.Node(IntraModalMerge(hmc=True, to_lps=False), name='distorted_merge')
        # Use the distorted mask for eddy
        workflow.connect([
            (gather_inputs, distorted_merge, [('topup_imain', 'in_files')]),
            (distorted_merge, pre_eddy_b0_ref_wf, [('out_avg', 'inputnode.b0_template')]),
            (pre_eddy_b0_ref_wf, eddy, [('outputnode.dwi_mask', 'in_mask')]),
        ])  # fmt:skip

    if (
        fieldmap_type in ('epi', 'rpe_series')
        and 'drbuddi' in config.workflow.pepolar_method.lower()
    ):
        outputnode.inputs.sdc_method = 'DRBUDDI'
        config.loggers.workflow.info('Running DRBUDDI for SDC')

        # Let gather_inputs know we're doing pepolar, even though it's not topup
        gather_inputs.inputs.topup_requested = True
        if 'epi' in scan_groups['fieldmap_info']:
            gather_inputs.inputs.epi_fmaps = scan_groups['fieldmap_info']['epi']

        drbuddi_wf = init_drbuddi_wf(
            scan_groups=scan_groups,
            t2w_sdc=t2w_sdc,
        )

        workflow.connect([
            (split_eddy_lps, drbuddi_wf, [
                ('dwi_files', 'inputnode.dwi_files'),
                ('bval_files', 'inputnode.bval_files'),
                ('bvec_files', 'inputnode.bvec_files'),
            ]),
            (inputnode, drbuddi_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t2w_unfatsat', 'inputnode.t2w_unfatsat'),
                ('original_files', 'inputnode.original_files'),
            ]),
            (drbuddi_wf, outputnode, [
                ('outputnode.sdc_warps', 'to_dwi_ref_warps'),
                ('outputnode.sdc_scaling_images', 'sdc_scaling_images'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.fieldmap_type', 'fieldmap_type'),
                ('outputnode.b0_up_image', 'b0_up_image'),
                ('outputnode.b0_up_corrected_image', 'b0_up_corrected_image'),
                ('outputnode.b0_down_image', 'b0_down_image'),
                ('outputnode.b0_down_corrected_image', 'b0_down_corrected_image'),
                ('outputnode.up_fa_image', 'up_fa_image'),
                ('outputnode.up_fa_corrected_image', 'up_fa_corrected_image'),
                ('outputnode.down_fa_image', 'down_fa_image'),
                ('outputnode.down_fa_corrected_image', 'down_fa_corrected_image'),
                ('outputnode.t2w_image', 't2w_image'),
                ('outputnode.b0_ref', 'b0_template'),
            ]),
        ])  # fmt:skip

        return workflow

    if fieldmap_type in ('fieldmap', 'syn') or fieldmap_type.startswith('phase'):
        config.loggers.workflow.info(f'Computing fieldmap directly from {fieldmap_type}')
        outputnode.inputs.sdc_method = fieldmap_type
        b0_sdc_wf = init_sdc_wf(
            scan_groups['fieldmap_info'],
            dwi_metadata,
        )

        workflow.connect([
            # Send to SDC workflow
            (b0_ref_for_coreg, b0_sdc_wf, [
                ('outputnode.ref_image', 'inputnode.b0_ref'),
                ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                ('outputnode.dwi_mask', 'inputnode.b0_mask'),
            ]),
            (inputnode, b0_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
            ]),
            # These deformations will be applied later, use the unwarped image now
            (b0_sdc_wf, outputnode, [
                ('outputnode.out_warp', 'to_dwi_ref_warps'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.b0_ref', 'b0_template'),
            ]),
        ])  # fmt:skip

    if not fieldmap_type:
        outputnode.inputs.sdc_method = 'None'
        workflow.connect([
            (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
        ])  # fmt:skip
    return workflow
