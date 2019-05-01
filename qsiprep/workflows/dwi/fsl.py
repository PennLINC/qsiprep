"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_dwi_preproc_wf

"""

import os

import nibabel as nb
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import DiffusionSummary, GradientPlot
from ...interfaces.images import SplitDWIs, ConcatRPESplits
from ...interfaces.gradients import ExtractB0s
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from .merge import init_merge_and_denoise_wf
from .hmc import init_dwi_hmc_wf
from .util import init_dwi_reference_wf, _create_mem_gb, _get_wf_name, _list_squeeze
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf
from .derivatives import init_dwi_derivatives_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(impute_slice_threshold,
                    bidirectional_pepolar,
                    rpe_b0,
                    mem_gb=3,
                    omp_nthreads=1,
                    name="fsl_hmc_wf"):
    """
    This workflow controls the dwi preprocessing stages using FSL tools.


    **Parameters**


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

    **Outputs**


    **Subworkflows**

    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'b0_indices', 'bvecs', 'bvals', 'b0_images', 'original_files']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["final_template", "forward_transforms", "noise_free_dwis",
                    "optimization_data"]),
        name='outputnode')

    workflow = Workflow(name=name)

    gradient_plot = pe.Node(GradientPlot(), name='gradient_plot', run_without_submitting=True)
    # ds_report_gradients = pe.Node(
    #     DerivativesDataSink(suffix='sampling_scheme'),
    #     name='ds_report_gradients', run_without_submitting=True,
    #     mem_gb=DEFAULT_MEMORY_MIN_GB)

    # CONNECT TO DERIVATIVES #####################
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_prefix=output_prefix,
        source_file=source_file,
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template,
        write_local_bvecs=write_local_bvecs)

    workflow.connect([
        (inputnode, dwi_derivatives_wf, [('dwi_files', 'inputnode.source_file')]),
        (outputnode, dwi_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('bvals_t1', 'inputnode.bvals_t1'),
          ('bvecs_t1', 'inputnode.bvecs_t1'),
          ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
          ('t1_b0_ref', 'inputnode.t1_b0_ref'),
          ('t1_b0_series', 'inputnode.t1_b0_series'),
          ('gradient_table_t1', 'inputnode.gradient_table_t1'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('bvals_mni', 'inputnode.bvals_mni'),
          ('bvecs_mni', 'inputnode.bvecs_mni'),
          ('local_bvecs_mni', 'inputnode.local_bvecs_mni'),
          ('mni_b0_ref', 'inputnode.mni_b0_ref'),
          ('mni_b0_series', 'inputnode.mni_b0_series'),
          ('gradient_table_mni', 'inputnode.gradient_table_mni'),
          ('confounds', 'inputnode.confounds')])
    ])

    workflow.connect([
        (inputnode, b0_coreg_wf,
            [('t1_brain', 'inputnode.t1_brain'),
             ('t1_seg', 'inputnode.t1_seg'),
             ('subjects_dir', 'inputnode.subjects_dir'),
             ('subject_id', 'inputnode.subject_id'),
             ('t1_2_fsnative_reverse_transform',
              'inputnode.t1_2_fsnative_reverse_transform')]),
        (buffernode, b0_coreg_wf, [('b0_ref_image',
                                    'inputnode.ref_b0_brain')]),
    ])

    # confounds_wf = init_dwi_confs_wf(mem_gb=mem_gb['resampled'], metadata=[],
    #                                 impute_slice_threshold=impute_slice_threshold,
    #                                 name='confounds_wf')

    # Carpetplot and confounds plot
    # conf_plot = pe.Node(DMRISummary(), name='conf_plot', mem_gb=mem_gb['resampled'])
    # ds_report_dwi_conf = pe.Node(
    #    DerivativesDataSink(suffix='carpetplot'),
    #    name='ds_report_dwi_conf', run_without_submitting=True,
    #    mem_gb=DEFAULT_MEMORY_MIN_GB)

    #  workflow.connect([
    #     (buffernode, slice_check, [('ideal_images', 'ideal_image_files'),
    #                                ('dwi_files', 'uncorrected_dwi_files'),
    #                                ('b0_ref_mask', 'mask_image')]),
    #     (slice_check, confounds_wf, [('slice_stats', 'inputnode.sliceqc_file')]),
    #     (buffernode, confounds_wf, [('to_dwi_ref_affines', 'inputnode.hmc_affines'),
    #                                 ('bval_files', 'inputnode.bval_files'),
    #                                 ('bvec_files', 'inputnode.bvec_files'),
    #                                 ('original_grouping', 'inputnode.original_files')]),
    #     (confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),
    #
    #     (confounds_wf, conf_plot, [('outputnode.confounds_file', 'confounds_file')]),
    #     (slice_check, conf_plot, [('slice_stats', 'sliceqc_file')]),
    #     (buffernode, conf_plot, []),
    #     (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
    #     (buffernode, gradient_plot, [('bvec_files', 'orig_bvec_files'),
    #                                  ('bval_files', 'orig_bval_files'),
    #                                  ('original_grouping', 'source_files')]),
    #     (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])
    #
    # ])

    if "T1w" in output_spaces:
        transform_dwis_t1 = init_dwi_trans_wf(name='transform_dwis_t1',
                                              template="ACPC",
                                              mem_gb=mem_gb['resampled'],
                                              use_fieldwarp=(fmaps is not None or use_syn),
                                              omp_nthreads=omp_nthreads,
                                              use_compression=False,
                                              to_mni=False,
                                              write_local_bvecs=write_local_bvecs
                                              )
        gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
        workflow.connect([
            (buffernode, transform_dwis_t1, [
                ('dwi', 'inputnode.dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('b0_ref_mask', 'inputnode.dwi_mask'),
                ('b0_indices', 'inputnode.b0_indices')]),
            (inputnode, transform_dwis_t1, [
                ('dwi_sampling_grid', 'inputnode.output_grid')]),
            (b0_coreg_wf, transform_dwis_t1, [
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_t1, outputnode, [('outputnode.bvals', 'bvals_t1'),
                                             ('outputnode.rotated_bvecs', 'bvecs_t1'),
                                             ('outputnode.dwi_resampled', 'dwi_t1'),
                                             ('outputnode.local_bvecs', 'local_bvecs_t1'),
                                             ('outputnode.dwi_mask_resampled', 'dwi_mask_t1'),
                                             ('outputnode.b0_series', 't1_b0_series'),
                                             ('outputnode.dwi_ref_resampled', 't1_b0_ref')]),
            (outputnode, gradient_plot, [('bvecs_t1', 'final_bvec_file')]),
            (transform_dwis_t1, gtab_t1, [('outputnode.bvals', 'bval_file'),
                                          ('outputnode.rotated_bvecs', 'bvec_file')]),
            (gtab_t1, outputnode, [('gradient_file', 'gradient_table_t1')])
        ])

    if "template" in output_spaces:
        transform_dwis_mni = init_dwi_trans_wf(name='transform_dwis_mni',
                                               template=template,
                                               mem_gb=mem_gb['resampled'],
                                               use_fieldwarp=(fmaps is not None or use_syn),
                                               omp_nthreads=omp_nthreads,
                                               use_compression=False,
                                               to_mni=True,
                                               write_local_bvecs=write_local_bvecs
                                               )
        gtab_mni = pe.Node(MRTrixGradientTable(), name='gtab_mni')
        workflow.connect([
            (buffernode, transform_dwis_mni, [
                ('dwi_files', 'dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('b0_ref_mask', 'inputnode.dwi_mask'),
                ('b0_indices', 'inputnode.b0_indices'),
                ('to_dwi_ref_affines', 'inputnode.hmc_xforms'),
                ('to_dwi_ref_warps', 'inputnode.fieldwarps')]),
            (inputnode, transform_dwis_mni, [
                ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                ('dwi_sampling_grid', 'inputnode.output_grid')]),
            (b0_coreg_wf, transform_dwis_mni, [
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_mni, outputnode, [('outputnode.bvals', 'bvals_mni'),
                                              ('outputnode.rotated_bvecs', 'bvecs_mni'),
                                              ('outputnode.dwi_resampled', 'dwi_mni'),
                                              ('outputnode.dwi_mask_resampled', 'dwi_mask_mni'),
                                              ('outputnode.b0_series', 'mni_b0_series'),
                                              ('outputnode.local_bvecs', 'local_bvecs_mni'),
                                              ('outputnode.dwi_ref_resampled', 'mni_b0_ref')]),
            (transform_dwis_mni, gtab_mni, [('outputnode.bvals', 'bval_file'),
                                            ('outputnode.rotated_bvecs', 'bvec_file')]),
            (gtab_mni, outputnode, [('gradient_file', 'gradient_table_mni')])
        ])
        if "T1w" not in output_spaces:
            workflow.connect([(outputnode, gradient_plot, [('bvecs_mni', 'final_bvec_file')])])

    # REPORTING ############################################################
    # ds_report_summary = pe.Node(
    #     DerivativesDataSink(suffix='summary'),
    #     name='ds_report_summary',
    #     run_without_submitting=True,
    #     mem_gb=DEFAULT_MEMORY_MIN_GB)

    # workflow.connect([
    #   (summary, ds_report_summary, [('out_report', 'in_file')])
    # ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = source_file

    return workflow
