"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf

"""

import os

from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, utility as niu

from ...interfaces import DerivativesDataSink
from ...niworkflows.interfaces.registration import SimpleBeforeAfterRPT
from ...interfaces.reports import GradientPlot
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from .util import _create_mem_gb
from .resampling import init_dwi_trans_wf
from .derivatives import init_dwi_derivatives_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_finalize_wf(scan_groups,
                         name,
                         output_prefix,
                         ignore,
                         hmc_model,
                         shoreline_iters,
                         reportlets_dir,
                         output_spaces,
                         template,
                         output_dir,
                         omp_nthreads,
                         write_local_bvecs,
                         low_mem,
                         use_syn,
                         make_intramodal_template,
                         layout=None):
    """
    This workflow controls the resampling parts of the dwi preprocessing workflow.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.base import init_dwi_finalize_wf(scan_groups, name, output_prefix, ignore, hmc_model, shoreline_iters, reportlets_dir, output_spaces, template, output_dir, omp_nthreads, write_local_bvecs, low_mem, use_syn, make_intramodal_template, layout)
        wf = init_dwi_finalize_wf(name='finalize_wf',
                                  omp_nthreads=1,
                                  ignore=[],
                                  reportlets_dir='.',
                                  output_dir='.',
                                  template='MNI152NLin2009cAsym',
                                  output_spaces=['T1w', 'template'],
                                  low_mem=False,
                                  output_prefix='',
                                  write_local_bvecs=False,
                                  num_dwi=1)

    **Parameters**

        output_prefix : str
            beginning of the output file name (eg 'sub-1_buds-j')
        ignore : list
            Preprocessing steps to skip (eg "fieldmaps")
        reportlets_dir : str
            Directory in which to save reportlets
        output_spaces : list
            List of output spaces functional images are to be resampled to.
            Some parts of pipeline will only be instantiated for some output s
            paces.

            Valid spaces:

                - T1w
                - template

        template : str
            Name of template targeted by ``template`` output space
        output_dir : str
            Directory in which to save derivatives
        omp_nthreads : int
            Maximum number of threads an individual process may use
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        layout : BIDSLayout
            BIDSLayout structure to enable metadata retrieval

    **Inputs**

        t1_preproc
            Bias-corrected structural template image
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_output_grid
            Image to write out DWIs aligned to t1
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to
            FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed
            subject space to T1w
        dwi_sampling_grid
            A NIfTI1 file with the grid spacing and FoV to resample the DWIs
        b0_ref_image
            A Nifti of the b0 reference that was used for hmc and sdc
        intramodal_template
            The intramodal template image created from all b0 ref images

    **Outputs**

        dwi_t1
            dwi series, resampled to T1w space
        dwi_mask_t1
            dwi series mask in T1w space
        bvals_t1
            bvalues of the dwi series
        bvecs_t1
            bvecs after aligning to the T1w and resampling
        local_bvecs_t1
            voxelwise bvecs accounting for local displacements
        gradient_table_t1
            MRTrix-style gradient table

        dwi_mni
            dwi series, resampled to template space
        dwi_mask_mni
            dwi series mask in template space
        bvals_mni
            bvalues of the dwi series
        bvecs_mni
            bvecs after aligning to the T1w and resampling
        local_bvecs_mni
            voxelwise bvecs accounting for local displacements
        gradient_table_mni
            MRTrix-style gradient table

    """
    # Check the inputs
    if layout is not None:
        all_dwis = scan_groups['dwi_series']
        source_file = all_dwis[0]
        fieldmap_info = scan_groups['fieldmap_info']
    else:
        all_dwis = ['/fake/testing/path.nii.gz']
        source_file = all_dwis[0]
        fieldmap_info = {'suffix': None}

    fieldmap_type = fieldmap_info['suffix']

    fieldmap_type = fieldmap_info['suffix']
    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    dwi_nvols = 10

    # Determine resource usage
    for scan in all_dwis:
        if not os.path.exists(scan):
            # For docs building
            continue
        _dwi_nvols, _mem_gb = _create_mem_gb(scan)
        dwi_nvols += _dwi_nvols
        mem_gb['filesize'] += _mem_gb['filesize']
        mem_gb['resampled'] += _mem_gb['resampled']
        mem_gb['largemem'] += _mem_gb['largemem']

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_b0_to_t1',
            'b0_to_intramodal_template_transforms',
            'intramodal_template_to_t1_affine',
            'intramodal_template_to_t1_warp',
            't1_2_mni_forward_transform',
            'hmc_optimization_data',
            'dwi_files',
            'cnr_map',
            'bval_files',
            'bvec_files',
            'b0_ref_image',
            'intramodal_template',
            'b0_indices',
            'dwi_mask',
            'original_files',
            'hmc_xforms',
            'fieldwarps',
            'output_grid',
            'sbref_file', 'subjects_dir', 'subject_id',
            't1_preproc', 't1_brain', 't1_mask', 'mni_mask', 't1_seg', 't1_tpms',
            't1_aseg', 't1_aparc',
            't1_2_mni_reverse_transform', 't1_2_fsnative_forward_transform',
            't1_2_fsnative_reverse_transform', 'dwi_sampling_grid'
        ]),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_t1', 'dwi_mask_t1', 'cnr_map_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1',
            't1_b0_ref', 't1_b0_series', 'dwi_mni', 'dwi_mask_mni', 'cnr_map_mni', 'bvals_mni',
            'bvecs_mni', 'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'confounds',
            'gradient_table_mni', 'gradient_table_t1', 'hmc_optimization_data'
        ]),
        name='outputnode')

    gradient_plot = pe.Node(GradientPlot(), name='gradient_plot', run_without_submitting=True)
    ds_report_gradients = pe.Node(
        DerivativesDataSink(suffix='sampling_scheme'),
        name='ds_report_gradients', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    if make_intramodal_template:
        b0_to_im_template = pe.Node(SimpleBeforeAfterRPT(), name='b0_to_im_template')
        ds_report_intramodal = pe.Node(
            DerivativesDataSink(suffix='to_intramodal'),
            name='ds_report_intramodal', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, b0_to_im_template, [
                ('intramodal_template', 'after'),
                ('b0_ref_image', 'before')]),
            (b0_to_im_template, ds_report_intramodal, [('out_report', 'in_file')])
        ])

    # CONNECT TO DERIVATIVES #####################
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_prefix=output_prefix,
        source_file=source_file,
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template,
        write_local_bvecs=write_local_bvecs,
        hmc_model=hmc_model,
        shoreline_iters=shoreline_iters)

    workflow.connect([
        (inputnode, dwi_derivatives_wf, [('dwi_files', 'inputnode.source_file')]),
        (inputnode, outputnode, [('hmc_optimization_data', 'hmc_optimization_data')]),
        (outputnode, dwi_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('cnr_map_t1', 'inputnode.cnr_map_t1'),
          ('bvals_t1', 'inputnode.bvals_t1'),
          ('bvecs_t1', 'inputnode.bvecs_t1'),
          ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
          ('t1_b0_ref', 'inputnode.t1_b0_ref'),
          ('t1_b0_series', 'inputnode.t1_b0_series'),
          ('gradient_table_t1', 'inputnode.gradient_table_t1'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('cnr_map_mni', 'inputnode.cnr_map_mni'),
          ('bvals_mni', 'inputnode.bvals_mni'),
          ('bvecs_mni', 'inputnode.bvecs_mni'),
          ('local_bvecs_mni', 'inputnode.local_bvecs_mni'),
          ('mni_b0_ref', 'inputnode.mni_b0_ref'),
          ('mni_b0_series', 'inputnode.mni_b0_series'),
          ('gradient_table_mni', 'inputnode.gradient_table_mni'),
          ('confounds', 'inputnode.confounds'),
          ('hmc_optimization_data', 'inputnode.hmc_optimization_data')])
    ])

    workflow.connect([
        (inputnode, gradient_plot, [
            ('bvec_files', 'orig_bvec_files'),
            ('bval_files', 'orig_bval_files'),
            ('original_files', 'source_files')]),
        (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])
    ])

    if "T1w" in output_spaces:
        transform_dwis_t1 = init_dwi_trans_wf(name='transform_dwis_t1',
                                              template="ACPC",
                                              mem_gb=mem_gb['resampled'],
                                              use_fieldwarp=(fieldmap_type is not None
                                                             or use_syn),
                                              omp_nthreads=omp_nthreads,
                                              use_compression=False,
                                              to_mni=False,
                                              write_local_bvecs=write_local_bvecs)
        gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
        workflow.connect([
            (inputnode, transform_dwis_t1, [
                ('b0_indices', 'inputnode.b0_indices'),
                ('bval_files', 'inputnode.bval_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('cnr_map', 'inputnode.cnr_map'),
                ('t1_mask', 'inputnode.t1_mask'),
                ('dwi_mask', 'inputnode.dwi_mask'),
                ('hmc_xforms', 'inputnode.hmc_xforms'),
                ('fieldwarps', 'inputnode.fieldwarps'),
                ('dwi_files', 'inputnode.dwi_files'),
                ('dwi_sampling_grid', 'inputnode.output_grid'),
                ('b0_to_intramodal_template_transforms',
                 'inputnode.b0_to_intramodal_template_transforms'),
                ('intramodal_template_to_t1_affine',
                 'inputnode.intramodal_template_to_t1_affine'),
                ('intramodal_template_to_t1_warp',
                 'inputnode.intramodal_template_to_t1_warp'),
                ('itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_t1, outputnode, [('outputnode.bvals', 'bvals_t1'),
                                             ('outputnode.rotated_bvecs', 'bvecs_t1'),
                                             ('outputnode.dwi_resampled', 'dwi_t1'),
                                             ('outputnode.cnr_map_resampled', 'cnr_map_t1'),
                                             ('outputnode.local_bvecs', 'local_bvecs_t1'),
                                             ('outputnode.b0_series', 't1_b0_series'),
                                             ('outputnode.dwi_ref_resampled', 't1_b0_ref'),
                                             ('outputnode.resampled_dwi_mask', 'dwi_mask_t1')]),
            (outputnode, gradient_plot, [('bvecs_t1', 'final_bvec_file')]),
            (transform_dwis_t1, gtab_t1, [('outputnode.bvals', 'bval_file'),
                                          ('outputnode.rotated_bvecs', 'bvec_file')]),
            (gtab_t1, outputnode, [('gradient_file', 'gradient_table_t1')])])

    if "template" in output_spaces:
        transform_dwis_mni = init_dwi_trans_wf(name='transform_dwis_mni',
                                               template=template,
                                               mem_gb=mem_gb['resampled'],
                                               use_fieldwarp=(fieldmap_type is not None
                                                              or use_syn),
                                               omp_nthreads=omp_nthreads,
                                               use_compression=False,
                                               to_mni=True,
                                               write_local_bvecs=write_local_bvecs)
        gtab_mni = pe.Node(MRTrixGradientTable(), name='gtab_mni')
        workflow.connect([
            (inputnode, transform_dwis_mni, [
                ('b0_indices', 'inputnode.b0_indices'),
                ('bval_files', 'inputnode.bval_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('cnr_map', 'inputnode.cnr_map'),
                ('dwi_mask', 'inputnode.dwi_mask'),
                ('hmc_xforms', 'inputnode.hmc_xforms'),
                ('fieldwarps', 'inputnode.fieldwarps'),
                ('dwi_files', 'inputnode.dwi_files'),
                ('dwi_sampling_grid', 'inputnode.output_grid'),
                ('b0_to_intramodal_template_transforms',
                 'inputnode.b0_to_intramodal_template_transforms'),
                ('intramodal_template_to_t1_affine',
                 'inputnode.intramodal_template_to_t1_affine'),
                ('intramodal_template_to_t1_warp',
                 'inputnode.intramodal_template_to_t1_warp'),
                ('itk_b0_to_t1', 'inputnode.itk_b0_to_t1'),
                ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform')]),
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

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = source_file
    return workflow


def init_mask_finalize_wf(name="mask_finalize_wf"):
    """Creates a final mask using a combination of the t1 mask and dwi2mask
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1_mask', 'resampled_b0s']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['mask_file']), name='outputnode')
    workflow = Workflow(name=name)
    resample_t1_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_t1_mask')
    b0mask = pe.Node(afni.Automask(outputtype='NIFTI_GZ'), name='b0mask')
    or_mask = pe.Node(afni.Calc(outputtype='NIFTI_GZ', expr='step(a+b)'), name='or_mask')
    workflow.connect([
        (inputnode, resample_t1_mask, [
            ('t1_mask', 'in_file'),
            ('resampled_b0s', 'master')]),
        (inputnode, b0mask, [('resampled_b0s', 'in_file')]),
        (b0mask, or_mask, [('out_file', 'in_file_a')]),
        (resample_t1_mask, or_mask, [('out_file', 'in_file_b')]),
        (or_mask, outputnode, [('out_file', 'mask_file')])
    ])

    return workflow
