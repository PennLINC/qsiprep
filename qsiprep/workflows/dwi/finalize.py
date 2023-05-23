"""
Final steps on the preprocessed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_finalize_wf

"""

import os

from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.base import isdefined
from ...interfaces import DerivativesDataSink
from ...niworkflows.interfaces.registration import SimpleBeforeAfterRPT
from ...interfaces.reports import GradientPlot, SeriesQC
from ...interfaces.mrtrix import MRTrixGradientTable, DWIBiasCorrect
from ...interfaces.dwi_merge import SplitResampledDWIs, MergeFinalConfounds
from ...interfaces.confounds import GatherConfounds
from ...interfaces.gradients import ExtractB0s
from ...interfaces.nilearn import Merge
from ...interfaces.dsi_studio import DSIStudioBTable
from ...engine import Workflow
from .util import init_dwi_reference_wf

# dwi workflows
from .util import _create_mem_gb
from .resampling import init_dwi_trans_wf
from .derivatives import init_dwi_derivatives_wf
from .qc import init_mask_overlap_wf, init_interactive_report_wf, init_modelfree_qc_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_finalize_wf(scan_groups,
                         name,
                         output_prefix,
                         hmc_model,
                         shoreline_iters,
                         reportlets_dir,
                         output_resolution,
                         template,
                         output_dir,
                         omp_nthreads,
                         b0_threshold,
                         write_local_bvecs,
                         do_biascorr,
                         make_intramodal_template,
                         source_file,
                         write_derivatives=True,
                         layout=None):
    """
    This workflow controls the resampling parts of the dwi preprocessing workflow.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.base import init_dwi_finalize_wf
        wf = init_dwi_finalize_wf(name='finalize_wf',
                                  omp_nthreads=1,
                                  reportlets_dir='.',
                                  output_dir='.',
                                  output_resolution=2.0,
                                  template='MNI152NLin2009cAsym',
                                  b0_threshold=100,
                                  low_mem=False,
                                  output_prefix='',
                                  make_intramodal_template=False,
                                  write_local_bvecs=False,
                                  do_biascorr=True,
                                  source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
                                  num_dwi=1)

    **Parameters**

        output_prefix : str
            beginning of the output file name (eg 'sub-1_buds-j')
        ignore : list
            Preprocessing steps to skip (eg "fieldmaps")
        reportlets_dir : str
            Directory in which to save reportlets
        template : str
            Name of template targeted by ``template`` output space
        output_dir : str
            Directory in which to save derivatives
        output_resolution : float
            Output voxel resolution in mm
        pepolar_method : str
            Either 'DRBUDDI' or 'TOPUP'. The method for SDC when EPI fieldmaps are used.
        omp_nthreads : int
            Maximum number of threads an individual process may use
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        layout : BIDSLayout
            BIDSLayout structure to enable metadata retrieval
        write_derivatives: bool
            Is this the final output? If so, write the final derivatives. If these
            resampled outputs will be combined with other distortion groups at the end,
            then return the resampled, non-concatenated images

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
        source_file
            The file name template used for derivatives
        raw_qc_file
            The QC file from the DWI data before any preprocessing
        raw_concatenated
            The original raw images in a single 4D file
        carpetplot_data
            File containing carpetplot data

    **Outputs**

        dwi_t1
            dwi series, resampled to T1w space. If write_derivitaves, this is a
            4d file. Otherwise it's a list of resampled images.
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

    """
    # Check the inputs
    if layout is not None:
        all_dwis = scan_groups['dwi_series']
        if 'rpe_series' in scan_groups:
            all_dwis += scan_groups['rpe_series']
    else:
        all_dwis = ['/fake/testing/path.nii.gz']

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
            't1_preproc', 't1_brain', 't1_mask', 't1_seg',
            't1_aseg', 't1_aparc',
            't1_2_mni_reverse_transform', 't1_2_fsnative_forward_transform',
            't1_2_fsnative_reverse_transform', 'dwi_sampling_grid', 'raw_qc_file',
            'coreg_score', 'raw_concatenated', 'confounds', 'carpetplot_data',
            'sdc_scaling_images'
        ]),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_t1', 'dwi_mask_t1', 'cnr_map_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1',
            't1_b0_ref', 'confounds', 'gradient_table_t1', 'btable_t1', 'hmc_optimization_data'
        ]),
        name='outputnode')

    if make_intramodal_template:
        b0_to_im_template = pe.Node(SimpleBeforeAfterRPT(), name='b0_to_im_template')
        ds_report_intramodal = pe.Node(
            DerivativesDataSink(suffix='tointramodal', source_file=source_file,
                                base_directory=reportlets_dir),
            name='ds_report_intramodal', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, b0_to_im_template, [
                ('intramodal_template', 'after'),
                ('b0_ref_image', 'before')]),
            (b0_to_im_template, ds_report_intramodal, [('out_report', 'in_file')])
        ])

    # Do the resampling
    transform_dwis_t1 = init_dwi_trans_wf(source_file=source_file,
                                          name='transform_dwis_t1',
                                          template="ACPC",
                                          mem_gb=mem_gb['resampled'],
                                          omp_nthreads=omp_nthreads,
                                          output_resolution=output_resolution,
                                          use_compression=False,
                                          to_mni=False,
                                          write_local_bvecs=write_local_bvecs,
                                          concatenate=write_derivatives)

    # Apply denoising to the interpolated data if requested
    final_denoise_wf = init_finalize_denoising_wf(
        name="final_denoise_wf",
        omp_nthreads=omp_nthreads,
        source_file=source_file,
        do_biascorr=do_biascorr and write_derivatives,
        num_dwi_acquisitions=len(all_dwis)
    )

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
            ('itk_b0_to_t1', 'inputnode.itk_b0_to_t1'),
            ('sdc_scaling_images', 'inputnode.sdc_scaling_images')]),
        (transform_dwis_t1, outputnode, [
            ('outputnode.bvals', 'bvals_t1'),
            ('outputnode.rotated_bvecs', 'bvecs_t1'),
            ('outputnode.cnr_map_resampled', 'cnr_map_t1'),
            ('outputnode.local_bvecs', 'local_bvecs_t1')]),
        (inputnode, final_denoise_wf, [('confounds', 'inputnode.confounds')]),
        (transform_dwis_t1, final_denoise_wf, [
            ('outputnode.dwi_resampled', 'inputnode.dwi_t1'),
            ('outputnode.bvals', 'inputnode.dwi_t1_bval'),
            ('outputnode.rotated_bvecs', 'inputnode.dwi_t1_bvec'),
            ('outputnode.b0_series', 'inputnode.t1_b0_series'),
            ('outputnode.dwi_ref_resampled', 'inputnode.t1_b0_ref'),
            ('outputnode.resampled_dwi_mask', 'inputnode.dwi_mask_t1'),
            ('outputnode.resampled_qc', 'inputnode.series_qc_t1')]),
        (final_denoise_wf, outputnode, [
            ('outputnode.confounds', 'confounds'),
            ('outputnode.dwi_t1', 'dwi_t1'),
            ('outputnode.t1_b0_ref', 't1_b0_ref'),
            ('outputnode.dwi_mask_t1', 'dwi_mask_t1'),
            ])
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = source_file

    # The workflow is done if we will be concatenating images later
    if not write_derivatives:
        return workflow

    # Finish up the derivatives process
    interactive_report_wf = init_interactive_report_wf()

    # We need to attach outputs to the interactive report
    workflow.connect([
        (inputnode, interactive_report_wf, [
            ('raw_concatenated', 'inputnode.raw_dwi_file'),
            ('carpetplot_data', 'inputnode.carpetplot_data'),
            ('confounds', 'inputnode.confounds_file')]),
        (transform_dwis_t1, interactive_report_wf, [
             ('outputnode.dwi_resampled', 'inputnode.processed_dwi_file'),
             ('outputnode.resampled_dwi_mask', 'inputnode.mask_file'),
             ('outputnode.bvals', 'inputnode.bval_file'),
             ('outputnode.rotated_bvecs', 'inputnode.bvec_file')]),
        (interactive_report_wf, outputnode, [('outputnode.out_report', 'interactive_report')])
    ])

    # CONNECT TO DERIVATIVES #####################
    gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
    btab_t1 = pe.Node(DSIStudioBTable(bvec_convention="DIPY"), name='btab_t1')
    t1_dice_calc = init_mask_overlap_wf(name='t1_dice_calc')
    gradient_plot = pe.Node(GradientPlot(), name='gradient_plot', run_without_submitting=True)
    ds_report_gradients = pe.Node(
        DerivativesDataSink(suffix='sampling_scheme', source_file=source_file),
        name='ds_report_gradients', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_prefix=output_prefix,
        source_file=source_file,
        output_dir=output_dir,
        template=template,
        write_local_bvecs=write_local_bvecs,
        hmc_model=hmc_model,
        shoreline_iters=shoreline_iters)

    # Combine all the QC measures for a series QC
    series_qc = pe.Node(SeriesQC(output_file_name=output_prefix), name='series_qc')
    ds_series_qc = pe.Node(
        DerivativesDataSink(desc='ImageQC', suffix='dwi', source_file=source_file,
                            base_directory=output_dir),
        name='ds_series_qc', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Write the carpetplot data
    ds_carpetplot = pe.Node(
        DerivativesDataSink(desc='SliceQC', suffix='dwi', source_file=source_file,
                            base_directory=output_dir),
        name='ds_carpetplot', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Write the interactive report json
    ds_interactive_report = pe.Node(
        DerivativesDataSink(suffix='dwiqc', source_file=source_file,
                            base_directory=output_dir),
        name='ds_interactive_report', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (interactive_report_wf, ds_interactive_report, [
            ('outputnode.out_report', 'in_file')]),
        (inputnode, series_qc, [
            ('raw_qc_file', 'pre_qc'),
            ('confounds', 'confounds_file')]),
        (inputnode, ds_carpetplot, [('carpetplot_data', 'in_file')]),
        (t1_dice_calc, series_qc, [('outputnode.dice_score', 't1_dice_score')]),
        (final_denoise_wf, series_qc, [
            ('outputnode.series_qc_postproc', 't1_qc_postproc')]),
        (series_qc, ds_series_qc, [('series_qc_file', 'in_file')]),
        (transform_dwis_t1, series_qc, [
            ('outputnode.cnr_map_resampled', 't1_cnr_file')]),
        (final_denoise_wf, series_qc, [
            ('outputnode.dwi_mask_t1', 't1_mask_file'),
            ('outputnode.t1_b0_series', 't1_b0_series')]),
        (series_qc, interactive_report_wf, [('series_qc_file', 'inputnode.series_qc_file')]),
        (inputnode, dwi_derivatives_wf, [('dwi_files', 'inputnode.source_file')]),
        (inputnode, outputnode, [('hmc_optimization_data', 'hmc_optimization_data')]),
        (transform_dwis_t1, series_qc, [('outputnode.resampled_qc', 't1_qc')]),
        (transform_dwis_t1, t1_dice_calc, [
            ('outputnode.resampled_dwi_mask', 'inputnode.dwi_mask')]),
        (outputnode, gradient_plot, [('bvecs_t1', 'final_bvec_file')]),
        (transform_dwis_t1, gtab_t1, [('outputnode.bvals', 'bval_file'),
                                      ('outputnode.rotated_bvecs', 'bvec_file')]),
        (transform_dwis_t1, btab_t1, [('outputnode.bvals', 'bval_file'),
                                      ('outputnode.rotated_bvecs', 'bvec_file')]),
        (inputnode, t1_dice_calc, [
            ('t1_mask', 'inputnode.anatomical_mask')]),
        (gtab_t1, outputnode, [('gradient_file', 'gradient_table_t1')]),
        (btab_t1, outputnode, [('btable_file', 'btable_t1')]),
        (outputnode, dwi_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('cnr_map_t1', 'inputnode.cnr_map_t1'),
          ('bvals_t1', 'inputnode.bvals_t1'),
          ('bvecs_t1', 'inputnode.bvecs_t1'),
          ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
          ('t1_b0_ref', 'inputnode.t1_b0_ref'),
          ('gradient_table_t1', 'inputnode.gradient_table_t1'),
          ('btable_t1', 'inputnode.btable_t1'),
          ('confounds', 'inputnode.confounds'),
          ('hmc_optimization_data', 'inputnode.hmc_optimization_data')]),
        (inputnode, gradient_plot, [
            ('bvec_files', 'orig_bvec_files'),
            ('bval_files', 'orig_bval_files'),
            ('original_files', 'source_files')]),
        (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])
    ])
    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            if not isdefined(workflow.get_node(node).inputs.source_file):
                workflow.get_node(node).inputs.source_file = source_file

    return workflow


def init_finalize_denoising_wf(name,
                               omp_nthreads,
                               source_file,
                               do_biascorr,
                               num_dwi_acquisitions,
                               split_biascorr=False,
                               do_patch2self=False,
                               ):
    """
    Some denoising can only happen after images have been aligned
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'confounds',
            'dwi_t1',
            'dwi_t1_bval',
            'dwi_t1_bvec',
            't1_b0_series',
            't1_b0_ref',
            'dwi_mask_t1',
            'series_qc_t1']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'confounds',
            'dwi_t1',
            't1_b0_series',
            't1_b0_ref',
            'dwi_mask_t1',
            'series_qc_postproc']),
        name="outputnode")
    workflow = Workflow(name=name)
    # If no denoising, send it back
    if not (do_biascorr or do_patch2self):
        workflow.connect([
            (inputnode, outputnode, [
                ('confounds', 'confounds'),
                ('dwi_t1', 'dwi_t1'),
                ('t1_b0_series', 't1_b0_series'),
                ('t1_b0_ref', 't1_b0_ref'),
                ('dwi_mask_t1', 'dwi_mask_t1')
            ])
        ])
        return workflow

    # Hold the results of maybe bias correction
    bias_corrected = pe.Node(
        niu.IdentityInterface(
            fields=["dwi_t1", "bias_field", "bias_confounds"]),
        name="bias_corrected")

    if do_biascorr:
        if not split_biascorr:
            biascorr = pe.Node(
                DWIBiasCorrect(method='ants'),
                name='biascorr',
                n_procs=omp_nthreads)
            ds_report_biascorr = pe.Node(
                DerivativesDataSink(suffix=name + '_biascorr',
                                    source_file=source_file),
                name='ds_report_' + name + '_biascorr',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, biascorr, [
                    ("dwi_t1", "in_file"),
                    ("dwi_t1_bval", "in_bval"),
                    ("dwi_t1_bvec", "in_bvec"),
                    ("dwi_mask_t1", "mask")]),
                (biascorr, ds_report_biascorr, [('out_report', 'in_file')]),
                (biascorr, bias_corrected, [
                    ('out_file', 'dwi_t1'),
                    ('bias_image', 'bias_field'),
                    ('nmse_text', 'bias_confounds')])
            ])

        else:
            biascorrs = []
            reports = []
            scan_split = pe.Node(
                SplitResampledDWIs(n_images=num_dwi_acquisitions),
                name="scan_split")
            gather_corrected_images = pe.Node(niu.Merge(num_dwi_acquisitions),
                                              name="gather_corrected_images")
            gather_nmse_txts = pe.Node(niu.Merge(num_dwi_acquisitions),
                                       name="gather_nmse_txts")
            gather_bias_images = pe.Node(niu.Merge(num_dwi_acquisitions),
                                         name="gather_bias_images")

            merge_corrected_images = pe.Node(Merge(is_dwi=True, dtype="f4"),
                                             name="merge_corrected_images")
            merge_bias_images = pe.Node(Merge(is_dwi=True, dtype="f4"),
                                        name="merge_bias_images")

            # Loop over each original image
            for scan_num in range(num_dwi_acquisitions):
                scan_num += 1
                biascorrs.append(
                    pe.Node(DWIBiasCorrect(method='ants'),
                            name='biascorr%d' % scan_num,
                            n_procs=omp_nthreads))
                reports.append(
                    pe.Node(
                        DerivativesDataSink(suffix=name + '_biascorr%d' % scan_num),
                        name='ds_report_' + name + '_biascorr%d' % scan_num,
                        run_without_submitting=True,
                        mem_gb=DEFAULT_MEMORY_MIN_GB))
                workflow.connect([
                    (inputnode, biascorrs[-1], [("dwi_mask_t1", "mask")]),
                    (scan_split, biascorrs[-1], [
                        ('dwi_file_%d'%scan_num, 'in_file'),
                        ('bval_file_%d'%scan_num, 'in_bval'),
                        ('bvec_file_%d'%scan_num, 'in_bvec')]),
                    (biascorrs[-1], gather_corrected_images, [
                        ('out_file', 'in%d'%scan_num)]),
                    (biascorrs[-1], gather_nmse_txts, [
                        ('nmse_text', 'in%d'%scan_num)]),
                    (biascorrs[-1], gather_bias_images, [
                        ('bias_image', 'in%d'%scan_num)]),
                    (biascorrs[-1], reports[-1], [
                        ('out_report', 'in_file')]),
                    (scan_split, reports[-1], [
                        ("source_file_%d"%scan_num, "source_file")]),
                ])
            workflow.connect([
                (inputnode, scan_split, [
                    ("dwi_t1", "dwi_file"),
                    ("dwi_t1_bval", "bval_file"),
                    ("dwi_t1_bvec", "bvec_file"),
                    ("confounds", "confounds")
                ]),
                (gather_corrected_images, merge_corrected_images, [("out", "in_files")]),
                (merge_corrected_images, bias_corrected, [("out_file", "dwi_t1")]),
                (gather_nmse_txts, bias_corrected, [("out", "bias_confounds")]),
                (gather_bias_images, merge_bias_images, [("out", "in_files")]),
                (merge_bias_images, bias_corrected, [("out_file", "bias_field")])
            ])

    p2s_buffernode = pe.Node(
        niu.IdentityInterface(fields=["denoised_image", "noise_image", "nmse_text"]),
        name="p2s_buffernode")
    if do_patch2self:
        raise NotImplementedError()
    else:
        workflow.connect([
            (bias_corrected, p2s_buffernode, [("dwi_t1", "dwi_t1")])
        ])

    # Extract some additional derivatives from the corrected
    extract_b0_series = pe.Node(ExtractB0s(), name="extract_b0_series")
    final_b0_ref = init_dwi_reference_wf(register_t1=False, gen_report=True,
                                         desc='resampled', name='final_b0_ref',
                                         source_file=source_file, omp_nthreads=omp_nthreads)

    # Calculate QC metrics on the resampled data
    calculate_qc = init_modelfree_qc_wf(omp_nthreads=omp_nthreads,
                                        bvec_convention="DIPY",  # Resampled is always LPS+
                                        name='calculate_qc')
    update_confounds = pe.Node(MergeFinalConfounds(), name='update_confounds')

    workflow.connect([
        # Get the new b=0 data from the denoised images
        (p2s_buffernode, extract_b0_series, [("dwi_t1", "dwi_series")]),
        (inputnode, extract_b0_series, [("dwi_t1_bval", "bval_file")]),
        (extract_b0_series, final_b0_ref, [('b0_average', 'inputnode.b0_template')]),
        (inputnode, final_b0_ref, [('dwi_mask_t1', 'inputnode.t1_mask')]),
        (extract_b0_series, outputnode, [('b0_series', 't1_b0_series')]),
        (final_b0_ref, outputnode, [
            ('outputnode.ref_image', 't1_b0_ref'),
            ('outputnode.dwi_mask', 'dwi_mask_t1')]),

        # New QC on the whole series
        (inputnode, calculate_qc, [
            ('dwi_t1_bval', 'inputnode.bval_file'),
            ('dwi_t1_bvec', 'inputnode.bvec_file')]),
        (p2s_buffernode, calculate_qc, [('dwi_t1', 'inputnode.dwi_file')]),
        (calculate_qc, outputnode, [('outputnode.qc_summary', 'series_qc_postproc')]),

        # Update confounds with new nmse's
        (inputnode, update_confounds, [("confounds", "confounds")]),
        (bias_corrected, update_confounds, [
            ("bias_confounds", "bias_correction_confounds")]),
        (p2s_buffernode, update_confounds, [
            ("nmse_text", "patch2self_correction_confounds")]),

        # The final version
        (p2s_buffernode, outputnode, [
            ('dwi_t1', 'dwi_t1')]),

    ])
    return workflow