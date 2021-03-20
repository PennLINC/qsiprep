"""
Merging Distortion Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""
import logging
from nipype.interfaces import utility as niu
import nipype.pipeline.engine as pe
from .derivatives import init_dwi_derivatives_wf
from .qc import init_modelfree_qc_wf
from .util import init_dwi_reference_wf
from ...engine import Workflow
from ...interfaces import DerivativesDataSink
from ...interfaces.mrtrix import MRTrixGradientTable
from ...interfaces.reports import GradientPlot, SeriesQC
from ...interfaces.dwi_merge import AveragePEPairs, MergeDWIs
from ...interfaces.nilearn import Merge
from .qc import init_mask_overlap_wf, init_interactive_report_wf


DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_distortion_group_merge_wf(merging_strategy, inputs_list, hmc_model, reportlets_dir,
                                   harmonize_b0_intensities, b0_threshold, output_prefix,
                                   source_file, output_dir, template, shoreline_iters,
                                   mem_gb=3, omp_nthreads=1,
                                   name="distortion_group_merge_wf"):
    """Create an unbiased intramodal template for a subject. This aligns the b=0 references
    from all the scans of a subject. Can be rigid, affine or nonlinear (BSplineSyN).

    **Parameters**
        inputs_list: list of inputs
            List if identifiers for inputs. There will be bvals, bvecs, niis and original
            bvecs.
        merging_strategy: str
            'average': averages images that originally sampled the same q-space coordinate
            'concat': concatenates images in the 4th dimension


    **Inputs**

        [workflow_name]_image...
            One input for each volume in each input image.
        [workflow_name]_bval...
            One input for each input image. path to the corresponding bval file
        [workflow_name]_bvec...
            One input for each input image. path to the corresponding final bvec file
        [workflow_name]_original_bvec...
            One input for each input image. Path to the original bvec file
        [workflow_name]_original_image...
            One input for each input image. Path to the original dwi file
        [workflow_name]_raw_concatenated_image
            One input for each input image. Path to the original images after concatenation
        [workflow_name]_confounds
            One input for each input image. Path to the confounds files
        [workflow_name]_b0_ref
            One input for each input image. Path to the b=0 reference image
        [workflow_name]_carpetplot_data
            One input for each input image. Path to the hmc carpetplot data

    **Outputs**
        merged_image
            The input images merged into a single image (averaged or concatenated)
        merged_bval
            The bvals corresponding to merged_image
        merged_bvec
            The bvecs corresponding to merged_image
        merged_qc
            The Before/After QC file
        merged_interactive_report
            The interactive report data
    """

    workflow = Workflow(name=name)
    source_file = "dwi/" + source_file
    sanitized_inputs = [name.replace('-', '_') for name in inputs_list]
    input_names = ['t1_brain', 't1_mask', 't1_seg']
    for suffix in ["_image", "_bval", "_bvec", "_original_bvec", "_b0_ref", "_cnr",
                   "_carpetplot_data", "_original_image", "_raw_concatenated_image",
                   "_confounds"]:
        input_names += [name + suffix for name in sanitized_inputs]
    inputnode = pe.Node(niu.IdentityInterface(fields=input_names), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["merged_image", "merged_bval", "merged_bvec", "merged_qc", "merged_cnr_map"
                    'dwi_mask_t1', 'cnr_map_t1', 'merged_bval', 'bvecs_t1',
                    'local_bvecs_t1', 't1_b0_ref', 'confounds',
                    'gradient_table_t1', 'hmc_optimization_data']),
        name='outputnode')

    num_inputs = len(input_names)
    merge_images = pe.Node(niu.Merge(num_inputs), name='merge_images')
    merge_bval = pe.Node(niu.Merge(num_inputs), name='merge_bval')
    merge_bvec = pe.Node(niu.Merge(num_inputs), name='merge_bvec')
    merge_original_bvec = pe.Node(niu.Merge(num_inputs), name='merge_original_bvec')
    merge_original_image = pe.Node(niu.Merge(num_inputs), name='merge_original_image')
    merge_b0_refs = pe.Node(niu.Merge(num_inputs), name='merge_b0_refs')
    merge_raw_concatenated_image = pe.Node(niu.Merge(num_inputs),
                                           name='merge_raw_concatenated_image')
    merge_confounds = pe.Node(niu.Merge(num_inputs), name='merge_confounds')
    merge_cnrs = pe.Node(niu.Merge(num_inputs), name='merge_cnrs')
    merge_carpetplot_data = pe.Node(niu.Merge(num_inputs), name='merge_carpetplot_data')

    # Merge the input data from each distortion group: safe even if eddy was used
    for input_num, input_name in enumerate(sanitized_inputs):
        merge_input_name = 'in%d' % (input_num + 1)
        workflow.connect([
            (inputnode, merge_images, [(input_name + "_image", merge_input_name)]),
            (inputnode, merge_bval, [(input_name + "_bval", merge_input_name)]),
            (inputnode, merge_bvec, [(input_name + "_bvec", merge_input_name)]),
            (inputnode, merge_original_bvec, [(input_name + "_original_bvec", merge_input_name)]),
            (inputnode, merge_original_image, [(input_name + "_original_image",
                                                merge_input_name)]),
            (inputnode, merge_raw_concatenated_image, [(input_name + "_raw_concatenated_image",
                                                        merge_input_name)]),
            (inputnode, merge_b0_refs, [(input_name + "_b0_ref", merge_input_name)]),
            (inputnode, merge_cnrs, [(input_name + "_cnr", merge_input_name)]),
            (inputnode, merge_confounds, [(input_name + "_confounds", merge_input_name)]),
            (inputnode, merge_carpetplot_data, [
                (input_name + "_carpetplot_data", merge_input_name)])
        ])

    if merging_strategy.lower() == 'average':
        distortion_merger = pe.Node(AveragePEPairs(), name='distortion_merger')
        workflow.connect([
            (merge_original_bvec, distortion_merger, [('out', 'original_bvec_files')]),
            (merge_carpetplot_data, distortion_merger, [('out', 'carpetplot_data')])
        ])
    elif merging_strategy.startswith('concat'):
        distortion_merger = pe.Node(MergeDWIs(), name='distortion_merger')
    b0_ref_wf = init_dwi_reference_wf(name='merged_b0_ref', register_t1=False,
                                      gen_report=True, source_file=source_file)
    concat_cnr_images = pe.Node(Merge(), name='concat_cnr_images')

    workflow.connect([
        (merge_images, distortion_merger, [('out', 'dwi_files')]),
        (merge_bval, distortion_merger, [('out', 'bval_files')]),
        (merge_bvec, distortion_merger, [('out', 'bvec_files')]),
        (merge_original_image, distortion_merger, [('out', 'bids_dwi_files')]),
        (merge_raw_concatenated_image, distortion_merger, [('out', 'raw_concatenated_files')]),
        (merge_b0_refs, distortion_merger, [('out', 'b0_refs')]),
        (merge_confounds, distortion_merger, [('out', 'denoising_confounds')]),
        (merge_cnrs, concat_cnr_images, [('out', 'in_files')]),
        (concat_cnr_images, outputnode, [('out_file', 'cnr_map_t1')])
    ])

    # Calculate QC on the merged raw and processed data
    raw_qc_wf = init_modelfree_qc_wf(name='raw_qc_wf')
    processed_qc_wf = init_modelfree_qc_wf(name='processed_qc_wf')
    # Combine all the QC measures for a series QC
    series_qc = pe.Node(SeriesQC(output_file_name=output_prefix), name='series_qc')
    ds_series_qc = pe.Node(
        DerivativesDataSink(desc='ImageQC', suffix='dwi', source_file=source_file,
                            base_directory=output_dir),
        name='ds_series_qc', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    interactive_report_wf = init_interactive_report_wf()
    # Write the interactive report json
    ds_interactive_report = pe.Node(
        DerivativesDataSink(suffix='dwiqc', source_file=source_file,
                            base_directory=output_dir),
        name='ds_interactive_report', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    # CONNECT TO DERIVATIVES
    gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
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
        output_spaces=["T1w"],
        template=template,
        write_local_bvecs=False,
        hmc_model=hmc_model,
        shoreline_iters=shoreline_iters)

    workflow.connect([
        # Mask the new b=0 reference
        (distortion_merger, b0_ref_wf, [
            ('merged_b0_ref', 'inputnode.b0_template')]),
        (inputnode, b0_ref_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg')]),
        (b0_ref_wf, outputnode, [
            ('outputnode.ref_image', 't1_b0_ref'),
            ('outputnode.dwi_mask', 'dwi_mask_t1')]),

        # QC connections
        (distortion_merger, raw_qc_wf, [
            ('merged_raw_dwi', 'inputnode.dwi_file'),
            ('merged_raw_bvec', 'inputnode.bvec_file'),
            ('out_bval', 'inputnode.bval_file')]),
        (distortion_merger, processed_qc_wf, [
            ('out_dwi', 'inputnode.dwi_file'),
            ('out_bvec', 'inputnode.bvec_file'),
            ('out_bval', 'inputnode.bval_file')]),
        (distortion_merger, series_qc, [('merged_denoising_confounds', 'confounds_file')]),
        (raw_qc_wf, series_qc, [
            ('outputnode.qc_summary', 'pre_qc')]),
        (processed_qc_wf, series_qc, [
            ('outputnode.qc_summary', 't1_qc')]),
        (b0_ref_wf, t1_dice_calc, [
            ('outputnode.dwi_mask', 'inputnode.dwi_mask')]),
        (inputnode, t1_dice_calc, [
            ('t1_mask', 'inputnode.anatomical_mask')]),
        (t1_dice_calc, series_qc, [('outputnode.dice_score', 't1_dice_score')]),
        (series_qc, ds_series_qc, [('series_qc_file', 'in_file')]),

        (distortion_merger, outputnode, [
            ('out_bval', 'merged_bval'),
            ('out_bvec', 'bvecs_t1'),
            ('out_dwi', 'merged_image'),
            ('merged_denoising_confounds', 'confounds')
        ]),

        # Report the merged gradients
        (outputnode, gradient_plot, [('bvecs_t1', 'final_bvec_file')]),
        (distortion_merger, gradient_plot, [
            ('out_bvec', 'orig_bvec_files'),
            ('out_bval', 'orig_bval_files'),
            ('original_images', 'source_files')]),
        (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')]),
        (distortion_merger, gtab_t1, [('out_bval', 'bval_file'),
                                      ('out_bvec', 'bvec_file')]),
        (gtab_t1, outputnode, [('gradient_file', 'gradient_table_t1')]),

        # Connections for the interactive report
        (distortion_merger, interactive_report_wf, [
            ('merged_raw_dwi', 'inputnode.raw_dwi_file'),
            ('out_dwi', 'inputnode.processed_dwi_file'),
            ('out_bval', 'inputnode.bval_file'),
            ('out_bvec', 'inputnode.bvec_file'),
            ('merged_carpetplot_data', 'inputnode.carpetplot_data'),
            ('merged_denoising_confounds', 'inputnode.confounds_file')]),
        (interactive_report_wf, outputnode, [
            ('outputnode.out_report', 'interactive_report')]),
        (b0_ref_wf, interactive_report_wf, [
            ('outputnode.dwi_mask', 'inputnode.mask_file')]),
        (series_qc, interactive_report_wf, [('series_qc_file', 'inputnode.series_qc_file')]),
        (interactive_report_wf, ds_interactive_report, [
            ('outputnode.out_report', 'in_file')]),

        # Connect merged results to outputs
        (outputnode, dwi_derivatives_wf, [
            ('merged_image', 'inputnode.dwi_t1'),
            ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
            ('cnr_map_t1', 'inputnode.cnr_map_t1'),
            ('merged_bval', 'inputnode.bvals_t1'),
            ('bvecs_t1', 'inputnode.bvecs_t1'),
            ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
            ('t1_b0_ref', 'inputnode.t1_b0_ref'),
            ('gradient_table_t1', 'inputnode.gradient_table_t1'),
            ('confounds', 'inputnode.confounds'),
            ('hmc_optimization_data', 'inputnode.hmc_optimization_data')]),
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir

    return workflow
