"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.dwi_merge import MergeDWIs
from ...interfaces.nilearn import Merge

# dwi workflows
from .merge import gen_denoising_boilerplate, init_merge_and_denoise_wf
from .qc import init_modelfree_qc_wf
from .util import get_source_file

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_pre_hmc_wf(
    scan_groups,
    preprocess_rpe_series,
    orientation,
    source_file,
    calculate_qc=True,
    name='pre_hmc_wf',
):
    """
    This workflow merges and denoises dwi scans. The outputs from this workflow is
    a single dwi file (optionally denoised) and corresponding bvals, bvecs.

    In the general case, a single warped group will be sent to this workflow. However,
    since eddy expects a single 4D input file, two warped groups can be processed
    separately and merged into a 4D file. This happens when ``preprocess_rpe_series`` is
    ``True``. FSL's eddy also requires data in LAS+ orientation.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf
        wf = init_dwi_pre_hmc_wf(['/completely/made/up/path/sub-01_dwi.nii.gz'],
                                  preprocess_rpe_series=False,
                                  orientation="LPS",

    **Parameters**

        orientation : str
            'LPS' or 'LAS'

    **Outputs**
        dwi_file
            a (potentially-denoised) dwi file
        bvec_file
            a bvec file
        bval_file
            a bval files
        sidecar_file
            a json sidecar file for the scan data
        b0_indices
            list of the positions of the b0 images in the dwi series
        b0_images
            list of paths to single-volume b0 images
        original_files
            list of paths to the original files that the single volumes came from
        original_grouping
            list of warped space group ids
        raw_concatenated
            4d image of the raw inputs concatenated (for QC and visualization)
    """
    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bval_file',
                'bvec_file',
                'json_file',
                'original_files',
                'denoising_confounds',
                'noise_images',
                'bias_images',
                'qc_file',
                'raw_concatenated',
                'validation_reports',
            ]
        ),
        name='outputnode',
    )
    dwi_series_pedir = scan_groups['dwi_series_pedir']
    dwi_series = scan_groups['dwi_series']

    workflow.__postdesc__ = gen_denoising_boilerplate()

    # Doing biascorr here is the old way.
    do_biascorr = False
    if config.workflow.b1_biascorrect_stage == 'legacy':
        do_biascorr = True
        config.loggers.workflow.warning('Applying bias correction before merging. Check results!')

    # Special case: Two reverse PE DWI series are going to get combined for eddy
    if preprocess_rpe_series:
        workflow.__desc__ = 'Images were grouped into two phase encoding polarity groups. '
        rpe_series = scan_groups['fieldmap_info']['rpe_series']
        # Merge, denoise, split, hmc on the plus series
        plus_files, minus_files = (
            (rpe_series, dwi_series)
            if dwi_series_pedir.endswith('-')
            else (dwi_series, rpe_series)
        )
        pe_axis = dwi_series_pedir[0]
        plus_source_file = get_source_file(plus_files, suffix='_PEplus')
        merge_plus = init_merge_and_denoise_wf(
            raw_dwi_files=plus_files,
            orientation=orientation,
            source_file=plus_source_file,
            phase_id=f'{pe_axis}+ phase-encoding direction',
            calculate_qc=False,
            do_biascorr=do_biascorr,
            name='merge_plus',
        )

        # Merge, denoise, split, hmc on the minus series
        minus_source_file = get_source_file(minus_files, suffix='_PEminus')
        merge_minus = init_merge_and_denoise_wf(
            raw_dwi_files=minus_files,
            orientation=orientation,
            source_file=minus_source_file,
            phase_id=f'{pe_axis}- phase-encoding direction',
            calculate_qc=False,
            do_biascorr=do_biascorr,
            name='merge_minus',
        )

        # Combine the original images from the splits into one 4D series + bvals/bvecs
        pm_validation = pe.Node(niu.Merge(2), name='pm_validation')
        pm_dwis = pe.Node(niu.Merge(2), name='pm_dwis')
        pm_bids_dwis = pe.Node(niu.Merge(2), name='pm_bids_dwis')
        pm_bvals = pe.Node(niu.Merge(2), name='pm_bvals')
        pm_bvecs = pe.Node(niu.Merge(2), name='pm_bvecs')
        pm_bias = pe.Node(niu.Merge(2), name='pm_bias')
        pm_noise_images = pe.Node(niu.Merge(2), name='pm_noise')
        pm_denoising_confounds = pe.Node(niu.Merge(2), name='pm_denoising_confounds')
        pm_raw_images = pe.Node(niu.Merge(2), name='pm_raw_images')
        rpe_concat = pe.Node(
            MergeDWIs(harmonize_b0_intensities=not config.workflow.no_b0_harmonization),
            name='rpe_concat',
        )
        raw_rpe_concat = pe.Node(Merge(is_dwi=True), name='raw_rpe_concat')
        qc_wf = init_modelfree_qc_wf(bvec_convention='DIPY' if orientation == 'LPS' else 'FSL')

        workflow.connect([
            # combine PE+
            (merge_plus, pm_dwis, [('outputnode.merged_image', 'in1')]),
            (merge_plus, pm_bids_dwis, [('outputnode.original_files', 'in1')]),
            (merge_plus, pm_bvals, [('outputnode.merged_bval', 'in1')]),
            (merge_plus, pm_bvecs, [('outputnode.merged_bvec', 'in1')]),
            (merge_plus, pm_bias, [('outputnode.bias_images', 'in1')]),
            (merge_plus, pm_noise_images, [('outputnode.noise_images', 'in1')]),
            (merge_plus, pm_raw_images, [('outputnode.merged_raw_image', 'in1')]),
            (merge_plus, pm_denoising_confounds, [('outputnode.denoising_confounds', 'in1')]),
            (merge_plus, pm_validation, [('outputnode.validation_reports', 'in1')]),

            # combine PE-
            (merge_minus, pm_dwis, [('outputnode.merged_image', 'in2')]),
            (merge_minus, pm_bids_dwis, [('outputnode.original_files', 'in2')]),
            (merge_minus, pm_bvals, [('outputnode.merged_bval', 'in2')]),
            (merge_minus, pm_bvecs, [('outputnode.merged_bvec', 'in2')]),
            (merge_minus, pm_bias, [('outputnode.bias_images', 'in2')]),
            (merge_minus, pm_noise_images, [('outputnode.noise_images', 'in2')]),
            (merge_minus, pm_raw_images, [('outputnode.merged_raw_image', 'in2')]),
            (merge_minus, pm_denoising_confounds, [('outputnode.denoising_confounds', 'in2')]),
            (merge_minus, pm_validation, [('outputnode.validation_reports', 'in2')]),

            (pm_dwis, rpe_concat, [('out', 'dwi_files')]),
            (pm_bids_dwis, rpe_concat, [('out', 'bids_dwi_files')]),
            (pm_bvals, rpe_concat, [('out', 'bval_files')]),
            (pm_bvecs, rpe_concat, [('out', 'bvec_files')]),
            (pm_denoising_confounds, rpe_concat, [('out', 'denoising_confounds')]),

            # Connect to the outputnode
            (rpe_concat, outputnode, [
                ('out_dwi', 'dwi_file'),
                ('out_bval', 'bval_file'),
                ('out_bvec', 'bvec_file'),
                ('original_images', 'original_files'),
                ('merged_denoising_confounds', 'denoising_confounds'),
            ]),
            (pm_validation, outputnode, [('out', 'validation_reports')]),
            (pm_noise_images, outputnode, [('out', 'noise_images')]),
            (pm_bias, outputnode, [('out', 'bias_images')]),
            (pm_raw_images, raw_rpe_concat, [('out', 'in_files')]),
            (raw_rpe_concat, outputnode, [('out_file', 'raw_concatenated')]),

            # Send the slice timings from "plus" to the next steps
            (merge_plus, outputnode, [('outputnode.merged_json', 'json_file')]),

            # Connect to the QC calculator
            (raw_rpe_concat, qc_wf, [('out_file', 'inputnode.dwi_file')]),
            (rpe_concat, qc_wf, [
                ('out_bval', 'inputnode.bval_file'),
                ('out_bvec', 'inputnode.bvec_file'),
            ]),
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_file')]),
        ])  # fmt:skip

        workflow.__postdesc__ += (
            'Both distortion groups were then merged into a '
            'single file, as required for the FSL workflows.\n\n'
        )
        return workflow

    workflow.__postdesc__ += '\n\n'
    merge_dwis = init_merge_and_denoise_wf(
        raw_dwi_files=dwi_series,
        orientation=orientation,
        calculate_qc=True,
        phase_id=dwi_series_pedir,
        do_biascorr=do_biascorr,
        source_file=source_file,
    )

    workflow.connect([
        (merge_dwis, outputnode, [
            ('outputnode.merged_image', 'dwi_file'),
            ('outputnode.merged_bval', 'bval_file'),
            ('outputnode.merged_bvec', 'bvec_file'),
            ('outputnode.merged_json', 'json_file'),
            ('outputnode.bias_images', 'bias_images'),
            ('outputnode.noise_images', 'noise_images'),
            ('outputnode.validation_reports', 'validation_reports'),
            ('outputnode.denoising_confounds', 'denoising_confounds'),
            ('outputnode.original_files', 'original_files'),
            ('outputnode.merged_raw_image', 'raw_concatenated'),
        ]),
    ])  # fmt:skip

    if calculate_qc:
        qc_wf = init_modelfree_qc_wf(bvec_convention='DIPY' if orientation == 'LPS' else 'FSL')
        workflow.connect([
            (merge_dwis, qc_wf, [
                ('outputnode.merged_raw_image', 'inputnode.dwi_file'),
                ('outputnode.merged_bval', 'inputnode.bval_file'),
                ('outputnode.merged_bvec', 'inputnode.bvec_file'),
            ]),
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_file')]),
        ])  # fmt:skip

    return workflow
