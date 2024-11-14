"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf

"""

import os

from nipype.interfaces import utility as niu
from nipype.interfaces.base import isdefined
from nipype.pipeline import engine as pe

from ... import config
from ...engine import Workflow
from ...interfaces import DerivativesDataSink, DerivativesMaybeDataSink
from ...interfaces.confounds import DMRISummary
from ...interfaces.reports import DiffusionSummary
from ...interfaces.utils import TestInput
from ..fieldmap.pepolar import init_extended_pepolar_report_wf

# dwi workflows
from ..fieldmap.unwarp import init_fmap_unwarp_report_wf
from .confounds import init_dwi_confs_wf
from .fsl import init_fsl_hmc_wf
from .hmc_sdc import init_qsiprep_hmcsdc_wf
from .pre_hmc import init_dwi_pre_hmc_wf
from .registration import init_b0_to_anat_registration_wf, init_direct_b0_acpc_wf
from .util import _create_mem_gb, _get_wf_name

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_preproc_wf(
    scan_groups,
    t2w_sdc,
    output_prefix,
    source_file,
    anatomical_template,
) -> Workflow:
    """
    This workflow controls the dwi preprocessing stages of qsiprep.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.base import init_dwi_preproc_wf
        wf = init_dwi_preproc_wf(scan_groups={'dwi_series': ['fake.nii'],
                                  'fieldmap_info': {'suffix': None},
                                  'dwi_series_pedir': 'j'},
                                 output_prefix='',
                                 source_file='/data/bids/sub-1/dwi/sub-1_dwi.nii.gz')

    Parameters
    ----------
    scan_groups : list of dicts
        List of dicts grouping files by PE-dir
    t2w_sdc : bool
        Include T2w scans in distortion correction
    output_prefix : str
        beginning of the output file name (eg 'sub-1_buds-j')
    source_file : str
        The file name template used for derivatives

    Inputs
    ------
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

    Outputs
    -------
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
    confounds_file
        estimated motion parameters and zipper scores
    raw_qc_file
        DSI Studio QC file for the raw data
    raw_concatenated
        concatenated raw images for a qc report
    carpetplot_data
        path to a file containing carpetplot data

    See Also
    --------

    * :py:func:`~qsiprep.workflows.dwi.hmc.init_dwi_hmc_wf`
    * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_t1_trans_wf`
    * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_reg_wf`
    * :py:func:`~qsiprep.workflows.dwi.confounds.init_dwi_confounds_wf`

    """
    dwi_only = config.workflow.anat_modality == "none"
    output_dir = config.execution.output_dir

    # Check the inputs
    if config.execution.layout is not None:
        all_dwis = scan_groups["dwi_series"]
        fieldmap_info = scan_groups["fieldmap_info"]
        dwi_metadata = config.execution.layout.get_metadata(all_dwis[0])
    else:
        all_dwis = ["/fake/testing/path.nii.gz"]
        fieldmap_info = {"suffix": None}
        dwi_metadata = {}

    fieldmap_type = fieldmap_info["suffix"]
    doing_bidirectional_pepolar = fieldmap_type == "rpe_series"
    if fieldmap_type is not None:
        fmap_key = "phase1" if fieldmap_type == "phase" else fieldmap_type

        if fieldmap_type != "syn":
            fieldmap_file = fieldmap_info[fmap_key]
            # There can be a bunch of rpe series, so don't get the info yet
            if fmap_key not in ("rpe_series", "epi", "dwi"):
                fieldmap_info["metadata"] = config.execution.layout.get_metadata(fieldmap_file)

    mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}
    dwi_nvols = 10
    # Determine resource usage
    for scan in all_dwis:
        if not os.path.exists(scan):
            # For docs building
            continue
        _dwi_nvols, _mem_gb = _create_mem_gb(scan)
        dwi_nvols += _dwi_nvols
        mem_gb["filesize"] += _mem_gb["filesize"]
        mem_gb["resampled"] += _mem_gb["resampled"]
        mem_gb["largemem"] += _mem_gb["largemem"]

    wf_name = _get_wf_name(output_prefix)
    workflow = Workflow(name=wf_name)
    config.loggers.workflow.debug(
        "Creating DWI processing workflow for <%s> (%.2f GB). "
        "Memory resampled/largemem=%.2f/%.2f GB.",
        source_file,
        mem_gb["filesize"],
        mem_gb["resampled"],
        mem_gb["largemem"],
    )
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_files",
                "sbref_file",
                "subjects_dir",
                "subject_id",
                "t1_preproc",
                "t1_brain",
                "t1_mask",
                "t1_seg",
                "t1_aseg",
                "t1_aparc",
                "t1_2_mni_forward_transform",
                "t2w_unfatsat",
                "t1_2_mni_reverse_transform",
                "t1_2_fsnative_forward_transform",
                "t1_2_fsnative_reverse_transform",
                "t2w_files",
                "dwi_sampling_grid",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "confounds",
                "hmc_optimization_data",
                "itk_b0_to_t1",
                "noise_images",
                "bias_images",
                "dwi_files",
                "cnr_map",
                "bval_files",
                "bvec_files",
                "b0_ref_image",
                "b0_indices",
                "dwi_mask",
                "hmc_xforms",
                "fieldwarps",
                "sbref_file",
                "original_files",
                "original_bvecs",
                "raw_qc_file",
                "coreg_score",
                "raw_concatenated",
                "carpetplot_data",
                "sdc_scaling_images",
            ]
        ),
        name="outputnode",
    )
    workflow.__desc__ = """

Diffusion data preprocessing

: """

    pre_hmc_wf = init_dwi_pre_hmc_wf(
        scan_groups=scan_groups,
        preprocess_rpe_series=doing_bidirectional_pepolar,
        orientation="LAS" if config.workflow.hmc_model == "eddy" else "LPS",
        source_file=source_file,
    )
    test_pre_hmc_connect = pe.Node(TestInput(), name="test_pre_hmc_connect")
    if config.workflow.hmc_model in ("none", "3dSHORE", "tensor"):
        if not config.workflow.hmc_model == "none" and config.workflow.shoreline_iters < 1:
            raise Exception(
                f"--shoreline-iters must be > 0 when --hmc-model is {config.workflow.hmc_model}"
            )
        hmc_wf = init_qsiprep_hmcsdc_wf(
            scan_groups=scan_groups,
            source_file=source_file,
            dwi_metadata=dwi_metadata,
            t2w_sdc=t2w_sdc,
            anatomical_template=anatomical_template,
        )

    elif config.workflow.hmc_model == "eddy":
        hmc_wf = init_fsl_hmc_wf(
            scan_groups=scan_groups,
            source_file=source_file,
            dwi_metadata=dwi_metadata,
            t2w_sdc=t2w_sdc,
            name="hmc_sdc_wf",
        )

    workflow.connect([
        (pre_hmc_wf, hmc_wf, [
            ('outputnode.dwi_file', 'inputnode.dwi_file'),
            ('outputnode.bval_file', 'inputnode.bval_file'),
            ('outputnode.bvec_file', 'inputnode.bvec_file'),
            ('outputnode.json_file', 'inputnode.json_file'),
            ('outputnode.original_files', 'inputnode.original_files')]),
        (inputnode, hmc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t2w_unfatsat', 'inputnode.t2w_unfatsat'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform')]),
        (pre_hmc_wf, outputnode, [
            ('outputnode.qc_file', 'raw_qc_file'),
            ('outputnode.original_files', 'original_files'),
            ('outputnode.bvec_file', 'original_bvecs'),
            ('outputnode.bias_images', 'bias_images'),
            ('outputnode.noise_images', 'noise_images'),
            ('outputnode.raw_concatenated', 'raw_concatenated')]),
        (pre_hmc_wf, test_pre_hmc_connect, [('outputnode.raw_concatenated', 'test1')])
    ])  # fmt:skip

    if not dwi_only:
        # calculate dwi registration to T1w
        b0_coreg_wf = init_b0_to_anat_registration_wf(write_report=True)
    else:
        b0_coreg_wf = init_direct_b0_acpc_wf(write_report=True)

    ds_report_coreg = pe.Node(
        DerivativesDataSink(
            datatype="figures",
            suffix="dwi",
            desc="acpc" if dwi_only else "coreg",
            source_file=source_file,
        ),
        name="ds_report_coreg",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # Fieldmap reports should vary depending on which type of correction is performed
    # PEPOLAR (epi, rpe series) will produce potentially much more detailed reports
    doing_topup = (
        fieldmap_type in ("epi", "rpe_series")
        and "topup" in config.workflow.pepolar_method.lower()
    )
    if fieldmap_type not in ("epi", "rpe_series", None) or doing_topup:
        fmap_unwarp_report_wf = init_fmap_unwarp_report_wf()
        ds_report_sdc = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="sdc",
                suffix="b0",
                source_file=source_file,
            ),
            name="ds_report_sdc",
            mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True,
        )

        workflow.connect([
            (inputnode, fmap_unwarp_report_wf, [
                ('t1_seg', 'inputnode.in_seg')]),
            (hmc_wf, outputnode, [
                ('outputnode.sdc_scaling_images', 'sdc_scaling_images')]),
            (hmc_wf, fmap_unwarp_report_wf, [
                ('outputnode.pre_sdc_template', 'inputnode.in_pre'),
                ('outputnode.b0_template', 'inputnode.in_post')]),
            (b0_coreg_wf, fmap_unwarp_report_wf, [
                ('outputnode.itk_b0_to_t1', 'inputnode.in_xfm')]),
            (fmap_unwarp_report_wf, ds_report_sdc, [('outputnode.report', 'in_file')])
        ])  # fmt:skip

    # DRBUDDI has some extra reports that we want to save. Make sure we get them!
    if (
        fieldmap_type in ("epi", "rpe_series")
        and "drbuddi" in config.workflow.pepolar_method.lower()
    ):

        if os.path.exists(t2w_sdc):
            extended_pepolar_report_wf = init_extended_pepolar_report_wf(segment_t2w=t2w_sdc)
        else:
            extended_pepolar_report_wf = init_extended_pepolar_report_wf()

        ds_report_fa_sdc = pe.Node(
            DerivativesMaybeDataSink(
                datatype="figures",
                desc="sdc",
                suffix="fa",
                source_file=source_file,
            ),
            name="ds_report_fa_sdc",
            mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True,
        )

        ds_report_b0_sdc = pe.Node(
            DerivativesMaybeDataSink(
                datatype="figures",
                desc="sdcdrbuddi",
                suffix="b0" if not t2w_sdc else "b0t2w",
                source_file=source_file,
            ),
            name="ds_report_b0_sdc",
            mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True,
        )

        workflow.connect([
            (hmc_wf, extended_pepolar_report_wf, [
                ("outputnode.b0_template", "inputnode.b0_ref"),
                ("outputnode.fieldmap_type", "inputnode.fieldmap_type"),
                ("outputnode.b0_up_image", "inputnode.b0_up_image"),
                ("outputnode.b0_up_corrected_image", "inputnode.b0_up_corrected_image"),
                ("outputnode.b0_down_image", "inputnode.b0_down_image"),
                ("outputnode.b0_down_corrected_image", "inputnode.b0_down_corrected_image"),
                ("outputnode.up_fa_image", "inputnode.up_fa_image"),
                ("outputnode.up_fa_corrected_image", "inputnode.up_fa_corrected_image"),
                ("outputnode.down_fa_image", "inputnode.down_fa_image"),
                ("outputnode.down_fa_corrected_image", "inputnode.down_fa_corrected_image"),
                ("outputnode.t2w_image", "inputnode.t2w_image")]),
            (b0_coreg_wf, extended_pepolar_report_wf, [
                ('outputnode.itk_b0_to_t1', 'inputnode.t1w_seg_transform')]),
            (inputnode, extended_pepolar_report_wf, [
                ("t1_seg", "inputnode.t1w_seg")]),
            (extended_pepolar_report_wf, ds_report_fa_sdc, [
                ("outputnode.fa_sdc_report", "in_file")]),
            (extended_pepolar_report_wf, ds_report_b0_sdc, [
                ("outputnode.b0_sdc_report", "in_file")])
        ])  # fmt:skip

    summary = pe.Node(
        DiffusionSummary(
            pe_direction=scan_groups["dwi_series_pedir"],
            hmc_model=config.workflow.hmc_model,
            b0_to_t1w_transform=config.workflow.b0_to_t1w_transform,
            hmc_transform=config.workflow.hmc_transform,
            denoise_method=config.workflow.denoise_method,
            dwi_denoise_window=config.workflow.dwi_denoise_window,
        ),
        name="summary",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, b0_coreg_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1_2_fsnative_reverse_transform',
             'inputnode.t1_2_fsnative_reverse_transform')]),
        (hmc_wf, b0_coreg_wf, [('outputnode.b0_template',
                                'inputnode.ref_b0_brain')]),
        (hmc_wf, summary, [('outputnode.sdc_method', 'distortion_correction')]),
        (b0_coreg_wf, ds_report_coreg, [('outputnode.report', 'in_file')]),
        (b0_coreg_wf, outputnode, [
            (('outputnode.itk_b0_to_t1', _get_first), 'itk_b0_to_t1'),
            ('outputnode.coreg_metric', 'coreg_score')])
    ])  # fmt:skip

    # Compute and gather confounds
    confounds_wf = init_dwi_confs_wf()
    ds_confounds = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=str(output_dir),
            desc="confounds",
            suffix="timeseries",
        ),
        name="ds_confounds",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(confounds_wf, ds_confounds, [("outputnode.confounds_file", "in_file")])])

    # Carpetplot and confounds plot
    conf_plot = pe.Node(DMRISummary(), name="conf_plot", mem_gb=mem_gb["resampled"])
    ds_report_dwi_conf = pe.Node(
        DerivativesDataSink(
            datatype="figures",
            desc="carpetplot",
            suffix="dwi",
            source_file=source_file,
        ),
        name="ds_report_dwi_conf",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (hmc_wf, confounds_wf, [
            ('outputnode.slice_quality', 'inputnode.sliceqc_file'),
            ('outputnode.motion_params', 'inputnode.motion_params')]),
        (pre_hmc_wf, confounds_wf, [
            ('outputnode.denoising_confounds', 'inputnode.denoising_confounds'),
            ('outputnode.bval_file', 'inputnode.bval_file'),
            ('outputnode.bvec_file', 'inputnode.bvec_file'),
            ('outputnode.original_files', 'inputnode.original_files')]),
        (hmc_wf, conf_plot, [
            ('outputnode.slice_quality', 'sliceqc_file'),
            ('outputnode.b0_template_mask', 'sliceqc_mask')]),
        (confounds_wf, conf_plot, [
            ('outputnode.confounds_file', 'confounds_file')]),
        (confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),
        (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('carpetplot_json', 'carpetplot_data')]),
        (hmc_wf, outputnode, [
            ('outputnode.hmc_optimization_data', 'hmc_optimization_data'),
            ('outputnode.b0_indices', 'b0_indices'),
            ('outputnode.bval_files', 'bval_files'),
            ('outputnode.bvec_files_to_transform', 'bvec_files'),
            ('outputnode.b0_template', 'b0_ref_image'),
            ('outputnode.cnr_map', 'cnr_map'),
            ('outputnode.b0_template_mask', 'dwi_mask'),
            ('outputnode.to_dwi_ref_affines', 'hmc_xforms'),
            ('outputnode.to_dwi_ref_warps', 'fieldwarps'),
            ('outputnode.dwi_files_to_transform', 'dwi_files')])
    ])  # fmt:skip

    # Reporting
    ds_report_summary = pe.Node(
        DerivativesDataSink(
            datatype="figures",
            desc="summary",
            suffix="dwi",
            source_file=source_file,
        ),
        name="ds_report_summary",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (pre_hmc_wf, summary, [('outputnode.validation_reports', 'validation_reports')]),
        (summary, ds_report_summary, [('out_report', 'in_file')])
    ])  # fmt:skip

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.base_directory = str(config.execution.output_dir)
            src_file = workflow.get_node(node).inputs.source_file
            if not isdefined(src_file) or src_file is None:
                workflow.get_node(node).inputs.source_file = source_file

    return workflow


def _get_first(lll):
    return lll[0]
