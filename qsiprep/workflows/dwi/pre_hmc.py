"""Orchestrating the dwi-preprocessing workflow.

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
from ...utils.bids import get_source_file

# dwi workflows
from .merge import SERIES_LIST_FIELDS, gen_denoising_boilerplate, init_merge_dwis_wf
from .qc import init_modelfree_qc_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_pre_hmc_wf(
    unit,
    orientation,
    source_file,
    do_biascorr,
    calculate_qc=True,
    name='pre_hmc_wf',
):
    """Build a workflow that merges denoised dwi scans before head motion correction.

    The member series arrive already conformed and denoised (see
    :func:`~qsiprep.workflows.dwi.merge.init_dwi_series_denoise_wf`), one list
    entry per file in ``unit.dwi_files`` and in the same order. The outputs from
    this workflow are a single dwi file and corresponding bvals, bvecs.

    In the general case, a single warped group will be sent to this workflow. However,
    since eddy expects a single 4D input file, two warped groups can be processed
    separately and merged into a 4D file. This happens when the unit has both phase
    encoding polarities (``unit.has_bidirectional_dwi``). FSL's eddy also requires
    data in LAS+ orientation.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        wf = init_dwi_pre_hmc_wf(
            make_preproc_unit(['/completely/made/up/path/sub-01_dwi.nii.gz']),
            orientation="LPS",
            source_file='/completely/made/up/path/sub-01_dwi.nii.gz',
            do_biascorr=True,
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The DWI series to merge. When the unit has both phase encoding
        polarities, each polarity is merged separately.
    orientation : str
        'LPS' or 'LAS', the orientation the series were conformed to
    source_file : str
        Source file used to name the merged outputs when the unit has a single
        phase encoding polarity.
    do_biascorr : bool
        Whether bias correction is applied to this output; used to write the
        methods boilerplate.
    calculate_qc : bool, optional
        Whether to calculate model-free QC metrics for the merged series when the
        unit has a single phase encoding polarity. Default is True.
    name : str, optional
        Name of workflow (default: ``pre_hmc_wf``)

    Inputs
    ------
    dwi_files
        conformed, denoised series, one per file in ``unit.dwi_files``
    bval_files
        bvals of each series
    bvec_files
        conformed bvecs of each series
    raw_dwi_files
        conformed, not denoised series
    noise_images
        noise image of each series
    denoising_confounds
        denoising confounds of each series
    validation_reports
        conformation report of each series

    Outputs
    -------
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
    inputnode = pe.Node(niu.IdentityInterface(fields=list(SERIES_LIST_FIELDS)), name='inputnode')
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
                'qc_file',
                'raw_concatenated',
                'validation_reports',
            ]
        ),
        name='outputnode',
    )
    workflow.__postdesc__ = gen_denoising_boilerplate(do_biascorr)

    # Special case: Two reverse PE DWI series are going to get combined for eddy
    if unit.has_bidirectional_dwi:
        workflow.__desc__ = 'Images were grouped into two phase encoding polarity groups. '
        all_files = list(unit.dwi_files)
        plus_files = list(unit.plus_files)
        minus_files = list(unit.minus_files)
        pe_axis = unit.pe_axis
        plus_source_file = get_source_file(plus_files, suffix='_PEplus')
        merge_plus = init_merge_dwis_wf(
            unit=unit,
            raw_dwi_files=plus_files,
            orientation=orientation,
            source_file=plus_source_file,
            phase_id=f'{pe_axis}+ phase-encoding direction',
            calculate_qc=False,
            name='merge_plus',
        )

        # Merge, split, hmc on the minus series
        minus_source_file = get_source_file(minus_files, suffix='_PEminus')
        merge_minus = init_merge_dwis_wf(
            unit=unit,
            raw_dwi_files=minus_files,
            orientation=orientation,
            source_file=minus_source_file,
            phase_id=f'{pe_axis}- phase-encoding direction',
            calculate_qc=False,
            name='merge_minus',
        )

        # The series lists cover the whole unit; each polarity takes its members
        # by position (known at build time from unit.dwi_files).
        select_plus = pe.Node(
            niu.Function(
                function=_select_polarity,
                input_names=['indices', *SERIES_LIST_FIELDS],
                output_names=list(SERIES_LIST_FIELDS),
            ),
            name='select_plus',
            run_without_submitting=True,
        )
        select_plus.inputs.indices = [all_files.index(path) for path in plus_files]
        select_minus = select_plus.clone('select_minus')
        select_minus.inputs.indices = [all_files.index(path) for path in minus_files]
        passthrough = [(field, field) for field in SERIES_LIST_FIELDS]
        into_merge = [(field, f'inputnode.{field}') for field in SERIES_LIST_FIELDS]

        # Combine the original images from the splits into one 4D series + bvals/bvecs
        pm_validation = pe.Node(niu.Merge(2), name='pm_validation')
        pm_dwis = pe.Node(niu.Merge(2), name='pm_dwis')
        pm_bids_dwis = pe.Node(niu.Merge(2), name='pm_bids_dwis')
        pm_bvals = pe.Node(niu.Merge(2), name='pm_bvals')
        pm_bvecs = pe.Node(niu.Merge(2), name='pm_bvecs')
        pm_noise_images = pe.Node(niu.Merge(2), name='pm_noise')
        pm_denoising_confounds = pe.Node(niu.Merge(2), name='pm_denoising_confounds')
        pm_raw_images = pe.Node(niu.Merge(2), name='pm_raw_images')
        rpe_concat = pe.Node(
            MergeDWIs(
                harmonize_b0_intensities=not config.workflow.no_b0_harmonization,
                merged_prefix=unit.output_name,
            ),
            name='rpe_concat',
        )
        raw_rpe_concat = pe.Node(Merge(is_dwi=True), name='raw_rpe_concat')
        qc_wf = init_modelfree_qc_wf(bvec_convention='DIPY' if orientation == 'LPS' else 'FSL')

        workflow.connect([
            (inputnode, select_plus, passthrough),
            (inputnode, select_minus, passthrough),
            (select_plus, merge_plus, into_merge),
            (select_minus, merge_minus, into_merge),

            # combine PE+
            (merge_plus, pm_dwis, [('outputnode.merged_image', 'in1')]),
            (merge_plus, pm_bids_dwis, [('outputnode.original_files', 'in1')]),
            (merge_plus, pm_bvals, [('outputnode.merged_bval', 'in1')]),
            (merge_plus, pm_bvecs, [('outputnode.merged_bvec', 'in1')]),
            (merge_plus, pm_noise_images, [('outputnode.noise_images', 'in1')]),
            (merge_plus, pm_raw_images, [('outputnode.merged_raw_image', 'in1')]),
            (merge_plus, pm_denoising_confounds, [('outputnode.denoising_confounds', 'in1')]),
            (merge_plus, pm_validation, [('outputnode.validation_reports', 'in1')]),

            # combine PE-
            (merge_minus, pm_dwis, [('outputnode.merged_image', 'in2')]),
            (merge_minus, pm_bids_dwis, [('outputnode.original_files', 'in2')]),
            (merge_minus, pm_bvals, [('outputnode.merged_bval', 'in2')]),
            (merge_minus, pm_bvecs, [('outputnode.merged_bvec', 'in2')]),
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
    merge_dwis = init_merge_dwis_wf(
        unit=unit,
        raw_dwi_files=list(unit.dwi_files),
        orientation=orientation,
        calculate_qc=True,
        phase_id=unit.pe_dir,
        source_file=source_file,
    )

    workflow.connect([
        (inputnode, merge_dwis, [
            (field, f'inputnode.{field}') for field in SERIES_LIST_FIELDS
        ]),
        (merge_dwis, outputnode, [
            ('outputnode.merged_image', 'dwi_file'),
            ('outputnode.merged_bval', 'bval_file'),
            ('outputnode.merged_bvec', 'bvec_file'),
            ('outputnode.merged_json', 'json_file'),
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


def _select_polarity(
    indices,
    dwi_files,
    bval_files,
    bvec_files,
    raw_dwi_files,
    noise_images,
    denoising_confounds,
    validation_reports,
):
    """Pick one polarity's member series out of the unit-wide series lists.

    ``indices`` are the positions of that polarity's files in ``unit.dwi_files``.
    A list that is empty because no step produced it (noise images and
    confounds when denoising is off) stays empty.
    """

    def _pick(items):
        if not items:
            return []
        return [items[index] for index in indices]

    return (
        _pick(dwi_files),
        _pick(bval_files),
        _pick(bvec_files),
        _pick(raw_dwi_files),
        _pick(noise_images),
        _pick(denoising_confounds),
        _pick(validation_reports),
    )
