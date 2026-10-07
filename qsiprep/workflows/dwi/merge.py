# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Merge and denoise dwi images.

Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import pandas as pd
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from qsiplan.models import strip_nii_ext

from ... import config
from ...interfaces import ConformDwi, DerivativesDataSink
from ...interfaces.dipy import Patch2Self
from ...interfaces.dwi_merge import MergeDWIs, PhaseToRad, StackConfounds
from ...interfaces.mrtrix import (
    ComplexToMagnitude,
    DWIDenoise,
    DWIDenoise2,
    MRDeGibbs,
    MRTrixGradientTable,
    PolarToComplex,
)
from ...interfaces.nilearn import Merge
from ...interfaces.svht import SVHTDeGibbs, SVHTDenoise
from ...interfaces.tortoise import Gibbs
from ...utils.bids import IMPORTANT_DWI_FIELDS, update_metadata_from_nifti_header
from ...utils.misc import (
    SVHT_DEFAULTS,
    check_dwidenoise2_demodulation,
    check_svht_extent,
    check_svht_phase,
    describe_dwidenoise2,
    describe_svht,
    load_dwidenoise2_config,
    load_svht_config,
)
from .qc import init_modelfree_qc_wf
from .util import _get_wf_name

DEFAULT_MEMORY_MIN_GB = 0.01


# Each raw DWI series is conformed and denoised exactly once, by
# init_dwi_series_denoise_wf at the subject level, and the results travel
# down to the per-unit workflows as lists ordered like ``unit.dwi_files``.
# (series workflow output, list field carried downstream)
SERIES_FIELDS = (
    ('dwi_file', 'dwi_files'),
    ('bval_file', 'bval_files'),
    ('bvec_file', 'bvec_files'),
    ('raw_dwi_file', 'raw_dwi_files'),
    ('noise_image', 'noise_images'),
    ('confounds', 'denoising_confounds'),
    ('validation_report', 'validation_reports'),
)
SERIES_LIST_FIELDS = tuple(list_field for _, list_field in SERIES_FIELDS)


def init_dwi_series_denoise_wf(
    dwi_file,
    metadata,
    phase_file=None,
    orientation='LPS',
    name=None,
):
    """Build a workflow that conforms and denoises one raw DWI series.

    A series may belong to several preprocessing units (virtual acquisitions
    list it under more than one ``MultipartID``), and denoising depends only on
    the series itself, so this workflow is built once per raw file at the
    subject level and its outputs are shared by every unit that uses the file.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.merge import init_dwi_series_denoise_wf
        wf = init_dwi_series_denoise_wf(
            '/path/to/dwi/sub-1_dwi.nii.gz',
            metadata={'PhaseEncodingDirection': 'j'},
        )

    Parameters
    ----------
    dwi_file : str
        the raw DWI series, in its original BIDS directory
    metadata : dict
        the sidecar metadata of ``dwi_file`` (``PreprocUnit.metadata_for``)
    phase_file : str, optional
        the ``part-phase`` companion of ``dwi_file``, when the acquisition is
        complex-valued and the phase has not been ignored
    orientation : str, optional
        Orientation the series is conformed to ('LPS' or 'LAS').
    name : str, optional
        Name of workflow. Derived from the file name when omitted.

    Outputs
    -------
    dwi_file
        the series, conformed and denoised
    bval_file
        bvals of the series
    bvec_file
        bvecs of the series, conformed
    raw_dwi_file
        the series, conformed but not denoised
    noise_image
        noise image estimated by the denoiser (when one ran)
    confounds
        per-volume denoising/unringing confounds (when any step ran)
    validation_report
        HTML segment reporting header problems found while conforming
    """
    if name is None:
        _, fname, _ = split_filename(dwi_file)
        name = _get_wf_name(fname).replace('preproc', 'denoise')
    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[series_field for series_field, _ in SERIES_FIELDS]),
        name='outputnode',
    )

    row = get_acq_parameters_df([dwi_file], metadata_lookup=lambda _: metadata).iloc[0]

    conform_dwi = pe.Node(
        ConformDwi(orientation=orientation, dwi_file=dwi_file),
        name='conform_dwi',
    )
    use_phase = phase_file is not None and 'phase' not in config.workflow.ignore
    if phase_file is not None:
        config.loggers.workflow.info('Phase file found for %s', dwi_file)
    denoise_wf = init_dwi_denoising_wf(
        partial_fourier=row.PartialFourier,
        phase_encoding_direction=row.PhaseEncodingAxis,
        source_file=dwi_file,
        n_volumes=row.NumVolumes,
        use_phase=use_phase,
        name='denoise_wf',
    )
    workflow.connect([
        (conform_dwi, denoise_wf, [
            ('bval_file', 'inputnode.bval_file'),
            ('bvec_file', 'inputnode.bvec_file'),
            ('dwi_file', 'inputnode.dwi_file'),
        ]),
        (conform_dwi, outputnode, [
            ('dwi_file', 'raw_dwi_file'),
            ('out_report', 'validation_report'),
        ]),
        (denoise_wf, outputnode, [
            ('outputnode.dwi_file', 'dwi_file'),
            ('outputnode.bval_file', 'bval_file'),
            ('outputnode.bvec_file', 'bvec_file'),
            ('outputnode.noise_image', 'noise_image'),
            ('outputnode.confounds', 'confounds'),
        ]),
    ])  # fmt:skip

    if use_phase:
        conform_phase = pe.Node(
            ConformDwi(orientation=orientation, dwi_file=phase_file),
            name='conform_phase',
        )
        workflow.connect([
            (conform_phase, denoise_wf, [('dwi_file', 'inputnode.dwi_phase_file')]),
        ])  # fmt:skip

    return workflow


def init_merge_dwis_wf(
    unit,
    raw_dwi_files,
    orientation,
    source_file,
    calculate_qc=False,
    phase_id='same',
    name='merge_dwis_wf',
):
    """Build a workflow that merges already-denoised DWI series.

    The series arrive conformed and denoised from
    :func:`init_dwi_series_denoise_wf`, one list entry per file in
    ``raw_dwi_files`` and in the same order.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.merge import init_merge_dwis_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        wf = init_merge_dwis_wf(
            make_preproc_unit(['/path/to/dwi/sub-1_dwi.nii.gz']),
            ['/path/to/dwi/sub-1_dwi.nii.gz'],
            orientation='LPS',
            source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        the unit these series belong to; sidecar metadata comes from its
        records, so the layout is never re-read
    raw_dwi_files : list
        list of raw (in their original BIDS directory) dwi nifti files
    orientation : str
        Orientation the series were conformed to ('LPS' or 'LAS'). Selects the
        bvec convention used for QC ('DIPY' for 'LPS', 'FSL' otherwise).
    source_file : str
        Source file whose name (without extension) is used as the prefix of the
        merged outputs.
    calculate_qc : bool, optional
        Whether to calculate DSI Studio QC metrics on the merged raw data.
        Default is False.
    phase_id : str, optional
        Label for the distortion group, used in the methods boilerplate.
        Default is ``'same'``.
    name : str, optional
        Name of workflow (default: ``merge_dwis_wf``)

    Inputs
    ------
    dwi_files
        conformed, denoised series, one per entry of ``raw_dwi_files``
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
    merged_image
        dwi series, conformed, denoised if requested
    merged_raw_image
        dwi series, conformed, raw
    merged_bval
        bvals from merged images
    merged_bvec
        bvecs from merged images
    merged_json
        JSON file containing slice timings for slice2vol
    noise_images
        image(s) created by ``dwidenoise``
    denoising_confounds
        confounds from denoising, aligned to the merged series
    original_files
        names of the original files for each volume
    qc_summary
        DSI Studio QC text file
    validation_reports
        the conformation reports, passed through

    """
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads
    inputnode = pe.Node(niu.IdentityInterface(fields=list(SERIES_LIST_FIELDS)), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'merged_image',
                'merged_raw_image',
                'merged_bval',
                'merged_bvec',
                'merged_json',
                'noise_images',
                'denoising_confounds',
                'original_files',
                'qc_summary',
                'validation_reports',
            ]
        ),
        name='outputnode',
    )
    desc = []

    merge_dwis = pe.Node(
        MergeDWIs(
            bids_dwi_files=raw_dwi_files,
            b0_threshold=config.workflow.b0_threshold,
            harmonize_b0_intensities=not config.workflow.no_b0_harmonization,
            merged_prefix=strip_nii_ext(source_file),
            scan_metadata={scan: unit.metadata_for(scan) for scan in raw_dwi_files},
        ),
        name='merge_dwis',
        n_procs=omp_nthreads,
    )
    num_dwis = len(raw_dwi_files)
    if num_dwis > 1:
        desc.append(
            'A total of %d DWI series in the %s distortion group were '
            'concatenated, with preprocessing operations performed on individual '
            'DWI series before concatenation.' % (num_dwis, phase_id)
        )
    workflow.__desc__ = ' '.join(desc)

    # Get an orientation-conformed version of the raw inputs and their gradients
    raw_merge = pe.Node(Merge(is_dwi=True), name='raw_merge', n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, merge_dwis, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('denoising_confounds', 'denoising_confounds'),
        ]),
        (inputnode, raw_merge, [('raw_dwi_files', 'in_files')]),
        (inputnode, outputnode, [
            ('noise_images', 'noise_images'),
            ('validation_reports', 'validation_reports'),
        ]),
        (raw_merge, outputnode, [('out_file', 'merged_raw_image')]),
        (merge_dwis, outputnode, [
            ('out_dwi', 'merged_image'),
            ('merged_denoising_confounds', 'denoising_confounds'),
            ('original_images', 'original_files'),
            ('out_bval', 'merged_bval'),
            ('out_bvec', 'merged_bvec'),
            ('merged_metadata', 'merged_json'),
        ]),
    ])  # fmt:skip

    # Get a QC score for the raw data
    if calculate_qc:
        qc_wf = init_modelfree_qc_wf(
            bvec_convention='DIPY' if orientation == 'LPS' else 'FSL',
        )
        workflow.connect([
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_summary')]),
            (raw_merge, qc_wf, [('out_file', 'inputnode.dwi_file')]),
            (merge_dwis, qc_wf, [
                ('out_bval', 'inputnode.bval_file'),
                ('out_bvec', 'inputnode.bvec_file'),
            ]),
        ])  # fmt:skip

    return workflow


class _ImageChain:
    """Track the image flowing through the denoising steps, and its data domain.

    The denoising steps form a linear chain, each reading the previous step's output.
    Whether that image is complex-valued is not a property of the chain position but of
    which steps have run: ``dwidenoise`` handed complex data emits complex data, and so
    does ``mrdegibbs`` on MRtrix3's development branch, while ``dwibiascorrect`` and
    TORTOISE's ``rpg`` are magnitude-only. ``svht_denoise`` handed phase data emits the
    signed real-axis rotation, which is not magnitude either and is flagged the same
    way. Tracking the domain alongside the current image lets :func:`to_magnitude`
    insert the split at the last step that can still consume complex data, instead of
    always splitting right after denoising.
    """

    def __init__(self, workflow, node, field, omp_nthreads):
        self.workflow = workflow
        self.source = (node, field)
        self.is_complex = False
        self._omp_nthreads = omp_nthreads

    def feed(self, node, field='in_file'):
        """Connect the current image to ``field`` on ``node``."""
        source_node, source_field = self.source
        self.workflow.connect([(source_node, node, [(source_field, field)])])

    def advance(self, node, field='out_file', is_complex=False):
        """Make ``field`` on ``node`` the current image."""
        self.source = (node, field)
        self.is_complex = is_complex

    def to_magnitude(self):
        """Reduce the current image to magnitude, if it is not already.

        A no-op on real-valued data, so callers can invoke it before any
        magnitude-only step without checking first. Only one ``split_complex``
        node is ever created, because the first call clears ``is_complex``.
        """
        if not self.is_complex:
            return

        split_complex = pe.Node(
            ComplexToMagnitude(),
            name='split_complex',
            n_procs=self._omp_nthreads,
        )
        self.feed(split_complex, 'complex_file')
        self.advance(split_complex, 'out_file', is_complex=False)


def init_dwi_denoising_wf(
    source_file,
    partial_fourier,
    phase_encoding_direction,
    n_volumes,
    use_phase,
    name='denoise_wf',
):
    """Build a workflow to denoise a DWI series.

    Parameters
    ----------
    source_file : str
        path to the original dwi file
    partial_fourier : float
        fraction of k-space acquired
    phase_encoding_direction : str
        direction of phase encoding
    n_volumes : int
        number of volumes in the DWI series.
        Used to determine the window size for denoising if dwidenoise is used
        and the 'auto' option is selected.
    use_phase : bool
        True if phase data are available for the DWI scan.
        If True, and ``denoise_method`` is ``dwidenoise`` or ``dwidenoise2``, then
        the denoiser will be run on the complex-valued data. If ``denoise_method``
        is ``svht``, then ``svht_denoise`` will rotate the magnitude and phase data
        onto the real axis before denoising.
    name : str, optional
        name of the workflow

    Inputs
    ------
    dwi_file
        path to the dwi file
    bval_file
        path to the bval file
    bvec_file
        path to the bvec file
    dwi_phase_file
        path to the dwi phase file (optional)

    Outputs
    -------
    dwi_file
        path to the denoised dwi file
    bval_file
        path to the denoised bval file
    bvec_file
        path to the denoised bvec file
    noise_image
        path to the noise image
    confounds
        path to the confounds file
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'dwi_phase_file']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bval_file',
                'bvec_file',
                'noise_image',
                'confounds',
            ],
        ),
        name='outputnode',
    )
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads
    desc = '\n\n'

    # The chain starts at the raw input file, in the magnitude domain
    chain = _ImageChain(workflow, inputnode, 'dwi_file', omp_nthreads)

    workflow.connect([
        # XXX: Why pass the bval and bvec files through unmodified?
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
    ])  # fmt:skip

    # Which steps to apply?
    denoise_method = config.workflow.denoise_method
    dwidenoise2_params = {}
    if denoise_method == 'dwidenoise2':
        if config.workflow.denoise_config is not None:
            dwidenoise2_params = load_dwidenoise2_config(config.workflow.denoise_config)
        check_dwidenoise2_demodulation(dwidenoise2_params, use_phase)
    svht_params = dict(SVHT_DEFAULTS)
    if denoise_method == 'svht':
        if config.workflow.denoise_config is not None:
            svht_params.update(load_svht_config(config.workflow.denoise_config))
        check_svht_phase(svht_params, use_phase)
        check_svht_extent(svht_params, n_volumes)

    unringing_method = config.workflow.unringing_method
    do_denoise = denoise_method in ('patch2self', 'dwidenoise', 'dwidenoise2', 'svht')
    do_unringing = config.workflow.unringing_method in ('mrdegibbs', 'rpg', 'svht')
    harmonize_b0s = not config.workflow.no_b0_harmonization

    # Only the dwidenoise variants denoise complex data; the rest use magnitude alone,
    # except svht_denoise, which reads the phase itself and denoises the real-axis rotation.
    denoise_complex = do_denoise and denoise_method.startswith('dwidenoise') and use_phase
    denoise_real_axis = denoise_method == 'svht' and use_phase
    # Only the development-branch mrdegibbs reads and writes complex data.
    unring_complex = (
        denoise_complex
        and unringing_method == 'mrdegibbs'
        and config.workflow.mrtrix_version == 'dev'
    )
    if denoise_complex and unringing_method == 'mrdegibbs' and not unring_complex:
        config.loggers.workflow.warning(
            'Complex-valued Gibbs unringing is available with --mrtrix-version dev. '
            'The magnitude data will be unrung instead.'
        )

    # How many steps in the denoising pipeline
    num_steps = sum(map(int, [do_denoise, do_unringing, harmonize_b0s]))
    merge_confounds = pe.Node(niu.Merge(num_steps), name='merge_confounds')

    # Add the steps
    step_num = 1  # Merge inputs start at 1
    last_step = ''
    if do_denoise:
        ds_report_denoising = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='denoising',
                source_file=source_file,
            ),
            name=f'ds_report_{name}_denoising',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        # Build the denoiser. The node is the same whether it is handed magnitude-only or
        # complex-valued data; only the data feeding it differs, which is wired up below.
        if denoise_method == 'dwidenoise2':
            # dwidenoise2 sizes its patches per iteration from its multi-resolution schedule,
            # so there is no kernel to configure and dwidenoise_window does not apply here.
            denoiser = pe.Node(
                DWIDenoise2(nthreads=omp_nthreads, **dwidenoise2_params),
                name='denoiser',
                n_procs=omp_nthreads,
            )

            # dwidenoise2 needs the gradient table to demean by shell. Temporary
            # workaround for a bug in dwidenoise2: supply the gradients as a single
            # MRtrix-format table instead of using -fslgrad.
            gradient_table = pe.Node(MRTrixGradientTable(), name='gradient_table')
            workflow.connect([
                (inputnode, gradient_table, [
                    ('bval_file', 'bval_file'),
                    ('bvec_file', 'bvec_file'),
                ]),
                (gradient_table, denoiser, [('gradient_file', 'grad_file')]),
            ])  # fmt:skip
        elif denoise_method == 'dwidenoise':
            dwidenoise_window = config.workflow.dwidenoise_window
            auto_str = ''
            if dwidenoise_window == 'auto':
                # Configure the denoising window
                import numpy as np

                dwidenoise_window = closest_odd(int(np.ceil(np.cbrt(n_volumes))))
                dwidenoise_window = max(dwidenoise_window, 3)
                config.loggers.workflow.info(
                    f'Automatically using {dwidenoise_window}, {dwidenoise_window}, '
                    f'{dwidenoise_window} window for dwidenoise'
                )
                auto_str = 'n automatically-determined'

            denoiser = pe.Node(
                DWIDenoise(
                    extent=(dwidenoise_window, dwidenoise_window, dwidenoise_window),
                    nthreads=omp_nthreads,
                ),
                name='denoiser',
                n_procs=omp_nthreads,
            )
        elif denoise_method == 'svht':
            # svht_denoise sizes its patches from the volume count unless --denoise-config
            # sets "extent"; --dwidenoise-window is for dwidenoise alone
            svht_kwargs = dict(svht_params)
            if denoise_real_axis:
                # PhaseToRad (below) supplies the phase in radians
                svht_kwargs['phase_units'] = 'radians'

            denoiser = pe.Node(
                SVHTDenoise(nthreads=omp_nthreads, **svht_kwargs),
                name='denoiser',
                n_procs=omp_nthreads,
            )
            if svht_params['demean']:
                # The shells to demean are read from the b-values
                workflow.connect([(inputnode, denoiser, [('bval_file', 'bval_file')])])
        else:
            denoiser = pe.Node(
                Patch2Self(),
                name='denoiser',
                n_procs=omp_nthreads,
            )
            workflow.connect([(inputnode, denoiser, [('bval_file', 'bval_file')])])

        if denoise_method.startswith('dwidenoise'):
            if denoise_method == 'dwidenoise2':
                # dwidenoise2 turns on a number of methods by default, each with its own
                # citation, so the description is compiled from the parameters in effect
                mppca_desc = describe_dwidenoise2(dwidenoise2_params, complex_data=denoise_complex)
            else:
                mppca_desc = (
                    'denoised using the Marchenko-Pastur PCA method implemented in dwidenoise '
                    '[@mrtrix3; @dwidenoise1; @dwidenoise2] '
                    f'with a{auto_str} window size of {dwidenoise_window} voxels. '
                )

            if denoise_complex:
                desc += (
                    'Magnitude and phase DWI data were combined into a complex-valued file, then '
                    f'{mppca_desc}'
                )
                if not unring_complex:
                    desc += (
                        'After denoising, the complex-valued data were split back into '
                        'magnitude and phase, and the denoised magnitude data were retained. '
                    )
            else:
                desc += f'DWI data were {mppca_desc}'

            last_step = 'After MP-PCA, '
        elif denoise_method == 'svht':
            desc += describe_svht(svht_params, real_axis=denoise_real_axis)
            last_step = 'After denoising, '
        else:
            desc += (
                "DWI data were denoised using DiPy's Patch2Self algorithm [@dipy; @patch2self] "
                'with an automatically-defined window size. '
            )
            last_step = 'After `patch2self`, '

        # Wiring that is the same for every denoising method
        workflow.connect([
            (denoiser, ds_report_denoising, [('out_report', 'in_file')]),
            (denoiser, merge_confounds, [('nmse_text', f'in{step_num}')]),
            # The noise image is a derivative, so it always comes straight from the denoiser
            (denoiser, outputnode, [('noise_image', 'noise_image')]),
        ])  # fmt:skip

        # The complex-valued path only changes what feeds the denoiser; the denoiser
        # wiring itself is the same either way.
        if denoise_complex or denoise_real_axis:
            phase_to_radians = pe.Node(
                PhaseToRad(),
                name='phase_to_radians',
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, phase_to_radians, [('dwi_phase_file', 'phase_file')]),
            ])  # fmt:skip
        if denoise_complex:
            combine_complex = pe.Node(
                PolarToComplex(),
                name='combine_complex',
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
            ])  # fmt:skip
            chain.feed(combine_complex, 'mag_file')
            chain.advance(combine_complex, 'out_file', is_complex=True)
        elif denoise_real_axis:
            # svht_denoise takes the magnitude and phase separately
            workflow.connect([
                (phase_to_radians, denoiser, [('phase_file', 'phase_file')]),
            ])  # fmt:skip

        chain.feed(denoiser, 'in_file')
        chain.advance(denoiser, 'out_file', is_complex=denoise_complex or denoise_real_axis)
        # Hold the complex data if unringing can use them; otherwise split here
        if not unring_complex:
            chain.to_magnitude()

        step_num += 1

    if do_unringing:
        if unringing_method == 'mrdegibbs':
            if unring_complex:
                desc += (
                    f'{last_step}Gibbs ringing was removed from the complex-valued data using '
                    'MRtrix3 [@mrtrix3; @mrdegibbs]. The complex-valued data were then split '
                    'back into magnitude and phase, and the magnitude data were retained. '
                )
            else:
                desc += (
                    f'{last_step}Gibbs ringing was removed from the magnitude data using '
                    'MRtrix3 [@mrtrix3; @mrdegibbs]. '
                )
            degibbser = pe.Node(
                MRDeGibbs(nthreads=omp_nthreads),
                name='degibbser',
                n_procs=omp_nthreads,
            )
        elif unringing_method == 'rpg':
            desc += f'{last_step}Gibbs ringing was removed using TORTOISE [@pfgibbs]. '

            pe_code = {
                'i': 0,
                'i-': 0,
                'j': 1,
                'j-': 1,
                'k': 2,
            }.get(phase_encoding_direction)
            if pe_code is None:
                raise Exception('rpg requires an i[-],j[-] or k[-] PhaseEncodingDirection')

            degibbser = pe.Node(
                Gibbs(
                    kspace_coverage=partial_fourier,
                    phase_encoding_dir=pe_code,
                    num_threads=omp_nthreads,
                ),
                name='degibbser',
                n_procs=omp_nthreads,
            )
        elif unringing_method == 'svht':
            svht_pf = _svht_partial_fourier(partial_fourier, phase_encoding_direction)
            svht_kwargs = {}
            pf_desc = ''
            if svht_pf is not None:
                svht_kwargs['partial_fourier'] = svht_pf
                pf_desc = (
                    f', together with the ringing induced by the partial Fourier ({svht_pf}) '
                    'acquisition [@pfgibbs]'
                )
            desc += (
                f'{last_step}Gibbs ringing was removed from the magnitude data by local '
                f'subvoxel shifts [@mrdegibbs]{pf_desc}, as implemented in `svht_denoise` '
                '[@svht_denoise]. '
            )
            degibbser = pe.Node(
                SVHTDeGibbs(nthreads=omp_nthreads, **svht_kwargs),
                name='degibbser',
                n_procs=omp_nthreads,
            )

        last_step = 'After unringing, '

        ds_report_unringing = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='unringing',
                extension='.svg',
                source_file=source_file,
            ),
            name=f'ds_report_{name}_unringing',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (degibbser, ds_report_unringing, [('out_report', 'in_file')]),
            (degibbser, merge_confounds, [('nmse_text', f'in{step_num}')]),
        ])  # fmt:skip
        chain.feed(degibbser, 'in_file')
        chain.advance(degibbser, 'out_file', is_complex=unring_complex)
        step_num += 1

    # The workflow always hands downstream steps magnitude data
    chain.to_magnitude()
    chain.feed(outputnode, 'dwi_file')

    if not last_step:
        desc = 'No denoising steps were applied to the DWI data.'

    # If any denoising operations were run, collect their confounds
    if step_num > 1:
        hstack_confounds = pe.Node(StackConfounds(axis=1), name='hstack_confounds')
        workflow.connect([
            (merge_confounds, hstack_confounds, [('out', 'in_files')]),
            (hstack_confounds, outputnode, [('confounds_file', 'confounds')]),
        ])  # fmt:skip

    workflow.__desc__ = desc

    return workflow


# The partial Fourier factors svht_denoise implements; it refuses any other
_SVHT_PARTIAL_FOURIER = (0.875, 0.75)


def _svht_partial_fourier(partial_fourier, phase_encoding_direction):
    """Resolve the ``-pF`` factor for ``svht_denoise`` unringing.

    Returns ``None`` for full k-space, including when the factor is not recorded.
    ``svht_denoise`` implements only 7/8 and 6/8, and only along the second voxel
    axis, so anything else raises rather than silently leaving the ringing in place.
    """
    if partial_fourier is None or pd.isna(partial_fourier) or partial_fourier > 0.99:
        return None

    for factor in _SVHT_PARTIAL_FOURIER:
        if abs(partial_fourier - factor) < 0.01:
            break
    else:
        raise ValueError(
            '--unringing-method svht supports partial Fourier factors of 7/8 (0.875) and '
            f'6/8 (0.75), but this series has a PartialFourier of {partial_fourier}. '
            'Use --unringing-method rpg instead.'
        )

    if phase_encoding_direction not in ('j', 'j-'):
        raise ValueError(
            '--unringing-method svht corrects partial Fourier ringing only along the j '
            f'axis, but this series is phase-encoded along {phase_encoding_direction}. '
            'Use --unringing-method rpg instead.'
        )

    return factor


def _as_list(item):
    return [item]


def gen_denoising_boilerplate(do_biascorr):
    """Generate a methods boilerplate for the denoising workflow.

    ``do_biascorr`` is the resolved decision for this output, not the
    ``--dwi-biascorrect`` mode: under ``auto`` the mode alone cannot say whether
    N4 actually ran, so reading the config here would state the wrong thing.
    """
    no_b0_harmonization = config.workflow.no_b0_harmonization
    b0_threshold = config.workflow.b0_threshold
    desc = [
        f'Any images with a b-value less than {b0_threshold} s/mm^2 were treated as a *b*=0 image.'
    ]
    harmonize_b0s = not no_b0_harmonization
    last_step = ''

    if harmonize_b0s:
        desc.append(
            'The mean intensity of the DWI series was adjusted '
            'so all the mean intensity of the b=0 images matched across each'
            'separate DWI scanning sequence.'
        )
        last_step = True

    if do_biascorr:
        desc.append(
            'B1 field inhomogeneity was corrected using '
            '`dwibiascorrect` from MRtrix3 with the N4 algorithm '
            '[@n4] after corrected images were resampled.'
        )
        last_step = True

    if not last_step:
        return 'No denoising steps were applied to the DWI data.'

    return ' '.join(desc)


def get_acq_parameters_df(dwi_file_list, metadata_lookup):
    """Tabulate each file's acquisition parameters.

    ``metadata_lookup`` maps a path to its sidecar metadata (usually
    ``PreprocUnit.metadata_for``, so nothing is re-read from disk); the NIfTI
    header still supplies image-derived fields like the volume count.
    """
    file_rows = []
    for dwi_file in dwi_file_list:
        metadata = dict(metadata_lookup(dwi_file))
        update_metadata_from_nifti_header(metadata, dwi_file)
        metadata['BIDSFile'] = dwi_file
        file_rows.append(metadata)

    merged_acq_params = pd.DataFrame(
        file_rows,
        columns=['BIDSFile'] + IMPORTANT_DWI_FIELDS,
    )
    merged_acq_params['PhaseEncodingAxis'] = merged_acq_params[
        'PhaseEncodingDirection'
    ].str.replace('-', '')
    return merged_acq_params


def get_merged_parameter(parameter_df, parameter_name, selection_mode='all'):
    """Return a single parameter from a parameter dataframe."""
    col = parameter_df[parameter_name]
    unique_values = col.unique()
    if len(unique_values) > 1:
        config.loggers.workflow.warn(
            'Found %d unique values for %s',
            parameter_name,
            len(unique_values),
        )

    # Require that all the values are the same
    if selection_mode == 'all':
        if len(unique_values) > 1:
            raise Exception(
                'More than one value for %s was found (%s): exiting!',
                parameter_name,
                str(unique_values),
            )

        return unique_values[0]

    if selection_mode == 'mode':
        return col.mode()[0]

    raise Exception("selection_mode must be 'all' or 'mode'")


def closest_odd(x):
    if x % 2 == 0:
        return x + 1
    else:
        return x
