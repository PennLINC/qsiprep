# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import pandas as pd
from bids.layout import Query
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from qsiplan.models import strip_nii_ext

from ... import config
from ...interfaces import ConformDwi, DerivativesDataSink
from ...interfaces.dipy import Patch2Self
from ...interfaces.dwi_merge import MergeDWIs, PhaseToRad, StackConfounds
from ...interfaces.gradients import ExtractB0s
from ...interfaces.mrtrix import (
    ComplexToMagnitude,
    DWIBiasCorrect,
    DWIDenoise,
    DWIDenoise2,
    MRDeGibbs,
    MRTrixGradientTable,
    PolarToComplex,
)
from ...interfaces.nilearn import MaskEPI, Merge
from ...interfaces.tortoise import Gibbs
from ...utils.bids import IMPORTANT_DWI_FIELDS, update_metadata_from_nifti_header
from ...utils.misc import describe_dwidenoise2, parse_denoise_method
from .qc import init_modelfree_qc_wf
from .util import _get_wf_name

DEFAULT_MEMORY_MIN_GB = 0.01


def init_merge_and_denoise_wf(
    unit,
    raw_dwi_files,
    orientation,
    source_file,
    do_biascorr,
    calculate_qc=False,
    phase_id='same',
    name='merge_and_denoise_wf',
):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_merge_and_denoise_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        wf = init_merge_and_denoise_wf(
            make_preproc_unit(['/path/to/dwi/sub-1_dwi.nii.gz']),
            ['/path/to/dwi/sub-1_dwi.nii.gz'],
            source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        the unit these series belong to; sidecar metadata comes from its
        records instead of layout re-reads (the layout is still probed for
        part-phase companion files, which the grouping does not model)
    raw_dwi_files : list
        list of raw (in their original BIDS directory) dwi nifti files


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
    noise_image
        image(s) created by ``dwidenoise``
    original_files
        names of the original files for each volume
    qc_summary
        DSI Studio QC text file

    """
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads
    denoise_before_combining = not config.workflow.denoise_after_combining
    layout = config.execution.layout
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'merged_image',
                'merged_raw_image',
                'merged_bval',
                'merged_bvec',
                'merged_json',
                'noise_images',
                'bias_images',
                'denoising_confounds',
                'original_files',
                'qc_summary',
                'validation_reports',
            ]
        ),
        name='outputnode',
    )
    desc = []

    # DWIs will be merged at some point.
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
    # Create a denoising workflow for each input image
    num_dwis = len(raw_dwi_files)
    if num_dwis > 1:
        if denoise_before_combining:
            order = 'on individual DWI series before concatenation'
        else:
            order = 'on the concatenated DWI series'
        desc.append(
            'A total of %d DWI series in the %s distortion group were '
            'concatenated, with preprocessing operations performed %s.'
            % (num_dwis, phase_id, order)
        )
    workflow.__desc__ = ' '.join(desc)
    conformed_bvals = pe.Node(niu.Merge(num_dwis), name='conformed_bvals')
    conformed_bvecs = pe.Node(niu.Merge(num_dwis), name='conformed_bvecs')
    conformed_images = pe.Node(niu.Merge(num_dwis), name='conformed_images')
    conformed_raw_images = pe.Node(niu.Merge(num_dwis), name='conformed_raw_images')
    conformation_reports = pe.Node(niu.Merge(num_dwis), name='conformation_reports')
    # derivatives from denoising
    denoising_confounds = pe.Node(niu.Merge(num_dwis), name='denoising_confounds')
    noise_images = pe.Node(niu.Merge(num_dwis), name='noise_images')
    bias_images = pe.Node(niu.Merge(num_dwis), name='bias_images')
    # Collect conformers and denoisers
    conformers = []
    denoising_wfs = []

    # Get a data frame of the raw_dwi_files and their imaging parameters:
    dwi_df = get_acq_parameters_df(raw_dwi_files, metadata_lookup=unit.metadata_for)
    for i_dwi, row in dwi_df.iterrows():
        dwi_num = i_dwi + 1  # start at 1
        dwi_file = row.BIDSFile

        # Conform each image to the requested orientation
        conformers.append(
            pe.Node(ConformDwi(orientation=orientation), name=f'conform_dwis{dwi_num:02d}'),
        )
        conformers[-1].inputs.dwi_file = dwi_file

        if denoise_before_combining:
            # Build the denoising workflow
            _, fname, _ = split_filename(dwi_file)
            wf_name = _get_wf_name(fname).replace('preproc', 'denoise')

            # Set up a strict query for a phase file based on the magnitude file.
            all_entities = layout.get_entities(metadata=False)
            # No other non-matching entities allowed
            query = dict.fromkeys(all_entities.keys(), Query.NONE)
            query.update(layout.get_file(dwi_file).get_entities())
            query['part'] = 'phase'
            phase_files = layout.get(**query)
            phase_available = False
            if len(phase_files) == 1:
                phase_available = True
                config.loggers.workflow.info('Phase file found for %s', dwi_file)
                phase_file = phase_files[0].path

            use_phase = phase_available and 'phase' not in config.workflow.ignore
            if use_phase:
                conform_phase = pe.Node(
                    ConformDwi(
                        orientation=orientation,
                        dwi_file=phase_file,
                    ),
                    name=f'conform_phase{dwi_num}',
                )

            n_volumes = row.NumVolumes
            denoising_wfs.append(
                init_dwi_denoising_wf(
                    partial_fourier=row.PartialFourier,
                    phase_encoding_direction=row.PhaseEncodingAxis,
                    source_file=dwi_file,
                    n_volumes=n_volumes,
                    use_phase=use_phase,
                    do_biascorr=do_biascorr,
                    name=wf_name,
                ),
            )
            workflow.connect([
                (conformers[-1], denoising_wfs[-1], [
                    ('bval_file', 'inputnode.bval_file'),
                    ('bvec_file', 'inputnode.bvec_file'),
                    ('dwi_file', 'inputnode.dwi_file'),
                ]),
                (denoising_wfs[-1], denoising_confounds, [
                    ('outputnode.confounds', f'in{dwi_num}'),
                ]),
                (denoising_wfs[-1], noise_images, [('outputnode.noise_image', f'in{dwi_num}')]),
                (denoising_wfs[-1], bias_images, [('outputnode.bias_image', f'in{dwi_num}')]),
            ])  # fmt:skip

            if use_phase:
                workflow.connect([
                    (conform_phase, denoising_wfs[-1], [('dwi_file', 'inputnode.dwi_phase_file')]),
                ])  # fmt:skip

            dwi_source = denoising_wfs[-1]
            edge_prefix = 'outputnode.'
        else:
            dwi_source = conformers[-1]
            edge_prefix = ''

        workflow.connect([
            (dwi_source, conformed_images, [(f'{edge_prefix}dwi_file', f'in{dwi_num}')]),
            (conformers[-1], conformed_raw_images, [('dwi_file', f'in{dwi_num}')]),
            (dwi_source, conformed_bvals, [(f'{edge_prefix}bval_file', f'in{dwi_num}')]),
            (dwi_source, conformed_bvecs, [(f'{edge_prefix}bvec_file', f'in{dwi_num}')]),
            (conformers[-1], conformation_reports, [('out_report', f'in{dwi_num}')]),
        ])  # fmt:skip

    # Get an orientation-conformed version of the raw inputs and their gradients
    raw_merge = pe.Node(Merge(is_dwi=True), name='raw_merge', n_procs=omp_nthreads)

    # Merge the either conformed-only or conformed-and-denoised data
    workflow.connect([
        (conformed_images, merge_dwis, [('out', 'dwi_files')]),
        (conformed_raw_images, raw_merge, [('out', 'in_files')]),
        (raw_merge, outputnode, [('out_file', 'merged_raw_image')]),
        (conformed_bvals, merge_dwis, [('out', 'bval_files')]),
        (conformed_bvecs, merge_dwis, [('out', 'bvec_files')]),
        (conformation_reports, outputnode, [('out', 'validation_reports')]),
        (merge_dwis, outputnode, [
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

    # We have denoised and combined, therefore we are done
    if denoise_before_combining:
        workflow.connect([
            (denoising_confounds, merge_dwis, [('out', 'denoising_confounds')]),
            (merge_dwis, outputnode, [
                ('out_dwi', 'merged_image'),
                ('merged_denoising_confounds', 'denoising_confounds'),
            ]),
            (noise_images, outputnode, [('out', 'noise_images')]),
            (bias_images, outputnode, [('out', 'bias_images')]),
        ])  # fmt:skip

        return workflow

    # Send the merged series for denoising
    merge_confounds = pe.Node(niu.Merge(2), name='merge_confounds')
    hstack_confounds = pe.Node(StackConfounds(axis=1), name='hstack_confounds')
    n_volumes = dwi_df['NumVolumes'].sum()
    denoising_wf = init_dwi_denoising_wf(
        partial_fourier=get_merged_parameter(dwi_df, 'PartialFourier', 'all'),
        phase_encoding_direction=get_merged_parameter(dwi_df, 'PhaseEncodingAxis', 'all'),
        source_file=source_file,
        n_volumes=n_volumes,
        use_phase=False,  # can't use phase with concatenated data
        do_biascorr=do_biascorr,
        name='merged_denoise',
    )

    workflow.connect([
        (merge_dwis, denoising_wf, [
            ('out_bval', 'inputnode.bval_file'),
            ('out_dwi', 'inputnode.dwi_file'),
            ('out_bvec', 'inputnode.bvec_file'),
        ]),
        (merge_dwis, merge_confounds, [('merged_denoising_confounds', 'in1')]),
        (denoising_wf, merge_confounds, [('outputnode.confounds', 'in2')]),
        (merge_confounds, hstack_confounds, [('out', 'in_files')]),
        (hstack_confounds, outputnode, [('confounds_file', 'denoising_confounds')]),
        (denoising_wf, outputnode, [
            ('outputnode.dwi_file', 'merged_image'),
            (('outputnode.noise_image', _as_list), 'noise_images'),
            (('outputnode.bias_image', _as_list), 'bias_images'),
        ]),
    ])  # fmt:skip

    return workflow


class _ImageChain:
    """Track the image flowing through the denoising steps, and its data domain.

    The denoising steps form a linear chain, each reading the previous step's output.
    Whether that image is complex-valued is not a property of the chain position but of
    which steps have run: ``dwidenoise`` handed complex data emits complex data, and so
    does ``mrdegibbs`` on MRtrix3's development branch, while ``dwibiascorrect`` and
    TORTOISE's ``rpg`` are magnitude-only. Tracking the domain alongside the current
    image lets :func:`to_magnitude` insert the split at the last step that can still
    consume complex data, instead of always splitting right after denoising.
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
    do_biascorr,
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
        If True, and ``denoise_method`` is ``dwidenoise``, then ``dwidenoise``
        will be run on the complex-valued data.
    do_biascorr : bool
        If True run dwi_biascorrect
    name : str
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
    bias_image
        path to the bias image
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
                'bias_image',
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
    denoise_method, dwidenoise2_params = parse_denoise_method(
        config.workflow.denoise_method,
        use_phase=use_phase,
    )

    unringing_method = config.workflow.unringing_method
    do_denoise = denoise_method in ('patch2self', 'dwidenoise', 'dwidenoise2')
    do_unringing = config.workflow.unringing_method in ('mrdegibbs', 'rpg')
    harmonize_b0s = not config.workflow.no_b0_harmonization

    # Only the dwidenoise variants denoise complex data; the rest use magnitude alone.
    denoise_complex = do_denoise and denoise_method.startswith('dwidenoise') and use_phase
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
    num_steps = sum(map(int, [do_denoise, do_unringing, do_biascorr, harmonize_b0s]))
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
            # so there is no kernel to configure and dwi_denoise_window does not apply here.
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
            dwi_denoise_window = config.workflow.dwi_denoise_window
            auto_str = ''
            if dwi_denoise_window == 'auto':
                # Configure the denoising window
                import numpy as np

                dwi_denoise_window = closest_odd(int(np.ceil(np.cbrt(n_volumes))))
                dwi_denoise_window = max(dwi_denoise_window, 3)
                config.loggers.workflow.info(
                    f'Automatically using {dwi_denoise_window}, {dwi_denoise_window}, '
                    f'{dwi_denoise_window} window for dwidenoise'
                )
                auto_str = 'n automatically-determined'

            denoiser = pe.Node(
                DWIDenoise(
                    extent=(dwi_denoise_window, dwi_denoise_window, dwi_denoise_window),
                    nthreads=omp_nthreads,
                ),
                name='denoiser',
                n_procs=omp_nthreads,
            )
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
                    f'with a{auto_str} window size of {dwi_denoise_window} voxels. '
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
        if denoise_complex:
            phase_to_radians = pe.Node(
                PhaseToRad(),
                name='phase_to_radians',
                n_procs=omp_nthreads,
            )
            combine_complex = pe.Node(
                PolarToComplex(),
                name='combine_complex',
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, phase_to_radians, [('dwi_phase_file', 'phase_file')]),
                (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
            ])  # fmt:skip
            chain.feed(combine_complex, 'mag_file')
            chain.advance(combine_complex, 'out_file', is_complex=True)

        chain.feed(denoiser, 'in_file')
        chain.advance(denoiser, 'out_file', is_complex=denoise_complex)
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

    if do_biascorr:
        desc += (
            f'{last_step}B1 field inhomogeneity was corrected using '
            '`dwibiascorrect` from MRtrix3 with the N4 algorithm [@n4]. '
        )
        last_step = True

        biascorr = pe.Node(
            DWIBiasCorrect(method='ants', mrtrix_version=config.workflow.mrtrix_version),
            name='biascorr',
            n_procs=omp_nthreads,
        )
        ds_report_biascorr = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='biascorr',
                source_file=source_file,
            ),
            name=f'ds_report_{name}_biascorr',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        get_b0s = pe.Node(ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='get_b0s')
        quick_mask = pe.Node(MaskEPI(lower_cutoff=0.02), name='quick_mask')

        # dwibiascorrect is magnitude-only, and so is the mask built from its input
        chain.to_magnitude()
        chain.feed(biascorr, 'in_file')
        chain.feed(get_b0s, 'dwi_series')
        workflow.connect([
            (inputnode, get_b0s, [('bval_file', 'bval_file')]),
            (get_b0s, quick_mask, [('b0_series', 'in_files')]),
            (quick_mask, biascorr, [('out_mask', 'mask')]),
            (biascorr, outputnode, [('bias_image', 'bias_image')]),
            (biascorr, ds_report_biascorr, [('out_report', 'in_file')]),
            (biascorr, merge_confounds, [('nmse_text', f'in{step_num}')]),
            (inputnode, biascorr, [
                ('bval_file', 'in_bval'),
                ('bvec_file', 'in_bvec'),
            ]),
        ])  # fmt:skip
        chain.advance(biascorr, 'out_file')
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


def _as_list(item):
    return [item]


def gen_denoising_boilerplate():
    """Generate a methods boilerplate for the denoising workflow."""

    b1_biascorrect_stage = config.workflow.b1_biascorrect_stage
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

    if b1_biascorrect_stage == 'final':
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
