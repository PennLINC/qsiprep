"""
HMC + SDC backend that wraps TORTOISEV4 DIFFPREP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_diffprep_hmc_wf

DIFFPREP fits a signal model over arbitrary q-space and corrects **head motion
and eddy currents on shelled and non-shelled data** (where FSL eddy cannot run).
Susceptibility distortion correction is performed with **TORTOISE-native** tools
where possible:

* reverse-PE (``epi`` / ``rpe_series``) -> DRBUDDI (:func:`init_drbuddi_wf`)
* GRE / phase fieldmaps -> qsiprep's fieldmap machinery (:func:`init_sdc_wf`)
* fieldmap-less with a T2w -> TORTOISE ``--epi T2Wreg`` (baked in), else SyN
* nothing -> HMC only

This mirrors the SDC coverage of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf`.
"""

import json
from importlib.resources import files

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradients import ExtractB0s, SliceQC
from ...interfaces.nilearn import EnhanceB0
from ...interfaces.shoreline import CalculateCNR
from ...interfaces.tortoise import (
    DIFFPREP,
    ConcatenateDIFFPREPGroups,
    DIFFPREPMotionParams,
    DIFFPREPSplitOutputs,
    SplitDWIsByDistortionGroup,
    SynthesizeDWIs,
    TORTOISEConvert,
    generate_diffprep_boilerplate,
)
from ...utils.gpu import gpu_enabled
from ...utils.resources import as_path
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf
from ..fieldmap.synb0 import init_synb0_wf
from .util import init_dwi_reference_wf, tortoise_convert_mem_gb

# BIDS PhaseEncodingDirection axes already match what TORTOISEProcess expects
# in its own JSON input -- TORTOISE consumes "i", "j", "k" (with optional "-")
# straight from the BIDS sidecar.
_VALID_PE = {'i', 'i-', 'j', 'j-', 'k', 'k-'}


def _listify(value):
    """Wrap a single transform path in a list for MapNode/MultiObject inputs."""
    return [value]


def _load_diffprep_config(config_path):
    """Load a --diffprep-config JSON, or return defaults."""
    if config_path is None:
        config_path = as_path(files('qsiprep.data') / 'diffprep_params.json')
    with open(config_path) as fobj:
        cfg = json.load(fobj)
    cfg.setdefault('b0_id', -1)
    cfg.setdefault('is_human_brain', True)
    cfg.setdefault('rot_eddy_center', 'isocenter')
    cfg.setdefault('extra_args', [])
    # --hmc-model exposes a single "tortoise" value, so this is the only way to
    # reach DIFFPREP's rigid-only ('motion') or 'cubic' eddy modes.
    cfg.setdefault('correction_mode', 'quadratic')
    # No default for "use_cuda": its absence must stay observable so a shipped
    # default is never mistaken for user intent (see _legacy_use_cuda below).
    # Opt-in MAPMRI shell synthesis for DRBUDDI's registration target;
    # None/0 = off.
    cfg.setdefault('drbuddi_synth_shell_bval', None)
    cfg.setdefault('drbuddi_synth_shell_ndirs', 30)
    return cfg


def _resolve_phase_encoding(pe_dir):
    """Validate a BIDS PhaseEncodingDirection value, falling back to 'j'."""
    if pe_dir in _VALID_PE:
        return pe_dir
    config.loggers.workflow.warning(
        'PhaseEncodingDirection %r is missing or unrecognized; assuming "j" '
        '(anterior-posterior) for DIFFPREP.',
        pe_dir,
    )
    return 'j'


def _write_sidecar_json(nii_file, phase_encoding_direction, working_dir=None):
    """Function-node helper. Writes a BIDS-style JSON sidecar with the same
    basename as ``nii_file`` so TORTOISEProcess can read PhaseEncodingDirection
    from it.

    The sidecar goes into this node's own cwd, not next to ``nii_file``; only
    the basename is reused. Keeping the output in the node's directory lets the
    ``copyfile=True`` propagation to the ``diffprep`` node stage a valid sidecar
    even after the upstream node's cache is cleared.
    """
    import json
    import os
    import os.path as op

    base = op.basename(nii_file)
    if base.endswith('.nii.gz'):
        base = base[: -len('.nii.gz')]
    elif base.endswith('.nii'):
        base = base[: -len('.nii')]
    out_dir = working_dir if working_dir else os.getcwd()
    json_file = op.abspath(op.join(out_dir, base + '.json'))
    with open(json_file, 'w') as fobj:
        json.dump({'PhaseEncodingDirection': phase_encoding_direction}, fobj)
    return json_file


def _write_pe_json_node(name):
    """A function-node that writes a PhaseEncodingDirection sidecar for TORTOISE."""
    return pe.Node(
        niu.Function(
            input_names=['nii_file', 'phase_encoding_direction', 'working_dir'],
            output_names=['json_file'],
            function=_write_sidecar_json,
        ),
        name=name,
    )


def _build_rpe_diffprep_stage(
    workflow, inputnode, diffprep_kwargs, pe_axis, b0_threshold, n_procs, mem_gb, sidecars
):
    """Wire the per-phase-encoding-direction DIFFPREP stage for ``rpe_series``.

    Mirrors TORTOISE's own ``for(int PE=0;PE<2;PE++)`` loop: the already-merged
    (denoised + b0-harmonized) series is re-split into its two PE groups, each
    single-PE group is corrected by its own DIFFPREP run, and the two corrected
    outputs are recombined in the original volume order. The returned node
    exposes the same ``corrected_dwi_file`` / ``corrected_bmtxt_file`` /
    ``transformations_file`` triple as a single DIFFPREP node, so everything
    downstream (split, QC, CNR, motion params, DRBUDDI) is identical.

    Returns
    -------
    nipype.pipeline.engine.Node
        The ``ConcatenateDIFFPREPGroups`` node providing the recombined triple.
    """
    split_groups = pe.Node(
        SplitDWIsByDistortionGroup(pe_axis=pe_axis, b0_threshold=b0_threshold, sidecars=sidecars),
        name='split_rpe_groups',
        mem_gb=mem_gb,
    )
    workflow.connect([
        (inputnode, split_groups, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('original_files', 'original_files'),
        ]),
    ])  # fmt:skip

    recombine = pe.Node(ConcatenateDIFFPREPGroups(), name='recombine_rpe_groups', mem_gb=mem_gb)
    workflow.connect([
        (split_groups, recombine, [('group_assignments', 'group_assignments')]),
    ])  # fmt:skip

    for group_id in (1, 2):
        convert = pe.Node(TORTOISEConvert(), name=f'tortoise_convert_g{group_id}', mem_gb=mem_gb)
        write_json = _write_pe_json_node(f'write_pe_json_g{group_id}')
        diffprep = pe.Node(
            DIFFPREP(**diffprep_kwargs),
            name=f'diffprep_g{group_id}',
            n_procs=n_procs,
        )
        workflow.connect([
            (split_groups, convert, [
                (f'group{group_id}_dwi_file', 'dwi_file'),
                (f'group{group_id}_bval_file', 'bval_file'),
                (f'group{group_id}_bvec_file', 'bvec_file'),
            ]),
            (convert, write_json, [('dwi_file', 'nii_file')]),
            (split_groups, write_json, [(f'group{group_id}_pe_dir', 'phase_encoding_direction')]),
            (convert, diffprep, [
                ('dwi_file', 'dwi_file'),
                ('bmtxt_file', 'bmtxt_file'),
            ]),
            (write_json, diffprep, [('json_file', 'json_file')]),
            (diffprep, recombine, [
                ('corrected_dwi_file', f'group{group_id}_dwi_file'),
                ('corrected_bmtxt_file', f'group{group_id}_bmtxt_file'),
                ('transformations_file', f'group{group_id}_transformations_file'),
            ]),
        ])  # fmt:skip

    return recombine


def init_diffprep_hmc_wf(
    unit,
    source_file,
    t2w_sdc,
    name='diffprep_hmc_wf',
):
    """HMC + SDC workflow that uses TORTOISEV4 DIFFPREP for motion + eddy
    correction, and TORTOISE-native SDC (DRBUDDI / T2Wreg) or qsiprep's own
    fieldmap machinery.

    Drop-in peer of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf` with an
    identical inputnode/outputnode contract. The TORTOISE binary writes the
    corrected DWI directly, so this workflow follows the same "bake the
    correction in, emit identity per-volume affines" pattern that
    ``init_fsl_hmc_wf`` uses for FSL eddy.

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The DWI series to correct together and the fieldmap that corrects them.
    source_file : str
        Path to the source DWI file (used for report naming).
    t2w_sdc : bool
        Whether a T2w image is available for distortion correction (used for
        DRBUDDI's multi-modal registration and for the fieldmap-less T2Wreg
        path).
    name : str
        Workflow name.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'original_files',
                't1_preproc',
                't1_brain',
                't1_mask',
                't1_seg',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
                'to_template_affine_transform',
                'acpc_inv_transform',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_template',
                'b0_template_mask',
                'pre_sdc_template',
                'hmc_optimization_data',
                'sdc_method',
                'slice_quality',
                'motion_params',
                'ec_file',
                'cnr_map',
                'bvec_files_to_transform',
                'dwi_files_to_transform',
                'b0_indices',
                'bval_files',
                'to_dwi_ref_affines',
                'to_dwi_ref_warps',
                'sdc_scaling_images',
                'fieldmap_type',
                'b0_up_image',
                'b0_up_corrected_image',
                'b0_down_image',
                'b0_down_corrected_image',
                'up_fa_image',
                'up_fa_corrected_image',
                'down_fa_image',
                'down_fa_corrected_image',
                't2w_image',
            ],
        ),
        name='outputnode',
    )

    # Several nodes below hold the whole series as float32; size their memory
    # from the data rather than guessing.
    series_mem_gb = tortoise_convert_mem_gb(list(unit.dwi_files))

    # TORTOISE-native T2Wreg replaces SyN for the fieldmap-less case when a
    # structural target is available: DIFFPREP runs the ``--epi T2Wreg`` stage
    # in the same TORTOISEProcess call and bakes the correction into its
    # output. The plan names the target: 'synb0' (a synthetic distortion-free
    # b=0 generated from the T1w, preferred when --use-synb0 was given) or
    # 't2w' (the subject's T2w, additionally gated by ``t2w_sdc``).
    is_fieldmapless = not unit.has_scanner_measured_fieldmap
    t2wreg_stage = unit.run.stage_with('t2wreg')
    synb0_target = t2wreg_stage is not None and t2wreg_stage.structural_target == 'synb0'
    # Classic SyN is fieldmap-less too, but has its own path (init_sdc_wf below).
    use_t2wreg = is_fieldmapless and not unit.is_nipreps_syn and (synb0_target or bool(t2w_sdc))
    epi_mode = 'T2Wreg' if use_t2wreg else 'off'

    # Load any user-supplied DIFFPREP config (or our defaults)
    diffprep_cfg = _load_diffprep_config(config.workflow.diffprep_config)
    # "use_cuda" counts as user intent only when the user's own config file sets
    # it -- the shipped default must not "conflict" with --gpu on every run.
    _legacy_use_cuda = diffprep_cfg.get('use_cuda') if config.workflow.diffprep_config else None
    diffprep_gpu = gpu_enabled('diffprep', config_file_value=_legacy_use_cuda)
    drbuddi_gpu = gpu_enabled('drbuddi', config_file_value=_legacy_use_cuda)

    synth_shell_bval = diffprep_cfg.get('drbuddi_synth_shell_bval')

    pe_dir = _resolve_phase_encoding(unit.pe_dir or None)

    correction_mode = diffprep_cfg['correction_mode']

    # --sloppy uses ONLY --niter 0 plus rigid-only correction. Clearing
    # ``is_human_brain`` would also disable the iterative pass but is not a
    # speed knob: it changes DIFFPREP's auto-masking and the T2Wreg structural
    # masking. And --niter 0 alone does not bound runtime on low-b data (the
    # always-run first pass fits a 24-parameter quadratic per volume), so the
    # correction mode is dropped to 'motion' as well.
    effective_correction_mode = correction_mode
    if config.execution.sloppy:
        effective_correction_mode = 'motion'
        sloppy_kwargs = {'niter': 0}
        config.loggers.workflow.warning(
            '--sloppy: running DIFFPREP rigid-only (-c motion, --niter 0) instead '
            'of %s. Eddy-current correction is DISABLED; this is for smoke-testing '
            'the pipeline, not for real data.',
            correction_mode,
        )
    else:
        sloppy_kwargs = {}

    # Described after the --sloppy downgrade so the methods section reports the
    # correction that actually ran, not the one that was asked for.
    workflow.__desc__ = generate_diffprep_boilerplate(effective_correction_mode)

    diffprep_kwargs = dict(
        # num_threads only sets OMP_NUM_THREADS, which TORTOISEProcess ignores
        # (it sizes its thread pool from hardware concurrency and calls
        # omp_set_num_threads itself); ncores is the knob that actually bounds
        # it, so nipype's n_procs declaration and real CPU use agree.
        num_threads=config.nipype.omp_nthreads,
        ncores=config.nipype.omp_nthreads,
        # Only forwarded when explicitly set: the right value depends on the
        # GPU/CPU balance of the node, and TORTOISE's own default should stand
        # rather than qsiprep inventing one.
        **(
            {'gpu_cpu_ratio': config.workflow.tortoise_gpu_cpu_ratio}
            if config.workflow.tortoise_gpu_cpu_ratio
            else {}
        ),
        correction_mode=effective_correction_mode,
        b0_id=diffprep_cfg['b0_id'],
        is_human_brain=diffprep_cfg['is_human_brain'],
        rot_eddy_center=diffprep_cfg['rot_eddy_center'],
        extra_args=diffprep_cfg['extra_args'],
        epi_mode=epi_mode,
        use_cuda=diffprep_gpu,
        **sloppy_kwargs,
    )

    # DIFFPREP stage. A reverse-PE *series* is corrected once per phase-encoding
    # direction (DIFFPREP models a single phase axis / single b=0 reference for
    # a whole file), then recombined; every other case is a single DIFFPREP run.
    # Both expose the identical ``corrected_dwi_file`` / ``corrected_bmtxt_file``
    # / ``transformations_file`` triple via ``corrected_node``.
    if unit.has_bidirectional_dwi:
        corrected_node = _build_rpe_diffprep_stage(
            workflow,
            inputnode,
            diffprep_kwargs,
            pe_axis=pe_dir[0],
            b0_threshold=config.workflow.b0_threshold,
            n_procs=config.nipype.omp_nthreads,
            mem_gb=series_mem_gb,
            sidecars=unit.sidecar_overrides(),
        )
    else:
        # Convert gzipped niftis + FSL gradients into TORTOISE format (.nii + .bmtxt).
        tortoise_convert = pe.Node(
            TORTOISEConvert(), name='tortoise_convert', mem_gb=series_mem_gb
        )

        # TORTOISE reads PhaseEncodingDirection from a BIDS-style JSON next to the
        # .nii. Generate one so DIFFPREP (and T2Wreg) can pick the right phase axis.
        write_pe_json = _write_pe_json_node('write_pe_json')
        write_pe_json.inputs.phase_encoding_direction = pe_dir

        diffprep = pe.Node(
            DIFFPREP(**diffprep_kwargs),
            name='diffprep',
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (inputnode, tortoise_convert, [
                ('dwi_file', 'dwi_file'),
                ('bval_file', 'bval_file'),
                ('bvec_file', 'bvec_file'),
            ]),
            (tortoise_convert, write_pe_json, [('dwi_file', 'nii_file')]),
            (tortoise_convert, diffprep, [
                ('dwi_file', 'dwi_file'),
                ('bmtxt_file', 'bmtxt_file'),
            ]),
            (write_pe_json, diffprep, [('json_file', 'json_file')]),
        ])  # fmt:skip

        # T2Wreg bakes SDC into the DIFFPREP call: feed the structural target.
        if use_t2wreg and synb0_target:
            # Generate the synthetic distortion-free b=0 from the T1w and the
            # raw (distorted) b=0 reference, and register DIFFPREP's EPI stage
            # to it instead of a T2w -- its contrast matches the b=0 exactly.
            raw_b0s = pe.Node(
                ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='raw_b0s'
            )
            synb0_b0_ref_wf = init_dwi_reference_wf(
                gen_report=False,
                desc='b0_for_synb0',
                name='synb0_b0_ref_wf',
                source_file=source_file,
            )
            synb0_wf = init_synb0_wf()
            workflow.connect([
                (inputnode, raw_b0s, [
                    ('dwi_file', 'dwi_series'),
                    ('bval_file', 'bval_file'),
                ]),
                (raw_b0s, synb0_b0_ref_wf, [('b0_average', 'inputnode.b0_template')]),
                (inputnode, synb0_wf, [
                    ('t1_preproc', 'inputnode.t1_preproc'),
                    ('t1_brain', 'inputnode.t1_brain'),
                    ('t1_seg', 'inputnode.t1_seg'),
                    ('to_template_affine_transform',
                     'inputnode.to_template_affine_transform'),
                    ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
                ]),
                (synb0_b0_ref_wf, synb0_wf, [
                    ('outputnode.ref_image', 'inputnode.b0_ref'),
                    ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                ]),
                (synb0_wf, diffprep, [('outputnode.synthetic_b0', 'structural_image')]),
            ])  # fmt:skip
        elif use_t2wreg:
            workflow.connect([
                (inputnode, diffprep, [('t2w_unfatsat', 'structural_image')]),
            ])  # fmt:skip

        corrected_node = diffprep

    split_outputs = pe.Node(
        DIFFPREPSplitOutputs(b0_threshold=config.workflow.b0_threshold),
        name='split_outputs',
        mem_gb=series_mem_gb,
    )

    motion_params = pe.Node(DIFFPREPMotionParams(), name='motion_params')

    # Build a pre-SDC template from the corrected b=0 series for the report.
    extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold),
        name='extract_b0s',
    )
    enhance_pre_sdc = pe.Node(EnhanceB0(), name='enhance_pre_sdc')

    # A "true" b0 reference for downstream coregistration (matches fsl path).
    b0_ref_for_coreg = init_dwi_reference_wf(
        gen_report=False,
        desc='b0_for_coreg',
        name='b0_ref_for_coreg',
        source_file=source_file,
    )

    # Slice-wise QC for the carpet plot: fit a MAPMRI model to the corrected DWI
    # and synthesize an "ideal" volume at every corrected gradient, then score
    # observed-vs-ideal per slice with the same SliceQC node SHORELine uses.
    synth_dwis = pe.Node(
        SynthesizeDWIs(num_threads=config.nipype.omp_nthreads),
        name='synth_dwis',
        n_procs=config.nipype.omp_nthreads,
        mem_gb=series_mem_gb,
    )
    slice_qc = pe.Node(SliceQC(), name='slice_qc')

    # CNR from the same MAPMRI synthesis, using SHORELine's CalculateCNR so the
    # map means the same thing (per-voxel var(predicted) / var(predicted -
    # observed)) as it does for the other backends. DIFFPREP emits no CNR of its
    # own, and the map is a required downstream ApplyTransforms input.
    calculate_cnr = pe.Node(CalculateCNR(), name='calculate_cnr', mem_gb=2)

    workflow.connect([
        (corrected_node, split_outputs, [
            ('corrected_dwi_file', 'corrected_dwi_file'),
            ('corrected_bmtxt_file', 'corrected_bmtxt_file'),
        ]),
        (corrected_node, motion_params, [('transformations_file', 'transformations_file')]),

        # Outputnode plumbing (per-volume corrected data + identity affines)
        (split_outputs, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices'),
            ('forward_transforms', 'to_dwi_ref_affines'),
        ]),
        (motion_params, outputnode, [
            ('spm_motion_file', 'motion_params'),
            ('diffprep_ec_file', 'ec_file'),
        ]),

        # Pre-SDC enhancement (report)
        (corrected_node, extract_b0s, [('corrected_dwi_file', 'dwi_series')]),
        (split_outputs, extract_b0s, [('b0_indices', 'b0_indices')]),
        (extract_b0s, enhance_pre_sdc, [('b0_average', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [('enhanced_file', 'pre_sdc_template')]),

        # b0 reference for coregistration (b0_template is wired below: on the
        # T2Wreg path it has to be the SDC-corrected b=0, not the raw one)
        (inputnode, b0_ref_for_coreg, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
        ]),
        (b0_ref_for_coreg, outputnode, [('outputnode.dwi_mask', 'b0_template_mask')]),

        # Carpet-plot QC + CNR (both from the shared MAPMRI synthesis)
        (corrected_node, synth_dwis, [
            ('corrected_dwi_file', 'dwi_file'),
            ('corrected_bmtxt_file', 'bmtxt_file'),
        ]),
        (b0_ref_for_coreg, synth_dwis, [('outputnode.dwi_mask', 'mask_file')]),
        (split_outputs, slice_qc, [('dwi_files', 'uncorrected_dwi_files')]),
        (synth_dwis, slice_qc, [
            ('per_volume_synth', 'ideal_image_files'),
            ('qc_mask', 'mask_image'),
        ]),
        (slice_qc, outputnode, [('slice_stats', 'slice_quality')]),
        (split_outputs, calculate_cnr, [('dwi_files', 'hmc_warped_images')]),
        (synth_dwis, calculate_cnr, [
            ('per_volume_synth', 'predicted_images'),
            ('qc_mask', 'mask_image'),
        ]),
        (calculate_cnr, outputnode, [('cnr_image', 'cnr_map')]),
    ])  # fmt:skip

    # The b=0 that coregistration sees must be susceptibility-corrected. On every
    # other path DIFFPREP's output already is (there is no in-TORTOISE SDC, so the
    # correction is a downstream warp) -- but on the T2Wreg path the EPI stage ran
    # and we deliberately took its *pre*-EPI image, so apply the field here.
    if use_t2wreg:
        apply_sdc_to_b0 = pe.Node(
            ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
            name='apply_sdc_to_b0',
        )
        workflow.connect([
            (extract_b0s, apply_sdc_to_b0, [
                ('b0_average', 'input_image'),
                ('b0_average', 'reference_image'),
            ]),
            (corrected_node, apply_sdc_to_b0, [(('sdc_warp', _listify), 'transforms')]),
            (apply_sdc_to_b0, b0_ref_for_coreg, [('output_image', 'inputnode.b0_template')]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (extract_b0s, b0_ref_for_coreg, [('b0_average', 'inputnode.b0_template')]),
        ])  # fmt:skip

    # SDC decision tree (TORTOISE-native where possible).

    # 1. PEPOLAR -> DRBUDDI. Both an ``epi`` fieldmap (a reverse-PE b=0/EPI in
    #    fmap/) and a reverse-PE *series* (``rpe_series``) land here. Non-shelled
    #    series take the same path; DRBUDDI's plain tensor fit is adequate for a
    #    q-space grid, and shell synthesis is opt-in via
    #    "drbuddi_synth_shell_bval" for the cases where it is not.
    if unit.is_pepolar:
        # For rpe_series the per-direction DIFFPREP stage above already produced
        # a single recombined series in the original merged order, so
        # GatherDRBUDDIInputs re-splits it into up/down exactly as it does for
        # the FSL backend.
        drbuddi_wf = init_drbuddi_wf(
            unit=unit,
            t2w_sdc=t2w_sdc,
            use_cuda=drbuddi_gpu,
            synth_shell_bval=synth_shell_bval,
            synth_shell_ndirs=diffprep_cfg.get('drbuddi_synth_shell_ndirs', 30),
        )

        workflow.connect([
            (split_outputs, drbuddi_wf, [
                ('dwi_files', 'inputnode.dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
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

    # 2. Fieldmap-less with a T2w -> TORTOISE T2Wreg. The EPI stage's displacement
    #    field is carried out as a warp (see DIFFPREP._list_outputs) rather than
    #    baked in, so it composes with HMC and coregistration and the data is
    #    resampled once -- the same contract as the DRBUDDI branch above. This
    #    keeps DIFFPREP's output in the native grid and leaves coregistration and
    #    ACPC alignment to qsiprep instead of TORTOISE's StructuralAlignment.
    if use_t2wreg:
        outputnode.inputs.sdc_method = 'T2Wreg (SynB0)' if synb0_target else 'T2Wreg'
        workflow.connect([
            (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
            (corrected_node, outputnode, [
                (('sdc_warp', _listify), 'to_dwi_ref_warps'),
                ('b0_up_image', 'b0_up_image'),
                ('b0_up_corrected_image', 'b0_up_corrected_image'),
                ('structural_image', 't2w_image'),
            ]),
        ])  # fmt:skip
        return workflow

    # 3. GRE / phase fieldmaps, or classic fieldmap-less SyN -> qsiprep's
    #    init_sdc_wf. The warp is applied downstream (to_dwi_ref_warps),
    #    decoupled from HMC.
    if unit.is_gre or unit.is_nipreps_syn:
        b0_sdc_wf = init_sdc_wf(unit)
        b0_sdc_wf.inputs.inputnode.template = config.workflow.anatomical_template

        workflow.connect([
            (b0_ref_for_coreg, b0_sdc_wf, [
                ('outputnode.ref_image', 'inputnode.b0_ref'),
                ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                ('outputnode.dwi_mask', 'inputnode.b0_mask'),
            ]),
            (inputnode, b0_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
            ]),
            (b0_sdc_wf, outputnode, [
                ('outputnode.method', 'sdc_method'),
                ('outputnode.b0_ref', 'b0_template'),
                ('outputnode.out_warp', 'to_dwi_ref_warps'),
            ]),
        ])  # fmt:skip
        return workflow

    # 4. No fieldmap, no T2w -> HMC only.
    outputnode.inputs.sdc_method = 'None'
    workflow.connect([
        (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
    ])  # fmt:skip

    return workflow
