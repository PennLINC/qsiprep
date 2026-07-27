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
import os.path as op
from importlib.resources import files

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradients import ExtractB0s, SliceQC
from ...interfaces.nilearn import EnhanceB0
from ...interfaces.shoreline import B0Mean, CalculateCNR, ExtractDWIsForModel, SignalPrediction
from ...interfaces.tortoise import (
    DIFFPREP,
    DRBUDDI,
    ConcatenateDIFFPREPGroups,
    DIFFPREPMotionParams,
    DIFFPREPSplitOutputs,
    DRBUDDIAggregateOutputs,
    MergeVolumes4D,
    SplitCorrectedByGroup,
    SplitDWIsByDistortionGroup,
    StageDRBUDDIPair,
    SynthesizeDWIs,
    TORTOISEConvert,
    WriteBmatTORTOISE,
    WriteDRBUDDIJSON,
    WriteFSLGradFiles,
    equally_distributed_directions,
    generate_diffprep_boilerplate,
)
from ...utils.resources import as_path
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf
from .util import init_dwi_reference_wf

# BIDS PhaseEncodingDirection axes already match what TORTOISEProcess expects
# in its own JSON input -- TORTOISE consumes "i", "j", "k" (with optional "-")
# straight from the BIDS sidecar.
_VALID_PE = {'i', 'i-', 'j', 'j-', 'k', 'k-'}


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
    # None => auto-detect whether a reverse-PE series is tensor-fittable
    # as-is (see _rpe_series_is_shelled). true/false forces the decision.
    cfg.setdefault('rpe_series_shelled', None)
    return cfg


def _sibling_bval(nii_file):
    """Return the FSL ``.bval`` sibling path for a BIDS DWI nii."""
    for ext in ('.nii.gz', '.nii'):
        if nii_file.endswith(ext):
            return nii_file[: -len(ext)] + '.bval'
    return op.splitext(nii_file)[0] + '.bval'


def _rpe_series_is_shelled(
    scan_groups,
    b0_threshold,
    override=None,
    tol=100.0,
    max_tensor_bval=1500.0,
    min_shell_dirs=6,
):
    """Decide whether a reverse-PE series is tensor-fittable as-is.

    DRBUDDI's ``rpe_series`` path fits its own tensor per phase-encoding
    direction and uses ``[b0, FA]`` as a 2-channel registration target. That
    tensor fit is well-conditioned only when the data contains a populous,
    tensor-fittable low-b shell -- true for DTI and multi-shell HARDI, false for
    a CS-DSI q-space grid, where DRBUDDI would silently produce a poor A0/FA.

    Returns ``True`` (use the corrected series directly) when a shell centre in
    ``[b0_threshold, max_tensor_bval]`` holds at least ``min_shell_dirs`` volumes
    within ``tol`` s/mm^2 of it; ``False`` (synthesize a shell) otherwise. A
    user override (``--diffprep-config`` ``"rpe_series_shelled"``) wins. When the
    b-values cannot be read -- e.g. docs/graph builds with placeholder paths --
    we default to shelled, which is safe for real data because BIDS DWI always
    ships ``.bval`` files.

    .. note::
       This is a heuristic and the DIFFPREP ``rpe_series`` path has no CS-DSI CI
       dataset yet, so it must be validated against real data. Override it with
       ``"rpe_series_shelled": true|false`` in ``--diffprep-config`` if a
       particular acquisition is misclassified.
    """
    import numpy as np
    from dipy.core.gradients import unique_bvals_tolerance

    if override is not None:
        return bool(override)

    fieldmap_info = scan_groups['fieldmap_info']
    nii_files = list(scan_groups.get('dwi_series', [])) + list(
        fieldmap_info.get('rpe_series', [])
    )
    all_bvals = []
    for nii in nii_files:
        bval_path = _sibling_bval(nii)
        try:
            all_bvals.append(np.loadtxt(bval_path).reshape(-1))
        except (OSError, ValueError):
            config.loggers.workflow.warning(
                'rpe_series shelled/non-shelled detection could not read %s; '
                'defaulting to the shelled (stock DRBUDDI) path. Set '
                '"rpe_series_shelled" in --diffprep-config to override.',
                bval_path,
            )
            return True

    if not all_bvals:
        return True
    bvals = np.concatenate(all_bvals)
    non_b0 = bvals[bvals >= b0_threshold]
    if non_b0.size == 0:
        return True

    for centre in unique_bvals_tolerance(non_b0, tol=tol):
        if b0_threshold <= centre <= max_tensor_bval:
            if int(np.sum(np.abs(non_b0 - centre) <= tol)) >= min_shell_dirs:
                return True
    return False


def _resolve_phase_encoding(pe_dir):
    """Validate a BIDS PhaseEncodingDirection value. Falls back to 'j'
    (anterior-posterior) when missing or not recognised -- which is what most
    clinical DWI protocols use."""
    if pe_dir in _VALID_PE:
        return pe_dir
    return 'j'


def _write_sidecar_json(nii_file, phase_encoding_direction, working_dir=None):
    """Function-node helper. Writes a BIDS-style JSON sidecar with the same
    basename as ``nii_file`` so TORTOISEProcess can read PhaseEncodingDirection
    from it.

    The JSON is written into **this node's own working directory** (the
    function-node cwd), NOT into the directory ``nii_file`` lives in. Only the
    basename of ``nii_file`` is used (so the sidecar stem matches the ``.nii``
    TORTOISE looks up); its directory is intentionally discarded. Writing into
    the node's own cwd keeps each node's outputs inside its own directory, so
    the standard nipype ``copyfile=True`` propagation to the ``diffprep`` node
    carries a valid sidecar regardless of how the upstream node's cache was
    cleared.
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
    workflow, inputnode, diffprep_kwargs, pe_axis, b0_threshold, n_procs
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
        SplitDWIsByDistortionGroup(pe_axis=pe_axis, b0_threshold=b0_threshold),
        name='split_rpe_groups',
    )
    workflow.connect([
        (inputnode, split_groups, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('original_files', 'original_files'),
        ]),
    ])  # fmt:skip

    recombine = pe.Node(ConcatenateDIFFPREPGroups(), name='recombine_rpe_groups')
    workflow.connect([
        (split_groups, recombine, [('group_assignments', 'group_assignments')]),
    ])  # fmt:skip

    for group_id in (1, 2):
        convert = pe.Node(TORTOISEConvert(), name=f'tortoise_convert_g{group_id}')
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


def _init_diffprep_predict_shell_wf(
    n_directions=32, bval=1000.0, minimal_q_distance=2.0, name='predict_shell_wf'
):
    """Synthesize a tensor-fittable ``[b0 + n*bval]`` shell for one PE side.

    Adapted from the validated ``csdsi_preproc.predict.init_predict_shell_wf``,
    but **without** its per-volume HMC ``ApplyTransforms`` step: DIFFPREP has
    already baked motion+eddy correction into the volumes (identity affines from
    :class:`DIFFPREPSplitOutputs`), so the corrected non-b=0 volumes are already
    aligned and feed straight into 3dSHORE :class:`SignalPrediction`. Volume 0 of
    the emitted shell is the side's own b=0 mean (measured, not predicted); the
    ``n`` non-zero directions are the deterministic
    :func:`equally_distributed_directions` set, so the up and down shells are
    directly comparable to DRBUDDI.
    """
    workflow = Workflow(name=name)
    bvecs_full, bvals_full = equally_distributed_directions(n=n_directions, bval=bval)
    target_bvecs = bvecs_full[1:]
    target_bvals = bvals_full[1:]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_files',
                'bvec_files',
                'bval_files',
                'transforms',
                'b0_indices',
                'b0_template',
                'b0_template_mask',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['predicted_4d', 'predicted_bvec_file', 'predicted_bval_file']
        ),
        name='outputnode',
    )

    extract = pe.Node(ExtractDWIsForModel(), name='extract_non_b0')
    predict = pe.MapNode(
        SignalPrediction(model='3dSHORE', minimal_q_distance=minimal_q_distance),
        iterfield=['bvec_to_predict', 'bval_to_predict'],
        name='predict_directions',
    )
    predict.inputs.bvec_to_predict = list(target_bvecs)
    predict.inputs.bval_to_predict = list(target_bvals.astype(float))

    merge_4d = pe.Node(MergeVolumes4D(), name='merge_4d')
    write_grad = pe.Node(WriteFSLGradFiles(), name='write_grad')
    write_grad.inputs.bvecs = bvecs_full
    write_grad.inputs.bvals = bvals_full

    workflow.connect([
        (inputnode, extract, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('transforms', 'transforms'),
            ('b0_indices', 'b0_indices'),
        ]),
        (extract, predict, [
            ('model_dwi_files', 'aligned_dwis'),
            ('model_bvecs', 'aligned_bvecs'),
            ('model_bvals', 'bvals'),
        ]),
        (inputnode, predict, [
            ('b0_template', 'aligned_b0_mean'),
            ('b0_template_mask', 'aligned_mask'),
        ]),
        (inputnode, merge_4d, [('b0_template', 'b0_image')]),
        (predict, merge_4d, [('predicted_image', 'predicted_images')]),
        (merge_4d, outputnode, [('merged_4d', 'predicted_4d')]),
        (write_grad, outputnode, [
            ('bvec_file', 'predicted_bvec_file'),
            ('bval_file', 'predicted_bval_file'),
        ]),
    ])  # fmt:skip
    return workflow


def _build_rpe_synthesis_sdc(
    workflow, inputnode, outputnode, split_outputs, mask_source, scan_groups, dwi_metadata, t2w_sdc
):
    """DRBUDDI SDC for a **non-shelled** reverse-PE series (e.g. CS-DSI).

    DRBUDDI's ``rpe_series`` path fits its own ``[b0, FA]`` per direction; on a
    q-space grid that fit is ill-conditioned. So per PE direction we synthesize a
    tensor-fittable ``[b0 + 32*b1000]`` shell (3dSHORE prediction on the
    already-corrected volumes) and hand THOSE to the ``DRBUDDI`` interface. The
    predicted shell of a side lives in the same distorted geometry as that side's
    real volumes, so the deformation DRBUDDI estimates is applied to the *real*
    data via the **stock** :class:`DRBUDDIAggregateOutputs` (unchanged), using the
    real series' per-volume ``blip_assignments``.

    This mirrors the validated ``csdsi_preproc`` recipe (stage synthesized shells,
    call DRBUDDI directly), adapted to qsiprep's single-``scan_groups`` path where
    one DRBUDDI run corrects both directions. ``init_drbuddi_wf`` is intentionally
    NOT reused here: its ``GatherDRBUDDIInputs`` rebuilds DRBUDDI's registration
    images from the real (non-shelled) series, which is exactly what we must avoid.
    """
    split_corrected = pe.Node(SplitCorrectedByGroup(), name='split_corrected_by_group')
    workflow.connect([
        (split_outputs, split_corrected, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('forward_transforms', 'forward_transforms'),
            ('b0_indices', 'b0_indices'),
        ]),
        (inputnode, split_corrected, [('original_files', 'original_files')]),
    ])  # fmt:skip

    stage_pair = pe.Node(StageDRBUDDIPair(), name='stage_drbuddi_pair')
    for side in ('up', 'down'):
        b0_mean = pe.Node(B0Mean(), name=f'{side}_b0_mean')
        predict = _init_diffprep_predict_shell_wf(name=f'predict_{side}_shell')
        write_bmat = pe.Node(WriteBmatTORTOISE(), name=f'{side}_write_bmat')
        workflow.connect([
            (split_corrected, b0_mean, [(f'{side}_b0_files', 'b0_images')]),
            (split_corrected, predict, [
                (f'{side}_dwi_files', 'inputnode.dwi_files'),
                (f'{side}_bval_files', 'inputnode.bval_files'),
                (f'{side}_bvec_files', 'inputnode.bvec_files'),
                (f'{side}_transforms', 'inputnode.transforms'),
                (f'{side}_b0_indices', 'inputnode.b0_indices'),
            ]),
            (b0_mean, predict, [('average_image', 'inputnode.b0_template')]),
            (mask_source, predict, [('outputnode.dwi_mask', 'inputnode.b0_template_mask')]),
            (predict, write_bmat, [
                ('outputnode.predicted_bvec_file', 'bvec_file'),
                ('outputnode.predicted_bval_file', 'bval_file'),
            ]),
            (predict, stage_pair, [('outputnode.predicted_4d', f'{side}_image')]),
            (write_bmat, stage_pair, [('bmat_file', f'{side}_bmat')]),
        ])  # fmt:skip

    write_json = pe.Node(
        WriteDRBUDDIJSON(phase_encoding_direction=scan_groups['dwi_series_pedir']),
        name='write_drbuddi_json',
    )
    if dwi_metadata:
        if 'TotalReadoutTime' in dwi_metadata:
            write_json.inputs.total_readout_time = float(dwi_metadata['TotalReadoutTime'])
        if 'EchoTime' in dwi_metadata:
            write_json.inputs.echo_time = float(dwi_metadata['EchoTime'])

    drbuddi = pe.Node(
        DRBUDDI(
            fieldmap_type='rpe_series',
            num_threads=config.nipype.omp_nthreads,
            sloppy=config.execution.sloppy,
            start_with_diffeomorphic_for_rigid_reg=config.execution.sloppy,
        ),
        name='drbuddi',
        n_procs=config.nipype.omp_nthreads,
    )

    aggregate = pe.Node(
        DRBUDDIAggregateOutputs(fieldmap_type='rpe_series'), name='aggregate_drbuddi'
    )

    workflow.connect([
        (stage_pair, drbuddi, [
            ('up_image', 'blip_up_image'),
            ('up_bmat', 'blip_up_bmat'),
            ('down_image', 'blip_down_image'),
            ('down_bmat', 'blip_down_bmat'),
        ]),
        (write_json, drbuddi, [('out_json', 'blip_up_json')]),
        (split_corrected, aggregate, [('blip_assignments', 'blip_assignments')]),
        (drbuddi, aggregate, [
            ('undistorted_reference', 'undistorted_reference'),
            ('bdown_to_bup_rigid_trans_h5', 'bdown_to_bup_rigid_trans_h5'),
            ('blip_down_b0', 'blip_down_b0'),
            ('blip_down_b0_corrected', 'blip_down_b0_corrected'),
            ('blip_down_b0_corrected_jac', 'blip_down_b0_corrected_jac'),
            ('blip_down_b0_quad', 'blip_down_b0_quad'),
            ('blip_up_b0', 'blip_up_b0'),
            ('blip_up_b0_corrected', 'blip_up_b0_corrected'),
            ('blip_up_b0_corrected_jac', 'blip_up_b0_corrected_jac'),
            ('blip_up_b0_quad', 'blip_up_b0_quad'),
            ('deformation_finv', 'deformation_finv'),
            ('deformation_minv', 'deformation_minv'),
            ('blip_up_FA', 'blip_up_FA'),
            ('blip_down_FA', 'blip_down_FA'),
            ('structural_image', 'structural_image'),
        ]),
    ])  # fmt:skip

    if t2w_sdc:
        # Only DRBUDDI takes the T2w; aggregate_drbuddi.structural_image is fed
        # from drbuddi.structural_image above (matching stock init_drbuddi_wf).
        # Connecting inputnode here too would double-connect that input and make
        # nipype raise at build time.
        workflow.connect([
            (inputnode, drbuddi, [('t2w_unfatsat', 'structural_image')]),
        ])  # fmt:skip

    outputnode.inputs.sdc_method = 'DRBUDDI (predicted shell)'
    outputnode.inputs.fieldmap_type = 'rpe_series'
    workflow.connect([
        (aggregate, outputnode, [
            ('sdc_warps', 'to_dwi_ref_warps'),
            ('sdc_scaling_images', 'sdc_scaling_images'),
            ('up_fa_corrected_image', 'up_fa_corrected_image'),
            ('down_fa_corrected_image', 'down_fa_corrected_image'),
            ('b0_ref', 'b0_template'),
        ]),
        (drbuddi, outputnode, [
            ('blip_up_b0', 'b0_up_image'),
            ('blip_down_b0', 'b0_down_image'),
            ('blip_up_b0_corrected', 'b0_up_corrected_image'),
            ('blip_down_b0_corrected', 'b0_down_corrected_image'),
            ('blip_up_FA', 'up_fa_image'),
            ('blip_down_FA', 'down_fa_image'),
            ('structural_image', 't2w_image'),
        ]),
    ])  # fmt:skip


def init_diffprep_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    correction_mode='quadratic',
    dwi_metadata=None,
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
    scan_groups : dict
        Same scan-groups dict the other HMC backends consume.
    source_file : str
        Path to the source DWI file (used for report naming).
    t2w_sdc : bool
        Whether a T2w image is available for distortion correction (used for
        DRBUDDI's multi-modal registration and for the fieldmap-less T2Wreg
        path).
    correction_mode : str
        One of ``'motion'`` (rigid only), ``'quadratic'`` (recommended), or
        ``'cubic'``. Forwarded to TORTOISE as ``-c``.
    dwi_metadata : dict, optional
        BIDS sidecar metadata (used for the PE direction and for SDC).
    name : str
        Workflow name.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = generate_diffprep_boilerplate(correction_mode)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'original_files',
                't1_brain',
                't1_mask',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
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

    fieldmap_info = scan_groups['fieldmap_info']
    fieldmap_type = fieldmap_info['suffix']

    # TORTOISE-native T2Wreg replaces SyN for the fieldmap-less case when a T2w
    # structural is available: DIFFPREP runs the ``--epi T2Wreg`` stage in the
    # same TORTOISEProcess call and bakes the correction into its output.
    is_fieldmapless = fieldmap_type is None or fieldmap_type == 'syn'
    use_t2wreg = is_fieldmapless and bool(t2w_sdc)
    epi_mode = 'T2Wreg' if use_t2wreg else 'off'

    # Load any user-supplied DIFFPREP config (or our defaults)
    diffprep_cfg = _load_diffprep_config(config.workflow.diffprep_config)

    # For a reverse-PE series, decide up front whether DRBUDDI can tensor-fit
    # the corrected series as-is (shelled) or needs a synthesized single shell
    # for its [b0, FA] target (non-shelled, e.g. CS-DSI). See
    # _rpe_series_is_shelled. The DIFFPREP split/recombine stage below is the
    # same either way; only the DRBUDDI-input derivation differs.
    rpe_shelled = None
    if fieldmap_type == 'rpe_series':
        rpe_shelled = _rpe_series_is_shelled(
            scan_groups,
            config.workflow.b0_threshold,
            override=diffprep_cfg.get('rpe_series_shelled'),
        )

    pe_dir = _resolve_phase_encoding((dwi_metadata or {}).get('PhaseEncodingDirection'))

    # --sloppy asks for underpowered-but-fast registration (the same contract
    # DRBUDDI honours via its own ``sloppy`` input). Every DWI is registered to
    # the b=0 regardless; what costs the time is TORTOISE's *second* pass, which
    # fits DTI+MAPMRI to the corrected data, synthesizes a contrast-matched
    # target per volume and re-registers against it. That pass is gated on
    # ``iterative = (is_human_brain && high_bval) || s2v || repol``, and
    # ``--niter 0`` sets ``iterative=false`` outright (DIFFPREP.cxx:1359).
    #
    # Use ONLY --niter 0 here. Clearing ``is_human_brain`` would reach the same
    # flag but is not a speed knob: it also makes DIFFPREP's auto-masking read a
    # ``<stem>_noise.nii`` (DIFFPREP.cxx:2349) and changes structural-target
    # masking on the T2Wreg path (TORTOISE.cxx:1079).
    #
    # ``--niter 0`` only bites on high-b (>1200) data, so it alone does not bound
    # runtime for DTI-regime test data. The first pass -- which always runs --
    # fits a 24-parameter quadratic per volume; dropping to rigid ('motion') is
    # what actually bounds it. Both are sloppy-only; production runs still get
    # the correction mode the user asked for.
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

    diffprep_kwargs = dict(
        correction_mode=effective_correction_mode,
        b0_id=diffprep_cfg['b0_id'],
        is_human_brain=diffprep_cfg['is_human_brain'],
        rot_eddy_center=diffprep_cfg['rot_eddy_center'],
        extra_args=diffprep_cfg['extra_args'],
        epi_mode=epi_mode,
        **sloppy_kwargs,
    )

    # DIFFPREP stage. A reverse-PE *series* is corrected once per phase-encoding
    # direction (DIFFPREP models a single phase axis / single b=0 reference for
    # a whole file), then recombined; every other case is a single DIFFPREP run.
    # Both expose the identical ``corrected_dwi_file`` / ``corrected_bmtxt_file``
    # / ``transformations_file`` triple via ``corrected_node``.
    if fieldmap_type == 'rpe_series':
        corrected_node = _build_rpe_diffprep_stage(
            workflow,
            inputnode,
            diffprep_kwargs,
            pe_axis=pe_dir[0],
            b0_threshold=config.workflow.b0_threshold,
            n_procs=config.nipype.omp_nthreads,
        )
    else:
        # Convert gzipped niftis + FSL gradients into TORTOISE format (.nii + .bmtxt).
        tortoise_convert = pe.Node(TORTOISEConvert(), name='tortoise_convert')

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

        # T2Wreg bakes SDC into the DIFFPREP call: feed the T2w structural.
        if use_t2wreg:
            workflow.connect([
                (inputnode, diffprep, [('t2w_unfatsat', 'structural_image')]),
            ])  # fmt:skip

        corrected_node = diffprep

    split_outputs = pe.Node(
        DIFFPREPSplitOutputs(b0_threshold=config.workflow.b0_threshold),
        name='split_outputs',
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
        (motion_params, outputnode, [('spm_motion_file', 'motion_params')]),

        # Pre-SDC enhancement (report)
        (corrected_node, extract_b0s, [('corrected_dwi_file', 'dwi_series')]),
        (split_outputs, extract_b0s, [('b0_indices', 'b0_indices')]),
        (extract_b0s, enhance_pre_sdc, [('b0_average', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [('enhanced_file', 'pre_sdc_template')]),

        # b0 reference for coregistration
        (extract_b0s, b0_ref_for_coreg, [('b0_average', 'inputnode.b0_template')]),
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

    # -----------------------------------------------------------------------
    # SDC decision tree (TORTOISE-native where possible)
    # -----------------------------------------------------------------------

    # 1. PEPOLAR -> DRBUDDI. Both an ``epi`` fieldmap (a reverse-PE b=0/EPI in
    #    fmap/) and a reverse-PE *series* (``rpe_series``) land here.
    if fieldmap_type in ('epi', 'rpe_series'):
        if 'topup' in config.workflow.pepolar_method.lower():
            raise Exception(
                'TOPUP-based pepolar correction is not supported with '
                '--hmc-model diffprep_*; choose --pepolar-method DRBUDDI.'
            )

        # Non-shelled reverse-PE series: DRBUDDI cannot tensor-fit a usable
        # [b0, FA] from a q-space grid, so synthesize a single shell per PE
        # direction and hand those to DRBUDDI (see _build_rpe_synthesis_sdc).
        if fieldmap_type == 'rpe_series' and not rpe_shelled:
            config.loggers.workflow.warning(
                'Non-shelled reverse-PE series detected: using the EXPERIMENTAL '
                'predicted-shell DRBUDDI path (a tensor-fittable shell is '
                'synthesized per phase-encoding direction). This path has no '
                'CS-DSI CI dataset yet and must be validated against real data; '
                'inspect the SDC report. Override detection with '
                '"rpe_series_shelled" in --diffprep-config.'
            )
            _build_rpe_synthesis_sdc(
                workflow,
                inputnode,
                outputnode,
                split_outputs,
                b0_ref_for_coreg,
                scan_groups,
                dwi_metadata,
                t2w_sdc,
            )
            return workflow

        # ``epi`` fieldmaps and *shelled* reverse-PE series go through the stock
        # DRBUDDI workflow unchanged: for rpe_series the per-direction DIFFPREP
        # stage above already produced a single recombined series in the original
        # merged order, so GatherDRBUDDIInputs re-splits it into up/down exactly
        # as it does for the FSL backend.
        drbuddi_wf = init_drbuddi_wf(scan_groups=scan_groups, t2w_sdc=t2w_sdc)

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

    # 2. Fieldmap-less with a T2w -> TORTOISE T2Wreg (already baked into DIFFPREP)
    if use_t2wreg:
        outputnode.inputs.sdc_method = 'T2Wreg'
        workflow.connect([
            (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
        ])  # fmt:skip
        return workflow

    # 3. GRE / phase fieldmaps, or SyN fallback (no T2w) -> qsiprep's init_sdc_wf.
    #    The warp is applied downstream (to_dwi_ref_warps), decoupled from HMC.
    if fieldmap_type in ('fieldmap', 'syn') or (
        fieldmap_type is not None and fieldmap_type.startswith('phase')
    ):
        b0_sdc_wf = init_sdc_wf(scan_groups['fieldmap_info'], dwi_metadata)
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
