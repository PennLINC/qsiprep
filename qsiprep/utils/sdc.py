"""Susceptibility-distortion-correction gating helpers.

Pure functions over the method selection and collected subject data - no
config reads - shared by workflow construction and its tests.
"""

from qsiplan.methods import HmcMethod, SdcTool


def t2w_sdc_enabled(selection):
    """Whether the selected methods have a stage that can consume a T2w for SDC.

    DRBUDDI's multimodal ``--structural`` is reached whenever DRBUDDI is among
    the PEPOLAR tools; DIFFPREP's ``--epi T2Wreg`` covers the fieldmap-less
    case and is not gated on the PEPOLAR tool choice.
    """
    return SdcTool.DRBUDDI in selection.pepolar_tools or selection.hmc is HmcMethod.TORTOISE


def t2w_available_for_sdc(subject_data, selection, anat_modality):
    """Whether a T2w should drive susceptibility distortion correction.

    True only when the subject has a T2w, anatomical processing runs
    (``anat_modality`` != ``'none'``), and the selected methods actually have
    a T2w-consuming stage. Every T2w consumer takes the anatomical workflow's
    ``t2w_unfatsat``, which is only produced when ``init_anat_preproc_wf`` is
    asked for additional T2ws (see ``additional_t2ws`` in
    :func:`qsiprep.workflows.base.init_single_subject_wf`, which must stay in
    sync with this). Requesting T2w-based SDC without it leaves those nodes
    with an empty input.
    """
    return bool(subject_data.get('t2w')) and anat_modality != 'none' and t2w_sdc_enabled(selection)


def pe_readout_time(unit):
    """``TotalReadoutTime`` (s) of a unit's lead phase-encode series, or ``None``.

    Turning TOPUP's off-resonance field (Hz) into a displacement field needs the
    readout time: the voxel shift is ``field_Hz * TotalReadoutTime``. The lead
    series is the ``+`` polarity one for a reverse-PE pair, matching
    :attr:`~qsiplan.adapters.PreprocUnit.pe_dir`. Returns ``None`` when the
    metadata is absent so callers can decline rather than fail.
    """
    lead = unit.plus_files[0] if unit.has_bidirectional_dwi else unit.dwi_files[0]
    trt = unit.sidecar_overrides().get(lead, {}).get('TotalReadoutTime')
    if trt is None:
        trt = unit.dwi_metadata.get('TotalReadoutTime')
    return float(trt) if trt is not None else None


def t2wreg_target(unit, t2w_sdc):
    """The structural target DIFFPREP's T2Wreg stage registers to, or ``None``.

    Mirrors ``use_t2wreg``/``synb0_target`` in
    :mod:`qsiprep.workflows.dwi.diffprep`. A SynB0 unit carries the T2Wreg
    stage without being a ``CorrectionMethod.T2WREG`` estimation, so the stage
    (not the method) is what says T2Wreg runs. ``'synb0'`` needs no T2w; the
    ``t2w_sdc`` bool additionally honors --anat-modality/--ignore t2w for the
    ``'t2w'`` target, which the plan does not see.
    """
    stage = unit.run.stage_with('t2wreg')
    if stage is None:
        return None
    if stage.structural_target == 'synb0':
        return 'synb0'
    return 't2w' if t2w_sdc else None


def sdc_warp_source(unit, t2w_sdc):
    """Where the SDC displacement-field derivative comes from, and its method label.

    Returns ``(source, estimation_method)``. ``source`` is ``'fieldwarp'`` when a
    susceptibility method wrote a standalone warp into ``fieldwarps`` --
    DRBUDDI, a GRE/phasediff fieldmap, fieldmap-less SyN, or TORTOISE T2Wreg --
    and ``'topup'`` when only TOPUP ran: eddy applies its field and leaves no
    standalone warp, so it is rebuilt from the off-resonance field, which needs
    a readout time. ``(None, None)`` when no susceptibility correction ran.
    """
    target = t2wreg_target(unit, t2w_sdc)
    topup = unit.run.stage_with('topup')
    if unit.run.stage_with('drbuddi') is not None:
        return 'fieldwarp', 'DRBUDDI'
    if unit.is_gre:
        return 'fieldwarp', 'GRE fieldmap'
    if unit.is_nipreps_syn:
        return 'fieldwarp', 'SyN (fieldmap-less)'
    if target is not None:
        return 'fieldwarp', 'TORTOISE T2Wreg (SynB0)' if target == 'synb0' else 'TORTOISE T2Wreg'
    if topup is not None and pe_readout_time(unit) is not None:
        return 'topup', 'TOPUP (SynB0)' if topup.structural_target == 'synb0' else 'TOPUP'
    return None, None
