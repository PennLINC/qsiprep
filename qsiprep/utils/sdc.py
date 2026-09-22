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


def resolve_t2wreg_target(unit, t2w_sdc):
    """The structural target DIFFPREP's T2Wreg stage registers to, or ``None``.

    Mirrors ``use_t2wreg``/``synb0_target`` in
    :mod:`qsiprep.workflows.dwi.diffprep`. T2Wreg does real susceptibility
    distortion correction but carries no measured fieldmap, so without this
    predicate the fieldmap-less case would fall through the reportlet gate and
    produce no SDC figure. The plan encodes the stage and its target
    (``'synb0'`` needs no T2w); the ``t2w_sdc`` bool additionally honors
    --anat-modality/--ignore t2w for the ``'t2w'`` target.

    Used by ``init_dwi_preproc_wf``'s SDC reportlet gate and by
    :mod:`qsiprep.utils.jacobian_provenance`, which asks the same question to
    decide whether a TORTOISE unit's T2Wreg stage actually reaches
    ``fieldwarps``.
    """
    stage = unit.run.stage_with('t2wreg')
    if stage is None:
        return None
    if stage.structural_target == 'synb0':
        return 'synb0'
    return 't2w' if t2w_sdc else None
