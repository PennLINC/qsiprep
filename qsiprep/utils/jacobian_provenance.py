"""Per-unit Jacobian-modulation provenance for the sidecar.

Which spatial-distortion corrections a DWI run's own Jacobian weight
derivative modulated is a build-time fact about *that run*: it depends on
which HMC/SDC backend the run's compiled plan selected, whether that run has
a gradwarp field, and whether a susceptibility warp reaches ``fieldwarps``
externally rather than being baked into the HMC backend's own resampling
(``eddy``). None of that is invocation-global -- a multi-run or multi-subject
invocation can mix runs with and without gradwarp, or PEPOLAR runs alongside
uncorrected ones -- so it must be computed once per :class:`~qsiplan.
adapters.PreprocUnit`, not accumulated into a single shared list across the
whole invocation (see the removed ``config.record_applied``/
``config.record_unmodulated`` and the git history of this module for why that
was wrong).

:func:`jacobian_provenance_for` is the sole entry point. It mirrors, without
executing, the branching that ``qsiprep.workflows.dwi.base``,
``qsiprep.workflows.dwi.fsl``, ``qsiprep.workflows.dwi.hmc_sdc`` and
``qsiprep.workflows.dwi.diffprep`` already perform when they build a run's
Jacobian-consuming nodes and warp connections -- see the module docstring of
each for the corresponding wiring. Keeping the two in sync is a discipline
this module cannot enforce by itself; the workflow modules cross-reference it
in comments at each branch point instead of duplicating this reasoning.
"""

from .. import config
from .diffprep_config import load_diffprep_config
from .eddy_config import eddy_applies_gre, eddy_modulates_distortion, load_eddy_args
from .sdc import t2wreg_target

#: Reason string for both eddy-current and susceptibility unmodulated entries
#: under a non-'jac' eddy resampling method -- eddy has already baked its
#: resampling in for both, under the same configuration knob.
_EDDY_UNMODULATED_REASON = 'FSL eddy ran with --resamp={method} rather than jac'

_DIFFPREP_CUBIC_UNMODULATED_REASON = (
    "DIFFPREP ran with correction_mode='cubic', whose eddy-current polynomial "
    '(cubic Okan terms) has no implemented Jacobian determinant.'
)

_T2WREG_UNMODULATED_REASON = (
    'TORTOISE applies the fieldmap-less T2Wreg (EPIREG) field without intensity '
    'modulation: its final registration stage is not restricted to the '
    'phase-encoding direction, so its Jacobian is not a volume change. Pass '
    '--force jacobian to modulate by the phase-encoding component anyway.'
)


def t2wreg_is_weighted(unit, t2w_sdc):
    """Whether this unit's susceptibility field gets an intensity weight.

    Only TORTOISE's T2Wreg (EPIREG) field is exempt, and ``--force jacobian``
    lifts the exemption. Every other field (DRBUDDI, GRE, SyN) is weighted.
    """
    if t2wreg_target(unit, t2w_sdc) is None:
        return True
    return 'jacobian' in (config.workflow.force or [])


def _gradwarp_applied(unit):
    """True when this unit's gradwarp field reaches ComposeJacobianWeights.

    Mirrors ``qsiprep.workflows.dwi.base``'s gradwarp block: a spatial warp is
    built (and consumed by ``ComposeJacobianWeights``, via
    ``init_dwi_trans_wf``) exactly when ``resolve_gradwarp_plan(unit)`` is not
    ``None`` and its ``warp_dim`` is not ``None`` (a DIS3D unit needing no
    spatial correction produces a plan with ``warp_dim=None``).
    """
    # Imported here, not at module level: resolve_gradwarp_plan lives in a
    # workflow module that imports describe_jacobian_modulation from this one.
    from ..workflows.dwi.gradwarp import resolve_gradwarp_plan

    plan = resolve_gradwarp_plan(unit)
    return plan is not None and plan.warp_dim is not None


def _shoreline_sdc_applied(unit):
    """Mirrors ``qsiprep.workflows.dwi.hmc_sdc``'s SDC recording.

    No TOPUP-in-eddy carve-out exists for this backend: every SDC warp it
    produces (PEPOLAR -- including a TOPUP-only plan, via DRBUDDI's own
    wrapping -- or a GRE/SyN fieldmap) is carried in ``to_dwi_ref_warps`` and
    reaches ``ComposeJacobianWeights`` externally.

    This is *not* simply whether any correction method was selected --
    ``unit.method`` is also non-``None`` for the fieldmap-less methods
    (SYNB0, T2Wreg), and those are only ever applied by the TORTOISE backend
    (``init_sdc_wf``'s own docstring, ``qsiprep/workflows/fieldmap/base.py``);
    on SHORELine (and eddy), ``qsiplan.plan._stages_for_unit`` never attaches
    an ``ESTIMATE_AND_APPLY`` stage for either, so no warp is ever built. The
    condition mirrors ``init_sdc_wf``'s own ``does_sdc`` gate exactly
    (``qsiprep/workflows/fieldmap/base.py:114-115``): a real SDC warp exists
    here precisely when there is a scanner-measured fieldmap (PEPOLAR or GRE)
    or classic NiPreps SyN.
    """
    return unit.has_scanner_measured_fieldmap or unit.is_nipreps_syn


def _eddy_provenance(unit):
    """Mirrors ``qsiprep.workflows.dwi.fsl``'s SDC/eddy-current recording.

    ``eddy``'s own resampling bakes in TOPUP's field and a GRE fieldmap handed
    to it (:func:`~qsiprep.utils.eddy_config.eddy_applies_gre`), never DRBUDDI's
    or SyN's, which are applied downstream of ``eddy`` and Jacobian-modulated by
    QSIPrep itself, so:

    * 'eddy-current' is unmodulated whenever ``eddy`` did not run with
      ``--resamp=jac`` -- unconditionally, since eddy-current is entirely
      internal to ``eddy``.
    * 'susceptibility' is unmodulated under that same condition *only* when
      TOPUP or a GRE fieldmap eddy applied is this run's susceptibility source.
      This is independent of whether DRBUDDI also runs afterwards (a
      TOPUP+DRBUDDI refine plan can have TOPUP's component unmodulated while
      DRBUDDI's own warp is separately, externally applied) -- see the F2
      regression test for exactly this combination.
    * 'sdc' is applied whenever a warp reaches ``to_dwi_ref_warps``
      externally: DRBUDDI (whether or not TOPUP also ran), SyN, or a GRE
      fieldmap applied after ``eddy`` (``--gre-sdc-after-eddy``). TOPUP-only and
      a GRE fieldmap eddy applied never apply 'sdc' -- they are baked into
      ``eddy``.
    """
    applied = []
    unmodulated = []
    reason = None

    eddy_args = load_eddy_args()
    eddy_will_modulate = eddy_modulates_distortion(eddy_args)
    run_topup = unit.run.stage_with('topup') is not None
    run_drbuddi = unit.run.stage_with('drbuddi') is not None
    gre_in_eddy = eddy_applies_gre(unit)

    if not eddy_will_modulate:
        reason = _EDDY_UNMODULATED_REASON.format(method=eddy_args.get('method'))
        unmodulated.append('eddy-current')
        if run_topup or gre_in_eddy:
            unmodulated.append('susceptibility')

    # ``fsl.py:594`` branches on ``run_drbuddi`` alone (no ``is_pepolar``
    # conjunct); the extra ``unit.is_pepolar`` guard here is redundant with
    # it, not a divergence: ``qsiplan.plan._stages_for_unit`` only ever calls
    # ``_pepolar_stages`` (the sole place a ``drbuddi`` ``PlanStage`` is ever
    # constructed, both under eddy and under DIFFPREP/SHORELine) when
    # ``unit.is_pepolar`` is true (``qsiplan/plan.py``, verified against both
    # the checked-out dev tree and the pinned ``qsiplan==0.4.0`` tag). So
    # ``run_drbuddi`` already implies ``unit.is_pepolar`` by construction of
    # the planner, and this conjunct can never actually diverge from the
    # builder's condition -- kept anyway as a documented, load-bearing
    # invariant check rather than trusting an external package silently.
    if (
        (unit.is_pepolar and run_drbuddi)
        or (unit.is_gre and not gre_in_eddy)
        or unit.is_nipreps_syn
    ):
        applied.append('sdc')

    return applied, unmodulated, reason


def _tortoise_provenance(unit, t2w_sdc):
    """Mirrors ``qsiprep.workflows.dwi.diffprep``'s SDC/eddy-current recording.

    The eddy-current component is DIFFPREP's own Okan quadratic-transform
    Jacobian, gated on the *effective* correction mode (post ``--sloppy``
    downgrade, which forces 'motion'): 'quadratic' is applied, 'cubic' is
    unmodulated (no implemented determinant), 'motion' has no eddy-current
    component at all so it is neither.

    The SDC component reaches ``fieldwarps`` externally for every branch
    DIFFPREP's own decision tree takes except "no fieldmap, no T2w": PEPOLAR
    (DRBUDDI), the T2Wreg fieldmap-less case (``t2wreg_target``
    mirrors the same ``use_t2wreg`` gate ``diffprep.py`` computes, including
    its ``t2w_sdc``/``--anat-modality`` dependency), and GRE/SyN.
    """
    applied = []
    unmodulated = []
    reason = None

    diffprep_cfg = load_diffprep_config(config.workflow.diffprep_config)
    correction_mode = diffprep_cfg['correction_mode']
    effective_correction_mode = 'motion' if config.execution.sloppy else correction_mode
    if effective_correction_mode == 'quadratic':
        applied.append('eddy-current')
    elif effective_correction_mode == 'cubic':
        unmodulated.append('eddy-current')
        reason = _DIFFPREP_CUBIC_UNMODULATED_REASON

    if unit.is_pepolar:
        applied.append('sdc')
    elif unit.is_gre or unit.is_nipreps_syn:
        applied.append('sdc')
    elif t2wreg_target(unit, t2w_sdc) is not None:
        if t2wreg_is_weighted(unit, t2w_sdc):
            applied.append('sdc')
        else:
            unmodulated.append('sdc')
            reason = (
                f'{reason} {_T2WREG_UNMODULATED_REASON}' if reason else _T2WREG_UNMODULATED_REASON
            )

    return applied, unmodulated, reason


def jacobian_provenance_for(unit, t2w_sdc=False):
    """Which corrections this unit's run Jacobian-modulated, and why not the rest.

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The correction unit a single HMC+SDC run compiles, carrying the
        compiled ``run`` (stage sequence) the actual workflow builders
        dispatch on.
    t2w_sdc : bool
        Whether a T2w is available for TORTOISE's fieldmap-less T2Wreg stage,
        honoring ``--anat-modality``/``--ignore t2w`` -- the same value
        ``init_dwi_preproc_wf``/``init_diffprep_hmc_wf`` receive. Irrelevant
        off the TORTOISE backend.

    Returns
    -------
    tuple[list[str], list[str], str | None]
        ``(applied, unmodulated, reason)`` -- which corrections QSIPrep
        itself Jacobian-modulated for this run, which ran without modulation,
        and why (a single reason string, since no configuration produces two
        distinct unmodulated reasons for one run).
    """
    hmc_tool = unit.run.hmc_stage.tool

    applied = []
    unmodulated = []
    reason = None

    if _gradwarp_applied(unit):
        applied.append('gradwarp')

    if hmc_tool == 'shoreline':
        if _shoreline_sdc_applied(unit):
            applied.append('sdc')
    elif hmc_tool == 'eddy':
        backend_applied, unmodulated, reason = _eddy_provenance(unit)
        applied.extend(backend_applied)
    elif hmc_tool == 'tortoise':
        backend_applied, unmodulated, reason = _tortoise_provenance(unit, t2w_sdc)
        applied.extend(backend_applied)
    else:
        raise ValueError(f'Unknown HMC tool: {hmc_tool!r}')

    return applied, unmodulated, reason


_JACOBIAN_SENTENCE = {
    True: (
        ' A Jacobian intensity correction was applied to compensate for the '
        'local volume change this correction introduces.'
    ),
    False: (
        ' This correction was applied without Jacobian intensity modulation '
        '(--ignore jacobian), so the local volume change it introduces '
        'was not compensated for.'
    ),
}


def describe_jacobian_modulation():
    """Methods text: whether *QSIPrep* itself Jacobian-modulated a displacement field.

    Reads ``config.workflow.ignore`` directly (``--ignore jacobian``). This is
    display vocabulary describing what ``ComposeJacobianWeights`` did, not
    routing. Used by ``gradwarp_boilerplate`` for the gradient nonlinearity
    field.
    """
    return _JACOBIAN_SENTENCE['jacobian' not in (config.workflow.ignore or [])]
