"""Issue collection and validation rules for DWI grouping.

Two layers of checks live here:

1. **Grouping-time rules** are applied while a :class:`~.models.DWIGrouping`
   is being built (called from ``inference.py``). They are about the data
   itself and hold no matter which processing software runs later.
2. **Backend feasibility checks** (:func:`check_backend`) are pure functions
   over a finished grouping that answer "could the fsl / tortoise / mixed
   pipeline actually process this?". They keep all tool-specific knowledge
   out of the grouping model.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from .models import DWIGrouping, FieldmapEstimation

#: Backends a grouping can be previewed/validated against.
BACKENDS = ('fsl', 'tortoise', 'mixed')

BACKEND_DESCRIPTIONS = {
    'fsl': 'FSL path (TOPUP fieldmap estimation + eddy HMC/SDC)',
    'tortoise': 'TORTOISE path (DIFFPREP HMC + DRBUDDI SDC)',
    'mixed': 'Two-stage path (TOPUP + eddy, then DRBUDDI or T2Wreg refinement)',
}


class GroupingError(RuntimeError):
    """Raised when the grouping contains error-severity issues."""


@dataclasses.dataclass(frozen=True)
class GroupingIssue:
    """One problem or notable decision discovered while grouping.

    ``scope`` names the concatenation group (MultipartID) an issue belongs
    to, when it belongs to exactly one; reports use it to show each issue
    under the right output. ``run`` names the execution-plan processing run
    an issue is about, when it is about exactly one.
    """

    severity: str  # 'error' | 'warning'
    code: str  # stable machine-readable identifier
    message: str
    files: tuple[str, ...] = ()
    scope: str | None = None
    run: str | None = None

    def render(self) -> str:
        return f'{self.severity.upper()} [{self.code}]: {self.message}'


def error(code: str, message: str, files=(), scope=None) -> GroupingIssue:
    return GroupingIssue('error', code, message, tuple(files), scope)


def warning(code: str, message: str, files=(), scope=None) -> GroupingIssue:
    return GroupingIssue('warning', code, message, tuple(files), scope)


def raise_for_errors(grouping: DWIGrouping):
    """Raise :class:`GroupingError` if any error-severity issue was collected."""
    errors = grouping.errors
    if errors:
        rendered = '\n'.join(issue.render() for issue in errors)
        raise GroupingError(
            f'The DWI grouping for sub-{grouping.subject_id} has '
            f'{len(errors)} unresolvable problem(s):\n{rendered}'
        )


def _pepolar_signature_count(grouping: DWIGrouping, estimation: FieldmapEstimation) -> int:
    """Number of distinct distortion signatures among an estimation's EPI sources."""
    signatures = set()
    for path in estimation.sources:
        record = grouping.files.get(path)
        if record is not None and record.is_epi_like and record.signature.pe_dir:
            signatures.add(record.signature.key)
    return len(signatures)


def blip_pair_polarities(grouping: DWIGrouping, estimation: FieldmapEstimation) -> dict:
    """Blip-pair identity ``(pe_axis, readout_time, shim)`` -> polarities present.

    DRBUDDI corrects one matched blip-up/blip-down pair at a time: two EPI-like
    sources that share this identity (same axis, readout time and shim) and carry
    opposite polarity. A group with both polarities is a complete pair; one
    polarity is unpaired. (TOPUP, by contrast, pools every group into a single
    estimation regardless of readout.)
    """
    polarities: dict[tuple, set] = defaultdict(set)
    for path in estimation.sources:
        record = grouping.files.get(path)
        if record is not None and record.is_epi_like and record.signature.pe_dir:
            sig = record.signature
            polarities[(sig.pe_axis, sig.readout_time, sig.shim)].add(sig.pe_polarity)
    return dict(polarities)


def blip_sort_key(key: tuple) -> tuple:
    """Deterministic ordering for blip-pair keys (readout/shim may be None)."""
    axis, readout, shim = key
    return (axis, float('-inf') if readout is None else readout, str(shim))


def describe_blip_group(key: tuple) -> str:
    """Human-readable label for a blip-pair identity, e.g. 'axis j, TRT 0.05s'."""
    axis, readout, _shim = key
    return f'axis {axis}' + (f', TRT {readout:g}s' if readout is not None else '')


def dwi_blip_pairs(grouping: DWIGrouping, estimation: FieldmapEstimation) -> list:
    """Blip-pair identities with reverse phase-encoded *dMRI series* in both
    polarities among the sources - the matched pairs DRBUDDI can *refine*.

    Unlike :func:`blip_pair_polarities` (which counts epi fieldmaps too), a lone
    reverse b=0 was already consumed by TOPUP and adds nothing more; only a
    reverse-PE dMRI pair carries new information. Keys on the full blip-pair
    identity (axis, readout time and shim), since DRBUDDI needs a readout match.
    Sorted.
    """
    polarities: dict[tuple, set] = defaultdict(set)
    for path in estimation.sources:
        record = grouping.files.get(path)
        if record is not None and record.is_dwi and record.signature.pe_dir:
            sig = record.signature
            polarities[(sig.pe_axis, sig.readout_time, sig.shim)].add(sig.pe_polarity)
    return sorted((key for key, pols in polarities.items() if len(pols) == 2), key=blip_sort_key)


def structural_target(grouping: DWIGrouping) -> tuple[str, list[str]] | None:
    """The structural image registration-based stages should use.

    Returns ``(kind, paths)`` where kind is ``'synb0'`` (a synthetic
    undistorted b=0 from the T1w - preferred when SyNb0 was requested, even
    over a real T2w, since its contrast matches the b=0 exactly) or ``'t2w'``;
    ``None`` when neither is available.
    """
    t1ws = grouping.anat_files('T1w')
    if grouping.synb0_requested and t1ws:
        return 'synb0', t1ws
    t2ws = grouping.anat_files('T2w')
    if t2ws:
        return 't2w', t2ws
    return None


#: Maximum b-values within one output may differ by this much before warning.
MAXB_TOLERANCE = 100.0


def check_data_compatibility(
    records: dict,
    correction_units: dict,
    concatenation_groups: dict,
    ignore_fov: bool = False,
) -> list[GroupingIssue]:
    """Backend-independent data checks on the grouped series.

    These read properties of the images themselves (b-values, NIfTI grids)
    and therefore run at grouping time, once, rather than per backend:

    - **Maximum b-value spread** (per final output): scanners often adjust
      acquisition parameters (TE, gradient timings) when the maximum b-value
      changes, so concatenating a b=1000 series with a b=3000 series
      deserves a warning.
    - **Field of view** (per correction unit - raw series are only ever
      stacked within a unit; final concatenation happens after resampling):
      a pure translation offset is fixable by overwriting affines (warning,
      with shim evidence); differing orientations break axis-aligned
      distortion correction (error, downgradable with ``ignore_fov``);
      differing matrix/voxel sizes cannot be stacked at all (error, not
      downgradable).

    Series whose b-values or headers could not be read are skipped.
    """
    issues: list[GroupingIssue] = []

    unit_scope = {
        unit_key: multipart_id
        for multipart_id, concat in concatenation_groups.items()
        for unit_key in concat.correction_units
    }

    for multipart_id, concat in sorted(concatenation_groups.items()):
        members = [records[path] for path in concat.dwi_files if path in records]

        # --- maximum b-value spread ---------------------------------------
        max_bvals = {
            record.filename: record.max_bval for record in members if record.max_bval is not None
        }
        if max_bvals and max(max_bvals.values()) - min(max_bvals.values()) > MAXB_TOLERANCE:
            described = ', '.join(
                f'{name}: b={int(val)}' for name, val in sorted(max_bvals.items())
            )
            issues.append(
                warning(
                    'maxb-mismatch',
                    f"Series concatenated in output '{concat.output_name}' have "
                    f'different maximum b-values ({described}). Many scanners '
                    'adjust acquisition parameters (TE, gradient timings) when '
                    'the maximum b-value changes, so these series may differ in '
                    'more than diffusion weighting. If they should be processed '
                    'separately, give them different MultipartID values.',
                    tuple(record.path for record in members),
                    scope=multipart_id,
                )
            )

    for unit_key, unit in sorted(correction_units.items()):
        multipart_id = unit_scope.get(unit_key)
        members = [records[path] for path in unit.dwi_files if path in records]

        # --- field of view -------------------------------------------------
        gridded = [record for record in members if record.grid is not None]
        if len(gridded) < 2:
            continue
        reference = gridded[0]
        worst = {'grid': [], 'oblique': [], 'shifted': []}
        max_shift = 0.0
        max_rotation = 0.0
        for record in gridded[1:]:
            relation = reference.grid.compare(record.grid)
            if relation == 'match':
                continue
            worst[relation].append(record)
            if relation == 'shifted':
                max_shift = max(max_shift, reference.grid.shift_mm(record.grid))
            if relation == 'oblique':
                max_rotation = max(max_rotation, reference.grid.rotation_deg(record.grid))

        involved = tuple(record.path for record in gridded)
        separate_advice = (
            'process them as separate outputs by giving them different '
            'MultipartID values or using --separate-all-dwis'
        )

        if worst['grid']:
            names = ', '.join(record.filename for record in [reference] + worst['grid'])
            issues.append(
                error(
                    'fov-grid-mismatch',
                    f"Series stacked in correction unit '{unit.key}' are "
                    f'sampled on different voxel grids ({names}: matrix size or '
                    'voxel size differs). They cannot be stacked volumewise. '
                    f'Either {separate_advice}, or resample them to a common '
                    'grid before running qsiprep.',
                    involved,
                    scope=multipart_id,
                )
            )
        elif worst['oblique']:
            make_issue = warning if ignore_fov else error
            proceed = (
                'Proceeding anyway because field-of-view checking is disabled: '
                'expect distortion corrections to be misapplied.'
                if ignore_fov
                else 'To proceed anyway, accepting misapplied corrections, '
                'disable field-of-view checking (ignore_fov).'
            )
            issues.append(
                make_issue(
                    'fov-oblique',
                    f"Series stacked in correction unit '{unit.key}' have "
                    f'differently-oriented fields of view (slice orientations '
                    f'differ by up to {max_rotation:.1f} degrees). Susceptibility '
                    'and eddy-current distortions act along the acquisition axes, '
                    'so corrections cannot be applied correctly to a naive '
                    f'concatenation. Either {separate_advice}. {proceed}',
                    involved,
                    scope=multipart_id,
                )
            )
        elif worst['shifted']:
            shims = {record.signature.shim for record in gridded}
            if None in shims or () in shims:
                shim_evidence = (
                    'No ShimSetting is recorded in the sidecars, so whether a '
                    're-shim occurred cannot be verified.'
                )
            elif len(shims) == 1:
                shim_evidence = (
                    'The recorded ShimSetting values match, so a re-shim does '
                    'not appear to have occurred and aligning them is safe.'
                )
            else:
                shim_evidence = (
                    'The recorded ShimSetting values differ, confirming a '
                    're-shim: these series do NOT share susceptibility '
                    'distortions.'
                )
            issues.append(
                warning(
                    'fov-shifted',
                    f"Series stacked in correction unit '{unit.key}' share "
                    f'a grid but their fields of view are offset by up to '
                    f'{max_shift:.1f} mm. The affines can be overwritten to align '
                    'them, but many scanners force a re-shim when the field of '
                    'view is moved, in which case the series no longer share '
                    f'susceptibility distortions. {shim_evidence} To keep them '
                    f'apart instead, {separate_advice}.',
                    involved,
                    scope=multipart_id,
                )
            )

    return issues


def check_backend(grouping: DWIGrouping, backend: str) -> list[GroupingIssue]:
    """Validate a finished grouping against one processing backend.

    Returns issues only - never raises - so reports can show all three
    backends side by side. The rules live in the plan compiler
    (:func:`~.plan.compile_plan`); this returns the compiled plan's issues
    for the backend's canonical method selection.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {BACKENDS}.")

    from .methods import canonical_selection
    from .plan import compile_plan

    return list(compile_plan(grouping, canonical_selection(backend)).issues)
