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
import os.path as op
from collections import defaultdict

from .models import DWIGrouping, EstimationMethod, FieldmapEstimation, Provenance

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
    under the right output.
    """

    severity: str  # 'error' | 'warning'
    code: str  # stable machine-readable identifier
    message: str
    files: tuple[str, ...] = ()
    scope: str | None = None

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


def dwi_bidirectional_axes(grouping: DWIGrouping, estimation: FieldmapEstimation) -> frozenset:
    """Axes covered by *dMRI series* in both polarities among the sources.

    Unlike :attr:`FieldmapEstimation.bidirectional_axes` (which counts epi
    fieldmaps too), this answers "is there reverse phase-encoded dMRI data?" -
    the requirement for DRBUDDI to *refine* an eddy correction, since a lone
    reverse b=0 was already consumed by TOPUP and adds nothing more.
    """
    polarities = defaultdict(set)
    for path in estimation.sources:
        record = grouping.files.get(path)
        if record is not None and record.is_dwi and record.signature.pe_axis:
            polarities[record.signature.pe_axis].add(record.signature.pe_polarity)
    return frozenset(axis for axis, pols in polarities.items() if len(pols) == 2)


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
    concatenation_groups: dict,
    ignore_fov: bool = False,
) -> list[GroupingIssue]:
    """Backend-independent data checks on each output's member series.

    These read properties of the images themselves (b-values, NIfTI grids)
    and therefore run at grouping time, once, rather than per backend:

    - **Maximum b-value spread**: scanners often adjust acquisition
      parameters (TE, gradient timings) when the maximum b-value changes, so
      concatenating a b=1000 series with a b=3000 series deserves a warning.
    - **Field of view**: the series concatenated into one output must share a
      sampling grid. A pure translation offset is fixable by overwriting
      affines (warning, with shim evidence); differing orientations break
      axis-aligned distortion correction (error, downgradable with
      ``ignore_fov``); differing matrix/voxel sizes cannot be stacked at all
      (error, not downgradable).

    Series whose b-values or headers could not be read are skipped.
    """
    issues: list[GroupingIssue] = []

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
                    f"Series concatenated in output '{concat.output_name}' are "
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
                    f"Series concatenated in output '{concat.output_name}' have "
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
                    f"Series concatenated in output '{concat.output_name}' share "
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


def _check_shelling(grouping, backend, multipart_id, concat) -> list[GroupingIssue]:
    """Data-level shelled/non-shelled rules for one output.

    eddy models the signal on shells, so the FSL path (and the mixed path,
    whose first stage is eddy) requires every series in an output to be
    shelled. TORTOISE handles either sampling, but a mixture within one
    concatenated output deserves an informational note. Series whose b-values
    could not be read (``shelled is None``) are skipped.
    """
    issues = []
    shelled = [path for path in concat.dwi_files if grouping.files[path].shelled is True]
    non_shelled = [path for path in concat.dwi_files if grouping.files[path].shelled is False]

    if non_shelled and backend in ('fsl', 'mixed'):
        names = ', '.join(op.basename(path) for path in non_shelled)
        issues.append(
            error(
                'eddy-requires-shelled',
                f'eddy requires shelled (DTI/multi-shell) q-space sampling, but '
                f"{names} in output '{concat.output_name}' is not shelled. "
                'Use --hmc-model tortoise, which handles non-shelled data.',
                tuple(non_shelled),
                scope=multipart_id,
            )
        )

    if shelled and non_shelled and backend == 'tortoise':
        issues.append(
            warning(
                'mixed-shelled-nonshelled',
                f"Output '{concat.output_name}' concatenates shelled and "
                f'non-shelled series ({len(shelled)} shelled, '
                f'{len(non_shelled)} non-shelled). TORTOISE can process both, '
                'but consider whether these acquisitions belong in one output '
                '(MultipartID can separate them).',
                tuple(shelled + non_shelled),
                scope=multipart_id,
            )
        )

    return issues


def check_backend(grouping: DWIGrouping, backend: str) -> list[GroupingIssue]:
    """Validate a finished grouping against one processing backend.

    Returns issues only - never raises - so reports can show all three
    backends side by side.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {BACKENDS}.")

    issues = []
    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
        issues.extend(_check_shelling(grouping, backend, multipart_id, concat))
        estimations = {
            dgroup.b0field_source
            for dgroup in grouping.distortion_groups_in(multipart_id)
            if dgroup.b0field_source is not None
        }
        uncorrected = [
            dgroup.key
            for dgroup in grouping.distortion_groups_in(multipart_id)
            if dgroup.b0field_source is None
        ]

        if uncorrected and not estimations:
            # With the fieldmap-less fallback, an uncorrected group means the
            # subject has no usable T2w either (or the series has no PE info).
            issues.append(
                warning(
                    'no-sdc',
                    f"Output '{concat.output_name}' has no fieldmap and this subject "
                    'has no T2w image (or the series lacks PhaseEncodingDirection): '
                    'no susceptibility distortion correction will be performed.',
                    concat.dwi_files,
                    scope=multipart_id,
                )
            )

        for b0field_id in sorted(estimations):
            estimation = grouping.estimations[b0field_id]

            if estimation.method is EstimationMethod.ANAT_CONTRAST:
                # T2Wreg lives in TORTOISE's DIFFPREP; the FSL path (and the
                # fsl-based first stage of the mixed path) cannot reach it.
                if backend in ('fsl', 'mixed'):
                    demanded = estimation.provenance in (Provenance.FORCED, Provenance.CURATED)
                    make_issue = error if demanded else warning
                    issues.append(
                        make_issue(
                            'anat-sdc-unsupported',
                            f"Estimation '{b0field_id}' is a T2w registration "
                            f'(T2Wreg), which only the TORTOISE path implements. '
                            + (
                                'Use --hmc-model tortoise to run it.'
                                if demanded
                                else f"On this path '{concat.output_name}' gets no "
                                'susceptibility distortion correction.'
                            ),
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )
                continue

            if estimation.method is EstimationMethod.SYNB0:
                # The synthetic b=0 is a target image: TOPUP's missing blip on
                # the fsl path, the registration target for T2Wreg-style
                # stages on the tortoise and mixed paths. Every backend can
                # consume it; DRBUDDI's dual-blip refinement simply never runs
                # for these series (there is no reverse-PE dMRI data).
                continue

            if not estimation.is_pepolar:
                # GRE-style fieldmaps route to the classic fieldmap workflow on
                # every backend; the only note is that the mixed path's DRBUDDI
                # stage has nothing to refine.
                if backend == 'mixed':
                    issues.append(
                        warning(
                            'mixed-non-pepolar',
                            f"Estimation '{b0field_id}' is not PEPOLAR; the DRBUDDI "
                            f'second stage only refines PEPOLAR corrections, so '
                            f"'{concat.output_name}' will get single-stage "
                            'correction.',
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )
                continue

            if backend == 'fsl':
                if _pepolar_signature_count(grouping, estimation) < 2:
                    issues.append(
                        error(
                            'topup-single-signature',
                            f"Estimation '{b0field_id}' has only one distortion "
                            f'signature among its sources. TOPUP needs at least two '
                            f'(e.g. opposite phase encoding directions) to estimate '
                            f"a fieldmap for '{concat.output_name}'.",
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )
            elif backend == 'mixed' and not dwi_bidirectional_axes(grouping, estimation):
                # DRBUDDI refinement needs reverse phase-encoded dMRI *series*;
                # a lone reverse b=0 (epi fieldmap) was already consumed by
                # TOPUP, so a second DRBUDDI pass would reuse the same
                # information. Users cannot be stopped from requesting it, but
                # they should know it is probably not useful. The preview
                # instead assumes T2Wreg against a structural target, or
                # single-stage when there is none.
                fallback = (
                    'The second stage is T2Wreg against a structural image instead.'
                    if structural_target(grouping) is not None
                    else 'No T2w or SyNb0 synthetic b=0 is available for a T2Wreg '
                    f"second stage either, so '{concat.output_name}' gets "
                    'single-stage (TOPUP+eddy) correction.'
                )
                issues.append(
                    warning(
                        'drbuddi-refinement-not-useful',
                        f"Estimation '{b0field_id}' has no reverse phase-encoded "
                        'dMRI series: a DRBUDDI second stage would reuse the same '
                        'b=0 images TOPUP already consumed and is probably not '
                        f'useful. {fallback}',
                        estimation.sources,
                        scope=multipart_id,
                    )
                )
            else:  # tortoise, and mixed with reverse-PE dMRI: DRBUDDI runs
                if len(estimation.bidirectional_axes) == 0:
                    issues.append(
                        error(
                            'drbuddi-no-opposing-pair',
                            f"Estimation '{b0field_id}' has no axis with both "
                            f'phase encoding polarities. DRBUDDI requires an '
                            f'opposing (blip-up/blip-down) pair to correct '
                            f"'{concat.output_name}'.",
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )
                elif len(estimation.pe_axes) > 1:
                    issues.append(
                        error(
                            'drbuddi-cross-axis',
                            f"Estimation '{b0field_id}' includes sources on "
                            f'multiple phase encoding axes '
                            f'({", ".join(sorted(estimation.pe_axes))}). DRBUDDI '
                            f'estimates distortion along a single axis; split this '
                            f'estimation per axis (e.g. with per-axis '
                            f'B0FieldIdentifiers) to use it for '
                            f"'{concat.output_name}'.",
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )
                elif _pepolar_signature_count(grouping, estimation) > 2:
                    issues.append(
                        error(
                            'drbuddi-too-many-signatures',
                            f"Estimation '{b0field_id}' draws on more than two "
                            f'distortion signatures. DRBUDDI takes exactly one '
                            f'blip-up and one blip-down group; harmonize the '
                            f'acquisitions or curate per-pair B0FieldIdentifiers '
                            f"to correct '{concat.output_name}'.",
                            estimation.sources,
                            scope=multipart_id,
                        )
                    )

    return issues
