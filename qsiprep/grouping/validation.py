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

from .models import DWIGrouping, FieldmapEstimation

#: Backends a grouping can be previewed/validated against.
BACKENDS = ('fsl', 'tortoise', 'mixed')

BACKEND_DESCRIPTIONS = {
    'fsl': 'FSL path (TOPUP fieldmap estimation + eddy HMC/SDC)',
    'tortoise': 'TORTOISE path (DIFFPREP HMC + DRBUDDI SDC)',
    'mixed': 'Two-stage path (TOPUP + eddy, then DRBUDDI refinement)',
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


def check_backend(grouping: DWIGrouping, backend: str) -> list[GroupingIssue]:
    """Validate a finished grouping against one processing backend.

    Returns issues only - never raises - so reports can show all three
    backends side by side.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {BACKENDS}.")

    issues = []
    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
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
            issues.append(
                warning(
                    'no-sdc',
                    f"Output '{concat.output_name}' has no fieldmap. "
                    + (
                        'The TORTOISE path can register the b=0 to a T2w image (T2Wreg) '
                        'if one was acquired; otherwise no '
                        if backend != 'fsl'
                        else 'No '
                    )
                    + 'susceptibility distortion correction will be performed.',
                    concat.dwi_files,
                    scope=multipart_id,
                )
            )

        for b0field_id in sorted(estimations):
            estimation = grouping.estimations[b0field_id]
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
            else:  # tortoise and mixed both end in DRBUDDI
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
