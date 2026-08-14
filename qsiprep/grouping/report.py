"""Human-readable renderings of a DWIGrouping.

Two views are provided:

- :func:`report_text` describes the grouping itself: which files were grouped
  with which, and whether each decision was curated, translated from
  IntendedFor, or inferred.
- :func:`describe_processing` previews what a processing backend (``fsl``,
  ``tortoise``, or ``mixed``) would actually do with the grouping, including
  any errors that backend would raise. The grouping model itself knows
  nothing about backends; all tool knowledge lives here and in
  :func:`~.validation.check_backend`.
"""

from __future__ import annotations

import os.path as op
from collections import defaultdict

from .models import DWIGrouping, EstimationMethod, FieldmapEstimation
from .validation import BACKEND_DESCRIPTIONS, check_backend

_METHOD_LABELS = {
    EstimationMethod.PEPOLAR: 'PEPOLAR (reverse phase-encoding)',
    EstimationMethod.DIRECT: 'precomputed fieldmap',
    EstimationMethod.PHASEDIFF: 'GRE phase difference',
    EstimationMethod.PHASES: 'GRE two-phase',
    EstimationMethod.ANAT_CONTRAST: 'anatomical registration',
}


def _basename(path: str) -> str:
    return op.basename(path)


def report_text(grouping: DWIGrouping) -> str:
    """The grouping decisions, with provenance, as printable text."""
    lines = []
    title = f'DWI grouping for sub-{grouping.subject_id}'
    lines.append(title)
    lines.append('=' * len(title))
    lines.append('')

    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
        lines.append(
            f'Output "{concat.output_name}" '
            f'(MultipartID {multipart_id} {concat.provenance.tag()}): '
            f'{len(concat.dwi_files)} series'
        )
        for dgroup in grouping.distortion_groups_in(multipart_id):
            lines.append(f'  Distortion group {dgroup.key} ({dgroup.signature.describe()}):')
            for path in dgroup.dwi_files:
                lines.append(f'    - {_basename(path)}')
            if dgroup.b0field_source is None:
                lines.append('    corrected by: nothing (no fieldmap found)')
            else:
                provenance = grouping.application_provenance[dgroup.dwi_files[0]]
                lines.append(f'    corrected by: {dgroup.b0field_source} {provenance.tag()}')
                losing = [
                    candidate
                    for candidate in grouping.application_candidates.get(dgroup.dwi_files[0], ())
                    if candidate != dgroup.b0field_source
                ]
                if losing:
                    lines.append(f'    (also eligible: {", ".join(losing)})')
        borrowed = grouping.borrowed_sources(multipart_id)
        for b0field_id, paths in sorted(borrowed.items()):
            lines.append(
                f'  Borrows for fieldmap estimation ({b0field_id}), not included in this output:'
            )
            for path in paths:
                lines.append(f'    - {_basename(path)}')
        lines.append('')

    if grouping.estimations:
        lines.append('Fieldmap estimations:')
        for b0field_id, estimation in sorted(grouping.estimations.items()):
            lines.append(
                f'  {b0field_id} {estimation.provenance.tag()}: '
                f'{_METHOD_LABELS[estimation.method]}'
            )
            for path in estimation.sources:
                lines.append(f'    - {_basename(path)}')
            if estimation.pe_axes:
                axes = ', '.join(
                    f'{axis} (bidirectional)'
                    if axis in estimation.bidirectional_axes
                    else f'{axis} (one direction only)'
                    for axis in sorted(estimation.pe_axes)
                )
                lines.append(f'    phase encoding axes: {axes}')
        lines.append('')
    else:
        lines.append('Fieldmap estimations: none found.')
        lines.append('')

    if grouping.issues:
        lines.append('Notes:')
        for issue in grouping.issues:
            lines.append(f'  {issue.render()}')
        lines.append('')

    return '\n'.join(lines)


def _group_estimations(grouping: DWIGrouping, multipart_id: str) -> dict[str, list[str]]:
    """estimation id -> keys of member distortion groups it corrects."""
    corrected = defaultdict(list)
    for dgroup in grouping.distortion_groups_in(multipart_id):
        if dgroup.b0field_source is not None:
            corrected[dgroup.b0field_source].append(dgroup.key)
    return corrected


def _pepolar_signature_count(grouping: DWIGrouping, estimation: FieldmapEstimation) -> int:
    signatures = set()
    for path in estimation.sources:
        record = grouping.files.get(path)
        if record is not None and record.is_epi_like and record.signature.pe_dir:
            signatures.add(record.signature.key)
    return len(signatures)


def _split_polarities(grouping: DWIGrouping, multipart_id: str, axis: str):
    """Member distortion groups on ``axis``, split into (blip-up, blip-down)."""
    up, down = [], []
    for dgroup in grouping.distortion_groups_in(multipart_id):
        if dgroup.signature.pe_axis != axis:
            continue
        (up if dgroup.signature.pe_polarity == 1 else down).append(dgroup)
    return up, down


def describe_processing(grouping: DWIGrouping, backend: str) -> str:
    """Preview what ``backend`` would do with this grouping.

    ``backend`` is ``'fsl'`` (TOPUP+eddy), ``'tortoise'`` (DIFFPREP+DRBUDDI),
    or ``'mixed'`` (TOPUP+eddy then DRBUDDI).
    """
    backend_issues = check_backend(grouping, backend)

    lines = []
    title = f'Processing preview: {BACKEND_DESCRIPTIONS[backend]}'
    lines.append(title)
    lines.append('-' * len(title))
    lines.append('')

    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
        n_series = len(concat.dwi_files)
        plural = 'series' if n_series != 1 else 'single series'
        lines.append(f'Output "{concat.output_name}" ({n_series} {plural}):')

        # --- Pre-HMC stage (identical across backends) -------------------
        if n_series > 1:
            lines.append(
                '  1. Each series is denoised on its own, then all '
                f'{n_series} series are concatenated. '
                '(--denoise-after-combining reverses this order.)'
            )
        else:
            lines.append('  1. The series is denoised.')

        corrected = _group_estimations(grouping, multipart_id)
        group_errors = [issue for issue in backend_issues if issue.scope in (None, multipart_id)]

        if backend == 'fsl':
            _describe_fsl(lines, grouping, multipart_id, corrected)
        elif backend == 'tortoise':
            _describe_tortoise(lines, grouping, multipart_id, corrected)
        else:
            _describe_mixed(lines, grouping, multipart_id, corrected)

        for issue in group_errors:
            lines.append(f'  !! {issue.render()}')
        lines.append('')

    return '\n'.join(lines)


def _borrow_note(grouping: DWIGrouping, multipart_id: str, b0field_id: str) -> str | None:
    borrowed = grouping.borrowed_sources(multipart_id).get(b0field_id)
    if not borrowed:
        return None
    names = ', '.join(_basename(path) for path in borrowed)
    return (
        f'b=0 images are borrowed from {names} for the estimation; '
        'those series are NOT included in this output.'
    )


def _describe_fsl(lines, grouping, multipart_id, corrected):
    step = 2
    pepolar_ids = [
        b0field_id
        for b0field_id in sorted(corrected)
        if grouping.estimations[b0field_id].is_pepolar
    ]
    gre_ids = [
        b0field_id
        for b0field_id in sorted(corrected)
        if not grouping.estimations[b0field_id].is_pepolar
    ]

    if pepolar_ids:
        all_sources = set()
        signatures = set()
        for b0field_id in pepolar_ids:
            estimation = grouping.estimations[b0field_id]
            all_sources.update(estimation.sources)
            signatures.update(
                grouping.files[path].signature.key
                for path in estimation.sources
                if path in grouping.files and grouping.files[path].is_epi_like
            )
        fmap_only = sorted(
            path
            for path in all_sources
            if path in grouping.files and grouping.files[path].datatype == 'fmap'
        )
        lines.append(
            f'  {step}. TOPUP estimates the susceptibility field from b=0 images '
            f'spanning {len(signatures)} distortion groups '
            f'({len(signatures)} rows in the acquisition-parameters file).'
        )
        if fmap_only:
            lines.append(
                f'     Extra b=0 images from fmap/: '
                f'{", ".join(_basename(path) for path in fmap_only)}.'
            )
        for b0field_id in pepolar_ids:
            note = _borrow_note(grouping, multipart_id, b0field_id)
            if note:
                lines.append(f'     {note}')
        if len(pepolar_ids) > 1:
            lines.append(
                f'     Note: {len(pepolar_ids)} separate fieldmap estimations feed '
                'this output; the FSL path estimates one combined field from all '
                'of their b=0 images.'
            )
        step += 1
        lines.append(
            f'  {step}. eddy corrects head motion, eddy currents, and susceptibility '
            'distortion in one model, using the TOPUP field. Every volume is '
            'assigned to its distortion group in the eddy index file.'
        )
        step += 1
    elif gre_ids:
        for b0field_id in gre_ids:
            estimation = grouping.estimations[b0field_id]
            lines.append(
                f'  {step}. A fieldmap is computed from the '
                f'{_METHOD_LABELS[estimation.method]} acquisition '
                f'({", ".join(_basename(path) for path in estimation.sources)}).'
            )
            step += 1
        lines.append(
            f'  {step}. eddy corrects head motion and eddy currents; the fieldmap '
            'is applied to unwarp the results.'
        )
        step += 1
    else:
        lines.append(
            f'  {step}. eddy corrects head motion and eddy currents. No fieldmap is '
            'available: susceptibility distortion is NOT corrected.'
        )
        step += 1

    lines.append(f'  {step}. The corrected series is written as one output file.')


def _describe_tortoise(lines, grouping, multipart_id, corrected):
    step = 2
    member_dgroups = grouping.distortion_groups_in(multipart_id)
    pepolar_ids = [
        b0field_id
        for b0field_id in sorted(corrected)
        if grouping.estimations[b0field_id].is_pepolar
    ]
    gre_ids = [
        b0field_id
        for b0field_id in sorted(corrected)
        if not grouping.estimations[b0field_id].is_pepolar
    ]

    if len(member_dgroups) > 1:
        keys = ', '.join(dgroup.key for dgroup in member_dgroups)
        lines.append(
            f'  {step}. DIFFPREP corrects head motion and eddy currents separately '
            f'for each distortion group ({keys}).'
        )
    else:
        lines.append(f'  {step}. DIFFPREP corrects head motion and eddy currents.')
    step += 1

    if pepolar_ids:
        for b0field_id in pepolar_ids:
            estimation = grouping.estimations[b0field_id]
            for axis in sorted(estimation.bidirectional_axes):
                up, down = _split_polarities(grouping, multipart_id, axis)
                up_names = ', '.join(dgroup.key for dgroup in up) or 'borrowed series'
                down_names = ', '.join(dgroup.key for dgroup in down) or 'borrowed series'
                lines.append(
                    f'  {step}. DRBUDDI estimates distortion along the {axis} axis '
                    f'from the blip-up ({up_names}) and blip-down ({down_names}) '
                    'data and applies the correction to every volume.'
                )
                step += 1
            note = _borrow_note(grouping, multipart_id, b0field_id)
            if note:
                lines.append(f'     {note}')
    elif gre_ids:
        for b0field_id in gre_ids:
            estimation = grouping.estimations[b0field_id]
            lines.append(
                f'  {step}. A fieldmap is computed from the '
                f'{_METHOD_LABELS[estimation.method]} acquisition and applied to '
                'unwarp the DIFFPREP output.'
            )
            step += 1
    else:
        lines.append(
            f'  {step}. No fieldmap is available. If a T2w image was acquired, '
            'DRBUDDI registers the b=0 image to it to estimate distortion '
            '(T2Wreg); otherwise distortion is NOT corrected.'
        )
        step += 1

    lines.append(f'  {step}. The corrected series is written as one output file.')


def _describe_mixed(lines, grouping, multipart_id, corrected):
    step = 2
    pepolar_ids = [
        b0field_id
        for b0field_id in sorted(corrected)
        if grouping.estimations[b0field_id].is_pepolar
    ]

    if pepolar_ids:
        lines.append(
            f'  {step}. TOPUP estimates an initial susceptibility field from the '
            'b=0 images (as in the FSL path).'
        )
        step += 1
        lines.append(
            f'  {step}. eddy corrects head motion and eddy currents using the TOPUP field.'
        )
        step += 1
        for b0field_id in pepolar_ids:
            estimation = grouping.estimations[b0field_id]
            for axis in sorted(estimation.bidirectional_axes):
                lines.append(
                    f'  {step}. DRBUDDI re-estimates distortion along the {axis} axis '
                    'from the eddy-corrected blip-up/blip-down data, refining the '
                    'TOPUP correction.'
                )
                step += 1
            note = _borrow_note(grouping, multipart_id, b0field_id)
            if note:
                lines.append(f'     {note}')
    else:
        lines.append(
            f'  {step}. Without a PEPOLAR fieldmap the DRBUDDI second stage adds '
            'nothing: processing follows the FSL path.'
        )
        step += 1

    lines.append(f'  {step}. The corrected series is written as one output file.')


def full_report(grouping: DWIGrouping) -> str:
    """The grouping report followed by all three backend previews."""
    sections = [report_text(grouping)]
    for backend in ('fsl', 'tortoise', 'mixed'):
        sections.append(describe_processing(grouping, backend))
    return '\n'.join(sections)
