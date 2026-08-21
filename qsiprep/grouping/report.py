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
import re
from collections import Counter, defaultdict
from typing import NamedTuple

from .models import CorrectionMethod, DWIGrouping
from .validation import (
    BACKEND_DESCRIPTIONS,
    blip_pair_polarities,
    blip_sort_key,
    check_backend,
    describe_blip_group,
    dwi_blip_pairs,
    structural_target,
)

_METHOD_LABELS = {
    CorrectionMethod.PEPOLAR: 'PEPOLAR (reverse phase-encoding)',
    CorrectionMethod.DIRECT: 'precomputed fieldmap',
    CorrectionMethod.PHASEDIFF: 'GRE phase difference',
    CorrectionMethod.PHASES: 'GRE two-phase',
    CorrectionMethod.SYNB0: 'SyNb0 synthetic b=0',
    CorrectionMethod.T2WREG: 'T2w registration (T2Wreg)',
    CorrectionMethod.NIPREPS_SYN: 'fieldmap-less SyN',
}


def _basename(path: str) -> str:
    return op.basename(path)


def _shell_tag(record) -> str:
    """Bracketed sampling-scheme annotation for a DWI file line."""
    if record.shelled is True:
        shells = '/'.join(str(int(centre)) for centre in record.shells)
        return f' [shelled: b={shells}]'
    if record.shelled is False:
        return ' [non-shelled q-space sampling]'
    return ''


def report_text(grouping: DWIGrouping) -> str:
    """The grouping decisions, with provenance, as printable text."""
    lines = []
    title = f'DWI grouping for sub-{grouping.subject_id}'
    lines.append(title)
    lines.append('=' * len(title))
    lines.append('')

    for concat_key, concat in sorted(grouping.concatenation_groups.items()):
        lines.append(
            f'Output "{concat.output_name}" '
            f'(MultipartID {concat.multipart_id} {concat.provenance.tag()}): '
            f'{len(concat.dwi_files)} series'
        )
        for dgroup in grouping.distortion_groups_in(concat_key):
            lines.append(f'  Distortion group {dgroup.key} ({dgroup.signature.describe()}):')
            for path in dgroup.dwi_files:
                lines.append(f'    - {_basename(path)}{_shell_tag(grouping.files[path])}')
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
        borrowed = grouping.borrowed_sources(concat_key)
        for b0field_id, paths in sorted(borrowed.items()):
            lines.append(
                f'  Borrows for fieldmap estimation ({b0field_id}), not included in this output:'
            )
            for path in paths:
                lines.append(f'    - {_basename(path)}')
        lines.append('')

    if grouping.estimations:
        applied_ids = {b0field_id for b0field_id in grouping.application.values() if b0field_id}
        lines.append('Fieldmap estimations:')
        for b0field_id, estimation in sorted(grouping.estimations.items()):
            unused = '' if b0field_id in applied_ids else ' (not used)'
            lines.append(
                f'  {b0field_id} {estimation.provenance.tag()}: '
                f'{_METHOD_LABELS[estimation.method]}{unused}'
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


def _unit_corrected(grouping: DWIGrouping, unit) -> dict[str, list[str]]:
    """estimation id -> keys of the unit's distortion groups it corrects."""
    corrected = defaultdict(list)
    for key in unit.distortion_groups:
        dgroup = grouping.distortion_groups[key]
        if dgroup.b0field_source is not None:
            corrected[dgroup.b0field_source].append(key)
    return corrected


def _split_polarities(dgroups, pair_key: tuple):
    """Distortion groups in one blip pair (pe_axis, readout, shim), split into
    (blip-up, blip-down)."""
    up, down = [], []
    for dgroup in dgroups:
        sig = dgroup.signature
        if (sig.pe_axis, sig.readout_time, sig.shim) != pair_key:
            continue
        (up if sig.pe_polarity == 1 else down).append(dgroup)
    return up, down


def _describe_unit(lines, grouping, backend, multipart_id, unit, step) -> int:
    """One correction unit's HMC+SDC steps; returns the next step number."""
    corrected = _unit_corrected(grouping, unit)
    dgroups = [grouping.distortion_groups[key] for key in unit.distortion_groups]
    if backend == 'fsl':
        return _describe_fsl(lines, grouping, multipart_id, corrected, dgroups, step)
    if backend == 'tortoise':
        return _describe_tortoise(lines, grouping, multipart_id, corrected, dgroups, step)
    return _describe_mixed(lines, grouping, multipart_id, corrected, dgroups, step)


def _output_step_lines(grouping, backend, multipart_id, backend_issues) -> list[str]:
    """The numbered step lines for one output, as printed by describe_processing."""
    concat = grouping.concatenation_groups[multipart_id]
    units = grouping.correction_units_in(multipart_id)
    n_series = len(concat.dwi_files)
    lines = []

    if len(units) == 1:
        if n_series > 1:
            lines.append(
                '  1. Each series is denoised on its own, then all '
                f'{n_series} series are concatenated. '
                '(--denoise-after-combining reverses this order.)'
            )
        else:
            lines.append('  1. The series is denoised.')
        step = _describe_unit(lines, grouping, backend, multipart_id, units[0], step=2)
        lines.append(f'  {step}. The corrected series is written as one output file.')
    else:
        lines.append(
            f'  1. Each series is denoised on its own. The {n_series} series span '
            f'{len(units)} correction units; each unit is concatenated and '
            'corrected in its own pipeline.'
        )
        step = 2
        for unit in units:
            names = ', '.join(_basename(path) for path in unit.dwi_files)
            lines.append(f"  {step}. Correction unit '{unit.key}' ({names}):")
            step = _describe_unit(lines, grouping, backend, multipart_id, unit, step + 1)
        lines.append(
            f'  {step}. The corrected results of the {len(units)} units are '
            'resampled onto one grid and concatenated into one output file.'
        )

    for issue in backend_issues:
        if issue.scope in (None, multipart_id):
            lines.append(f'  !! {issue.render()}')
    return lines


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
        lines.extend(_output_step_lines(grouping, backend, multipart_id, backend_issues))
        lines.append('')

    return '\n'.join(lines)


def processing_steps(grouping: DWIGrouping, backend: str) -> dict[str, list[str]]:
    """The describe_processing steps, structured for non-text renderings.

    Returns ``{output_name: [step, ...]}`` where each step is the text of one
    numbered line with its continuation notes folded in. Backend feasibility
    issues keep their leading ``'!! '`` marker so renderers can style them.
    """
    backend_issues = check_backend(grouping, backend)
    result = {}
    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
        steps = []
        for line in _output_step_lines(grouping, backend, multipart_id, backend_issues):
            text = line.strip()
            numbered = re.match(r'\d+\.\s+(.*)', text)
            if numbered:
                steps.append(numbered.group(1))
            elif text.startswith('!!') or not steps:
                steps.append(text)
            else:
                steps[-1] += ' ' + text
        result[concat.output_name] = steps
    return result


def _borrow_note(grouping: DWIGrouping, multipart_id: str, b0field_id: str) -> str | None:
    borrowed = grouping.borrowed_sources(multipart_id).get(b0field_id)
    if not borrowed:
        return None
    names = ', '.join(_basename(path) for path in borrowed)
    return (
        f'b=0 images are borrowed from {names} for the estimation; '
        'those series are NOT included in this output.'
    )


class MethodGroups(NamedTuple):
    """A group's applied estimation ids, split by how backends treat them."""

    pepolar: list[str]
    gre: list[str]
    synb0: list[str]
    anat: list[str]
    syn: list[str]


def _ids_by_kind(grouping, corrected) -> MethodGroups:
    kinds = MethodGroups([], [], [], [], [])
    for b0field_id in sorted(corrected):
        method = grouping.estimations[b0field_id].method
        if method is CorrectionMethod.PEPOLAR:
            kinds.pepolar.append(b0field_id)
        elif method is CorrectionMethod.SYNB0:
            kinds.synb0.append(b0field_id)
        elif method is CorrectionMethod.T2WREG:
            kinds.anat.append(b0field_id)
        elif method is CorrectionMethod.NIPREPS_SYN:
            kinds.syn.append(b0field_id)
        else:
            kinds.gre.append(b0field_id)
    return kinds


def _no_sdc_sentence(grouping) -> str:
    """Definite no-correction wording: the grouping knows the T2w inventory."""
    if grouping.anat_files('T2w'):
        # A T2w exists but no estimation was applied: only possible when the
        # series has no usable PE information.
        return (
            'No fieldmap could be set up (missing phase encoding '
            'information): susceptibility distortion is NOT corrected.'
        )
    return (
        'No fieldmap is available and this subject has no T2w image: '
        'susceptibility distortion is NOT corrected.'
    )


def _structural_phrase(grouping) -> str | None:
    """Name the structural target registration-based stages would use."""
    target = structural_target(grouping)
    if target is None:
        return None
    kind, paths = target
    names = ', '.join(_basename(path) for path in paths)
    if kind == 'synb0':
        override = ', in place of the T2w image' if grouping.anat_files('T2w') else ''
        return f'a SyNb0 synthetic b=0 (from {names}{override})'
    return f'the T2w image ({names})'


def _structural_note(grouping) -> str | None:
    """DRBUDDI's multimodal --structural line, when a target exists."""
    phrase = _structural_phrase(grouping)
    if phrase is None:
        return None
    return (
        f'     DRBUDDI additionally uses {phrase} as a structural '
        'registration target (multimodal correction).'
    )


def _describe_fsl(lines, grouping, multipart_id, corrected, dgroups, step):
    kinds = _ids_by_kind(grouping, corrected)
    pepolar_ids = kinds.pepolar
    gre_ids = kinds.gre

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
        step += 1
        lines.append(
            f'  {step}. eddy corrects head motion, eddy currents, and susceptibility '
            'distortion in one model, using the TOPUP field. Every volume is '
            'assigned to its distortion group in the eddy index file.'
        )
        step += 1
    elif kinds.synb0:
        for b0field_id in kinds.synb0:
            estimation = grouping.estimations[b0field_id]
            t1w_names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. SyNb0 synthesizes an undistorted b=0 image from the T1w ({t1w_names}).'
            )
            step += 1
        lines.append(
            f'  {step}. TOPUP estimates the susceptibility field from the acquired '
            'b=0 images plus the synthetic b=0, which enters as a zero-readout-time '
            'volume (an extra row in the acquisition-parameters file).'
        )
        step += 1
        lines.append(
            f'  {step}. eddy corrects head motion, eddy currents, and susceptibility '
            'distortion in one model, using the TOPUP field.'
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
    elif kinds.anat:
        lines.append(
            f'  {step}. eddy corrects head motion and eddy currents. The selected '
            'correction is a T2w registration (T2Wreg), which only the TORTOISE '
            'path implements: distortion is NOT corrected on this path.'
        )
        step += 1
    elif kinds.syn:
        for b0field_id in kinds.syn:
            estimation = grouping.estimations[b0field_id]
            names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. eddy corrects head motion and eddy currents; a standalone '
                'fieldmap-less SyN warp (constrained ANTs registration of the T1w to a '
                f'fieldmap atlas: {names}) is applied to unwarp the results.'
            )
            step += 1
    else:
        lines.append(
            f'  {step}. eddy corrects head motion and eddy currents. ' + _no_sdc_sentence(grouping)
        )
        step += 1

    return step


def _describe_tortoise(lines, grouping, multipart_id, corrected, dgroups, step):
    kinds = _ids_by_kind(grouping, corrected)
    pepolar_ids = kinds.pepolar
    gre_ids = kinds.gre

    if len(dgroups) > 1:
        keys = ', '.join(dgroup.key for dgroup in dgroups)
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
            pairs = blip_pair_polarities(grouping, estimation)
            complete = sorted(
                (key for key, pols in pairs.items() if len(pols) == 2), key=blip_sort_key
            )
            unpaired = sorted(
                (key for key, pols in pairs.items() if len(pols) < 2), key=blip_sort_key
            )
            if len(complete) > 1:
                labels = '; '.join(describe_blip_group(key) for key in complete)
                lines.append(
                    f'     Estimation forms {len(complete)} blip pairs ({labels}); each is '
                    'corrected in its own DIFFPREP+DRBUDDI pipeline and the corrected '
                    'results are combined.'
                )
            axis_counts = Counter(key[0] for key in complete)
            for key in complete:
                up, down = _split_polarities(dgroups, key)
                up_names = ', '.join(dgroup.key for dgroup in up) or 'borrowed series'
                down_names = ', '.join(dgroup.key for dgroup in down) or 'borrowed series'
                where = f'the {key[0]} axis'
                if key[1] is not None and axis_counts[key[0]] > 1:
                    where += f' (TRT {key[1]:g}s)'
                lines.append(
                    f'  {step}. DRBUDDI estimates distortion along {where} '
                    f'from the blip-up ({up_names}) and blip-down ({down_names}) '
                    'data and applies the correction to every volume.'
                )
                step += 1
            if unpaired:
                groups = '; '.join(describe_blip_group(key) for key in unpaired)
                fallback = (
                    'DIFFPREP registers them to the T2w instead (T2Wreg)'
                    if grouping.anat_files('T2w')
                    else 'they are left uncorrected (no T2w for a T2Wreg fallback)'
                )
                lines.append(
                    f'  {step}. DRBUDDI has no opposing blip for {groups}, so {fallback}.'
                )
                step += 1
            note = _borrow_note(grouping, multipart_id, b0field_id)
            if note:
                lines.append(f'     {note}')
        structural = _structural_note(grouping)
        if structural:
            lines.append(structural)
    elif kinds.synb0:
        for b0field_id in kinds.synb0:
            estimation = grouping.estimations[b0field_id]
            t1w_names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. SyNb0 synthesizes an undistorted b=0 image from the T1w '
                f'({t1w_names}); DIFFPREP registers the data to it (T2Wreg mode) '
                'to estimate and correct distortion.'
            )
            step += 1
    elif kinds.anat:
        for b0field_id in kinds.anat:
            estimation = grouping.estimations[b0field_id]
            t2w_names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. DIFFPREP registers the b=0 image to the T2w '
                f'({t2w_names}) to estimate and correct distortion (T2Wreg).'
            )
            step += 1
    elif gre_ids:
        for b0field_id in gre_ids:
            estimation = grouping.estimations[b0field_id]
            lines.append(
                f'  {step}. A fieldmap is computed from the '
                f'{_METHOD_LABELS[estimation.method]} acquisition and applied to '
                'unwarp the DIFFPREP output.'
            )
            step += 1
    elif kinds.syn:
        for b0field_id in kinds.syn:
            estimation = grouping.estimations[b0field_id]
            t1w_names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. A standalone fieldmap-less SyN warp is estimated by a '
                f'constrained ANTs registration of the T1w ({t1w_names}) to a fieldmap '
                'atlas, and applied to unwarp the DIFFPREP output.'
            )
            step += 1
    else:
        lines.append(f'  {step}. ' + _no_sdc_sentence(grouping))
        step += 1

    return step


def _describe_mixed(lines, grouping, multipart_id, corrected, dgroups, step):
    kinds = _ids_by_kind(grouping, corrected)
    pepolar_ids = kinds.pepolar

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
        # The refinement stage depends on the data: DRBUDDI needs reverse
        # phase-encoded dMRI *series* (a lone reverse b=0 was already consumed
        # by TOPUP); otherwise T2Wreg against a structural target, or nothing.
        all_pairs: dict = {}
        for b0field_id in pepolar_ids:
            all_pairs.update(blip_pair_polarities(grouping, grouping.estimations[b0field_id]))
        refine_pairs = sorted(
            {
                key
                for b0field_id in pepolar_ids
                for key in dwi_blip_pairs(grouping, grouping.estimations[b0field_id])
            },
            key=blip_sort_key,
        )
        if not refine_pairs:
            # No reverse-PE dMRI pair for DRBUDDI to refine (a lone reverse b=0 was
            # already consumed by TOPUP): T2Wreg against a structural target, else
            # single-stage.
            phrase = _structural_phrase(grouping)
            if phrase:
                lines.append(
                    f'  {step}. There is no reverse phase-encoded dMRI series for '
                    f'DRBUDDI to refine with; instead, T2Wreg registers the '
                    f'eddy-corrected b=0 to {phrase}, refining the TOPUP correction.'
                )
                step += 1
            else:
                lines.append(
                    f'  {step}. There is no reverse phase-encoded dMRI series for '
                    'DRBUDDI to refine with, and no T2w or synthetic b=0 for a '
                    'T2Wreg stage: correction is single-stage (TOPUP + eddy).'
                )
                step += 1
        elif len(all_pairs) > 1:
            # DRBUDDI's single pass corrects one matched pair; a multi-group unit
            # gets single-stage TOPUP+eddy (which pools every group correctly).
            labels = '; '.join(
                describe_blip_group(key) for key in sorted(all_pairs, key=blip_sort_key)
            )
            lines.append(
                f'  {step}. The estimation spans {len(all_pairs)} blip groups ({labels}); '
                'TOPUP+eddy corrects them together, but the single-pass DRBUDDI refinement '
                'covers one matched pair, so no DRBUDDI refinement is applied.'
            )
            step += 1
        else:
            axis_counts = Counter(key[0] for key in refine_pairs)
            for key in refine_pairs:
                where = f'the {key[0]} axis'
                if key[1] is not None and axis_counts[key[0]] > 1:
                    where += f' (TRT {key[1]:g}s)'
                lines.append(
                    f'  {step}. DRBUDDI re-estimates distortion along {where} '
                    'from the eddy-corrected blip-up/blip-down dMRI series, refining '
                    'the TOPUP correction.'
                )
                step += 1
            for b0field_id in pepolar_ids:
                note = _borrow_note(grouping, multipart_id, b0field_id)
                if note:
                    lines.append(f'     {note}')
            structural = _structural_note(grouping)
            if structural:
                lines.append(structural)
    elif kinds.synb0:
        for b0field_id in kinds.synb0:
            estimation = grouping.estimations[b0field_id]
            t1w_names = ', '.join(_basename(path) for path in estimation.sources)
            lines.append(
                f'  {step}. SyNb0 synthesizes an undistorted b=0 image from the T1w '
                f'({t1w_names}); TOPUP treats it as a zero-readout-time volume and '
                'eddy applies the resulting field.'
            )
            step += 1
        lines.append(
            f'  {step}. T2Wreg registers the eddy-corrected b=0 to the synthetic '
            'b=0, refining the TOPUP correction.'
        )
        step += 1
    elif kinds.anat:
        lines.append(
            f'  {step}. The selected correction is a T2w registration (T2Wreg), '
            'which only the TORTOISE path implements: processing follows the FSL '
            'path with NO distortion correction.'
        )
        step += 1
    elif kinds.gre:
        lines.append(
            f'  {step}. Without a PEPOLAR fieldmap the DRBUDDI second stage adds '
            'nothing: processing follows the FSL path (fieldmap-based correction).'
        )
        step += 1
    elif kinds.syn:
        lines.append(
            f'  {step}. Without a PEPOLAR fieldmap the DRBUDDI second stage adds '
            'nothing: processing follows the FSL path with a fieldmap-less SyN warp.'
        )
        step += 1
    else:
        lines.append(f'  {step}. ' + _no_sdc_sentence(grouping))
        step += 1

    return step


def full_report(grouping: DWIGrouping) -> str:
    """The grouping report followed by all three backend previews."""
    sections = [report_text(grouping)]
    for backend in ('fsl', 'tortoise', 'mixed'):
        sections.append(describe_processing(grouping, backend))
    return '\n'.join(sections)
