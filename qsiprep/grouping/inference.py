"""Infer virtual B0FieldIdentifier/B0FieldSource/MultipartID values.

The grouping model is always expressed in BIDS curation vocabulary. When the
user curated their sidecars, those values are used verbatim. When they did
not, this module fills in equivalent values, with strict precedence:

1. **Curated** ``B0FieldIdentifier``/``B0FieldSource`` (step E1) always win.
2. **IntendedFor** on fmap/ files is translated into estimations (step E2).
3. A **heuristic** groups DWI series with differing phase encoding into one
   PEPOLAR estimation (step E3), which handles HCP-style acquisitions with
   zero curation. It runs only in sessions with no curated fieldmap linkage
   at all: once anything in a session is curated, QSIPrep stops guessing
   for the rest of it.

The heuristic operates per (session, shim-compatible bucket), so a DWI series
with no reverse-PE partner of its own can still *borrow* compatible series
from elsewhere in the session for fieldmap estimation - even when those
series are concatenated into a different output. Estimation membership and
concatenation membership are independent by design.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from .models import (
    AUTO_PREFIX,
    ConcatenationGroup,
    CorrectionMethod,
    CorrectionUnit,
    DistortionGroup,
    DWIGrouping,
    FieldmapEstimation,
    FileRecord,
    GroupingPolicy,
    Provenance,
    derive_output_name,
    strip_nii_ext,
)
from .validation import GroupingIssue, error, warning

_METHOD_RANK = {
    CorrectionMethod.PEPOLAR: 0,
    CorrectionMethod.DIRECT: 1,
    CorrectionMethod.PHASEDIFF: 2,
    CorrectionMethod.PHASES: 3,
    CorrectionMethod.SYNB0: 4,
    CorrectionMethod.T2WREG: 5,
    CorrectionMethod.NIPREPS_SYN: 6,
}
_PROVENANCE_RANK = {
    Provenance.CURATED: 0,
    Provenance.TRANSLATED: 1,
    Provenance.FORCED: 2,
    Provenance.INFERRED: 3,
}


def _entity_stem(path: str) -> str:
    """Filename without extension and without the trailing suffix token.

    ``sub-01_acq-x_phasediff.nii.gz`` and ``sub-01_acq-x_magnitude1.nii.gz``
    share the stem ``sub-01_acq-x``, which is how sidecar-companion files
    (phasediff + magnitudes, phase1 + phase2) are recognized.
    """
    fname = strip_nii_ext(path)
    return fname.rsplit('_', 1)[0] if '_' in fname else fname


def _classify_method(records: list[FileRecord]) -> CorrectionMethod | None:
    suffixes = {record.suffix for record in records}
    if 'fieldmap' in suffixes:
        return CorrectionMethod.DIRECT
    if 'phasediff' in suffixes:
        return CorrectionMethod.PHASEDIFF
    if 'phase1' in suffixes and 'phase2' in suffixes:
        return CorrectionMethod.PHASES
    # An anatomical source marks a fieldmap-less registration estimation, even
    # when EPI files share the identifier (they are its registration movers).
    if suffixes.intersection(('T1w', 'T2w')):
        return CorrectionMethod.T2WREG
    if any(record.is_epi_like for record in records):
        return CorrectionMethod.PEPOLAR
    return None


def _pe_axes(records: list[FileRecord]) -> tuple[frozenset, frozenset]:
    """(axes covered, axes with both polarities) among EPI-like records."""
    polarities = defaultdict(set)
    for record in records:
        signature = record.signature
        if record.is_epi_like and signature.pe_axis:
            polarities[signature.pe_axis].add(signature.pe_polarity)
    axes = frozenset(polarities)
    bidirectional = frozenset(axis for axis, pols in polarities.items() if len(pols) == 2)
    return axes, bidirectional


def _make_estimation(
    b0field_id: str,
    method: CorrectionMethod,
    records: list[FileRecord],
    provenance: Provenance,
) -> FieldmapEstimation:
    axes, bidirectional = _pe_axes(records)
    return FieldmapEstimation(
        b0field_id=b0field_id,
        method=method,
        sources=tuple(sorted(record.path for record in records)),
        provenance=provenance,
        pe_axes=axes,
        bidirectional_axes=bidirectional,
    )


def _check_estimation_shims(
    estimation_id: str,
    records: list[FileRecord],
    ignore_shims: bool,
    make_issue,  # `error` for curated estimations, `warning` for translated
    issues: list[GroupingIssue],
):
    epi_like = [record for record in records if record.is_epi_like]
    for rec_a, rec_b in combinations(epi_like, 2):
        if not rec_a.signature.compatible_shim(rec_b.signature, ignore_shims=ignore_shims):
            issues.append(
                make_issue(
                    'estimation-shim-mismatch',
                    f"Files combined for fieldmap estimation '{estimation_id}' were "
                    f'acquired with different shim settings '
                    f'({rec_a.filename} vs {rec_b.filename}). The estimated field '
                    f'may not match either image. Re-run with shims ignored to '
                    f'silence this if it is intentional.',
                    (rec_a.path, rec_b.path),
                )
            )
            return


def _unique_id(base: str, taken, scope: str | None = None) -> str:
    if base not in taken:
        return base
    # The same series curated into several output groups yields identically
    # named distortion groups/units; disambiguate with the output scope (a
    # readable MultipartID) before falling back to a numeric suffix.
    if scope:
        scoped = f'{base}_{scope}'
        if scoped not in taken:
            return scoped
        base = scoped
    counter = 2
    while f'{base}+{counter}' in taken:
        counter += 1
    return f'{base}+{counter}'


def resolve_estimations(
    records: list[FileRecord],
    ignore_shims: bool = False,
):
    """Steps E1-E3: build every FieldmapEstimation for this subject.

    Returns ``(estimations, targets, issues)`` where ``targets`` maps each
    estimation id to the set of DWI paths it is known to correct (used by
    :func:`resolve_application`; for INFERRED estimations the targets are the
    estimation's own DWI sources).
    """
    issues: list[GroupingIssue] = []
    estimations: dict[str, FieldmapEstimation] = {}
    targets: dict[str, set[str]] = {}
    by_path = {record.path: record for record in records}

    # ------------------------------------------------------------------ E1
    # A B0FieldIdentifier must be unique within a subject: sessions are
    # reshimmed, so one field cannot span them. Session-less sources (a
    # shared fmap) declare it once and are exempt.
    curated_members = defaultdict(list)
    for record in records:
        for identifier in record.b0field_identifiers:
            curated_members[identifier].append(record)

    for identifier, members in sorted(curated_members.items()):
        if identifier.startswith(AUTO_PREFIX):
            issues.append(
                error(
                    'reserved-b0field-prefix',
                    f"B0FieldIdentifier '{identifier}' uses the reserved '{AUTO_PREFIX}' "
                    'prefix. Rename it in your sidecars.',
                    tuple(record.path for record in members),
                )
            )
            continue
        declaring_sessions = sorted(
            {record.session for record in members if record.session is not None}, key=str
        )
        if len(declaring_sessions) > 1:
            sessions_txt = ', '.join(f'ses-{session}' for session in declaring_sessions)
            issues.append(
                error(
                    'b0field-multisession',
                    f"B0FieldIdentifier '{identifier}' is declared by files in "
                    f'{len(declaring_sessions)} sessions ({sessions_txt}). A '
                    'B0FieldIdentifier must be unique within a subject: the scanner is '
                    'reshimmed between sessions, so one fieldmap cannot span them. Give '
                    f"each session its own identifier (e.g. '{identifier}_ses-...').",
                    tuple(record.path for record in members),
                )
            )
            continue
        method = _classify_method(members)
        if method is None:
            issues.append(
                error(
                    'curated-estimation-unclassifiable',
                    f'Could not determine an estimation method for B0FieldIdentifier '
                    f"'{identifier}' from its files "
                    f'({", ".join(record.filename for record in members)}).',
                    tuple(record.path for record in members),
                )
            )
            continue
        estimation = _make_estimation(identifier, method, members, Provenance.CURATED)
        if method is CorrectionMethod.PEPOLAR:
            signatures = {
                record.signature.key
                for record in members
                if record.is_epi_like and record.signature.pe_dir
            }
            if len(signatures) < 2:
                issues.append(
                    error(
                        'curated-pepolar-single-signature',
                        f"All files with B0FieldIdentifier '{identifier}' share one "
                        'distortion signature; a PEPOLAR fieldmap cannot be estimated '
                        'from them. Add a reverse phase-encoded acquisition to this '
                        'identifier or remove it.',
                        estimation.sources,
                    )
                )
            _check_estimation_shims(identifier, members, ignore_shims, error, issues)
        estimations[identifier] = estimation
        targets[identifier] = {
            record.path
            for record in records
            if record.is_dwi and identifier in record.b0field_sources
        }

    for record in records:
        if record.datatype == 'fmap' and record.b0field_identifiers and record.intended_for:
            issues.append(
                warning(
                    'intendedfor-superseded',
                    f'{record.filename} carries both B0FieldIdentifier and IntendedFor. '
                    'IntendedFor is deprecated; the B0FieldIdentifier/B0FieldSource links '
                    'are used exclusively and the IntendedFor entries are ignored.',
                    (record.path,),
                )
            )

    # ------------------------------------------------------------------ E2
    # Cluster uncurated fmap files by entity stem so sidecar companions
    # (phasediff+magnitudes, phase1+phase2) stay together.
    stems = defaultdict(list)
    for record in records:
        if record.datatype == 'fmap' and not record.b0field_identifiers:
            stems[_entity_stem(record.path)].append(record)

    translated = []  # (cluster records, target dwi paths, method)
    for _, members in sorted(stems.items()):
        cluster_targets = set()
        for record in members:
            cluster_targets.update(record.intended_for)
        if not cluster_targets:
            epi_like = [record for record in members if record.is_epi_like]
            if epi_like:
                issues.append(
                    warning(
                        'unlinked-fmap',
                        f'{", ".join(record.filename for record in epi_like)} in fmap/ has '
                        'no B0FieldIdentifier or IntendedFor linking it to a DWI series, '
                        'so it will not be used. Add B0FieldIdentifier/B0FieldSource '
                        'metadata to use it. (IntendedFor is also honored, but '
                        'deprecated.)',
                        tuple(record.path for record in epi_like),
                    )
                )
            continue
        method = _classify_method(members)
        if method is None:
            issues.append(
                warning(
                    'intendedfor-unclassifiable',
                    'Could not determine an estimation method for fmap files '
                    f'{", ".join(record.filename for record in members)}; they will '
                    'not be used.',
                    tuple(record.path for record in members),
                )
            )
            continue
        translated.append((members, cluster_targets, method))

    # Merge PEPOLAR clusters that target the same DWI files: multiple epi
    # fmaps intended for the same series jointly estimate one field.
    merged_pepolar = {}
    for members, cluster_targets, method in translated:
        if method is CorrectionMethod.PEPOLAR:
            key = frozenset(cluster_targets)
            merged_pepolar.setdefault(key, []).extend(members)
    translated_final = [
        (members, set(key), CorrectionMethod.PEPOLAR)
        for key, members in sorted(merged_pepolar.items(), key=lambda kv: sorted(kv[0]))
    ] + [item for item in translated if item[2] is not CorrectionMethod.PEPOLAR]

    for members, cluster_targets, method in translated_final:
        fmap_sources = list(members)
        if method is CorrectionMethod.PEPOLAR:
            # The target DWIs participate in a PEPOLAR estimation: their b=0
            # images supply the other phase encoding direction(s).
            fmap_sources = members + [
                by_path[path] for path in sorted(cluster_targets) if path in by_path
            ]
        base_id = AUTO_PREFIX + 'fmap+' + derive_output_name([record.path for record in members])
        b0field_id = _unique_id(base_id, estimations)
        estimation = _make_estimation(b0field_id, method, fmap_sources, Provenance.TRANSLATED)
        _check_estimation_shims(b0field_id, fmap_sources, ignore_shims, warning, issues)
        estimations[b0field_id] = estimation
        targets[b0field_id] = set(cluster_targets)

    # ------------------------------------------------------------------ E3
    # The reverse phase-encoding heuristic runs only in sessions with NO
    # curated fieldmap linkage at all. In an uncurated dataset, absent
    # metadata means "nobody looked" and guessing is a service; once any
    # series in the session is linked - by B0FieldIdentifier/B0FieldSource
    # or by an IntendedFor naming it - absent metadata means "somebody
    # looked and chose not to link these", so QSIPrep stops guessing.
    # (Fieldmap-less correction still applies: unlike sidecar metadata, it
    # is under the user's control at the command line.)
    intendedfor_covered = set()
    for b0field_id, estimation in estimations.items():
        if estimation.provenance is Provenance.TRANSLATED:
            intendedfor_covered.update(targets[b0field_id])

    #: Sessions where any file carries B0Field* metadata (a curated fmap or
    #: anat counts even if no DWI sources it: the curator was here).
    curated_sessions = {
        record.session
        for record in records
        if record.b0field_identifiers or record.b0field_sources
    }

    def _linked(record: FileRecord) -> bool:
        return bool(
            record.b0field_identifiers
            or record.b0field_sources
            or record.path in intendedfor_covered
        )

    dwi_records = [record for record in records if record.is_dwi]
    for session, session_records in sorted(
        _by_session(dwi_records).items(), key=lambda kv: str(kv[0])
    ):
        unlinked = [record for record in session_records if not _linked(record)]
        if session in curated_sessions or len(unlinked) < len(session_records):
            if unlinked:
                names = ', '.join(record.filename for record in unlinked)
                issues.append(
                    warning(
                        'reverse-pe-not-inferred',
                        f'{names}: this session has curated fieldmap metadata, so '
                        'QSIPrep does not infer reverse phase-encoding pairings for '
                        'the remaining series. Add B0FieldIdentifier/B0FieldSource '
                        'to correct them (fieldmap-less correction can still be '
                        'requested at the command line).',
                        tuple(record.path for record in unlinked),
                    )
                )
            continue

        shim_groups = _shim_groups(session_records, ignore_shims, issues)
        for shim_index, shim_records in enumerate(shim_groups):
            encoded = [record for record in shim_records if record.signature.pe_dir]
            directions = {record.signature.pe_dir for record in encoded}
            if len(directions) < 2:
                continue
            # Any two differing phase encodings jointly determine the
            # susceptibility field - opposite polarity on one axis is the
            # well-conditioned special case, not a requirement - so ALL
            # differing-PE series in the bucket estimate one field together.
            # Whether a backend can consume the resulting shape (multiple
            # axes, unpaired polarities) is check_backend's business.
            axes = ''.join(sorted({record.signature.pe_axis for record in encoded}))
            id_parts = [AUTO_PREFIX + 'pepolar']
            if session:
                id_parts.append(f'ses-{session}')
            if len(shim_groups) > 1:
                id_parts.append(f'shim{shim_index + 1}')
            id_parts.append(axes)
            b0field_id = _unique_id('+'.join(id_parts), estimations)
            estimation = _make_estimation(
                b0field_id, CorrectionMethod.PEPOLAR, encoded, Provenance.INFERRED
            )
            estimations[b0field_id] = estimation
            # The inferred estimation corrects exactly the series it was
            # built from - including single-direction "borrowers".
            targets[b0field_id] = {record.path for record in encoded}

    return estimations, targets, issues


def _by_session(records: list[FileRecord]) -> dict:
    sessions = defaultdict(list)
    for record in records:
        sessions[record.session].append(record)
    return sessions


def _shim_groups(
    records: list[FileRecord],
    ignore_shims: bool,
    issues: list[GroupingIssue],
) -> list[list[FileRecord]]:
    """Partition a session's DWI records into shim-compatible buckets.

    Records without a ShimSetting are wildcards: they join every bucket.
    """
    distinct = sorted({record.signature.shim for record in records if record.signature.shim})
    if len(distinct) <= 1:
        return [records]
    if ignore_shims:
        issues.append(
            warning(
                'shims-ignored',
                f'{len(distinct)} different shim settings found in one session, but '
                'shim checking is disabled: all series are treated as compatible '
                'for fieldmap estimation.',
                tuple(record.path for record in records),
            )
        )
        return [records]

    issues.append(
        warning(
            'session-multiple-shims',
            f'{len(distinct)} different shim settings found in one session. Series '
            'are only combined for fieldmap estimation within a matching shim '
            'setting. Use ignore_shims to override.',
            tuple(record.path for record in records),
        )
    )
    wildcards = [record for record in records if not record.signature.shim]
    if wildcards:
        issues.append(
            warning(
                'shim-wildcard',
                f'{len(wildcards)} DWI series have no ShimSetting and are treated as '
                'compatible with every shim group.',
                tuple(record.path for record in wildcards),
            )
        )
    return [
        [record for record in records if record.signature.shim in (shim, None, ())]
        for shim in distinct
    ]


def resolve_application(
    records: list[FileRecord],
    estimations: dict[str, FieldmapEstimation],
    targets: dict[str, set[str]],
):
    """Decide which estimation corrects each DWI file (its B0FieldSource)."""
    issues: list[GroupingIssue] = []
    application: dict[str, str | None] = {}
    provenance: dict[str, Provenance] = {}
    candidates_out: dict[str, tuple[str, ...]] = {}

    for record in sorted((record for record in records if record.is_dwi), key=lambda r: r.path):
        candidates = []  # (method_rank, provenance_rank, b0field_id)

        for source_id in record.b0field_sources:
            if source_id not in estimations:
                issues.append(
                    error(
                        'unresolvable-b0fieldsource',
                        f"{record.filename} names B0FieldSource '{source_id}', but no "
                        'file in this subject carries that B0FieldIdentifier.',
                        (record.path,),
                    )
                )
                continue
            candidates.append((source_id, Provenance.CURATED))

        if not candidates:
            for b0field_id, estimation in estimations.items():
                if (
                    estimation.provenance is Provenance.TRANSLATED
                    and record.path in targets[b0field_id]
                ):
                    candidates.append((b0field_id, Provenance.TRANSLATED))

        if not candidates:
            for b0field_id, estimation in estimations.items():
                if (
                    estimation.provenance is Provenance.INFERRED
                    and record.path in targets[b0field_id]
                ):
                    candidates.append((b0field_id, Provenance.INFERRED))

        candidates.sort(
            key=lambda item: (
                _METHOD_RANK[estimations[item[0]].method],
                _PROVENANCE_RANK[item[1]],
                item[0],
            )
        )
        candidates_out[record.path] = tuple(b0field_id for b0field_id, _ in candidates)
        if candidates:
            chosen_id, chosen_provenance = candidates[0]
            application[record.path] = chosen_id
            provenance[record.path] = chosen_provenance
        else:
            application[record.path] = None
            provenance[record.path] = Provenance.INFERRED

    applied_provenances = {
        provenance[path] for path, chosen in application.items() if chosen is not None
    }
    if Provenance.CURATED in applied_provenances and len(applied_provenances) > 1:
        curated = sorted(
            path
            for path, chosen in application.items()
            if chosen and provenance[path] is Provenance.CURATED
        )
        uncurated = sorted(
            path
            for path, chosen in application.items()
            if chosen and provenance[path] is not Provenance.CURATED
        )
        issues.append(
            warning(
                'mixed-application-provenance',
                f'{len(curated)} DWI series have curated B0FieldSource metadata but '
                f'{len(uncurated)} do not; fieldmaps for the latter were assigned '
                'automatically. Curate B0FieldSource on every series to make this '
                'fully explicit.',
                tuple(uncurated),
            )
        )

    return application, provenance, candidates_out, issues


def _anat_for_session(records, session, suffix):
    """Anatomical images for a session: same session, else session-less, else any."""
    anat = [record for record in records if record.is_anat and record.suffix == suffix]
    for candidates in (
        [record for record in anat if record.session == session],
        [record for record in anat if record.session is None],
        anat,
    ):
        if candidates:
            return sorted(candidates, key=lambda record: record.path)
    return []


def resolve_fieldmapless(
    records: list[FileRecord],
    estimations: dict[str, FieldmapEstimation],
    application: dict[str, str | None],
    provenance: dict[str, Provenance],
    candidates: dict[str, tuple[str, ...]],
    force_t2wreg: bool = False,
    use_synb0: bool = False,
    use_nipreps_syn_sdc: bool = False,
):
    """Apply the fieldmap-less ladder to the application map (in place).

    The fieldmap-less methods are mutually exclusive per run - each corrects a
    series entirely on its own, never layered together:

    - ``force_t2wreg`` overrides every DWI's fieldmap with a T2w registration
      (T2Wreg) estimation and overrides the other two if they were also asked
      for.
    - ``use_nipreps_syn_sdc`` is the standalone niworkflows SyN-SDC: a
      constrained ANTs SyN registration of an inverted T1w (or a synthetic b=0)
      to a fieldmap atlas. It corrects every still-uncorrected series and is
      never combined with SyNb0 (SyNb0 wins if both are requested).
    - ``use_synb0`` gives still-uncorrected series a SyNb0 synthetic-b=0
      estimation.
    - Finally, uncorrected series in a subject with a T2w fall back to an
      inferred T2Wreg estimation (today's automatic TORTOISE behavior, made
      explicit).

    Anatomical estimations are created per session, with the anatomical
    image(s) as their only sources - the DWIs they correct are targets, since
    each output registers its own b=0. A DWI without a PhaseEncodingDirection
    cannot be corrected along an axis: it is skipped by the fallback and is a
    hard error when SyNb0 or SyN-SDC was explicitly requested.
    """
    issues: list[GroupingIssue] = []
    dwi_records = {record.path: record for record in records if record.is_dwi}

    def _apply(paths, session, method, id_stem, suffix, prov):
        anat = _anat_for_session(records, session, suffix)
        if not anat:
            return False
        id_parts = [AUTO_PREFIX + id_stem]
        if session:
            id_parts.append(f'ses-{session}')
        b0field_id = '+'.join(id_parts)
        if b0field_id not in estimations:
            estimations[b0field_id] = _make_estimation(b0field_id, method, anat, prov)
        for path in paths:
            application[path] = b0field_id
            provenance[path] = prov
            candidates[path] = (b0field_id, *candidates.get(path, ()))
        return True

    by_session = defaultdict(list)
    for path, record in sorted(dwi_records.items()):
        by_session[record.session].append(path)

    if force_t2wreg:
        if use_synb0 or use_nipreps_syn_sdc:
            issues.append(
                warning(
                    'fieldmapless-overridden',
                    'Forcing T2Wreg overrides the other fieldmap-less methods; '
                    'SyNb0 and SyN-SDC are not used.',
                )
            )
        for session, paths in sorted(by_session.items(), key=lambda kv: str(kv[0])):
            if not _apply(
                paths, session, CorrectionMethod.T2WREG, 't2wreg', 'T2w', Provenance.FORCED
            ):
                issues.append(
                    error(
                        't2wreg-requires-t2w',
                        'T2w-registration SDC (T2Wreg) was requested, but this subject '
                        'has no T2w image.',
                        tuple(paths),
                    )
                )
        return issues

    # SyN-SDC is a standalone fieldmap-less workflow; it is never layered with
    # SyNb0. If both are requested, SyNb0 wins and SyN-SDC is dropped.
    if use_synb0 and use_nipreps_syn_sdc:
        issues.append(
            warning(
                'syn-sdc-standalone',
                'SyN-SDC is a standalone fieldmap-less method and cannot be combined '
                'with SyNb0; SyNb0 is used and SyN-SDC is not.',
            )
        )
        use_nipreps_syn_sdc = False

    if use_synb0:
        for session, paths in sorted(by_session.items(), key=lambda kv: str(kv[0])):
            uncorrected = [path for path in paths if application[path] is None]
            if not uncorrected:
                continue
            missing_pedir = [
                path for path in uncorrected if dwi_records[path].signature.pe_dir is None
            ]
            if missing_pedir:
                issues.append(
                    error(
                        'synb0-missing-pedir',
                        'SyNb0 was requested, but these DWI series have no '
                        'PhaseEncodingDirection, which the synthetic-b=0 correction '
                        'requires.',
                        tuple(missing_pedir),
                    )
                )
            correctable = [path for path in uncorrected if path not in missing_pedir]
            if correctable and not _apply(
                correctable, session, CorrectionMethod.SYNB0, 'synb0', 'T1w', Provenance.FORCED
            ):
                issues.append(
                    error(
                        'synb0-requires-t1w',
                        'SyNb0 was requested, but this subject has no T1w image to '
                        'synthesize an undistorted b=0 from.',
                        tuple(correctable),
                    )
                )

    if use_nipreps_syn_sdc:
        for session, paths in sorted(by_session.items(), key=lambda kv: str(kv[0])):
            uncorrected = [path for path in paths if application[path] is None]
            if not uncorrected:
                continue
            missing_pedir = [
                path for path in uncorrected if dwi_records[path].signature.pe_dir is None
            ]
            if missing_pedir:
                issues.append(
                    error(
                        'syn-missing-pedir',
                        'SyN-SDC was requested, but these DWI series have no '
                        'PhaseEncodingDirection, which the fieldmap-less SyN correction '
                        'requires.',
                        tuple(missing_pedir),
                    )
                )
            correctable = [path for path in uncorrected if path not in missing_pedir]
            if correctable and not _apply(
                correctable, session, CorrectionMethod.NIPREPS_SYN, 'syn', 'T1w', Provenance.FORCED
            ):
                issues.append(
                    error(
                        'syn-requires-t1w',
                        'SyN-SDC was requested, but this subject has no T1w image to '
                        'register against a template.',
                        tuple(correctable),
                    )
                )

    # Automatic fallback: a subject with a T2w gets T2Wreg for anything that
    # still has no fieldmap (executable on the TORTOISE path only - the
    # backend checks say so explicitly).
    for session, paths in sorted(by_session.items(), key=lambda kv: str(kv[0])):
        fallback = [
            path
            for path in paths
            if application[path] is None and dwi_records[path].signature.pe_dir is not None
        ]
        if fallback:
            _apply(
                fallback,
                session,
                CorrectionMethod.T2WREG,
                't2wreg',
                'T2w',
                Provenance.INFERRED,
            )

    return issues


def build_distortion_groups(
    records: list[FileRecord],
    application: dict[str, str | None],
    separate_all_dwis: bool,
) -> dict[str, DistortionGroup]:
    """Group DWI files that share a distortion and a correction.

    Files are grouped by their distortion parameters, the fieldmap applied to
    them, and their output (MultipartID or ``separate_all_dwis``), so a single
    group can never span two fieldmaps or two output files. A series curated
    into several MultipartIDs (virtual acquisitions) contributes to one group per
    output scope, so the group's scope is stored, never re-derived.
    """
    dwi_records = [record for record in records if record.is_dwi]
    buckets = defaultdict(list)
    for record in dwi_records:
        if separate_all_dwis:
            walls = (record.path,)
        else:
            # Dedupe repeated sidecar ids; `or (None,)` keeps uncurated
            # series (empty MultipartID) in a single None-scoped group.
            walls = tuple(dict.fromkeys(record.multipart_id)) or (None,)
        for wall in walls:
            buckets[(record.session, record.signature.key, application[record.path], wall)].append(
                record
            )

    groups: dict[str, DistortionGroup] = {}
    for (_, _, applied, wall), members in sorted(buckets.items(), key=lambda kv: kv[1][0].path):
        scope = None if separate_all_dwis else wall
        key = _unique_id(
            derive_output_name([record.path for record in members]), groups, scope=scope
        )
        groups[key] = DistortionGroup(
            key=key,
            signature=members[0].signature,
            dwi_files=tuple(sorted(record.path for record in members)),
            b0field_source=applied,
            multipart_scope=scope,
        )
    return groups


def build_correction_units(
    records: list[FileRecord],
    distortion_groups: dict[str, DistortionGroup],
    separate_all_dwis: bool,
) -> dict[str, CorrectionUnit]:
    """Partition distortion groups into correction units.

    A unit's raw series are stacked into one HMC+SDC pipeline, so its groups
    must share ONE applied correction: within a (session, MultipartID scope)
    wall, the groups applying the same non-null estimation form one unit (a
    PEPOLAR pair's two polarities, corrected jointly); uncorrected groups -
    and every group under ``separate_all_dwis`` - stand alone. Units never
    span sessions or MultipartIDs.
    """
    dwi_records = {record.path: record for record in records if record.is_dwi}

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for key, dgroup in distortion_groups.items():
        if separate_all_dwis or dgroup.b0field_source is None:
            bucket = ('single', key)
        else:
            record = dwi_records[dgroup.dwi_files[0]]
            bucket = ('shared', record.session, dgroup.multipart_scope, dgroup.b0field_source)
        buckets[bucket].append(key)

    units: dict[str, CorrectionUnit] = {}
    # Order by smallest member key so unit-key disambiguation is deterministic.
    for member_keys in sorted(buckets.values(), key=min):
        member_keys = tuple(sorted(member_keys))
        dwi_files = tuple(
            sorted(path for key in member_keys for path in distortion_groups[key].dwi_files)
        )
        # A unit never spans scopes (the partition key includes the scope), so
        # any member's scope is the unit's.
        scope = distortion_groups[member_keys[0]].multipart_scope
        unit_key = _unique_id(derive_output_name(dwi_files), units, scope=scope)
        units[unit_key] = CorrectionUnit(
            key=unit_key,
            distortion_groups=member_keys,
            dwi_files=dwi_files,
            b0field_source=distortion_groups[member_keys[0]].b0field_source,
            multipart_scope=scope,
            session=dwi_records[dwi_files[0]].session,
        )
    return units


def build_concatenation_groups(
    records: list[FileRecord],
    distortion_groups: dict[str, DistortionGroup],
    correction_units: dict[str, CorrectionUnit],
    separate_all_dwis: bool,
    distortion_group_merge: str = 'concat',
):
    """Package correction units into final outputs.

    Each unit is preprocessed independently; a final output spanning several
    units concatenates (or averages, per ``distortion_group_merge``) their
    *corrected* results. Curated MultipartIDs define the final outputs
    verbatim. Otherwise, all corrected units in a session are packaged into
    one final output ('concat'/'average') or kept separate ('none');
    uncorrected units always stand alone - corrected and uncorrected volumes
    never share a file.
    """
    issues: list[GroupingIssue] = []
    dwi_records = {record.path: record for record in records if record.is_dwi}
    curated_ids = {mid for record in dwi_records.values() for mid in record.multipart_id}

    # Assign each correction unit to a MultipartID
    assignments: dict[str, tuple[str, Provenance]] = {}  # unit key -> (id, provenance)

    if separate_all_dwis:
        if curated_ids:
            issues.append(
                warning(
                    'multipartid-overridden',
                    'separate_all_dwis is enabled, overriding the MultipartID values '
                    'in the sidecars: every DWI series will be a separate output.',
                )
            )
        for key, unit in correction_units.items():
            stem = _entity_stem(unit.dwi_files[0])
            assignments[key] = (AUTO_PREFIX + 'single+' + stem, Provenance.INFERRED)
    elif curated_ids:
        uncurated_files = []
        for key, unit in correction_units.items():
            multipart_id = unit.multipart_scope
            if multipart_id:
                if multipart_id.startswith(AUTO_PREFIX):
                    issues.append(
                        error(
                            'reserved-multipartid-prefix',
                            f"MultipartID '{multipart_id}' uses the reserved "
                            f"'{AUTO_PREFIX}' prefix. Rename it in your sidecars.",
                            unit.dwi_files,
                        )
                    )
                assignments[key] = (multipart_id, Provenance.CURATED)
            else:
                stem = _entity_stem(unit.dwi_files[0])
                assignments[key] = (AUTO_PREFIX + 'single+' + stem, Provenance.INFERRED)
                uncurated_files.extend(unit.dwi_files)
        if uncurated_files:
            issues.append(
                warning(
                    'partial-multipart',
                    f'{len(uncurated_files)} DWI series have no MultipartID while other '
                    'series in this subject do. Series without one are NOT packaged '
                    'with the curated groups: each of their correction units becomes '
                    'its own output. Set MultipartID on every series (or on none) to '
                    'control the packaging explicitly.',
                    tuple(sorted(uncurated_files)),
                )
            )
        # Virtual acquisition mode: a series listing several MultipartIDs is a
        # deliberate request to preprocess it once per group. Surface it
        # loudly - the same raw data lands in multiple outputs on purpose.
        overlapping = sorted(
            record.path for record in dwi_records.values() if len(record.multipart_id) > 1
        )
        if overlapping:
            issues.append(
                warning(
                    'multipart-overlap',
                    f'Virtual acquisition mode: {len(overlapping)} DWI series with multiple '
                    'MultipartIDs; each is preprocessed once per group and appears in '
                    'each of those outputs.',
                    tuple(overlapping),
                )
            )
    else:
        # Inferred: per session, corrected units package into one final
        # output (their corrected results concatenate cleanly, whatever the
        # shim or estimation boundaries between them). Uncorrected units
        # stand alone.
        by_session = defaultdict(list)
        for key, unit in correction_units.items():
            by_session[dwi_records[unit.dwi_files[0]].session].append(key)

        packages: list[tuple[str | None, list[str]]] = []  # (session, unit keys)
        for session, session_keys in sorted(by_session.items(), key=lambda kv: str(kv[0])):
            corrected = sorted(key for key in session_keys if correction_units[key].b0field_source)
            uncorrected = sorted(
                key for key in session_keys if not correction_units[key].b0field_source
            )
            if distortion_group_merge == 'none':
                packages.extend((session, [key]) for key in corrected)
            elif corrected:
                packages.append((session, corrected))
            packages.extend((session, [key]) for key in uncorrected)

        for index, (session, unit_keys) in enumerate(packages):
            id_parts = [AUTO_PREFIX + 'concat']
            if session:
                id_parts.append(f'ses-{session}')
            id_parts.append(str(index))
            for key in unit_keys:
                assignments[key] = ('+'.join(id_parts), Provenance.INFERRED)

    # An id reused across sessions makes one output per session, never a
    # cross-session concatenation. The dict key is the bare id, session-
    # qualified only when the id recurs across sessions.
    members_by_group = defaultdict(list)  # (session, id) -> [(unit key, provenance)]
    for key, (multipart_id, provenance) in assignments.items():
        members_by_group[(correction_units[key].session, multipart_id)].append((key, provenance))

    sessions_per_id = defaultdict(set)
    for session, multipart_id in members_by_group:
        sessions_per_id[multipart_id].add(session)

    def _group_key(session, multipart_id):
        if len(sessions_per_id[multipart_id]) == 1:
            return multipart_id
        return f'{multipart_id}+ses-{session}'

    materialized = [
        (_group_key(session, multipart_id), session, multipart_id, members)
        for (session, multipart_id), members in members_by_group.items()
    ]

    concatenation_groups: dict[str, ConcatenationGroup] = {}
    output_names: dict[str, str] = {}
    for group_key, session, multipart_id, members in sorted(materialized):
        unit_keys = tuple(sorted(key for key, _ in members))
        keys = tuple(
            sorted(
                dgroup_key
                for unit_key in unit_keys
                for dgroup_key in correction_units[unit_key].distortion_groups
            )
        )
        dwi_files = tuple(
            sorted(path for key in keys for path in distortion_groups[key].dwi_files)
        )
        provenance = members[0][1]

        # A curated MultipartID of the form 'acq-<label>' does double duty:
        # it groups the series AND renames the acq- entity of the output.
        acq = None
        if provenance is Provenance.CURATED and multipart_id.startswith('acq-'):
            label = multipart_id[len('acq-') :]
            if re.fullmatch('[0-9a-zA-Z]+', label):
                acq = label
            else:
                issues.append(
                    error(
                        'multipartid-acq-invalid',
                        f"MultipartID '{multipart_id}' begins with 'acq-', which names "
                        f"the output's acq- entity, but '{label}' is not a valid BIDS "
                        'label (alphanumeric characters only). Rename it.',
                        dwi_files,
                    )
                )
        output_name = derive_output_name(dwi_files, acq=acq)

        if output_name in output_names:
            issues.append(
                error(
                    'output-name-collision',
                    f"Two output groups ('{output_names[output_name]}' and "
                    f"'{multipart_id}') would both be named '{output_name}'. "
                    'Add distinguishing entities to the filenames, or use '
                    "'acq-'-prefixed MultipartIDs (e.g. 'acq-multishell') to "
                    'name the outputs explicitly.',
                    dwi_files,
                )
            )
        output_names[output_name] = multipart_id
        concatenation_groups[group_key] = ConcatenationGroup(
            multipart_id=multipart_id,
            provenance=provenance,
            distortion_groups=keys,
            correction_units=unit_keys,
            dwi_files=dwi_files,
            output_name=output_name,
            key=group_key,
            session=session,
        )

    # An estimation whose targets span correction units is legal (borrowing),
    # but the user should know the same fieldmap will be estimated once per
    # unit pipeline.
    spanned_units = defaultdict(set)
    for unit_key, unit in correction_units.items():
        if unit.b0field_source is not None:
            spanned_units[unit.b0field_source].add(unit_key)
    for b0field_id, unit_keys in sorted(spanned_units.items()):
        if len(unit_keys) > 1:
            files = tuple(
                sorted(
                    path for unit_key in unit_keys for path in correction_units[unit_key].dwi_files
                )
            )
            issues.append(
                warning(
                    'estimation-spans-outputs',
                    f"Fieldmap estimation '{b0field_id}' corrects DWI series in "
                    f'{len(unit_keys)} different correction units; it will be '
                    'estimated once per unit.',
                    files,
                )
            )

    return concatenation_groups, issues


def build_grouping(
    records: list[FileRecord],
    subject_id: str,
    separate_all_dwis: bool = False,
    ignore_fieldmaps: bool = False,
    ignore_shims: bool = False,
    ignore_fov: bool = False,
    ignore_sdc: bool = False,
    force_t2wreg: bool = False,
    use_synb0: bool = False,
    use_nipreps_syn_sdc: bool = False,
    distortion_group_merge: str | None = 'concat',
    extra_issues: list[GroupingIssue] | None = None,
) -> DWIGrouping:
    """Assemble the full :class:`~.models.DWIGrouping` from indexed records.

    ``ignore_fieldmaps`` took effect during indexing; it is accepted here only
    so the recorded :class:`~.models.GroupingPolicy` is complete.
    """
    distortion_group_merge = distortion_group_merge or 'concat'
    if distortion_group_merge not in ('concat', 'average', 'none'):
        raise ValueError(
            f"distortion_group_merge must be 'concat', 'average', or 'none', "
            f'not {distortion_group_merge!r}.'
        )
    policy = GroupingPolicy(
        separate_all_dwis=separate_all_dwis,
        ignore_fieldmaps=ignore_fieldmaps,
        ignore_shims=ignore_shims,
        ignore_fov=ignore_fov,
        ignore_sdc=ignore_sdc,
        force_t2wreg=force_t2wreg,
        use_synb0=use_synb0,
        use_nipreps_syn_sdc=use_nipreps_syn_sdc,
        distortion_group_merge=distortion_group_merge,
    )
    issues = list(extra_issues or [])

    if ignore_sdc:
        # No susceptibility distortion correction at all: no fieldmaps, no
        # reverse-PE heuristic, no fieldmap-less fallback. Every series is left
        # uncorrected, but still grouped and concatenated for head-motion
        # correction.
        estimations = {}
        application = {record.path: None for record in records if record.is_dwi}
        app_provenance = {}
        candidates = {}
    else:
        estimations, targets, estimation_issues = resolve_estimations(records, ignore_shims)
        issues.extend(estimation_issues)

        application, app_provenance, candidates, application_issues = resolve_application(
            records, estimations, targets
        )
        issues.extend(application_issues)

        issues.extend(
            resolve_fieldmapless(
                records,
                estimations,
                application,
                app_provenance,
                candidates,
                force_t2wreg=force_t2wreg,
                use_synb0=use_synb0,
                use_nipreps_syn_sdc=use_nipreps_syn_sdc,
            )
        )

        # Drop heuristic estimations nothing references. Ones that lost an
        # application contest survive (so reports can show what "(also eligible)"
        # ids refer to); curated and translated ones are always kept and flagged,
        # since a person asked for them.
        applied_ids = {b0field_id for b0field_id in application.values() if b0field_id}
        candidate_ids = {b0field_id for ids in candidates.values() for b0field_id in ids}
        for b0field_id in sorted(set(estimations) - applied_ids):
            estimation = estimations[b0field_id]
            if estimation.provenance is Provenance.INFERRED:
                if b0field_id not in candidate_ids:
                    del estimations[b0field_id]
            else:
                issues.append(
                    warning(
                        'estimation-unused',
                        f"Fieldmap estimation '{b0field_id}' "
                        f'{estimation.provenance.tag()} does not correct any DWI series.',
                        estimation.sources,
                    )
                )

        # One estimation correcting several sessions is only reachable through
        # explicit session-less curation (a shared fmap/); honored, but worth
        # a warning since sessions usually mean a reshim.
        record_by_path = {record.path: record for record in records}
        applied_sessions = defaultdict(set)
        for path, chosen in application.items():
            if chosen is not None and record_by_path[path].session is not None:
                applied_sessions[chosen].add(record_by_path[path].session)
        for b0field_id, sessions in sorted(applied_sessions.items()):
            if len(sessions) > 1:
                sessions_txt = ', '.join(f'ses-{session}' for session in sorted(sessions, key=str))
                issues.append(
                    warning(
                        'cross-session-fieldmap-application',
                        f"Fieldmap estimation '{b0field_id}' corrects DWI series in "
                        f'multiple sessions ({sessions_txt}). The linkage is explicit '
                        'and will be honored, but sessions are usually re-shimmed - '
                        'verify that one fieldmap really applies to all of them.',
                        tuple(
                            sorted(
                                path
                                for path, chosen in application.items()
                                if chosen == b0field_id
                            )
                        ),
                    )
                )

    distortion_groups = build_distortion_groups(records, application, separate_all_dwis)

    for record in records:
        if record.is_dwi and record.signature.pe_dir is None:
            issues.append(
                warning(
                    'missing-pedir',
                    f'{record.filename} has no PhaseEncodingDirection; it cannot be '
                    'combined with other series or corrected with a PEPOLAR fieldmap.',
                    (record.path,),
                )
            )

    correction_units = build_correction_units(records, distortion_groups, separate_all_dwis)

    concatenation_groups, concat_issues = build_concatenation_groups(
        records,
        distortion_groups,
        correction_units,
        separate_all_dwis,
        distortion_group_merge,
    )
    issues.extend(concat_issues)

    from .validation import check_data_compatibility

    # Raw series are stacked per correction unit, so grid/FoV compatibility
    # is a unit-level requirement. Across units the final concatenation
    # happens after resampling to the output grid.
    issues.extend(
        check_data_compatibility(
            {record.path: record for record in records if record.is_dwi},
            correction_units,
            concatenation_groups,
            ignore_fov=ignore_fov,
        )
    )

    grouping = DWIGrouping(
        subject_id=subject_id,
        files={record.path: record for record in records},
        estimations=estimations,
        application=application,
        application_provenance=app_provenance,
        application_candidates=candidates,
        distortion_groups=distortion_groups,
        correction_units=correction_units,
        concatenation_groups=concatenation_groups,
        issues=issues,
        synb0_requested=use_synb0,
        policy=policy,
    )

    from .integrity import check_model_integrity

    violations = check_model_integrity(grouping)
    if violations:
        rendered = '\n  - '.join(violations)
        raise RuntimeError(
            'Internal grouping model inconsistency (this is a qsiprep bug, '
            f'not a data problem):\n  - {rendered}'
        )
    return grouping
