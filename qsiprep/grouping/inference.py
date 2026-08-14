"""Infer virtual B0FieldIdentifier/B0FieldSource/MultipartID values.

The grouping model is always expressed in BIDS curation vocabulary. When the
user curated their sidecars, those values are used verbatim. When they did
not, this module fills in equivalent values, with strict precedence:

1. **Curated** ``B0FieldIdentifier``/``B0FieldSource`` (step E1) always win.
2. **IntendedFor** on fmap/ files is translated into estimations (step E2).
3. A **heuristic** pairs reverse phase-encoded DWI series (step E3), which
   handles HCP-style acquisitions with zero curation.

The heuristic operates per (session, shim-compatible bucket), so a DWI series
with no reverse-PE partner of its own can still *borrow* compatible series
from elsewhere in the session for fieldmap estimation - even when those
series are concatenated into a different output. Estimation membership and
concatenation membership are independent by design.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .models import (
    AUTO_PREFIX,
    ConcatenationGroup,
    DistortionGroup,
    DWIGrouping,
    EstimationMethod,
    FieldmapEstimation,
    FileRecord,
    Provenance,
    derive_output_name,
    strip_nii_ext,
)
from .validation import GroupingIssue, error, warning

_METHOD_RANK = {
    EstimationMethod.PEPOLAR: 0,
    EstimationMethod.DIRECT: 1,
    EstimationMethod.PHASEDIFF: 2,
    EstimationMethod.PHASES: 3,
    EstimationMethod.SYNB0: 4,
    EstimationMethod.ANAT_CONTRAST: 5,
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


def _classify_method(records: list[FileRecord]) -> EstimationMethod | None:
    suffixes = {record.suffix for record in records}
    if 'fieldmap' in suffixes:
        return EstimationMethod.DIRECT
    if 'phasediff' in suffixes:
        return EstimationMethod.PHASEDIFF
    if 'phase1' in suffixes and 'phase2' in suffixes:
        return EstimationMethod.PHASES
    # An anatomical source marks a fieldmap-less registration estimation, even
    # when EPI files share the identifier (they are its registration movers).
    if suffixes.intersection(('T1w', 'T2w')):
        return EstimationMethod.ANAT_CONTRAST
    if any(record.is_epi_like for record in records):
        return EstimationMethod.PEPOLAR
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
    method: EstimationMethod,
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


def _unique_id(base: str, taken) -> str:
    if base not in taken:
        return base
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
        if method is EstimationMethod.PEPOLAR:
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
                        'no IntendedFor or B0FieldIdentifier linking it to a DWI series, '
                        'so it will not be used. Add B0FieldIdentifier/B0FieldSource '
                        '(preferred) or IntendedFor metadata to use it.',
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
        if method is EstimationMethod.PEPOLAR:
            key = frozenset(cluster_targets)
            merged_pepolar.setdefault(key, []).extend(members)
    translated_final = [
        (members, set(key), EstimationMethod.PEPOLAR)
        for key, members in sorted(merged_pepolar.items(), key=lambda kv: sorted(kv[0]))
    ] + [item for item in translated if item[2] is not EstimationMethod.PEPOLAR]

    for members, cluster_targets, method in translated_final:
        fmap_sources = list(members)
        if method is EstimationMethod.PEPOLAR:
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
    dwi_records = [record for record in records if record.is_dwi]
    for session, session_records in sorted(
        _by_session(dwi_records).items(), key=lambda kv: str(kv[0])
    ):
        shim_groups = _shim_groups(session_records, ignore_shims, issues)
        for shim_index, shim_records in enumerate(shim_groups):
            polarities = defaultdict(set)
            for record in shim_records:
                if record.signature.pe_axis:
                    polarities[record.signature.pe_axis].add(record.signature.pe_polarity)
            for axis in sorted(axis for axis, pols in polarities.items() if len(pols) == 2):
                members = [record for record in shim_records if record.signature.pe_axis == axis]
                id_parts = [AUTO_PREFIX + 'pepolar']
                if session:
                    id_parts.append(f'ses-{session}')
                if len(shim_groups) > 1:
                    id_parts.append(f'shim{shim_index + 1}')
                id_parts.append(axis)
                b0field_id = _unique_id('+'.join(id_parts), estimations)
                estimation = _make_estimation(
                    b0field_id, EstimationMethod.PEPOLAR, members, Provenance.INFERRED
                )
                estimations[b0field_id] = estimation
                # An inferred axis estimation corrects exactly the series it
                # was built from - including single-polarity "borrowers".
                targets[b0field_id] = {record.path for record in members}

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
):
    """Apply the fieldmap-less ladder to the application map (in place).

    Order: ``force_t2wreg`` overrides every DWI's fieldmap with a T2w
    registration (T2Wreg) estimation; ``use_synb0`` gives still-uncorrected
    series a SyNb0 synthetic-b=0 estimation; finally, uncorrected series in a
    subject with a T2w fall back to an inferred T2Wreg estimation (today's
    automatic TORTOISE behavior, made explicit).

    Anatomical estimations are created per session, with the anatomical
    image(s) as their only sources - the DWIs they correct are targets, since
    each output registers its own b=0. A DWI without a PhaseEncodingDirection
    cannot be corrected along an axis: it is skipped by the fallback and is a
    hard error when SyNb0 was explicitly requested.
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
        if use_synb0:
            issues.append(
                warning(
                    'synb0-overridden',
                    'Both T2Wreg and SyNb0 were requested; forcing T2Wreg wins '
                    'and SyNb0 is not used.',
                )
            )
        for session, paths in sorted(by_session.items(), key=lambda kv: str(kv[0])):
            if not _apply(
                paths, session, EstimationMethod.ANAT_CONTRAST, 't2wreg', 'T2w', Provenance.FORCED
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
                correctable, session, EstimationMethod.SYNB0, 'synb0', 'T1w', Provenance.FORCED
            ):
                issues.append(
                    error(
                        'synb0-requires-t1w',
                        'SyNb0 was requested, but this subject has no T1w image to '
                        'synthesize an undistorted b=0 from.',
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
                EstimationMethod.ANAT_CONTRAST,
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
    """Partition DWI files by (signature, applied estimation, curation walls).

    Curated MultipartIDs (and ``separate_all_dwis``) are part of the partition
    key so that a distortion group can never span two outputs.
    """
    dwi_records = [record for record in records if record.is_dwi]
    buckets = defaultdict(list)
    for record in dwi_records:
        if separate_all_dwis:
            wall = record.path
        else:
            wall = record.multipart_id
        buckets[(record.session, record.signature.key, application[record.path], wall)].append(
            record
        )

    groups: dict[str, DistortionGroup] = {}
    for (_, _, applied, _), members in sorted(buckets.items(), key=lambda kv: kv[1][0].path):
        key = _unique_id(derive_output_name([record.path for record in members]), groups)
        groups[key] = DistortionGroup(
            key=key,
            signature=members[0].signature,
            dwi_files=tuple(sorted(record.path for record in members)),
            b0field_source=applied,
        )
    return groups


def build_concatenation_groups(
    records: list[FileRecord],
    distortion_groups: dict[str, DistortionGroup],
    estimations: dict[str, FieldmapEstimation],
    separate_all_dwis: bool,
    ignore_shims: bool,
):
    """Decide which distortion groups are concatenated in the outputs."""
    issues: list[GroupingIssue] = []
    dwi_records = {record.path: record for record in records if record.is_dwi}
    curated_ids = {record.multipart_id for record in dwi_records.values() if record.multipart_id}

    # Assign each distortion group to a MultipartID
    assignments: dict[str, tuple[str, Provenance]] = {}  # dgroup key -> (id, provenance)

    if separate_all_dwis:
        if curated_ids:
            issues.append(
                warning(
                    'multipartid-overridden',
                    'separate_all_dwis is enabled, overriding the MultipartID values '
                    'in the sidecars: every DWI series will be a separate output.',
                )
            )
        for key, dgroup in distortion_groups.items():
            stem = _entity_stem(dgroup.dwi_files[0])
            assignments[key] = (AUTO_PREFIX + 'single+' + stem, Provenance.INFERRED)
    elif curated_ids:
        for key, dgroup in distortion_groups.items():
            multipart_id = dwi_records[dgroup.dwi_files[0]].multipart_id
            if multipart_id:
                if multipart_id.startswith(AUTO_PREFIX):
                    issues.append(
                        error(
                            'reserved-multipartid-prefix',
                            f"MultipartID '{multipart_id}' uses the reserved "
                            f"'{AUTO_PREFIX}' prefix. Rename it in your sidecars.",
                            dgroup.dwi_files,
                        )
                    )
                assignments[key] = (multipart_id, Provenance.CURATED)
            else:
                stem = _entity_stem(dgroup.dwi_files[0])
                assignments[key] = (AUTO_PREFIX + 'single+' + stem, Provenance.INFERRED)
    else:
        # Inferred: union-find over distortion groups, per session.
        parent = {key: key for key in distortion_groups}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(key_a, key_b):
            root_a, root_b = find(key_a), find(key_b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

        by_session = defaultdict(list)
        for key, dgroup in distortion_groups.items():
            by_session[dwi_records[dgroup.dwi_files[0]].session].append(key)

        loud_merges = []
        for session_keys in by_session.values():
            for key_a, key_b in combinations(sorted(session_keys), 2):
                group_a = distortion_groups[key_a]
                group_b = distortion_groups[key_b]
                same_estimation = (
                    group_a.b0field_source is not None
                    and group_a.b0field_source == group_b.b0field_source
                )
                both_corrected = (
                    group_a.b0field_source is not None
                    and group_b.b0field_source is not None
                    and group_a.signature.compatible_shim(
                        group_b.signature, ignore_shims=ignore_shims
                    )
                )
                identical_signature = (
                    group_a.signature.pe_dir is not None
                    and group_a.signature.key == group_b.signature.key
                )
                if same_estimation or both_corrected or identical_signature:
                    if both_corrected and not (same_estimation or identical_signature):
                        loud_merges.append((key_a, key_b))
                    union(key_a, key_b)

        components = defaultdict(list)
        for key in distortion_groups:
            components[find(key)].append(key)

        for merge_a, merge_b in loud_merges:
            if find(merge_a) == find(merge_b):
                issues.append(
                    warning(
                        'inferred-concat-merge',
                        f"Distortion groups '{merge_a}' and '{merge_b}' are corrected "
                        'by different fieldmaps but share a session and shim '
                        'settings, so their outputs will be concatenated. Use '
                        'MultipartID (or separate_all_dwis) to keep them separate.',
                        distortion_groups[merge_a].dwi_files
                        + distortion_groups[merge_b].dwi_files,
                    )
                )

        for index, (_, component) in enumerate(sorted(components.items(), key=lambda kv: kv[0])):
            session = dwi_records[distortion_groups[component[0]].dwi_files[0]].session
            id_parts = [AUTO_PREFIX + 'concat']
            if session:
                id_parts.append(f'ses-{session}')
            id_parts.append(str(index))
            for key in component:
                assignments[key] = ('+'.join(id_parts), Provenance.INFERRED)

    # Materialize the groups
    members_by_id = defaultdict(list)
    for key, (multipart_id, provenance) in assignments.items():
        members_by_id[multipart_id].append((key, provenance))

    concatenation_groups: dict[str, ConcatenationGroup] = {}
    output_names: dict[str, str] = {}
    for multipart_id, members in sorted(members_by_id.items()):
        keys = sorted(key for key, _ in members)
        dwi_files = sorted(path for key in keys for path in distortion_groups[key].dwi_files)
        output_name = derive_output_name(dwi_files)
        if output_name in output_names:
            issues.append(
                error(
                    'output-name-collision',
                    f"Two output groups ('{output_names[output_name]}' and "
                    f"'{multipart_id}') would both be named '{output_name}'. "
                    'Add distinguishing entities to the filenames or curate '
                    'MultipartID to give them distinct memberships.',
                    tuple(dwi_files),
                )
            )
        output_names[output_name] = multipart_id
        concatenation_groups[multipart_id] = ConcatenationGroup(
            multipart_id=multipart_id,
            provenance=members[0][1],
            distortion_groups=tuple(keys),
            dwi_files=tuple(dwi_files),
            output_name=output_name,
        )

    # An estimation whose targets span outputs is legal (borrowing), but the
    # user should know the same fieldmap will be estimated for each output.
    for b0field_id, estimation in sorted(estimations.items()):
        spanned = {
            multipart_id
            for multipart_id, concat in concatenation_groups.items()
            for key in concat.distortion_groups
            if distortion_groups[key].b0field_source == b0field_id
        }
        if len(spanned) > 1:
            issues.append(
                warning(
                    'estimation-spans-outputs',
                    f"Fieldmap estimation '{b0field_id}' corrects DWI series in "
                    f'{len(spanned)} different outputs; it will be estimated once '
                    'per output.',
                    estimation.sources,
                )
            )

    return concatenation_groups, issues


def build_grouping(
    records: list[FileRecord],
    subject_id: str,
    separate_all_dwis: bool = False,
    ignore_shims: bool = False,
    ignore_fov: bool = False,
    force_t2wreg: bool = False,
    use_synb0: bool = False,
    extra_issues: list[GroupingIssue] | None = None,
) -> DWIGrouping:
    """Assemble the full :class:`~.models.DWIGrouping` from indexed records."""
    issues = list(extra_issues or [])

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

    concatenation_groups, concat_issues = build_concatenation_groups(
        records, distortion_groups, estimations, separate_all_dwis, ignore_shims
    )
    issues.extend(concat_issues)

    from .validation import check_data_compatibility

    issues.extend(
        check_data_compatibility(
            {record.path: record for record in records if record.is_dwi},
            concatenation_groups,
            ignore_fov=ignore_fov,
        )
    )

    return DWIGrouping(
        subject_id=subject_id,
        files={record.path: record for record in records},
        estimations=estimations,
        application=application,
        application_provenance=app_provenance,
        application_candidates=candidates,
        distortion_groups=distortion_groups,
        concatenation_groups=concatenation_groups,
        issues=issues,
        synb0_requested=use_synb0,
    )
