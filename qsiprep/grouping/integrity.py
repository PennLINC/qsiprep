"""Internal consistency checks for a completed :class:`~.models.DWIGrouping`.

Model-integrity checks, not data validation: a violation means the grouping
implementation broke its own model, never that the user's data is wrong. The
checks hold even for groupings carrying error-severity issues; only
output-name uniqueness is suspended when ``output-name-collision`` is already
reported.
"""

from __future__ import annotations

from .models import DWIGrouping, Provenance

__all__ = ['check_model_integrity']


def check_model_integrity(grouping: DWIGrouping) -> list[str]:
    """Every violated model invariant, as human-readable strings (empty = OK)."""
    violations: list[str] = []
    files = grouping.files
    dwi_paths = {path for path, rec in files.items() if rec.is_dwi}

    # --- referential integrity -------------------------------------------
    for b0field_id, est in grouping.estimations.items():
        if est.b0field_id != b0field_id:
            violations.append(f"estimation key '{b0field_id}' != its id '{est.b0field_id}'")
        for src in est.sources:
            if src not in files:
                violations.append(f"estimation '{b0field_id}' sources unindexed file {src}")
    for path, chosen in grouping.application.items():
        if path not in dwi_paths:
            violations.append(f'application maps non-DWI path {path}')
        if chosen is not None and chosen not in grouping.estimations:
            violations.append(f"application of {path} names unknown estimation '{chosen}'")
    for path, ids in grouping.application_candidates.items():
        for b0field_id in ids:
            if b0field_id not in grouping.estimations:
                violations.append(f"candidate '{b0field_id}' for {path} is not an estimation")

    # --- distortion groups ------------------------------------------------
    for key, dgroup in grouping.distortion_groups.items():
        if dgroup.key != key:
            violations.append(f"distortion group key '{key}' != stored key '{dgroup.key}'")
        if not dgroup.dwi_files:
            violations.append(f"distortion group '{key}' is empty")
        if len(set(dgroup.dwi_files)) != len(dgroup.dwi_files):
            violations.append(f"distortion group '{key}' lists a file more than once")
        sessions = set()
        for path in dgroup.dwi_files:
            if path not in dwi_paths:
                violations.append(f"distortion group '{key}' member {path} is not an indexed DWI")
                continue
            sessions.add(files[path].session)
            if grouping.application.get(path) != dgroup.b0field_source:
                violations.append(
                    f"distortion group '{key}' correction "
                    f"'{dgroup.b0field_source}' != member application "
                    f"'{grouping.application.get(path)}' for {path}"
                )
            if (
                not grouping.policy.separate_all_dwis
                and dgroup.multipart_scope is not None
                and dgroup.multipart_scope not in files[path].multipart_id
            ):
                violations.append(
                    f"distortion group '{key}' scope '{dgroup.multipart_scope}' is not "
                    f'declared by member {path}'
                )
        if len(sessions) > 1:
            spanned = sorted(sessions, key=str)
            violations.append(f"distortion group '{key}' spans sessions {spanned}")

    # --- correction units -------------------------------------------------
    claimed_groups: list[str] = []
    for key, unit in grouping.correction_units.items():
        if unit.key != key:
            violations.append(f"correction unit key '{key}' != stored key '{unit.key}'")
        member_files: list[str] = []
        for dgroup_key in unit.distortion_groups:
            claimed_groups.append(dgroup_key)
            dgroup = grouping.distortion_groups.get(dgroup_key)
            if dgroup is None:
                violations.append(f"unit '{key}' references unknown group '{dgroup_key}'")
                continue
            member_files.extend(dgroup.dwi_files)
            if dgroup.b0field_source != unit.b0field_source:
                violations.append(
                    f"unit '{key}' correction '{unit.b0field_source}' != group "
                    f"'{dgroup_key}' correction '{dgroup.b0field_source}'"
                )
            if dgroup.multipart_scope != unit.multipart_scope:
                violations.append(
                    f"unit '{key}' scope '{unit.multipart_scope}' != group "
                    f"'{dgroup_key}' scope '{dgroup.multipart_scope}'"
                )
            for path in dgroup.dwi_files:
                if path in files and files[path].session != unit.session:
                    violations.append(
                        f"unit '{key}' session '{unit.session}' != member {path} "
                        f"session '{files[path].session}'"
                    )
        if tuple(sorted(member_files)) != unit.dwi_files:
            violations.append(f"unit '{key}' dwi_files != union of its distortion groups")
    if len(set(claimed_groups)) != len(claimed_groups):
        violations.append('a distortion group belongs to more than one correction unit')
    if set(claimed_groups) != set(grouping.distortion_groups):
        violations.append('correction units do not cover every distortion group exactly once')

    # --- concatenation groups ---------------------------------------------
    claimed_units: list[str] = []
    output_names: list[str] = []
    for key, concat in grouping.concatenation_groups.items():
        if concat.key != key:
            violations.append(f"concatenation key '{key}' != stored key '{concat.key}'")
        output_names.append(concat.output_name)
        member_groups: list[str] = []
        unit_sources: list[str | None] = []
        for unit_key in concat.correction_units:
            claimed_units.append(unit_key)
            unit = grouping.correction_units.get(unit_key)
            if unit is None:
                violations.append(f"output '{key}' references unknown unit '{unit_key}'")
                continue
            member_groups.extend(unit.distortion_groups)
            unit_sources.append(unit.b0field_source)
            if unit.session != concat.session:
                violations.append(
                    f"output '{key}' session '{concat.session}' != unit '{unit_key}' "
                    f"session '{unit.session}'"
                )
            if concat.provenance is Provenance.CURATED and unit.multipart_scope != (
                concat.multipart_id
            ):
                violations.append(
                    f"curated output '{key}' label '{concat.multipart_id}' != unit "
                    f"'{unit_key}' scope '{unit.multipart_scope}'"
                )
        if tuple(sorted(member_groups)) != concat.distortion_groups:
            violations.append(f"output '{key}' distortion_groups != union of its units")
        expected_files = tuple(
            sorted(
                path
                for dgroup_key in concat.distortion_groups
                if dgroup_key in grouping.distortion_groups
                for path in grouping.distortion_groups[dgroup_key].dwi_files
            )
        )
        if expected_files != concat.dwi_files:
            violations.append(f"output '{key}' dwi_files != union of its distortion groups")
        # Corrected and uncorrected units never share an inferred output.
        if concat.provenance is not Provenance.CURATED and len(unit_sources) > 1:
            if any(src is None for src in unit_sources) and any(
                src is not None for src in unit_sources
            ):
                violations.append(f"inferred output '{key}' mixes corrected/uncorrected units")
    if len(set(claimed_units)) != len(claimed_units):
        violations.append('a correction unit belongs to more than one output')
    if set(claimed_units) != set(grouping.correction_units):
        violations.append('outputs do not cover every correction unit exactly once')
    collision_reported = any(issue.code == 'output-name-collision' for issue in grouping.issues)
    if not collision_reported and len(set(output_names)) != len(output_names):
        violations.append('duplicate output_name without an output-name-collision error')

    # --- membership counts ------------------------------------------------
    # One output per declared MultipartID (virtual acquisitions duplicate on
    # purpose); exactly one with no MultipartID or under separate_all_dwis.
    appearances: dict[str, int] = dict.fromkeys(dwi_paths, 0)
    for concat in grouping.concatenation_groups.values():
        for path in concat.dwi_files:
            if path in appearances:
                appearances[path] += 1
    for path, count in appearances.items():
        record = files[path]
        if grouping.policy.separate_all_dwis or not record.multipart_id:
            expected = 1
        else:
            expected = len(set(record.multipart_id))
        if count != expected:
            violations.append(
                f'{path} appears in {count} outputs, expected {expected} '
                f'(MultipartID {list(record.multipart_id)!r})'
            )

    return violations
