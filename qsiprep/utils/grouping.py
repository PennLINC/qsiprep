# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to group scans based on their acquisition parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import logging
import os
import pprint
import warnings
from collections import defaultdict

from nipype.utils.filemanip import split_filename

from .. import config
from ..interfaces.bids import get_bids_params

LOGGER = logging.getLogger('nipype.workflow')

_DISTORTION_FIELDS = (
    'session',
    'PhaseEncodingDirection',
    'ShimSetting',
    'TotalReadoutTime',
    'B0FieldIdentifier',
)


def group_dwi_scans(
    layout,
    subject_data,
    combine_scans=True,
    ignore_fieldmaps=False,
    estimate_per_axis=False,
):
    """Determine groupings for DWI preprocessing.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
        A PyBIDS layout.
    subject_data : :obj:`dict`
        A dictionary of BIDS data for a single subject.
        The ``dwi`` key is required.
    combine_scans : :obj:`bool`, optional
        If True, group scans together based on their BIDS entities.
    ignore_fieldmaps : :obj:`bool`, optional
        If True, do not use files in ``fmap/`` for distortion correction.
    estimate_per_axis : :obj:`bool`, optional
        If True, limit PE direction–based grouping to reverse-PED pairs only.

    Returns
    -------
    distortion_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of raw DWI file paths.
    fmap_estimation_groups : dict[str, list[str]]
        Keys are field-map identifiers, values are lists mixing distortion-group
        IDs (for DWI data) and raw file paths (for fmap data).
    fmap_application_groups : dict[str, list[str]]
        Keys are field-map identifiers, values are distortion-group IDs that
        will be corrected by that field map.
    concatenation_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of distortion-group IDs.
    """
    config.loggers.workflow.info('Grouping DWI scans')

    distortion_groups = build_distortion_groups(layout, subject_data, combine_scans)

    fmap_estimation_groups = build_fmap_estimation_groups(
        layout,
        subject_data,
        distortion_groups,
        ignore_fieldmaps,
        estimate_per_axis,
    )

    fmap_application_groups = build_fmap_application_groups(
        layout,
        subject_data,
        distortion_groups,
        fmap_estimation_groups,
    )

    concatenation_groups = build_concatenation_groups(
        layout,
        subject_data,
        distortion_groups,
        combine_scans,
    )

    validate_group_consistency(
        distortion_groups,
        fmap_estimation_groups,
        concatenation_groups,
        combine_scans,
    )

    initial_distortion_groups = distortion_groups
    distortion_groups = refine_distortion_groups(distortion_groups, fmap_estimation_groups)
    if distortion_groups != initial_distortion_groups:
        (
            fmap_estimation_groups,
            fmap_application_groups,
            concatenation_groups,
        ) = _remap_groups_after_refinement(
            initial_distortion_groups,
            distortion_groups,
            fmap_estimation_groups,
            fmap_application_groups,
            concatenation_groups,
        )

    config.loggers.workflow.info('Finished grouping DWI scans')
    return distortion_groups, fmap_estimation_groups, fmap_application_groups, concatenation_groups


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def build_distortion_groups(layout, subject_data, combine_scans):
    """Group DWI files that share distortion-relevant metadata.

    Files are grouped by (session, PhaseEncodingDirection, ShimSetting,
    TotalReadoutTime).  When *combine_scans* is False every file becomes its
    own singleton group.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
        A PyBIDS layout.
    subject_data : :obj:`dict`
        Must contain a ``dwi`` key with a list of DWI file paths.
    combine_scans : :obj:`bool`
        If False, each file is its own distortion group.

    Returns
    -------
    distortion_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of raw DWI file paths.
    """
    all_dwis = subject_data['dwi']

    if not combine_scans:
        names = _get_unique_concatenated_bids_names([[dwi] for dwi in all_dwis])
        return {name: [dwi] for name, dwi in zip(names, all_dwis, strict=False)}

    dwi_info = {}
    for dwi in all_dwis:
        metadata = layout.get_file(dwi).get_metadata()
        session = layout.get_file(dwi).entities.get('session')
        shim = metadata.get('ShimSetting')
        dwi_info[dwi] = {
            'session': session,
            'PhaseEncodingDirection': metadata.get('PhaseEncodingDirection'),
            'ShimSetting': tuple(shim) if isinstance(shim, list) else shim,
            'TotalReadoutTime': metadata.get('TotalReadoutTime'),
            'B0FieldIdentifier': tuple(_ensure_list(metadata.get('B0FieldIdentifier')))
            if metadata.get('B0FieldIdentifier') is not None
            else None,
        }

    groups = []
    group_exemplars = []
    for dwi in all_dwis:
        info = dwi_info[dwi]
        matched = False
        for idx, exemplar_info in enumerate(group_exemplars):
            if all(info[k] == exemplar_info[k] for k in _DISTORTION_FIELDS):
                groups[idx].append(dwi)
                matched = True
                break

        if not matched:
            groups.append([dwi])
            group_exemplars.append(info)

    names = _get_unique_concatenated_bids_names(groups)
    return dict(zip(names, groups, strict=False))


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def build_fmap_estimation_groups(
    layout,
    subject_data,
    distortion_groups,
    ignore_fieldmaps,
    estimate_per_axis,
):
    """Determine which files contribute to each field-map estimation.

    Priority:
    1. B0FieldIdentifier present on any file.
    2. IntendedFor present on fmap files (no B0FieldIdentifier).
    3. Heuristic pairing of DWI distortion groups by PE direction.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    subject_data : :obj:`dict`
    distortion_groups : dict[str, list[str]]
    ignore_fieldmaps : :obj:`bool`
    estimate_per_axis : :obj:`bool`

    Returns
    -------
    fmap_estimation_groups : dict[str, list[str]]
        Keys are field-map identifiers, values are lists mixing distortion-group
        IDs and raw fmap file paths.
    """
    file_to_dg = _build_file_to_dg_map(distortion_groups)
    all_dwis = subject_data['dwi']

    fmap_files = _get_fmap_files(layout, all_dwis, ignore_fieldmaps)

    # ------------------------------------------------------------------
    # Path 1: B0FieldIdentifier
    # ------------------------------------------------------------------
    b0field_groups = defaultdict(set)
    has_b0field = False

    for dwi in all_dwis:
        b0fi = _get_metadata_field(layout, dwi, 'B0FieldIdentifier')
        if b0fi is not None:
            has_b0field = True
            for val in _ensure_list(b0fi):
                b0field_groups[val].add(file_to_dg[dwi])

    for fmap_file in fmap_files:
        b0fi = _get_metadata_field(layout, fmap_file, 'B0FieldIdentifier')
        if b0fi is not None:
            has_b0field = True
            for val in _ensure_list(b0fi):
                b0field_groups[val].add(fmap_file)

    if has_b0field:
        _check_b0field_axis_conflict(
            b0field_groups,
            distortion_groups,
            layout,
            estimate_per_axis,
        )
        grouped = _split_named_member_groups_by_session(
            layout,
            distortion_groups,
            {k: sorted(v) for k, v in b0field_groups.items()},
        )
        _check_distortion_metadata_compatibility(
            layout,
            distortion_groups,
            grouped,
            group_kind='field-map estimation',
        )
        return grouped

    # ------------------------------------------------------------------
    # Path 2: IntendedFor on fmap files
    # ------------------------------------------------------------------
    if fmap_files:
        intended_groups = _build_intendedfor_groups(
            layout,
            fmap_files,
            all_dwis,
            file_to_dg,
        )
        if intended_groups:
            grouped = _split_named_member_groups_by_session(
                layout,
                distortion_groups,
                intended_groups,
            )
            _check_distortion_metadata_compatibility(
                layout,
                distortion_groups,
                grouped,
                group_kind='field-map estimation',
            )
            return grouped

    # ------------------------------------------------------------------
    # Path 3: Heuristic — pair distortion groups by PE direction
    # ------------------------------------------------------------------
    return _build_heuristic_estimation_groups(
        layout,
        distortion_groups,
        estimate_per_axis,
    )


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------


def build_fmap_application_groups(
    layout,
    subject_data,
    distortion_groups,
    fmap_estimation_groups,
):
    """Map each field-map identifier to the distortion groups it will correct.

    Priority:
    1. B0FieldSource on DWI files.
    2. IntendedFor on fmap files.
    3. Heuristic — same members as the estimation group (all are both
       sources and targets).

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    subject_data : :obj:`dict`
    distortion_groups : dict[str, list[str]]
    fmap_estimation_groups : dict[str, list[str]]

    Returns
    -------
    fmap_application_groups : dict[str, list[str]]
        Keys are field-map identifiers, values are lists of distortion-group
        IDs that will be corrected.
    """
    all_dwis = subject_data['dwi']
    file_to_dg = _build_file_to_dg_map(distortion_groups)

    # Path 1: B0FieldSource
    b0source_map = defaultdict(set)
    has_b0source = False
    for dwi in all_dwis:
        b0fs = _get_metadata_field(layout, dwi, 'B0FieldSource')
        if b0fs is not None:
            has_b0source = True
            for val in _ensure_list(b0fs):
                b0source_map[val].add(file_to_dg[dwi])

    if has_b0source:
        grouped = _split_named_member_groups_by_session(
            layout,
            distortion_groups,
            {k: sorted(v) for k, v in b0source_map.items()},
        )
        _check_distortion_metadata_compatibility(
            layout,
            distortion_groups,
            grouped,
            group_kind='field-map application',
        )
        return grouped

    # Path 2 / 3: Derive from estimation groups — every DWI distortion group
    # that appears in an estimation group is also an application target.
    app_groups = {}
    for key, members in fmap_estimation_groups.items():
        dg_ids = [m for m in members if m in distortion_groups]
        if dg_ids:
            app_groups[key] = sorted(dg_ids)

    return app_groups


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------


def build_concatenation_groups(layout, subject_data, distortion_groups, combine_scans):
    """Group distortion groups into final output concatenations.

    Priority:
    1. combine_scans=False → each distortion group is its own concatenation group.
    2. MultipartID present → group distortion groups whose files share MultipartID.
    3. Otherwise → group all distortion groups within the same session.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    subject_data : :obj:`dict`
    distortion_groups : dict[str, list[str]]
    combine_scans : :obj:`bool`

    Returns
    -------
    concatenation_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of distortion-group IDs.
    """
    all_dwis = subject_data['dwi']
    has_multipart = any(
        _get_metadata_field(layout, dwi, 'MultipartID') is not None for dwi in all_dwis
    )

    if not combine_scans:
        if has_multipart:
            warnings.warn(
                'MultipartID metadata is present but combine_scans is False. '
                'Each DWI file will remain a separate group.',
                UserWarning,
                stacklevel=2,
            )
        return {dg_id: [dg_id] for dg_id in distortion_groups}

    if has_multipart:
        return _build_multipartid_concatenation_groups(layout, distortion_groups)

    return _build_session_concatenation_groups(layout, distortion_groups)


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------


def validate_group_consistency(
    distortion_groups,
    fmap_estimation_groups,
    concatenation_groups,
    combine_scans,
):
    """Check that groups are consistent with each other.

    Raises
    ------
    ValueError
        If a field-map estimation group is not a subset of exactly one
        concatenation group.
    """
    if not fmap_estimation_groups:
        return

    # When scans are intentionally kept separate, field-map estimation groups
    # can legitimately span many singleton concatenation groups.
    if not combine_scans:
        return

    dg_to_concat = defaultdict(set)
    for cg_id, dg_ids in concatenation_groups.items():
        for dg_id in dg_ids:
            dg_to_concat[dg_id].add(cg_id)

    # A distortion group should belong to exactly one concatenation group.
    for dg_id, cg_ids in dg_to_concat.items():
        if len(cg_ids) > 1:
            raise ValueError(
                f'Distortion group {dg_id!r} appears in multiple concatenation groups '
                f'({cg_ids}). A field-map estimation group must be a subset of exactly one '
                f'concatenation group.'
            )

    for fme_key, fme_members in fmap_estimation_groups.items():
        fme_dg_ids = {m for m in fme_members if m in distortion_groups}
        if not fme_dg_ids:
            continue

        concat_ids = set()
        for dg_id in fme_dg_ids:
            concat_ids.update(dg_to_concat.get(dg_id, set()))
        if len(concat_ids) > 1:
            raise ValueError(
                f'Field-map estimation group {fme_key!r} spans multiple '
                f'concatenation groups ({concat_ids}). A field-map estimation '
                f'group must be a subset of exactly one concatenation group.'
            )


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------


def refine_distortion_groups(distortion_groups, fmap_estimation_groups):
    """Split distortion groups that span multiple field-map estimation groups.

    If a distortion group contains files assigned to different estimation
    groups (e.g. because B0FieldIdentifier splits runs), the distortion
    group is split so each piece falls entirely within one estimation group.

    Parameters
    ----------
    distortion_groups : dict[str, list[str]]
    fmap_estimation_groups : dict[str, list[str]]

    Returns
    -------
    refined : dict[str, list[str]]
    """
    if not fmap_estimation_groups:
        return distortion_groups

    dg_to_fme = defaultdict(set)
    for fme_key, fme_members in fmap_estimation_groups.items():
        for member in fme_members:
            if member in distortion_groups:
                dg_to_fme[member].add(fme_key)

    needs_split = {dg_id for dg_id, fme_keys in dg_to_fme.items() if len(fme_keys) > 1}
    if not needs_split:
        return distortion_groups

    file_to_fme = {}
    for fme_key, fme_members in fmap_estimation_groups.items():
        for member in fme_members:
            if member not in distortion_groups:
                continue
            for dwi in distortion_groups[member]:
                file_to_fme[dwi] = fme_key

    refined = {}
    for dg_id, files in distortion_groups.items():
        if dg_id not in needs_split:
            refined[dg_id] = files
            continue

        sub_groups = defaultdict(list)
        for dwi in files:
            fme_key = file_to_fme.get(dwi, 'unknown')
            sub_groups[fme_key].append(dwi)

        sub_group_lists = list(sub_groups.values())
        sub_names = _get_unique_concatenated_bids_names(sub_group_lists)
        for name, sub_files in zip(sub_names, sub_group_lists, strict=False):
            refined[name] = sub_files

    return refined


# ---------------------------------------------------------------------------
# Internal helpers for the stage functions
# ---------------------------------------------------------------------------


def _build_file_to_dg_map(distortion_groups):
    """Return a mapping from DWI file path → distortion-group ID."""
    file_to_dg = {}
    for dg_id, files in distortion_groups.items():
        for f in files:
            file_to_dg[f] = dg_id
    return file_to_dg


def _get_fmap_files(layout, dwi_files, ignore_fieldmaps):
    """Return fmap/ NIfTI files for the same subject, or [] if ignored."""
    if ignore_fieldmaps or not dwi_files:
        return []

    sub_id = layout.get_file(dwi_files[0]).entities.get('subject')
    try:
        files = layout.get(
            subject=sub_id,
            datatype='fmap',
            extension=['.nii.gz', '.nii'],
            return_type='file',
        )
    except Exception:
        files = []

    return sorted(files)


def _get_metadata_field(layout, filepath, field):
    """Safely read a single metadata field from a BIDS file."""
    try:
        return layout.get_file(filepath).get_metadata().get(field)
    except Exception:
        return None


def _ensure_list(value):
    """Wrap scalars in a list; pass through lists unchanged."""
    if isinstance(value, list):
        return value
    return [value]


def _get_pe_axis(pe_dir):
    """Extract the axis letter from a PhaseEncodingDirection string."""
    if pe_dir is None:
        return None
    return pe_dir.rstrip('-')


def _dg_pe_direction(layout, distortion_groups, dg_id):
    """Return the PhaseEncodingDirection of a distortion group's exemplar file."""
    exemplar = distortion_groups[dg_id][0]
    return _get_metadata_field(layout, exemplar, 'PhaseEncodingDirection')


def _check_b0field_axis_conflict(
    b0field_groups,
    distortion_groups,
    layout,
    estimate_per_axis,
):
    """Raise if estimate_per_axis=True and a B0FieldIdentifier spans PE axes."""
    if not estimate_per_axis:
        return

    for b0fi_key, members in b0field_groups.items():
        axes = set()
        for member in members:
            if member in distortion_groups:
                pe = _dg_pe_direction(layout, distortion_groups, member)
            else:
                pe = _get_metadata_field(layout, member, 'PhaseEncodingDirection')

            axis = _get_pe_axis(pe)
            if axis is not None:
                axes.add(axis)

        if len(axes) > 1:
            raise ValueError(
                f'B0FieldIdentifier {b0fi_key!r} groups files across PE axes '
                f'{axes}, but estimate_per_axis is True. A single B0 field '
                f'estimation group cannot span multiple axis pairs.'
            )


def _resolve_intended_for(intended_for_paths, dwi_files):
    """Map IntendedFor target paths to actual DWI file paths."""
    basename_to_file = {os.path.basename(f): f for f in dwi_files}
    resolved = []
    for target in intended_for_paths:
        if target.startswith('bids::'):
            target = target[len('bids::') :]
        target_basename = os.path.basename(target)
        if target_basename in basename_to_file:
            resolved.append(basename_to_file[target_basename])
    return resolved


def _build_intendedfor_groups(layout, fmap_files, all_dwis, file_to_dg):
    """Build estimation groups from IntendedFor metadata on fmap files."""
    raw_groups = {}
    auto_counter = 0

    for fmap_file in fmap_files:
        intended_for = _get_metadata_field(layout, fmap_file, 'IntendedFor')
        if intended_for is None:
            continue

        intended_for = _ensure_list(intended_for)
        targeted_dwis = _resolve_intended_for(intended_for, all_dwis)
        targeted_dg_ids = sorted({file_to_dg[f] for f in targeted_dwis if f in file_to_dg})
        if not targeted_dg_ids:
            continue

        target_key = tuple(targeted_dg_ids)
        if target_key not in raw_groups:
            key = f'auto_{auto_counter:05d}'
            auto_counter += 1
            raw_groups[target_key] = (key, set(targeted_dg_ids))

        raw_groups[target_key][1].add(fmap_file)

    if not raw_groups:
        return {}

    return {key: sorted(members) for _, (key, members) in raw_groups.items()}


def _build_heuristic_estimation_groups(layout, distortion_groups, estimate_per_axis):
    """Pair distortion groups by PE direction when no curator metadata exists."""
    fme_groups = {}
    auto_counter = 0

    # Never create heuristic estimation groups across sessions.
    dgs_by_session = defaultdict(list)
    for dg_id, files in distortion_groups.items():
        session = layout.get_file(files[0]).entities.get('session')
        dgs_by_session[session].append(dg_id)

    for session in sorted(dgs_by_session):
        session_dg_ids = dgs_by_session[session]

        # In fully automatic mode (no curator-defined fmap links), do not build
        # heuristic fmap groups across MultipartID boundaries.
        multipart_partitions = defaultdict(list)
        for dg_id in session_dg_ids:
            multipart_key = _get_distortion_group_multipart_key(layout, distortion_groups, dg_id)
            partition_key = multipart_key if multipart_key is not None else '_no_multipart'
            multipart_partitions[partition_key].append(dg_id)

        for partition_dg_ids in multipart_partitions.values():
            pe_map = {}
            for dg_id in partition_dg_ids:
                pe_map[dg_id] = _dg_pe_direction(layout, distortion_groups, dg_id)

            pe_axes = defaultdict(lambda: defaultdict(list))
            for dg_id, pe in pe_map.items():
                if pe is None:
                    continue
                axis = _get_pe_axis(pe)
                pe_axes[axis][pe].append(dg_id)

            if estimate_per_axis:
                for axis, dir_map in sorted(pe_axes.items()):
                    plus_dir = axis
                    minus_dir = axis + '-'
                    plus_dgs = dir_map.get(plus_dir, [])
                    minus_dgs = dir_map.get(minus_dir, [])
                    if plus_dgs and minus_dgs:
                        key = f'auto_{auto_counter:05d}'
                        auto_counter += 1
                        fme_groups[key] = sorted(plus_dgs + minus_dgs)
            else:
                all_dgs_with_pe = [dg_id for dg_id, pe in pe_map.items() if pe is not None]
                unique_pes = {pe for pe in pe_map.values() if pe is not None}
                if len(unique_pes) >= 2 and all_dgs_with_pe:
                    key = f'auto_{auto_counter:05d}'
                    auto_counter += 1
                    fme_groups[key] = sorted(all_dgs_with_pe)

    return fme_groups


def _build_multipartid_concatenation_groups(layout, distortion_groups):
    """Group distortion groups by MultipartID metadata."""
    mp_groups = defaultdict(set)
    none_counter = 0

    for dg_id, files in distortion_groups.items():
        session = layout.get_file(files[0]).entities.get('session')
        mp_ids = set()
        for dwi in files:
            mp = _get_metadata_field(layout, dwi, 'MultipartID')
            if mp is not None:
                mp_ids.add(mp)

        if mp_ids:
            for mp_id in mp_ids:
                mp_groups[(session, mp_id)].add(dg_id)
        else:
            mp_groups[(session, f'_none_{none_counter}')] = {dg_id}
            none_counter += 1

    all_dg_lists = [sorted(dg_ids) for dg_ids in mp_groups.values()]
    all_file_lists = []
    for dg_ids in all_dg_lists:
        files = []
        for dg_id in dg_ids:
            files.extend(distortion_groups[dg_id])
        all_file_lists.append(files)

    names = _get_unique_concatenated_bids_names(all_file_lists)
    return dict(zip(names, all_dg_lists, strict=False))


def _build_session_concatenation_groups(layout, distortion_groups):
    """Group all distortion groups within the same session."""
    session_dgs = defaultdict(list)
    for dg_id, files in distortion_groups.items():
        session = layout.get_file(files[0]).entities.get('session')
        session_dgs[session].append(dg_id)

    all_dg_lists = [sorted(dg_ids) for dg_ids in session_dgs.values()]
    all_file_lists = []
    for dg_ids in all_dg_lists:
        files = []
        for dg_id in dg_ids:
            files.extend(distortion_groups[dg_id])
        all_file_lists.append(files)

    names = _get_unique_concatenated_bids_names(all_file_lists)
    return dict(zip(names, all_dg_lists, strict=False))


def _get_member_session(layout, distortion_groups, member):
    """Return session entity for a group member (DG ID or file path)."""
    if member in distortion_groups:
        exemplar = distortion_groups[member][0]
        return layout.get_file(exemplar).entities.get('session')
    return layout.get_file(member).entities.get('session')


def _split_named_member_groups_by_session(layout, distortion_groups, named_groups):
    """Split named groups so members never span sessions.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    distortion_groups : dict[str, list[str]]
    named_groups : dict[str, list[str]]
        Group-keyed members, where members are DG IDs and/or file paths.

    Returns
    -------
    dict[str, list[str]]
        Same structure as input, but any cross-session group is split into
        per-session groups with deterministic suffixed keys.
    """
    split_groups = {}
    for key, members in named_groups.items():
        by_session = defaultdict(list)
        for member in members:
            by_session[_get_member_session(layout, distortion_groups, member)].append(member)

        if len(by_session) <= 1:
            split_groups[key] = sorted(members)
            continue

        for session in sorted(by_session, key=lambda s: '' if s is None else str(s)):
            session_suffix = f'ses-{session}' if session is not None else 'ses-none'
            split_groups[f'{key}_{session_suffix}'] = sorted(by_session[session])

    return split_groups


def _get_distortion_group_multipart_key(layout, distortion_groups, dg_id):
    """Return a deterministic MultipartID key for a distortion group."""
    mp_ids = set()
    for dwi in distortion_groups[dg_id]:
        mp = _get_metadata_field(layout, dwi, 'MultipartID')
        if mp is not None:
            mp_ids.add(mp)

    if not mp_ids:
        return None
    if len(mp_ids) == 1:
        return next(iter(mp_ids))
    return tuple(sorted(mp_ids))


def _get_distortion_group_signature(layout, distortion_groups, dg_id):
    """Return (ShimSetting, TotalReadoutTime) for a distortion group exemplar."""
    exemplar = distortion_groups[dg_id][0]
    metadata = layout.get_file(exemplar).get_metadata()
    shim = metadata.get('ShimSetting')
    shim = tuple(shim) if isinstance(shim, list) else shim
    trt = metadata.get('TotalReadoutTime')
    return shim, trt


def _check_distortion_metadata_compatibility(
    layout,
    distortion_groups,
    named_groups,
    group_kind,
):
    """Raise if a metadata-defined group spans incompatible shim/TRT signatures."""
    for group_key, members in named_groups.items():
        dg_members = [m for m in members if m in distortion_groups]
        if len(dg_members) <= 1:
            continue

        signatures = {
            _get_distortion_group_signature(layout, distortion_groups, dg_id)
            for dg_id in dg_members
        }
        if len(signatures) > 1:
            raise ValueError(
                f'{group_kind.capitalize()} group {group_key!r} contains distortion groups '
                'with conflicting ShimSetting and/or TotalReadoutTime metadata. '
                'These files cannot be grouped together for field-map processing.'
            )


def _remap_group_members_with_new_dg_ids(group_members, old_to_new):
    """Replace old DG IDs in a member list with remapped new DG IDs."""
    remapped = []
    for member in group_members:
        if member in old_to_new:
            remapped.extend(sorted(old_to_new[member]))
        else:
            remapped.append(member)
    return sorted(set(remapped))


def _remap_groups_after_refinement(
    old_distortion_groups,
    new_distortion_groups,
    fmap_estimation_groups,
    fmap_application_groups,
    concatenation_groups,
):
    """Remap DG IDs in returned groups after distortion-group refinement."""
    file_to_new_dg = {}
    for new_dg_id, files in new_distortion_groups.items():
        for dwi in files:
            file_to_new_dg[dwi] = new_dg_id

    old_to_new = {}
    for old_dg_id, files in old_distortion_groups.items():
        old_to_new[old_dg_id] = {file_to_new_dg[dwi] for dwi in files if dwi in file_to_new_dg}

    remapped_fme = {}
    for key, members in fmap_estimation_groups.items():
        remapped_fme[key] = _remap_group_members_with_new_dg_ids(members, old_to_new)

    remapped_fma = {}
    for key, members in fmap_application_groups.items():
        remapped_fma[key] = _remap_group_members_with_new_dg_ids(members, old_to_new)

    remapped_cg = {}
    for key, members in concatenation_groups.items():
        remapped_cg[key] = _remap_group_members_with_new_dg_ids(members, old_to_new)

    return remapped_fme, remapped_fma, remapped_cg


def get_entity_groups(layout, subject_data, combine_all_dwis):
    """Handle the grouping of multiple DWI files.

    This function will group DWI files based on the MultipartID metadata field,
    when available, and will default to considering all DWIs in a session as a
    "group" when it is not.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
        A PyBIDS layout
    subject_data : :obj:`dict`
        A dictionary of BIDS data for a single subject
    combine_all_dwis : :obj:`bool`
        If True, combine all DWI files within a session into a single group

    Returns
    -------
    dwi_entity_groups : :obj:`list` of :obj:`list`
        A list of lists of DWI files.
        Each list of DWI files is a group of scans that can be concatenated together.
    """
    all_dwis = subject_data['dwi']
    if not combine_all_dwis:
        dwi_groups = [[dwi] for dwi in all_dwis]
        return dwi_groups

    all_metadata = []
    for f in all_dwis:
        metadata = layout.get_file(f).get_metadata()
        all_metadata.append(metadata)

    grouping_method = 'entities'  # default is to group by entities
    if any('MultipartID' in metadata for metadata in all_metadata):
        grouping_method = 'metadata'

    dwi_entities = {}
    grouping_metadata = {}
    for i_file, f in enumerate(all_dwis):
        if grouping_method == 'metadata':
            # One MultipartID in any DWI metadata file means we use MultipartID to group
            # If a DWI has no MultipartID, it will be placed in a group by itself
            grouping_metadata[f] = all_metadata[i_file].get('MultipartID', None)
        else:
            # Group by entity instead
            f_entities = layout.get_file(f).get_entities()
            for k, v in f_entities.items():
                if k in dwi_entities:
                    if v not in dwi_entities[k]:
                        dwi_entities[k].append(v)
                else:
                    dwi_entities[k] = [v]

    if grouping_method == 'metadata':
        # Overwrite the existing dwi_groups (list) with a dict of lists
        LOGGER.info('Using MultipartID to group DWI files')
        dwi_groups = {}
        none_counter = 0
        for f in all_dwis:
            group = grouping_metadata[f]
            if group is None:
                # ! is not common, so it should be safe to use here without
                # conflicting with a valid MultipartID
                group = f'!none{none_counter}'
                none_counter += 1

            if group not in dwi_groups:
                dwi_groups[group] = []

            dwi_groups[group].append(f)

        for multipart_id, group_files in dwi_groups.items():
            if multipart_id.startswith('!'):
                LOGGER.info(
                    '\t- %d scan without MultipartID (set to %s)',
                    len(group_files),
                    multipart_id[1:],
                )
            else:
                LOGGER.info(
                    '\t- %d scans with MultipartID %s',
                    len(group_files),
                    multipart_id,
                )

        # Convert to list of lists
        dwi_groups = list(dwi_groups.values())

    elif grouping_method == 'entities':
        LOGGER.info('Combining all DWI files within each available session')
        # Group by session
        dwi_groups = []
        sessions = dwi_entities.get('session', [None])
        for session in sessions:
            session_files = [
                img for img in all_dwis if layout.get_file(img).entities.get('session') == session
            ]

            LOGGER.info(
                '\t- %d scans in session %s',
                len(session_files),
                session,
            )
            dwi_groups.append(session_files)

    return dwi_groups


FMAP_PRIORITY = {
    'dwi': 0,
    'epi': 1,
    'fieldmap': 2,
    'phasediff': 3,
    'phase1': 4,
    'phase': 4,
    'syn': 5,
}


def get_highest_priority_fieldmap(fmap_infos):
    """Return a dictionary describing the highest priority fieldmap.

    Parameters
    ----------
    fmap_infos : :obj:`list` of :obj:`dict`
        A list of dictionaries describing fieldmaps. Each dictionary must have a
        ``suffix`` key and may have an ``epi`` key.

    Returns
    -------
    selected_fmap_info : :obj:`dict`
        The dictionary describing the highest priority fieldmap.
        This will be the entry from ``fmap_infos`` with the highest priority value.
        If no fieldmaps are found, the dictionary will have a ``suffix`` key with a
        value of ``None``.

    Examples
    --------
    Invent some potential fieldmaps
    >>> epi_fmap1 = {"epi": "/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz", "suffix": "epi"}
    >>> epi_fmap2 = {"epi": "/data/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz", "suffix": "epi"}
    >>> epi_fmap3 = {"epi": "/data/sub-1/fmap/sub-1_dir-PA_epi.nii.gz", "suffix": "epi"}
    >>>
    >>> phasediff_fmap = {"phasediff": "/data/sub-1/fmap/sub-1_phasediff.nii.gz",
    ...                   "suffix": "phasediff"}
    >>> phases_fmap = {"phase1": "/data/sub-1/fmap/sub-1_phase1.nii.gz",
    ...                "suffix": "phase1"}
    >>>
    >>> dwi_fmap1 = {"dwi": "/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz", "suffix": "dwi"}
    >>> dwi_fmap2 = {'suffix': 'dwi',
    ...   'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']}

    When there are no fieldmaps in ``fmaps/``, but a reverse PE DWI series
    >>> get_highest_priority_fieldmap([dwi_fmap1])
    {'dwi': '/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz', 'suffix': 'dwi'}

    There is both an epi fieldmap and a phase1/phase2 GRE fieldmap
    >>> get_highest_priority_fieldmap([epi_fmap1, phases_fmap])
    {'suffix': 'epi', 'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz']}

    Multiple EPI fieldmaps
    >>> get_highest_priority_fieldmap(
    ...     [epi_fmap1, epi_fmap2, epi_fmap3]) # doctest: +NORMALIZE_WHITESPACE
    {'suffix': 'epi',
     'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz',
             '/data/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
             '/data/sub-1/fmap/sub-1_dir-PA_epi.nii.gz']}

    An EPI fieldmap from ``fmap/`` should be chosen over a reverse PE DWI series
    >>> get_highest_priority_fieldmap([epi_fmap1, dwi_fmap2])
    {'suffix': 'epi', 'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz']}

    """
    # Find fieldmaps
    default_priority = max(FMAP_PRIORITY.values()) + 1
    priority = default_priority
    selected_fmap_info = {'suffix': None}

    # collapse multiple EPI fieldmaps into one entry
    epi_fmaps = sorted(
        [fmap_info['epi'] for fmap_info in fmap_infos if fmap_info.get('suffix') == 'epi']
    )
    if epi_fmaps:
        epi_info = {'suffix': 'epi', 'epi': epi_fmaps}
        fmap_infos = [
            fmap_info for fmap_info in fmap_infos if fmap_info.get('suffix') != 'epi'
        ] + [epi_info]

    # Select the highest priority fieldmap
    for fmap_info in fmap_infos:
        if fmap_info.get('suffix') == 'phase':
            fmap_info['suffix'] = 'phase1'

        fmap_type = fmap_info.get('suffix')
        if fmap_type not in FMAP_PRIORITY:
            continue

        this_priority = FMAP_PRIORITY[fmap_type]
        if this_priority < priority:
            priority = this_priority
            selected_fmap_info = fmap_info

    return selected_fmap_info


def find_fieldmaps_from_other_dwis(dwi_files, dwi_file_metadatas, estimate_per_axis):
    """Find a list of files in the dwi/ directory that can be used for distortion correction.

    It is common to acquire DWI scans with opposite phase encoding directions so they can be
    used to correct each other's EPI distortion. There is currently no mechanism in BIDS to
    specify whether b=0 scans in dwi/ can be used as fieldmaps for one another.

    Parameters
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.
    dwi_file_metadatas : :obj:`list` of :obj:`dict`
        A list of dictionaries containing metadata for each dwi file.
        Each dictionary should have a ``PhaseEncodingDirection`` key.
    estimate_per_axis : :obj:`bool`
        If True, limit phase encoding direction-based grouping to reverse PED scans.
        Otherwise, group all scans with different phase encoding directions together.
        Default is False.

    Returns
    -------
    dwi_series_fieldmaps : :obj:`dict`
        A dictionary where the keys are the full paths to dwi files and the values are
        dictionaries describing the fieldmap. If no fieldmap is found, the dictionary
        will be empty.

    Examples
    --------

    A single scan with no opportunities to SDC with a DWI scan
    >>> from qsiprep.utils.grouping import find_fieldmaps_from_other_dwis
    >>> single_dwi_file = ["/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz"]
    >>> single_dwi_file_metadatas = [{"PhaseEncodingDirection": "j"}]
    >>> find_fieldmaps_from_other_dwis(single_dwi_file, single_dwi_file_metadatas)
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {}}

    Two scans with the same PE direction: again no opportunities to SDC
    >>> repeat_dwi_files = ["/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz",
    ...                     "/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz"]
    >>> repeat_dwi_file_metadatas = [{"PhaseEncodingDirection": "j"},
    ...                              {"PhaseEncodingDirection": "j"}]
    >>> find_fieldmaps_from_other_dwis(repeat_dwi_files,
    ...     repeat_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {},
     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz': {}}

    Paired scans, each in opposite PE directions
    >>> paired_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz"]
    >>> paired_dwi_file_metadatas = [
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j-"}]
    >>> find_fieldmaps_from_other_dwis(paired_dwi_files,
    ...     paired_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']}}

    Multiple scans in multiple PE directions
    >>> multi_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz"]
    >>> multi_dwi_file_metadatas = [
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j-"},
    ...     {"PhaseEncodingDirection": "j-"}]
    >>> find_fieldmaps_from_other_dwis(multi_dwi_files,
    ...     multi_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']}}

    No information available
    >>> empty_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz"]
    >>> empty_dwi_file_metadatas = [
    ...     {},
    ...     {}]
    >>> find_fieldmaps_from_other_dwis(empty_dwi_files,
    ...     empty_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz': {},
     '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz': {}}
    """

    scans_to_pe_dirs = {
        fname: meta.get('PhaseEncodingDirection', None)
        for fname, meta in zip(dwi_files, dwi_file_metadatas, strict=False)
    }
    pe_dirs_to_scans = defaultdict(list)
    for scan_name, scan_dir in scans_to_pe_dirs.items():
        pe_dirs_to_scans[scan_dir].append(scan_name)

    dwi_series_fieldmaps = {}
    if not estimate_per_axis:
        unique_pe_dirs = set(pe_dirs_to_scans.keys())
        unique_nonnone_pe_dirs = [pe_dir for pe_dir in unique_pe_dirs if pe_dir is not None]

        if len(unique_nonnone_pe_dirs) <= 1:
            # No compatible PED scans, return empty fieldmaps
            for dwi_file in dwi_files:
                dwi_series_fieldmaps[dwi_file] = {}
            return dwi_series_fieldmaps

        # Group all scans with different phase encoding directions together
        # Exclude scans with no phase encoding direction
        for dwi_file in pe_dirs_to_scans.get(None, []):
            dwi_series_fieldmaps[dwi_file] = {}

        for pe_dir in unique_nonnone_pe_dirs:
            for dwi_file in pe_dirs_to_scans[pe_dir]:
                rpe_dwis = [f for f in dwi_files if f != dwi_file]
                rpe_dwis = [f for f in rpe_dwis if f not in pe_dirs_to_scans.get(None, [])]
                dwi_series_fieldmaps[dwi_file] = {'suffix': 'dwi', 'dwi': sorted(rpe_dwis)}
    else:
        # Group scans with reverse PEDs together
        dwi_series_fieldmaps = {}
        for dwi_file in dwi_files:
            dwi_series_fieldmaps[dwi_file] = {}
            pe_dir = scans_to_pe_dirs[dwi_file]
            # if there is no information, don't assume it's ok to combine
            if pe_dir is None:
                continue

            opposite_pe = pe_dir[0] if pe_dir.endswith('-') else pe_dir + '-'
            rpe_dwis = pe_dirs_to_scans[opposite_pe]

            if rpe_dwis:
                dwi_series_fieldmaps[dwi_file] = {'suffix': 'dwi', 'dwi': sorted(rpe_dwis)}

    return dwi_series_fieldmaps


def split_by_phase_encoding_direction(dwi_files, metadatas):
    """If no fieldmaps have been found for a group of dwi files, split them by PE direction.

    Parameters
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.
    metadatas : :obj:`list` of :obj:`dict`
        A list of dictionaries containing metadata for each dwi file.
        The only field that is used i "PhaseEncodingDirection".

    Returns
    -------
    dwi_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files. Each dictionary
        has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    Examples
    --------

    One of each direction (Not likely to see in the wild)
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-LR_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-SI_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {'PhaseEncodingDirection': 'i'},
    ...     {'PhaseEncodingDirection': 'i-'},
    ...     {'PhaseEncodingDirection': 'k'},
    ...     {'PhaseEncodingDirection': 'k-'}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i',
      'concatenated_bids_name': 'sub-1_dir-RL'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-LR_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i-',
      'concatenated_bids_name': 'sub-1_dir-LR'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k',
      'concatenated_bids_name': 'sub-1_dir-IS'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-SI_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k-',
      'concatenated_bids_name': 'sub-1_dir-SI'}]

    Repeats of some:
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {'PhaseEncodingDirection': 'i'}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i',
      'concatenated_bids_name': 'sub-1_dir-RL'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    Some missing metadata
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-3_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-4_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-5_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {},
    ...     {}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-3_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_run-3'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-4_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-4'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-5_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-5'}]
    """
    pe_dir_groups = defaultdict(list)
    unknowns = []
    for dwi_file, meta in zip(dwi_files, metadatas, strict=False):
        pe_dir = meta.get('PhaseEncodingDirection')
        if pe_dir:
            pe_dir_groups[pe_dir].append(dwi_file)
        else:
            unknowns.append(dwi_file)

    dwi_groups = []
    for pe_dir, dwi_group in sorted(pe_dir_groups.items()):
        dwi_groups.append(
            {
                'dwi_series': dwi_group,
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': pe_dir,
                'concatenated_bids_name': get_concatenated_bids_name(dwi_group),
            }
        )
    for unknown in unknowns:
        dwi_groups.append(
            {
                'dwi_series': [unknown],
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': '',
                'concatenated_bids_name': get_concatenated_bids_name([unknown]),
            }
        )

    return dwi_groups


def group_by_warpspace(dwi_files, layout, ignore_fieldmaps, estimate_per_axis):
    """Groups a session's DWI files by their acquisition parameters.

    DWIs are grouped by their **warped space**. Two DWI series that are
    listed in the IntendedFor field of a fieldmap are assumed to have the same
    susceptibility distortions and therefore be in the same warped space. The goal
    of this function is to combine DWI series into groups of acquisitions that
    are in the same warped space into a list of scans that can be combined after
    unwarping.

    Parameters
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree
    layout : :obj:`pybids.BIDSLayout`
        A representation of the BIDS tree
    ignore_fieldmaps : :obj:`bool`
        If True, ignore any fieldmaps in the ``fmap/`` directory.
        Images in ``dwi/`` will still be considered for SDC.
    estimate_per_axis : :obj:`bool`
        If True, limit phase encoding direction-based grouping to reverse PED scans.
        Otherwise, group all scans with different phase encoding directions together.
        Default is False.

    Returns
    -------
    dwi_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files. Each dictionary
        has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    Examples
    --------

    Set up tests
    >>> from pprint import pprint
    >>> from qsiprep.utils.bids import collect_data
    >>> SUBJECT_ID = "1"

    No fieldmap data, a single DWI series
    >>> subject_data, layout = collect_data("easy", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['...sub-1_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWIs with the same PE direction, to be concatenated
    >>> subject_data, layout = collect_data("concat1", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../concat1/sub-1/dwi/sub-1_run-01_dwi.nii.gz',
                     '.../concat1/sub-1/dwi/sub-1_run-02_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWI series intended to SDC each other
    >>> subject_data, layout = collect_data("opposite", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    Multiple DWI series in two different PE directions
    >>> subject_data, layout = collect_data("opposite_concat", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
               '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
                     '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
               '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    A phasediff fieldmap defines the warped group
    >>> subject_data, layout = collect_data("phasediff", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
                        'suffix': 'phasediff'},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, each with its own fieldmap/warped space
    >>> subject_data, layout = collect_data("separate_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../separate_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../separate_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-2'}]

    Same as above but ignoring fieldmaps. Data gets concatenated
    >>> subject_data, layout = collect_data("separate_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, True)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, opposite PE directions, dedicated EPI fieldmap for each
    >>> subject_data, layout = collect_data("mixed_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
       'dwi_series_pedir': 'j-',
       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]

    Same as last one, but ignore fieldmaps. The DWI series will be used for SDC instead
    >>> subject_data, layout = collect_data("mixed_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, True)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]


    There is no metadata related to epi distortion: don't concatenate anything
    >>> subject_data, layout = collect_data("missing_info", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../missing_info/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../missing_info/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]

    A bizarre mix of PE directions and some missing data
    >>> subject_data, layout = collect_data("wtf", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../wtf/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../wtf/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
        '.../wtf/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k-',
      'concatenated_bids_name': 'sub-1_dir-IS'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-1'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-2'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '.../wtf/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../wtf/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
        '.../wtf/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]
    """
    # For doc-building
    if layout is None:
        LOGGER.warning("Assuming we're building docs")
        return [
            {
                'dwi_series': dwi_files,
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': 'j',
                'concatenated_bids_name': 'sub-1',
            }
        ]

    # Get the metadata from every dwi file
    dwi_metadatas = [layout.get_metadata(dwi_file) for dwi_file in dwi_files]
    # Check for any data in dwi/ that could be used for distortion correction
    dwi_series_fieldmaps = find_fieldmaps_from_other_dwis(
        dwi_files=dwi_files,
        dwi_file_metadatas=dwi_metadatas,
        estimate_per_axis=estimate_per_axis,
    )

    # Find the best fieldmap for each file.
    best_fieldmap = {}
    grouped_by_fmap = defaultdict(list)
    for dwi_file in dwi_files:
        all_fmaps = [dwi_series_fieldmaps[dwi_file]]
        if not ignore_fieldmaps:
            fmap_fmaps = layout.get_fieldmap(dwi_file, return_list=True)
            all_fmaps += fmap_fmaps

        # Find the highest priority fieldmap for this dwi file
        best_fmap = get_highest_priority_fieldmap(all_fmaps)
        best_fieldmap[dwi_file] = best_fmap

        # Add the dwi file to a list of those corrected by this fieldmap
        fmap_key = tuple(best_fmap[best_fmap['suffix']]) if best_fmap['suffix'] else 'None'
        grouped_by_fmap[fmap_key].append(dwi_file)

    # Create the final groups
    dwi_groups = []
    for fmap_key, dwi_group in grouped_by_fmap.items():
        if fmap_key == 'None':
            dwi_groups.extend(
                split_by_phase_encoding_direction(
                    dwi_group,
                    [layout.get_metadata(dwi_file) for dwi_file in dwi_group],
                )
            )
        else:
            example_dwi_file = dwi_group[0]
            pe_direction = layout.get_metadata(example_dwi_file).get('PhaseEncodingDirection')
            dwi_groups.append(
                {
                    'dwi_series': dwi_group,
                    'fieldmap_info': best_fieldmap[example_dwi_file],
                    'dwi_series_pedir': pe_direction,
                    'concatenated_bids_name': get_concatenated_bids_name(dwi_group),
                }
            )

    config.loggers.workflow.info(
        f'Found {len(dwi_groups)} groups of DWI series based on their warp spaces:\n'
        f'{pprint.pformat(dwi_groups, indent=2, width=120)}'
    )

    return dwi_groups


def merge_dwi_groups(dwi_groups_plus, dwi_groups_minus):
    """Convert two dwi groups into a single group that will be concatenated for FSL.

    Parameters
    ----------
    dwi_groups_plus : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    dwi_groups_minus : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the same keys as ``dwi_groups_plus``.

    Returns
    -------
    merged_group : :obj:`dict`
        A dictionary describing the merged group of dwi files.
        The dictionary has the same keys as ``dwi_groups_plus``.

    Examples
    --------

    Set up tests
    >>> SUBJECT_ID = "1"

    AP/PA fieldmaps and paired DWI series
    >>> plus_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j-',
    ...       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
       '.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    Two series SDC each other
    >>> plus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    An odd case: one has an EPI
    >>> plus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    """
    dwi_files = []
    rpe_files = []
    fmap_files = []

    for dwi_group in dwi_groups_plus:
        dwi_files += dwi_group['dwi_series']
        fmap_type = dwi_group['fieldmap_info'].get('suffix')
        if fmap_type == 'dwi':
            rpe_files += dwi_group['fieldmap_info']['dwi']
        elif fmap_type == 'epi':
            fmap_files += dwi_group['fieldmap_info']['epi']
        pe_dir = dwi_group['dwi_series_pedir']

    for dwi_group in dwi_groups_minus:
        rpe_files += dwi_group['dwi_series']
        fmap_type = dwi_group['fieldmap_info'].get('suffix')
        if fmap_type == 'dwi':
            dwi_files += dwi_group['fieldmap_info']['dwi']
        elif fmap_type == 'epi':
            fmap_files += dwi_group['fieldmap_info']['epi']

    dwi_files = sorted(set(dwi_files))
    rpe_files = sorted(set(rpe_files))
    fmap_files = sorted(set(fmap_files))
    fieldmap_info = {'suffix': 'rpe_series', 'rpe_series': rpe_files}
    if fmap_files:
        fieldmap_info['epi'] = fmap_files

    merged_group = {
        'dwi_series': dwi_files,
        'dwi_series_pedir': pe_dir,
        'fieldmap_info': fieldmap_info,
        'concatenated_bids_name': get_concatenated_bids_name(dwi_files + rpe_files),
    }
    return merged_group


def group_for_eddy(all_dwi_fmap_groups):
    """Find matched pairs of phase encoding directions that can be combined for TOPUP/eddy.

    Any groups that don't have a phase encoding direction won't be correctable by eddy/TOPUP.

    Parameters
    ----------
    all_dwi_fmap_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.

    Returns
    -------
    eddy_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.
    concatenation_grouping : :obj:`dict`
        A dictionary mapping the concatenated BIDS name of each group to the name of the
        group that it should be concatenated with.

    Examples
    --------

    Paired DWI series to correct each other:
    >>> from pprint import pprint
    >>> dwi_groups = [
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
    ...   'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...   'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

    AP/PA EPI fieldmaps
    >>> dwi_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j-',
    ...       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
                                 '.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz'],
                         'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

    Val's scenario
    >>> dwi_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j',
    ...       'concatenated_bids_name': 'sub-1_dir-AP_run-2'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1_dir-AP_run-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz'],
                         'suffix': 'epi'}},
      {'concatenated_bids_name': 'sub-1_dir-AP_run-2',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz'],
                         'suffix': 'epi'}}],
        {'sub-1_dir-AP_run-1': 'sub-1_dir-AP_run-1',
         'sub-1_dir-AP_run-2': 'sub-1_dir-AP_run-2'})

    Repeated scans per PE direction
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...               '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'},
    ...     {'dwi_series': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...                     '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...               '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                      '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'rpe_series': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
                                        '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

    A phasediff fieldmap (Not used by eddy)
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
    ...                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
    ...                        'suffix': 'phasediff'},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1_dir-AP',
       'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                      '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
                         'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
                         'suffix': 'phasediff'}}],
     {'sub-1_dir-AP': 'sub-1_dir-AP'})

    """
    eddy_dwi_groups = []
    eddy_compatible_suffixes = ('dwi', 'epi')
    session_groups = _group_by_sessions(all_dwi_fmap_groups)
    for _, dwi_fmap_groups in session_groups.items():
        for pe_dir in 'ijk':
            plus_series = [
                dwi_group
                for dwi_group in dwi_fmap_groups
                if dwi_group.get('dwi_series_pedir') == pe_dir
                and dwi_group['fieldmap_info'].get('suffix') in eddy_compatible_suffixes
            ]
            minus_series = [
                dwi_group
                for dwi_group in dwi_fmap_groups
                if dwi_group.get('dwi_series_pedir') == pe_dir + '-'
                and dwi_group['fieldmap_info'].get('suffix') in eddy_compatible_suffixes
            ]

            # Can these be grouped?
            if plus_series and minus_series:
                eddy_dwi_groups.append(merge_dwi_groups(plus_series, minus_series))
            else:
                eddy_dwi_groups.extend(plus_series + minus_series)

        # Add separate groups for non-compatible fieldmaps
        for dwi_group in dwi_fmap_groups:
            if dwi_group['fieldmap_info'].get('suffix') not in eddy_compatible_suffixes:
                eddy_dwi_groups.append(dwi_group)

    config.loggers.workflow.info(
        f'Found {len(eddy_dwi_groups)} groups of DWI series that can be corrected by eddy:\n'
        f'{pprint.pformat(eddy_dwi_groups, indent=2, width=120)}'
    )

    return eddy_dwi_groups, {
        group['concatenated_bids_name']: group['concatenated_bids_name']
        for group in eddy_dwi_groups
    }


def get_concatenated_bids_name(dwi_group):
    """Derive the output file name for a group of dwi files.

    Strip away non-shared key/values from the input list of files. This function
    assumes you have already split the dwi group into something meaningful and
    really want to combine all the inputs.

    Parameters
    ----------
    dwi_group : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.

    Returns
    -------
    fname : :obj:`str`
        The BIDS name of the concatenated dwi series.

    Examples
    --------
    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-4_dwi.nii.gz'
    ... ])
    'sub-1_dir-AP'

    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'
    ... ])
    'sub-1'


    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_acq-HCP-dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-PA_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-PA_run-2_dwi.nii.gz'
    ... ])
    'sub-1_acq-HCP'

    """
    # If a single file, use its name, otherwise use the common prefix
    if len(dwi_group) > 1:
        fname = _get_common_bids_fields(dwi_group)
        parts = fname.split('_')
        full_parts = [part for part in parts if not part.endswith('-')]
        fname = '_'.join(full_parts)
    else:
        input_fname = dwi_group[0]
        fname = split_filename(input_fname)[1]

    if fname.endswith('_dwi'):
        fname = fname[:-4]

    return fname.replace('.', '').replace(' ', '')


def _add_acq_entity(concatenated_bids_name, index):
    """Add or modify an acquisition entity in a concatenated BIDS name.

    Parameters
    ----------
    concatenated_bids_name : :obj:`str`
        The concatenated BIDS name.
    index : :obj:`int`
        Index used to make a duplicate name unique.

    Returns
    -------
    :obj:`str`
        A BIDS-like name with an acquisition label.

    Examples
    --------
    >>> _add_acq_entity('sub-1_dir-AP', 1)
    'sub-1_acq-1_dir-AP'
    >>> _add_acq_entity('sub-1_acq-HCP_dir-AP', 2)
    'sub-1_acq-HCP+2_dir-AP'
    """
    name_parts = concatenated_bids_name.split('_')
    acq_idx = next((idx for idx, part in enumerate(name_parts) if part.startswith('acq-')), None)
    if acq_idx is not None:
        acq_label = name_parts[acq_idx][len('acq-') :]
        name_parts[acq_idx] = f'acq-{acq_label}+{index}'
        return '_'.join(name_parts)

    ses_idx = next((idx for idx, part in enumerate(name_parts) if part.startswith('ses-')), None)
    sub_idx = next((idx for idx, part in enumerate(name_parts) if part.startswith('sub-')), None)
    insert_idx = ses_idx + 1 if ses_idx is not None else sub_idx + 1 if sub_idx is not None else 0
    name_parts.insert(insert_idx, f'acq-{index}')
    return '_'.join(name_parts)


def _get_unique_concatenated_bids_names(dwi_groups):
    """Get unique concatenated BIDS names for a list of DWI groups.

    Parameters
    ----------
    dwi_groups : :obj:`list` of :obj:`list` of :obj:`str`
        A list of DWI groups.

    Returns
    -------
    :obj:`list` of :obj:`str`
        One generalized BIDS name per input group.

    Notes
    -----
    `get_concatenated_bids_name` intentionally keeps shared entities only, which
    can produce duplicate names for different groups. This helper resolves only
    collisions by adding indexed `acq-` entities.
    """
    concatenated_names = [get_concatenated_bids_name(dwi_group) for dwi_group in dwi_groups]
    grouped_name_indices = defaultdict(list)
    for idx, concatenated_name in enumerate(concatenated_names):
        grouped_name_indices[concatenated_name].append(idx)

    unique_names = list(concatenated_names)
    for concatenated_name, duplicate_indices in grouped_name_indices.items():
        if len(duplicate_indices) == 1:
            continue

        for duplicate_idx, group_idx in enumerate(duplicate_indices, start=1):
            unique_names[group_idx] = _add_acq_entity(concatenated_name, duplicate_idx)

    return unique_names


def _get_common_bids_fields(fnames):
    """Find the common fields in a list of BIDS filenames.

    Parameters
    ----------
    fnames : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.

    Returns
    -------
    fname : :obj:`str`
        The common fields in the filenames.
    """
    bids_keys = defaultdict(set)
    for fname in fnames:
        basename = split_filename(fname)[1]
        for token in basename.split('_'):
            parts = token.split('-')
            if len(parts) == 2:
                key, value = parts
                bids_keys[key].update((value,))

    # Find all the keys with a single unique value
    common_bids = []
    for key in ['sub', 'ses', 'acq', 'dir', 'run']:
        if len(bids_keys[key]) == 1:
            common_bids.append(key + '-' + bids_keys[key].pop())

    return '_'.join(common_bids)


def _group_by_sessions(dwi_fmap_groups):
    """Create a lookup of distortion groups by session

    Parameters
    ----------
    dwi_fmap_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.

    Returns
    -------
    ses_lookup : :obj:`dict`
        A dictionary mapping session ids to lists of dwi_fmap_groups.

    Examples
    --------
    Paired DWI series to correct each other:
    >>> dwi_groups = [
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-PA_run-2_dwi.nii.gz']},
    ...   'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_ses-1_dir-AP_run-1'},
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-AP_run-1_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...   'concatenated_bids_name': 'sub-1_ses-1_dir-PA_run-2'},
    ...  {'dwi_series': [
    ...          '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2/dir-AP_run-1_dwi.nii.gz',
    ...          '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2/dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_ses-2_dir-AP'},
    ...     {'dwi_series': [
    ...           '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-1_dwi.nii.gz',
    ...           '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-AP_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-AP_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_ses-2_dir-PA'}]
    """
    ses_lookup = defaultdict(list)
    for group in dwi_fmap_groups:
        bids_info = get_bids_params(group['concatenated_bids_name'])
        ses_lookup[bids_info['session_id']].append(group)

    return ses_lookup
