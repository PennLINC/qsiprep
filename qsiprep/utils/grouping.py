# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to group scans based on their acquisition parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

from nipype.utils.filemanip import split_filename

from .. import config
from ..interfaces.bids import get_bids_params
from .fieldmaps import FieldmapEstimation, FieldmapFile

LOGGER = logging.getLogger('nipype.workflow')

_DISTORTION_FIELDS = (
    'session',
    'PhaseEncodingDirection',
    'ShimSetting',
    'TotalReadoutTime',
    'B0FieldIdentifier',
)


@dataclass(slots=True)
class DwiSeries:
    """A DWI file plus its distortion-relevant metadata, read once."""

    path: str
    session: object
    pe_dir: object
    shim: object
    trt: object
    b0_identifier: object
    b0_source: object
    multipart_id: object

    @classmethod
    def from_layout(cls, layout, path):
        bidsfile = layout.get_file(path)
        metadata = bidsfile.get_metadata()
        shim = metadata.get('ShimSetting')
        b0fi = metadata.get('B0FieldIdentifier')
        return cls(
            path=path,
            session=bidsfile.entities.get('session'),
            pe_dir=metadata.get('PhaseEncodingDirection'),
            shim=tuple(shim) if isinstance(shim, list) else shim,
            trt=metadata.get('TotalReadoutTime'),
            b0_identifier=tuple(b0fi) if isinstance(b0fi, list) else b0fi,
            b0_source=metadata.get('B0FieldSource'),
            multipart_id=metadata.get('MultipartID'),
        )

    @property
    def distortion_signature(self):
        """The physical-distortion key (no estimator constraint yet).

        ``B0FieldIdentifier`` is normalized to a tuple exactly as the legacy
        ``build_distortion_groups`` did (``tuple(_ensure_list(...))``), so a
        scalar identifier and a single-element list group identically.
        """
        b0 = self.b0_identifier
        if b0 is not None and not isinstance(b0, tuple):
            b0 = (b0,)
        return (self.session, self.pe_dir, self.shim, self.trt, b0)


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
        If True, disable field-map usage entirely. Field-map estimation and
        application groups will be returned as ``None``.
    estimate_per_axis : :obj:`bool`, optional
        If True, limit PE direction-based grouping to reverse-PED pairs only.

    Returns
    -------
    distortion_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of raw DWI file paths.
    fmap_estimation_groups : dict[str, list[str]] or None
        Keys are field-map identifiers, values are lists mixing distortion-group
        IDs (for DWI data) and raw file paths (for fmap data). ``None`` when
        ``ignore_fieldmaps`` is True.
    fmap_application_groups : dict[str, list[str]] or None
        Keys are field-map identifiers, values are distortion-group IDs that
        will be corrected by that field map. ``None`` when
        ``ignore_fieldmaps`` is True.
    concatenation_groups : dict[str, list[str]]
        Keys are unique BIDS names, values are lists of distortion-group IDs.
    """
    config.loggers.workflow.info('Grouping DWI scans')

    series_list = [DwiSeries.from_layout(layout, dwi) for dwi in subject_data['dwi']]

    if ignore_fieldmaps:
        estimators, series_to_estimator = [], {}
    else:
        estimators, series_to_estimator = find_estimators(
            layout=layout,
            series=series_list,
            ignore_fieldmaps=ignore_fieldmaps,
            estimate_per_axis=estimate_per_axis,
        )

    # Final distortion groups fold the estimator id into the physical signature,
    # so a group never spans two estimators (no post-hoc refine/remap needed).
    if combine_scans:
        distortion_groups = _final_distortion_groups(series_list, series_to_estimator)
    else:
        names = _get_unique_concatenated_bids_names([[s.path] for s in series_list])
        distortion_groups = {
            name: [s.path] for name, s in zip(names, series_list, strict=False)
        }

    if ignore_fieldmaps:
        fmap_estimation_groups = None
        fmap_application_groups = None
    else:
        fmap_estimation_groups = _serialize_estimation_groups(estimators, distortion_groups)
        # Application is derived from the (final) distortion + estimation groups,
        # reproducing the many-to-many B0FieldSource mapping exactly.
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

    config.loggers.workflow.info('Finished grouping DWI scans')
    return distortion_groups, fmap_estimation_groups, fmap_application_groups, concatenation_groups


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
                f'({sorted(cg_ids)}). A field-map estimation group must be a subset of '
                f'exactly one concatenation group.'
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
                f'concatenation groups ({sorted(concat_ids)}). A field-map estimation '
                f'group must be a subset of exactly one concatenation group.'
            )


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
                f'{sorted(axes)}, but estimate_per_axis is True. A single B0 field '
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


FMAP_PRIORITY = {
    'dwi': 0,
    'epi': 1,
    'fieldmap': 2,
    'phasediff': 3,
    'phase1': 4,
    'phase': 4,
    'syn': 5,
}


def _find_sibling_fmap_file(fmap_file, suffix, sibling_suffix):
    """Find a file next to *fmap_file* whose BIDS suffix is *sibling_suffix*."""
    dirname, basename = os.path.split(fmap_file)
    sibling_basename = basename.replace(f'_{suffix}.', f'_{sibling_suffix}.')
    if sibling_basename == basename:
        return None

    sibling = os.path.join(dirname, sibling_basename)
    if os.path.exists(sibling):
        return sibling
    return None


def classify_fmap_files(fmap_files):
    """Sort raw fmap file paths into pepolar EPI files and typed GRE fieldmaps.

    Magnitude and phase2 images are attached to the phasediff/phase1/fieldmap
    image they accompany (found by filename, like pybids' ``get_fieldmap``),
    even when they are not listed in *fmap_files* themselves.

    Parameters
    ----------
    fmap_files : :obj:`list` of :obj:`str`
        Paths to NIfTI files in a subject's ``fmap/`` directory.

    Returns
    -------
    epi_files : :obj:`list` of :obj:`str`
        Sorted paths to pepolar EPI fieldmap files.
    gre_fmap_infos : :obj:`list` of :obj:`dict`
        One dictionary per GRE fieldmap, in the ``fieldmap_info`` format
        expected by the downstream SDC workflows (a ``suffix`` key plus
        type-specific file keys).
    """
    epi_files = []
    gre_fmap_infos = []
    handled = set()

    for fmap_file in sorted(fmap_files):
        if fmap_file in handled:
            continue

        suffix = split_filename(fmap_file)[1].split('_')[-1]
        if suffix == 'epi':
            epi_files.append(fmap_file)
        elif suffix == 'phasediff':
            info = {'suffix': 'phasediff', 'phasediff': fmap_file}
            for mag_suffix in ('magnitude1', 'magnitude2'):
                sibling = _find_sibling_fmap_file(fmap_file, suffix, mag_suffix)
                if sibling is not None:
                    info[mag_suffix] = sibling
                    handled.add(sibling)
            gre_fmap_infos.append(info)
        elif suffix == 'phase1':
            info = {'suffix': 'phase1', 'phase1': fmap_file}
            for sibling_suffix in ('phase2', 'magnitude1', 'magnitude2'):
                sibling = _find_sibling_fmap_file(fmap_file, suffix, sibling_suffix)
                if sibling is not None:
                    info[sibling_suffix] = sibling
                    handled.add(sibling)
            gre_fmap_infos.append(info)
        elif suffix == 'fieldmap':
            info = {'suffix': 'fieldmap', 'fieldmap': fmap_file}
            sibling = _find_sibling_fmap_file(fmap_file, suffix, 'magnitude')
            if sibling is not None:
                info['magnitude'] = sibling
                handled.add(sibling)
            gre_fmap_infos.append(info)
        # magnitude*/phase2 files are only useful as companions of the files
        # handled above, so anything else is ignored.

    return epi_files, gre_fmap_infos


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


# ---------------------------------------------------------------------------
# Series-level field-map estimator discovery (new architecture).
#
# These helpers port the behavior of the distortion-group-level functions
# above (``build_fmap_estimation_groups`` and friends) to operate over
# ``DwiSeries`` objects, returning ``FieldmapEstimation`` objects. The
# distortion-group originals are retained for now and removed in a later task.
# ---------------------------------------------------------------------------


def _as_list(value):
    """Wrap scalars in a list; unpack lists and tuples.

    Ported from :func:`_ensure_list`, but also unpacks tuples because
    :class:`DwiSeries` normalizes list-valued ``B0FieldIdentifier`` to a tuple
    (for hashability in the distortion signature). Preserves the exact None
    handling (``None`` becomes ``[None]``); callers filter ``None`` themselves.
    """
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _subject_fmap_files(layout, series):
    """Sorted fmap/ NIfTI paths for the subject owning these series.

    Ported from :func:`_get_fmap_files` (the ``ignore_fieldmaps`` short-circuit
    is handled by the caller).
    """
    if not series:
        return []
    sub_id = layout.get_file(series[0].path).entities.get('subject')
    try:
        files = layout.get(
            subject=sub_id,
            datatype='fmap',
            extension=['.nii.gz', '.nii'],
            return_type='file',
        )
    except Exception:  # noqa: BLE001
        files = []
    return sorted(files)


def _suffix_key(base_key, session):
    """Append a session suffix to a group key (mirrors the legacy scheme)."""
    suffix = f'ses-{session}' if session is not None else 'ses-none'
    return f'{base_key}_{suffix}'


def _session_of_path(layout, by_path, path):
    """Session entity of a series path (via DwiSeries) or an fmap path."""
    if path in by_path:
        return by_path[path].session
    bidsfile = layout.get_file(path)
    return bidsfile.entities.get('session') if bidsfile is not None else None


def _split_paths_by_session(layout, by_path, member_paths, base_key):
    """Partition member paths by session into keyed groups.

    Mirrors :func:`_split_named_member_groups_by_session`: when members occupy a
    single session the group keeps ``base_key``; otherwise it splits into
    ``f'{base_key}_ses-{session}'`` groups (sorted by session, ``None`` ->
    ``'ses-none'``). Returns a list of ``(key, sorted_paths)``.
    """
    by_session = defaultdict(list)
    for p in member_paths:
        by_session[_session_of_path(layout, by_path, p)].append(p)
    if len(by_session) <= 1:
        return [(base_key, sorted(member_paths))]
    return [
        (_suffix_key(base_key, session), sorted(by_session[session]))
        for session in sorted(by_session, key=lambda s: '' if s is None else str(s))
    ]


def _check_axis_conflict(sources, ident, estimate_per_axis):
    """Raise if estimate_per_axis=True and a B0 group spans multiple PE axes.

    Ported from :func:`_check_b0field_axis_conflict`, reading PE directions
    directly from the source files' metadata. It deliberately does not build a
    :class:`FieldmapEstimation`, because a single file may carry several
    ``B0FieldIdentifier`` values (belonging to several groups) -- exactly as the
    legacy check, which operated on the identifier->members map.
    """
    if not estimate_per_axis:
        return

    axes = set()
    for f in sources:
        pe = f.metadata.get('PhaseEncodingDirection')
        axis = _get_pe_axis(pe)
        if axis is not None:
            axes.add(axis)

    if len(axes) > 1:
        raise ValueError(
            f'B0FieldIdentifier {ident!r} groups files across PE axes '
            f'{sorted(axes)}, but estimate_per_axis is True. A single B0 field '
            f'estimation group cannot span multiple axis pairs.'
        )


def _check_series_metadata_compatibility(member_series, group_key, group_kind):
    """Raise if a metadata-defined group spans incompatible shim/TRT signatures.

    Ported from :func:`_check_distortion_metadata_compatibility`, operating over
    DWI-series members (fmap paths are not series and carry no signature here).
    """
    if len(member_series) <= 1:
        return

    signatures = {(s.shim, s.trt) for s in member_series}
    if len(signatures) > 1:
        raise ValueError(
            f'{group_kind.capitalize()} group {group_key!r} contains distortion groups '
            'with conflicting ShimSetting and/or TotalReadoutTime metadata. '
            'These files cannot be grouped together for field-map processing.'
        )


def _resolve_intended_for_series(intended_for_paths, by_path):
    """Map IntendedFor target paths to actual DWI series paths.

    Ported from :func:`_resolve_intended_for`; ``by_path`` maps series paths to
    :class:`DwiSeries`.
    """
    basename_to_path = {os.path.basename(p): p for p in by_path}
    resolved = []
    for target in intended_for_paths:
        if target.startswith('bids::'):
            target = target[len('bids::') :]
        target_basename = os.path.basename(target)
        if target_basename in basename_to_path:
            resolved.append(basename_to_path[target_basename])
    return resolved


def _intendedfor_buckets(layout, fmap_files, by_path):
    """Build estimation buckets from IntendedFor metadata on fmap files.

    Ported from :func:`_build_intendedfor_groups`. Returns a list of
    ``(tuple_of_sorted_target_series_paths, set_of_fmap_paths)`` in
    first-appearance order while iterating ``fmap_files`` (which the caller passes
    sorted, matching legacy ``_get_fmap_files``). This reproduces the legacy
    ``auto_`` numbering, which assigned ids by first appearance of each target key
    during sorted-fmap iteration -- not by sorted target key.
    """
    raw_groups = {}  # target_key -> set of fmap paths (insertion order preserved)

    for fmap_file in fmap_files:
        intended_for = layout.get_metadata(fmap_file).get('IntendedFor')
        if intended_for is None:
            continue

        intended_for = _as_list(intended_for)
        targeted = _resolve_intended_for_series(intended_for, by_path)
        targeted_paths = sorted(set(targeted))
        if not targeted_paths:
            continue

        target_key = tuple(targeted_paths)
        raw_groups.setdefault(target_key, set()).add(fmap_file)

    return list(raw_groups.items())


def _heuristic_series_groups(series, estimate_per_axis):
    """Pair series by PE direction when no curator metadata exists.

    Ported from :func:`_build_heuristic_estimation_groups`. Yields lists of
    series paths in the same iteration order (sorted sessions -> MultipartID
    partitions -> sorted PE axes).
    """
    groups = []

    # Never create heuristic estimation groups across sessions.
    by_session = defaultdict(list)
    for s in series:
        by_session[s.session].append(s)

    for session in sorted(by_session, key=lambda v: '' if v is None else str(v)):
        session_series = by_session[session]

        # Do not build heuristic fmap groups across MultipartID boundaries.
        multipart_partitions = defaultdict(list)
        for s in session_series:
            partition_key = s.multipart_id if s.multipart_id is not None else '_no_multipart'
            multipart_partitions[partition_key].append(s)

        for partition_series in multipart_partitions.values():
            pe_axes = defaultdict(lambda: defaultdict(list))
            for s in partition_series:
                pe = s.pe_dir
                if pe is None:
                    continue
                axis = _get_pe_axis(pe)
                pe_axes[axis][pe].append(s.path)

            if estimate_per_axis:
                for axis, dir_map in sorted(pe_axes.items()):
                    plus_dir = axis
                    minus_dir = axis + '-'
                    plus_paths = dir_map.get(plus_dir, [])
                    minus_paths = dir_map.get(minus_dir, [])
                    if plus_paths and minus_paths:
                        groups.append(sorted(plus_paths + minus_paths))
            else:
                all_paths = [s.path for s in partition_series if s.pe_dir is not None]
                unique_pes = {s.pe_dir for s in partition_series if s.pe_dir is not None}
                if len(unique_pes) >= 2 and all_paths:
                    groups.append(sorted(all_paths))

    return groups


def find_estimators(*, layout, series, ignore_fieldmaps, estimate_per_axis):
    """Discover field-map estimators for a set of DWI series.

    Priority (highest first): B0FieldIdentifier -> IntendedFor -> PE-direction
    heuristic. Mirrors sdcflows' ``find_estimators`` shape but returns QSIPrep
    estimator objects plus a per-series application assignment.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    series : list[DwiSeries]
    ignore_fieldmaps : :obj:`bool`
    estimate_per_axis : :obj:`bool`

    Returns
    -------
    estimators : list[FieldmapEstimation]
    series_to_estimator : dict[str, str]
        Maps a DWI series path to the ``bids_id`` of the estimator that corrects
        it. (Application mapping; mirrors today's fmap_application_groups.)
    """
    estimators = []
    series_to_estimator = {}
    auto_counter = 0

    by_path = {s.path: s for s in series}
    fmap_files = [] if ignore_fieldmaps else _subject_fmap_files(layout, series)

    def _fieldmap_file(path):
        return FieldmapFile(path, metadata=layout.get_metadata(path))

    # ----- Priority 1: B0FieldIdentifier on any DWI or fmap file -----
    b0_members = defaultdict(set)  # identifier -> {paths}
    has_b0field = False
    for s in series:
        b0fi = s.b0_identifier
        if b0fi is not None:
            has_b0field = True
            for value in _as_list(b0fi):
                if value is not None:
                    b0_members[value].add(s.path)
    for fpath in fmap_files:
        b0fi = layout.get_metadata(fpath).get('B0FieldIdentifier')
        if b0fi is not None:
            has_b0field = True
            for value in _as_list(b0fi):
                if value is not None:
                    b0_members[value].add(fpath)

    if has_b0field:
        # Axis conflicts are checked on the unsplit identifier groups, matching
        # the legacy order (all axis checks before any session split / compat).
        for ident in sorted(b0_members):
            sources = [_fieldmap_file(p) for p in sorted(b0_members[ident])]
            _check_axis_conflict(sources, ident, estimate_per_axis)

        # A named estimation group is split so it never spans sessions, using the
        # same suffixed-key scheme as _split_named_member_groups_by_session.
        for ident in sorted(b0_members):
            for key, part_paths in _split_paths_by_session(
                layout, by_path, b0_members[ident], ident
            ):
                member_series = [by_path[p] for p in part_paths if p in by_path]
                _check_series_metadata_compatibility(
                    member_series, key, group_kind='field-map estimation'
                )
                sources = [_fieldmap_file(p) for p in sorted(part_paths)]
                # bids_id is passed explicitly: a source may carry several
                # B0FieldIdentifiers (belonging to several groups), so per-source
                # resolution would wrongly flag a conflict.
                est = FieldmapEstimation(sources, bids_id=key)
                estimators.append(est)

        # Application assignment, derived independently of estimation (mirrors the
        # legacy build_fmap_application_groups): if any series declares a
        # B0FieldSource, split those by session; otherwise every member series is a
        # target of its (session-split) estimation group.
        any_b0_source = any(s.b0_source is not None for s in series)
        if any_b0_source:
            b0source_members = defaultdict(list)  # value -> [series paths]
            for s in series:
                if s.b0_source is not None:
                    for value in _as_list(s.b0_source):
                        if value is not None:
                            b0source_members[value].append(s.path)
            for value, paths in b0source_members.items():
                split = len({by_path[p].session for p in paths}) > 1
                for p in paths:
                    sess = by_path[p].session
                    series_to_estimator[p] = _suffix_key(value, sess) if split else value
        else:
            for ident in sorted(b0_members):
                for key, part_paths in _split_paths_by_session(
                    layout, by_path, b0_members[ident], ident
                ):
                    for p in part_paths:
                        if p in by_path:
                            series_to_estimator[p] = key

        return estimators, series_to_estimator

    # ----- Priority 2: IntendedFor on fmap files -----
    if fmap_files:
        intended = _intendedfor_buckets(layout, fmap_files, by_path)
        for target_paths, fmap_paths in intended:
            auto_id = f'auto_{auto_counter:05d}'
            auto_counter += 1
            # Split by session over the combined members (targets + fmaps),
            # mirroring the legacy split. In practice IntendedFor resolves by
            # basename (which carries the ses- entity), so groups are already
            # single-session and the split is a no-op.
            combined = list(target_paths) + sorted(fmap_paths)
            sessions = {_session_of_path(layout, by_path, p) for p in combined}
            split = len(sessions) > 1
            ordered = (
                sorted(sessions, key=lambda s: '' if s is None else str(s))
                if split
                else [None]
            )
            for session in ordered:
                if split:
                    key = _suffix_key(auto_id, session)
                    sess_fmaps = [
                        p
                        for p in fmap_paths
                        if _session_of_path(layout, by_path, p) == session
                    ]
                    sess_targets = [p for p in target_paths if by_path[p].session == session]
                else:
                    key = auto_id
                    sess_fmaps = sorted(fmap_paths)
                    sess_targets = list(target_paths)
                if not sess_fmaps:
                    # Degenerate cross-session intent (a fmap in one session
                    # intending a target in another). The legacy code emitted a
                    # source-less estimation group here; we drop it rather than
                    # raise 'Insufficient sources'. Unreachable when IntendedFor
                    # entries carry the ses- entity (the normal case).
                    continue
                member_series = [by_path[p] for p in sess_targets if p in by_path]
                _check_series_metadata_compatibility(
                    member_series, key, group_kind='field-map estimation'
                )
                # The IntendedFor targets are part of the estimation group (legacy
                # _build_intendedfor_groups records target dg ids as members), so
                # they are included as sources here. This is harmless to method
                # inference and to_fieldmap_info (which select fmaps by suffix).
                sources = [
                    _fieldmap_file(p) for p in sorted(sess_fmaps) + sorted(sess_targets)
                ]
                est = FieldmapEstimation(sources, bids_id=key)
                estimators.append(est)
                for tpath in sess_targets:
                    series_to_estimator[tpath] = est.bids_id
        if estimators:
            return estimators, series_to_estimator

    # ----- Priority 3: PE-direction heuristic -----
    for group_paths in _heuristic_series_groups(series, estimate_per_axis):
        sources = [_fieldmap_file(p) for p in sorted(group_paths)]
        est = FieldmapEstimation(sources, auto_id=f'auto_{auto_counter:05d}')
        auto_counter += 1
        estimators.append(est)
        for p in group_paths:
            series_to_estimator[p] = est.bids_id

    return estimators, series_to_estimator


def _final_distortion_groups(series_list, series_to_estimator):
    """Group DwiSeries by ``(physical signature, estimator id)``.

    Folding the estimator id into the signature means a distortion group never
    spans two estimators, so no post-hoc split/remap is needed. Groups (and the
    files within them) are kept in first-appearance order over ``series_list``,
    matching the legacy ``build_distortion_groups``.
    """
    groups = []
    keys = []
    for s in series_list:
        key = (s.distortion_signature, series_to_estimator.get(s.path))
        if key in keys:
            groups[keys.index(key)].append(s.path)
        else:
            keys.append(key)
            groups.append([s.path])
    names = _get_unique_concatenated_bids_names(groups)
    return dict(zip(names, groups, strict=False))


def _serialize_estimation_groups(estimators, distortion_groups):
    """Serialize estimators into the legacy ``fmap_estimation_groups`` dict.

    ``fmap_estimation_groups[bids_id]`` is the sorted list of members, mixing
    distortion-group ids (for DWI sources, mapped via the file -> dg lookup) and
    raw ``fmap/`` file paths (kept as paths), exactly as the legacy
    ``build_fmap_estimation_groups`` produced. Application groups are derived
    separately by :func:`build_fmap_application_groups`, which faithfully
    reproduces the many-to-many B0FieldSource mapping (a DWI may declare several
    sources).
    """
    path_to_dg = {}
    for dg_id, files in distortion_groups.items():
        for f in files:
            path_to_dg[f] = dg_id

    fmap_estimation_groups = {}
    for est in estimators:
        members = set()
        for source in est.sources:
            path = str(source.path)
            members.add(path_to_dg.get(path, path))
        fmap_estimation_groups[est.bids_id] = sorted(members)

    return fmap_estimation_groups
