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

    # Distortion groups are determined by physical signature alone (legacy
    # build_distortion_groups). They are built first so that field-map discovery
    # can resolve curator links (IntendedFor) against whole distortion groups,
    # exactly as the legacy dg-level pipeline did.
    if combine_scans:
        distortion_groups = _final_distortion_groups(series_list)
    else:
        names = _get_unique_concatenated_bids_names([[s.path] for s in series_list])
        distortion_groups = {name: [s.path] for name, s in zip(names, series_list, strict=False)}

    if ignore_fieldmaps:
        estimators = []
    else:
        estimators = find_estimators(
            layout=layout,
            series=series_list,
            distortion_groups=distortion_groups,
            ignore_fieldmaps=ignore_fieldmaps,
            estimate_per_axis=estimate_per_axis,
        )

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


def _build_file_to_dg_map(distortion_groups):
    """Return a mapping from DWI file path → distortion-group ID."""
    file_to_dg = {}
    for dg_id, files in distortion_groups.items():
        for f in files:
            file_to_dg[f] = dg_id
    return file_to_dg


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


# ---------------------------------------------------------------------------
# Series-level field-map estimator discovery.
#
# These helpers reproduce the behavior of QSIPrep's historical
# distortion-group-level field-map selection, operating over ``DwiSeries``
# objects and returning ``FieldmapEstimation`` value objects.
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


def _intendedfor_buckets(layout, fmap_files, by_path, series_to_dg):
    """Build estimation buckets from IntendedFor metadata on fmap files.

    Ported from :func:`_build_intendedfor_groups`. Targets are resolved to
    *distortion-group ids* (legacy keyed buckets on target dg ids, not raw
    files), so several fmaps pointing at series that collapse into one distortion
    group share a single bucket. Returns a list of
    ``(tuple_of_sorted_target_dg_ids, set_of_fmap_paths)`` in first-appearance
    order while iterating ``fmap_files`` (which the caller passes sorted, matching
    legacy ``_get_fmap_files``). This reproduces the legacy ``auto_`` numbering,
    which assigned ids by first appearance of each target key during sorted-fmap
    iteration -- not by sorted target key.
    """
    raw_groups = {}  # target_dg_key -> set of fmap paths (insertion order preserved)

    for fmap_file in fmap_files:
        intended_for = layout.get_metadata(fmap_file).get('IntendedFor')
        if intended_for is None:
            continue

        intended_for = _as_list(intended_for)
        targeted = _resolve_intended_for_series(intended_for, by_path)
        target_dgs = sorted({series_to_dg[p] for p in targeted if p in series_to_dg})
        if not target_dgs:
            continue

        target_key = tuple(target_dgs)
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


def find_estimators(*, layout, series, distortion_groups, ignore_fieldmaps, estimate_per_axis):
    """Discover field-map estimators for a set of DWI series.

    Priority (highest first): B0FieldIdentifier -> IntendedFor -> PE-direction
    heuristic. Mirrors sdcflows' ``find_estimators`` shape, adapted to QSIPrep's
    distortion groups. Curator links (``IntendedFor``) are resolved against whole
    distortion groups, reproducing the legacy dg-level pipeline.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
    series : list[DwiSeries]
    distortion_groups : dict[str, list[str]]
        The (final) distortion groups, used to collapse curator links that point
        at series sharing a distortion group.
    ignore_fieldmaps : :obj:`bool`
    estimate_per_axis : :obj:`bool`

    Returns
    -------
    estimators : list[FieldmapEstimation]
    """
    estimators = []
    auto_counter = 0

    by_path = {s.path: s for s in series}
    series_to_dg = {}
    for dg_id, files in distortion_groups.items():
        for f in files:
            series_to_dg[f] = dg_id
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

        return estimators

    # ----- Priority 2: IntendedFor on fmap files -----
    if fmap_files:
        intended = _intendedfor_buckets(layout, fmap_files, by_path, series_to_dg)
        for target_dgs, fmap_paths in intended:
            auto_id = f'auto_{auto_counter:05d}'
            auto_counter += 1
            # The target series are every DWI in the targeted distortion groups.
            target_paths = [p for dg in target_dgs for p in distortion_groups[dg]]
            # Split by session over the combined members (targets + fmaps),
            # mirroring the legacy split. In practice IntendedFor resolves by
            # basename (which carries the ses- entity), so groups are already
            # single-session and the split is a no-op.
            combined = list(target_paths) + sorted(fmap_paths)
            sessions = {_session_of_path(layout, by_path, p) for p in combined}
            split = len(sessions) > 1
            ordered = (
                sorted(sessions, key=lambda s: '' if s is None else str(s)) if split else [None]
            )
            for session in ordered:
                if split:
                    key = _suffix_key(auto_id, session)
                    sess_fmaps = [
                        p for p in fmap_paths if _session_of_path(layout, by_path, p) == session
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
                sources = [_fieldmap_file(p) for p in sorted(sess_fmaps) + sorted(sess_targets)]
                est = FieldmapEstimation(sources, bids_id=key)
                estimators.append(est)
        if estimators:
            return estimators

    # ----- Priority 3: PE-direction heuristic -----
    for group_paths in _heuristic_series_groups(series, estimate_per_axis):
        sources = [_fieldmap_file(p) for p in sorted(group_paths)]
        est = FieldmapEstimation(sources, auto_id=f'auto_{auto_counter:05d}')
        auto_counter += 1
        estimators.append(est)

    return estimators


def _final_distortion_groups(series_list):
    """Group DwiSeries by their physical distortion signature.

    This reproduces the legacy ``build_distortion_groups`` exactly. The legacy
    ``refine_distortion_groups`` step is a no-op in practice: two series sharing a
    physical signature also share their ``B0FieldIdentifier`` (it is part of the
    signature) and resolve any ``IntendedFor`` to the same group, so their
    estimation membership is always identical and the refine split collapses to a
    single sub-group. Grouping by signature alone therefore matches legacy while
    avoiding the build-then-refine-then-remap dance. Groups (and the files within
    them) are kept in first-appearance order over ``series_list``.
    """
    groups = []
    keys = []
    for s in series_list:
        key = s.distortion_signature
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
