"""Read every grouping-relevant sidecar field, exactly once, in one place.

This module is the only part of qsiprep's grouping that touches JSON
sidecars. It converts each dwi/ and fmap/ NIfTI into a
:class:`~.models.FileRecord`, normalizing:

- ``B0FieldIdentifier`` / ``B0FieldSource``: string-or-list to tuple.
- ``IntendedFor``: relative-to-subject paths, ``bids::`` URIs, and (with a
  warning) absolute paths, all resolved to absolute paths. pybids'
  ``layout.get_fieldmap`` is deliberately not used - it only understands
  relative-path IntendedFor and cannot query B0Field* fields at all.
- ``ShimSetting``: list to tuple, so signatures are hashable.
- ``TotalReadoutTime``: rounded to :data:`~.models.READOUT_TOLERANCE` so
  float jitter between sidecars does not split groups.
"""

from __future__ import annotations

import os.path as op
import re

from bids.layout import parse_file_entities

from .models import READOUT_TOLERANCE, DistortionSignature, FileRecord
from .validation import GroupingIssue, warning

#: fmap/ suffixes that can participate in fieldmap estimation.
FMAP_SUFFIXES = (
    'epi',
    'fieldmap',
    'phasediff',
    'phase1',
    'phase2',
    'magnitude',
    'magnitude1',
    'magnitude2',
)

_BIDS_URI = re.compile(r'^bids:[^:]*:(?P<relpath>.+)$')


def _normalize_to_tuple(value) -> tuple[str, ...]:
    """B0FieldIdentifier/B0FieldSource/IntendedFor may be a string or a list."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def build_signature(metadata: dict) -> DistortionSignature:
    """Extract the distortion-relevant parameters from a merged sidecar."""
    readout_time = metadata.get('TotalReadoutTime')
    if readout_time is not None:
        readout_time = round(float(readout_time) / READOUT_TOLERANCE) * READOUT_TOLERANCE
    shim = metadata.get('ShimSetting')
    if shim is not None:
        shim = tuple(shim)
    return DistortionSignature(
        pe_dir=metadata.get('PhaseEncodingDirection'),
        readout_time=readout_time,
        shim=shim,
        parallel_factor=metadata.get('ParallelReductionFactorInPlane'),
        multiband_factor=metadata.get('MultibandAccelerationFactor'),
    )


def resolve_intended_for(
    entries: tuple[str, ...],
    fmap_path: str,
    bids_root: str,
    subject_id: str,
    known_dwi_files: set[str],
    issues: list[GroupingIssue],
) -> tuple[str, ...]:
    """Resolve IntendedFor entries to absolute paths of this subject's DWIs.

    Entries that resolve to non-DWI targets (e.g. BOLD runs) are legitimate
    uses of a shared fieldmap and are skipped silently. Entries that resolve
    to nothing on disk produce a warning - they are usually curation typos.
    """
    resolved = []
    for entry in entries:
        match = _BIDS_URI.match(entry)
        if match:
            candidate = op.join(bids_root, match.group('relpath'))
        elif op.isabs(entry):
            issues.append(
                warning(
                    'intendedfor-absolute-path',
                    f'IntendedFor entry "{entry}" in {op.basename(fmap_path)} is an '
                    'absolute path, which is not portable BIDS. Using it as-is.',
                    (fmap_path,),
                )
            )
            candidate = entry
        else:
            # The BIDS-legal relative form: relative to the subject directory
            candidate = op.join(bids_root, f'sub-{subject_id}', entry)

        candidate = op.abspath(candidate)
        if candidate in known_dwi_files:
            resolved.append(candidate)
        elif not op.exists(candidate):
            issues.append(
                warning(
                    'intendedfor-missing-target',
                    f'IntendedFor entry "{entry}" in {op.basename(fmap_path)} does not '
                    'match any file in the dataset. Check the path for typos.',
                    (fmap_path,),
                )
            )
        # else: exists but is not one of this subject's DWIs (e.g. a BOLD
        # target of a shared fieldmap) - fine, just not our concern.
    return tuple(resolved)


def _record_from_file(
    path: str,
    layout,
    bids_root: str,
    subject_id: str,
    known_dwi_files: set[str],
    issues: list[GroupingIssue],
) -> FileRecord:
    path = op.abspath(path)
    metadata = layout.get_metadata(path)
    entities = parse_file_entities(path)
    datatype = entities.get('datatype') or ('dwi' if path in known_dwi_files else 'fmap')

    intended_for = ()
    if datatype == 'fmap':
        intended_for = resolve_intended_for(
            _normalize_to_tuple(metadata.get('IntendedFor')),
            fmap_path=path,
            bids_root=bids_root,
            subject_id=subject_id,
            known_dwi_files=known_dwi_files,
            issues=issues,
        )

    return FileRecord(
        path=path,
        datatype=datatype,
        suffix=entities.get('suffix', ''),
        session=entities.get('session'),
        signature=build_signature(metadata),
        b0field_identifiers=_normalize_to_tuple(metadata.get('B0FieldIdentifier')),
        b0field_sources=_normalize_to_tuple(metadata.get('B0FieldSource')),
        multipart_id=metadata.get('MultipartID'),
        intended_for=intended_for,
        metadata=metadata,
    )


def index_subject(
    layout,
    subject_data: dict,
    ignore_fieldmaps: bool = False,
) -> tuple[list[FileRecord], list[GroupingIssue]]:
    """Build a :class:`~.models.FileRecord` for every dwi/ and fmap/ image.

    Parameters
    ----------
    layout : :class:`bids.BIDSLayout`
        Used for sidecar reading (``get_metadata`` applies BIDS inheritance)
        and for discovering fmap/ files when ``subject_data`` lacks them.
    subject_data : dict
        As produced by :func:`qsiprep.utils.bids.collect_data`: at minimum a
        ``'dwi'`` key listing this subject's DWI files; an optional ``'fmap'``
        key listing fieldmap files.
    ignore_fieldmaps : bool
        Skip fmap/ indexing entirely (``--ignore fieldmaps``). The DWI-based
        PEPOLAR heuristic still applies downstream.
    """
    issues: list[GroupingIssue] = []
    dwi_files = [op.abspath(path) for path in subject_data.get('dwi', [])]
    if not dwi_files:
        raise ValueError('subject_data contains no DWI files to group.')

    known_dwi_files = set(dwi_files)
    bids_root = str(layout.root)
    subject_id = parse_file_entities(dwi_files[0])['subject']

    fmap_files = []
    if not ignore_fieldmaps:
        if subject_data.get('fmap') is not None:
            fmap_files = [op.abspath(path) for path in subject_data['fmap']]
        else:
            fmap_files = layout.get(
                return_type='file',
                subject=subject_id,
                datatype='fmap',
                extension=['.nii', '.nii.gz'],
            )
            fmap_files = [op.abspath(path) for path in fmap_files]
        fmap_files = [
            path for path in fmap_files if parse_file_entities(path).get('suffix') in FMAP_SUFFIXES
        ]

    records = [
        _record_from_file(path, layout, bids_root, subject_id, known_dwi_files, issues)
        for path in sorted(known_dwi_files) + sorted(fmap_files)
    ]
    return records, issues
