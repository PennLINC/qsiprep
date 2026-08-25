"""Read everything the grouping needs from the input files, in one place.

This module is the only part of qsiprep's grouping that reads input data:
JSON sidecars, ``.bval`` files, and NIfTI headers. It converts each dwi/,
fmap/, and anat/ image into a :class:`~.models.FileRecord`, normalizing:

- ``B0FieldIdentifier`` / ``B0FieldSource``: string-or-list to tuple.
- ``IntendedFor``: relative-to-subject paths, ``bids::`` URIs, and (with a
  warning) absolute paths, all resolved to absolute paths. pybids'
  ``layout.get_fieldmap`` is deliberately not used - it only understands
  relative-path IntendedFor and cannot query B0Field* fields at all.
- ``ShimSetting``: list to tuple, so signatures are hashable.
- ``TotalReadoutTime``: rounded to :data:`~.models.READOUT_TOLERANCE` so
  float jitter between sidecars does not split groups.
- b-values: shelled/non-shelled classification and the maximum b-value.
- NIfTI headers: the sampling grid, for field-of-view checks.

Everything read from data (as opposed to metadata) degrades to "undetermined"
when a file is unreadable - test skeletons and docs builds use zero-byte
placeholders - and the downstream checks skip undetermined values.
"""

from __future__ import annotations

import os.path as op
import re

import numpy as np
from bids.layout import parse_file_entities

from .models import READOUT_TOLERANCE, DistortionSignature, FileRecord, GridInfo
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

#: Default b=0 threshold (matches qsiprep's ``--b0-threshold`` default).
#: Diffusion-weighting at or below this is treated as b=0. Callers with a
#: configured threshold pass it explicitly.
B0_THRESHOLD = 100.0


def unique_bvals(bvals, tol: float = 20.0):
    """Cluster b-values within ``tol`` of each other, greedily over sorted values.

    Mirrors dipy's ``unique_bvals_tolerance`` (its only use here) so the
    grouping does not need dipy: each sorted unique value starts a new
    cluster when it exceeds the previous representative by more than ``tol``.
    """
    values = np.unique(np.asarray(bvals, dtype=float).reshape(-1))
    representatives = [values[0]]
    for value in values[1:]:
        if value - representatives[-1] > tol:
            representatives.append(value)
    return np.asarray(representatives)


def read_bvals_bvecs(bval_file: str, bvec_file: str):
    """``(bvals, bvecs)`` arrays from sidecar files, FSL row-form transposed.

    Mirrors the dipy reader this replaced: bvals flatten to ``(N,)``, a 3xN
    bvec table becomes ``(N, 3)``, and a volume-count mismatch raises
    ``ValueError`` (callers treat unreadable gradients as absent).
    """
    bvals = np.loadtxt(bval_file).reshape(-1)
    bvecs = np.atleast_2d(np.loadtxt(bvec_file))
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if bvecs.shape != (bvals.size, 3):
        raise ValueError(
            f'{bvec_file} has shape {bvecs.shape}, expected ({bvals.size}, 3)'
        )
    return bvals, bvecs


def _sibling(nii_file: str, ext: str) -> str:
    for nii_ext in ('.nii.gz', '.nii'):
        if nii_file.endswith(nii_ext):
            return nii_file[: -len(nii_ext)] + ext
    return op.splitext(nii_file)[0] + ext


def sibling_bval(nii_file: str) -> str:
    """The FSL ``.bval`` sibling path for a BIDS DWI nii."""
    return _sibling(nii_file, '.bval')


def sibling_bvec(nii_file: str) -> str:
    """The FSL ``.bvec`` sibling path for a BIDS DWI nii."""
    return _sibling(nii_file, '.bvec')


def evaluate_shells(
    bvals,
    b0_threshold: float | None = None,
    tol: float = 100.0,
    min_shell_dirs: int = 6,
    max_shells: int = 7,
) -> tuple[bool | None, tuple[float, ...]]:
    """Classify one series' b-values as shelled or non-shelled sampling.

    Returns ``(shelled, shell_centres)``. Two conditions must both hold for a
    shelled classification (adapted from the retired
    ``_side_is_shelled`` detector in ``workflows/dwi/diffprep.py``):

    1. **Grid guard** - the non-b=0 values cluster into at most ``max_shells``
       distinct shells. A CS-DSI q-space grid fragments into many clusters
       (real HASC55 has ~18 per phase-encoding direction), while DTI has 1 and
       multi-shell HARDI a handful. This is the decisive test: a grid can
       still pack ``min_shell_dirs`` samples near some radius, so a population
       count alone misclassifies it.
    2. **A populous shell** - at least one cluster holds ``min_shell_dirs``
       or more volumes. Unlike the retired DRBUDDI detector, no upper b-value
       limit applies: a single-shell b=2000 acquisition is shelled for eddy's
       purposes even though it is not tensor-fittable at low b.

    ``shelled`` is ``None`` (undetermined) when there are no diffusion-weighted
    volumes to classify.
    """
    if b0_threshold is None:
        b0_threshold = B0_THRESHOLD
    non_b0 = np.asarray(bvals, dtype=float).reshape(-1)
    non_b0 = non_b0[non_b0 >= b0_threshold]
    if non_b0.size == 0:
        return None, ()
    centres = unique_bvals(non_b0, tol=tol)
    shell_centres = tuple(round(float(centre)) for centre in centres)
    if len(centres) > max_shells:
        return False, shell_centres
    shelled = any(
        int(np.sum(np.abs(non_b0 - centre) <= tol)) >= min_shell_dirs for centre in centres
    )
    return shelled, shell_centres


def _read_gradients(
    nii_file: str, b0_threshold: float | None = None
) -> tuple[bool | None, tuple[float, ...], float | None]:
    """(shelled, shell centres, max b-value) from a DWI's sibling .bval file.

    Missing or unreadable b-values (docs builds, test skeletons) leave
    everything undetermined rather than guessing.
    """
    try:
        bvals = np.loadtxt(sibling_bval(nii_file)).reshape(-1)
    except (OSError, ValueError):
        return None, (), None
    if bvals.size == 0:
        return None, (), None
    shelled, shells = evaluate_shells(bvals, b0_threshold=b0_threshold)
    return shelled, shells, float(np.max(bvals))


def _read_grid(nii_file: str) -> GridInfo | None:
    """The sampling grid of a NIfTI file, or None when unreadable.

    Test skeletons and docs builds use zero-byte placeholder files; those
    leave the grid undetermined and the field-of-view checks skip.
    """
    import nibabel as nb

    try:
        img = nb.load(nii_file)
        shape = tuple(int(dim) for dim in img.shape[:3])
        zooms = tuple(round(float(zoom), 3) for zoom in img.header.get_zooms()[:3])
        affine = tuple(tuple(float(val) for val in row) for row in np.asarray(img.affine))
    except Exception:  # noqa: BLE001 - nibabel raises a menagerie on bad files
        return None
    return GridInfo(shape=shape, zooms=zooms, affine=affine)


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
    b0_threshold: float | None = None,
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

    shelled, shells, max_bval, grid = (None, (), None, None)
    if datatype == 'dwi':
        shelled, shells, max_bval = _read_gradients(path, b0_threshold=b0_threshold)
        grid = _read_grid(path)

    return FileRecord(
        path=path,
        datatype=datatype,
        suffix=entities.get('suffix', ''),
        session=entities.get('session'),
        signature=build_signature(metadata),
        b0field_identifiers=_normalize_to_tuple(metadata.get('B0FieldIdentifier')),
        b0field_sources=_normalize_to_tuple(metadata.get('B0FieldSource')),
        multipart_id=_normalize_to_tuple(metadata.get('MultipartID')),
        intended_for=intended_for,
        metadata=metadata,
        shelled=shelled,
        shells=shells,
        max_bval=max_bval,
        grid=grid,
    )


def _collect_datatype_files(layout, subject_data, subject_id, key, datatype, suffixes):
    """Files of one datatype, from subject_data when present, else the layout."""
    if subject_data.get(key) is not None:
        files = [op.abspath(path) for path in subject_data[key]]
    else:
        files = layout.get(
            return_type='file',
            subject=subject_id,
            datatype=datatype,
            extension=['.nii', '.nii.gz'],
        )
        files = [op.abspath(path) for path in files]
    return [path for path in files if parse_file_entities(path).get('suffix') in suffixes]


def index_subject(
    layout,
    subject_data: dict,
    ignore_fieldmaps: bool = False,
    b0_threshold: float | None = None,
) -> tuple[list[FileRecord], list[GroupingIssue]]:
    """Build a :class:`~.models.FileRecord` for every relevant image.

    Indexes the subject's dwi/ series, fmap/ files, and the anatomical images
    that can drive fieldmap-less correction (T1w for SyNb0, T2w for T2Wreg).
    Whether anatomical processing actually runs (``--anat-modality none``) is
    a workflow concern applied downstream, not here.

    Parameters
    ----------
    layout : :class:`bids.BIDSLayout`
        Used for sidecar reading (``get_metadata`` applies BIDS inheritance)
        and for discovering fmap/anat files when ``subject_data`` lacks them.
    subject_data : dict
        As produced by :func:`qsiprep.utils.bids.collect_data`: at minimum a
        ``'dwi'`` key listing this subject's DWI files; optional ``'fmap'``,
        ``'t1w'``, and ``'t2w'`` keys.
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
        fmap_files = _collect_datatype_files(
            layout, subject_data, subject_id, 'fmap', 'fmap', FMAP_SUFFIXES
        )

    anat_files = sorted(
        _collect_datatype_files(layout, subject_data, subject_id, 't1w', 'anat', ('T1w',))
        + _collect_datatype_files(layout, subject_data, subject_id, 't2w', 'anat', ('T2w',))
    )

    records = [
        _record_from_file(
            path,
            layout,
            bids_root,
            subject_id,
            known_dwi_files,
            issues,
            b0_threshold=b0_threshold,
        )
        for path in sorted(known_dwi_files) + sorted(fmap_files) + anat_files
    ]
    return records, issues
