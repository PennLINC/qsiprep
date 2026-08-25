"""Build :class:`~qsiprep.grouping.adapters.PreprocUnit` objects in memory.

The workflow-construction tests need units without materializing a BIDS layout.
:func:`make_preproc_unit` assembles the minimal :class:`~qsiprep.grouping.models`
objects a unit's accessors touch (file records, one estimation) so builders can
be constructed and their graphs asserted.
"""

from __future__ import annotations

import os.path as op

from qsiprep.grouping.adapters import PreprocUnit
from qsiprep.grouping.models import (
    AUTO_PREFIX,
    CorrectionMethod,
    DistortionSignature,
    DWIGrouping,
    FieldmapEstimation,
    FileRecord,
    Provenance,
    derive_output_name,
)

_SUFFIX_TAILS = (
    'phasediff',
    'magnitude1',
    'magnitude2',
    'magnitude',
    'phase1',
    'phase2',
    'fieldmap',
    'epi',
    'sbref',
    'dwi',
)


def _suffix_of(path: str) -> str:
    stem = op.basename(path).split('.')[0]
    for tail in _SUFFIX_TAILS:
        if stem.endswith(tail):
            return tail
    return 'dwi'


def _datatype_of(suffix: str) -> str:
    if suffix == 'dwi':
        return 'dwi'
    return 'fmap'


def make_preproc_unit(
    dwi_files,
    *,
    method: CorrectionMethod | None = None,
    pe_dir: str = 'j',
    pe_dirs: dict[str, str] | None = None,
    estimation_sources=None,
    readout_time: float = 0.05,
    metadata: dict | None = None,
    per_file_metadata: dict[str, dict] | None = None,
    shelled: bool = True,
    provenance: Provenance = Provenance.INFERRED,
    b0field_id: str | None = None,
    output_name: str | None = None,
) -> PreprocUnit:
    """Assemble a :class:`PreprocUnit` from bare file paths.

    ``pe_dirs`` overrides the phase-encoding direction per file (keyed by path);
    ``estimation_sources`` overrides the estimation's source list (defaults to
    the member DWIs). Non-DWI sources get a file record whose suffix is inferred
    from the filename (``*_epi`` -> ``epi``, ``*_phasediff`` -> ``phasediff`` ...).
    ``per_file_metadata`` overrides sidecar keys for individual files (keyed by
    path), layered on top of ``metadata``.
    """
    dwi_files = list(dwi_files)
    metadata = dict(metadata or {})
    pe_dirs = dict(pe_dirs or {})
    sources = list(dwi_files if estimation_sources is None else estimation_sources)

    files: dict[str, FileRecord] = {}
    for path in dict.fromkeys(dwi_files + sources):
        suffix = _suffix_of(path)
        this_pe = pe_dirs.get(path, pe_dir)
        record_meta = {'PhaseEncodingDirection': this_pe, 'TotalReadoutTime': readout_time}
        record_meta.update(metadata)
        record_meta.update((per_file_metadata or {}).get(path, {}))
        files[path] = FileRecord(
            path=path,
            datatype=_datatype_of(suffix),
            suffix=suffix,
            session=None,
            signature=DistortionSignature(pe_dir=this_pe, readout_time=readout_time),
            metadata=record_meta,
            shelled=shelled if suffix == 'dwi' else None,
        )

    estimation = None
    if method is not None:
        estimation = FieldmapEstimation(
            b0field_id=b0field_id or (AUTO_PREFIX + 'fmap'),
            method=method,
            sources=tuple(sorted(sources)),
            provenance=provenance,
        )

    grouping = DWIGrouping(
        subject_id='01',
        files=files,
        estimations={estimation.b0field_id: estimation} if estimation else {},
        application={},
        application_provenance={},
        application_candidates={},
        distortion_groups={},
        concatenation_groups={},
    )
    return PreprocUnit(
        grouping=grouping,
        output_name=output_name or derive_output_name(dwi_files),
        dwi_files=tuple(dwi_files),
        estimation=estimation,
    )
