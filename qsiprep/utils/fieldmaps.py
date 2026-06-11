# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Field-map value objects for QSIPrep.

This module mirrors the names and structure of ``sdcflows.fieldmaps`` so that a
future offload of distortion correction to SDCFlows is a deletion-and-delegation
rather than a rewrite. It uses stdlib dataclasses (rather than SDCFlows' attrs)
to avoid taking on a new direct dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto

from nipype.utils.filemanip import split_filename


class EstimatorType(Enum):
    """The kind of field-map estimation a set of sources supports."""

    UNKNOWN = auto()
    PEPOLAR = auto()
    PHASEDIFF = auto()
    MAPPED = auto()
    ANAT = auto()


MODALITIES = {
    'dwi': EstimatorType.PEPOLAR,
    'epi': EstimatorType.PEPOLAR,
    'fieldmap': EstimatorType.MAPPED,
    'magnitude': None,
    'magnitude1': None,
    'magnitude2': None,
    'phase1': EstimatorType.PHASEDIFF,
    'phase2': EstimatorType.PHASEDIFF,
    'phasediff': EstimatorType.PHASEDIFF,
    'T1w': EstimatorType.ANAT,
    'T2w': EstimatorType.ANAT,
}


def _suffix_of(path):
    """Return the BIDS suffix of a NIfTI path (e.g. 'phasediff')."""
    return split_filename(path)[1].split('_')[-1]


@dataclass(slots=True)
class FieldmapFile:
    """A single file usable in field-map estimation, with metadata read once.

    Parameters
    ----------
    path : str
        Path to a NIfTI file in a BIDS tree.
    metadata : dict, optional
        Metadata for the file. Caller is responsible for providing BIDS-resolved
        metadata (QSIPrep reads it from the pybids layout).
    """

    path: str
    metadata: dict = field(default_factory=dict)
    suffix: str = field(init=False)

    def __post_init__(self):
        self.suffix = _suffix_of(self.path)

    def find_siblings(self, suffixes):
        """Return ``{suffix: path}`` for sibling files that exist on disk.

        Siblings are located by replacing this file's suffix token in its
        basename, matching pybids' ``get_fieldmap`` filename convention.
        """
        dirname, basename = os.path.split(self.path)
        found = {}
        for sibling_suffix in suffixes:
            sibling_basename = basename.replace(f'_{self.suffix}.', f'_{sibling_suffix}.')
            if sibling_basename == basename:
                continue
            candidate = os.path.join(dirname, sibling_basename)
            if os.path.exists(candidate):
                found[sibling_suffix] = candidate
        return found


_PEPOLAR_SUFFIXES = ('dwi', 'epi')
# ``phase2`` is intentionally absent: it is always a sibling of ``phase1`` never an
# estimation primary, so an estimation is never keyed on it. This keeps method
# inference honest and avoids a degenerate phase2-only source set inferring
# PHASEDIFF and then failing to find a phase1 primary in ``to_fieldmap_info``.
_GRE_SUFFIXES = ('fieldmap', 'phasediff', 'phase1')


@dataclass(slots=True)
class FieldmapEstimation:
    """A set of :class:`FieldmapFile`s that together define one field map.

    The estimation ``method`` is inferred from the suffixes of the sources. The
    ``bids_id`` is the common ``B0FieldIdentifier`` of the sources, or the
    provided ``auto_id`` fallback when none is set.

    Parameters
    ----------
    sources : list[FieldmapFile]
    auto_id : str, optional
        Identifier to use when the sources carry no ``B0FieldIdentifier``.
    bids_id : str, optional
        Explicit identifier for this estimation. When provided, it is used
        verbatim and the per-source ``B0FieldIdentifier`` resolution (and its
        conflict check) is skipped. This is needed when a single source file
        carries several ``B0FieldIdentifier`` values and thus belongs to several
        estimations -- the caller already knows which one this is.
    """

    sources: list
    auto_id: str | None = None
    bids_id: str | None = None
    method: EstimatorType = field(init=False, default=EstimatorType.UNKNOWN)

    def __post_init__(self):
        suffixes = {f.suffix for f in self.sources}

        gre = suffixes.intersection(_GRE_SUFFIXES)
        pepolar = suffixes.intersection(_PEPOLAR_SUFFIXES)

        if gre:
            # phase1/phase2 are the same method; any other mix is invalid.
            if len(gre) > 1 and gre - {'phase1', 'phase2'}:
                raise ValueError(f'Incompatible field-map suffixes: {sorted(gre)}')
            self.method = MODALITIES[sorted(gre)[0]]
        elif pepolar:
            self.method = EstimatorType.PEPOLAR
        else:
            raise ValueError('Insufficient sources to estimate a field map.')

        if self.bids_id is not None:
            # Caller supplied an explicit id; skip per-source resolution.
            return

        b0_ids = [
            f.metadata['B0FieldIdentifier']
            for f in self.sources
            if f.metadata.get('B0FieldIdentifier') is not None
        ]
        # Flatten list-valued identifiers.
        flat_ids = set()
        for value in b0_ids:
            flat_ids.update(value if isinstance(value, list) else [value])

        if flat_ids:
            if len(flat_ids) > 1:
                raise ValueError(f'Conflicting B0FieldIdentifiers: {sorted(flat_ids)}')
            self.bids_id = flat_ids.pop()
        else:
            self.bids_id = self.auto_id

    def paths(self):
        """Sorted tuple of source paths."""
        return tuple(sorted(str(f.path) for f in self.sources))

    def to_fieldmap_info(self, epi_files=None, rpe_files=None):
        """Build the legacy ``fieldmap_info`` dict for downstream workflows.

        For GRE field maps (MAPPED / PHASEDIFF) the dict is built from the
        sources and their sibling magnitude/phase files. For PEPOLAR field maps
        the caller supplies the target-specific ``epi_files`` (EPIs from
        ``fmap/``) and ``rpe_files`` (reverse-PE DWI series), because which
        series are sources vs. targets depends on the distortion group being
        corrected.
        """
        if self.method is EstimatorType.MAPPED:
            src = next(f for f in self.sources if f.suffix == 'fieldmap')
            info = {'suffix': 'fieldmap', 'fieldmap': str(src.path)}
            mag = self._sibling_path(src, 'magnitude')
            if mag is not None:
                info['magnitude'] = mag
            return info

        if self.method is EstimatorType.PHASEDIFF:
            phasediff = next((f for f in self.sources if f.suffix == 'phasediff'), None)
            if phasediff is not None:
                info = {'suffix': 'phasediff', 'phasediff': str(phasediff.path)}
                for mag in ('magnitude1', 'magnitude2'):
                    sib = self._sibling_path(phasediff, mag)
                    if sib is not None:
                        info[mag] = sib
                return info
            # phase1/phase2
            phase1 = next(f for f in self.sources if f.suffix == 'phase1')
            info = {'suffix': 'phase1', 'phase1': str(phase1.path)}
            for sib_suffix in ('phase2', 'magnitude1', 'magnitude2'):
                sib = self._sibling_path(phase1, sib_suffix)
                if sib is not None:
                    info[sib_suffix] = sib
            return info

        # PEPOLAR
        epi_files = sorted(epi_files or [])
        rpe_files = sorted(rpe_files or [])
        if rpe_files:
            info = {'suffix': 'rpe_series', 'rpe_series': rpe_files}
            if epi_files:
                info['epi'] = epi_files
            return info
        if epi_files:
            return {'suffix': 'epi', 'epi': epi_files}
        return {'suffix': None}

    def _sibling_path(self, source, suffix):
        """Return a sibling path already among sources, else discover on disk."""
        for f in self.sources:
            if f.suffix == suffix:
                return str(f.path)
        found = source.find_siblings((suffix,))
        return found.get(suffix)
