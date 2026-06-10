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
            sibling_basename = basename.replace(
                f'_{self.suffix}.', f'_{sibling_suffix}.'
            )
            if sibling_basename == basename:
                continue
            candidate = os.path.join(dirname, sibling_basename)
            if os.path.exists(candidate):
                found[sibling_suffix] = candidate
        return found
