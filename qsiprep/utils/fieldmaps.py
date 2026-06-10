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
