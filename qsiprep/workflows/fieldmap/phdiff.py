#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field inhomogeneity inside the scanner (fieldmap) is proportional to the
phase drift between two subsequent :abbr:`GRE (gradient recall echo)`
sequence.


Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image
8.9.2 in BIDS 1.0.0: two phases and at least one magnitude image

"""

from nipype.interfaces import ants, afni, utility as niu
from nipype.pipeline import engine as pe
from .utils import cleanup_edge_pipeline, siemens2rads, demean_image
from ...niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ...niworkflows.interfaces.bids import ReadSidecarJSON
from ...niworkflows.interfaces.images import IntraModalMerge
from ...niworkflows.interfaces.masks import BrainExtractionRPT

from ...interfaces import Phasediff2Fieldmap, Phases2Fieldmap, DerivativesDataSink
