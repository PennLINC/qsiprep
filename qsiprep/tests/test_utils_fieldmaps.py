"""Tests for the field-map value-object model."""

import pytest

from qsiprep.utils import fieldmaps as fm


def test_estimator_types_exist():
    assert {t.name for t in fm.EstimatorType} >= {
        'UNKNOWN',
        'PEPOLAR',
        'PHASEDIFF',
        'MAPPED',
        'ANAT',
    }


def test_modalities_mapping():
    assert fm.MODALITIES['dwi'] is fm.EstimatorType.PEPOLAR
    assert fm.MODALITIES['epi'] is fm.EstimatorType.PEPOLAR
    assert fm.MODALITIES['phasediff'] is fm.EstimatorType.PHASEDIFF
    assert fm.MODALITIES['phase1'] is fm.EstimatorType.PHASEDIFF
    assert fm.MODALITIES['fieldmap'] is fm.EstimatorType.MAPPED
    assert fm.MODALITIES['T1w'] is fm.EstimatorType.ANAT
    assert fm.MODALITIES['magnitude1'] is None
