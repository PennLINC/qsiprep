"""Tests for the field-map value-object model."""

import os

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


def _touch(path, content=''):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def test_fieldmapfile_reads_suffix_and_metadata(tmp_path):
    nii = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    f = fm.FieldmapFile(nii, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})
    assert f.suffix == 'phasediff'
    assert f.metadata['EchoTime1'] == 0.004
    assert f.path == nii


def test_fieldmapfile_finds_sibling_magnitudes(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    pd = _touch(fmap_dir / 'sub-01_phasediff.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude1.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude2.nii.gz')
    f = fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})
    siblings = f.find_siblings(('magnitude1', 'magnitude2'))
    assert sorted(os.path.basename(s) for s in siblings.values()) == [
        'sub-01_magnitude1.nii.gz',
        'sub-01_magnitude2.nii.gz',
    ]


def test_estimation_infers_pepolar(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    assert est.method is fm.EstimatorType.PEPOLAR


def test_estimation_infers_phasediff(tmp_path):
    pd = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    m1 = _touch(tmp_path / 'sub-01_magnitude1.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006}),
            fm.FieldmapFile(m1),
        ]
    )
    assert est.method is fm.EstimatorType.PHASEDIFF


def test_estimation_uses_b0fieldidentifier_as_id(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j', 'B0FieldIdentifier': 'pp1'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-', 'B0FieldIdentifier': 'pp1'}),
        ]
    )
    assert est.bids_id == 'pp1'


def test_estimation_auto_id_when_unnamed(tmp_path):
    pd = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    m1 = _touch(tmp_path / 'sub-01_magnitude1.nii.gz')
    est = fm.FieldmapEstimation(
        [fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006}), fm.FieldmapFile(m1)],
        auto_id='auto_00000',
    )
    assert est.bids_id == 'auto_00000'
