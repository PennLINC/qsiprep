"""Tests for the field-map value-object model."""

import os

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
            fm.FieldmapFile(
                ap, metadata={'PhaseEncodingDirection': 'j', 'B0FieldIdentifier': 'pp1'}
            ),
            fm.FieldmapFile(
                pa, metadata={'PhaseEncodingDirection': 'j-', 'B0FieldIdentifier': 'pp1'}
            ),
        ]
    )
    assert est.bids_id == 'pp1'


def test_estimation_auto_id_when_unnamed(tmp_path):
    pd = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    m1 = _touch(tmp_path / 'sub-01_magnitude1.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006}),
            fm.FieldmapFile(m1),
        ],
        auto_id='auto_00000',
    )
    assert est.bids_id == 'auto_00000'


def test_fieldmap_info_phasediff(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    pd = _touch(fmap_dir / 'sub-01_phasediff.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude1.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude2.nii.gz')
    est = fm.FieldmapEstimation(
        [fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})]
    )
    info = est.to_fieldmap_info()
    assert info['suffix'] == 'phasediff'
    assert info['phasediff'].endswith('sub-01_phasediff.nii.gz')
    assert info['magnitude1'].endswith('sub-01_magnitude1.nii.gz')
    assert info['magnitude2'].endswith('sub-01_magnitude2.nii.gz')
    assert 'epi' not in info


def test_fieldmap_info_fieldmap(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    fmapf = _touch(fmap_dir / 'sub-01_fieldmap.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude.nii.gz')
    est = fm.FieldmapEstimation([fm.FieldmapFile(fmapf, metadata={'Units': 'Hz'})])
    info = est.to_fieldmap_info()
    assert info['suffix'] == 'fieldmap'
    assert info['fieldmap'].endswith('sub-01_fieldmap.nii.gz')
    assert info['magnitude'].endswith('sub-01_magnitude.nii.gz')


def test_fieldmap_info_pepolar_epi_only(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    info = est.to_fieldmap_info(epi_files=[ap, pa], rpe_files=[])
    assert info == {'suffix': 'epi', 'epi': sorted([ap, pa])}


def test_fieldmap_info_pepolar_rpe_series(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    rpe = _touch(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(rpe, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    info = est.to_fieldmap_info(epi_files=[ap], rpe_files=[rpe])
    assert info['suffix'] == 'rpe_series'
    assert info['rpe_series'] == [rpe]
    assert info['epi'] == [ap]
