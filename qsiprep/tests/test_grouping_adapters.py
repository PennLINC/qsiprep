"""Tests for :mod:`qsiprep.grouping.adapters`.

The adapter bridges the grouping model to the legacy ``scan_groups`` dicts the
workflow builders still consume. Two things are checked here:

- **Differential parity** against the retired ``group_dwi_scans`` on the shared
  skeleton fixtures, with every intended divergence asserted explicitly.
- **Per-method shapes**: that each :class:`~qsiprep.grouping.models.EstimationMethod`
  renders the ``fieldmap_info`` the downstream workflows expect.
"""

import os.path as op

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.grouping import backend_for_config, build_dwi_grouping, to_legacy_scan_groups
from qsiprep.tests.grouping_scenarios import load_scenario
from qsiprep.tests.utils import get_test_data_path
from qsiprep.utils.grouping import group_dwi_scans


def _basenames(value):
    """Recursively replace absolute paths with basenames for readable asserts."""
    if isinstance(value, dict):
        return {key: _basenames(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_basenames(val) for val in value]
    if isinstance(value, str) and '/' in value:
        return op.basename(value)
    return value


def _normalize(scan_groups):
    """Order-independent, basename-only rendering of a scan_groups list."""
    return sorted(
        (_basenames(group) for group in scan_groups),
        key=lambda group: (group['concatenated_bids_name'], group['dwi_series']),
    )


def _load_skeleton(name, tmp_path):
    bids_dir = tmp_path / name
    generate_bids_skeleton(str(bids_dir), op.join(get_test_data_path(), f'{name}.yml'))
    layout = BIDSLayout(str(bids_dir), validate=False)
    dwi_files = layout.get(
        suffix='dwi', datatype='dwi', extension=['.nii', '.nii.gz'], return_type='file'
    )
    return layout, {'dwi': sorted(dwi_files)}


def _adapt(scenario, tmp_path, **kwargs):
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    scan_groups, scheme = to_legacy_scan_groups(grouping)
    return scan_groups, scheme


@pytest.mark.parametrize(
    ('hmc_model', 'pepolar_method', 'expected'),
    [
        ('eddy', 'TOPUP', 'fsl'),
        ('eddy', 'DRBUDDI', 'mixed'),
        ('eddy', 'TOPUP+DRBUDDI', 'mixed'),
        ('tortoise', 'TOPUP', 'tortoise'),
        ('tortoise', 'DRBUDDI', 'tortoise'),
        ('3dSHORE', 'TOPUP', 'tortoise'),
        ('tensor', 'TOPUP', 'tortoise'),
        ('none', 'TOPUP', 'tortoise'),
    ],
)
def test_backend_for_config(hmc_model, pepolar_method, expected):
    assert backend_for_config(hmc_model, pepolar_method) == expected


def test_multiped_matches_legacy(tmp_path):
    """No fieldmaps, four PE directions: the adapter reproduces group_dwi_scans."""
    layout, subject_data = _load_skeleton('skeleton_simple_multiped', tmp_path)
    legacy, _ = group_dwi_scans(layout, subject_data)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)
    new, scheme = to_legacy_scan_groups(grouping)

    assert _normalize(new) == _normalize(legacy)
    assert scheme == {name: name for name in scheme}


def test_relpaths_curation_outranks_reverse_pe(tmp_path):
    """Each DWI has a dedicated PEPOLAR fieldmap and a reverse-PE sibling.

    Legacy paired the DWI series against each other into one merged output;
    the model instead honors the curated fieldmap linkage, correcting each
    series with its own epi fieldmap in a separate output.
    """
    layout, subject_data = _load_skeleton('skeleton_complex_relpaths', tmp_path)
    legacy, _ = group_dwi_scans(layout, subject_data)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)
    new, _ = to_legacy_scan_groups(grouping)

    # Legacy collapsed everything into a single reverse-PE-series output.
    assert len(legacy) == 1
    assert legacy[0]['fieldmap_info']['suffix'] == 'rpe_series'

    # The model keeps the two curated groups apart, each with its own epi fmap.
    by_name = {group['concatenated_bids_name']: _basenames(group) for group in new}
    assert set(by_name) == {'sub-01_dir-AP', 'sub-01_dir-PA'}
    assert by_name['sub-01_dir-AP']['fieldmap_info'] == {
        'suffix': 'epi',
        'epi': ['sub-01_dir-PA_epi.nii.gz'],
    }
    assert by_name['sub-01_dir-AP']['dwi_series'] == [
        'sub-01_dir-AP_run-1_dwi.nii.gz',
        'sub-01_dir-AP_run-2_dwi.nii.gz',
    ]
    assert by_name['sub-01_dir-PA']['fieldmap_info'] == {
        'suffix': 'epi',
        'epi': ['sub-01_dir-AP_epi.nii.gz'],
    }


def test_identity_concatenation_scheme(tmp_path):
    _, scheme = _adapt('mixed_trt', tmp_path)
    assert scheme == {name: name for name in scheme}


def test_pepolar_rpe_series_shape(tmp_path):
    """Opposite-PE DWI series correcting each other become one rpe_series group."""
    (group,), _ = _adapt('mixed_trt', tmp_path)
    info = _basenames(group['fieldmap_info'])
    assert info['suffix'] == 'rpe_series'
    assert _basenames(group['dwi_series']) == ['sub-01_dir-PA_dwi.nii.gz']
    assert info['rpe_series'] == ['sub-01_dir-AP_dwi.nii.gz']
    assert group['dwi_series_pedir'] == 'j'
    assert 'epi' not in info


def test_pepolar_epi_shape(tmp_path):
    """A DWI with a dedicated epi fieldmap becomes an epi group."""
    (group,), _ = _adapt('abcd_style', tmp_path)
    info = _basenames(group['fieldmap_info'])
    assert info == {'suffix': 'epi', 'epi': ['sub-01_dir-PA_epi.nii.gz']}
    assert _basenames(group['dwi_series']) == ['sub-01_dir-AP_dwi.nii.gz']
    assert group['dwi_series_pedir'] == 'j-'


def test_borrowed_series_become_epi(tmp_path):
    """DWI series borrowed from another output ride in the epi list."""
    scan_groups, _ = _adapt('multipart_splits_estimation', tmp_path)
    by_name = {group['concatenated_bids_name']: _basenames(group) for group in scan_groups}
    info = by_name['sub-01_run-1']['fieldmap_info']
    assert info['suffix'] == 'rpe_series'
    assert info['rpe_series'] == ['sub-01_dir-AP_run-1_dwi.nii.gz']
    # The run-2 series live in a different output; they only lend b=0 images.
    assert info['epi'] == [
        'sub-01_dir-AP_run-2_dwi.nii.gz',
        'sub-01_dir-PA_run-2_dwi.nii.gz',
    ]


def test_gre_phasediff_shape(tmp_path):
    """A phasediff fieldmap keys its files by suffix and carries its metadata."""
    (group,), _ = _adapt('gre_phasediff', tmp_path)
    info = _basenames(group['fieldmap_info'])
    assert info['suffix'] == 'phasediff'
    assert info['phasediff'] == 'sub-01_phasediff.nii.gz'
    assert info['magnitude1'] == 'sub-01_magnitude1.nii.gz'
    assert info['magnitude2'] == 'sub-01_magnitude2.nii.gz'
    assert isinstance(info['metadata'], dict)


def test_t2wreg_is_fieldmapless(tmp_path):
    """T2Wreg is expressed as fieldmap-less; SDC is driven by t2w_sdc elsewhere."""
    (group,), _ = _adapt('fieldmapless_t2w', tmp_path)
    assert group['fieldmap_info'] == {'suffix': None}
    assert group['dwi_series']


def test_uncorrected_group_has_no_fieldmap(tmp_path):
    """A series with no distortion information gets a None fieldmap."""
    scan_groups, _ = _adapt('missing_pedir', tmp_path)
    assert all(group['fieldmap_info'] == {'suffix': None} for group in scan_groups)


def test_synb0_not_implemented(tmp_path):
    """SyNb0 has no legacy shape yet; the adapter says so loudly."""
    grouping = load_scenario('fieldmapless_t1w_only', tmp_path, strict=False, use_synb0=True)
    with pytest.raises(NotImplementedError, match='SyNb0'):
        to_legacy_scan_groups(grouping)
