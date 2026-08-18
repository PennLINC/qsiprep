"""Tests for :mod:`qsiprep.grouping.adapters`.

The adapter bridges the grouping model to the legacy ``scan_groups`` dicts the
workflow builders still consume. Two things are checked here:

- **Differential parity** against the retired ``group_dwi_scans`` on the shared
  skeleton fixtures, with every intended divergence asserted explicitly.
- **Per-method shapes**: that each :class:`~qsiprep.grouping.models.CorrectionMethod`
  renders the ``fieldmap_info`` the downstream workflows expect.
"""

import os.path as op

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.grouping import (
    CorrectionMethod,
    backend_for_config,
    build_dwi_grouping,
    to_legacy_scan_groups,
    to_preproc_units,
)
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


def test_multiped_pools_all_directions(tmp_path):
    """No fieldmaps, four PE directions: one pooled estimation, one output.

    Legacy built one reverse-PE output per axis. Any two differing phase
    encodings jointly determine the susceptibility field, so the model pools
    all four series into a single estimation and output; whether a backend
    can consume that shape is check_backend's call (TOPUP can, DRBUDDI
    raises drbuddi-cross-axis).
    """
    layout, subject_data = _load_skeleton('skeleton_simple_multiped', tmp_path)
    legacy, _ = group_dwi_scans(layout, subject_data)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)
    new, scheme = to_legacy_scan_groups(grouping)

    assert len(legacy) == 2  # legacy: one output per axis

    (estimation,) = grouping.estimations.values()
    assert estimation.pe_axes == {'i', 'j'}
    assert estimation.bidirectional_axes == {'i', 'j'}

    (group,) = new
    names = _basenames(group)
    assert group['concatenated_bids_name'] == 'sub-01'
    assert names['fieldmap_info']['suffix'] == 'rpe_series'
    assert len(names['dwi_series']) + len(names['fieldmap_info']['rpe_series']) == 4
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


def test_syn_legacy_and_unit_shape(tmp_path):
    """SyN units expose is_nipreps_syn and render the legacy 'syn' fieldmap shape."""
    (unit,) = _units('fieldmapless_t1w_only', tmp_path, use_nipreps_syn_sdc=True)
    assert unit.is_nipreps_syn
    assert not unit.has_scanner_measured_fieldmap
    assert unit.to_legacy_dict()['fieldmap_info'] == {'suffix': 'syn'}


def test_synb0_not_implemented(tmp_path):
    """SyNb0 has no legacy shape yet; the adapter says so loudly."""
    grouping = load_scenario('fieldmapless_t1w_only', tmp_path, strict=False, use_synb0=True)
    with pytest.raises(NotImplementedError, match='SyNb0'):
        to_legacy_scan_groups(grouping)


# -- Native PreprocUnit surface ------------------------------------------------


def _units(scenario, tmp_path, **kwargs):
    return to_preproc_units(load_scenario(scenario, tmp_path, strict=False, **kwargs))


def test_preproc_units_match_legacy_partition(tmp_path):
    """Units and legacy scan groups are the same partition, named identically."""
    grouping = load_scenario('mixed_trt', tmp_path, strict=False)
    units = to_preproc_units(grouping)
    legacy, _ = to_legacy_scan_groups(grouping)
    assert [unit.output_name for unit in units] == [g['concatenated_bids_name'] for g in legacy]


def test_preproc_unit_pepolar_bidirectional(tmp_path):
    """A reverse-PE pair exposes its plus/minus split natively."""
    (unit,) = _units('mixed_trt', tmp_path)
    assert unit.method is CorrectionMethod.PEPOLAR
    assert unit.has_bidirectional_dwi
    assert _basenames(list(unit.plus_files)) == ['sub-01_dir-PA_dwi.nii.gz']
    assert _basenames(list(unit.minus_files)) == ['sub-01_dir-AP_dwi.nii.gz']
    assert unit.pe_dir == 'j'
    assert unit.extra_b0 == ()


def test_preproc_unit_extra_b0_from_epi_fmap(tmp_path):
    """A dedicated epi fieldmap is exposed as an extra b=0 source, not a member."""
    (unit,) = _units('abcd_style', tmp_path)
    assert unit.method is CorrectionMethod.PEPOLAR
    assert not unit.has_bidirectional_dwi
    assert _basenames(list(unit.extra_b0)) == ['sub-01_dir-PA_epi.nii.gz']
    assert unit.pe_dir == 'j-'


def test_preproc_unit_gre_files(tmp_path):
    """A GRE unit exposes its fieldmap files keyed by BIDS suffix."""
    (unit,) = _units('gre_phasediff', tmp_path)
    assert unit.method is CorrectionMethod.PHASEDIFF
    gre = {suffix: op.basename(path) for suffix, path in unit.gre_files().items()}
    assert gre['phasediff'] == 'sub-01_phasediff.nii.gz'
    assert gre['magnitude1'] == 'sub-01_magnitude1.nii.gz'


def test_preproc_unit_uncorrected(tmp_path):
    """A series with no fieldmap has no estimation and no method."""
    units = _units('missing_pedir', tmp_path)
    assert units
    assert all(unit.estimation is None and unit.method is None for unit in units)


def test_sidecar_overrides_carry_pe_and_readout(tmp_path):
    """The override map carries each series' PE dir and readout time from the model."""
    (unit,) = _units('mixed_trt', tmp_path)
    overrides = unit.sidecar_overrides()
    assert set(_basenames(list(overrides))) == {
        'sub-01_dir-AP_dwi.nii.gz',
        'sub-01_dir-PA_dwi.nii.gz',
    }
    for spec in overrides.values():
        assert spec['PhaseEncodingDirection'] in ('j', 'j-')
        assert 'TotalReadoutTime' in spec


def test_concatenation_scheme_multi_unit(tmp_path):
    """Two corrected units in one final output map to the shared final name."""

    scan_groups, scheme = _adapt('two_gre_fmaps', tmp_path)
    assert scheme == {
        'sub-01_dir-AP_run-1': 'sub-01_dir-AP',
        'sub-01_dir-AP_run-2': 'sub-01_dir-AP',
    }
    assert {group['concatenated_bids_name'] for group in scan_groups} == set(scheme)


def test_concatenation_scheme_identity_for_single_unit(tmp_path):
    """A single-unit output maps to itself."""
    _, scheme = _adapt('hcp_style', tmp_path)
    assert scheme == {'sub-01': 'sub-01'}
