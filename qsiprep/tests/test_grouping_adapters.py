"""Tests for :mod:`qsiprep.grouping.adapters`.

The adapter bridges the grouping model to the :class:`PreprocUnit` objects the
workflow builders consume. Checked here: the per-method unit shapes (PEPOLAR
splits, GRE files, fieldmap-less markers), the backend-aware decomposition,
and the concatenation scheme. Routing parity against the compiled execution
plan lives in ``test_grouping_plan.py``.
"""

import os.path as op

from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.grouping import (
    CorrectionMethod,
    build_dwi_grouping,
    concatenation_scheme,
    to_preproc_units,
)
from qsiprep.tests.grouping_scenarios import load_scenario
from qsiprep.tests.utils import get_test_data_path


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


def _units(scenario, tmp_path, backend='fsl', **kwargs):
    return to_preproc_units(load_scenario(scenario, tmp_path, strict=False, **kwargs), backend)


def test_multiped_pools_all_directions(tmp_path):
    """No fieldmaps, four PE directions: one pooled estimation, one output.

    Any two differing phase encodings jointly determine the susceptibility
    field, so the model pools all four series into a single estimation and
    output. FSL keeps that pooled shape (one TOPUP+eddy); TORTOISE splits it
    per axis - see ``test_multiped_tortoise_splits_per_axis``.
    """
    layout, subject_data = _load_skeleton('skeleton_simple_multiped', tmp_path)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)

    (estimation,) = grouping.estimations.values()
    assert estimation.pe_axes == {'i', 'j'}
    assert estimation.bidirectional_axes == {'i', 'j'}

    (unit,) = to_preproc_units(grouping, 'fsl')
    assert unit.output_name == 'sub-01'
    assert unit.has_bidirectional_dwi
    assert unit.pepolar_fieldmap_type == 'rpe_series'
    assert len(unit.dwi_files) == 4
    scheme = concatenation_scheme(grouping, 'fsl')
    assert scheme == {name: name for name in scheme}


def test_multiped_tortoise_splits_per_axis(tmp_path):
    """TORTOISE splits the pooled multi-axis cluster into one unit per axis.

    The four-direction cluster is a single pooled estimation (one TOPUP+eddy for
    FSL), but DRBUDDI corrects one axis at a time. For the tortoise backend the
    adapter yields one PreprocUnit per phase-encoding axis, each an opposing
    pair seeing only its own axis, and the concatenation scheme maps both to the
    shared final output so the corrected results are recombined by the merge.
    """
    layout, subject_data = _load_skeleton('skeleton_simple_multiped', tmp_path)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)

    # FSL keeps one pooled unit, identity scheme.
    fsl_units = to_preproc_units(grouping, backend='fsl')
    assert [unit.output_name for unit in fsl_units] == ['sub-01']

    # TORTOISE: one unit per axis, both bidirectional, both -> 'sub-01'.
    units = to_preproc_units(grouping, backend='tortoise')
    assert len(units) == 2
    assert all(unit.has_bidirectional_dwi for unit in units)
    assert {unit.pe_axis for unit in units} == {'i', 'j'}
    for unit in units:
        # each split unit sees only its own axis - no cross-axis borrowing
        assert {grouping.files[path].signature.pe_axis for path in unit.dwi_files} == {
            unit.pe_axis
        }
        assert unit.extra_b0 == ()

    scheme = concatenation_scheme(grouping, backend='tortoise')
    assert set(scheme) == {unit.output_name for unit in units}  # keyed by unit name
    assert len(scheme) == 2  # two distinct per-axis unit names
    assert set(scheme.values()) == {'sub-01'}


def test_multi_readout_splits_per_pair(tmp_path):
    """TORTOISE splits one axis at two readout times into per-readout pairs.

    DRBUDDI must match the readout time along with the axis, so a pooled
    estimation carrying two blip pairs on the same axis (acq-fast/acq-slow)
    becomes one per-readout PreprocUnit, each seeing only its own readout; FSL
    keeps the single pooled estimation.
    """
    grouping = load_scenario('multi_readout', tmp_path, strict=False)

    assert [unit.output_name for unit in to_preproc_units(grouping, backend='fsl')] == ['sub-01']

    units = to_preproc_units(grouping, backend='tortoise')
    assert {unit.output_name for unit in units} == {'sub-01_acq-fast', 'sub-01_acq-slow'}
    for unit in units:
        assert unit.has_bidirectional_dwi
        assert unit.pe_axis == 'j'
        readouts = {grouping.files[path].signature.readout_time for path in unit.dwi_files}
        assert len(readouts) == 1  # one readout time per split unit
        assert unit.extra_b0 == ()

    scheme = concatenation_scheme(grouping, backend='tortoise')
    assert scheme == {'sub-01_acq-fast': 'sub-01', 'sub-01_acq-slow': 'sub-01'}

    # The pooled FSL/mixed unit is multi-group (mixed skips DRBUDDI for it); each
    # per-pair TORTOISE sub-unit is a single blip pair that DRBUDDI can consume.
    (pooled,) = to_preproc_units(grouping, backend='fsl')
    assert not pooled.is_single_blip_pair
    assert all(unit.is_single_blip_pair for unit in units)


def test_single_unit_acq_output_exposes_multipartid_label(tmp_path):
    """A single-unit curated ``acq-`` output must expose its MultipartID label
    as the final output name, distinct from the entity-derived unit key.

    ``base.py`` names each single-unit output's derivatives from the concatenation
    scheme's final name (not the unit key), so this is what carries a curated
    ``acq-<label>`` into the derivative filenames. Regression guard: when this
    was the identity, single-unit virtual-acquisition outputs lost their
    labels (e.g. ``acq-dsi258`` fell back to a bare name).
    """
    grouping = load_scenario('virtual_acq_multipart', tmp_path, strict=False)

    # Every output here is a single correction unit (no cross-unit merge).
    scheme = concatenation_scheme(grouping, backend='fsl')
    assert scheme == {
        'sub-01_dir-AP': 'sub-01_acq-solo_dir-AP',
        'sub-01': 'sub-01_acq-pair',
    }
    # The final name adds the curated label and differs from the unit key, so
    # naming derivatives from it (not the key) is what preserves the label.
    for unit_key, final_name in scheme.items():
        assert final_name.startswith('sub-01_acq-')
        assert final_name != unit_key


def test_partial_pair_routes_pair_and_singleton(tmp_path):
    """TORTOISE routes each blip group on its own: the matched pair to DRBUDDI,
    the unmatched singleton to the fieldmap-less fallback (estimation dropped, so
    the workflow does T2Wreg with a T2w). Both concatenate into one output.
    """
    grouping = load_scenario('partial_pair', tmp_path, strict=False)

    units = {unit.output_name: unit for unit in to_preproc_units(grouping, backend='tortoise')}
    assert set(units) == {'sub-01_acq-fast', 'sub-01_acq-slow_dir-AP'}

    pair = units['sub-01_acq-fast']
    assert pair.is_pepolar  # DRBUDDI pair
    assert pair.has_bidirectional_dwi

    singleton = units['sub-01_acq-slow_dir-AP']
    assert singleton.estimation is None  # fieldmap-less -> T2Wreg / HMC-only
    assert not singleton.has_scanner_measured_fieldmap

    assert set(concatenation_scheme(grouping, backend='tortoise').values()) == {'sub-01'}
    # FSL keeps the single pooled unit.
    assert [unit.output_name for unit in to_preproc_units(grouping, backend='fsl')] == ['sub-01']


def test_relpaths_curation_outranks_reverse_pe(tmp_path):
    """Each DWI has a dedicated PEPOLAR fieldmap and a reverse-PE sibling.

    The model honors the curated fieldmap linkage, correcting each series with
    its own epi fieldmap in a separate output rather than pairing the DWI
    series against each other into one merged output.
    """
    layout, subject_data = _load_skeleton('skeleton_complex_relpaths', tmp_path)
    grouping = build_dwi_grouping(layout, subject_data, strict=False)

    units = {unit.output_name: unit for unit in to_preproc_units(grouping, 'fsl')}
    assert set(units) == {'sub-01_dir-AP', 'sub-01_dir-PA'}

    ap = units['sub-01_dir-AP']
    assert ap.pepolar_fieldmap_type == 'epi'
    assert _basenames(list(ap.extra_b0)) == ['sub-01_dir-PA_epi.nii.gz']
    assert _basenames(list(ap.dwi_files)) == [
        'sub-01_dir-AP_run-1_dwi.nii.gz',
        'sub-01_dir-AP_run-2_dwi.nii.gz',
    ]
    pa = units['sub-01_dir-PA']
    assert _basenames(list(pa.extra_b0)) == ['sub-01_dir-AP_epi.nii.gz']


def test_identity_concatenation_scheme(tmp_path):
    scheme = concatenation_scheme(load_scenario('mixed_trt', tmp_path, strict=False))
    assert scheme == {name: name for name in scheme}


def test_pepolar_rpe_series_shape(tmp_path):
    """Opposite-PE DWI series correcting each other are one bidirectional unit."""
    (unit,) = _units('mixed_trt', tmp_path)
    assert unit.pepolar_fieldmap_type == 'rpe_series'
    assert _basenames(list(unit.plus_files)) == ['sub-01_dir-PA_dwi.nii.gz']
    assert _basenames(list(unit.minus_files)) == ['sub-01_dir-AP_dwi.nii.gz']
    assert unit.pe_dir == 'j'
    assert unit.extra_b0 == ()


def test_pepolar_epi_shape(tmp_path):
    """A DWI with a dedicated epi fieldmap is a single-polarity unit + extra b=0."""
    (unit,) = _units('abcd_style', tmp_path)
    assert unit.pepolar_fieldmap_type == 'epi'
    assert not unit.has_bidirectional_dwi
    assert _basenames(list(unit.extra_b0)) == ['sub-01_dir-PA_epi.nii.gz']
    assert _basenames(list(unit.dwi_files)) == ['sub-01_dir-AP_dwi.nii.gz']
    assert unit.pe_dir == 'j-'


def test_borrowed_series_become_extra_b0(tmp_path):
    """DWI series borrowed from another output ride in as extra b=0 sources."""
    units = {unit.output_name: unit for unit in _units('multipart_splits_estimation', tmp_path)}
    unit = units['sub-01_run-1']
    assert unit.has_bidirectional_dwi
    # The run-2 series live in a different output; they only lend b=0 images.
    assert _basenames(list(unit.extra_b0)) == [
        'sub-01_dir-AP_run-2_dwi.nii.gz',
        'sub-01_dir-PA_run-2_dwi.nii.gz',
    ]


def test_gre_phasediff_shape(tmp_path):
    """A phasediff fieldmap keys its files by suffix and carries its metadata."""
    (unit,) = _units('gre_phasediff', tmp_path)
    assert unit.method is CorrectionMethod.PHASEDIFF
    assert unit.gre_suffix == 'phasediff'
    gre = {suffix: op.basename(path) for suffix, path in unit.gre_files().items()}
    assert gre['phasediff'] == 'sub-01_phasediff.nii.gz'
    assert gre['magnitude1'] == 'sub-01_magnitude1.nii.gz'
    assert gre['magnitude2'] == 'sub-01_magnitude2.nii.gz'
    assert isinstance(unit.metadata_for(unit.gre_files()['phasediff']), dict)


def test_t2wreg_is_fieldmapless(tmp_path):
    """T2Wreg is fieldmap-less; SDC is driven by the t2wreg plan stage elsewhere."""
    (unit,) = _units('fieldmapless_t2w', tmp_path)
    assert unit.using_t2w_for_sdc
    assert not unit.has_scanner_measured_fieldmap
    assert unit.dwi_files


def test_uncorrected_units_have_no_method(tmp_path):
    """A series with no fieldmap has no estimation and no method."""
    units = _units('missing_pedir', tmp_path)
    assert units
    assert all(unit.estimation is None and unit.method is None for unit in units)


def test_syn_unit_shape(tmp_path):
    """SyN units expose is_nipreps_syn and no scanner-measured fieldmap."""
    (unit,) = _units('fieldmapless_t1w_only', tmp_path, use_nipreps_syn_sdc=True)
    assert unit.is_nipreps_syn
    assert not unit.has_scanner_measured_fieldmap


def test_synb0_unit_shape(tmp_path):
    """SyNb0 units carry their method; the synthesis workflow arrives later."""
    (unit,) = _units('fieldmapless_t1w_only', tmp_path, use_synb0=True)
    assert unit.method is CorrectionMethod.SYNB0
    assert not unit.has_scanner_measured_fieldmap


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
    grouping = load_scenario('two_gre_fmaps', tmp_path, strict=False)
    scheme = concatenation_scheme(grouping)
    assert scheme == {
        'sub-01_dir-AP_run-1': 'sub-01_dir-AP',
        'sub-01_dir-AP_run-2': 'sub-01_dir-AP',
    }
    assert {unit.output_name for unit in to_preproc_units(grouping)} == set(scheme)


def test_concatenation_scheme_identity_for_single_unit(tmp_path):
    """A single-unit output maps to itself."""
    scheme = concatenation_scheme(load_scenario('hcp_style', tmp_path, strict=False))
    assert scheme == {'sub-01': 'sub-01'}
