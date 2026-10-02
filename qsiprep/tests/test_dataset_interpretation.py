"""How qsiprep reads a dataset: the grouping and plan, without running anything.

A real single-session dataset with a GRE fieldmap (forrest_gump) and the TRXScan fixtures
each have one right interpretation: which series form an output, which fieldmap estimation
corrects it, with what method and provenance, and which stages the plan runs. These tests
pin that in seconds; the end-to-end runs then only have to score what the stages produce.
"""

import os
from pathlib import Path

import pytest


def _plan(dataset_dir, subject, hmc='eddy', sdc='topup'):
    from bids.layout import BIDSLayout
    from qsiplan import build_dwi_grouping, report_text
    from qsiplan.adapters import plan_concatenation_scheme, plan_preproc_units
    from qsiplan.methods import selection_for_config
    from qsiplan.plan import compile_plan

    from qsiprep.utils.bids import collect_data

    layout = BIDSLayout(str(dataset_dir), validate=False)
    subject_data = collect_data(layout, subject, bids_validate=False)[0]
    grouping = build_dwi_grouping(
        layout=layout, subject_data=subject_data, b0_threshold=100, strict=False
    )
    plan = compile_plan(grouping, selection_for_config(hmc, sdc))
    units = plan_preproc_units(grouping, plan)
    return grouping, plan, units, plan_concatenation_scheme(plan), report_text(grouping)


def _stage_tools(unit):
    return [
        (stage.role.value, stage.tool, stage.method.value if stage.method else None)
        for stage in unit.run.stages
    ]


def _fixture(data_dir, name):
    from qsiprep.tests.trxscan_fixtures import RECIPES

    if not data_dir:
        pytest.skip('--data_dir was not provided')
    path = Path(data_dir) / 'trxscan' / name
    if not (path / 'dataset_description.json').exists():
        pytest.skip(f'TRXScan fixture {name} is not under {data_dir}')
    return path, RECIPES[name][1]['subject']


def test_forrest_gump_is_one_series_with_a_phasediff_fieldmap(data_dir):
    """Test the grouping of a real session: one DWI, PE j, corrected by its GRE phasediff."""
    if not data_dir:
        pytest.skip('--data_dir was not provided')
    dataset = Path(data_dir) / 'forrest_gump'
    if not (dataset / 'dataset_description.json').exists():
        pytest.skip('forrest_gump dataset is unavailable')
    grouping, plan, units, concat, text = _plan(dataset, '01')

    assert not grouping.errors
    assert not plan.issues
    assert [u.output_name for u in units] == ['sub-01_ses-forrestgump']
    (unit,) = units
    assert [os.path.basename(f) for f in unit.dwi_files] == ['sub-01_ses-forrestgump_dwi.nii.gz']
    assert unit.pe_dir == 'j'
    assert unit.is_gre
    assert not unit.is_nipreps_syn
    assert unit.estimation.method.value == 'phasediff'
    assert unit.estimation.provenance.value == 'intendedfor'
    assert sorted(os.path.basename(f) for f in unit.estimation.sources) == [
        'sub-01_ses-forrestgump_magnitude1.nii.gz',
        'sub-01_ses-forrestgump_phasediff.nii.gz',
    ]
    assert _stage_tools(unit) == [
        ('hmc', 'eddy', None),
        ('estimate+apply', 'fieldmap', 'phasediff'),
    ]
    assert concat == {'sub-01_ses-forrestgump': 'sub-01_ses-forrestgump'}
    assert 'PE j, TRT 0.0472s' in text
    assert 'GRE phase difference' in text


def test_rpe_fixture_is_a_reverse_pe_pair(data_dir):
    """Test the reverse-PE pair: two series, opposite polarities, one output through TOPUP."""
    dataset, subject = _fixture(data_dir, 'rpe')
    grouping, plan, units, concat, text = _plan(dataset, subject)

    assert not grouping.errors
    assert not plan.issues
    assert len(units) == 1
    (unit,) = units
    assert sorted(os.path.basename(f) for f in unit.dwi_files) == [
        f'sub-{subject}_dir-AP_dwi.nii.gz',
        f'sub-{subject}_dir-PA_dwi.nii.gz',
    ]
    assert unit.estimation.method.value == 'pepolar'
    tools = {stage[1] for stage in _stage_tools(unit)}
    assert {'topup', 'eddy'} <= tools, _stage_tools(unit)
    assert any(stage[2] == 'pepolar' for stage in _stage_tools(unit)), _stage_tools(unit)


def test_epi_fixture_uses_its_pepolar_fieldmap(data_dir):
    """Test one series plus an epi fieldmap under a shared B0FieldIdentifier."""
    dataset, subject = _fixture(data_dir, 'epi')
    grouping, plan, units, concat, text = _plan(dataset, subject)

    assert not grouping.errors
    assert not plan.issues
    (unit,) = units
    assert [os.path.basename(f) for f in unit.dwi_files] == [f'sub-{subject}_dir-AP_dwi.nii.gz']
    assert unit.estimation.method.value == 'pepolar'
    assert unit.estimation.provenance.value == 'curated'  # from B0FieldIdentifier/Source
    assert any(os.path.basename(f).endswith('_epi.nii.gz') for f in unit.estimation.sources)


def test_phasediff_fixture_is_a_gre_estimation(data_dir):
    """Test one series with a synthetic GRE phasediff fieldmap."""
    dataset, subject = _fixture(data_dir, 'phasediff')
    grouping, plan, units, concat, text = _plan(dataset, subject)

    assert not grouping.errors
    assert not plan.issues
    (unit,) = units
    assert unit.is_gre
    assert unit.estimation.method.value == 'phasediff'


def test_t2wreg_fixture_has_no_fieldmap_and_takes_t2wreg_under_tortoise(data_dir):
    """Test the fieldmap-less single series: no estimation with eddy, T2Wreg with DIFFPREP."""
    dataset, subject = _fixture(data_dir, 't2wreg')
    grouping, plan, units, concat, text = _plan(dataset, subject)
    assert not grouping.errors
    (unit,) = units
    assert [os.path.basename(f) for f in unit.dwi_files] == [f'sub-{subject}_dir-AP_dwi.nii.gz']
    assert unit.estimation is None or unit.estimation.method.value == 'none'

    grouping, plan, units, concat, text = _plan(dataset, subject, hmc='tortoise', sdc='drbuddi')
    (unit,) = units
    assert unit.run.stage_with('t2wreg') is not None
