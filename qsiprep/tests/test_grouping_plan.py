"""Parity harness for the execution-plan compiler.

The compiler must reproduce the legacy routing byte-for-byte before anything
consumes it: run keys, file sets and estimations against
``to_preproc_units``; assemblies against ``concatenation_scheme``; issues
against ``check_backend`` (order and text). Every scenario is checked under
every canonical selection, plus the flag variants the golden reports cover.
Stage shapes - the ordered estimate/HMC/refine sequences - are pinned
separately for each method family.
"""

import itertools
import json

import pytest

from qsiprep.grouping import check_backend, concatenation_scheme, to_preproc_units
from qsiprep.grouping.integrity import check_plan
from qsiprep.grouping.methods import canonical_selection, selection_for_config
from qsiprep.grouping.models import CorrectionMethod
from qsiprep.grouping.plan import ExecutionPlan, StageRole, compile_plan
from qsiprep.grouping.validation import BACKENDS
from qsiprep.tests.grouping_scenarios import SCENARIOS, load_scenario

#: (scenario, build kwargs) - every scenario plain, plus the golden flag variants.
CASES = [(scenario, {}) for scenario in SCENARIOS] + [
    ('fieldmapless_t1w_only', {'use_nipreps_syn_sdc': True}),
    ('fieldmapless_t1w_only', {'use_synb0': True}),
    ('t2w_hcp', {'use_synb0': True}),
    ('curated_t2wreg', {'force_t2wreg': True}),
]


def _case_id(case):
    scenario, kwargs = case
    suffix = '+'.join(sorted(kwargs)) if kwargs else 'plain'
    return f'{scenario}-{suffix}'


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_plan_matches_legacy_routing(tmp_path, case, backend):
    scenario, kwargs = case
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    selection = canonical_selection(backend)
    plan = compile_plan(grouping, selection)

    # Runs <-> PreprocUnits: same keys in the same order, same files, same
    # (possibly pair-restricted) estimations.
    units = to_preproc_units(grouping, backend)
    assert [run.key for run in plan.runs] == [unit.output_name for unit in units]
    for run, unit in zip(plan.runs, units, strict=True):
        assert run.dwi_files == unit.dwi_files
        assert run.estimation == unit.estimation

    # Issues <-> check_backend: order and text.
    assert list(plan.issues) == check_backend(grouping, backend)

    # Assemblies <-> concatenation_scheme, inverted.
    scheme = concatenation_scheme(grouping, backend)
    from_plan = {
        run_key: assembly.output_name
        for assembly in plan.outputs
        for run_key in assembly.input_runs
    }
    assert from_plan == scheme

    # The compiler kept its own invariants.
    assert check_plan(grouping, plan) == []

    # Serialization is complete and stable.
    payload = json.dumps(plan.to_dict(), sort_keys=True)
    assert json.dumps(plan.to_dict(), sort_keys=True) == payload


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_stage_order_per_method_family(tmp_path, case):
    """eddy estimates before HMC (TOPUP integrated); the others correct first."""
    scenario, kwargs = case
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    selections = [canonical_selection(backend) for backend in BACKENDS] + [
        selection_for_config('shoreline', 'drbuddi')
    ]
    for selection in selections:
        plan = compile_plan(grouping, selection)
        for run in plan.runs:
            roles = [stage.role for stage in run.stages]
            assert roles, f'{run.key} has no stages'
            hmc_roles = {StageRole.HMC, StageRole.HMC_WITH_FIELD}
            assert sum(role in hmc_roles for role in roles) == 1
            if run.estimation is not None and run.estimation.is_pepolar:
                topup = run.stage_with('topup')
                if topup is not None:
                    # The integrated path: the field is estimated first and
                    # eddy consumes it during motion correction.
                    assert topup.role is StageRole.ESTIMATE
                    eddy = run.stage_with('eddy')
                    assert eddy.role is StageRole.HMC_WITH_FIELD
                    assert eddy.consumes == topup.index
                elif run.stage_with('drbuddi') is not None:
                    # DRBUDDI always estimates after motion correction.
                    assert roles[0] in hmc_roles


def _plan_for(tmp_path, scenario, hmc, sdc, **kwargs):
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    return compile_plan(grouping, selection_for_config(hmc, sdc))


def _role_tool(run):
    return [(stage.role.value, stage.tool) for stage in run.stages]


def test_eddy_topup_sequence(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'eddy', 'topup')
    (run,) = plan.runs
    assert _role_tool(run) == [('estimate', 'topup'), ('hmc-with-field', 'eddy')]
    assert run.stages[1].consumes == 0


def test_eddy_topup_drbuddi_sequence(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'eddy', 'topup+drbuddi')
    (run,) = plan.runs
    assert _role_tool(run) == [
        ('estimate', 'topup'),
        ('hmc-with-field', 'eddy'),
        ('refine', 'drbuddi'),
    ]


def test_eddy_drbuddi_only_corrects_after_hmc(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'eddy', 'drbuddi')
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'eddy'), ('estimate+apply', 'drbuddi')]


def test_diffprep_drbuddi_sequence(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'diffprep', 'drbuddi')
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'diffprep'), ('estimate+apply', 'drbuddi')]


def test_shoreline_is_named_not_diffprep(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'shoreline', 'drbuddi')
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'shoreline'), ('estimate+apply', 'drbuddi')]


def test_gre_fieldmap_estimates_after_hmc(tmp_path):
    plan = _plan_for(tmp_path, 'gre_phasediff', 'eddy', 'topup')
    stages = {run.key: _role_tool(run) for run in plan.runs}
    assert all(
        seq == [('hmc', 'eddy'), ('estimate+apply', 'fieldmap')] for seq in stages.values()
    ), stages


def test_diffprep_t2wreg_fallback_stage(tmp_path):
    plan = _plan_for(tmp_path, 'fieldmapless_t2w', 'diffprep', 'drbuddi')
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'diffprep'), ('estimate+apply', 't2wreg')]
    assert run.stages[1].structural_target == 't2w'


def test_eddy_cannot_run_t2wreg(tmp_path):
    plan = _plan_for(tmp_path, 'fieldmapless_t2w', 'eddy', 'topup')
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'eddy')]
    assert any(issue.code == 'anat-sdc-unsupported' for issue in plan.issues)


def test_synb0_prefers_synthetic_target_on_diffprep(tmp_path):
    plan = _plan_for(tmp_path, 't2w_hcp', 'diffprep', 'drbuddi', use_synb0=True)
    targets = {
        stage.structural_target
        for run in plan.runs
        for stage in run.stages
        if stage.tool == 't2wreg'
    }
    assert targets in (set(), {'synb0'})


def test_syn_stage_targets_t1w(tmp_path):
    plan = _plan_for(
        tmp_path, 'fieldmapless_t1w_only', 'eddy', 'topup', use_nipreps_syn_sdc=True
    )
    (run,) = plan.runs
    assert _role_tool(run) == [('hmc', 'eddy'), ('estimate+apply', 'syn')]
    assert run.stages[1].structural_target == 't1w'
    assert run.stages[1].method is CorrectionMethod.NIPREPS_SYN


def test_decomposed_pairs_get_their_own_runs(tmp_path):
    """A multi-readout PEPOLAR unit splits per blip pair on decomposing methods."""
    grouping = load_scenario('multi_readout', tmp_path, strict=False)
    pooled = compile_plan(grouping, selection_for_config('eddy', 'topup'))
    split = compile_plan(grouping, selection_for_config('diffprep', 'drbuddi'))
    assert len(split.runs) > len(pooled.runs)
    assert {run.logical_unit for run in split.runs} == {
        run.logical_unit for run in pooled.runs
    }
    # Every complete pair corrects with DRBUDDI; each run's blip files stay
    # inside its own pair.
    for run in split.runs:
        drbuddi = run.stage_with('drbuddi')
        if drbuddi is not None:
            assert run.estimation is not None
            assert set(drbuddi.fieldmap_sources) == set(run.estimation.sources)


def test_plan_serialization_shape(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'eddy', 'topup+drbuddi')
    payload = plan.to_dict()
    assert payload['schema_version'] == 1
    assert payload['selection']['hmc'] == 'eddy'
    assert payload['selection']['pepolar_tools'] == ['topup', 'drbuddi']
    assert payload['selection']['label'] == 'eddy + TOPUP→DRBUDDI'
    (run,) = payload['runs']
    assert [stage['role'] for stage in run['stages']] == [
        'estimate',
        'hmc-with-field',
        'refine',
    ]
    assert isinstance(plan, ExecutionPlan)


def test_run_lookup_helpers(tmp_path):
    plan = _plan_for(tmp_path, 'hcp_style', 'eddy', 'topup')
    (run,) = plan.runs
    assert plan.run(run.key) is run
    assert plan.runs_for(run.output_group) == [run]
    with pytest.raises(KeyError):
        plan.run('nope')


def test_every_backend_selection_pair_is_reachable():
    """canonical_selection covers BACKENDS; product with CASES stays in sync."""
    assert {canonical_selection(b).legacy_backend for b in BACKENDS} == set(BACKENDS)
    assert len(CASES) == len(SCENARIOS) + 4
    assert len(list(itertools.product(CASES, BACKENDS))) == 3 * len(CASES)


def test_plan_step_records_follow_stage_order(tmp_path):
    from qsiprep.grouping.report import plan_step_records

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    plan = compile_plan(grouping, selection_for_config('eddy', 'topup+drbuddi'))
    (records,) = plan_step_records(grouping, plan).values()
    kinds = [record.kind for record in records]
    assert kinds == ['denoise', 'sdc-estimate', 'hmc', 'refine', 'assemble']
    tools = [record.tool for record in records if record.tool]
    assert tools == ['topup', 'eddy', 'drbuddi']
    assert 'TOPUP estimates' in records[1].text
    assert 'using the TOPUP field' in records[2].text


def test_plan_step_records_scope_issues_to_their_output(tmp_path):
    from qsiprep.grouping.report import plan_step_records

    grouping = load_scenario('nonshelled_pair', tmp_path, strict=False)
    plan = compile_plan(grouping, selection_for_config('eddy', 'topup'))
    (records,) = plan_step_records(grouping, plan).values()
    issues = [record for record in records if record.kind == 'issue']
    assert issues
    assert issues[0].severity == 'error'
    assert 'eddy-requires-shelled' in issues[0].text
