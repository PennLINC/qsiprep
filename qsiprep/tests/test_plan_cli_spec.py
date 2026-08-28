"""Contract: qsiprep's CLI conforms to qsiplan's shared plan-CLI spec.

qsiplan owns the plan-relevant option surface (:mod:`qsiplan.cli_spec`) - the
real flag spellings, their choices, and the policy/selection field each drives.
qsiprep builds its own (grouped, deprecation-aware) parser by hand, so this test
asserts that parser stays conformant: a flag renamed or a choice added in
qsiplan turns this red until qsiprep catches up, instead of drifting silently -
as the config-to-selection bridge already did (see the regression test below).
"""

import argparse

from qsiplan import cli_spec

from qsiprep.cli.parser import _build_parser


def _actions():
    parser = _build_parser()
    return {flag: action for action in parser._actions for flag in action.option_strings}


def test_parser_realizes_every_implemented_plan_option():
    actions = _actions()
    for option in cli_spec.PLAN_OPTIONS:
        if option.planned:
            continue
        action = actions.get(option.flag)
        assert action is not None, f'qsiprep parser is missing {option.flag}'
        missing = set(option.owned_choices()) - set(action.choices or ())
        assert not missing, f'{option.flag}: qsiprep is missing choices {sorted(missing)}'


def test_planned_options_are_the_only_gaps():
    actions = _actions()
    absent = sorted(o.flag for o in cli_spec.PLAN_OPTIONS if o.flag not in actions)
    planned = sorted(o.flag for o in cli_spec.PLAN_OPTIONS if o.planned)
    assert absent == planned  # today: [] (--use-synb0 is wired)


def test_use_synb0_flag_reaches_the_grouping_policy(tmp_path):
    bids_dir = tmp_path / 'bids'
    bids_dir.mkdir()
    parser = _build_parser()
    base = [str(bids_dir), str(tmp_path / 'out'), 'participant', '--output-resolution', '2']
    assert not parser.parse_args(base).use_synb0
    namespace = parser.parse_args([*base, '--use-synb0'])
    assert namespace.use_synb0
    assert cli_spec.policy_from_namespace(namespace).use_synb0


def test_shared_helpers_read_a_qsiprep_namespace():
    # qsiprep's flag dests match the spec, so qsiplan's helpers work on a
    # qsiprep-parsed namespace unchanged: the drop-in replacement for the
    # hand-written config-to-objects wiring in workflows/base.py.
    namespace = argparse.Namespace(
        hmc_method='tortoise',
        shoreline_model=None,
        sdc_method='drbuddi',
        separate_all_dwis=False,
        ignore=['fieldmaps', 'shims'],
        force=[],
        use_syn_sdc=False,
        use_synb0=False,
        distortion_group_merge='concat',
    )
    assert cli_spec.selection_from_namespace(namespace).label() == 'TORTOISE + DRBUDDI'
    policy = cli_spec.policy_from_namespace(namespace)
    assert policy.ignore_fieldmaps
    assert policy.ignore_shims


def test_config_selection_bridge_resolves_without_error():
    # Regression: the config-to-selection bridge used to pass use_syn=/
    # force_t2wreg= kwargs that qsiplan's selection_for_config dropped, raising
    # TypeError. It now resolves cleanly to a MethodSelection.
    from qsiprep import config
    from qsiprep.utils.plan import method_selection_from_config

    saved = (config.workflow.hmc_method, config.workflow.sdc_method)
    try:
        config.workflow.hmc_method, config.workflow.sdc_method = 'eddy', 'topup'
        selection = method_selection_from_config()
    finally:
        config.workflow.hmc_method, config.workflow.sdc_method = saved
    assert selection.hmc.value == 'eddy'


def test_subject_plan_bridge_constructs_grouping_policy(monkeypatch):
    """Regression: the subject workflow passed an undefined ``policy`` name."""
    from qsiprep.workflows import base

    grouping = object()
    plan = object()
    received = {}

    def fake_grouping(**kwargs):
        received.update(kwargs)
        return grouping

    monkeypatch.setattr(base, 'build_dwi_grouping', fake_grouping)
    monkeypatch.setattr(base, 'compile_plan', lambda value, selection: plan)

    assert base._build_dwi_plan({'dwi': ['scan.nii.gz']}, object()) == (grouping, plan)
    assert received['subject_data'] == {'dwi': ['scan.nii.gz']}
    assert received['strict'] is False
    assert 'separate_all_dwis' in received
