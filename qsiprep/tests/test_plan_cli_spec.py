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
    assert absent == planned  # today: []


def _parse_minimal(tmp_path, *extra):
    bids_dir = tmp_path / 'bids'
    bids_dir.mkdir(exist_ok=True)
    base = [str(bids_dir), str(tmp_path / 'out'), 'participant', '--output-resolution', '2']
    return _build_parser().parse_args([*base, *extra])


def test_sdc_anat_reference_flag_reaches_the_grouping_policy(tmp_path):
    assert _parse_minimal(tmp_path).sdc_anat_reference == 'none'
    namespace = _parse_minimal(tmp_path, '--sdc-anat-reference', 'synb0')
    assert namespace.sdc_anat_reference == 'synb0'
    policy = cli_spec.policy_from_namespace(namespace)
    assert policy.sdc_anat_reference == 'synb0'
    assert policy.force_sdc_anat_reference is False


def test_force_sdc_anat_reference_reaches_the_grouping_policy(tmp_path):
    namespace = _parse_minimal(
        tmp_path, '--sdc-anat-reference', 't2w', '--force', 'sdc-anat-reference'
    )
    policy = cli_spec.policy_from_namespace(namespace)
    assert policy.sdc_anat_reference == 't2w'
    assert policy.force_sdc_anat_reference is True


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
        sdc_anat_reference='none',
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


def test_complex_dwi_reaches_the_plan_as_a_magnitude_companion(tmp_path):
    """Contract across the qsiprep<->qsiplan seam for complex-valued DWI.

    Exercises the real grouping/plan (no monkeypatching): qsiprep collects both
    parts (the ``part`` pre-filter is retired), qsiplan indexes only the
    magnitude and carries the phase as its companion, and the unit the workflow
    consumes exposes it through ``dwi_phase_files`` -- keyed by the magnitude
    path, the way ``init_merge_and_denoise_wf`` looks it up -- with no phase
    leaking into any series list or sidecar override.
    """
    import os.path as op

    from bids.layout import BIDSLayout
    from qsiplan import build_dwi_grouping
    from qsiplan.adapters import plan_preproc_units
    from qsiplan.methods import selection_for_config
    from qsiplan.plan import compile_plan

    from qsiprep.tests.utils import (
        COMPLEX_DWI_SKELETON,
        SHARED_DWI_GRADIENTS,
        build_test_dataset,
    )
    from qsiprep.utils.bids import collect_data

    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={
            **SHARED_DWI_GRADIENTS,
            'sub-01/dwi/sub-01_dwi.json': {
                'PhaseEncodingDirection': 'j-',
                'TotalReadoutTime': 0.05,
            },
        },
        n_volumes=2,
    )
    layout = BIDSLayout(root, validate=False)

    # B1: the ``part`` pre-filter is gone, so both parts reach qsiplan.
    subject_data = collect_data(layout, '01', bids_validate=False)[0]
    assert sorted(op.basename(f) for f in subject_data['dwi']) == [
        'sub-01_part-mag_dwi.nii.gz',
        'sub-01_part-phase_dwi.nii.gz',
    ]

    # qsiplan is the authoritative split: only the magnitude is a series, and
    # the phase rides on its record as a companion.
    grouping = build_dwi_grouping(layout, subject_data, strict=False)
    assert [op.basename(f) for f in grouping.dwi_files] == ['sub-01_part-mag_dwi.nii.gz']
    (magnitude,) = grouping.dwi_files
    assert op.basename(grouping.files[magnitude].phase_path) == 'sub-01_part-phase_dwi.nii.gz'

    # B2: the shape the workflow consumes -- phase reached only through
    # ``dwi_phase_files``, never as a member series or a sidecar override.
    plan = compile_plan(grouping, selection_for_config('eddy', 'topup'))
    (unit,) = plan_preproc_units(grouping, plan)
    assert unit.dwi_phase_files == {magnitude: grouping.files[magnitude].phase_path}
    series = (*unit.dwi_files, *unit.plus_files, *unit.minus_files)
    assert not [path for path in series if 'part-phase' in path]
    assert not [path for path in unit.sidecar_overrides() if 'part-phase' in path]
