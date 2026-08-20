"""Model-integrity checks over every scenario and flag variant, plus
corruption tests proving the checker detects broken models."""

import dataclasses

import pytest

from qsiprep.grouping.integrity import check_model_integrity
from qsiprep.tests.grouping_scenarios import SCENARIOS, load_scenario
from qsiprep.tests.test_grouping_report import FLAG_VARIANTS

CASES = [pytest.param(scenario, {}, id=scenario) for scenario in SCENARIOS] + [
    pytest.param(scenario, kwargs, id=scenario + '+' + '+'.join(sorted(kwargs)))
    for scenario, kwargs in FLAG_VARIANTS
]


@pytest.mark.parametrize(('scenario', 'kwargs'), CASES)
def test_model_integrity_holds(tmp_path, scenario, kwargs):
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    assert check_model_integrity(grouping) == []


def test_detects_cross_session_output(tmp_path):
    """A hand-corrupted output spanning sessions is flagged."""
    grouping = load_scenario('multi_session', tmp_path)
    (key_a, key_b) = sorted(grouping.concatenation_groups)
    merged = dataclasses.replace(
        grouping.concatenation_groups[key_a],
        correction_units=tuple(
            sorted(
                grouping.concatenation_groups[key_a].correction_units
                + grouping.concatenation_groups[key_b].correction_units
            )
        ),
    )
    corrupted = dataclasses.replace(
        grouping,
        concatenation_groups={key_a: merged, key_b: grouping.concatenation_groups[key_b]},
    )
    violations = check_model_integrity(corrupted)
    assert any('session' in violation for violation in violations)


def test_detects_membership_undercount(tmp_path):
    """Dropping an output of a list-MultipartID (virtual acquisition) series
    is flagged."""
    grouping = load_scenario('virtual_acq_multipart', tmp_path)
    trimmed = dict(grouping.concatenation_groups)
    removed = trimmed.pop(sorted(trimmed)[0])
    assert removed.key not in trimmed
    corrupted = dataclasses.replace(grouping, concatenation_groups=trimmed)
    violations = check_model_integrity(corrupted)
    assert any('appears in' in violation for violation in violations)
