"""Golden tests for the grouping report and per-backend previews.

Every scenario's full report (grouping + fsl/tortoise/mixed previews) is
frozen as a text file under ``qsiprep/tests/data/grouping_reports/``. The
golden files double as documentation: they are the exact text a user sees.

To regenerate after an intentional behavior change::

    QSIPREP_REGEN_GROUPING_REPORTS=1 pytest qsiprep/tests/test_grouping_report.py

then review the diffs like any other code change.
"""

import os
import os.path as op

import pytest

from qsiprep.grouping import full_report
from qsiprep.tests.grouping_scenarios import SCENARIOS, load_scenario
from qsiprep.tests.utils import get_test_data_path

GOLDEN_DIR = op.join(get_test_data_path(), 'grouping_reports')
REGEN = bool(os.getenv('QSIPREP_REGEN_GROUPING_REPORTS'))

#: (scenario, kwargs, golden file stem) - scenarios whose flags matter get
#: extra entries.
CASES = [(scenario, {}, scenario) for scenario in SCENARIOS] + [
    ('hcp_style', {'separate_all_dwis': True}, 'hcp_style_separate_all_dwis'),
    ('reshim', {'ignore_shims': True}, 'reshim_ignore_shims'),
    ('abcd_style', {'ignore_fieldmaps': True}, 'abcd_style_ignore_fieldmaps'),
]


@pytest.mark.parametrize(('scenario', 'kwargs', 'stem'), CASES)
def test_full_report_matches_golden(tmp_path, scenario, kwargs, stem):
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    report = full_report(grouping)

    golden_path = op.join(GOLDEN_DIR, f'{stem}.txt')
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, 'w') as fobj:
            fobj.write(report)
        pytest.skip('regenerated golden file')

    assert op.exists(golden_path), (
        f'Missing golden report {golden_path}. Run with '
        'QSIPREP_REGEN_GROUPING_REPORTS=1 to create it, then review it.'
    )
    with open(golden_path) as fobj:
        expected = fobj.read()
    assert report == expected
