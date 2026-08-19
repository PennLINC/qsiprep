"""Report tests: a small curated golden set plus universal invariants.

Structured tests (test_grouping_inference.py) own the grouping *logic*.
Here, golden files freeze the exact *prose* - the preview text is the
product, and the golden files double as documentation - but only for a
curated set of scenarios chosen to cover every narration branch once:

=============================================== ============================
Narration branch                                Golden stem
=============================================== ============================
inferred PEPOLAR on all three backends          hcp_style
IntendedFor epi fmap + fmap-only b=0 extras     abcd_style
GRE phasediff + curated-boundary outputs        two_gre_fmaps
inferred T2Wreg (anat-sdc-unsupported on fsl)   fieldmapless_t2w
fieldmap-less SyN (--use-syn-sdc)               fieldmapless_t1w_only_syn
SyNb0 (--use-synb0)                             fieldmapless_t1w_only_synb0
SyNb0 overriding a real T2w structural target   t2w_hcp_synb0
no PE information, no SDC at all                missing_pedir
partial curation: curation disables inference   partial_curation
borrowing + estimation-spans-outputs            multipart_splits_estimation
cross-axis pooling (DRBUDDI infeasible)         cross_axis_unpaired
cross-axis split (per-axis DRBUDDI, recombined) cross_axis_b0field
multi-readout split (per-readout DRBUDDI)       multi_readout
matched pair + singleton (DRBUDDI + T2Wreg)     partial_pair
shelled/non-shelled tags + mixture warning      shell_mix
eddy-requires-shelled error                     nonshelled_pair
=============================================== ============================

Known-uncovered branch: the fsl "N separate fieldmap estimations feed this
output" note, reachable only through a curated MultipartID spanning multiple
curated estimations; add a scenario here if that wording starts to matter.

Every scenario (and every flag variant), golden or not, runs through
:func:`test_report_invariants`: rendering never crashes, every output,
series, estimation, and issue is mentioned, and step numbering is
contiguous. Invariants never need regeneration.

To regenerate the golden files after an intentional behavior change::

    QSIPREP_REGEN_GROUPING_REPORTS=1 pytest qsiprep/tests/test_grouping_report.py

then review the diffs like any other code change.
"""

import os
import os.path as op
import re
from collections import Counter

import pytest

from qsiprep.grouping import full_report
from qsiprep.tests.grouping_scenarios import SCENARIOS, load_scenario
from qsiprep.tests.utils import get_test_data_path

GOLDEN_DIR = op.join(get_test_data_path(), 'grouping_reports')
REGEN = bool(os.getenv('QSIPREP_REGEN_GROUPING_REPORTS'))

#: (scenario, kwargs, golden file stem) - the curated branch-coverage set.
GOLDEN_CASES = [
    ('hcp_style', {}, 'hcp_style'),
    ('abcd_style', {}, 'abcd_style'),
    ('two_gre_fmaps', {}, 'two_gre_fmaps'),
    ('fieldmapless_t2w', {}, 'fieldmapless_t2w'),
    ('fieldmapless_t1w_only', {'use_nipreps_syn_sdc': True}, 'fieldmapless_t1w_only_syn'),
    ('fieldmapless_t1w_only', {'use_synb0': True}, 'fieldmapless_t1w_only_synb0'),
    ('t2w_hcp', {'use_synb0': True}, 't2w_hcp_synb0'),
    ('missing_pedir', {}, 'missing_pedir'),
    ('partial_curation', {}, 'partial_curation'),
    ('multipart_splits_estimation', {}, 'multipart_splits_estimation'),
    ('cross_axis_unpaired', {}, 'cross_axis_unpaired'),
    ('cross_axis_b0field', {}, 'cross_axis_b0field'),
    ('multi_readout', {}, 'multi_readout'),
    ('partial_pair', {}, 'partial_pair'),
    ('shell_mix', {}, 'shell_mix'),
    ('nonshelled_pair', {}, 'nonshelled_pair'),
]

#: Flag variants with no golden file: invariant coverage only.
FLAG_VARIANTS = [
    ('hcp_style', {'separate_all_dwis': True}),
    ('reshim', {'ignore_shims': True}),
    ('abcd_style', {'ignore_fieldmaps': True}),
    ('t2w_hcp', {'force_t2wreg': True}),
    ('fieldmapless_t1w_only', {'use_synb0': True}),
    ('fieldmapless_t1w_only', {'use_nipreps_syn_sdc': True}),
    ('missing_pedir', {'use_synb0': True}),
    ('fov_oblique', {'ignore_fov': True}),
    ('t2w_hcp', {'use_synb0': True}),
]

INVARIANT_CASES = [pytest.param(scenario, {}, id=scenario) for scenario in SCENARIOS] + [
    pytest.param(scenario, kwargs, id=scenario + '+' + '+'.join(sorted(kwargs)))
    for scenario, kwargs in FLAG_VARIANTS
]


@pytest.mark.parametrize(('scenario', 'kwargs', 'stem'), GOLDEN_CASES)
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


@pytest.mark.parametrize(('scenario', 'kwargs'), INVARIANT_CASES)
def test_report_invariants(tmp_path, scenario, kwargs):
    """Rendering properties every scenario must satisfy, golden or not."""
    grouping = load_scenario(scenario, tmp_path, strict=False, **kwargs)
    report = full_report(grouping)

    # The grouping section plus all three backend previews are present.
    assert report.count('Processing preview:') == 3

    # Every output appears once in the grouping section and once per backend.
    name_counts = Counter(concat.output_name for concat in grouping.concatenation_groups.values())
    for output_name, count in name_counts.items():
        assert report.count(f'Output "{output_name}"') == 4 * count

    # Every series, estimation, and issue is mentioned somewhere.
    for path in grouping.dwi_files:
        assert op.basename(path) in report
    for b0field_id in grouping.estimations:
        assert b0field_id in report
    for issue in grouping.issues:
        assert issue.code in report

    # Preview step numbering restarts at 1 for each output and is contiguous.
    step = None
    for line in report.splitlines():
        if line.startswith('Output "'):
            step = 0
            continue
        numbered = re.match(r'  (\d+)\. ', line)
        if numbered and step is not None:
            step += 1
            assert int(numbered.group(1)) == step, f'non-contiguous step: {line!r}'
