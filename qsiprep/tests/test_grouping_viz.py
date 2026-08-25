"""Structural tests for the explanatory HTML grouping page.

These assert on the content the renderer must contain (every estimation,
output, and issue present; one row per scan) rather than on exact markup, so
they survive cosmetic changes.
"""

from __future__ import annotations

import pytest

from qsiprep.grouping import render_html
from qsiprep.tests.grouping_scenarios import SCENARIOS, load_scenario


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_render_html_covers_the_grouping(tmp_path, scenario):
    """Every estimation, output, distortion group, and issue is on the page."""
    grouping = load_scenario(scenario, tmp_path, strict=False)
    page = render_html(grouping)
    assert page.startswith('<!doctype html>')

    for eid in grouping.estimations:
        assert eid in page
    for concat in grouping.concatenation_groups.values():
        assert concat.output_name in page
    for key in grouping.distortion_groups:
        assert key in page
    for issue in grouping.issues:
        assert issue.code in page


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_render_html_shows_each_scan_once_per_output(tmp_path, scenario):
    """A scan appears as one row per output it belongs to: exactly once in the
    common case, and once per virtual acquisition when a MultipartID list
    places it in several outputs. Estimation membership is a chip, not a
    second listing of the filename."""
    grouping = load_scenario(scenario, tmp_path, strict=False)
    page = render_html(grouping)
    # The page renders one scan row per distortion-group membership; every
    # distortion group belongs to exactly one output, so this equals the total
    # of each output's series count.
    expected = sum(len(dgroup.dwi_files) for dgroup in grouping.distortion_groups.values())
    assert page.count('class="scan"') == expected


def test_grouping_display_has_interactive_plan_controls(tmp_path):
    """The page ends with the interactive processing-plan display: CLI-flag
    dropdowns over every reachable method combination, an embedded payload
    index keyed by combination_key, and per-key prose. The concatenated
    sampling scheme stays shown directly."""
    import json
    import re

    from qsiprep.grouping.methods import reachable_selections

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    page = render_html(grouping)
    assert 'class="plan-interactive"' in page
    for control in ('ctl-hmc', 'ctl-model', 'ctl-sdc'):
        assert f'class="{control}"' in page
    # One re-render host; the payload index covers every reachable selection.
    assert page.count('class="pipeline-viewer plan-host"') == 1
    payload_json = re.search(
        r'<script type="application/json" class="plan-payloads">(.*?)</script>',
        page,
        re.S,
    ).group(1)
    index = json.loads(payload_json.replace('<\\/', '</'))
    keys = {selection.combination_key() for selection in reachable_selections()}
    assert set(index) == keys
    assert page.count('class="plan-prose"') == len(keys)
    # The eddy prose narrates estimation-before-HMC; SHORELine is itself.
    assert 'TOPUP estimates' in page
    assert 'SHORELine iteratively corrects head motion' in page
    # The sampling scheme is still shown, directly.
    assert 'scheme-block' in page
    assert 'qspace-viewer' in page


def test_report_segment_shows_only_the_selected_plan(tmp_path):
    """The subject report draws the single plan that actually ran: no tabs."""
    from qsiprep.grouping import render_report_segment
    from qsiprep.grouping.methods import selection_for_config

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    segment = render_report_segment(grouping, selection_for_config('eddy', 'topup'))
    assert 'how the data was corrected' in segment
    assert 'class="plan-tabs"' not in segment
    assert segment.count('class="pipeline-viewer"') == 1
    assert 'eddy + TOPUP' in segment
    # Without a selection the plan section is simply absent (legacy callers).
    bare = render_report_segment(grouping)
    assert 'class="pipeline-viewer"' not in bare


def test_report_is_past_tense_and_plan_is_future_tense(tmp_path):
    """The report describes a run that happened (past tense); the CLI plan page
    previews one that has not yet (future tense). Same markup, different wording."""
    from qsiprep.grouping import render_report_segment

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    plan = render_html(grouping)
    report = render_report_segment(grouping)
    assert 'How QSIPrep will process' in plan
    assert 'how distortion will be measured' in plan
    assert 'How QSIPrep processed' in report
    assert 'how distortion was measured' in report
    assert 'were combined, and how each was corrected' in report
    assert 'will process' not in report


def test_render_html_marks_borrowed_sources(tmp_path):
    grouping = load_scenario('multipart_splits_estimation', tmp_path, strict=False)
    page = render_html(grouping)
    assert page.count('class="borrow"') == sum(
        len(grouping.borrowed_sources(multipart_id))
        for multipart_id in grouping.concatenation_groups
    )


def test_render_html_is_self_contained(tmp_path):
    """No external assets: the page must not reference remote URLs."""
    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    page = render_html(grouping)
    assert 'src="http' not in page
    assert 'href="http' not in page


def test_render_report_segment_inlines_scoped_and_self_contained(tmp_path):
    """The report fragment inlines natively (no iframe, no document wrapper),
    carries its own assets, and scopes every style rule under the container so
    it neither restyles the host report nor leaks in."""
    import re

    from qsiprep.grouping import render_report_segment
    from qsiprep.grouping.interactive import ROOT_CLASS

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    segment = render_report_segment(grouping)

    # Inlines directly: no iframe and no full-document wrapper.
    assert '<iframe' not in segment
    lowered = segment.lower()
    assert '<!doctype' not in lowered
    assert '<body' not in lowered
    # One scoping container holds the whole widget.
    assert f'<div class="{ROOT_CLASS}">' in segment
    # Self-contained: no external assets.
    assert 'src="http' not in segment
    assert 'href="http' not in segment

    # Every grouping rule is scoped: no bare body/h1/h2/.badge selector that
    # would reach out into (or be reached by) the report's Bootstrap CSS.
    style = re.search(r'<style>(.*?)</style>', segment, re.S).group(1)
    grouping_css = style.split('.qspace-viewer', 1)[0]  # drop the shared viewer CSS
    for bare in ('\nbody{', '\nh1{', '\nh2{', '\n.badge{', '\n.output{', '\n*{'):
        assert bare not in grouping_css, bare
    assert f'.{ROOT_CLASS} .badge{{' in grouping_css
    assert f'.{ROOT_CLASS} .output{{' in grouping_css


def test_render_html_wraps_the_report_segment(tmp_path):
    """The standalone page is the inline fragment inside a minimal page shell."""
    from qsiprep.grouping.interactive import ROOT_CLASS

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    page = render_html(grouping)
    assert page.startswith('<!doctype html>')
    assert f'<div class="{ROOT_CLASS}">' in page


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_render_html_shows_correction_units(tmp_path, scenario):
    """Multi-unit outputs render an explicit unit tier and a final-combine
    note; single-unit outputs stay uncluttered."""
    grouping = load_scenario(scenario, tmp_path, strict=False)
    page = render_html(grouping)
    multi_unit_outputs = [
        concat
        for concat in grouping.concatenation_groups.values()
        if len(concat.correction_units) > 1
    ]
    expected_units = sum(len(c.correction_units) for c in multi_unit_outputs)
    assert page.count('class="cunit"') == expected_units
    assert page.count('class="final-concat"') == len(multi_unit_outputs)


def _write_series(directory, stem, bvals, bvecs, pe):
    """Write a touch-only nii plus its .bval/.bvec; return (path, record)."""
    import types

    nii = directory / f'{stem}.nii.gz'
    nii.touch()
    (directory / f'{stem}.bval').write_text(' '.join(str(b) for b in bvals))
    (directory / f'{stem}.bvec').write_text(
        '\n'.join(' '.join(str(v) for v in row) for row in bvecs)
    )
    record = types.SimpleNamespace(signature=types.SimpleNamespace(pe_dir=pe))
    return str(nii), record


def test_scheme_tab_builds_viewer_from_gradients(tmp_path):
    """The sampling-scheme tab embeds a viewer payload built from the sibling
    .bval/.bvec, one point per volume tagged by source file and PE direction,
    with the b0 threshold carried through."""
    import json
    import types

    from qsiprep.grouping import interactive

    bvals = [0, 1000, 1000, 2000]
    # x, y, z rows over four volumes (b=0 has no direction). Non-square so the
    # reader can orient it unambiguously.
    bvecs = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ap, ap_rec = _write_series(tmp_path, 'sub-01_dir-AP_dwi', bvals, bvecs, 'j-')
    pa, pa_rec = _write_series(tmp_path, 'sub-01_dir-PA_dwi', bvals, bvecs, 'j')
    grouping = types.SimpleNamespace(files={ap: ap_rec, pa: pa_rec})
    concat = types.SimpleNamespace(dwi_files=[ap, pa], output_name='sub-01_desc-preproc_dwi')

    data = interactive._scheme_data(grouping, concat)
    assert data['files'] == [
        'sub-01_dir-AP_dwi.nii.gz',
        'sub-01_dir-PA_dwi.nii.gz',
    ]
    assert data['pes'] == ['j-', 'j']
    assert len(data['meta']) == 8  # four volumes per series
    assert data['b0Threshold'] == 100.0
    # AP and PA sample the same directions, so their points coincide: one shared
    # origin (the two b=0s) plus the three shared directions = four coordinates.
    coords = data['panels'][0]['coords']
    distinct = {tuple(round(value, 6) for value in point) for point in coords}
    assert len(distinct) == 4

    view = interactive._scheme_view(grouping, concat)
    assert view.startswith('<div class="qspace-viewer">')
    payload = view.split('application/json">', 1)[1].split('</script>', 1)[0]
    assert json.loads(payload)['files'] == data['files']


def test_scheme_tab_degrades_without_gradients(tmp_path):
    """A DWI with no readable .bval/.bvec yields a notice, not an error."""
    import types

    from qsiprep.grouping import interactive

    nii = tmp_path / 'sub-01_dwi.nii.gz'
    nii.touch()
    record = types.SimpleNamespace(signature=types.SimpleNamespace(pe_dir='j'))
    grouping = types.SimpleNamespace(files={str(nii): record})
    concat = types.SimpleNamespace(dwi_files=[str(nii)], output_name='sub-01_desc-preproc_dwi')
    assert interactive._scheme_data(grouping, concat) is None
    assert 'scheme-missing' in interactive._scheme_view(grouping, concat)


def test_pipeline_payload_structure(tmp_path):
    """The diagram payload carries runs, ordered stages, outputs and issues."""
    from qsiprep.grouping.methods import selection_for_config
    from qsiprep.grouping.plan import compile_plan
    from qsiprep.viz.pipeline import plan_payload

    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    plan = compile_plan(grouping, selection_for_config('eddy', 'topup+drbuddi'))
    payload = plan_payload(grouping, plan)

    assert payload['schemaVersion'] == 1
    assert payload['selection']['label'] == 'eddy + TOPUP→DRBUDDI'
    (run,) = payload['runs']
    assert [stage['role'] for stage in run['stages']] == [
        'estimate',
        'hmc-with-field',
        'refine',
    ]
    assert run['stages'][1]['consumes'] == 0
    assert run['stages'][0]['letter'] == 'A'
    assert run['stages'][0]['text'].startswith('TOPUP estimates')
    (output,) = payload['outputs']
    assert output['runs'] == [run['key']]


def test_pipeline_div_escapes_script_closer():
    from qsiprep.viz.pipeline import pipeline_div

    markup = pipeline_div({'evil': 'a</script><b>'})
    assert '</script><b>' not in markup.replace('<\\/script>', '')
    assert 'class="pipeline-viewer"' in markup
