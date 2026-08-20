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


def test_render_html_previews_every_backend(tmp_path):
    grouping = load_scenario('hcp_style', tmp_path, strict=False)
    page = render_html(grouping, backend='tortoise')
    # The chosen backend is the expanded preview; the others are alternates.
    assert '(<b>tortoise</b> workflow)' in page
    assert 'if run with the fsl workflow instead' in page
    assert 'if run with the mixed workflow instead' in page
    assert 'DRBUDDI' in page
    assert 'TOPUP' in page


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
