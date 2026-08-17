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
def test_render_html_shows_each_scan_exactly_once(tmp_path, scenario):
    """Scans appear as one row each; estimation membership is a chip, not a
    second listing of the filename."""
    grouping = load_scenario(scenario, tmp_path, strict=False)
    page = render_html(grouping)
    assert page.count('class="scan"') == len(grouping.dwi_files)


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
