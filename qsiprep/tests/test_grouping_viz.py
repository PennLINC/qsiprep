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
    # The sampling scheme is the default tab; each backend has its own preview
    # tab, and the one selected on the command line is flagged.
    assert 'data-tab="scheme"' in page
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert f'data-tab="{backend}"' in page
    assert 'class="tab-btn sel" data-tab="tortoise"' in page
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

    tab = interactive._scheme_tab(grouping, concat)
    assert tab.startswith('<div class="qspace-viewer">')
    payload = tab.split('application/json">', 1)[1].split('</script>', 1)[0]
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
    assert 'scheme-missing' in interactive._scheme_tab(grouping, concat)
