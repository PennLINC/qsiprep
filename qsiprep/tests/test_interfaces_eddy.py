"""Tests for the qsiprep.interfaces.eddy module."""

import stat

import qsiprep.interfaces.eddy as eddy_mod
from qsiprep.interfaces.eddy import ExtendedEddy, _find_eddy_cuda


def _make_exe(path):
    """Create an executable stub file at ``path`` (a pathlib.Path)."""
    path.write_text('#!/bin/sh\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_find_eddy_cuda_single(tmp_path, monkeypatch):
    """A single eddy_cuda binary on PATH is returned as-is."""
    _make_exe(tmp_path / 'eddy_cuda11.0')
    monkeypatch.setenv('PATH', str(tmp_path))
    assert _find_eddy_cuda() == 'eddy_cuda11.0'


def test_find_eddy_cuda_multiple_picks_newest(tmp_path, monkeypatch):
    """With several binaries, the newest version wins and a warning is logged."""
    _make_exe(tmp_path / 'eddy_cuda10.2')
    _make_exe(tmp_path / 'eddy_cuda11.0')
    monkeypatch.setenv('PATH', str(tmp_path))

    warnings = []
    monkeypatch.setattr(eddy_mod.LOGGER, 'warning', lambda *a, **k: warnings.append(a))

    assert _find_eddy_cuda() == 'eddy_cuda11.0'
    assert warnings, 'expected a warning when multiple binaries are found'


def test_find_eddy_cuda_none_returns_default(tmp_path, monkeypatch):
    """With no binaries on PATH, the default name is returned and a warning logged."""
    monkeypatch.setenv('PATH', str(tmp_path))

    warnings = []
    monkeypatch.setattr(eddy_mod.LOGGER, 'warning', lambda *a, **k: warnings.append(a))

    assert _find_eddy_cuda() == 'eddy_cuda10.2'
    assert warnings, 'expected a warning when no binary is found'


def test_find_eddy_cuda_ignores_non_versioned(tmp_path, monkeypatch):
    """Non-versioned eddy binaries (eddy, eddy_cpu) are not matched."""
    _make_exe(tmp_path / 'eddy')
    _make_exe(tmp_path / 'eddy_cpu')
    monkeypatch.setenv('PATH', str(tmp_path))
    # No eddy_cuda<ver> present -> fallback default.
    assert _find_eddy_cuda() == 'eddy_cuda10.2'


def test_extended_eddy_cmd_uses_finder(tmp_path, monkeypatch):
    """ExtendedEddy(use_cuda=True) resolves its command via _find_eddy_cuda."""
    _make_exe(tmp_path / 'eddy_cuda11.0')
    monkeypatch.setenv('PATH', str(tmp_path))
    eddy = ExtendedEddy(use_cuda=True)
    assert eddy.cmd == 'eddy_cuda11.0'


def test_extended_eddy_cmd_cpu():
    """ExtendedEddy(use_cuda=False) uses the CPU binary name."""
    eddy = ExtendedEddy(use_cuda=False)
    assert eddy.cmd == 'eddy_cpu'


def test_gather_eddy_inputs_exports_no_warps(tmp_path, monkeypatch):
    """eddy bakes TOPUP's field in, so it must export no SDC warp downstream.

    If ``forward_warps`` ever carried the TOPUP field, it would reach
    ``fieldwarps`` and ``ComposeJacobianWeights`` would derive a determinant
    for a distortion ``eddy`` has already Jacobian-modulated internally --
    applying it twice. The empty list is load-bearing, not incidental.

    Asserted on the interface's actual *output*, not on its source text: a
    source-text check passes even if later code overwrites the list.
    """
    import pandas as pd

    import qsiprep.interfaces.epi_fmap as epi_fmap_mod
    from qsiprep.data import load as load_data
    from qsiprep.interfaces.eddy import GatherEddyInputs
    from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients

    # get_best_b0_topup_inputs_from scores b=0 candidates with TORTOISE's
    # SelectBestB0 binary, which is not installed in this test environment
    # (see test_b0_selection.py's own skipif). That scoring is irrelevant to
    # what this test checks -- GatherEddyInputs' forward_warps/forward_
    # transforms outputs -- so it is stubbed out rather than skipping the
    # test, which would leave this invariant unguarded wherever the real
    # binary happens to be missing.
    def _fake_select_best_b0_report(b0_files, prefix, num_threads=1):
        return pd.DataFrame(
            {
                'mean_cc': [1.0] * len(b0_files),
                'translation_total_mm': [0.0] * len(b0_files),
                'rotation_total_deg': [0.0] * len(b0_files),
            }
        )

    monkeypatch.setattr(epi_fmap_mod, 'select_best_b0_report', _fake_select_best_b0_report)

    dwi = write_dwi_with_gradients(tmp_path / 'sub-1_dwi.nii.gz', nvols=4)
    stem = str(dwi).split('.nii')[0]
    json_file = tmp_path / 'sub-1_dwi.json'
    json_file.write_text('{"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}')

    result = GatherEddyInputs(
        dwi_file=dwi,
        bval_file=stem + '.bval',
        bvec_file=stem + '.bvec',
        json_file=str(json_file),
        original_files=[dwi] * 4,
        topup_requested=True,
        # A single measured distortion group would otherwise make TOPUP
        # refuse to run ("not enough distortion groups"); synb0_requested
        # tells it a synthetic zero-readout b=0 supplies the second group
        # downstream, which is enough to exercise the TOPUP+eddy path here
        # without fabricating a second real fieldmap.
        synb0_requested=True,
        # GatherEddyInputs requires eddy_config; use qsiprep's shipped default
        # rather than inventing one, matching how the workflow wires it.
        eddy_config=str(load_data('eddy_params.json')),
    ).run(cwd=str(tmp_path))

    assert result.outputs.forward_warps == []
    assert result.outputs.forward_transforms == []
