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
