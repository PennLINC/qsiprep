"""Tests for phase-correction config defaults."""

from qsiprep import config


def test_phase_correction_config_defaults():
    assert config.workflow.dwi_phase_correction is None
    assert config.workflow.dwi_phase_tv_weight == 6.0
    assert config.workflow.dwi_phase_dc_kernel == 'Opt5'
