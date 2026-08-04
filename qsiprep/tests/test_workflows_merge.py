"""Tests for the qsiprep.workflows.dwi.merge module."""

import pytest

from qsiprep import config

from qsiprep.interfaces import mrtrix
from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf


@pytest.mark.parametrize('use_phase', [False, True])
def test_dwidenoise_workflow_uses_dwidenoise(monkeypatch, use_phase):
    """Build a DWIDenoise node, not Patch2Self, when ``dwidenoise`` is requested."""
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise')
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=use_phase,
        do_biascorr=False,
    )
    denoiser = workflow.get_node('denoiser')

    assert isinstance(denoiser.interface, mrtrix.DWIDenoise)
    assert denoiser.inputs.extent == (5, 5, 5)
    assert denoiser.inputs.nthreads == 1


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_dwidenoise_workflow_resolves_auto_window(monkeypatch, denoise_method):
    """Resolve the default ``auto`` window size for every dwidenoise variant."""
    monkeypatch.setattr(config.workflow, 'denoise_method', denoise_method)
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 'auto')
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
        do_biascorr=False,
    )
    denoiser = workflow.get_node('denoiser')

    # cbrt(30) rounded up to the closest odd integer
    assert denoiser.inputs.extent == (5, 5, 5)


def test_dwidenoise2_cli_parameters_reach_workflow(monkeypatch):
    """Forward parsed DWIDenoise2 parameters to the workflow node."""
    monkeypatch.setattr(
        config.workflow,
        'denoise_method',
        'dwidenoise2;demodulate:nonlinear;decomposition:bdcsvd',
    )
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
        do_biascorr=False,
    )
    denoiser = workflow.get_node('denoiser')

    assert denoiser.inputs.demodulate == 'nonlinear'
    assert denoiser.inputs.decomposition == 'bdcsvd'
    assert denoiser.inputs.onepass is True
    assert denoiser.inputs.subsample == 1
