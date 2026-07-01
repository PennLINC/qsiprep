"""Workflow-build tests for phase correction in the denoising workflow."""

import pytest

from qsiprep import config
from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf


@pytest.fixture
def _denoise_defaults(monkeypatch):
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise', raising=False)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none', raising=False)
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5, raising=False)
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True, raising=False)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1, raising=False)


def _node_names(wf):
    # Top-level nodes in this workflow have flat names (no subworkflow prefix).
    return set(wf.list_node_names())


def _build(use_phase):
    return init_dwi_denoising_wf(
        source_file='/data/sub-01/dwi/sub-01_part-mag_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=10,
        use_phase=use_phase,
        do_biascorr=False,
        name='test_denoise',
    )


@pytest.mark.usefixtures('_denoise_defaults')
def test_phase_correction_node_present_when_enabled(monkeypatch):
    monkeypatch.setattr(config.workflow, 'dwi_phase_correction', 'tv', raising=False)
    wf = _build(use_phase=True)
    names = _node_names(wf)
    assert 'phase_correct' in names
    assert 'split_complex' not in names


@pytest.mark.usefixtures('_denoise_defaults')
def test_magnitude_path_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr(config.workflow, 'dwi_phase_correction', 'none', raising=False)
    wf = _build(use_phase=True)
    names = _node_names(wf)
    assert 'split_complex' in names
    assert 'phase_correct' not in names


from nipype.interfaces.base import isdefined

from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf


def _get_node(wf, name):
    return wf.get_node(name)


def test_derivatives_tagged_part_real_when_enabled(monkeypatch):
    monkeypatch.setattr(config.workflow, 'dwi_phase_correction', 'tv', raising=False)
    monkeypatch.setattr(config.workflow, 'hmc_model', 'eddy', raising=False)
    wf = init_dwi_derivatives_wf(source_file='/data/sub-01/dwi/sub-01_part-mag_dwi.nii.gz')
    for node_name in (
        'ds_dwi_t1',
        'ds_bvals_t1',
        'ds_bvecs_t1',
        'ds_gradient_table_t1',
        'ds_btable_t1',
    ):
        assert _get_node(wf, node_name).inputs.part == 'real'
    # Derived references/maps are not tagged
    assert not isdefined(_get_node(wf, 'ds_t1_b0_ref').inputs.part)
    assert not isdefined(_get_node(wf, 'ds_dwi_mask_t1').inputs.part)


def test_derivatives_not_tagged_when_disabled(monkeypatch):
    monkeypatch.setattr(config.workflow, 'dwi_phase_correction', 'none', raising=False)
    monkeypatch.setattr(config.workflow, 'hmc_model', 'eddy', raising=False)
    wf = init_dwi_derivatives_wf(source_file='/data/sub-01/dwi/sub-01_part-mag_dwi.nii.gz')
    assert not isdefined(_get_node(wf, 'ds_dwi_t1').inputs.part)
