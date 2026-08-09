"""Tests for per-task GPU selection (``--gpu`` / ``qsiprep.utils.gpu``)."""

import pytest

from qsiprep.utils.gpu import (
    GPU_TASKS,
    check_gpu_available,
    gpu_enabled,
    resolve_gpu_tasks,
)


@pytest.fixture
def gpu_config():
    """Pin ``config.workflow.gpu`` and restore it afterwards.

    Starts from ``[]`` -- ``--gpu`` given but selecting nothing. Tests that care
    about the *absent* flag set ``None`` explicitly, since the two differ.
    """
    from qsiprep import config

    saved = getattr(config.workflow, 'gpu', None)
    config.workflow.gpu = []
    try:
        yield config
    finally:
        config.workflow.gpu = saved


# ---------------------------------------------------------------------------
# resolve_gpu_tasks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('requested', 'expected'),
    [
        (None, set()),
        ([], set()),
        (['none'], set()),
        (['eddy'], {'eddy'}),
        (['diffprep', 'drbuddi'], {'diffprep', 'drbuddi'}),
        (['all'], set(GPU_TASKS)),
        # "none" is unambiguous: it wins over anything else on the same line.
        (['all', 'none'], set()),
        (['eddy', 'none'], set()),
    ],
)
def test_resolve_gpu_tasks(requested, expected):
    assert resolve_gpu_tasks(requested) == expected


def test_every_task_is_individually_selectable():
    """The point of --gpu taking a list: an 8 GB card runs some tasks, not others."""
    for task in GPU_TASKS:
        assert resolve_gpu_tasks([task]) == {task}


# ---------------------------------------------------------------------------
# gpu_enabled
# ---------------------------------------------------------------------------


def test_gpu_enabled_reads_the_cli_list(gpu_config):
    gpu_config.workflow.gpu = ['diffprep', 'drbuddi']
    assert gpu_enabled('diffprep') is True
    assert gpu_enabled('drbuddi') is True
    # The real motivating case: dMRI tools on the GPU, the memory-hungry
    # anatomical ones left on CPU.
    assert gpu_enabled('synthstrip') is False
    assert gpu_enabled('synthseg') is False


def test_gpu_enabled_defaults_to_cpu(gpu_config):
    for task in GPU_TASKS:
        assert gpu_enabled(task) is False


def test_gpu_enabled_rejects_unknown_tasks(gpu_config):
    with pytest.raises(ValueError, match='Unknown GPU task'):
        gpu_enabled('not_a_task')


def test_absent_gpu_flag_leaves_legacy_config_files_working(gpu_config, caplog):
    """No --gpu at all: an existing --eddy-config "use_cuda" must still apply.

    This is the difference between ``--gpu none`` and omitting the flag. Getting
    it wrong would silently drop existing GPU runs to CPU -- and because the CUDA
    and CPU builds are not numerically identical, silently change their results.
    """
    gpu_config.workflow.gpu = None
    caplog.clear()
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert gpu_enabled('eddy', config_file_value=True) is True
        assert gpu_enabled('eddy', config_file_value=False) is False
        assert gpu_enabled('diffprep') is False
    # Nothing to disagree with, so nothing to warn about.
    assert 'conflicts' not in caplog.text


def test_explicit_gpu_none_overrides_the_config_file(gpu_config, caplog):
    """``--gpu none`` is an explicit "off" that beats a legacy use_cuda=true."""
    gpu_config.workflow.gpu = ['none']
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert gpu_enabled('eddy', config_file_value=True) is False
    assert 'conflicts' in caplog.text


def test_cli_overrides_the_config_file_and_warns(gpu_config, caplog):
    """--gpu wins over legacy "use_cuda", and the disagreement is not silent."""
    gpu_config.workflow.gpu = ['eddy']

    # Config file says off, CLI says on -> on, with a warning.
    caplog.clear()
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert gpu_enabled('eddy', config_file_value=False) is True
    assert 'conflicts' in caplog.text

    # Config file says on, CLI omits it -> off, with a warning.
    caplog.clear()
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert gpu_enabled('diffprep', config_file_value=True) is False
    assert 'conflicts' in caplog.text


def test_agreement_with_the_config_file_is_quiet(gpu_config, caplog):
    gpu_config.workflow.gpu = ['eddy']
    caplog.clear()
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert gpu_enabled('eddy', config_file_value=True) is True
        assert gpu_enabled('diffprep', config_file_value=False) is False
    assert 'conflicts' not in caplog.text


# ---------------------------------------------------------------------------
# check_gpu_available (preflight)
# ---------------------------------------------------------------------------


def test_preflight_is_a_noop_without_gpu_tasks(monkeypatch):
    """No --gpu means no GPU probing at all, on any machine."""
    import qsiprep.utils.gpu as gpu_mod

    def _boom():
        raise AssertionError('must not probe for a GPU when none was requested')

    monkeypatch.setattr(gpu_mod, '_gpu_visible', _boom)
    check_gpu_available([])
    check_gpu_available(['none'])


def test_preflight_raises_when_no_device_is_visible(monkeypatch):
    """The common mistake: --gpu without `docker run --gpus all`."""
    import qsiprep.utils.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, '_gpu_visible', lambda: False)
    with pytest.raises(RuntimeError, match='no CUDA device is visible'):
        check_gpu_available(['diffprep'])


def test_preflight_raises_when_the_gpu_build_is_missing(monkeypatch):
    import qsiprep.utils.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, '_gpu_visible', lambda: True)
    monkeypatch.setattr(
        gpu_mod,
        '_missing_binary',
        lambda task: 'TORTOISEProcess_cuda' if task == 'diffprep' else None,
    )
    with pytest.raises(RuntimeError, match='TORTOISEProcess_cuda'):
        check_gpu_available(['diffprep', 'drbuddi'])

    # The task that *is* installed passes on its own.
    check_gpu_available(['drbuddi'])


def test_preflight_passes_when_everything_is_present(monkeypatch):
    import qsiprep.utils.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, '_gpu_visible', lambda: True)
    monkeypatch.setattr(gpu_mod, '_missing_binary', lambda task: None)
    check_gpu_available(['all'])


def test_gpu_visible_is_false_without_nvidia_smi(monkeypatch):
    """Inside a container the toolkit only injects nvidia-smi when --gpus was used."""
    import qsiprep.utils.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, 'which', lambda _: None)
    monkeypatch.setattr(gpu_mod.os.path, 'exists', lambda _: False)
    assert gpu_mod._gpu_visible() is False


# ---------------------------------------------------------------------------
# Reaching the interfaces
# ---------------------------------------------------------------------------


def test_synthstrip_and_synthseg_reach_the_gpu(gpu_config):
    """Both were previously unreachable: SynthStrip's `gpu` was never set and
    SynthSeg's opt-OUT `cpu` was never overridden."""
    from qsiprep.interfaces.freesurfer import SynthSeg, SynthStrip

    gpu_config.workflow.gpu = ['synthstrip', 'synthseg']
    assert '-g' in SynthStrip(input_image=__file__, gpu=gpu_enabled('synthstrip')).cmdline
    assert '--cpu' not in SynthSeg(input_image=__file__, cpu=not gpu_enabled('synthseg')).cmdline

    gpu_config.workflow.gpu = []
    assert '-g' not in SynthStrip(input_image=__file__, gpu=gpu_enabled('synthstrip')).cmdline
    assert '--cpu' in SynthSeg(input_image=__file__, cpu=not gpu_enabled('synthseg')).cmdline


def test_diffprep_and_drbuddi_are_selected_independently(gpu_config):
    """DIFFPREP and DRBUDDI are separate binaries, so --gpu treats them separately."""
    from qsiprep.interfaces.tortoise import DIFFPREP, DRBUDDI

    gpu_config.workflow.gpu = ['drbuddi']
    assert DIFFPREP(use_cuda=gpu_enabled('diffprep')).cmd == 'TORTOISEProcess'
    assert DRBUDDI(use_cuda=gpu_enabled('drbuddi')).cmd == 'DRBUDDI_cuda'
