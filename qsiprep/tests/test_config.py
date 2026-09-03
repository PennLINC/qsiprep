"""Tests for qsiprep.config."""

import os
from pathlib import Path

import pytest

from qsiprep import config


@pytest.fixture
def mrtrix_trees(tmp_path, monkeypatch):
    """Declare two fake MRtrix3 installations and make PATH restorable.

    ``workflow.init()`` mutates ``os.environ['PATH']`` directly, which monkeypatch
    would not undo on its own. Setting PATH through monkeypatch first registers it
    for restoration at teardown.
    """
    stable = tmp_path / 'mrtrix3-stable'
    dev = tmp_path / 'mrtrix3-dev'
    (stable / 'bin').mkdir(parents=True)
    (dev / 'bin').mkdir(parents=True)
    monkeypatch.setenv('MRTRIX3_STABLE_HOME', str(stable))
    monkeypatch.setenv('MRTRIX3_DEV_HOME', str(dev))
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    return str(stable), str(dev)


@pytest.mark.parametrize('selected', ['stable', 'dev'])
def test_workflow_init_puts_the_selected_tree_first(monkeypatch, mrtrix_trees, selected):
    """Resolve commands from the requested MRtrix3, falling through to the other tree.

    The second entry is load-bearing: dwidenoise2 exists only in the development
    tree, so it must remain reachable when ``stable`` is selected.
    """
    stable, dev = mrtrix_trees
    monkeypatch.setattr(config.workflow, 'mrtrix_version', selected)

    config.workflow.init()

    entries = os.environ['PATH'].split(os.pathsep)
    expected_first = dev if selected == 'dev' else stable
    expected_second = stable if selected == 'dev' else dev
    assert entries[0] == str(Path(expected_first, 'bin'))
    assert entries[1] == str(Path(expected_second, 'bin'))
    assert config.environment.mrtrix3_home == expected_first


def test_workflow_init_is_a_noop_without_declared_trees(monkeypatch):
    """Leave PATH alone on a bare-metal install, which has one MRtrix3 already on it."""
    monkeypatch.delenv('MRTRIX3_STABLE_HOME', raising=False)
    monkeypatch.delenv('MRTRIX3_DEV_HOME', raising=False)
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'dev')

    config.workflow.init()

    assert os.environ['PATH'] == '/usr/bin:/bin'
    assert config.environment.mrtrix3_home is None


def test_workflow_init_raises_when_the_selected_tree_is_missing(monkeypatch, tmp_path):
    """Fail loudly rather than silently running the other version.

    Falling back would build the complex workflow path and then hand complex data
    to a released mrdegibbs that cannot read it.
    """
    dev = tmp_path / 'mrtrix3-dev'
    (dev / 'bin').mkdir(parents=True)
    monkeypatch.setenv('MRTRIX3_DEV_HOME', str(dev))
    monkeypatch.delenv('MRTRIX3_STABLE_HOME', raising=False)
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'stable')

    with pytest.raises(RuntimeError, match='stable'):
        config.workflow.init()


def test_workflow_init_does_not_accumulate_duplicates(monkeypatch, mrtrix_trees):
    """Keep PATH stable across reloads; the image already bakes both trees into it."""
    stable, dev = mrtrix_trees
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'stable')

    config.workflow.init()
    after_first = os.environ['PATH']
    config.workflow.init()

    assert os.environ['PATH'] == after_first
