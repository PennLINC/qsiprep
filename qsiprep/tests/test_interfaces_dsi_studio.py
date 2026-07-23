"""Tests for the qsiprep.interfaces.dsi_studio module."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nipype.interfaces.base import CommandLine

import qsiprep.interfaces.dsi_studio as dsi_studio_mod
from qsiprep.interfaces.dsi_studio import (
    DSIStudioCreateSrc,
    DSIStudioGQIReconstruction,
    DSIStudioSrcQC,
    _get_dsi_studio_environment,
)

APPTAINER_PATH = '/.singularity.d/libs'


def test_get_dsi_studio_environment_removes_apptainer_path():
    """Remove only the Apptainer library directory."""
    paths = ['/opt/freesurfer/lib', APPTAINER_PATH, '/opt/conda/lib']
    environment = {
        'LD_LIBRARY_PATH': os.pathsep.join(paths),
        'OTHER_VARIABLE': 'value',
    }

    result = _get_dsi_studio_environment(environment)

    expected = os.pathsep.join(['/opt/freesurfer/lib', '/opt/conda/lib'])
    assert result['LD_LIBRARY_PATH'] == expected
    assert result['OTHER_VARIABLE'] == 'value'


def test_get_dsi_studio_environment_handles_edge_cases():
    """Handle a trailing slash and a missing library path."""
    environment = {'LD_LIBRARY_PATH': APPTAINER_PATH + os.sep}
    assert _get_dsi_studio_environment(environment)['LD_LIBRARY_PATH'] == ''
    environment = {'OTHER_VARIABLE': 'value'}
    assert _get_dsi_studio_environment(environment) == environment


@pytest.mark.parametrize(
    'interface_class',
    [DSIStudioCreateSrc, DSIStudioGQIReconstruction],
)
def test_dsi_studio_command_line_sanitizes_environment(interface_class, monkeypatch, tmp_path):
    """Sanitize both DSI Studio command-line actions."""
    paths = [APPTAINER_PATH, '/opt/freesurfer/lib']
    monkeypatch.setenv('LD_LIBRARY_PATH', os.pathsep.join(paths))
    captured_environment = {}

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        captured_environment.update(self.inputs.environ)
        return runtime

    monkeypatch.setattr(CommandLine, '_run_interface', _run_interface)
    interface = interface_class()
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))

    assert captured_environment['LD_LIBRARY_PATH'] == '/opt/freesurfer/lib'


def test_dsi_studio_qc_sanitizes_environment(monkeypatch, tmp_path):
    """Sanitize the direct QC subprocess environment."""
    paths = ['/opt/freesurfer/lib', APPTAINER_PATH]
    monkeypatch.setenv('LD_LIBRARY_PATH', os.pathsep.join(paths))
    src_file = tmp_path / 'input.src.gz'
    src_file.touch()
    process = Mock()
    process.communicate.return_value = (b'', b'')
    popen = Mock(return_value=process)
    monkeypatch.setattr(dsi_studio_mod, 'Popen', popen)

    interface = DSIStudioSrcQC(src_file=str(src_file))
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))

    environment = popen.call_args.kwargs['env']
    assert environment['LD_LIBRARY_PATH'] == '/opt/freesurfer/lib'
