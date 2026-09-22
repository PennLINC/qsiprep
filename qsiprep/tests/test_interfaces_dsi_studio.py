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
    # A real qc.txt and a clean exit: the interface now rejects both a bad
    # exit status and an empty output, and this test is about the environment.
    (tmp_path / 'qc.txt').write_text('header\ninput\t32 39 34\t5 5 5\n')
    process = Mock()
    process.communicate.return_value = (b'', b'')
    process.returncode = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(dsi_studio_mod, 'Popen', popen)

    interface = DSIStudioSrcQC(src_file=str(src_file))
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))

    environment = popen.call_args.kwargs['env']
    assert environment['LD_LIBRARY_PATH'] == '/opt/freesurfer/lib'


# --- DSI Studio failing silently ---------------------------------------------
#
# DSI Studio can fail three ways without saying so: a non-zero exit, a crash on
# a signal, or exit 0 with an empty qc.txt. All three used to surface much
# later as "IndexError: list index out of range" in load_src_qc_file, naming
# neither the file, the command, nor the input. A segfault on a DWI series with
# extreme intensity outliers is one way to reach the third case.


def _qc_interface(tmp_path, monkeypatch, returncode, qc_contents):
    src_file = tmp_path / 'input.src.gz'
    src_file.touch()
    if qc_contents is not None:
        (tmp_path / 'qc.txt').write_text(qc_contents)

    process = Mock()
    process.communicate.return_value = (b'', b'')
    process.returncode = returncode
    monkeypatch.setattr(dsi_studio_mod, 'Popen', Mock(return_value=process))

    interface = DSIStudioSrcQC(src_file=str(src_file))
    return lambda: interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))


def test_qc_reports_a_segfault_by_name(tmp_path, monkeypatch):
    """A crash must name the signal, not be swallowed."""
    run = _qc_interface(tmp_path, monkeypatch, -11, '')

    with pytest.raises(RuntimeError, match='killed by SIGSEGV'):
        run()


def test_qc_reports_a_nonzero_exit(tmp_path, monkeypatch):
    run = _qc_interface(tmp_path, monkeypatch, 1, '')

    with pytest.raises(RuntimeError, match='exited with status 1'):
        run()


def test_qc_reports_an_empty_output_despite_success(tmp_path, monkeypatch):
    """Exit 0 plus an empty qc.txt is how DSI Studio reports giving up."""
    run = _qc_interface(tmp_path, monkeypatch, 0, '')

    with pytest.raises(RuntimeError, match='empty'):
        run()


def test_qc_reports_a_missing_output(tmp_path, monkeypatch):
    run = _qc_interface(tmp_path, monkeypatch, 0, None)

    with pytest.raises(RuntimeError, match='wrote no'):
        run()


def test_qc_accepts_a_real_measurement(tmp_path, monkeypatch):
    header = 'file name\tdimension\tresolution\n'
    run = _qc_interface(tmp_path, monkeypatch, 0, header + 'input\t32 39 34\t5 5 5\n')

    run()  # must not raise


def test_load_src_qc_file_names_the_empty_file(tmp_path):
    from qsiprep.interfaces.dsi_studio import load_src_qc_file

    empty = tmp_path / 'qc.txt'
    empty.write_text('')

    with pytest.raises(ValueError, match='no measurements.*file is empty'):
        load_src_qc_file(str(empty))


def test_load_src_qc_file_names_a_header_only_file(tmp_path):
    from qsiprep.interfaces.dsi_studio import load_src_qc_file

    header_only = tmp_path / 'qc.txt'
    header_only.write_text('file name\tdimension\tresolution\n')

    with pytest.raises(ValueError, match='only a header'):
        load_src_qc_file(str(header_only))


def test_load_src_qc_file_reports_an_unexpected_field_count(tmp_path):
    from qsiprep.interfaces.dsi_studio import load_src_qc_file

    odd = tmp_path / 'qc.txt'
    odd.write_text('header\n' + '\t'.join(['a', 'b', 'c']) + '\n')

    with pytest.raises(ValueError, match='expected 7, 8 or 9 tab-separated fields, got 3'):
        load_src_qc_file(str(odd))


def test_load_fib_qc_file_names_the_empty_file(tmp_path):
    from qsiprep.interfaces.dsi_studio import load_fib_qc_file

    empty = tmp_path / 'fib_qc.txt'
    empty.write_text('')

    with pytest.raises(ValueError, match='no measurements'):
        load_fib_qc_file(str(empty))


def test_load_fib_qc_file_still_reads_a_good_file(tmp_path):
    from qsiprep.interfaces.dsi_studio import load_fib_qc_file

    good = tmp_path / 'fib_qc.txt'
    good.write_text('header line\ncoherence index\t0.42\n')

    assert load_fib_qc_file(str(good)) == {'coherence_index': [0.42]}
