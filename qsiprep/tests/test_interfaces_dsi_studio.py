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
    """Test that only the Apptainer library directory is removed."""
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
    """Test that a trailing slash and a missing library path are handled."""
    environment = {'LD_LIBRARY_PATH': APPTAINER_PATH + os.sep}
    assert _get_dsi_studio_environment(environment)['LD_LIBRARY_PATH'] == ''
    environment = {'OTHER_VARIABLE': 'value'}
    assert _get_dsi_studio_environment(environment) == environment


@pytest.mark.parametrize(
    'interface_class',
    [DSIStudioCreateSrc, DSIStudioGQIReconstruction],
)
def test_dsi_studio_command_line_sanitizes_environment(interface_class, monkeypatch, tmp_path):
    """Test that both DSI Studio command-line actions sanitize the environment."""
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
    """Test that the direct QC subprocess environment is sanitized."""
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
# a signal, or exit 0 with an empty qc.txt. None was checked, and the only
# symptom used to be "IndexError: list index out of range" in load_src_qc_file.
# QC must not fail a run, so each case now logs a warning, carries the reason
# downstream, and the QC values become n/a. A segfault on a DWI series with
# extreme intensity outliers is how this was found.

import numpy as np  # noqa: E402

from qsiprep.interfaces.dsi_studio import (  # noqa: E402
    FIB_QC_MEASURES,
    QC_WARNINGS_COLUMN,
    SRC_QC_MEASURES,
    DSIStudioMergeQC,
    load_fib_qc_file,
    load_src_qc_file,
)

#: A real SRC QC row, from the passing forrest_gump run.
_SRC_HEADER = (
    'file name\tdimension\tresolution\tdwi count(b0/dwi)\tmax b-value\t'
    'DWI contrast\tneighboring DWI correlation\t'
    'neighboring DWI correlation(masked)\t#bad slices\n'
)
_SRC_ROW = (
    'scaled_merged\t32 39 34 \t5 5 5 \t1/32\t800.000000\t1.054243\t0.996735\t0.991714\t0\t\t\n'
)


def _run_qc(tmp_path, monkeypatch, returncode, qc_contents):
    """Run DSIStudioSrcQC against a stubbed dsi_studio; return its results."""
    src_file = tmp_path / 'input.src.gz'
    src_file.touch()
    if qc_contents is not None:
        (tmp_path / 'qc.txt').write_text(qc_contents)

    process = Mock()
    process.communicate.return_value = (b'', b'')
    process.returncode = returncode
    monkeypatch.setattr(dsi_studio_mod, 'Popen', Mock(return_value=process))

    interface = DSIStudioSrcQC(src_file=str(src_file))
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    return interface._results


def test_qc_warns_on_a_segfault_and_names_the_signal(tmp_path, monkeypatch, caplog):
    with caplog.at_level('WARNING'):
        results = _run_qc(tmp_path, monkeypatch, -11, '')

    assert 'SIGSEGV' in results['warning']
    assert any('SIGSEGV' in message for message in caplog.messages)


def test_qc_warns_on_a_nonzero_exit(tmp_path, monkeypatch):
    results = _run_qc(tmp_path, monkeypatch, 1, '')

    assert 'exited with status 1' in results['warning']


def test_qc_warns_on_an_empty_output_despite_success(tmp_path, monkeypatch):
    """Test that QC warns on an empty output despite success.

    Exit 0 plus an empty qc.txt is how DSI Studio reports giving up.
    """
    results = _run_qc(tmp_path, monkeypatch, 0, '')

    assert 'empty QC file' in results['warning']


def test_qc_creates_an_empty_file_when_none_was_written(tmp_path, monkeypatch):
    """Test that QC creates an empty file when none was written.

    qc_txt is declared exists=True, so a missing file would fail the node.
    """
    results = _run_qc(tmp_path, monkeypatch, 0, None)

    assert 'wrote no QC file' in results['warning']
    assert os.path.getsize(results['qc_txt']) == 0


def test_qc_is_silent_on_a_real_measurement(tmp_path, monkeypatch, caplog):
    with caplog.at_level('WARNING'):
        results = _run_qc(tmp_path, monkeypatch, 0, _SRC_HEADER + _SRC_ROW)

    assert 'warning' not in results
    assert not caplog.records


def test_load_src_qc_file_parses_a_real_row(tmp_path):
    qc = tmp_path / 'qc.txt'
    qc.write_text(_SRC_HEADER + _SRC_ROW)

    data = load_src_qc_file(str(qc))

    assert list(data) == list(SRC_QC_MEASURES)
    assert data['dimension_y'] == [39.0]
    assert data['neighbor_corr'] == [0.996735]
    assert data['num_directions'] == [32.0]


@pytest.mark.parametrize('contents', ['', _SRC_HEADER], ids=['empty', 'header-only'])
def test_load_src_qc_file_returns_na_with_every_column(tmp_path, contents):
    """Test that load_src_qc_file returns n/a with every column.

    Every run's image_qc.tsv must keep the same columns.
    """
    qc = tmp_path / 'qc.txt'
    qc.write_text(contents)

    data = load_src_qc_file(str(qc), prefix='raw_')

    assert list(data) == ['raw_' + name for name in SRC_QC_MEASURES]
    assert all(np.isnan(value[0]) for value in data.values())


def test_load_src_qc_file_still_rejects_an_unknown_format(tmp_path):
    """Test that load_src_qc_file still rejects an unknown format.

    A row DSI Studio did write, in a shape we do not know, is a version
    mismatch rather than a failed measurement, so it is not hidden as n/a.
    """
    qc = tmp_path / 'qc.txt'
    qc.write_text('header\n' + '\t'.join(['a', 'b', 'c']) + '\n')

    with pytest.raises(ValueError, match='expected 7, 8 or 9 tab-separated fields, got 3'):
        load_src_qc_file(str(qc))


def test_load_fib_qc_file_returns_na_when_empty(tmp_path):
    qc = tmp_path / 'fib_qc.txt'
    qc.write_text('')

    data = load_fib_qc_file(str(qc))

    assert list(data) == list(FIB_QC_MEASURES)
    assert np.isnan(data['coherence_index'][0])


def test_load_fib_qc_file_still_reads_a_good_file(tmp_path):
    good = tmp_path / 'fib_qc.txt'
    good.write_text('header line\ncoherence index\t0.42\n')

    assert load_fib_qc_file(str(good)) == {'coherence_index': [0.42]}


def _merge(tmp_path, src_contents, fib_contents, **warnings):
    src = tmp_path / 'src_qc.txt'
    src.write_text(src_contents)
    fib = tmp_path / 'fib_qc.txt'
    fib.write_text(fib_contents)
    interface = DSIStudioMergeQC(src_qc=str(src), fib_qc=str(fib), **warnings)
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    import pandas as pd

    return pd.read_csv(interface._results['qc_file'], keep_default_na=False)


def test_merge_qc_has_an_empty_warning_column_when_qc_succeeds(tmp_path):
    table = _merge(tmp_path, _SRC_HEADER + _SRC_ROW, 'header\ncoherence\t0.42\n')

    assert table[QC_WARNINGS_COLUMN][0] == ''
    assert table['neighbor_corr'][0] == pytest.approx(0.996735)


def test_merge_qc_passes_the_upstream_reason_through(tmp_path):
    """Test that merging QC passes the upstream reason through.

    The reason DSIStudioSrcQC gave is what reaches the report.
    """
    table = _merge(
        tmp_path,
        '',
        'header\ncoherence\t0.42\n',
        src_qc_warning='DSI Studio was killed by SIGSEGV.',
    )

    assert table[QC_WARNINGS_COLUMN][0] == 'SRC QC: DSI Studio was killed by SIGSEGV.'
    assert table['neighbor_corr'][0] == ''  # NaN, read back unconverted
    assert table['coherence_index'][0] == pytest.approx(0.42)


def test_merge_qc_trusts_nothing_from_a_stage_that_crashed(tmp_path):
    """Test that merging QC trusts nothing from a stage that crashed.

    After a crash, whatever DSI Studio wrote cannot be assumed complete.
    """
    table = _merge(
        tmp_path,
        _SRC_HEADER + _SRC_ROW,
        'header\ncoherence\t0.42\n',
        src_qc_warning='DSI Studio exited with status 1.',
    )

    assert table['neighbor_corr'][0] == ''


def test_merge_qc_notices_an_empty_file_with_no_upstream_reason(tmp_path):
    table = _merge(tmp_path, _SRC_HEADER + _SRC_ROW, '')

    assert table[QC_WARNINGS_COLUMN][0] == 'FIB QC: The QC file is empty.'
