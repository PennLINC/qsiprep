"""Coefficient files must survive TORTOISE's Siemens reader.

The expected behaviour throughout is that of ``GRADCAL::read_Siemens_format``
(TORTOISEV4 ``src/tools/gradnonlin/gradcal.cxx:99-183``), which has no comment
handling and no exception handling, so a line it cannot parse kills the tool.
"""

import logging

import pytest

from qsiprep.tests.gradient_fixtures import write_siemens_grad
from qsiprep.utils.gradcal import (
    copy_without_comments,
    reader_verdict,
    sanitize_siemens_coefficients,
)

#: The header that crashed a real Cima.X acquisition: the normalization
#: constants are documented in the same ``A(l,m)`` notation the data uses.
CIMAX_HEADER = (
    '# Siemens gradient coefficients\r\n'
    '# Normalization:\r\n'
    '#  A(1,0) = 1.0\r\n'
    '#  A(1,1) = 1.1547 (2/Sqrt[3])\r\n'
    '#  B(1,1) = 1.1547 (2/Sqrt[3])\r\n'
    ' 0.275 = R0\r\n'
)
CIMAX_TERMS = '  1 A( 3, 1) -0.023400 x\r\n  2 A( 3, 0)  0.045600 z\r\n'


def _write(path, text):
    path.write_bytes(text.encode('latin-1'))
    return path


def test_normalization_comment_aborts_the_reader():
    """The exact line from the crash: stoi succeeds, then stof is handed ' = 1.0'."""
    status, detail = reader_verdict('#  A(1,0) = 1.0\r')
    assert status == 'abort'
    assert 'std::stof' in detail
    assert "' = 1.0'" in detail


def test_a_line_ending_at_the_paren_aborts_on_an_empty_coefficient():
    """``size() - posA3 - 2`` underflows, so substr clamps and stof gets ''."""
    status, detail = reader_verdict(' 2 B(1,2)')
    assert status == 'abort'
    assert "std::stof on ''" in detail


def test_non_integer_indices_abort_on_stoi_instead():
    """Distinguishing the two matters: the C++ ``what()`` names one or the other."""
    status, detail = reader_verdict(' Coil (X,Y) info')
    assert status == 'abort'
    assert 'std::stoi' in detail


def test_a_real_coefficient_line_is_a_term():
    assert reader_verdict('  1 A( 3, 1) -0.023400 x') == ('term', 'x')


def test_crlf_does_not_break_a_coefficient_line():
    """getline keeps the \\r, so the axis letter stays in the coefficient
    substring -- stof stops at the space and the line still parses."""
    assert reader_verdict('  1 A( 3, 1) -0.023400 x\r') == ('term', 'x')


def test_a_term_with_no_axis_letter_is_reported():
    """The reader drops it silently; nothing downstream would ever say so."""
    assert reader_verdict('  1 A( 3, 1) -0.023400') == ('term', None)


def test_prose_without_parentheses_is_skipped():
    assert reader_verdict('# Siemens gradient coefficients') == ('skip', None)


def test_sanitize_drops_the_fatal_header_and_keeps_the_data(tmp_path):
    src = _write(tmp_path / 'CimaX_coeff.grad', CIMAX_HEADER + CIMAX_TERMS)
    dest = tmp_path / 'work'
    dest.mkdir()

    out = sanitize_siemens_coefficients(src, dest)

    assert out != src
    assert not [
        line for line in out.read_text().splitlines() if reader_verdict(line)[0] == 'abort'
    ]
    # The data survives untouched.
    assert '  1 A( 3, 1) -0.023400 x' in out.read_text()
    assert ' 0.275 = R0' in out.read_text()


def test_sanitize_never_modifies_the_users_file(tmp_path):
    original = CIMAX_HEADER + CIMAX_TERMS
    src = _write(tmp_path / 'CimaX_coeff.grad', original)
    dest = tmp_path / 'work'
    dest.mkdir()

    sanitize_siemens_coefficients(src, dest)

    assert src.read_bytes() == original.encode('latin-1')


def test_sanitized_copy_keeps_the_basename(tmp_path):
    """finalize.py records os.path.basename(coeff_file) in the graddev sidecar,
    so the sanitized copy has to carry the original name."""
    src = _write(tmp_path / 'CimaX_coeff.grad', CIMAX_HEADER + CIMAX_TERMS)
    dest = tmp_path / 'work'
    dest.mkdir()

    assert sanitize_siemens_coefficients(src, dest).name == 'CimaX_coeff.grad'


def test_a_commented_out_term_that_parses_is_dropped_too(tmp_path):
    """A comment the reader parses is silently folded into the expansion."""
    src = _write(
        tmp_path / 'c.grad',
        ' 0.275 = R0\r\n# 9 A( 3, 1) 0.111 x\r\n  1 A( 3, 1) -0.023400 x\r\n',
    )
    dest = tmp_path / 'work'
    dest.mkdir()

    out = sanitize_siemens_coefficients(src, dest)

    assert out != src
    assert '0.111' not in out.read_text()
    assert '-0.023400' in out.read_text()


def test_a_clean_file_is_passed_through_unchanged(tmp_path):
    src = write_siemens_grad(tmp_path / 'coeff.grad')
    dest = tmp_path / 'work'
    dest.mkdir()

    assert sanitize_siemens_coefficients(src, dest) == src
    assert not list(dest.iterdir())


def test_prose_comments_alone_do_not_trigger_a_copy(tmp_path):
    """A comment the reader skips is harmless, so leave the file alone."""
    src = _write(tmp_path / 'c.grad', '# just prose\r\n 0.275 = R0\r\n' + CIMAX_TERMS)
    dest = tmp_path / 'work'
    dest.mkdir()

    assert sanitize_siemens_coefficients(src, dest) == src


def test_a_fatal_data_line_is_rejected_with_its_line_number(tmp_path):
    """Dropping comments cannot fix this, so fail before the run starts."""
    src = _write(
        tmp_path / 'c.grad',
        '# prose\r\n 0.275 = R0\r\n  1 A( 3, 1) -0.023400 x\r\n  2 B(1,2)\r\n',
    )
    dest = tmp_path / 'work'
    dest.mkdir()

    with pytest.raises(ValueError, match='line 4'):
        sanitize_siemens_coefficients(src, dest)


def test_the_rejection_quotes_the_offending_line(tmp_path):
    src = _write(tmp_path / 'c.grad', ' 0.275 = R0\r\n  2 B(1,2)\r\n')
    dest = tmp_path / 'work'
    dest.mkdir()

    with pytest.raises(ValueError, match='cannot be read by TORTOISE') as excinfo:
        sanitize_siemens_coefficients(src, dest)
    assert '2 B(1,2)' in str(excinfo.value)


@pytest.mark.parametrize('name', ['coeff.dat', 'coeff.gc', 'field.nii', 'field.nii.gz'])
def test_non_siemens_inputs_are_left_alone(tmp_path, name):
    """Only ``.grad`` reaches read_Siemens_format; the other readers differ and
    a displacement field is never parsed at all."""
    src = _write(tmp_path / name, CIMAX_HEADER)
    dest = tmp_path / 'work'
    dest.mkdir()

    assert sanitize_siemens_coefficients(src, dest) == src


def test_dropped_lines_are_logged_with_their_original_numbers(tmp_path, caplog):
    src = _write(tmp_path / 'c.grad', CIMAX_HEADER + CIMAX_TERMS)
    dest = tmp_path / 'work'
    dest.mkdir()
    logger = logging.getLogger('qsiprep.tests.gradcal')

    with caplog.at_level(logging.WARNING, logger=logger.name):
        sanitize_siemens_coefficients(src, dest, logger=logger)

    messages = [record.getMessage() for record in caplog.records]
    # A(1,0) is line 3 of the original file, not line 1 of the comment block.
    assert any('dropped line 3' in message for message in messages)
    assert any('the original is unchanged' in message for message in messages)


def test_copy_without_comments_reports_how_many_it_removed(tmp_path):
    src = _write(tmp_path / 'c.grad', CIMAX_HEADER + CIMAX_TERMS)

    assert copy_without_comments(src, tmp_path / 'out.grad') == 5
    assert '#' not in (tmp_path / 'out.grad').read_text()


def test_parse_args_swaps_in_the_sanitized_copy(tmp_path):
    """End-to-end: a real CLI invocation must not hand TORTOISE the crashing file."""
    from niworkflows.utils.testing import generate_bids_skeleton

    from qsiprep import config
    from qsiprep.cli.parser import parse_args
    from qsiprep.tests.test_cli_run import long

    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(str(bids_dir), long)
    work_dir = tmp_path / 'work'
    coeff = _write(tmp_path / 'CimaX_coeff.grad', CIMAX_HEADER + CIMAX_TERMS)

    saved = config.workflow.gradient_file
    try:
        config.from_dict({'bids_dir': str(bids_dir), 'work_dir': str(work_dir)}, init=True)
        parse_args([
            str(bids_dir),
            str(tmp_path / 'out'),
            'participant',
            '--participant-label',
            '01',
            '--gradient-file',
            str(coeff),
            '--output-resolution',
            '2',
            '--work-dir',
            str(work_dir),
            '--skip-bids-validation',
        ])  # fmt:skip

        used = config.workflow.gradient_file
        assert used != coeff, 'the crashing file was passed straight through'
        assert used.parent == work_dir
        # The sidecar records os.path.basename(coeff_file); it must not change.
        assert used.name == coeff.name
        assert not [
            line for line in used.read_text().splitlines() if reader_verdict(line)[0] == 'abort'
        ]
        assert coeff.read_bytes() == (CIMAX_HEADER + CIMAX_TERMS).encode('latin-1')
    finally:
        config.workflow.gradient_file = saved
