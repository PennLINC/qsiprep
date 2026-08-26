"""The synthetic coefficient fixtures must satisfy TORTOISE's .grad grammar.

Every binary-backed test downstream feeds these files to the real TORTOISE
tools. A fixture that silently fails to parse would turn those tests into
no-ops, so the grammar is asserted here directly against the rules in
``src/tools/gradnonlin/gradcal.cxx:99-183``.
"""

import nibabel as nb
import numpy as np

from qsiprep.tests.gradient_fixtures import (
    write_dwi_with_gradients,
    write_itk_field,
    write_siemens_grad,
)


def _coefficient_lines(text):
    return [ln for ln in text.splitlines() if '(' in ln and ')' in ln]


def test_siemens_grad_axis_letter_is_last_character(tmp_path):
    """TORTOISE drops the final character when parsing the coefficient."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert line[-1] in 'xyz', line


def test_siemens_grad_axis_letter_appears_exactly_once(tmp_path):
    """The axis is chosen by searching the whole line for x, then y, then z."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert sum(line.count(axis) for axis in 'xyz') == 1, line


def test_siemens_grad_open_paren_position(tmp_path):
    """find_first_of("(", 3, 3) requires the paren at index >= 3, and < 10."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert 3 <= line.index('(') < 10, line


def test_siemens_grad_r0_occupies_columns_one_to_five(tmp_path):
    """R0 = atof(substr(1, 5)) * 1000, so 0.250 must sit at columns 1-5."""
    text = write_siemens_grad(tmp_path / 'coeff.grad', r0_m=0.250).read_text()
    r0_lines = [ln for ln in text.splitlines() if '= R0' in ln]
    assert len(r0_lines) == 1
    assert float(r0_lines[0][1:6]) * 1000 == 250.0


def test_siemens_grad_records_requested_terms(tmp_path):
    path = write_siemens_grad(
        tmp_path / 'coeff.grad', terms=[('x', 3, 1, -0.0234), ('z', 5, 0, 0.0011)]
    )
    lines = _coefficient_lines(path.read_text())
    assert len(lines) == 2
    assert lines[0].endswith('x')
    assert '( 3, 1)' in lines[0]
    assert lines[1].endswith('z')
    assert '( 5, 0)' in lines[1]


def test_itk_field_is_five_dimensional_vector_image(tmp_path):
    img = nb.load(str(write_itk_field(tmp_path / 'field.nii', shape=(4, 5, 6))))
    assert img.shape == (4, 5, 6, 1, 3)


def test_itk_field_is_nonzero(tmp_path):
    """A zero field would make a warp test pass for the wrong reason."""
    img = nb.load(str(write_itk_field(tmp_path / 'field.nii')))
    assert np.abs(np.asanyarray(img.dataobj)).max() > 0


def test_write_dwi_with_gradients_makes_siblings(tmp_path):
    path = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz', nvols=5)
    stem = str(path).split('.nii')[0]
    assert nb.load(path).shape == (8, 8, 8, 5)
    assert np.loadtxt(stem + '.bval').size == 5
    assert np.loadtxt(stem + '.bvec').shape == (3, 5)
