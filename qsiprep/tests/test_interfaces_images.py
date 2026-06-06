"""Tests for qsiprep.interfaces.images."""

from qsiprep.interfaces.images import _find_gradient_file


def _touch(path):
    path.write_text('0\n')
    return str(path)


def test_find_gradient_file_part_specific(tmp_path):
    """Prefers the part-specific gradient file when it exists."""
    dwi = str(tmp_path / 'sub-1_part-mag_dwi.nii.gz')
    part_bval = _touch(tmp_path / 'sub-1_part-mag_dwi.bval')
    _touch(tmp_path / 'sub-1_dwi.bval')  # part-less also present
    assert _find_gradient_file(dwi, '.bval') == part_bval


def test_find_gradient_file_strips_part_entity(tmp_path):
    """Falls back to the part-stripped gradient file (complex DWI convention)."""
    dwi = str(tmp_path / 'sub-1_dir-AP_part-mag_dwi.nii.gz')
    # Only the part-less gradient files exist (shared by mag/phase).
    bval = _touch(tmp_path / 'sub-1_dir-AP_dwi.bval')
    bvec = _touch(tmp_path / 'sub-1_dir-AP_dwi.bvec')
    assert _find_gradient_file(dwi, '.bval') == bval
    assert _find_gradient_file(dwi, '.bvec') == bvec


def test_find_gradient_file_missing_returns_part_specific(tmp_path):
    """When nothing exists, returns the part-specific candidate unchanged."""
    dwi = str(tmp_path / 'sub-1_part-mag_dwi.nii.gz')
    expected = str(tmp_path / 'sub-1_part-mag_dwi.bval')
    assert _find_gradient_file(dwi, '.bval') == expected


def test_find_gradient_file_no_part_entity(tmp_path):
    """A DWI without a part entity resolves to the sibling gradient file."""
    dwi = str(tmp_path / 'sub-1_dwi.nii.gz')
    bval = _touch(tmp_path / 'sub-1_dwi.bval')
    assert _find_gradient_file(dwi, '.bval') == bval
