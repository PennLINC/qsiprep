"""Tests for loading epi fieldmaps."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces import epi_fmap, fmap

IMPLEMENTATIONS = [epi_fmap.load_epi_dwi_fieldmaps, fmap.load_epi_dwi_fieldmaps]


def _write_epi(fname, num_vols=None, bvals=None):
    """Write a tiny epi image (3D if num_vols is None) and optionally a .bval file."""
    shape = (4, 4, 3) if num_vols is None else (4, 4, 3, num_vols)
    nb.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)).to_filename(fname)
    if bvals is not None:
        np.savetxt(fname.replace(".nii.gz", ".bval"), np.atleast_1d(bvals), fmt="%d")
    return fname


@pytest.mark.parametrize("load_fieldmaps", IMPLEMENTATIONS)
@pytest.mark.parametrize("three_d_first", [False, True])
def test_load_epi_dwi_fieldmaps_mixed_3d_4d(tmp_path, load_fieldmaps, three_d_first):
    """A mix of 3D and 4D fieldmap files should concatenate without error."""
    pa_file = _write_epi(str(tmp_path / "sub-1_dir-PA_epi.nii.gz"), num_vols=2)
    ap_file = _write_epi(str(tmp_path / "sub-1_dir-AP_epi.nii.gz"))

    fmap_list = [ap_file, pa_file] if three_d_first else [pa_file, ap_file]
    concatenated, b0_indices, original_files = load_fieldmaps(fmap_list, 100)

    assert concatenated.ndim == 4
    assert concatenated.shape[3] == 3
    assert b0_indices == [0, 1, 2]
    expected_files = [[fmap_file] * (2 if fmap_file == pa_file else 1) for fmap_file in fmap_list]
    assert original_files == sum(expected_files, [])


@pytest.mark.parametrize("load_fieldmaps", IMPLEMENTATIONS)
def test_load_epi_dwi_fieldmaps_3d_with_bval(tmp_path, load_fieldmaps):
    """A 3D fieldmap with a single-entry bval file is kept or excluded by b0_threshold."""
    pa_file = _write_epi(str(tmp_path / "sub-1_dir-PA_epi.nii.gz"), num_vols=2)
    b0_file = _write_epi(str(tmp_path / "sub-1_dir-AP_epi.nii.gz"), bvals=0)
    highb_file = _write_epi(str(tmp_path / "sub-1_dir-AP_run-2_epi.nii.gz"), bvals=1000)

    concatenated, b0_indices, original_files = load_fieldmaps(
        [pa_file, b0_file, highb_file], 100
    )

    assert concatenated.shape[3] == 4
    # The b=1000 volume (index 3) is not usable as a b=0
    assert b0_indices == [0, 1, 2]
    assert original_files == [pa_file, pa_file, b0_file, highb_file]
