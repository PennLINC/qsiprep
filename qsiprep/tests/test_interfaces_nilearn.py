"""Tests for the qsiprep.interfaces.nilearn module."""

import nibabel as nb
import numpy as np

from qsiprep.interfaces.nilearn import MaskWithinDWIFieldOfView


def test_mask_within_dwi_field_of_view_removes_zero_mean_b0_voxels(tmp_path):
    """Mask voxels without valid mean-b0 support should be excluded."""
    mask_file = tmp_path / 'mask.nii.gz'
    b0_file = tmp_path / 'b0_mean.nii.gz'

    mask_data = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_data[1, 1, 1] = 1
    mask_data[1, 1, 2] = 1

    b0_data = np.zeros((3, 3, 3), dtype=np.float32)
    b0_data[1, 1, 1] = 1000.0
    b0_data[1, 1, 2] = 0.0

    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)
    nb.Nifti1Image(b0_data, np.eye(4)).to_filename(b0_file)

    result = MaskWithinDWIFieldOfView(in_mask=str(mask_file), b0_image=str(b0_file)).run(
        cwd=str(tmp_path)
    )

    refined_mask = nb.load(result.outputs.out_mask).get_fdata()

    assert refined_mask[1, 1, 1] == 1
    assert refined_mask[1, 1, 2] == 0
    assert refined_mask.sum() == 1


def test_mask_within_dwi_field_of_view_averages_4d_b0_series(tmp_path):
    """The support check should use the mean across a 4D b0 series."""
    mask_file = tmp_path / 'mask4d.nii.gz'
    b0_file = tmp_path / 'b0_series.nii.gz'

    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 1
    mask_data[0, 0, 1] = 1

    b0_data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    b0_data[0, 0, 0, :] = [100.0, 120.0]
    b0_data[0, 0, 1, :] = [0.0, 0.0]

    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)
    nb.Nifti1Image(b0_data, np.eye(4)).to_filename(b0_file)

    result = MaskWithinDWIFieldOfView(in_mask=str(mask_file), b0_image=str(b0_file)).run(
        cwd=str(tmp_path)
    )

    refined_mask = nb.load(result.outputs.out_mask).get_fdata()

    assert refined_mask[0, 0, 0] == 1
    assert refined_mask[0, 0, 1] == 0
    assert refined_mask.sum() == 1