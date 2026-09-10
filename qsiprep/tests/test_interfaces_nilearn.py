"""Tests for the qsiprep.interfaces.nilearn module."""

import nibabel as nb
import numpy as np

from qsiprep.interfaces.nilearn import MaskWithinDWIFieldOfView


def _count_qc_risky_voxels(dwi, mask, bvals, b0_thr=50.0, eps=1e-8):
    """Mirror the QC script criterion for risky voxels inside mask."""
    mask_bool = mask > 0
    b0_idx = bvals <= b0_thr
    b0_data = dwi[..., b0_idx]
    b0_masked = b0_data[mask_bool, :]

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_b0 = np.nanmean(np.where(np.isfinite(b0_masked), b0_masked, np.nan), axis=1)

    risky = (~np.isfinite(mean_b0)) | (np.isfinite(mean_b0) & (mean_b0 <= eps))
    return int(np.count_nonzero(risky))


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



def test_mask_within_dwi_field_of_view_uses_full_dwi_support(tmp_path):
    """All-zero resampled DWI voxels should be excluded from the final mask."""
    mask_file = tmp_path / 'mask_fullsupport.nii.gz'
    b0_file = tmp_path / 'b0_fullsupport.nii.gz'
    dwi_file = tmp_path / 'dwi_fullsupport.nii.gz'

    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 1
    mask_data[0, 0, 1] = 1

    # b0 support alone would keep both voxels.
    b0_data = np.zeros((2, 2, 2), dtype=np.float32)
    b0_data[0, 0, 0] = 100.0
    b0_data[0, 0, 1] = 100.0

    # Full DWI support has one all-zero voxel across all volumes.
    dwi_data = np.zeros((2, 2, 2, 3), dtype=np.float32)
    dwi_data[0, 0, 0, :] = [100.0, 50.0, 25.0]
    dwi_data[0, 0, 1, :] = [0.0, 0.0, 0.0]

    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)
    nb.Nifti1Image(b0_data, np.eye(4)).to_filename(b0_file)
    nb.Nifti1Image(dwi_data, np.eye(4)).to_filename(dwi_file)

    result = MaskWithinDWIFieldOfView(
        in_mask=str(mask_file),
        b0_image=str(b0_file),
        dwi_series=str(dwi_file),
    ).run(cwd=str(tmp_path))

    refined_mask = nb.load(result.outputs.out_mask).get_fdata()

    assert refined_mask[0, 0, 0] == 1
    assert refined_mask[0, 0, 1] == 0
    assert refined_mask.sum() == 1


def test_mask_within_dwi_field_of_view_removes_nonfinite_dwi_voxels(tmp_path):
    """Voxels with non-finite DWI values should be excluded from the final mask."""
    mask_file = tmp_path / 'mask_nonfinite.nii.gz'
    b0_file = tmp_path / 'b0_nonfinite.nii.gz'
    dwi_file = tmp_path / 'dwi_nonfinite.nii.gz'

    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 1
    mask_data[0, 0, 1] = 1

    b0_data = np.zeros((2, 2, 2), dtype=np.float32)
    b0_data[0, 0, 0] = 100.0
    b0_data[0, 0, 1] = 100.0

    dwi_data = np.zeros((2, 2, 2, 3), dtype=np.float32)
    dwi_data[0, 0, 0, :] = [80.0, 40.0, 20.0]
    dwi_data[0, 0, 1, :] = [60.0, np.nan, 30.0]

    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)
    nb.Nifti1Image(b0_data, np.eye(4)).to_filename(b0_file)
    nb.Nifti1Image(dwi_data, np.eye(4)).to_filename(dwi_file)

    result = MaskWithinDWIFieldOfView(
        in_mask=str(mask_file),
        b0_image=str(b0_file),
        dwi_series=str(dwi_file),
    ).run(cwd=str(tmp_path))

    refined_mask = nb.load(result.outputs.out_mask).get_fdata()

    assert refined_mask[0, 0, 0] == 1
    assert refined_mask[0, 0, 1] == 0
    assert refined_mask.sum() == 1


def test_mask_refinement_eliminates_qc_script_risky_voxels(tmp_path):
    """Refined mask should have zero risky voxels under QC-script criterion."""
    mask_file = tmp_path / 'mask_qc.nii.gz'
    b0_file = tmp_path / 'b0_qc.nii.gz'
    dwi_file = tmp_path / 'dwi_qc.nii.gz'

    mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 1
    mask_data[0, 0, 1] = 1

    b0_data = np.zeros((2, 2, 2), dtype=np.float32)
    b0_data[0, 0, 0] = 100.0
    b0_data[0, 0, 1] = 0.0

    dwi_data = np.zeros((2, 2, 2, 4), dtype=np.float32)
    dwi_data[0, 0, 0, :] = [100.0, 90.0, 80.0, 70.0]
    dwi_data[0, 0, 1, :] = [0.0, 0.0, 0.0, 0.0]
    bvals = np.array([0.0, 10.0, 1000.0, 1000.0], dtype=float)

    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)
    nb.Nifti1Image(b0_data, np.eye(4)).to_filename(b0_file)
    nb.Nifti1Image(dwi_data, np.eye(4)).to_filename(dwi_file)

    result = MaskWithinDWIFieldOfView(
        in_mask=str(mask_file),
        b0_image=str(b0_file),
        dwi_series=str(dwi_file),
    ).run(cwd=str(tmp_path))

    refined_mask = nb.load(result.outputs.out_mask).get_fdata().astype(np.uint8)
    assert _count_qc_risky_voxels(dwi_data, refined_mask, bvals) == 0