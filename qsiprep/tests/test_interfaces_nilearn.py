"""Tests for the qsiprep.interfaces.nilearn module."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.nilearn import Merge, _grid_info, concat_imgs_on_first_grid

# The PE+ affine from a reported raw_rpe_concat crash, and the PE- affine, which differs
# from it only by float round-off in two rotation entries (largest difference 5.4e-8).
REF_AFFINE = np.array(
    [
        [-1.69622529e00, 1.12663135e-01, 1.12777678e-02, 1.17360825e02],
        [1.13213494e-01, 1.69011879e00, 1.43811643e-01, -1.21094513e02],
        [1.68148463e-03, -1.44243374e-01, 1.69386888e00, -4.14808731e01],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
ROUNDED_AFFINE = REF_AFFINE.copy()
ROUNDED_AFFINE[0, 2] = 1.12778209e-02
ROUNDED_AFFINE[2, 0] = 1.68153888e-03

# Origin jitter that qsiplan still calls 'match' (its bound is 0.05 mm)
JITTERED_AFFINE = REF_AFFINE.copy()
JITTERED_AFFINE[0, 3] += 0.03

SHIFTED_AFFINE = REF_AFFINE.copy()
SHIFTED_AFFINE[0, 3] += 1.0

_ANGLE = np.deg2rad(2.0)
ROTATED_AFFINE = REF_AFFINE.copy()
ROTATED_AFFINE[:3, :3] = (
    np.array(
        [
            [np.cos(_ANGLE), -np.sin(_ANGLE), 0.0],
            [np.sin(_ANGLE), np.cos(_ANGLE), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    @ REF_AFFINE[:3, :3]
)

SHAPE = (6, 6, 4)


def _write(path, affine, n_volumes, value, shape=SHAPE):
    data = np.full((*shape, n_volumes), value, dtype=np.float32)
    nb.Nifti1Image(data, affine).to_filename(path)
    return str(path)


def _relation(first, second):
    return _grid_info(nb.load(first)).compare(_grid_info(nb.load(second)))


def test_rounding_affines_fail_nilearn_tolerance():
    """Test that the reported affines are a pair np.allclose rejects."""
    assert not np.allclose(REF_AFFINE, ROUNDED_AFFINE)


@pytest.mark.parametrize(
    ('affine', 'relation'),
    [
        (ROUNDED_AFFINE, 'match'),
        (JITTERED_AFFINE, 'match'),
        (SHIFTED_AFFINE, 'shifted'),
        (ROTATED_AFFINE, 'oblique'),
    ],
    ids=['rounding', 'jitter', 'shifted', 'oblique'],
)
def test_merge_stacks_without_resampling(tmp_path, affine, relation):
    """Test that every series the plan allows on the first grid is stacked as is.

    Each takes the first series' affine and none is interpolated, so a field-of-view
    offset (shifted, or oblique under --ignore fov) is left to head-motion correction.
    """
    plus = _write(tmp_path / 'plus.nii.gz', REF_AFFINE, 2, 1.0)
    minus = _write(tmp_path / 'minus.nii.gz', affine, 3, 2.0)
    assert _relation(plus, minus) == relation
    # Still a mismatch for nilearn after the float32 round trip through the header
    assert not np.allclose(nb.load(plus).affine, nb.load(minus).affine)

    result = Merge(in_files=[plus, minus]).run(cwd=str(tmp_path))

    merged = nb.load(result.outputs.out_file)
    assert merged.shape == (*SHAPE, 5)
    np.testing.assert_allclose(merged.affine, nb.load(plus).affine)
    data = merged.get_fdata()
    np.testing.assert_array_equal(data[..., :2], 1.0)
    np.testing.assert_array_equal(data[..., 2:], 2.0)


def test_merge_rejects_a_different_voxel_grid(tmp_path):
    """Test that series on different matrices are an error, as they are when planning."""
    plus = _write(tmp_path / 'plus.nii.gz', REF_AFFINE, 2, 1.0)
    minus = _write(tmp_path / 'minus.nii.gz', REF_AFFINE, 2, 2.0, shape=(6, 6, 5))
    assert _relation(plus, minus) == 'grid'

    with pytest.raises(ValueError, match='minus.nii.gz .* cannot be stacked'):
        Merge(in_files=[plus, minus]).run(cwd=str(tmp_path))


def test_concat_names_in_memory_images():
    """Test that in-memory images (as MergeDWIs passes) are named by position."""
    first = nb.Nifti1Image(np.zeros((*SHAPE, 1), dtype=np.float32), REF_AFFINE)
    second = nb.Nifti1Image(np.zeros((6, 6, 5, 1), dtype=np.float32), REF_AFFINE)

    with pytest.raises(ValueError, match=r'image #1 .* grid of image #0'):
        concat_imgs_on_first_grid([first, second])
