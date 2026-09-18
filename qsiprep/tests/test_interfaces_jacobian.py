"""Unit tests for the Jacobian weighting helpers.

Pure-Python behaviour -- geometry validation, dedup keying, map arithmetic and
the positivity guard -- is tested unconditionally. Tests that shell out to ANTs
are guarded with ``shutil.which`` and skip when the binaries are absent. They
are not permanently offline: CircleCI's ``unit_tests`` job runs pytest inside
the ``pennlinc/qsiprep:test`` image, which ships ANTs.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.jacobian import (
    check_weight_map,
    compose_fields,
    jacobian_determinant,
    multiply_maps,
    validate_field_geometry,
    weight_key,
)
from qsiprep.tests.gradient_fixtures import write_itk_field


def _write_map(path, value, shape=(8, 8, 8), affine=None):
    affine = np.eye(4) if affine is None else affine
    data = np.full(shape, value, dtype='float32')
    nb.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def _write_linear_field(path, matrix, shape=(8, 8, 8)):
    """Write an ITK displacement field encoding phi(x) = matrix @ x."""
    coords = np.stack(
        np.meshgrid(*[np.arange(n, dtype='float32') for n in shape], indexing='ij'),
        axis=-1,
    )
    mapped = coords @ np.asarray(matrix, dtype='float32').T
    data = (mapped - coords).reshape(shape + (1, 3)).astype('float32')
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


# --- pure helpers ----------------------------------------------------------


def test_weight_key_dedups_identical_pairs():
    assert weight_key('g.nii.gz', 'f.nii.gz') == weight_key('g.nii.gz', 'f.nii.gz')


def test_weight_key_separates_different_fieldwarps():
    assert weight_key('g.nii.gz', 'up.nii.gz') != weight_key('g.nii.gz', 'down.nii.gz')


def test_weight_key_of_nothing_is_falsy():
    """No gradwarp and no fieldwarp means a unity weight, not a cache entry."""
    assert not weight_key(None, None)


def test_multiply_maps_multiplies_voxelwise(tmp_path):
    a = _write_map(tmp_path / 'a.nii.gz', 2.0)
    b = _write_map(tmp_path / 'b.nii.gz', 3.0)
    out = multiply_maps([a, b], str(tmp_path / 'out.nii.gz'))
    assert np.asanyarray(nb.load(out).dataobj)[0, 0, 0] == pytest.approx(6.0)


def test_multiply_maps_with_one_input_returns_it_unchanged(tmp_path):
    a = _write_map(tmp_path / 'a.nii.gz', 2.0)
    assert multiply_maps([a], str(tmp_path / 'out.nii.gz')) == a


def test_validate_field_geometry_accepts_matching_grid(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(8, 8, 8))
    validate_field_geometry(str(field), reference)


def test_validate_field_geometry_rejects_wrong_shape(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(6, 6, 6))
    with pytest.raises(ValueError, match='shape'):
        validate_field_geometry(str(field), reference)


def test_validate_field_geometry_rejects_wrong_affine(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0, affine=np.diag([2.0, 2, 2, 1]))
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(8, 8, 8))
    with pytest.raises(ValueError, match='affine'):
        validate_field_geometry(str(field), reference)


def test_check_weight_map_rejects_nonpositive_in_mask(tmp_path):
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[4, 4, 4] = -0.5
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    with pytest.raises(ValueError, match='non-positive'):
        check_weight_map(weights, mask)


def test_check_weight_map_rejects_nonfinite(tmp_path):
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[0, 0, 0] = np.nan
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    with pytest.raises(ValueError, match='finite'):
        check_weight_map(weights, mask)


def test_check_weight_map_warns_on_far_from_unity_median(tmp_path, caplog):
    weights = _write_map(tmp_path / 'w.nii.gz', 4.0)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    check_weight_map(weights, mask)
    assert 'median' in caplog.text


def test_check_weight_map_ignores_nonpositive_outside_mask(tmp_path):
    """Determinants outside the brain are not the guard's business."""
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[0, 0, 0] = -1.0
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask_data = np.zeros((8, 8, 8), dtype='uint8')
    mask_data[2:6, 2:6, 2:6] = 1
    mask = str(tmp_path / 'm.nii.gz')
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask)
    check_weight_map(weights, mask)


# --- ANTs-backed helpers ---------------------------------------------------


def test_jacobian_determinant_of_translation_is_unity(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    shape = (8, 8, 8)
    data = np.zeros(shape + (1, 3), dtype='float32')
    data[..., 0, 0] = 3.0
    field = tmp_path / 'translation.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(field))

    out = jacobian_determinant(str(field), str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=1e-3)


def test_jacobian_determinant_of_anisotropic_scaling(tmp_path):
    """phi(x) = diag(2, 1, 1) @ x has det = 2 everywhere."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    field = _write_linear_field(tmp_path / 'scale.nii.gz', np.diag([2.0, 1.0, 1.0]))

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 2.0, atol=5e-2)


def test_jacobian_determinant_of_shear_is_unity(tmp_path):
    """A shear moves voxels without changing volume, so det = 1."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    field = _write_linear_field(tmp_path / 'shear.nii.gz', shear)

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=5e-2)


def test_jacobian_determinant_rejects_a_fold_in_mask(tmp_path):
    """A folded warp has a negative determinant and must not be absolute-valued.

    Without this case the three positive-determinant tests above all pass while
    a fold at -0.5 silently becomes a weight of 0.5.
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    # phi(x) = diag(-1, 1, 1) @ x is an orientation reversal: det = -1.
    field = _write_linear_field(tmp_path / 'fold.nii.gz', np.diag([-1.0, 1.0, 1.0]))
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0)

    with pytest.raises(ValueError, match='non-positive'):
        jacobian_determinant(field, str(tmp_path / 'det.nii.gz'), mask_path=mask)


def test_jacobian_determinant_tolerates_edge_negatives_outside_mask(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    field = _write_linear_field(tmp_path / 'scale.nii.gz', np.diag([2.0, 1.0, 1.0]))
    mask_data = np.zeros((8, 8, 8), dtype='uint8')
    mask_data[3:5, 3:5, 3:5] = 1
    mask = str(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask)

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'), mask_path=mask)
    assert (np.asanyarray(nb.load(out).dataobj) >= 0).all()


def test_validate_scalar_geometry_rejects_a_wrong_grid(tmp_path):
    from qsiprep.interfaces.jacobian import validate_scalar_geometry

    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    other = _write_map(tmp_path / 'other.nii.gz', 1.0, shape=(6, 6, 6))
    with pytest.raises(ValueError, match='shape'):
        validate_scalar_geometry(other, reference)


def test_compose_fields_returns_a_single_field(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    first = write_itk_field(tmp_path / 'a.nii.gz', amplitude=0.4)
    second = write_itk_field(tmp_path / 'b.nii.gz', amplitude=0.2)

    out = compose_fields([str(first), str(second)], reference, str(tmp_path / 'c.nii.gz'))
    composed = nb.load(out)
    assert composed.shape[:3] == (8, 8, 8)
    assert np.isfinite(np.asanyarray(composed.dataobj)).all()
