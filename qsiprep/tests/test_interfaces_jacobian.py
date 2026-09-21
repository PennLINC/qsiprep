"""Unit tests for the Jacobian weighting helpers.

Pure-Python behaviour -- geometry validation, dedup keying, map arithmetic and
the positivity guard -- is tested unconditionally. Tests that shell out to ANTs
are guarded with ``shutil.which`` and skip when the binaries are absent. They
are not permanently offline: CircleCI's ``unit_tests`` job runs pytest inside
the ``pennlinc/qsiprep:test`` image, which ships ANTs.
"""

import shutil
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.jacobian import (
    check_weight_map,
    compose_fields,
    jacobian_determinant,
    multiply_maps,
    resample_like,
    validate_field_geometry,
    validate_scalar_geometry,
    weight_key,
)
from qsiprep.tests.gradient_fixtures import write_itk_field

#: The real DRBUDDI shape/affine mismatch this module's C1 fix addresses,
#: read directly off cached DRBUDDI outputs for a real dataset (see the
#: fix report): the native/input grid (e.g. ``b0_up.nii``) is (60, 60, 37)
#: with one origin, while DRBUDDI's own output grid (``b0_corrected_final.nii``,
#: ``deformation_FINV.nii.gz``) is (76, 76, 37) with a shifted origin -- same
#: voxel size, same orientation, genuinely different lattice, still the same
#: physical head.
NATIVE_SHAPE = (60, 60, 37)
NATIVE_AFFINE = np.array(
    [
        [-4.0, 0.0, 0.0, 119.0],
        [0.0, 4.0, 0.0, -101.522],
        [0.0, 0.0, 4.0, -60.338],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
DRBUDDI_SHAPE = (76, 76, 37)
DRBUDDI_AFFINE = np.array(
    [
        [-4.0, 0.0, 0.0, 151.0],
        [0.0, 4.0, 0.0, -133.522],
        [0.0, 0.0, 4.0, -60.338],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write_field(path, shape, affine, amplitude=0.05):
    """A small, smooth ITK displacement field on an arbitrary shape/affine."""
    path = Path(path)
    grid = np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing='ij')
    data = np.zeros(shape + (1, 3), dtype='float32')
    for component in range(3):
        data[..., 0, component] = amplitude * grid[component] ** 2
    # RAS -> LPS negation for x/y, matching write_itk_field's convention.
    data[..., 0, 0] *= -1
    data[..., 0, 1] *= -1
    img = nb.Nifti1Image(data, affine)
    img.header.set_intent(1007)
    img.to_filename(str(path))
    return str(path)


def _write_map(path, value, shape=(8, 8, 8), affine=None):
    affine = np.eye(4) if affine is None else affine
    data = np.full(shape, value, dtype='float32')
    nb.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def _write_linear_field(path, matrix, shape=(8, 8, 8), set_intent=True):
    """Write an ITK displacement field encoding phi(x) = matrix @ x.

    The x and y displacement components are negated before writing, matching
    the RAS-to-LPS convention ITK applies to any NIfTI it recognizes as a
    displacement field (i.e. tagged with the ``NIFTI_INTENT_VECTOR`` intent
    code -- see below). This is the same convention already established
    elsewhere in this codebase for hand-authored vector fields; compare
    ``FUGUEvsm2ANTSwarp`` and ``_fix_hdr`` in ``qsiprep/workflows/fieldmap/
    pepolar.py``, both of which negate components with a "ITK is LPS"
    comment. Without it, a diagonal field with a nonzero x or y scaling term
    reads back with the wrong sign once the intent code is set (measured: a
    2x scale came back as 0, not 2, when the naive un-negated data was
    intent-tagged).

    ``set_intent`` defaults to True, tagging the header with intent code 1007
    (``NIFTI_INTENT_VECTOR``), which is what ``CreateJacobianDeterminantImage``
    needs to apply the RAS-to-LPS conversion above; without it, ANTs instead
    treats the file as a generic vector array and silently folds
    cross-derivatives of the first vector component into the diagonal (see
    ``test_jacobian_determinant_normalizes_missing_vector_intent``). Pass
    ``set_intent=False`` to reproduce a header as a real producer -- e.g.
    ``MaskWarpDimensions``, which forwards its input header verbatim -- might
    emit it, while still encoding correctly-signed data as that same real
    producer would.
    """
    coords = np.stack(
        np.meshgrid(*[np.arange(n, dtype='float32') for n in shape], indexing='ij'),
        axis=-1,
    )
    mapped = coords @ np.asarray(matrix, dtype='float32').T
    displacement = (mapped - coords).astype('float32')
    displacement[..., 0] *= -1
    displacement[..., 1] *= -1
    data = displacement.reshape(shape + (1, 3)).astype('float32')
    img = nb.Nifti1Image(data, np.eye(4))
    if set_intent:
        img.header.set_intent(1007)
    img.to_filename(str(path))
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


def test_validate_field_geometry_accepts_a_differently_sampled_overlapping_field(tmp_path):
    """C1: a different lattice is not an error -- ANTs composes in physical space.

    Reproduces the real DRBUDDI shape/affine mismatch (see ``NATIVE_*`` /
    ``DRBUDDI_*`` above): before the fix, this raised on every DRBUDDI run.
    """
    reference = _write_map(
        tmp_path / 'ref.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE
    )
    field = _write_field(tmp_path / 'field.nii.gz', NATIVE_SHAPE, NATIVE_AFFINE)
    validate_field_geometry(str(field), reference)


def test_validate_field_geometry_rejects_wrong_component_count(tmp_path):
    """The one thing ANTs' physical-space composition cannot rescue."""
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    not_a_field = _write_map(tmp_path / 'notfield.nii.gz', 1.0, shape=(8, 8, 8, 1, 2))
    with pytest.raises(ValueError, match='vector components'):
        validate_field_geometry(str(not_a_field), reference)


def test_validate_field_geometry_rejects_a_3d_scalar_image_with_a_trailing_axis_of_3(tmp_path):
    """F3 regression: ``field.shape[-1]`` alone is not enough to detect a field.

    A plain 3D scalar image whose last spatial dimension happens to equal 3
    used to pass this guard (the previous check read ``field.shape[-1]`` as a
    component count for any non-5D image), letting a scalar map through
    composition as if it were a displacement field. Only the two real
    layouts -- 4D (X, Y, Z, 3) and 5D (X, Y, Z, 1, 3) -- are accepted.
    """
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    scalar = _write_map(tmp_path / 'scalar.nii.gz', 1.0, shape=(4, 4, 3))
    with pytest.raises(ValueError, match=r'\(4, 4, 3\)'):
        validate_field_geometry(str(scalar), reference)


def test_validate_field_geometry_rejects_a_disjoint_world_frame(tmp_path):
    """A field in a genuinely different coordinate domain still must raise."""
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0, shape=(8, 8, 8))
    far_affine = np.eye(4)
    far_affine[:3, 3] = 10_000.0
    field = _write_field(tmp_path / 'field.nii.gz', (8, 8, 8), far_affine)
    with pytest.raises(ValueError, match='overlapping world space'):
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


def test_check_weight_map_rejects_an_empty_mask(tmp_path):
    """C3: an all-zero (or fully-off-grid) resampled mask must raise, not
    silently skip the positivity/median guard.

    ``inside.size == 0`` previously short-circuited both checks below it via
    ``if inside.size and ...`` -- exactly the situation the guard exists for
    (something is very likely wrong: a world-frame mismatch between the
    weight map and the mask) disabled it instead of flagging it. The error
    names both images so the mismatch can actually be found.
    """
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    mask = _write_map(tmp_path / 'm.nii.gz', 0.0)
    with pytest.raises(ValueError, match='world.frame') as excinfo:
        check_weight_map(weights, mask)
    assert weights in str(excinfo.value)
    assert mask in str(excinfo.value)


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


def test_jacobian_determinant_normalizes_missing_vector_intent(tmp_path):
    """A field without the vector intent code must not silently corrupt weights.

    ``CreateJacobianDeterminantImage`` does not require the NIfTI
    ``NIFTI_INTENT_VECTOR`` (1007) intent code to run, but without it, it
    silently folds cross-derivatives of the first vector component into the
    diagonal: this shear, with det = 1, measures back at 0.7 when the intent
    code is missing (a different wrong number than the 1.3 seen for a
    differently-signed field in ``test_jacobian_determinant_of_shear_is_unity``
    -- the exact value depends on the field's own sign convention, which is
    the point: it is never flagged as wrong). It is positive and plausible, so
    nothing downstream -- the positivity guard included -- can catch it. A
    real producer can omit the intent code entirely: ``MaskWarpDimensions``
    forwards its input header verbatim, and a user-supplied ``--gradient-file``
    field is never checked. This test reproduces exactly that: a
    correctly-signed shear field written with no intent code must still come
    back at 1.0, proving ``jacobian_determinant`` normalizes the header rather
    than trusting it.
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    field = _write_linear_field(tmp_path / 'shear_no_intent.nii.gz', shear, set_intent=False)
    assert nb.load(field).header.get_intent('code')[0] != 1007

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


def test_validate_scalar_geometry_accepts_a_differently_sampled_overlapping_map(tmp_path):
    """C1: same relaxation as validate_field_geometry, for the mask/EC inputs."""
    reference = _write_map(
        tmp_path / 'ref.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE
    )
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    validate_scalar_geometry(mask, reference)


def test_validate_scalar_geometry_rejects_a_non_3d_map(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    not_scalar = _write_map(tmp_path / 'other.nii.gz', 1.0, shape=(8, 8, 8, 4))
    with pytest.raises(ValueError, match='3D scalar'):
        validate_scalar_geometry(not_scalar, reference)


def test_validate_scalar_geometry_rejects_a_singleton_trailing_dimension(tmp_path):
    """C4: a (X, Y, Z, 1) scalar map used to pass this guard, then blow up
    downstream -- ``check_weight_map``'s ``weights[mask]`` indexes a bare 3D
    weight array with a 4D boolean mask, raising a dimensionality error deep
    inside the guard rather than a clear message here. Nothing in this
    pipeline squeezes a validated scalar map before using it as a plain
    ndarray (``check_weight_map``, ``multiply_maps``), so a trailing singleton
    is rejected outright rather than accepted and silently mishandled later.
    """
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    singleton_4d = _write_map(tmp_path / 'singleton.nii.gz', 1.0, shape=(8, 8, 8, 1))
    with pytest.raises(ValueError, match='3D scalar'):
        validate_scalar_geometry(singleton_4d, reference)


def test_validate_scalar_geometry_rejects_a_disjoint_world_frame(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0, shape=(8, 8, 8))
    far_affine = np.eye(4)
    far_affine[:3, 3] = 10_000.0
    other = _write_map(tmp_path / 'other.nii.gz', 1.0, shape=(8, 8, 8), affine=far_affine)
    with pytest.raises(ValueError, match='overlapping world space'):
        validate_scalar_geometry(other, reference)


def test_resample_like_is_a_noop_when_grids_already_match(tmp_path):
    like = _write_map(tmp_path / 'like.nii.gz', 1.0)
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0)
    assert resample_like(mask, like, str(tmp_path / 'out.nii.gz')) == mask


def test_resample_like_noop_path_ignores_fill_value(tmp_path):
    """C1: ``force_resample=True`` is only ever passed on the branch that
    actually calls into nilearn. The matching-lattice early return is ours,
    not nilearn's, so a non-default ``fill_value`` must not disturb it --
    the common (already-matching) case must still cost no I/O and return the
    source path byte-identical, regardless of ``fill_value``.
    """
    like = _write_map(tmp_path / 'like.nii.gz', 1.0)
    source = _write_map(tmp_path / 'source.nii.gz', 1.0)
    out_path = str(tmp_path / 'out.nii.gz')
    result = resample_like(source, like, out_path, fill_value=1.0)
    assert result == source
    assert not Path(out_path).exists()


def test_resample_like_resamples_onto_the_target_grid(tmp_path):
    like = _write_map(tmp_path / 'like.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE)
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    out = resample_like(mask, like, str(tmp_path / 'out.nii.gz'))
    resampled = nb.load(out)
    assert resampled.shape[:3] == DRBUDDI_SHAPE
    assert np.allclose(resampled.affine, DRBUDDI_AFFINE)


def test_resample_like_default_fill_value_is_zero_for_masks(tmp_path):
    """C1: the default preserves current mask behaviour -- absent outside its
    own FOV, i.e. zero-filled -- only a determinant factor (``multiply_maps``)
    opts into ``fill_value=1.0``.
    """
    like = _write_map(tmp_path / 'like.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE)
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    out = resample_like(mask, like, str(tmp_path / 'out.nii.gz'))
    resampled = np.asanyarray(nb.load(out).dataobj)
    assert (resampled[0, 0, 0] == 0).all()


def test_resample_like_fill_value_reaches_outside_the_source_fov(tmp_path):
    """C1: with ``fill_value=1.0`` and the exact real-world lattice mismatch
    (an axis-aligned, whole-voxel padding offset -- DRBUDDI's own padding of
    DIFFPREP's native grid), voxels outside the source's field of view come
    back as the fill value, not zero. This is the mechanism the previous fix
    attempt (passing ``fill_value`` alone, without ``force_resample=True``)
    was verified NOT to guarantee -- nilearn's padding fast path ignores
    ``fill_value`` on exactly this kind of lattice relationship unless
    ``force_resample=True`` is also passed.
    """
    like = _write_map(tmp_path / 'like.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE)
    source = _write_map(tmp_path / 'source.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    out = resample_like(
        source, like, str(tmp_path / 'out.nii.gz'), interpolation='linear', fill_value=1.0
    )
    resampled = np.asanyarray(nb.load(out).dataobj)
    # A corner of the DRBUDDI-sized grid that is genuinely outside the
    # smaller native-grid footprint.
    assert resampled[0, 0, 0] == pytest.approx(1.0)
    assert not np.any(resampled == 0)


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


# --- ComposeJacobianWeights -------------------------------------------------


from nipype.interfaces.base import isdefined

from qsiprep.interfaces.jacobian import ComposeJacobianWeights


def _dwi_volumes(tmp_path, count):
    return [_write_map(tmp_path / f'dwi{i}.nii.gz', 1.0) for i in range(count)]


def test_compose_weights_with_no_fields_is_undefined(tmp_path):
    """Nothing to modulate means no weight, not a map of ones."""
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
    )
    result = interface.run(cwd=str(tmp_path))
    assert not isdefined(result.outputs.jacobian_weight_images)


def test_compose_weights_with_no_fields_ignores_a_mismatched_mask(tmp_path):
    """C2: the early no-op return must come before the mask is even looked at.

    A run with nothing to modulate must not be killed by a mask/reference
    mismatch it never needed to resolve -- even a mask in a totally disjoint
    world frame, which *would* raise if validated.
    """
    far_affine = np.eye(4)
    far_affine[:3, 3] = 10_000.0
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0, affine=far_affine),
    )
    result = interface.run(cwd=str(tmp_path))
    assert not isdefined(result.outputs.jacobian_weight_images)


def test_compose_weights_returns_one_path_per_volume(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        gradwarp_field=[str(write_itk_field(tmp_path / 'g.nii.gz'))],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    assert len(weights) == 4


def test_compose_weights_dedups_a_single_shared_field(tmp_path):
    """One gradwarp field for the whole run costs one determinant, not N."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        gradwarp_field=[str(write_itk_field(tmp_path / 'g.nii.gz'))],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    assert len(set(weights)) == 1


def test_compose_weights_keeps_two_blip_directions_distinct(tmp_path):
    """DRBUDDI rpe_series has one warp per blip direction, so two maps."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    up = str(write_itk_field(tmp_path / 'up.nii.gz', amplitude=0.4))
    down = str(write_itk_field(tmp_path / 'down.nii.gz', amplitude=0.2))
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[up, up, down, down],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    assert len(set(weights)) == 2
    assert weights[0] == weights[1]
    assert weights[2] == weights[3]


def test_compose_weights_broadcasts_a_single_fieldwarp(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[str(write_itk_field(tmp_path / 'f.nii.gz'))],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    assert len(weights) == 3
    assert len(set(weights)) == 1


def test_compose_weights_rejects_a_disjoint_field(tmp_path):
    """A field in a genuinely unrelated coordinate domain is still an error."""
    far_affine = np.eye(4)
    far_affine[:3, 3] = 10_000.0
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0, shape=(8, 8, 8)),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[str(_write_field(tmp_path / 'f.nii.gz', (8, 8, 8), far_affine))],
    )
    with pytest.raises(ValueError, match='overlapping world space'):
        interface.run(cwd=str(tmp_path))


def test_compose_weights_succeeds_on_a_native_mask_against_a_drbuddi_grid_reference(tmp_path):
    """C1 regression: the guard used to break every DRBUDDI run.

    Reproduces the real shape/affine mismatch measured on cached DRBUDDI
    outputs (see ``NATIVE_*``/``DRBUDDI_*`` above): ``b0_ref_image`` is
    DRBUDDI's own output grid (``fsl.py``/``diffprep.py`` override it from
    ``drbuddi_wf.outputnode.b0_ref``), the SDC warp DRBUDDI produces is on
    that same grid, and the brain mask is still on the native input grid.
    Before the fix, ``validate_scalar_geometry(mask, reference)`` raised here
    unconditionally. Now it must succeed and produce a sane (near-unity, in
    this near-identity synthetic case) determinant.
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')

    reference = _write_map(
        tmp_path / 'ref.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE
    )
    # DRBUDDI's own SDC warp: on DRBUDDI's own output grid, like the real
    # deformation_FINV.nii.gz.
    fieldwarp = _write_field(
        tmp_path / 'drbuddi_warp.nii.gz', DRBUDDI_SHAPE, DRBUDDI_AFFINE, amplitude=0.02
    )
    # The brain mask: still on the native (pre-DRBUDDI) input grid, like the
    # real b0_up.nii, genuinely different from the reference above.
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)

    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=reference,
        mask=mask,
        fieldwarps=[fieldwarp],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images

    assert len(weights) == 2
    for weight_path in set(weights):
        determinant = np.asanyarray(nb.load(weight_path).dataobj)
        assert np.isfinite(determinant).all()
        assert (determinant > 0).all()
        # A small, smooth synthetic warp should not move the determinant far
        # from unity -- this is the "sane" check the task asked for.
        assert 0.5 < float(np.median(determinant)) < 2.0


def test_compose_weights_reconciles_ec_jacobian_lattice_against_sdc_determinant(tmp_path):
    """F1 regression: the SDC determinant and the EC Jacobian on genuinely
    different lattices used to crash ``multiply_maps`` with a ``nilearn``
    ``ValueError`` (mismatched shape/affine) before either factor ever
    reached ``check_weight_map``.

    Reproduces the real ``--hmc-method tortoise --sdc-method drbuddi``
    ``correction_mode='quadratic'`` shapes: the SDC determinant lives on
    DRBUDDI's own output grid (``DRBUDDI_SHAPE``, 76x76x37 in the real run
    this was measured on), the per-volume EC Jacobian on DIFFPREP's native
    input grid (``NATIVE_SHAPE``, 60x60x37). No CI marker catches this
    because every TORTOISE marker passes ``--sloppy``, which forces
    ``correction_mode='motion'`` (no EC Jacobian at all).
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')

    reference = _write_map(
        tmp_path / 'ref.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE
    )
    fieldwarp = _write_field(
        tmp_path / 'drbuddi_warp.nii.gz', DRBUDDI_SHAPE, DRBUDDI_AFFINE, amplitude=0.02
    )
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    ec_images = [
        _write_map(
            tmp_path / f'ec{i}.nii.gz', 1.0 + 0.02 * i, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE
        )
        for i in range(2)
    ]

    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=reference,
        mask=mask,
        fieldwarps=[fieldwarp],
        ec_jacobian_images=ec_images,
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images

    assert len(weights) == 2
    for i, weight_path in enumerate(weights):
        img = nb.load(weight_path)
        # Reconciled onto b0_ref_image's grid -- see multiply_maps' like_path.
        assert img.shape[:3] == DRBUDDI_SHAPE
        assert np.allclose(img.affine, DRBUDDI_AFFINE)
        determinant = np.asanyarray(img.dataobj)
        # Positivity is only a contract inside the brain mask (that is what
        # check_weight_map enforces), but the native EC Jacobian's own
        # footprint does not cover the whole of DRBUDDI's larger output grid.
        # C1: voxels outside that footprint must NOT be zero-filled by
        # resampling -- a determinant factor undefined somewhere means "no
        # volume change known here", whose multiplicative identity is 1.0, not
        # 0. A zero-filled band there would flow through
        # ``_floor_nonpositive_weights`` (floor to 1e-3) and silently
        # annihilate real DWI signal in exactly the padded region DRBUDDI adds
        # to retain susceptibility-displaced signal.
        mask_on_weight_grid = resample_like(
            mask, weight_path, str(tmp_path / f'mask_check_{i}.nii.gz')
        )
        inmask = np.asanyarray(nb.load(mask_on_weight_grid).dataobj) > 0
        assert np.isfinite(determinant[inmask]).all()
        assert (determinant[inmask] > 0).all()

        outside = ~inmask
        # Sanity check that the native footprint genuinely does not cover the
        # whole DRBUDDI grid -- otherwise this test would vacuously pass.
        assert outside.any()
        assert np.isfinite(determinant[outside]).all()
        # atol=1e-2 comfortably separates "the multiplicative identity" from
        # "zero-filled" (the SDC determinant factor itself is not bit-exact
        # unity at its own edges -- ANTs' one-sided edge differences -- so a
        # tight tolerance around 1.0 would be the wrong thing to assert here).
        assert np.allclose(determinant[outside], 1.0, atol=1e-2), (
            'Voxels outside the native EC Jacobian footprint must reconcile to '
            'the multiplicative identity (1.0), never be zero-filled.'
        )


def test_compose_weights_reconciles_ec_jacobian_lattice_with_gradwarp(tmp_path):
    """Same F1 regression, with a gradwarp field also in the mix.

    Here the gradwarp+SDC composite is built directly onto ``b0_ref_image``'s
    grid (``compose_fields``' own ``reference_image``), so this exercises the
    "composed factor already matches the target grid, only the EC Jacobian
    needs resampling" path, rather than the lone-SDC-warp path the test above
    covers.
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')

    reference = _write_map(
        tmp_path / 'ref.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE
    )
    fieldwarp = _write_field(
        tmp_path / 'drbuddi_warp.nii.gz', DRBUDDI_SHAPE, DRBUDDI_AFFINE, amplitude=0.02
    )
    gradwarp = _write_field(
        tmp_path / 'gradwarp.nii.gz', NATIVE_SHAPE, NATIVE_AFFINE, amplitude=0.01
    )
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    ec_images = [
        _write_map(
            tmp_path / f'ec{i}.nii.gz', 1.0 + 0.02 * i, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE
        )
        for i in range(2)
    ]

    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=reference,
        mask=mask,
        gradwarp_field=[gradwarp],
        fieldwarps=[fieldwarp],
        ec_jacobian_images=ec_images,
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images

    assert len(weights) == 2
    for i, weight_path in enumerate(weights):
        img = nb.load(weight_path)
        assert img.shape[:3] == DRBUDDI_SHAPE
        determinant = np.asanyarray(img.dataobj)
        mask_on_weight_grid = resample_like(
            mask, weight_path, str(tmp_path / f'mask_check_{i}.nii.gz')
        )
        inmask = np.asanyarray(nb.load(mask_on_weight_grid).dataobj) > 0
        assert np.isfinite(determinant[inmask]).all()
        assert (determinant[inmask] > 0).all()


def test_compose_weights_transports_ec_jacobian_through_the_composed_warp(tmp_path):
    """C1 regression: the EC Jacobian and the SDC determinant live in
    different coordinate DOMAINS, not just different lattices.

    ``multiply_maps`` reconciles differing *sampling grids* via
    ``resample_like`` (lattice realignment in a shared world frame -- see its
    docstring), which is the right tool for ``test_compose_weights_reconciles_
    ec_jacobian_lattice_against_sdc_determinant`` above. It is the wrong tool
    here: the EC Jacobian is evaluated on DIFFPREP's distorted-native grid
    (``diffprep.py:578``'s ``extract_b0s.b0_average``), while the SDC
    determinant is evaluated in undistorted b0-reference space. A b0-reference
    point ``v`` and the native point sampled by plain lattice realignment are
    the *same voxel index after resampling*, not the same physical point --
    the SDC warp is precisely the function that relates them. The correct
    weight is ``J_S(v) * J_E(composed(v))``, where ``composed`` is the same
    gradwarp/SDC warp already used for ``J_S``, not ``J_S(v) * J_E(v))``.

    ``test_compose_weights_reconciles_ec_jacobian_lattice_against_sdc_
    determinant`` and its gradwarp twin cannot catch this: both use spatially
    *constant* EC maps, which are invariant under any transport, correct or
    not. This test uses an EC map that is linear in x (spatially varying) and
    an SDC warp that is an upper-triangular shear+scale mixing x and y
    (non-diagonal, so its action does not commute with -- does not preserve
    the level sets of -- the EC map's x-only dependence). Both are exactly
    representable by linear interpolation/finite differencing, so the
    "coordinate-correct" and "coordinate-wrong" (bug) formulas can be checked
    to near-float32 precision rather than a loose tolerance, and the
    comparison is restricted to an interior sub-box so that ``composed(v)``
    always lands inside the EC image's own footprint (no ``fill_value=1.0``
    contamination at the edges).

    Scale is chosen to be realistic: 1 grid unit stands for 1 mm (affine is
    ``eye(4)``, matching every other linear-field test in this module); the
    shear (5% along x, 8% cross term from y) and EC gradient (0.2%/mm) are
    within the ranges QSIPrep's own guards treat as plausible
    (``MEDIAN_WARN_RANGE`` = 0.9-1.1 for a 40mm-wide test volume).
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')

    shape = (40, 40, 40)
    # Upper-triangular shear+scale: det = 1.05 exactly (product of diagonal),
    # and mixes x and y so a purely-x-dependent EC map does not commute with it.
    matrix = np.array(
        [
            [1.05, 0.08, 0.0],
            [0.00, 1.00, 0.0],
            [0.00, 0.00, 1.0],
        ]
    )
    fieldwarp = _write_linear_field(tmp_path / 'sdc_warp.nii.gz', matrix, shape=shape)

    # EC Jacobian: linear in x only, on the same (native) grid/affine as
    # everything else in this synthetic test.
    xx, _, _ = np.meshgrid(
        np.arange(shape[0], dtype='float32'),
        np.arange(shape[1], dtype='float32'),
        np.arange(shape[2], dtype='float32'),
        indexing='ij',
    )
    ec_data = (1.0 + 0.002 * xx).astype('float32')
    ec_path = str(tmp_path / 'ec0.nii.gz')
    nb.Nifti1Image(ec_data, np.eye(4)).to_filename(ec_path)

    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0, shape=shape)
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=shape)

    interface = ComposeJacobianWeights(
        # Two volumes, not one: nipype's ``OutputMultiObject`` collapses a
        # single-element list to a bare scalar, which would silently turn
        # ``weights[0]`` into the first *character* of the path below.
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=reference,
        mask=mask,
        fieldwarps=[fieldwarp],
        ec_jacobian_images=[ec_path, ec_path],
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    combined = np.asanyarray(nb.load(weights[0]).dataobj)

    # Interior sub-box: composed_x = 1.05x + 0.08y stays well inside [0, 39]
    # for x, y in [5, 25] (min 5.65, max 28.25), clear of both the EC image's
    # own edge and any ANTs one-sided-difference/interpolation boundary noise.
    idx = np.arange(5, 26)
    zidx = np.arange(5, 36)
    xs, ys, zs = np.meshgrid(idx, idx, zidx, indexing='ij')
    got = combined[xs, ys, zs]

    det = 1.05  # det(matrix) -- exact, since the field is exactly linear.
    composed_x = 1.05 * xs + 0.08 * ys
    correct = det * (1.0 + 0.002 * composed_x)
    buggy = det * (1.0 + 0.002 * xs)  # the pre-fix behaviour: J_S(v) * J_E(v)

    max_diff = float(np.max(np.abs(correct - buggy)))
    assert max_diff > 1e-3, (
        f'test setup produced a negligible correct-vs-buggy gap ({max_diff}); '
        'strengthen the field before trusting this regression'
    )

    assert np.allclose(got, correct, atol=2e-3), (
        'Combined weight must equal J_S(v) * J_E(composed(v)) -- the EC '
        'Jacobian transported through the composed gradwarp/SDC warp -- not '
        f'evaluated at v directly. Max |correct - got| = '
        f'{float(np.max(np.abs(correct - got)))}, measured coordinate-domain '
        f'gap max |correct - buggy| = {max_diff}.'
    )


def test_compose_weights_rejects_mismatched_ec_count(tmp_path):
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        ec_jacobian_images=[_write_map(tmp_path / 'ec0.nii.gz', 1.0)],
    )
    with pytest.raises(ValueError, match='eddy-current'):
        interface.run(cwd=str(tmp_path))


def test_compose_weights_applies_ec_only(tmp_path):
    """TORTOISE EC with no gradwarp and no SDC still produces weights."""
    ec = [_write_map(tmp_path / f'ec{i}.nii.gz', 1.0 + 0.1 * i) for i in range(3)]
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        ec_jacobian_images=ec,
    )
    weights = interface.run(cwd=str(tmp_path)).outputs.jacobian_weight_images
    assert len(weights) == 3
    assert len(set(weights)) == 3


# --- OkanQuadraticJacobian ---------------------------------------------------

from qsiprep.interfaces.jacobian import OkanQuadraticJacobian

# An identity Okan row is NOT all zeros. Per the sourced formula (module
# docstring in qsiprep/interfaces/jacobian.py), columns 6-8 are the linear
# coefficients of the eddy-current phase-axis polynomial, and identity means
# the coefficient for the volume's own phase-encode axis is 1 (the other two
# are 0) -- SetIdentity() in TORTOISE's itkOkanQuadraticTransform.hxx sets
# exactly this. Phase=1 ("vertical"/j-axis) is TORTOISE's own default and the
# common AP/PA case, so column 7 (0-indexed) is the one set to 1 here.
_IDENTITY_ROW = [0.0] * 6 + [0.0, 1.0, 0.0] + [0.0] * 15


def _write_transformations(path, rows):
    """Write a DIFFPREP _moteddy_transformations.txt with 24 columns per row."""
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(' '.join(f'{value:.8f}' for value in row) + '\n')
    return str(path)


def test_okan_jacobian_of_identity_parameters_is_unity(tmp_path):
    """Identity parameters (see _IDENTITY_ROW) mean no eddy current, det = 1."""
    transformations = _write_transformations(tmp_path / 'x.txt', [_IDENTITY_ROW] * 3)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='quadratic',
    ).run()

    maps = result.outputs.ec_jacobian_images
    assert len(maps) == 3
    for path in maps:
        interior = np.asanyarray(nb.load(path).dataobj)[2:-2, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 1.0, atol=1e-4)


def test_okan_jacobian_ignores_the_rigid_columns(tmp_path):
    """Columns 0-5 are rigid motion, excluded by the scope policy.

    Two (identical) rows, not one: ``OutputMultiObject`` collapses a
    single-element list to a bare string, and this test wants a real list to
    index into.
    """
    rigid_row = [1.0, 2.0, 3.0, 0.05, 0.05, 0.05] + _IDENTITY_ROW[6:]
    transformations = _write_transformations(tmp_path / 'x.txt', [rigid_row, rigid_row])
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='quadratic',
    ).run()

    interior = np.asanyarray(
        nb.load(result.outputs.ec_jacobian_images[0]).dataobj
    )[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=1e-4)


def test_okan_jacobian_is_undefined_for_motion_only(tmp_path):
    """--sloppy forces correction_mode=motion, where no EC component exists."""
    transformations = _write_transformations(tmp_path / 'x.txt', [_IDENTITY_ROW] * 2)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='motion',
    ).run()
    assert not isdefined(result.outputs.ec_jacobian_images)


def test_okan_jacobian_is_undefined_for_cubic_and_does_not_raise(tmp_path):
    """Cubic is a valid existing mode, so it must degrade, not abort.

    Weighting is on by default, so raising here would newly break runs that
    work today and push users to --no-jacobian-weighting, losing gradwarp and
    SDC weighting as collateral. Silently applying the quadratic formula to
    cubic parameters is what is forbidden.
    """
    transformations = _write_transformations(tmp_path / 'x.txt', [_IDENTITY_ROW] * 2)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='cubic',
    ).run()
    assert not isdefined(result.outputs.ec_jacobian_images)


def test_okan_jacobian_rejects_a_short_row(tmp_path):
    transformations = _write_transformations(tmp_path / 'x.txt', [[0.0] * 20])
    with pytest.raises(ValueError, match='24'):
        OkanQuadraticJacobian(
            transformations_file=transformations,
            reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
            correction_mode='quadratic',
        ).run()


# --- StackJacobianWeights ---------------------------------------------------

from qsiprep.interfaces.jacobian import StackJacobianWeights


def test_stack_jacobian_weights_single_image_is_3d(tmp_path):
    """The collapsed (all-volumes-share-one-map) case stays 3D."""
    weight_image = _write_map(tmp_path / 'w0.nii.gz', 1.0)
    result = StackJacobianWeights(
        weight_images=[weight_image], weight_index=[0, 0, 0]
    ).run(cwd=str(tmp_path))
    out = nb.load(result.outputs.out_file)
    assert out.ndim == 3


def test_stack_jacobian_weights_two_images_stack_on_last_axis(tmp_path):
    """Multiple unique maps stack on the last axis, in first-appearance order."""
    first = _write_map(tmp_path / 'w0.nii.gz', 1.0)
    second = _write_map(tmp_path / 'w1.nii.gz', 2.0)
    result = StackJacobianWeights(
        weight_images=[first, second], weight_index=[0, 1, 0]
    ).run(cwd=str(tmp_path))
    out = nb.load(result.outputs.out_file)
    data = np.asanyarray(out.dataobj)
    assert out.ndim == 4
    assert data.shape[-1] == 2
    assert np.allclose(data[..., 0], 1.0)
    assert np.allclose(data[..., 1], 2.0)


def test_stack_jacobian_weights_undefined_input_stays_undefined(tmp_path):
    """No weights applied this run: no crash, and nothing is written.

    This is the case the brief's original ``mandatory=True`` input spec would
    have raised ``ValueError`` on -- see the task-12 report. Pinning it here
    keeps that regression from coming back.
    """
    result = StackJacobianWeights().run(cwd=str(tmp_path))
    assert not isdefined(result.outputs.out_file)
    assert not isdefined(result.outputs.meta_dict)


def test_stack_jacobian_weights_meta_dict_index_matches_shape(tmp_path):
    """The sidecar's index travels with the stacked file, unmodified.

    ``applied_corrections``/``unmodulated_corrections``/``unmodulated_reason``
    are build-time inputs set by the caller (see
    ``qsiprep.workflows.dwi.jacobian_provenance.jacobian_provenance_for``),
    not read from ``config.workflow`` -- so this test sets them directly on
    the interface rather than monkeypatching global config.
    """
    first = _write_map(tmp_path / 'w0.nii.gz', 1.0)
    second = _write_map(tmp_path / 'w1.nii.gz', 2.0)
    result = StackJacobianWeights(
        weight_images=[first, second],
        weight_index=[0, 0, 1, 0],
        applied_corrections=['gradwarp', 'sdc'],
        unmodulated_corrections=[],
        unmodulated_reason=None,
    ).run(cwd=str(tmp_path))

    out = nb.load(result.outputs.out_file)
    meta = result.outputs.meta_dict
    assert meta['JacobianWeightIndex'] == [0, 0, 1, 0]
    assert meta['AppliedCorrections'] == ['gradwarp', 'sdc']
    # every index must address a real frame of the stacked file
    assert max(meta['JacobianWeightIndex']) < out.shape[-1]
