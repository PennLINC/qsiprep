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


def test_resample_like_resamples_onto_the_target_grid(tmp_path):
    like = _write_map(tmp_path / 'like.nii.gz', 1.0, shape=DRBUDDI_SHAPE, affine=DRBUDDI_AFFINE)
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0, shape=NATIVE_SHAPE, affine=NATIVE_AFFINE)
    out = resample_like(mask, like, str(tmp_path / 'out.nii.gz'))
    resampled = nb.load(out)
    assert resampled.shape[:3] == DRBUDDI_SHAPE
    assert np.allclose(resampled.affine, DRBUDDI_AFFINE)


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


def test_stack_jacobian_weights_meta_dict_index_matches_shape(monkeypatch, tmp_path):
    """The sidecar's index travels with the stacked file, unmodified."""
    from qsiprep import config

    monkeypatch.setattr(config.workflow, 'jacobian_applied_corrections', ['gradwarp', 'sdc'])
    monkeypatch.setattr(config.workflow, 'jacobian_unmodulated_corrections', [])
    monkeypatch.setattr(config.workflow, 'jacobian_unmodulated_reason', None)

    first = _write_map(tmp_path / 'w0.nii.gz', 1.0)
    second = _write_map(tmp_path / 'w1.nii.gz', 2.0)
    result = StackJacobianWeights(
        weight_images=[first, second], weight_index=[0, 0, 1, 0]
    ).run(cwd=str(tmp_path))

    out = nb.load(result.outputs.out_file)
    meta = result.outputs.meta_dict
    assert meta['JacobianWeightIndex'] == [0, 0, 1, 0]
    assert meta['AppliedCorrections'] == ['gradwarp', 'sdc']
    # every index must address a real frame of the stacked file
    assert max(meta['JacobianWeightIndex']) < out.shape[-1]
