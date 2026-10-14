"""Test the in-brain gradient nonlinearity QC table."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.gradunwarp import GradDevQC, graddev_rotation_degrees


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_rotation_angle_ignores_scaling_shear_and_transposition():
    rotation = _rotation([0.6, 0.0, 0.8], 3.0)
    stretch = np.array([[1.04, 0.01, 0.0], [0.01, 0.97, 0.02], [0.0, 0.02, 1.01]])
    tensors = np.stack([np.eye(3), rotation, rotation @ stretch, (rotation @ stretch).T])

    np.testing.assert_allclose(graddev_rotation_degrees(tensors), [0, 3, 3, 3], atol=1e-6)


def test_rotation_angle_of_an_improper_tensor_is_finite():
    reflected = np.diag([1.0, 1.0, -1.0])
    assert graddev_rotation_degrees(reflected[None])[0] == pytest.approx(0.0)


def _save(data, path, affine):
    nb.Nifti1Image(np.asarray(data), affine).to_filename(path)
    return str(path)


def test_graddev_qc_summarizes_only_brain_voxels(tmp_path):
    import pandas as pd

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape = (6, 6, 6)
    brain = np.zeros(shape, dtype=np.uint8)
    brain[1:5, 1:5, 1:5] = 1

    angles = np.where(brain > 0, 2.0, 30.0)
    tensors = np.zeros(shape + (9,))
    for idx in np.ndindex(shape):
        tensors[idx] = _rotation([0, 0, 1], angles[idx]).ravel()

    displacement = np.where(brain > 0, 1.5, 9.0)
    field = np.zeros(shape + (1, 3))
    field[..., 0, 0] = displacement * 0.6
    field[..., 0, 1] = displacement * 0.8

    result = GradDevQC(
        grad_dev=_save(tensors, tmp_path / 'graddev.nii.gz', affine),
        grad_dev_mask=_save(brain, tmp_path / 'acpc_mask.nii.gz', affine),
        gradwarp_field=_save(field, tmp_path / 'field.nii.gz', affine),
        gradwarp_mask=_save(brain, tmp_path / 'native_mask.nii.gz', affine),
    ).run(cwd=str(tmp_path))

    table = pd.read_csv(result.outputs.out_file, sep='\t').set_index('measure')
    assert list(table.index) == ['displacement_mm', 'rotation_deg']
    assert (table['n_voxels'] == 64).all()
    np.testing.assert_allclose(table.loc['displacement_mm', ['p50', 'max']], 1.5, atol=1e-6)
    np.testing.assert_allclose(table.loc['rotation_deg', ['p50', 'max']], 2.0, atol=1e-4)


def test_graddev_qc_resamples_a_mask_on_another_grid(tmp_path):
    import pandas as pd

    shape = (6, 6, 6)
    tensors = np.tile(np.eye(3).ravel(), shape + (1,))
    fine = np.ones((12, 12, 12), dtype=np.uint8)

    result = GradDevQC(
        grad_dev=_save(tensors, tmp_path / 'graddev.nii.gz', np.diag([2.0, 2.0, 2.0, 1.0])),
        grad_dev_mask=_save(fine, tmp_path / 'mask.nii.gz', np.diag([1.0, 1.0, 1.0, 1.0])),
    ).run(cwd=str(tmp_path))

    table = pd.read_csv(result.outputs.out_file, sep='\t').set_index('measure')
    assert list(table.index) == ['rotation_deg']
    assert table.loc['rotation_deg', 'n_voxels'] > 0
