"""Conform keeps an oblique anatomical image where it is in world space."""

import nibabel as nb
import numpy as np

from qsiprep.interfaces.images import Conform, deoblique_grid


def _rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rz_ = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry_ = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx_ = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz_ @ ry_ @ rx_


def _world_centroid(img):
    data = np.asarray(img.dataobj, dtype=float)
    ijk = np.indices(data.shape).reshape(3, -1).astype(float)
    w = data.ravel()
    centre_vox = ijk @ w / w.sum()
    return (img.affine @ np.append(centre_vox, 1.0))[:3]


def _oblique_image(tmp_path, tilt_deg=(5.0, -3.0, 4.0)):
    shape = (40, 48, 36)
    data = np.zeros(shape, np.float32)
    ijk = np.indices(shape).astype(float)
    centre = np.array([14.0, 30.0, 20.0])  # an off-centre bright sphere
    data[np.sqrt(((ijk - centre[:, None, None, None]) ** 2).sum(0)) < 5] = 100.0
    affine = np.eye(4)
    affine[:3, :3] = _rotation(*np.radians(tilt_deg)) @ np.diag([-1.5, -1.5, 1.5])  # LPS, oblique
    affine[:3, 3] = [30.0, 40.0, -25.0]
    path = tmp_path / 'oblique_T1w.nii.gz'
    nb.Nifti1Image(data, affine).to_filename(path)
    return path, nb.Nifti1Image(data, affine)


def test_deoblique_grid_covers_the_oblique_volume(tmp_path):
    _, img = _oblique_image(tmp_path)
    affine, shape = deoblique_grid(img)
    assert np.allclose(
        np.abs(affine[:3, :3]), np.diag([1.5, 1.5, 1.5])
    )  # axis aligned, same zooms
    corners = np.array(
        [
            [i, j, k, 1.0]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ]
    )
    new_world = (affine @ corners.T)[:3]
    orig = np.array([[i, j, k, 1.0] for i in (0, 39) for j in (0, 47) for k in (0, 35)])
    old_world = (img.affine @ orig.T)[:3]
    assert np.all(new_world.min(1) <= old_world.min(1) + 1e-6)
    assert np.all(new_world.max(1) >= old_world.max(1) - 1e-6)


def test_conform_resamples_oblique_anatomy_in_place(tmp_path):
    path, img = _oblique_image(tmp_path)
    result = Conform(
        in_file=str(path),
        target_zooms=(1.5, 1.5, 1.5),
        target_shape=img.shape,
        deoblique_header=True,
    ).run(cwd=str(tmp_path))
    out = nb.load(result.outputs.out_file)
    assert np.allclose(np.abs(out.affine[:3, :3]), np.diag([1.5, 1.5, 1.5]))  # obliquity removed
    assert (
        np.linalg.norm(_world_centroid(out) - _world_centroid(img)) < 0.5
    )  # same place in the world
    assert np.isclose(np.asarray(out.dataobj).sum(), np.asarray(img.dataobj).sum(), rtol=0.05)
