"""InvertITKAffine writes the true inverse of ANTs' rigid and affine .mat transforms."""

import numpy as np
from scipy.io import loadmat, savemat

from qsiprep.interfaces.itk import InvertITKAffine, _itk_mat_to_matrix


def _euler(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def _check(tmp_path, key, params, fixed):
    in_file = tmp_path / 'fwd.mat'
    savemat(
        in_file,
        {
            key: np.asarray(params, float).reshape(-1, 1),
            'fixed': np.asarray(fixed, float).reshape(-1, 1),
        },
    )
    result = InvertITKAffine(in_file=str(in_file)).run(cwd=str(tmp_path))
    out = loadmat(result.outputs.out_file)
    inv = _itk_mat_to_matrix(
        'AffineTransform_double_3_3', out['AffineTransform_double_3_3'], out['fixed']
    )
    fwd = _itk_mat_to_matrix(key, params, fixed)
    assert np.allclose(fwd @ inv, np.eye(4), atol=1e-9)
    p = np.array([12.0, -30.0, 7.0, 1.0])  # a point mapped forward then back comes home
    assert np.allclose(inv @ (fwd @ p), p)


def test_invert_rigid_with_fixed_centre(tmp_path):
    _check(
        tmp_path,
        'Euler3DTransform_double_3_3',
        [0.1, -0.05, 0.2, 3.0, -4.0, 1.5],
        [10.0, -20.0, 5.0],
    )


def test_invert_affine_with_fixed_centre(tmp_path):
    mat = _euler(0.02, 0.1, -0.07) @ np.diag([1.05, 0.98, 1.0])
    _check(
        tmp_path,
        'AffineTransform_double_3_3',
        list(mat.ravel()) + [1.0, 2.0, -3.0],
        [-5.0, 8.0, 2.0],
    )
