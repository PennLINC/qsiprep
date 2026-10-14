"""InvertITKAffine writes the true inverse of ANTs' rigid and affine .mat transforms."""

import numpy as np
import SimpleITK as sitk

from qsiprep.interfaces.itk import InvertITKAffine, linear_transform, linear_transform_matrix


def _write_euler(path, rotation, translation, center):
    xfm = sitk.Euler3DTransform()
    xfm.SetCenter(center)
    xfm.SetRotation(*rotation)
    xfm.SetTranslation(translation)
    sitk.WriteTransform(xfm, str(path))
    return str(path)


def _write_affine(path, matrix, translation, center):
    xfm = sitk.AffineTransform(3)
    xfm.SetCenter(center)
    xfm.SetMatrix(np.asarray(matrix, float).ravel().tolist())
    xfm.SetTranslation(translation)
    sitk.WriteTransform(xfm, str(path))
    return str(path)


def _check(tmp_path, in_file):
    result = InvertITKAffine(in_file=in_file).run(cwd=str(tmp_path))
    fwd = linear_transform_matrix(in_file)
    inv = linear_transform_matrix(result.outputs.out_file)
    assert np.allclose(fwd @ inv, np.eye(4), atol=1e-9)
    # a point mapped forward by ITK and back by ITK comes home
    p = (12.0, -30.0, 7.0)
    back = sitk.ReadTransform(result.outputs.out_file).TransformPoint(
        sitk.ReadTransform(in_file).TransformPoint(p)
    )
    assert np.allclose(back, p, atol=1e-9)
    # the output is a plain affine with a zero centre
    assert np.allclose(linear_transform(result.outputs.out_file).GetCenter(), 0.0)


def test_invert_rigid_with_fixed_centre(tmp_path):
    _check(
        tmp_path,
        _write_euler(
            tmp_path / 'fwd.mat', (0.1, -0.05, 0.2), (3.0, -4.0, 1.5), (10.0, -20.0, 5.0)
        ),
    )


def test_invert_affine_with_fixed_centre(tmp_path):
    rot = sitk.Euler3DTransform()
    rot.SetRotation(0.02, 0.1, -0.07)
    mat = np.array(rot.GetMatrix()).reshape(3, 3) @ np.diag([1.05, 0.98, 1.0])
    _check(tmp_path, _write_affine(tmp_path / 'fwd.mat', mat, (1.0, 2.0, -3.0), (-5.0, 8.0, 2.0)))


def test_invert_accepts_a_one_element_list(tmp_path):
    """ANTs' forward_transforms is a list; the dwiref export hands it over as one."""
    in_file = _write_euler(
        tmp_path / 'fwd.mat', (0.1, -0.05, 0.2), (3.0, -4.0, 1.5), (1.0, 2.0, 3.0)
    )
    result = InvertITKAffine(in_file=[in_file]).run(cwd=str(tmp_path))
    fwd, inv = linear_transform_matrix(in_file), linear_transform_matrix(result.outputs.out_file)
    assert np.allclose(fwd @ inv, np.eye(4), atol=1e-9)
