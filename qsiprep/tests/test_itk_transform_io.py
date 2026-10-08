"""ITK transform files, however written, mean the same thing when read through SimpleITK.

Every place that reads, inverts or composes ANTs/ITK transforms goes through
:func:`qsiprep.interfaces.itk.linear_transform_matrix`, which derives the point map from
``TransformPoint`` so that centres of rotation, Euler conventions and composite ordering are
ITK's own. These tests pin the conventions that code relies on.
"""

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.io import savemat

from qsiprep.interfaces.itk import affine_from_matrix, linear_transform, linear_transform_matrix


def _euler_with_centre():
    xfm = sitk.Euler3DTransform()
    xfm.SetCenter((20.0, -10.0, 5.0))
    xfm.SetRotation(0.1, -0.05, 0.2)
    xfm.SetTranslation((3.0, -4.0, 1.5))
    return xfm


@pytest.mark.parametrize('ext', ['.txt', '.mat', '.h5'])
def test_point_map_survives_every_file_format(tmp_path, ext):
    """A rigid with a non-zero centre reads back identically from txt, mat and h5."""
    xfm = _euler_with_centre()
    path = tmp_path / f'xfm{ext}'
    sitk.WriteTransform(xfm, str(path))
    assert np.allclose(linear_transform_matrix(path), linear_transform_matrix(xfm), atol=1e-9)
    # and the file's own TransformPoint agrees with the matrix at an arbitrary point
    p = np.array([12.0, -30.0, 7.0])
    assert np.allclose(
        linear_transform_matrix(path)[:3, :3] @ p + linear_transform_matrix(path)[:3, 3],
        sitk.ReadTransform(str(path)).TransformPoint(tuple(p)),
        atol=1e-9,
    )


def test_point_map_folds_the_centre_into_the_offset():
    """``y = R (x - c) + c + t``: the matrix offset is ``t + c - R c``, not ``t``."""
    xfm = _euler_with_centre()
    m = linear_transform_matrix(xfm)
    R = np.array(xfm.GetMatrix()).reshape(3, 3)
    c, t = np.array(xfm.GetCenter()), np.array(xfm.GetTranslation())
    assert np.allclose(m[:3, :3], R)
    assert np.allclose(m[:3, 3], t + c - R @ c)
    assert not np.allclose(m[:3, 3], t)  # the centre matters


def test_euler_convention_is_z_x_y():
    """ITK's Euler3DTransform (ComputeZYX off) is ``R = Rz Rx Ry``."""
    ax, ay, az = 0.1, -0.05, 0.2
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    xfm = sitk.Euler3DTransform()
    xfm.SetRotation(ax, ay, az)
    assert np.allclose(linear_transform_matrix(xfm)[:3, :3], rz @ rx @ ry)


def test_composite_applies_the_last_added_transform_first():
    """``CompositeTransform([a, b])`` maps ``p`` to ``a(b(p))``."""
    a = affine_from_matrix(np.diag([2.0, 1.0, 1.0, 1.0]))
    b = affine_from_matrix(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
    composite = sitk.CompositeTransform([a, b])
    assert composite.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx((2.0, 0.0, 0.0))
    assert np.allclose(
        linear_transform_matrix(composite),
        linear_transform_matrix(a) @ linear_transform_matrix(b),
    )


@pytest.mark.parametrize('ext', ['.txt', '.mat', '.h5'])
def test_linear_transform_gives_the_typed_accessors(tmp_path, ext):
    """``linear_transform`` answers the typed accessors, also for a one-member composite h5."""
    xfm = _euler_with_centre()
    path = tmp_path / f'xfm{ext}'
    sitk.WriteTransform(sitk.CompositeTransform([xfm]) if ext == '.h5' else xfm, str(path))
    typed = linear_transform(path)
    assert typed.GetTranslation() == pytest.approx(xfm.GetTranslation())
    assert typed.GetCenter() == pytest.approx(xfm.GetCenter())
    assert np.allclose(typed.GetMatrix(), xfm.GetMatrix())


def test_affine_from_matrix_round_trips():
    m = linear_transform_matrix(_euler_with_centre())
    assert np.allclose(linear_transform_matrix(affine_from_matrix(m)), m, atol=1e-12)
    assert np.allclose(affine_from_matrix(m).GetCenter(), 0.0)


def test_inverse_through_simpleitk_matches_the_matrix_inverse():
    xfm = _euler_with_centre()
    assert np.allclose(
        linear_transform_matrix(xfm.GetInverse()),
        np.linalg.inv(linear_transform_matrix(xfm)),
        atol=1e-9,
    )


# Files written without SimpleITK, as oracles for the reader: the text layout ANTs' .txt uses and
# the MATLAB v4 container its .mat uses. Asymmetric angles, a non-zero centre and an off-axis test
# point, so no single flipped sign or swapped axis can cancel out.
_ANGLES = (0.3, -0.2, 0.1)
_TRANSLATION = np.array([3.0, -4.0, 1.5])
_CENTRE = np.array([20.0, -10.0, 5.0])
_POINT = np.array([12.0, -30.0, 7.0])


def _expected_rotation(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


def _expected_point_map(rotation):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = _TRANSLATION + _CENTRE - rotation @ _CENTRE
    return out


def test_reader_agrees_with_a_hand_written_euler_text_file(tmp_path):
    path = tmp_path / 'euler.txt'
    path.write_text(
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: Euler3DTransform_double_3_3\n'
        'Parameters: ' + ' '.join(map(str, (*_ANGLES, *_TRANSLATION))) + '\n'
        'FixedParameters: ' + ' '.join(map(str, _CENTRE)) + '\n'
    )
    expected = _expected_point_map(_expected_rotation(*_ANGLES))
    assert np.allclose(linear_transform_matrix(path), expected, atol=1e-9)
    assert np.allclose(
        sitk.ReadTransform(str(path)).TransformPoint(tuple(_POINT)),
        expected[:3, :3] @ _POINT + expected[:3, 3],
        atol=1e-9,
    )
    # a single flipped angle sign must not be confused with the original
    for i in range(3):
        flipped = list(_ANGLES)
        flipped[i] = -flipped[i]
        assert not np.allclose(
            _expected_point_map(_expected_rotation(*flipped)), expected, atol=1e-3
        )


def test_reader_agrees_with_a_hand_written_ants_style_mat_file(tmp_path):
    """ANTs writes a MATLAB v4 file with a float matrix key and a ``fixed`` centre."""
    rotation = _expected_rotation(*_ANGLES)
    path = tmp_path / 'affine.mat'
    savemat(
        str(path),
        {
            'AffineTransform_float_3_3': np.concatenate([rotation.ravel(), _TRANSLATION])
            .astype('float32')
            .reshape(-1, 1),
            'fixed': _CENTRE.astype('float32').reshape(-1, 1),
        },
        format='4',
    )
    expected = _expected_point_map(rotation)
    assert np.allclose(linear_transform_matrix(path), expected, atol=1e-5)
    typed = linear_transform(path)
    assert np.allclose(typed.GetCenter(), _CENTRE, atol=1e-5)
    assert np.allclose(typed.GetTranslation(), _TRANSLATION, atol=1e-5)


def test_world_to_lps_map_is_a_change_of_basis():
    """A RAS point map re-expressed in LPS sends the same physical point to the same place."""
    from qsiprep.tests.truth_scoring import world_to_lps_map

    T_ras = _expected_point_map(_expected_rotation(*_ANGLES))
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    p_ras = np.append(_POINT, 1.0)
    moved_ras = T_ras @ p_ras
    moved_lps = world_to_lps_map(T_ras) @ (flip @ p_ras)
    assert np.allclose(flip @ moved_lps, moved_ras)
    assert not np.allclose(world_to_lps_map(T_ras), T_ras)  # the flip is not a no-op here
