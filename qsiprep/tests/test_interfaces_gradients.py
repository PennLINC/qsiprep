"""Tests for the qsiprep.interfaces.gradients module."""

import os
import shutil

import nibabel as nb
import numpy as np
import pytest
import SimpleITK as sitk

from qsiprep.interfaces.gradients import get_fsl_motion_params, get_ras_motion_params


def test_get_ras_motion_params_no_axis_flip(tmp_path):
    """RAS export recovers applied motion with correct sign on a radiological grid.

    On a grid with a negative x on the affine diagonal, the FSL/LPS conventions
    flip x; ``get_ras_motion_params`` must report the applied RAS motion with no
    flip.
    """
    # radiological reference (negative x-diagonal, det < 0)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    ref_file = os.path.join(tmp_path, 'ref.nii.gz')
    nb.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), affine=affine).to_filename(ref_file)

    conv = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS <-> LPS
    itk_file = os.path.join(tmp_path, 'xfm.mat')

    def ras_to_itk(m_ras):
        m_lps = conv @ m_ras @ conv
        aff = sitk.AffineTransform(3)
        aff.SetMatrix(m_lps[:3, :3].ravel().tolist())
        aff.SetTranslation(m_lps[:3, 3].tolist())
        sitk.WriteTransform(aff, itk_file)

    # pure +3 mm translation along RAS x -> +3, not -3
    m = np.eye(4)
    m[0, 3] = 3.0
    ras_to_itk(m)
    params = get_ras_motion_params(itk_file, ref_file)
    assert params.shape == (12,)
    np.testing.assert_allclose(params[9:12], [3.0, 0.0, 0.0], atol=1e-6)

    # pure +5 deg rotation about RAS x -> rotvec x = +5 deg, not -5
    th = np.deg2rad(5.0)
    rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    m = np.eye(4)
    m[:3, :3] = rx
    ras_to_itk(m)
    params = get_ras_motion_params(itk_file, ref_file)
    np.testing.assert_allclose(params[6:9], [th, 0.0, 0.0], atol=1e-6)


def test_get_fsl_motion_params_identity_transform(tmp_path):
    """Test end-to-end motion parameter extraction using c3d_affine_tool."""
    if shutil.which('c3d_affine_tool') is None:
        pytest.skip('c3d_affine_tool is required for this test')
    ref_file = os.path.join(tmp_path, 'ref.nii.gz')
    itk_file = os.path.join(tmp_path, 'transform0GenericAffine.mat')

    # Small reference image is enough to exercise center/offset math.
    ref_img = nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), affine=np.eye(4))
    ref_img.to_filename(ref_file)

    # Write an identity ITK affine transform.
    sitk.WriteTransform(sitk.AffineTransform(3), itk_file)

    motion_params = get_fsl_motion_params(itk_file, ref_file, str(tmp_path))

    assert motion_params.shape == (12,)
    np.testing.assert_allclose(motion_params[:3], [1.0, 1.0, 1.0], atol=1e-8)  # scales
    np.testing.assert_allclose(motion_params[3:], np.zeros(9), atol=1e-8)  # shear/rot/trans


def test_compose_transforms_places_gradwarp_between_hmc_and_sdc():
    """TORTOISE composes motion/eddy, then gradwarp, then SDC.

    transform_order is native-to-target and reversed for ANTs, so gradwarp must
    sit immediately after hmc in the list.
    """
    from qsiprep.interfaces.gradients import ComposeTransforms

    order = ComposeTransforms._transform_order_names()
    assert order.index('gradwarp') == order.index('hmc') + 1
    assert order.index('gradwarp') < order.index('fieldwarp')


def test_compose_transforms_stage_names_match_the_runtime_lookup():
    """Every stage name must have an entry in _run_interface's by_name dict.

    A stage present in _TRANSFORM_STAGES but missing from that dict raises
    KeyError at runtime, long after the graph is built.
    """
    import inspect

    from qsiprep.interfaces.gradients import ComposeTransforms

    source = inspect.getsource(ComposeTransforms._run_interface)
    for stage in ComposeTransforms._TRANSFORM_STAGES:
        assert f"'{stage}':" in source, stage


def test_compose_transforms_gradwarp_is_not_forwarded_to_apply_transforms():
    """Every custom input must be popped before ifargs reaches ApplyTransforms."""
    from qsiprep.interfaces.gradients import ComposeTransforms

    assert 'gradwarp' in ComposeTransforms._popped_keys()
