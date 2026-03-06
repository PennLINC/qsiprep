"""Tests for the qsiprep.interfaces.gradients module."""

import os
import shutil

import nibabel as nb
import numpy as np
import SimpleITK as sitk

from qsiprep.interfaces.gradients import get_fsl_motion_params


def test_get_fsl_motion_params_identity_transform(tmp_path):
    """Test end-to-end motion parameter extraction using c3d_affine_tool."""
    assert shutil.which('c3d_affine_tool') is not None, 'c3d_affine_tool is required for this test'
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
