"""Tests for the qsiprep.interfaces.niworkflows module."""

import nibabel as nb
import niworkflows.interfaces.norm as niw_norm
import numpy as np
from nibabel.affines import apply_affine
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from qsiprep.interfaces.niworkflows import _create_cfm


def test_create_cfm_patch_installed():
    """qsiprep replaces the buggy upstream create_cfm at import time."""
    assert niw_norm.create_cfm is _create_cfm


def test_create_cfm_lesion_orientation(tmp_path):
    """A lesion stored in RAS is excluded at the correct world location even
    when in_file is stored in LPS (regression test for issue #1023)."""
    shape = (4, 5, 6)
    ras_affine = np.eye(4)

    # in_file: all-ones brain mask, stored in LPS orientation.
    ras_img = nb.Nifti1Image(np.ones(shape, dtype=np.uint8), ras_affine)
    xfm = ornt_transform(io_orientation(ras_img.affine), axcodes2ornt(('L', 'P', 'S')))
    in_img = ras_img.as_reoriented(xfm)
    in_file = str(tmp_path / 'in_lps.nii.gz')
    in_img.to_filename(in_file)

    # lesion: single voxel in RAS at world coordinate (1, 2, 3).
    lesion_data = np.zeros(shape, dtype=np.uint8)
    lesion_data[1, 2, 3] = 1
    lesion_file = str(tmp_path / 'lesion_ras.nii.gz')
    nb.Nifti1Image(lesion_data, ras_affine).to_filename(lesion_file)

    out = _create_cfm(
        in_file,
        lesion_mask=lesion_file,
        global_mask=True,
        out_path=str(tmp_path / 'cfm.nii.gz'),
    )

    cfm = nb.load(out)
    zeros = np.argwhere(np.asanyarray(cfm.dataobj) == 0)
    assert zeros.shape[0] == 1
    assert np.allclose(apply_affine(cfm.affine, zeros[0]), [1, 2, 3])
