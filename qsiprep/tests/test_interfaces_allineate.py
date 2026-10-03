"""Test the niimath -allineate transform conversion."""

import json

import numpy as np
import pytest

from qsiprep.interfaces.itk import _itk_mat_to_matrix
from qsiprep.interfaces.niimath import allineate_json_to_itk


def _read_itk_txt(path):
    lines = {ln.split(':')[0]: ln.split(':', 1)[1].split() for ln in open(path) if ':' in ln}
    return _itk_mat_to_matrix(
        'AffineTransform_double_3_3',
        [float(v) for v in lines['Parameters']],
        [float(v) for v in lines['FixedParameters']],
    )


def test_allineate_json_becomes_the_same_map_in_lps(tmp_path):
    rng = np.random.default_rng(0)
    ras = np.eye(4)
    ras[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0] @ np.diag([1.05, 0.95, 1.0])
    ras[:3, 3] = [5.0, -7.0, 3.0]
    js = tmp_path / 'a.json'
    js.write_text(json.dumps({'fixed_to_moving': ras.ravel().tolist(), 'space': 'world'}))
    out = allineate_json_to_itk(str(js), str(tmp_path / 'a.txt'))
    lps = _read_itk_txt(out)
    # a fixed point in RAS, mapped in RAS, must equal the LPS map of the same point
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    p = np.array([10.0, 20.0, -5.0, 1.0])
    np.testing.assert_allclose(flip @ (lps @ (flip @ p)), ras @ p, atol=1e-8)


def _have(cmd):
    import shutil

    return shutil.which(cmd) is not None


@pytest.mark.skipif(
    not (_have('niimath') and _have('antsApplyTransforms')), reason='niimath + ANTs'
)
def test_ants_applies_the_converted_transform_like_niimath_does(tmp_path):
    """ANTs must move the image exactly as niimath's own reslice did (fixed->moving, LPS).

    Checked on a synthetic head stored LPS, on an RAS-stored copy with reordered voxel axes
    and on a copy with an oblique header: the conversion is in world space, so storage must
    not matter. The inverted transform is the control.
    """
    import subprocess

    import nibabel as nb
    from scipy.spatial.transform import Rotation as R

    from qsiprep.interfaces.niimath import Allineate
    from qsiprep.tests.test_interfaces_niimath_skullstrip import _synthetic_head

    fixed = tmp_path / 'fixed.nii.gz'
    _synthetic_head(fixed)
    img = nb.load(str(fixed))
    # the moving image: the same head rotated 20 degrees and shifted, resampled by nibabel
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler('xyz', [20, -10, 8], degrees=True).as_matrix()
    rot[:3, 3] = [6.0, -9.0, 4.0]
    moving_lps = nb.Nifti1Image(np.asanyarray(img.dataobj), rot @ img.affine)
    variants = {
        'lps': moving_lps,
        'ras': nb.as_closest_canonical(moving_lps),
    }
    obl = moving_lps.affine.copy()
    obl[:3, :3] = R.from_euler('xyz', [12, -8, 5], degrees=True).as_matrix() @ obl[:3, :3]
    variants['oblique'] = nb.Nifti1Image(np.asanyarray(moving_lps.dataobj), obl)
    mask = np.asanyarray(img.dataobj) > 30
    for name, moving_img in variants.items():
        moving = tmp_path / f'moving_{name}.nii.gz'
        moving_img.to_filename(str(moving))
        res = Allineate(in_file=str(moving), reference=str(fixed)).run(cwd=str(tmp_path))
        by_niimath = np.asanyarray(nb.load(res.outputs.out_file).dataobj)
        out = {}
        for tag, xfm in (
            ('fwd', res.outputs.out_transform),
            ('inv', f'[{res.outputs.out_transform},1]'),
        ):
            o = tmp_path / f'ants_{name}_{tag}.nii.gz'
            subprocess.run(
                [
                    'antsApplyTransforms',
                    '-d',
                    '3',
                    '-i',
                    str(moving),
                    '-r',
                    str(fixed),
                    '-t',
                    xfm,
                    '-o',
                    str(o),
                    '--interpolation',
                    'Linear',
                ],
                check=True,
                capture_output=True,
            )
            out[tag] = np.asanyarray(nb.load(str(o)).dataobj)
        agree = np.corrcoef(out['fwd'][mask], by_niimath[mask])[0, 1]
        control = np.corrcoef(out['inv'][mask], by_niimath[mask])[0, 1]
        assert agree > 0.99, (name, agree)
        assert control < 0.9, (name, control)
