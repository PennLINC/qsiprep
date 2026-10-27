"""Test niimath -skullstrip through qsiprep's interface on a synthetic head."""

import shutil
import subprocess

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.niimath import SkullStrip


def _niimath_has_skullstrip():
    if shutil.which('niimath') is None:
        return False
    help_text = subprocess.run(['niimath'], capture_output=True, text=True, check=False).stdout
    return (
        '-skullstrip' in help_text
        and 'NOT in this build' not in help_text.split('-skullstrip', 1)[1].split('\n', 1)[0]
    )


def _synthetic_head(path):
    """Write a 2 mm head: a bright brain ellipsoid, a dark gap and a bright scalp shell."""
    n = (96, 112, 96)
    z, y, x = np.indices(n)
    c = np.array(n) / 2
    r_brain = ((x - c[0]) / 32) ** 2 + ((y - c[1]) / 40) ** 2 + ((z - c[2]) / 30) ** 2
    r_head = ((x - c[0]) / 40) ** 2 + ((y - c[1]) / 48) ** 2 + ((z - c[2]) / 38) ** 2
    rng = np.random.default_rng(0)
    img = rng.normal(5, 2, n)
    scalp = r_head < 1
    gap = (r_head < 0.9) & (r_brain > 1)
    brain = r_brain < 1
    img[scalp] = 60 + rng.normal(0, 5, n)[scalp]
    img[gap] = 15 + rng.normal(0, 3, n)[gap]
    img[brain] = 100 + rng.normal(0, 8, n)[brain]
    core = r_brain < 0.3
    img[core] = 70 + rng.normal(0, 5, n)[core]
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -np.array(n)
    nb.Nifti1Image(img.astype('float32'), affine).to_filename(str(path))
    return brain


@pytest.mark.skipif(not _niimath_has_skullstrip(), reason='niimath without -skullstrip')
def test_skullstrip_recovers_a_synthetic_brain(tmp_path):
    head = tmp_path / 'head.nii.gz'
    truth = _synthetic_head(head)

    result = SkullStrip(in_file=str(head)).run(cwd=str(tmp_path))

    mask_img = nb.load(result.outputs.mask_file)
    mask = np.asanyarray(mask_img.dataobj) > 0
    assert mask_img.get_data_dtype() == np.uint8
    np.testing.assert_allclose(mask_img.affine, nb.load(str(head)).affine)
    dice = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
    assert dice > 0.95, dice
    assert (mask & ~truth).sum() / mask.sum() < 0.1  # little scalp in the mask

    brain = np.asanyarray(nb.load(result.outputs.out_file).dataobj)
    assert (brain[mask] > brain.min()).all()
    assert (brain[~mask] == brain.min()).all()
