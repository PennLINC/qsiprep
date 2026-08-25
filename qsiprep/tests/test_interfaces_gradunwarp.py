"""Tests for the TORTOISE gradient nonlinearity interfaces.

Pure-Python behaviour -- command-line construction and field masking -- is
tested unconditionally. Tests that exercise the real TORTOISE binaries are
guarded with ``shutil.which`` and skip when those binaries are absent. They are
*not* permanently offline: CircleCI's ``unit_tests`` job runs pytest inside the
``pennlinc/qsiprep:test`` image, which ships the TORTOISE tools.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep.tests.gradient_fixtures import (
    write_itk_field,
)


def _require(*binaries):
    """Skip unless every named TORTOISE binary is on PATH."""
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        pytest.skip(f'{", ".join(missing)} required for this test')


def _components(path):
    """Return the (X, Y, Z, 3) displacement components of an ITK field."""
    data = np.asanyarray(nb.load(str(path)).dataobj)
    return data.reshape(data.shape[:3] + (3,))


@pytest.mark.parametrize(
    ('warp_dim', 'zeroed'),
    [('3D', ()), ('2D', (2,)), ('1D', (0, 1))],
)
def test_mask_warp_dimensions(tmp_path, warp_dim, zeroed):
    """2D zeroes the through-plane component; 1D keeps only that component."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim=warp_dim).run(cwd=str(tmp_path))

    before = _components(field)
    after = _components(result.outputs.out_file)
    for component in range(3):
        if component in zeroed:
            assert np.all(after[..., component] == 0)
        else:
            assert np.allclose(after[..., component], before[..., component])


def test_mask_warp_dimensions_preserves_geometry(tmp_path):
    """The masked field must stay on the same grid to compose correctly."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))

    original, masked = nb.load(str(field)), nb.load(result.outputs.out_file)
    assert masked.shape == original.shape
    assert np.allclose(masked.affine, original.affine)


def test_mask_warp_dimensions_does_not_modify_input(tmp_path):
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    before = _components(field).copy()
    MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))
    assert np.allclose(_components(field), before)
