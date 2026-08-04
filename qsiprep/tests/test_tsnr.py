"""Temporal SNR over the b=0 volumes of a diffusion series."""

import numpy as np
import pytest


def _write(path, data, affine=None):
    import nibabel as nb

    nb.Nifti1Image(data.astype('float32'), affine if affine is not None else np.eye(4)).to_filename(
        str(path)
    )
    return str(path)


@pytest.fixture
def series(tmp_path):
    """8 b=0 volumes with known noise, plus diffusion-weighted volumes.

    The DW volumes are given a wildly different mean so that including them
    would visibly corrupt the result -- that is what the b=0 restriction exists
    to prevent.
    """
    rng = np.random.default_rng(0)
    shape = (8, 8, 8)
    signal = np.full(shape, 100.0, dtype='float32')

    vols, bvals = [], []
    for _ in range(8):
        vols.append(signal + rng.normal(0, 5, shape))
        bvals.append(0)
    for _ in range(12):
        vols.append(signal * 0.2 + rng.normal(0, 5, shape))
        bvals.append(2000)

    data = np.stack(vols, axis=-1)
    dwi = _write(tmp_path / 'dwi.nii.gz', data)
    bval = tmp_path / 'dwi.bval'
    bval.write_text(' '.join(str(b) for b in bvals) + '\n')
    return dwi, str(bval)


def test_tsnr_uses_only_b0_volumes(series):
    """Mean/SD should reflect the b=0 set: ~100/5 = ~20, not the DW mixture."""
    from qsiprep.interfaces.tsnr import DWITSNR

    dwi, bval = series
    res = DWITSNR(dwi_file=dwi, bval_file=bval).run()
    assert res.outputs.n_b0 == 8
    # if DW volumes leaked in, the SD would balloon and TSNR would collapse
    assert 12 < res.outputs.median_tsnr < 32


def test_tsnr_map_geometry_matches_input(series):
    import nibabel as nb

    from qsiprep.interfaces.tsnr import DWITSNR

    dwi, bval = series
    res = DWITSNR(dwi_file=dwi, bval_file=bval).run()
    out, ref = nb.load(res.outputs.out_file), nb.load(dwi)
    assert out.shape == ref.shape[:3]
    assert np.allclose(out.affine, ref.affine)


def test_single_b0_yields_empty_map_not_garbage(tmp_path):
    """One b=0 gives no variance estimate; emit zeros rather than nonsense."""
    import nibabel as nb

    from qsiprep.interfaces.tsnr import DWITSNR

    rng = np.random.default_rng(1)
    data = np.stack([np.full((6, 6, 6), 100.0) + rng.normal(0, 3, (6, 6, 6)) for _ in range(4)], -1)
    dwi = _write(tmp_path / 'one.nii.gz', data)
    bval = tmp_path / 'one.bval'
    bval.write_text('0 1000 2000 3000\n')

    res = DWITSNR(dwi_file=dwi, bval_file=bval).run()
    assert res.outputs.n_b0 == 1
    assert res.outputs.median_tsnr == 0.0
    assert np.allclose(np.asanyarray(nb.load(res.outputs.out_file).dataobj), 0)


def test_mask_restricts_the_map(series, tmp_path):
    import nibabel as nb

    from qsiprep.interfaces.tsnr import DWITSNR

    dwi, bval = series
    mask = np.zeros((8, 8, 8), dtype='float32')
    mask[2:6, 2:6, 2:6] = 1
    mask_f = _write(tmp_path / 'mask.nii.gz', mask)

    res = DWITSNR(dwi_file=dwi, bval_file=bval, mask_file=mask_f).run()
    out = np.asanyarray(nb.load(res.outputs.out_file).dataobj)
    assert np.allclose(out[mask == 0], 0), 'values leaked outside the mask'
    assert out[mask > 0].mean() > 0


def test_tsnr_is_wired_into_derivatives():
    """It must reach the output tree, not just exist as an interface."""
    import inspect

    from qsiprep.workflows.dwi import derivatives

    src = inspect.getsource(derivatives)
    assert "desc='tsnr'" in src
    assert "(tsnr, ds_tsnr, [('out_file', 'in_file')])" in src
    # computed on the final resampled series, with its own bvals and mask
    assert "('dwi_t1', 'dwi_file')" in src
    assert "('bvals_t1', 'bval_file')" in src
