"""Tests for qsiprep.interfaces.phase."""

import numpy as np


def _synth_complex(shape=(32, 32, 8, 6), seed=0):
    """Build synthetic complex data: positive true signal on the real axis,
    rotated by a smooth background phase, plus Gaussian noise on both channels.
    Returns (complex_data, true_real_signal)."""
    rng = np.random.default_rng(seed)
    nr, nc, ns, nb = shape
    # True (real, positive) signal: a smooth blob per volume
    yy, xx = np.mgrid[0:nr, 0:nc]
    blob = np.exp(-(((xx - nc / 2) ** 2 + (yy - nr / 2) ** 2) / (2 * (nr / 5) ** 2)))
    true = np.zeros(shape)
    for v in range(nb):
        for z in range(ns):
            true[:, :, z, v] = (1.0 + 0.3 * v) * blob
    # Smooth background phase (low spatial frequency), same across volumes
    bg = 0.8 * np.sin(2 * np.pi * xx / nc) + 0.8 * np.cos(2 * np.pi * yy / nr)
    bg = bg[:, :, None, None] * np.ones((1, 1, ns, nb))
    clean = true * np.exp(1j * bg)
    # Moderate SNR: the regime where phase correction beats magnitude (the
    # magnitude noise floor biases the signal). At very high SNR the magnitude
    # already tracks the truth and the comparison is not meaningful.
    noise = rng.normal(scale=0.1, size=shape) + 1j * rng.normal(scale=0.1, size=shape)
    return clean + noise, true


def test_phasecorrect_interface_writes_real_channel(tmp_path):
    import nibabel as nb

    from qsiprep.interfaces.phase import PhaseCorrect

    carr, true = _synth_complex()
    affine = np.eye(4)
    complex_file = tmp_path / 'complex.nii.gz'
    nb.Nifti1Image(carr.astype(np.complex64), affine).to_filename(complex_file)

    interface = PhaseCorrect(
        complex_file=str(complex_file),
        method='tv',
        tv_weight=6.0,
    )
    results = interface.run(cwd=str(tmp_path))

    import os

    out_file = results.outputs.out_file
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape == carr.shape
    # Output is real-valued float32
    assert np.issubdtype(out_img.dataobj.dtype, np.floating)
    out_data = out_img.get_fdata()
    err_real = np.mean((out_data - true) ** 2)
    err_mag = np.mean((np.abs(carr) - true) ** 2)
    assert err_real < err_mag
    # A reportlet was produced
    assert os.path.isfile(results.outputs.out_report)
