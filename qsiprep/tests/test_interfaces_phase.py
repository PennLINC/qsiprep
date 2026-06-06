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


def test_rephase_tv_recovers_real_signal():
    from qsiprep.interfaces.phase import rephase_tv

    carr, true = _synth_complex()
    real, imag, phase_bg = rephase_tv(carr, weight=6.0)

    assert real.shape == carr.shape
    assert imag.shape == carr.shape
    # Real channel should track the true signal better than naive magnitude
    mag = np.abs(carr)
    err_real = np.mean((real - true) ** 2)
    err_mag = np.mean((mag - true) ** 2)
    assert err_real < err_mag
    # Imaginary residual should be near zero-mean
    assert abs(np.mean(imag)) < 0.05


def test_rephase_tv_does_not_imprint_magnitude_anatomy():
    """High-dynamic-range magnitude + smooth phase must not leak anatomy.

    Regression guard for the background-phase estimator: with a structured,
    high-dynamic-range magnitude and a purely smooth background phase (and no
    noise), perfect correction leaves the imaginary channel ~zero. A vectorial
    2-channel TV (``channel_axis=-1``) couples the magnitude edges into the
    phase estimate and leaks the magnitude structure into the imaginary channel;
    the volumetric estimate used by ``rephase_tv`` must not.
    """
    from qsiprep.interfaces.phase import rephase_tv

    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    # Structured magnitude spanning a large dynamic range (like real DWI).
    mag = np.full((n, n), 400.0)
    mag[16:48, 16:48] = 8000.0
    mag[26:38, 26:38] = 20000.0
    bg = 0.8 * np.sin(2 * np.pi * xx / n) + 0.8 * np.cos(2 * np.pi * yy / n)
    carr = (mag * np.exp(1j * bg))[:, :, None]  # 3D (single slice)

    _real, imag, _bg = rephase_tv(carr, weight=6.0)

    brain = mag > 4000.0
    leak = np.mean(np.abs(imag[:, :, 0][brain])) / np.mean(mag[brain])
    # channel_axis=-1 gives ~0.04 here; channel_axis=None gives <0.01.
    assert leak < 0.02


def _anatomy_imprint_volume():
    """High-dynamic-range structured magnitude + smooth phase (single slice)."""
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    mag = np.full((n, n), 400.0)
    mag[16:48, 16:48] = 8000.0
    mag[26:38, 26:38] = 20000.0
    bg = 0.8 * np.sin(2 * np.pi * xx / n) + 0.8 * np.cos(2 * np.pi * yy / n)
    return (mag * np.exp(1j * bg))[:, :, None], mag


def test_rephase_tv_complex_recovers_real_signal():
    from qsiprep.interfaces.phase import rephase_tv_complex

    carr, true = _synth_complex()
    real, imag, _phase_bg = rephase_tv_complex(carr, weight=6.0)

    assert real.shape == carr.shape
    mag = np.abs(carr)
    assert np.mean((real - true) ** 2) < np.mean((mag - true) ** 2)
    assert abs(np.mean(imag)) < 0.05


def test_rephase_tv_complex_does_not_imprint_magnitude_anatomy():
    """Paper-faithful complex TV must also not leak magnitude anatomy."""
    from qsiprep.interfaces.phase import rephase_tv_complex

    carr, mag = _anatomy_imprint_volume()
    _real, imag, _bg = rephase_tv_complex(carr, weight=6.0)
    brain = mag > 4000.0
    leak = np.mean(np.abs(imag[:, :, 0][brain])) / np.mean(mag[brain])
    assert leak < 0.02


def test_dc_kernels_present_and_normalized():
    from qsiprep.interfaces.phase import DC_KERNELS

    expected = {'B3', 'B5', 'G3F1', 'G5F2', 'G3F1H', 'G5F2H', 'Opt3', 'Opt5'}
    assert expected == set(DC_KERNELS)
    # Boxcar kernels sum to 1
    assert abs(DC_KERNELS['B3'].sum() - 1.0) < 1e-9
    assert DC_KERNELS['B3'].shape == (3, 3)
    assert DC_KERNELS['Opt5'].shape == (5, 5)


def test_rephase_dc_recovers_real_signal():
    from qsiprep.interfaces.phase import rephase_dc

    carr, true = _synth_complex()
    real, imag, phase_bg = rephase_dc(carr, kernel_name='Opt5', outlier_detection=False)

    assert real.shape == carr.shape
    mag = np.abs(carr)
    err_real = np.mean((real - true) ** 2)
    err_mag = np.mean((mag - true) ** 2)
    assert err_real < err_mag
    assert abs(np.mean(imag)) < 0.05


def test_rephase_dc_outlier_detection_runs():
    from qsiprep.interfaces.phase import rephase_dc

    carr, _ = _synth_complex()
    real, imag, phase_bg = rephase_dc(carr, kernel_name='Opt3', outlier_detection=True)
    assert real.shape == carr.shape


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
