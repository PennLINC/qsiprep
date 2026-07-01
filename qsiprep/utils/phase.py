# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: The rephasing helpers in this file (``rephase_tv``,
# ``rephase_tv_complex``, ``rephase_dc``, ``_dc_outlier_real``, the per-slice
# convolution helper, and the ``DC_KERNELS`` coefficients) are derived from the
# BSD-2-Clause-licensed MRItools package by Francesco Grussu (University College
# London), specifically the scripts:
#   - ``tools/imgrephaseTV.py`` (Eichner C et al, NeuroImage 2015, 122:373-384)
#   - ``tools/imgrephaseDC.py`` (Sprenger T et al, Magnetic Resonance in
#     Medicine 2017, 77:559-570)
#   https://github.com/fragrussu/MRItools
# The original file-based command-line scripts were refactored into array-based
# functions operating on a single complex NIfTI, file I/O was removed, the
# per-slice loops were preserved, the ``rephase_tv_complex`` variant was added,
# and the scikit-image calls were updated from the deprecated ``multichannel``
# argument to the modern ``channel_axis`` API.
#
# ORIGINAL WORK'S ATTRIBUTION NOTICE:
#
#    Code released under BSD Two-Clause license.
#
#    Copyright (c) 2020 University College London.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#    POSSIBILITY OF SUCH DAMAGE.
#
"""Utilities for complex-valued phase correction of DWI data.

The rephasing helpers below are adapted from the BSD-2-Clause-licensed MRItools
package by Francesco Grussu (University College London):
``imgrephaseTV.py`` (Eichner C et al, NeuroImage 2015, 122:373-384) and
``imgrephaseDC.py`` (Sprenger T et al, Magnetic Resonance in Medicine 2017,
77:559-570). See the BSD 2-Clause attribution notice at the top of this file.

These are pure-array utility functions; the Nipype interface that drives them
lives in :mod:`qsiprep.interfaces.phase`.
"""

import numpy as np
from scipy import ndimage, signal
from skimage.restoration import denoise_tv_bregman

DC_KERNELS = {
    'B3': np.full((3, 3), 1.0 / 9.0),
    'B5': np.full((5, 5), 1.0 / 25.0),
    'G3F1': np.array(
        [
            [0.075113607954111, 0.123841403152974, 0.075113607954111],
            [0.123841403152974, 0.204179955571658, 0.123841403152974],
            [0.075113607954111, 0.123841403152974, 0.075113607954111],
        ]
    ),
    'G5F2': np.array(
        [
            [
                0.023246839878294,
                0.033823952439922,
                0.038327559383904,
                0.033823952439922,
                0.023246839878294,
            ],
            [
                0.033823952439922,
                0.049213560408541,
                0.055766269846849,
                0.049213560408541,
                0.033823952439922,
            ],
            [
                0.038327559383904,
                0.055766269846849,
                0.063191462410265,
                0.055766269846849,
                0.038327559383904,
            ],
            [
                0.033823952439922,
                0.049213560408541,
                0.055766269846849,
                0.049213560408541,
                0.033823952439922,
            ],
            [
                0.023246839878294,
                0.033823952439922,
                0.038327559383904,
                0.033823952439922,
                0.023246839878294,
            ],
        ]
    ),
    'G3F1H': np.array(
        [
            [0.075113607954111, 0.123841403152974, 0.075113607954111],
            [0.123841403152974, 0.0, 0.123841403152974],
            [0.075113607954111, 0.123841403152974, 0.075113607954111],
        ]
    ),
    'G5F2H': np.array(
        [
            [
                0.023246839878294,
                0.033823952439922,
                0.038327559383904,
                0.033823952439922,
                0.023246839878294,
            ],
            [
                0.033823952439922,
                0.049213560408541,
                0.055766269846849,
                0.049213560408541,
                0.033823952439922,
            ],
            [0.038327559383904, 0.055766269846849, 0.0, 0.055766269846849, 0.038327559383904],
            [
                0.033823952439922,
                0.049213560408541,
                0.055766269846849,
                0.049213560408541,
                0.033823952439922,
            ],
            [
                0.023246839878294,
                0.033823952439922,
                0.038327559383904,
                0.033823952439922,
                0.023246839878294,
            ],
        ]
    ),
    'Opt3': np.array(
        [
            [0.107235538162453, 0.142764461837547, 0.107235538162453],
            [0.142764461837547, 0.0, 0.142764461837547],
            [0.107235538162453, 0.142764461837547, 0.107235538162453],
        ]
    ),
    'Opt5': np.array(
        [
            [
                0.025441320175391,
                0.037016902431746,
                0.041945645727859,
                0.037016902431746,
                0.025441320175391,
            ],
            [
                0.037016902431746,
                0.053859275233950,
                0.054719953999307,
                0.053859275233950,
                0.037016902431746,
            ],
            [0.041945645727859, 0.054719953999307, 0.0, 0.054719953999307, 0.041945645727859],
            [
                0.037016902431746,
                0.053859275233950,
                0.054719953999307,
                0.053859275233950,
                0.037016902431746,
            ],
            [
                0.025441320175391,
                0.037016902431746,
                0.041945645727859,
                0.037016902431746,
                0.025441320175391,
            ],
        ]
    ),
}


def rephase_tv(carr, weight):
    """Rephase complex MRI data using total-variation phase smoothing.

    Parameters
    ----------
    carr : :obj:`numpy.ndarray` of shape (X, Y, Z) or (X, Y, Z, V)
        Complex-valued image, 3D (single volume) or 4D (a series of ``V``
        volumes). The array is internally cast to ``complex128``. For a 4D
        input, each volume along the last axis is processed independently.
    weight : :obj:`float`
        Total-variation regularization weight (the ``lambda`` of
        :footcite:t:`eichner2015real`). Larger values yield a smoother estimated
        background phase. The weight passed to scikit-image's
        :func:`~skimage.restoration.denoise_tv_bregman` is ``2.0 * weight``.

    Returns
    -------
    real : :obj:`numpy.ndarray`
        Real channel after rephasing, ``float64``, same shape as ``carr``. This
        is the noise-floor-free channel retained for downstream processing.
    imag : :obj:`numpy.ndarray`
        Imaginary channel after rephasing, ``float64``, same shape as ``carr``.
        After successful rephasing this should contain mostly Gaussian noise.
    phase_bg : :obj:`numpy.ndarray`
        Estimated smooth background phase in radians, ``float64``, same shape as
        ``carr``.

    Notes
    -----
    This implements the total-variation rephasing of
    :footcite:t:`eichner2015real`. Diffusion-weighted images carry a smoothly
    varying background phase (from eddy currents, B0 inhomogeneity, and motion)
    on top of the desired real-valued signal. If that background phase is
    removed, the true signal collapses onto the real channel and the noise
    becomes Gaussian (rather than Rician), which avoids the magnitude noise
    floor and enables true signal averaging.

    The background phase is estimated per in-plane slice. The magnitude and
    wrapped phase are stacked and denoised jointly as a single image with
    total-variation (Split-Bregman) regularization, and the denoised phase
    channel is taken as the background phase ``phi_bg``. The data are then
    rephased by ``carr * exp(-1j * phi_bg)``, rotating the smooth background
    phase to zero. Because the phase is denoised directly, this estimator is
    sensitive to phase wrapping; the wrap-free :func:`rephase_tv_complex` and
    :func:`rephase_dc` variants avoid that limitation.

    References
    ----------
    .. footbibliography::
    """
    carr = np.asarray(carr, dtype=np.complex128)
    mag = np.abs(carr)
    phase = np.angle(carr)
    phase_bg = np.zeros(carr.shape, dtype=np.float64)

    # The magnitude/phase pair is denoised jointly as a single volumetric image
    # (``channel_axis=None``), matching the reference ``imgrephaseTV.py`` (which
    # relied on the old scikit-image ``multichannel=False`` default). Using
    # ``channel_axis=-1`` instead performs *vectorial* 2-channel TV, which
    # couples the (large dynamic range) magnitude edges into the phase estimate
    # and imprints anatomy into the background phase, leaving coherent anatomical
    # signal in the imaginary channel after rephasing.
    if carr.ndim == 3:
        for zz in range(carr.shape[2]):
            buff = denoise_tv_bregman(
                np.dstack((mag[:, :, zz], phase[:, :, zz])),
                weight=2.0 * float(weight),
                channel_axis=None,
            )
            phase_bg[:, :, zz] = buff[:, :, 1]
    elif carr.ndim == 4:
        for vv in range(carr.shape[3]):
            for zz in range(carr.shape[2]):
                buff = denoise_tv_bregman(
                    np.dstack((mag[:, :, zz, vv], phase[:, :, zz, vv])),
                    weight=2.0 * float(weight),
                    channel_axis=None,
                )
                phase_bg[:, :, zz, vv] = buff[:, :, 1]
    else:
        raise ValueError('rephase_tv expects a 3D or 4D array')

    rephased = carr * np.exp(-1j * phase_bg)
    return np.real(rephased), np.imag(rephased), phase_bg


def rephase_tv_complex(carr, weight):
    """Rephase complex MRI data using paper-faithful complex total variation.

    Parameters
    ----------
    carr : :obj:`numpy.ndarray` of shape (X, Y, Z) or (X, Y, Z, V)
        Complex-valued image, 3D (single volume) or 4D (a series of ``V``
        volumes). The array is internally cast to ``complex128``. For a 4D
        input, each volume along the last axis is processed independently.
    weight : :obj:`float`
        Total-variation regularization weight (the ``lambda`` of
        :footcite:t:`eichner2015real`). Larger values yield a smoother estimated
        background phase. The weight passed to scikit-image's
        :func:`~skimage.restoration.denoise_tv_bregman` is ``2.0 * weight``,
        matching the convention used by :func:`rephase_tv`.

    Returns
    -------
    real : :obj:`numpy.ndarray`
        Real channel after rephasing, ``float64``, same shape as ``carr``. This
        is the noise-floor-free channel retained for downstream processing.
    imag : :obj:`numpy.ndarray`
        Imaginary channel after rephasing, ``float64``, same shape as ``carr``.
        After successful rephasing this should contain mostly Gaussian noise.
    phase_bg : :obj:`numpy.ndarray`
        Estimated smooth background phase in radians, ``float64``, same shape as
        ``carr``.

    Notes
    -----
    This is a paper-faithful variant of the total-variation rephasing of
    :footcite:t:`eichner2015real`. Where :func:`rephase_tv` denoises a stacked
    ``(magnitude, phase)`` image and reads off the phase channel, this function
    estimates the background phase directly from a TV-denoised *complex* image,
    ``phi_bg = angle(TV(complex signal))``. This corresponds to the complex
    total-variation problem solved in the paper,

    .. math::

        \\hat{x} = \\arg\\min_{x \\in \\mathbb{C}^n}
        \\tfrac{1}{2}\\lVert y - x \\rVert_2^2
        + \\lambda \\lVert \\nabla x \\rVert_1,

    where ``y`` is the measured complex signal and ``angle(x_hat)`` is taken as
    the background phase. Operating on the complex signal is wrap-free (no phase
    unwrapping is required) and is intensity-weighted by construction, so it does
    not imprint magnitude anatomy onto the phase estimate.

    The complex TV is realized per in-plane slice as a vectorial (2-channel)
    total-variation denoising of the stacked real and imaginary parts. Before
    denoising, each slice is normalized by a robust (95th-percentile) magnitude
    so that the regularization ``weight`` is invariant to the large,
    scanner-dependent magnitude scale; without this normalization scikit-image's
    scale-dependent weight barely smooths DWI-scale data and ``phi_bg`` collapses
    onto the noisy measured phase. The data are then rephased by
    ``carr * exp(-1j * phi_bg)``.

    References
    ----------
    .. footbibliography::
    """
    carr = np.asarray(carr, dtype=np.complex128)
    phase_bg = np.zeros(carr.shape, dtype=np.float64)
    sk_weight = 2.0 * float(weight)

    def _slice_phase_bg(cslice):
        scale = np.percentile(np.abs(cslice), 95) + 1e-9
        cnorm = cslice / scale
        denoised = denoise_tv_bregman(
            np.dstack((cnorm.real, cnorm.imag)),
            weight=sk_weight,
            channel_axis=-1,  # vectorial TV over (real, imag) == complex TV
        )
        return np.angle(denoised[:, :, 0] + 1j * denoised[:, :, 1])

    if carr.ndim == 3:
        for zz in range(carr.shape[2]):
            phase_bg[:, :, zz] = _slice_phase_bg(carr[:, :, zz])
    elif carr.ndim == 4:
        for vv in range(carr.shape[3]):
            for zz in range(carr.shape[2]):
                phase_bg[:, :, zz, vv] = _slice_phase_bg(carr[:, :, zz, vv])
    else:
        raise ValueError('rephase_tv_complex expects a 3D or 4D array')

    rephased = carr * np.exp(-1j * phase_bg)
    return np.real(rephased), np.imag(rephased), phase_bg


def _convolve_per_slice(arr, kernel):
    out = np.zeros_like(arr)
    if arr.ndim == 3:
        for zz in range(arr.shape[2]):
            out[:, :, zz] = ndimage.convolve(arr[:, :, zz], kernel, mode='constant', cval=0.0)
    elif arr.ndim == 4:
        for vv in range(arr.shape[3]):
            for zz in range(arr.shape[2]):
                out[:, :, zz, vv] = ndimage.convolve(
                    arr[:, :, zz, vv], kernel, mode='constant', cval=0.0
                )
    else:
        raise ValueError('expected a 3D or 4D array')
    return out


def _dc_outlier_real(real, imag, ksize):
    """Replace rephased-real outliers with the magnitude (Sprenger 2017).

    A voxel is an outlier when |magnitude - real| exceeds a local MAD-based
    threshold computed on the imaginary channel within each in-plane slice.
    """
    mag = np.sqrt(real * real + imag * imag)
    delta = np.abs(mag - real)
    thresh = np.zeros_like(real)

    def _slice_thresh(islice):
        absdev = np.abs(islice - signal.medfilt(islice, ksize))
        medabsdev = signal.medfilt(absdev, ksize)
        return 2.5 * 1.4826 * medabsdev

    if real.ndim == 3:
        for zz in range(real.shape[2]):
            thresh[:, :, zz] = _slice_thresh(imag[:, :, zz])
    elif real.ndim == 4:
        for vv in range(real.shape[3]):
            for zz in range(real.shape[2]):
                thresh[:, :, zz, vv] = _slice_thresh(imag[:, :, zz, vv])
    else:
        raise ValueError('expected a 3D or 4D array')

    real_thresh = mag.copy()  # outliers fall back to magnitude
    keep = delta < thresh
    real_thresh[keep] = real[keep]
    return real_thresh


def rephase_dc(carr, kernel_name, outlier_detection=False):
    """Rephase complex MRI data using decorrelated phase filtering.

    Parameters
    ----------
    carr : :obj:`numpy.ndarray` of shape (X, Y, Z) or (X, Y, Z, V)
        Complex-valued image, 3D (single volume) or 4D (a series of ``V``
        volumes). The array is internally cast to ``complex128``. For a 4D
        input, each volume along the last axis is processed independently.
    kernel_name : :obj:`str`
        Name of the 2D smoothing kernel used to estimate the background phase.
        One of the keys of :data:`DC_KERNELS` (``'B3'``, ``'B5'``, ``'G3F1'``,
        ``'G5F2'``, ``'G3F1H'``, ``'G5F2H'``, ``'Opt3'``, ``'Opt5'``). The
        ``F``/``H`` and ``Opt`` kernels have a zeroed center, decoupling the
        background-phase estimate at a voxel from that voxel's own value.
    outlier_detection : :obj:`bool`, optional
        If ``True``, voxels whose rephased real channel departs strongly from
        the magnitude are flagged as outliers and replaced by the magnitude
        (default: ``False``). See :func:`_dc_outlier_real`.

    Returns
    -------
    real : :obj:`numpy.ndarray`
        Real channel after rephasing, ``float64``, same shape as ``carr``. This
        is the noise-floor-free channel retained for downstream processing. When
        ``outlier_detection`` is ``True``, flagged voxels hold the magnitude
        instead.
    imag : :obj:`numpy.ndarray`
        Imaginary channel after rephasing, ``float64``, same shape as ``carr``.
        After successful rephasing this should contain mostly Gaussian noise.
    phase_bg : :obj:`numpy.ndarray`
        Estimated smooth background phase in radians, ``float64``, same shape as
        ``carr``.

    Raises
    ------
    ValueError
        If ``kernel_name`` is not a key of :data:`DC_KERNELS`, or if ``carr`` is
        neither 3D nor 4D.

    Notes
    -----
    This implements the decorrelated-phase filtering of
    :footcite:t:`sprenger2017real`. As with the total-variation methods, the goal
    is to estimate and remove the smooth background phase so that the true signal
    is recovered on the real channel with Gaussian (rather than Rician) noise,
    avoiding the magnitude noise floor.

    Rather than total-variation denoising, the smooth background phase is
    obtained by convolving the real and imaginary parts (separately, per
    in-plane slice) with a small smoothing kernel and taking the phase of the
    result, ``phi_bg = angle(conv(real) + 1j * conv(imag))``. Because real and
    imaginary parts are smoothed independently, this is wrap-free. The
    center-zeroed kernels (the ``H``- and ``Opt``-family) implement the
    "decorrelation" of the paper: the phase estimate at each voxel is built only
    from its neighbors, so the estimate is statistically decorrelated from the
    voxel's own noise. The data are then rephased by ``carr * exp(-1j *
    phi_bg)``. Optionally, an outlier step (:func:`_dc_outlier_real`) detects
    voxels where rephasing failed and falls back to the magnitude there.

    References
    ----------
    .. footbibliography::
    """
    carr = np.asarray(carr, dtype=np.complex128)
    if kernel_name not in DC_KERNELS:
        raise ValueError(f'Unsupported DC kernel: {kernel_name}')
    kernel = DC_KERNELS[kernel_name]

    real_f = _convolve_per_slice(carr.real, kernel)
    imag_f = _convolve_per_slice(carr.imag, kernel)
    phase_bg = np.angle(real_f + 1j * imag_f)

    rephased = carr * np.exp(-1j * phase_bg)
    real = np.real(rephased)
    imag = np.imag(rephased)

    if outlier_detection:
        real = _dc_outlier_real(real, imag, kernel.shape)

    return real, imag, phase_bg
