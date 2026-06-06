# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for complex-valued phase correction of DWI data.

The rephasing helpers below are adapted from the BSD-2-Clause-licensed MRItools
package by Francesco Grussu (University College London):
``imgrephaseTV.py`` (Eichner C et al, NeuroImage 2015, 122:373-384) and
``imgrephaseDC.py`` (Sprenger T et al, Magnetic Resonance in Medicine 2017,
77:559-570).

STATEMENT OF CHANGES: the original file-based scripts were refactored into
array-based functions operating on a single complex NIfTI, the per-slice loops
were preserved, file I/O was removed, and the scikit-image call was updated to
the modern ``channel_axis`` API. Copyright (c) 2020 University College London.
All rights reserved.
"""

import os

import nibabel as nb
import numpy as np
from nilearn.image import threshold_img
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.viz.utils import compose_view, cuts_from_bbox
from scipy import ndimage, signal
from skimage.restoration import denoise_tv_bregman

from ..viz.utils import plot_denoise

LOGGER = logging.getLogger('nipype.interface')

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
    """Total-variation rephasing (Eichner 2015).

    Parameters
    ----------
    carr : numpy.ndarray
        Complex 3D or 4D array.
    weight : float
        Total-variation regularization weight.

    Returns
    -------
    real, imag, phase_bg : numpy.ndarray
        Real and imaginary channels after rephasing, and the estimated
        background phase.
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
    """Paper-faithful TV rephasing (Eichner 2015, Eqs. 5 & 8).

    Estimates the smooth background phase from a total-variation denoised
    *complex* image -- ``phi_bg = angle(TV(complex signal))`` -- rather than from
    a TV-denoised ``(magnitude, phase)`` stack as :func:`rephase_tv` does. This
    matches the paper, which solves the complex-valued TV problem
    ``argmin_x 1/2||y - x||_2^2 + lambda||grad x||_1`` over ``x in C^n`` and takes
    the phase of ``x`` as the background phase. Operating on the complex signal
    is wrap-free (no phase unwrapping) and intensity-weighted by construction.

    Implemented as vectorial (2-channel) TV on the real and imaginary parts,
    which is the complex TV of the paper. The signal is normalized by a robust
    per-slice magnitude before denoising so the regularization ``weight`` is
    invariant to the (large, scanner-dependent) magnitude scale; without this,
    scikit-image's scale-dependent ``weight`` barely smooths DWI-scale data and
    ``phi_bg`` collapses onto the noisy phase.

    Parameters
    ----------
    carr : numpy.ndarray
        Complex 3D or 4D array.
    weight : float
        Total-variation regularization weight (the paper's ``lambda``; default
        6.0 elsewhere). The scikit-image fidelity weight is ``2.0 * weight``,
        matching the convention used by :func:`rephase_tv`.

    Returns
    -------
    real, imag, phase_bg : numpy.ndarray
        Real and imaginary channels after rephasing, and the estimated
        background phase.
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
    """Decorrelated-phase rephasing (Sprenger 2017).

    Parameters
    ----------
    carr : numpy.ndarray
        Complex 3D or 4D array.
    kernel_name : str
        One of the keys in :data:`DC_KERNELS`.
    outlier_detection : bool
        If True, replace rephased-real outliers with the magnitude.

    Returns
    -------
    real, imag, phase_bg : numpy.ndarray
        Real and imaginary channels after rephasing, and the estimated
        background phase.
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


class _PhaseCorrectInputSpec(BaseInterfaceInputSpec):
    complex_file = File(exists=True, mandatory=True, desc='complex-valued denoised DWI NIfTI')
    method = traits.Enum('tv', 'tvc', 'dc', mandatory=True, desc='phase-correction method')
    tv_weight = traits.Float(6.0, usedefault=True, desc='TV regularization weight')
    dc_kernel = traits.Enum(
        'Opt5',
        'B3',
        'B5',
        'G3F1',
        'G5F2',
        'G3F1H',
        'G5F2H',
        'Opt3',
        usedefault=True,
        desc='DC convolution kernel',
    )
    dc_outlier_detection = traits.Bool(False, usedefault=True, desc='enable DC outlier detection')
    out_report = File(
        'phasecorrection_report.svg',
        usedefault=True,
        desc='filename for the visual report',
    )


class _PhaseCorrectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='real channel after phase correction')
    out_report = File(desc='visual report')


class PhaseCorrect(SimpleInterface):
    """Apply phase correction to complex DWI data and keep the real channel.

    Reads a complex-valued NIfTI (e.g. the output of complex ``dwidenoise``),
    estimates a smooth background phase using the selected method, rephases the
    data, and writes the real channel as a float32 NIfTI. The imaginary channel
    should then contain mostly Gaussian noise.
    """

    input_spec = _PhaseCorrectInputSpec
    output_spec = _PhaseCorrectOutputSpec

    def _rephase_volume(self, cvol):
        """Phase-correct a single 3D complex volume, returning (real, imag)."""
        if self.inputs.method == 'tv':
            real, imag, _ = rephase_tv(cvol, self.inputs.tv_weight)
        elif self.inputs.method == 'tvc':
            real, imag, _ = rephase_tv_complex(cvol, self.inputs.tv_weight)
        else:
            real, imag, _ = rephase_dc(
                cvol,
                self.inputs.dc_kernel,
                self.inputs.dc_outlier_detection,
            )
        return real, imag

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.complex_file)
        # IMPORTANT: get_fdata() would coerce to float and drop the imaginary
        # part; read the raw (complex) array instead. Keep it as complex64 (not
        # complex128) and process one volume at a time: upcasting a whole 4D DWI
        # series to complex128 would need tens of GB for a typical acquisition.
        cdata = np.asanyarray(img.dataobj)
        if not np.iscomplexobj(cdata):
            cdata = cdata.astype(np.complex64)
        elif cdata.dtype != np.complex64:
            cdata = cdata.astype(np.complex64)

        hdr = img.header.copy()
        hdr.set_data_dtype(np.float32)
        out_file = fname_presuffix(
            self.inputs.complex_file,
            suffix='_real.nii.gz',
            newpath=runtime.cwd,
            use_ext=False,
        )

        if cdata.ndim == 3:
            real, imag = self._rephase_volume(cdata)
            out = real.astype(np.float32, copy=False)
            report = (real, imag, real, imag)
        elif cdata.ndim == 4:
            nvol = cdata.shape[3]
            out = np.empty(cdata.shape, dtype=np.float32)
            vol_means = np.empty(nvol, dtype=np.float64)
            for vol in range(nvol):
                real_v, _imag_v = self._rephase_volume(cdata[..., vol])
                out[..., vol] = real_v
                vol_means[vol] = real_v.mean()
            # Recompute only the two report volumes to recover their imaginary
            # residuals (cheaper than retaining the full imaginary series).
            lowb_index = int(np.argmax(vol_means))
            highb_index = int(np.argmin(vol_means))
            real_low, imag_low = self._rephase_volume(cdata[..., lowb_index])
            real_high, imag_high = self._rephase_volume(cdata[..., highb_index])
            report = (real_low, imag_low, real_high, imag_high)
        else:
            raise ValueError('PhaseCorrect expects a 3D or 4D complex image')

        nb.Nifti1Image(out, img.affine, hdr).to_filename(out_file)
        self._results['out_file'] = out_file

        try:
            self._generate_report(report, img.affine)
            self._results['out_report'] = self._out_report
        except Exception as exc:  # reports are non-critical
            LOGGER.warning('Phase-correction reportlet failed: %s', exc)

        return runtime

    def _generate_report(self, report, affine):
        """Build a real-channel vs imaginary-residual reportlet."""
        self._out_report = os.path.abspath(self.inputs.out_report)

        real_low, imag_low, real_high, imag_high = report
        real_lowb = nb.Nifti1Image(real_low, affine)
        real_highb = nb.Nifti1Image(real_high, affine)
        imag_lowb = nb.Nifti1Image(imag_low, affine)
        imag_highb = nb.Nifti1Image(imag_high, affine)

        mask_nii = threshold_img(real_lowb, 1e-3)
        cuts = cuts_from_bbox(mask_nii, cuts=7)

        compose_view(
            plot_denoise(
                real_lowb,
                real_highb,
                'moving-image',
                estimate_brightness=True,
                cuts=cuts,
                label='Real (phase-corrected)',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            plot_denoise(
                imag_lowb,
                imag_highb,
                'fixed-image',
                estimate_brightness=True,
                cuts=cuts,
                label='Imaginary residual',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            out_file=self._out_report,
        )
