# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for complex-valued phase correction of DWI data.

The array-based rephasing helpers driven by :class:`PhaseCorrect` live in
:mod:`qsiprep.utils.phase`.
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

from ..utils.phase import rephase_dc, rephase_tv, rephase_tv_complex
from ..viz.utils import plot_denoise

LOGGER = logging.getLogger('nipype.interface')


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

    The background-phase estimators are implemented in
    :mod:`qsiprep.utils.phase` (:func:`~qsiprep.utils.phase.rephase_tv`,
    :func:`~qsiprep.utils.phase.rephase_tv_complex`, and
    :func:`~qsiprep.utils.phase.rephase_dc`).
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
