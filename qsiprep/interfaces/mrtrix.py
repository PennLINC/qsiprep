#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRtrix3 Interfaces
~~~~~~~~~~~~~~~~~~


"""
import os

import nibabel as nb
import numpy as np
from nilearn.image import load_img, threshold_img
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mrtrix3 import MRConvert
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec
from nipype.utils.filemanip import fname_presuffix, which
from niworkflows.viz.utils import compose_view, cuts_from_bbox

from ..viz.utils import plot_denoise
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
)

LOGGER = logging.getLogger("nipype.interface")
RC3_ROOT = which("average_response")  # Only exists in RC3
if RC3_ROOT is not None:
    # Use the directory containing average_response
    RC3_ROOT = os.path.split(RC3_ROOT)[0]


class MRTrixGradientTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)


class MRTrixGradientTableOutputSpec(TraitedSpec):
    gradient_file = File(exists=True)


class MRTrixGradientTable(SimpleInterface):
    input_spec = MRTrixGradientTableInputSpec
    output_spec = MRTrixGradientTableOutputSpec

    def _run_interface(self, runtime):
        gtab_fname = fname_presuffix(
            self.inputs.bval_file, suffix=".b", newpath=runtime.cwd, use_ext=False
        )
        _convert_fsl_to_mrtrix(self.inputs.bval_file, self.inputs.bvec_file, gtab_fname)
        self._results["gradient_file"] = gtab_fname
        return runtime


def _convert_fsl_to_mrtrix(bval_file, bvec_file, output_fname):
    vecs = np.loadtxt(bvec_file)
    vals = np.loadtxt(bval_file)
    gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
    np.savetxt(output_fname, gtab, fmt=["%.8f", "%.8f", "%.8f", "%d"])


class MRTrixIngressInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    suffix = traits.Str("", usedefault=True)


class MRTrixIngressOutputSpec(TraitedSpec):
    mif_file = File()


class MRTrixIngress(SimpleInterface):
    input_spec = MRTrixIngressInputSpec
    output_spec = MRTrixIngressOutputSpec

    def _run_interface(self, runtime):
        output_mif = fname_presuffix(
            self.inputs.dwi_file,
            suffix=self.inputs.suffix + ".mif",
            newpath=runtime.cwd,
            use_ext=False,
        )
        if isdefined(self.inputs.b_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file, grad_file=self.inputs.b_file, out_file=output_mif
            )
        elif isdefined(self.inputs.bval_file) and isdefined(self.inputs.bvec_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file,
                in_bval=self.inputs.bval_file,
                in_bvec=self.inputs.bvec_file,
                out_file=output_mif,
            )
        else:
            raise Exception("No valid mrtrix gradient files or fsl bval/bvec files specified")
        convert_run = convert.run()
        self._results["mif_file"] = convert_run.outputs.out_file

        return runtime


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(exists=True, argstr="%s", position=-2, mandatory=True, desc="input DWI image")
    mask = File(exists=True, argstr="-mask %s", position=1, desc="mask image")
    extent = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        argstr="-extent %d,%d,%d",
        desc="set the window size of the denoising filter. (default = 5,5,5)",
    )
    noise_image = File(
        argstr="-noise %s",
        name_template="%s_noise.nii.gz",
        name_source=["in_file"],
        keep_extension=False,
        desc="the output noise map",
    )
    out_file = File(
        name_template="%s_denoised.nii.gz",
        name_source=["in_file"],
        keep_extension=False,
        argstr="%s",
        position=-1,
        desc="the output denoised DWI image",
    )
    out_report = File(
        "dwidenoise_report.svg", usedefault=True, desc="filename for the visual report"
    )


class DWIDenoiseOutputSpec(SeriesPreprocReportOutputSpec):
    noise_image = File(desc="the output noise map", exists=True)
    out_file = File(desc="the output denoised DWI image", exists=True)


class DWIDenoise(SeriesPreprocReport, MRTrix3Base):
    """
    Denoise DWI data and estimate the noise level based on the optimal
    threshold for PCA.

    DWI data denoising and noise map estimation by exploiting data redundancy
    in the PCA domain using the prior knowledge that the eigenspectrum of
    random covariance matrices is described by the universal Marchenko Pastur
    distribution.

    Important note: image denoising must be performed as the first step of the
    image processing pipeline. The routine will fail if interpolation or
    smoothing has been applied to the data prior to denoising.

    Note that this function does not correct for non-Gaussian noise biases.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/dwidenoise.html>

    """

    _cmd = "dwidenoise"
    input_spec = DWIDenoiseInputSpec
    output_spec = DWIDenoiseOutputSpec

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get("out_file")
        denoised_nii = load_img(ref_name)
        noise_name = outputs["noise_image"]
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class DWIBiasCorrectInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(exists=True, argstr="%s", position=-2, mandatory=True, desc="input DWI image")
    mask = File(argstr="-mask %s", desc="input mask image for bias field estimation")
    method = traits.Enum("ants", "fsl", argstr="%s", position=1, usedefault=True)
    bias_image = File(
        argstr="-bias %s",
        name_source="in_file",
        name_template="%s_bias.nii.gz",
        keep_extension=False,
        desc="bias field",
    )
    out_file = File(
        name_source="in_file",
        keep_extension=False,
        argstr="%s",
        name_template="%s_N4.nii.gz",
        position=-1,
        desc="the output bias corrected DWI image",
    )
    ants_b = traits.Str(default_value="[150,3]", argstr="-ants.b %s", usedefault=True)
    ants_c = traits.Str(default_value="[200x200,1e-6]", argstr="-ants.c %s", usedefault=True)
    ants_s = traits.Str(default_value="4", argstr="-ants.s %s")
    out_report = File("n4_report.svg", usedefault=True, desc="filename for the visual report")
    bzero_max = traits.Int(
        argstr="-config BZeroThreshold %d",
        desc="Maximum b-value that can be considered a b=0",
    )


class DWIBiasCorrectOutputSpec(SeriesPreprocReportOutputSpec):
    bias_image = File(desc="the output bias field", exists=True)
    out_file = File(desc="the output bias corrected DWI image", exists=True)


class DWIBiasCorrect(SeriesPreprocReport, MRTrix3Base):
    """
    Perform B1 field inhomogeneity correction for a DWI volume series.
    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/scripts/dwibiascorrect.html>
    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bias_correct = mrt.DWIBiasCorrect()
    >>> bias_correct.inputs.in_file = 'dwi.mif'
    >>> bias_correct.inputs.method = 'ants'
    >>> bias_correct.cmdline
    'dwibiascorrect ants dwi.mif dwi_biascorr.mif'
    >>> bias_correct.run()                             # doctest: +SKIP
    """

    _cmd = "dwibiascorrect"
    input_spec = DWIBiasCorrectInputSpec
    output_spec = DWIBiasCorrectOutputSpec

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get("out_file")
        denoised_nii = load_img(ref_name)
        noise_name = outputs["bias_image"]
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class MRDeGibbsInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    out_report = File("degibbs_report.svg", usedefault=True, desc="filename for the visual report")
    in_file = File(exists=True, argstr="%s", position=-2, mandatory=True, desc="input DWI image")
    out_file = File(
        name_source="in_file",
        keep_extension=False,
        argstr="%s",
        name_template="%s_mrdegibbs.nii.gz",
        position=-1,
        desc="the output de-Gibbs'd DWI image",
    )
    mask = File(desc="input mask image for the visual report")
    nshifts = traits.Int(
        default=20, argstr="-nshifts %d", desc="discretization of subpixel spacing."
    )
    axes = traits.Enum(
        "0,1",
        "0,2",
        "1,2",
        default="0,1",
        argstr="-axes %s",
        desc="select the slice axes (default: 0,1 - i.e. x-y)",
    )
    minw = traits.Int(
        default=1, argstr="-minW %d", desc="left border of window used for TV computation"
    )
    maxw = traits.Int(
        default=3, argstr="-maxW %d", desc="right border of window used for TV computation"
    )


class MRDeGibbsOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(desc="the output de-Gibbs'd DWI image")


class MRDeGibbs(SeriesPreprocReport, MRTrix3Base):
    input_spec = MRDeGibbsInputSpec
    output_spec = MRDeGibbsOutputSpec
    _cmd = "mrdegibbs"

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get("out_file")
        denoised_nii = load_img(ref_name)
        return input_dwi, denoised_nii, None

    def _generate_report(self):
        """Generate a reportlet."""
        LOGGER.info("Generating denoising visual report")

        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
        image_intensities = np.array([img.mean() for img in image_data.T])
        lowb_index = int(np.argmax(image_intensities))
        highb_index = int(np.argmin(image_intensities))

        # Original images
        orig_lowb_nii = input_dwi.slicer[..., lowb_index]
        orig_highb_nii = input_dwi.slicer[..., highb_index]

        # Denoised images
        denoised_lowb_nii = denoised_nii.slicer[..., lowb_index]
        denoised_highb_nii = denoised_nii.slicer[..., highb_index]

        # Find spatial extent of the image
        contour_nii = mask_nii = None
        if isdefined(self.inputs.mask):
            contour_nii = load_img(self.inputs.mask)
        else:
            mask_nii = threshold_img(denoised_lowb_nii, 50)
        cuts = cuts_from_bbox(contour_nii or mask_nii, cuts=self._n_cuts)

        diff_lowb_nii = nb.Nifti1Image(
            orig_lowb_nii.get_fdata() - denoised_lowb_nii.get_fdata(),
            affine=denoised_lowb_nii.affine,
        )
        diff_highb_nii = nb.Nifti1Image(
            orig_highb_nii.get_fdata() - denoised_highb_nii.get_fdata(),
            affine=denoised_highb_nii.affine,
        )

        # Call composer
        compose_view(
            plot_denoise(
                denoised_lowb_nii,
                denoised_highb_nii,
                "moving-image",
                estimate_brightness=True,
                cuts=cuts,
                label="De-Gibbs",
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            plot_denoise(
                diff_lowb_nii,
                diff_highb_nii,
                "fixed-image",
                estimate_brightness=True,
                cuts=cuts,
                label="Estimated Ringing",
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            out_file=self._out_report,
        )

        self._calculate_nmse(input_dwi, denoised_nii)


class _ITKTransformConvertInputSpec(CommandLineInputSpec):
    in_transform = traits.File(exists=True, argstr="%s", mandatory=True, position=0)
    operation = traits.Enum(
        "itk_import", default="itk_import", usedefault=True, posision=1, argstr="%s"
    )
    out_transform = traits.File(
        argstr="%s",
        name_source="in_transform",
        name_template="%s.txt",
        keep_extension=False,
        position=-1,
    )


class _ITKTransformConvertOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ITKTransformConvert(CommandLine):
    _cmd = "transformconvert"
    input_spec = _ITKTransformConvertInputSpec
    output_spec = _ITKTransformConvertOutputSpec


class _TransformHeaderInputSpec(CommandLineInputSpec):
    transform_file = traits.File(exists=True, position=0, mandatory=True, argstr="-linear %s")
    in_image = traits.File(exists=True, mandatory=True, position=1, argstr="%s")
    out_image = traits.File(
        argstr="%s",
        name_source="in_image",
        name_template="%s_hdrxform.nii.gz",
        keep_extension=False,
        position=-1,
    )


class _TransformHeaderOutputSpec(TraitedSpec):
    out_image = File(exists=True)


class TransformHeader(CommandLine):
    input_spec = _TransformHeaderInputSpec
    output_spec = _TransformHeaderOutputSpec
    _cmd = "mrtransform -strides -1,-2,3"


class _PolarToComplexInputSpec(CommandLineInputSpec):
    mag_file = traits.File(exists=True, mandatory=True, position=0, argstr="%s")
    phase_file = traits.File(exists=True, mandatory=True, position=1, argstr="%s")
    out_file = traits.File(
        exists=False,
        name_source="mag_file",
        name_template="%s_complex.nii.gz",
        keep_extension=False,
        position=-1,
        argstr="-polar %s",
    )


class _PolarToComplexOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class PolarToComplex(CommandLine):
    """Convert a magnitude and phase image pair to a single complex image using mrcalc."""

    input_spec = _PolarToComplexInputSpec
    output_spec = _PolarToComplexOutputSpec

    _cmd = "mrcalc"


class _ComplexToMagnitudeInputSpec(CommandLineInputSpec):
    complex_file = traits.File(exists=True, mandatory=True, position=0, argstr="%s")
    out_file = traits.File(
        exists=False,
        name_source="complex_file",
        name_template="%s_mag.nii.gz",
        keep_extension=False,
        position=-1,
        argstr="-abs %s",
    )


class _ComplexToMagnitudeOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ComplexToMagnitude(CommandLine):
    """Extract the magnitude portion of a complex image using mrcalc."""

    input_spec = _ComplexToMagnitudeInputSpec
    output_spec = _ComplexToMagnitudeOutputSpec

    _cmd = "mrcalc"
