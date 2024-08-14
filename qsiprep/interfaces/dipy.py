#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import subprocess

import nibabel as nb
import numpy as np
from nilearn.image import load_img
from nipype import logging
from nipype.interfaces.base import File, SimpleInterface, traits
from nipype.utils.filemanip import fname_presuffix

from .. import config
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
)
from .patch2self import patch2self

LOGGER = logging.getLogger("nipype.interface")
TAU_DEFAULT = 1.0 / (4 * np.pi**2)


def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    config.loggers.interface.info(out)
    config.loggers.interface.info(err)


class Patch2SelfInputSpec(SeriesPreprocReportInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D diffusion MRI data file")
    patch_radius = traits.Either(traits.Int(0), traits.Str("auto"), desc="patch radius in voxels.")
    bval_file = File(exists=True, mandatory=True, desc="bval file containing b-values")
    model = traits.Str("ols", usedefault=True, desc="Regression model for Patch2Self")
    alpha = traits.Float(1.0, usedefault=True, desc="Regularization parameter for Ridge and Lasso")
    b0_threshold = traits.Float(50.0, usedefault=True, desc="Threshold to segregate b0s")
    mask = File(desc="mask image (unused)")
    b0_denoising = traits.Bool(True, usedefault=True, desc="denoise the b=0 images too")
    clip_negative_vals = traits.Bool(
        False, usedefault=True, desc="Sets negative values after denoising to 0"
    )
    shift_intensity = traits.Bool(
        True,
        usedefault=True,
        desc="Shifts the distribution of intensities per " "volume to give non-negative values",
    )
    out_report = File(
        "patch2self_report.svg", usedefault=True, desc="filename for the visual report"
    )


class Patch2SelfOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(exists=True, desc="Denoised version of the input image")
    noise_image = File(exists=True, desc="Residuals depicting suppressed noise")


class Patch2Self(SeriesPreprocReport, SimpleInterface):
    input_spec = Patch2SelfInputSpec
    output_spec = Patch2SelfOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file
        bval_file = self.inputs.bval_file
        denoised_file = fname_presuffix(
            in_file, suffix="_denoised_patch2self", newpath=runtime.cwd
        )
        noise_file = fname_presuffix(
            in_file, suffix="_denoised_residuals_patch2self", newpath=runtime.cwd
        )
        noisy_img = nb.load(in_file)
        noisy_arr = noisy_img.get_fdata()
        bvals = np.loadtxt(bval_file)

        # Determine the patch radius
        num_non_b0 = (bvals > self.inputs.b0_threshold).sum()
        very_few_directions = num_non_b0 < 20
        few_directions = num_non_b0 < 50
        if self.inputs.patch_radius == "auto":
            if very_few_directions:
                patch_radius = [3, 3, 3]
            elif few_directions:
                patch_radius = [1, 1, 1]
            else:
                patch_radius = [0, 0, 0]
        else:
            patch_radius = [self.inputs.patch_radius] * 3
            if self.inputs.patch_radius > 3 and not very_few_directions:
                LOGGER.info(
                    "a very large patch radius is not necessary when more than "
                    "20 gradient directions have been sampled."
                )
            elif self.inputs.patch_radius > 1 and not few_directions:
                LOGGER.info(
                    "a large patch radius is not necessary when more than "
                    "50 gradient directions have been sampled."
                )
            elif self.inputs.patch_radius == 0 and few_directions:
                LOGGER.warning(
                    "When < 50 gradient directions are available, it is "
                    "recommended to increase patch_radius to > 0."
                )

        denoised_arr, noise_residuals = patch2self(
            noisy_arr,
            bvals,
            model=self.inputs.model,
            alpha=self.inputs.alpha,
            patch_radius=patch_radius,
            b0_threshold=self.inputs.b0_threshold,
            verbose=True,
            b0_denoising=self.inputs.b0_denoising,
            clip_negative_vals=self.inputs.clip_negative_vals,
            shift_intensity=self.inputs.shift_intensity,
        )

        # Back to nifti
        denoised_img = nb.Nifti1Image(denoised_arr, noisy_img.affine, noisy_img.header)
        p2s_residuals = nb.Nifti1Image(noise_residuals, noisy_img.affine, noisy_img.header)
        denoised_img.to_filename(denoised_file)
        p2s_residuals.to_filename(noise_file)
        self._results["out_file"] = denoised_file
        self._results["noise_image"] = noise_file
        return runtime

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get("out_file")
        denoised_nii = load_img(ref_name)
        noise_name = outputs["noise_image"]
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii
