#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import nibabel as nb
import numpy as np

from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, SimpleInterface
)

LOGGER = logging.getLogger('nipype.interface')


class MedianOtsuInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="b0 template image")
    num_pass = traits.Int(4, usedefault=True, desc='Number of pass of the median filter')
    median_radius = traits.Int(4, usedefault=True,
                               desc='Radius (in voxels) of the applied median filter')
    dilate = traits.Int(6, usedefault=True, desc='Voxels to dilate after masking')


class MedianOtsuOutputSpec(TraitedSpec):
    masked_input = File(exists=True, desc='Masked version of the input image')
    out_mask = File(exists=True, desc='Median-Otsu mask of the input image')


class MedianOtsu(SimpleInterface):
    input_spec = MedianOtsuInputSpec
    output_spec = MedianOtsuOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file

        b0_img = nb.load(in_file)
        b0_data = b0_img.get_fdata()

        masked_data, data_mask = median_otsu(b0_data,
                                             median_radius=self.inputs.median_radius,
                                             numpass=self.inputs.num_pass,
                                             autocrop=False,
                                             dilate=self.inputs.dilate)

        self._results['out_mask'] = fname_presuffix(
            in_file, suffix='_mask', newpath=runtime.cwd)

        self._results['masked_input'] = fname_presuffix(
            in_file, suffix='_brain_masked', newpath=runtime.cwd)

        masked_img = nb.Nifti1Image(masked_data, b0_img.affine, b0_img.header)
        masked_img.to_filename(self._results['masked_input'])

        mask_img = nb.Nifti1Image(data_mask.astype('f8'), b0_img.affine, b0_img.header)
        mask_img.to_filename(self._results['out_mask'])

        return runtime


class HistEQInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to equalize')
    mask_file = File(exists=True, mandatory=True, desc='Mask image')


class HistEQOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='equalized file')


class HistEQ(SimpleInterface):
    input_spec = HistEQInputSpec
    output_spec = HistEQOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file

        uneq_img = nb.load(in_file)
        uneq_data = uneq_img.get_data()

        mask = nb.load(self.inputs.mask_file)
        bool_mask = mask.get_data() > 0
        data_voxels = uneq_data[bool_mask]

        # Do a clip on 2 to 98th percentile
        bottom_2, top_98 = np.percentile(data_voxels, np.array([1, 99]), axis=None)
        clipped_b0 = np.clip(data_voxels, 0, top_98)
        eq_data = histeq(clipped_b0, num_bins=512)
        output = np.zeros_like(mask.get_data())
        output[bool_mask] = eq_data
        eq_img = nb.Nifti1Image(output, uneq_img.affine, uneq_img.header)
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_equalized', newpath=runtime.cwd)
        eq_img.to_filename(self._results['out_file'])
        return runtime
