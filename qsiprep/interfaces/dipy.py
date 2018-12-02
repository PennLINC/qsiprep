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

from nilearn.masking import compute_epi_mask
from nilearn.image import concat_imgs

from dipy.core.histeq import histeq

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, InputMultiPath, SimpleInterface
)

LOGGER = logging.getLogger('nipype.interface')


class HistEQInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to equalize')
    mask_file = File(exists=True, mandatory=False, desc='Mask image')


class HistEQOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='equalized file')


class HistEQ(SimpleInterface):
    input_spec = HistEQInputSpec
    output_spec = HistEQOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file

        uneq_img = nb.load(in_file)
        uneq_data = uneq_img.get_data()

        if isdefined(self.inputs.mask_file):
            mask = nb.load(self.inputs.mask_file)
            mask_data = mask.get_data() > 0
            uneq_data[np.logical_not(mask_data)] = 0.0

        eq_data = histeq(uneq_data)
        eq_img = nb.Nifti1Image(eq_data.astype('f8'), uneq_img.affine, uneq_img.header)
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_equalized', newpath=runtime.cwd)
        eq_img.to_filename(self._results['out_mask'])
        return runtime


class MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input list of files to merge')
    dtype = traits.Enum('f4', 'f8', 'u1', 'u2', 'u4', 'i2', 'i4',
                        usedefault=True, desc='numpy dtype of output image')
    header_source = File(exists=True, desc='a Nifti file from which the header should be copied')
    compress = traits.Bool(True, usedefault=True, desc='Use gzip compression on .nii output')


class MergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output merged file')


class Merge(SimpleInterface):
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def _run_interface(self, runtime):
        ext = '.nii.gz' if self.inputs.compress else '.nii'
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_files[0], suffix='_merged' + ext, newpath=runtime.cwd, use_ext=False)
        new_nii = concat_imgs(self.inputs.in_files, dtype=self.inputs.dtype)

        if isdefined(self.inputs.header_source):
            src_hdr = nb.load(self.inputs.header_source).header
            new_nii.header.set_xyzt_units(t=src_hdr.get_xyzt_units()[-1])
            new_nii.header.set_zooms(list(new_nii.header.get_zooms()[:3]) +
                                     [src_hdr.get_zooms()[3]])

        new_nii.to_filename(self._results['out_file'])

        return runtime


def _enhance_t2_contrast(in_file, newpath=None, offset=0.5):
    """
    Performs a logarithmic transformation of intensity that
    effectively splits brain and background and makes the
    overall distribution more Gaussian.
    """
    out_file = fname_presuffix(in_file, suffix='_t1enh',
                               newpath=newpath)
    nii = nb.load(in_file)
    data = nii.get_data()
    maxd = data.max()
    newdata = np.log(offset + data / maxd)
    newdata -= newdata.min()
    newdata *= maxd / newdata.max()
    nii = nii.__class__(newdata, nii.affine, nii.header)
    nii.to_filename(out_file)
    return out_file
