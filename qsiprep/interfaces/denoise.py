#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for image denoising
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import numpy as np
import nibabel as nb
import pandas as pd
from nilearn.image import load_img, threshold_img, iter_img
from nipype import logging
from nipype.interfaces.base import traits, isdefined
from nipype.interfaces.mixins import reporting
from ..niworkflows.viz.utils import cuts_from_bbox, compose_view, plot_denoise

LOGGER = logging.getLogger('nipype.interface')


class SeriesPreprocReportInputSpec(reporting.ReportCapableInputSpec):
    nmse_text = traits.File(
        name_source='in_file',
        keep_extension=False,
        name_template='%s_nmse.txt')


class SeriesPreprocReportOutputSpec(reporting.ReportCapableOutputSpec):
    nmse_text = traits.File(desc='nmse between input and output volumes')


class SeriesPreprocReport(reporting.ReportCapableInterface):
    input_spec = SeriesPreprocReportInputSpec
    output_spce = SeriesPreprocReportOutputSpec
    _n_cuts = 7

    def __init__(self, **kwargs):
        """Instantiate SeriesPreprocReportlet."""
        self._n_cuts = kwargs.pop('n_cuts', self._n_cuts)
        super(SeriesPreprocReport, self).__init__(generate_report=True, **kwargs)

    def _calculate_nmse(self, original_nii, corrected_nii):
        """Calculate NMSE from the applied preprocessing operation."""
        outputs = self._list_outputs()
        output_file = outputs.get('nmse_text')
        pres = []
        posts = []
        differences = []
        for orig_img, corrected_img in zip(iter_img(original_nii), iter_img(corrected_nii)):
            orig_data = orig_img.get_fdata()
            corrected_data = corrected_img.get_fdata()
            baseline = orig_data.mean()
            pres.append(baseline)
            posts.append(corrected_data.mean())
            scaled_diff = np.abs(corrected_data - orig_data).mean() / baseline
            differences.append(scaled_diff)
        title = str(self.__class__)[:-2].split('.')[-1]
        pd.DataFrame({title+"_pre": pres,
                      title+"_post": posts,
                      title+"_change": differences}).to_csv(output_file, index=False)

    def _generate_report(self):
        """Generate a reportlet."""
        LOGGER.info('Generating denoising visual report')

        input_dwi, denoised_nii, field_nii = self._get_plotting_images()

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

        # What image should be contoured?
        if field_nii is None:
            lowb_field_nii = nb.Nifti1Image(denoised_lowb_nii.get_fdata()
                                            - orig_lowb_nii.get_fdata(),
                                            affine=denoised_lowb_nii.affine)
            highb_field_nii = nb.Nifti1Image(denoised_highb_nii.get_fdata()
                                             - orig_highb_nii.get_fdata(),
                                             affine=denoised_highb_nii.affine)
        else:
            lowb_field_nii = highb_field_nii = field_nii

        # Call composer
        compose_view(
            plot_denoise(orig_lowb_nii, orig_highb_nii, 'moving-image',
                         estimate_brightness=True,
                         cuts=cuts,
                         label='Raw Image',
                         lowb_contour=lowb_field_nii,
                         highb_contour=highb_field_nii,
                         compress=False),
            plot_denoise(denoised_lowb_nii, denoised_highb_nii, 'fixed-image',
                         estimate_brightness=True,
                         cuts=cuts,
                         label="Denoised",
                         lowb_contour=lowb_field_nii,
                         highb_contour=highb_field_nii,
                         compress=False),
            out_file=self._out_report
        )

        self._calculate_nmse(input_dwi, denoised_nii)

    def _get_plotting_images(self):
        """Implemented in subclasses to return the original image, the denoised image,
        and optionally an image created during the denoieing step."""
        raise NotImplementedError()
