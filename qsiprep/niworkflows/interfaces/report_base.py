# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" class mixin and utilities for enabling reports for nipype interfaces """
from __future__ import absolute_import, division, print_function, unicode_literals

from nilearn.masking import apply_mask, unmask
from nilearn.image import threshold_img, load_img

from nipype.interfaces.base import File, traits
from nipype.interfaces.mixins import reporting
from .. import NIWORKFLOWS_LOG
from ..viz.utils import cuts_from_bbox, compose_view


class SVGReportCapableInputSpec(reporting.ReportCapableInputSpec):
    out_report = File(
        'report.svg', usedefault=True, desc='filename for the visual report')
    compress_report = traits.Enum('auto', True, False, usedefault=True,
                                  desc="Compress the reportlet using SVGO or"
                                       "WEBP. 'auto' - compress if relevant "
                                       "software is installed, True = force,"
                                       "False - don't attempt to compress")


class RegistrationRC(reporting.ReportCapableInterface):
    """ An abstract mixin to registration nipype interfaces """
    _fixed_image = None
    _moving_image = None
    _fixed_image_mask = None
    _fixed_image_label = "fixed"
    _moving_image_label = "moving"
    _contour = None

    def _generate_report(self):
        """ Generates the visual report """
        from ...niworkflows.viz.utils import plot_registration
        NIWORKFLOWS_LOG.info('Generating visual report')

        fixed_image_nii = load_img(self._fixed_image)
        moving_image_nii = load_img(self._moving_image)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._fixed_image_mask:
            fixed_image_nii = unmask(apply_mask(fixed_image_nii,
                                                self._fixed_image_mask),
                                     self._fixed_image_mask)
            # since the moving image is already in the fixed image space we
            # should apply the same mask
            moving_image_nii = unmask(apply_mask(moving_image_nii,
                                                 self._fixed_image_mask),
                                      self._fixed_image_mask)
            mask_nii = load_img(self._fixed_image_mask)
        else:
            mask_nii = threshold_img(fixed_image_nii, 1e-3)

        n_cuts = 7
        if not self._fixed_image_mask and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(fixed_image_nii, 'fixed-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label=self._fixed_image_label,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            plot_registration(moving_image_nii, 'moving-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label=self._moving_image_label,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            out_file=self._out_report
        )


class SegmentationRC(reporting.ReportCapableInterface):

    """ An abstract mixin to segmentation nipype interfaces """

    def _generate_report(self):
        from ...niworkflows.viz.utils import plot_segs
        compose_view(
            plot_segs(
                image_nii=self._anat_file,
                seg_niis=self._seg_files,
                bbox_nii=self._mask_file,
                out_file=self.inputs.out_report,
                masked=self._masked,
                compress=self.inputs.compress_report
            ),
            fg_svgs=None,
            out_file=self._out_report
        )


class SurfaceSegmentationRC(reporting.ReportCapableInterface):

    """ An abstract mixin to registration nipype interfaces """
    _anat_file = None
    _mask_file = None
    _contour = None

    def _generate_report(self):
        """ Generates the visual report """
        from ...niworkflows.viz.utils import plot_registration
        NIWORKFLOWS_LOG.info('Generating visual report')

        anat = load_img(self._anat_file)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._mask_file:
            anat = unmask(apply_mask(anat, self._mask_file), self._mask_file)
            mask_nii = load_img(self._mask_file)
        else:
            mask_nii = threshold_img(anat, 1e-3)

        n_cuts = 7
        if not self._mask_file and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(anat, 'fixed-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              contour=contour_nii,
                              compress=self.inputs.compress_report),
            [],
            out_file=self._out_report
        )


class ReportingInterface(reporting.ReportCapableInterface):
    """Interface that always generates a report

    A subclass must define an ``input_spec`` and override ``_generate_report``.
    """
    output_spec = reporting.ReportCapableOutputSpec

    def __init__(self, generate_report=True, **kwargs):
        super(ReportingInterface, self).__init__(generate_report=generate_report, **kwargs)

    def _run_interface(self, runtime):
        return runtime
