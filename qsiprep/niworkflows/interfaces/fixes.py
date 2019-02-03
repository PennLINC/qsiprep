# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

from nipype.interfaces.ants.resampling import ApplyTransforms
from nipype.interfaces.ants.registration import Registration

from .. import __version__
from .utils import _copyxform


class FixHeaderApplyTransforms(ApplyTransforms):
    """
    A replacement for nipype.interfaces.ants.resampling.ApplyTransforms that
    fixes the resampled image header to match the xform of the reference
    image
    """

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderApplyTransforms, self)._run_interface(
            runtime, correct_return_codes)

        _copyxform(self.inputs.reference_image,
                   os.path.abspath(self._gen_filename('output_image')),
                   message='%s (niworkflows v%s)' % (
                       self.__class__.__name__, __version__))
        return runtime


class FixHeaderRegistration(Registration):
    """
    A replacement for nipype.interfaces.ants.registration.Registration that
    fixes the resampled image header to match the xform of the reference
    image
    """

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderRegistration, self)._run_interface(
            runtime, correct_return_codes)

        # Forward transform
        out_file = self._get_outputfilenames(inverse=False)
        if out_file is not None and out_file:
            _copyxform(
                self.inputs.fixed_image[0], os.path.abspath(out_file),
                message='%s (niworkflows v%s)' % (
                    self.__class__.__name__, __version__))

        # Inverse transform
        out_file = self._get_outputfilenames(inverse=True)
        if out_file is not None and out_file:
            _copyxform(
                self.inputs.moving_image[0], os.path.abspath(out_file),
                message='%s (niworkflows v%s)' % (
                    self.__class__.__name__, __version__))

        return runtime
