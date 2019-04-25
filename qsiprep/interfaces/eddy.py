#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath, OutputMultiPath,
    OutputMultiObject, SimpleInterface)
LOGGER = logging.getLogger('nipype.interface')


class GatherEddyInputsInputSpec(BaseInterfaceInputSpec):
    pass


class GatherEddyInputsOutputSpec(TraitedSpec):
    pass


class GatherEddyInputs(SimpleInterface):
    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):
        return runtime
