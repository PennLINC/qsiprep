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
import os

from tempfile import TemporaryDirectory
from time import time

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject
)
from nipype.interfaces import ants
from nipype.interfaces.ants.registration import RegistrationInputSpec
from .gradients import concatenate_bvecs, concatenate_bvals, GradientRotation
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis

LOGGER = logging.getLogger('nipype.interface')
