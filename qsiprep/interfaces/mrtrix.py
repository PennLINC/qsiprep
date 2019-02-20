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
    OutputMultiObject, isdefined
)
from nipype.interfaces import ants
from nipype.interfaces.ants.registration import RegistrationInputSpec
from .gradients import concatenate_bvecs, concatenate_bvals, GradientRotation
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from nipype.interfaces.mrtrix3 import EstimateFOD, Generate5tt, ComputeTDI, ResponseSD, MRConvert
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec

LOGGER = logging.getLogger('nipype.interface')


class MRTrixGradientTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)


class MRTrixGradientTableOutputSpec(TraitedSpec):
    gradient_file = File(exists=True)


class MRTrixGradientTable(SimpleInterface):
    input_spec = MRTrixGradientTableInputSpec
    output_spec = MRTrixGradientTableOutputSpec

    def _run_interface(self, runtime):
        gtab_fname = fname_presuffix(self.inputs.bval_file, suffix=".b", newpath=runtime.cwd,
                                     use_ext=False)
        vecs = np.loadtxt(self.inputs.bvec_file)
        vals = np.loadtxt(self.inputs.bval_file)
        gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
        np.savetxt(gtab_fname, gtab, fmt=["%.8f", "%.8f", "%.8f", "%d"])
        self._results['gradient_file'] = gtab_fname
        return runtime


class MRTrixIngressInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    suffix = traits.Str("_dwi", usedefault=True)


class MRTrixIngressOutputSpec(TraitedSpec):
    mif_file = File()


class MRTrixIngress(SimpleInterface):
    input_spec = MRTrixIngressInputSpec
    output_spec = MRTrixIngressOutputSpec

    def _run_interface(self, runtime):
        output_mif = fname_presuffix(self.inputs.dwi_file, suffix=self.inputs.suffix + ".mif",
                                     newpath=runtime.cwd, use_ext=False)
        if isdefined(self.inputs.b_file):
            convert = MRConvert(in_file=self.inputs.dwi_file,
                                grad_file=self.inputs.b_file,
                                out_file=output_mif)
        elif isdefined(self.inputs.bval_file) and isdefined(self.inputs.bvec_file):
            convert = MRConvert(in_file=self.inputs.dwi_file,
                                in_bval=self.inputs.bval_file,
                                in_bvec=self.inputs.bvec_file,
                                out_file=output_mif)
        else:
            raise Exception("No valid mrtrix gradient files or fsl bval/bvec files specified")
        convert_run = convert.run()
        self._results['mif_file'] = convert_run.outputs.out_file

        return runtime
