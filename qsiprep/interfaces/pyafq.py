#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interface for running a PyAFQ workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
from pkg_resources import resource_filename as pkgr
import nibabel as nb
from pyafq import
import numpy as np
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, isdefined
)


LOGGER = logging.getLogger('nipype.interface')
TAU_DEFAULT = 1. / (4 * np.pi**2)


class PyAFQInputSpec(BaseInterfaceInputSpec):
    b0_threshold = traits.Int(50, usedefault=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    big_delta = traits.Either(None, traits.Float(), usedefault=True)
    little_delta = traits.Either(None, traits.Float(), usedefault=True)
    b0_threshold = traits.CFloat(50, usedefault=True)


class PyAFQOutputSpec(TraitedSpec):
    out_directory = File()


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):

        # shim the expected inputs
        # shim_dir = op.join(runtime.cwd, "study/subject")
        # os.makedirs(shim_dir)
        # bval_file = fname_presuffix(self.inputs.bval_file,
        #                             newpath=shim_dir)
        # bvec_file = fname_presuffix(self.inputs.bvec_file,
        #                             newpath=shim_dir)
        # dwi_file = fname_presuffix(self.inputs.dwi_file,
        #                             newpath=shim_dir)
        # mask_file = fname_presuffix(self.inputs.mask_file,
        #                             newpath=shim_dir)
        # os.symlink(self.inputs.bval_file, bval_file)
        # os.symlink(self.inputs.bvec_file, bvec_file)
        # os.symlink(self.inputs.dwi_file, dwi_file)
        # os.symlink(self.inputs.mask_file, mask_file)


        self._results['directions_image'] = shim_dir + "/AMICO/NODDI/FIT_dir.nii.gz"
        self._results['icvf_image'] = shim_dir + "/AMICO/NODDI/FIT_ICVF.nii.gz"
        self._results['od_image'] = shim_dir + "/AMICO/NODDI/FIT_OD.nii.gz"
        self._results['isovf_image'] = shim_dir + "/AMICO/NODDI/FIT_ISOVF.nii.gz"
        self._results['config_file'] = shim_dir + "/AMICO/NODDI/config.pickle"

        return runtime
