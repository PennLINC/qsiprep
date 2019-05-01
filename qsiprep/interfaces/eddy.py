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
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiObject, OutputMultiObject,
    SimpleInterface)
LOGGER = logging.getLogger('nipype.interface')


class GatherEddyInputsInputSpec(BaseInterfaceInputSpec):
    rpe_b0 = File(exists=True)
    dwi_files = InputMultiObject(File(exists=True))
    bval_files = InputMultiObject(File(exists=True))
    bvec_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    b0_indices = traits.List()
    b0_images = InputMultiObject(File(exists=True))


class GatherEddyInputsOutputSpec(TraitedSpec):
    pass


class GatherEddyInputs(SimpleInterface):
    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.rpe_b0):
            pass

        # Gather inputs for TOPUP

        # Gather inputs for eddy

        return runtime


def read_nifti_sidecar(path):
    pth, fname, ext = split_filename(path)
    json_file = op.join(pth,fname) + ".json"
    with open(path, "r") as f:
        metadata = json.load(f)
    pe_dir = metadata['PhaseEncodingDirection']
    slice_times = metadata.get("SliceTiming")
    trt = metadata.get("TotalReadoutTime")
    if trt is None:
        pass

    return {"PhaseEncodingDirection": pe_dir,
            "SliceTiming": slice_times,
            "TotalReadoutTime": trt}


def specs_from_dwi_files(dwi_file_list):
    """Create a list of lines for a datain spec from a list of dwi files.

    """
    unique_files = list(set(dwi_file_list))
    spec_lookup = {}
    lines = {
        "i": '1 0 0 %.6f',
        "j": '0 1 0 %.6f',
        "k": '0 0 1 %.6f',
        "i-": '-1 0 0 %.6f',
        "j-": '0 -1 0 %.6f',
        "k-": '0 0 -1 %.6f'
    }
    for unique_dwi in unique_files:
