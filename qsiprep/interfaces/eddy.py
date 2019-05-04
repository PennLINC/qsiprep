#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
import os
import json
import os.path as op
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiObject, OutputMultiObject,
    SimpleInterface, isdefined)
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
    topup_datain = File(exists=True)
    topup_b0s = File(exists=True)
    apply_topup_image = File(exists=True)
    apply_topup_config = File(exists=True)
    eddy_config = File(exists=True)
    eddy_indices = File(exists=True)


class GatherEddyInputs(SimpleInterface):
    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):
        dwi_files_list = self.inputs.dwi_files
        rpe_b0 = self.inputs.rpe_b0
        original_files = self.inputs.original_files
        b0_indices = self.inputs.b0_indices

        # Should we get metadata from the rpe_b0?
        rpe_files = []
        if isdefined(rpe_b0):
            rpe_files = [rpe_b0]

        # Gather inputs for TOPUP
        topup_datain = op.join(runtime.cwd, "topup_datain.txt")
        b0_source_images = [original_files[idx] for idx in b0_indices] + rpe_files
        topup_file, slspec_file = specs_from_dwi_files(b0_source_images, topup_datain)
        self.outputs['topup_datain'] = topup_file
        self.outputs['topup_b0s'] =


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


def specs_from_dwi_files(dwi_file_list, datain_file):
    """Create a datain spec and a slspec from a list of dwi files."""
    unique_files = list(set(dwi_file_list))
    lines = {
        "i": '1 0 0 %.6f',
        "j": '0 1 0 %.6f',
        "k": '0 0 1 %.6f',
        "i-": '-1 0 0 %.6f',
        "j-": '0 -1 0 %.6f',
        "k-": '0 0 -1 %.6f'}
    spec_lookup = {}
    slicetime_lookup = {}
    for unique_dwi in unique_files:
        spec = read_nifti_sidecar(unique_dwi)
        spec_line = lines[spec['PhaseEncodingDirection']]
        spec_lookup[unique_dwi] = spec_line % spec['TotalReadoutTime']
        slicetime_lookup[unique_dwi] = spec['SliceTiming']

    # Write the datain.txt file
    datain_lines = [spec_lookup[dwi_file] for dwi_file in dwi_file_list]
    with open(datain_file, "w") as f:
        f.write("\n".join(datain_lines))

    # Check the slicetiming files
    return datain_file, None

def prepare_topup_file(b0_file_list, topup_file, image_to_unwarp):
    b0_images = [nb.load(b0_file) for b0_file in b0_file_list]

    # Check if the last image is 4d:
    last_image = b0_images.pop()
    if last_image.ndim == 4:
        last_image = nb.Nifti1Image(last_image.get_fdata()[..., 0], last_image.affine,
                                    last_image.header)
    b0_images.append(last_image)


    return
