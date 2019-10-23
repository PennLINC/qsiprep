#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Prepare files for TOPUP and eddy
~~~~~~~~~~~~~~~~~~

"""
import os
import json
from pkg_resources import resource_filename as pkgr_fn
import os.path as op
from collections import defaultdict
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nb
from nipype.interfaces import fsl
from .images import to_lps, reorient_to

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
    topup_imain = File(exists=True)
    topup_config = traits.Str()
    pre_topup_image = File(exists=True)
    eddy_acqp = File(exists=True)
    eddy_indices = File(exists=True)
    forward_transforms = traits.List()
    forward_warps = traits.List()


class GatherEddyInputs(SimpleInterface):
    """Manually prepare inputs for TOPUP and eddy.

    **Inputs**
        rpe_b0: str
            path to a file (3D or 4D) containing b=0 images with the reverse PE direction
        dwi_files: list
            list of paths to 3d DWI volumes
        bval_files, bvec_files: list
            lists of paths to bval, bvec files
    """
    input_spec = GatherEddyInputsInputSpec
    output_spec = GatherEddyInputsOutputSpec

    def _run_interface(self, runtime):
        rpe_b0 = self.inputs.rpe_b0
        original_files = self.inputs.original_files
        b0_indices = self.inputs.b0_indices
        dwi_files = self.inputs.dwi_files

        # Should we get metadata from the rpe_b0?
        rpe_files = []
        if isdefined(rpe_b0):
            rpe_files = [rpe_b0]

        # Gather inputs for TOPUP
        topup_prefix = op.join(runtime.cwd, "topup_")
        # get original files for their bids metadata
        b0_source_images = [original_files[idx] for idx in b0_indices] + rpe_files
        # this will be all the original b=0 files and the b=0 files from the rpe series
        b0_images = self.inputs.b0_images + rpe_files
        topup_datain_file, topup_imain_file = topup_inputs_from_dwi_files(
            b0_source_images, b0_images, topup_prefix, runtime.cwd)
        self._results['topup_datain'] = topup_datain_file
        self._results['topup_imain'] = topup_imain_file

        # If there are an odd number of slices, use b02b0_1.cnf
        example_b0 = nb.load(b0_images[0])
        self._results['topup_config'] = 'b02b0.cnf'
        if 1 in (example_b0.shape[0] % 2, example_b0.shape[1] % 2, example_b0.shape[2] % 2):
            LOGGER.warning(
                "Using slower b02b0_1.cnf because an axis has an odd number of slices")
            self._results['topup_config'] = pkgr_fn('qsiprep.data', 'b02b0_1.cnf')

        # For the apply topup report:
        eddy_prefix = op.join(runtime.cwd, "eddy_")
        self._results['pre_topup_image'] = b0_images[0]

        # Gather inputs for eddy
        acqp_file, index_file = eddy_inputs_from_dwi_files(original_files, dwi_files, eddy_prefix)
        self._results['eddy_acqp'] = acqp_file
        self._results['eddy_indices'] = index_file

        # these have already had HMC, SDC applied
        self._results['forward_transforms'] = []
        self._results['forward_warps'] = []
        return runtime


def read_nifti_sidecar(path):
    pth, fname, _ = split_filename(path)
    json_file = op.join(pth, fname) + ".json"
    with open(json_file, "r") as f:
        metadata = json.load(f)
    pe_dir = metadata['PhaseEncodingDirection']
    slice_times = metadata.get("SliceTiming")
    trt = metadata.get("TotalReadoutTime")
    if trt is None:
        pass

    return {"PhaseEncodingDirection": pe_dir,
            "SliceTiming": slice_times,
            "TotalReadoutTime": trt}


acqp_lines = {
    "i": '1 0 0 %.6f',
    "j": '0 1 0 %.6f',
    "k": '0 0 1 %.6f',
    "i-": '-1 0 0 %.6f',
    "j-": '0 -1 0 %.6f',
    "k-": '0 0 -1 %.6f'}


def eddy_inputs_from_dwi_files(origin_file_list, dwi_files, eddy_prefix):
    unique_files = list(set(origin_file_list))
    line_lookup = {}
    acqp_data = []
    for line_num, unique_dwi in enumerate(unique_files):
        spec = read_nifti_sidecar(unique_dwi)
        spec_line = acqp_lines[spec['PhaseEncodingDirection']]
        acqp_line = spec_line % spec['TotalReadoutTime']
        line_lookup[unique_dwi] = line_num + 1
        acqp_data.append(acqp_line)

    # Create the acqp file
    acqp_file = eddy_prefix + "acqp.txt"
    with open(acqp_file, "w") as f:
        f.write("\n".join(acqp_data))

    # Create the index file
    index_file = eddy_prefix + "index.txt"
    index_numbers = [line_lookup[dwi_file] for dwi_file in origin_file_list]
    with open(index_file, "w") as f:
        f.write(" ".join(map(str, index_numbers)))

    return acqp_file, index_file


def topup_inputs_from_dwi_files(dwi_file_list, b0_file_list, topup_prefix, cwd, max_per_spec=3):
    """Create a datain spec and a slspec from a list of dwi files."""
    unique_files = list(set(dwi_file_list))
    spec_lookup = {}
    slicetime_lookup = {}
    for unique_dwi in unique_files:
        spec = read_nifti_sidecar(unique_dwi)
        spec_line = acqp_lines[spec['PhaseEncodingDirection']]
        spec_lookup[unique_dwi] = spec_line % spec['TotalReadoutTime']
        slicetime_lookup[unique_dwi] = spec['SliceTiming']

    # Write the datain.txt file
    datain_lines = []
    imain_images = []
    image_data = []
    spec_counts = defaultdict(int)

    def atleast4d(data):
        if data.ndim == 4:
            return data
        if data.ndim == 3:
            return data[:, :, :, np.newaxis]
        raise Exception("Less than 3 dimensions in b0 image")

    for dwi_file, b0_file in zip(dwi_file_list, b0_file_list):
        img = nb.load(b0_file)
        line = spec_lookup[dwi_file]
        num_trs = 1 if len(img.shape) < 4 else img.shape[3]
        available_slots = max_per_spec - spec_counts[line]

        if available_slots <= 0:
            continue

        if available_slots >= num_trs:
            datain_lines.extend([line] * num_trs)
            spec_counts[line] += num_trs
            imain_images.append(b0_file)
            image_data.append(atleast4d(img.get_fdata()))
        else:
            # Too many images for this spec
            num_to_add = available_slots
            truncated_image = fname_presuffix(b0_file, newpath=cwd, suffix="truncated")
            orig_img = nb.load(b0_file)
            LOGGER.warning("Truncating %s to %d volumes", b0_file, num_to_add)
            nb.Nifti1Image(atleast4d(orig_img.get_fdata())[:, :, :, :num_to_add],
                           orig_img.affine, orig_img.header).to_filename(truncated_image)
            imain_images.append(truncated_image)
            datain_lines.extend([line] * num_to_add)
            spec_counts[line] += num_to_add

    # Make a 4d series, all conformed to LAS+
    images = [to_lps(nb.load(img), new_axcodes=('L', 'A', 'S')) for img in imain_images]
    image_data = [img.get_fdata()[..., np.newaxis] if len(img.shape) == 3 else img.get_fdata()
                  for img in images]
    imain_output = topup_prefix + "imain.nii.gz"
    imain_img = nb.Nifti1Image(np.concatenate(image_data, 3), images[0].affine, images[0].header)
    assert imain_img.shape[3] == len(datain_lines)
    imain_img.to_filename(imain_output)

    # Write the datain text file
    datain_file = topup_prefix + "datain.txt"
    with open(datain_file, "w") as f:
        f.write("\n".join(datain_lines))

    # Check the slicetiming files
    return datain_file, imain_output


class ExtendedEddyOutputSpec(fsl.epi.EddyOutputSpec):
    shell_PE_translation_parameters = File(
        exists=True,
        desc=('the translation along the PE-direction between the different shells'))
    outlier_map = File(
        exists=True, desc='All numbers are either 0, meaning that scan-slice '
        'is not an outliers, or 1 meaning that it is.')
    outlier_n_stdev_map = File(
        exists=True, desc='how many standard deviations off the mean difference '
        'between observation and prediction is.')
    outlier_n_sqr_stdev_map = File(
        exists=True, desc='how many standard deviations off the square root of the '
        'mean squared difference between observation and prediction is.')
    outlier_free_data = File(
        exists=True, desc=' the original data given by --imain not corrected for '
        'susceptibility or EC-induced distortions or subject movement, but with '
        'outlier slices replaced by the Gaussian Process predictions.')


class ExtendedEddy(fsl.Eddy):
    output_spec = ExtendedEddyOutputSpec

    _num_threads = 1

    def __init__(self, **inputs):
        super(ExtendedEddy, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not isdefined(self.inputs.num_threads):
            self.inputs.num_threads = self._num_threads
        else:
            self._num_threads_update()
        self.inputs.on_trait_change(self._use_cuda, 'use_cuda')
        if isdefined(self.inputs.use_cuda):
            self._use_cuda()

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected'] = os.path.abspath(
            '%s.nii.gz' % self.inputs.out_base)
        outputs['out_parameter'] = os.path.abspath(
            '%s.eddy_parameters' % self.inputs.out_base)

        # File generation might depend on the version of EDDY
        out_rotated_bvecs = os.path.abspath(
            '%s.eddy_rotated_bvecs' % self.inputs.out_base)
        out_movement_rms = os.path.abspath(
            '%s.eddy_movement_rms' % self.inputs.out_base)
        out_restricted_movement_rms = os.path.abspath(
            '%s.eddy_restricted_movement_rms' % self.inputs.out_base)
        out_shell_alignment_parameters = os.path.abspath(
            '%s.eddy_post_eddy_shell_alignment_parameters' % self.inputs.out_base)
        shell_PE_translation_parameters = op.abspath(
            '%s.eddy_post_eddy_shell_PE_translation_parameters' % self.inputs.out_base)
        out_outlier_report = os.path.abspath(
            '%s.eddy_outlier_report' % self.inputs.out_base)
        outlier_map = op.abspath(
            '%s.eddy_outlier_map' % self.inputs.out_base)
        outlier_n_stdev_map = op.abspath(
            '%s.eddy_outlier_n_stdev_map' % self.inputs.out_base)
        outlier_n_sqr_stdev_map = op.abspath(
            '%s.eddy_outlier_n_sqr_stdev_map' % self.inputs.out_base)

        if isdefined(self.inputs.cnr_maps) and self.inputs.cnr_maps:
            out_cnr_maps = os.path.abspath(
                '%s.eddy_cnr_maps.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_cnr_maps):
                outputs['out_cnr_maps'] = out_cnr_maps
        if isdefined(self.inputs.residuals) and self.inputs.residuals:
            out_residuals = os.path.abspath(
                '%s.eddy_residuals.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_residuals):
                outputs['out_residuals'] = out_residuals

        if os.path.exists(out_rotated_bvecs):
            outputs['out_rotated_bvecs'] = out_rotated_bvecs
        if os.path.exists(out_movement_rms):
            outputs['out_movement_rms'] = out_movement_rms
        if os.path.exists(out_restricted_movement_rms):
            outputs['out_restricted_movement_rms'] = \
                out_restricted_movement_rms
        if os.path.exists(out_shell_alignment_parameters):
            outputs['out_shell_alignment_parameters'] = \
                out_shell_alignment_parameters
        if os.path.exists(out_outlier_report):
            outputs['out_outlier_report'] = out_outlier_report

        if op.exists(shell_PE_translation_parameters):
            outputs['shell_PE_translation_parameters'] = shell_PE_translation_parameters
        if op.exists(outlier_map):
            outputs['outlier_map'] = outlier_map
        if op.exists(outlier_n_stdev_map):
            outputs['outlier_n_stdev_map'] = outlier_n_stdev_map
        if op.exists(outlier_n_sqr_stdev_map):
            outputs['outlier_n_sqr_stdev_map'] = outlier_n_sqr_stdev_map

        return outputs


    def _format_arg(self, name, spec, value):
        if name == 'field':
            pth, fname, ext = split_filename(value)
            return spec.argstr % op.join(pth, fname)
        return super(ExtendedEddy, self)._format_arg(name, spec, value)

class Eddy2SPMMotionInputSpec(BaseInterfaceInputSpec):
    eddy_motion = File(exists=True)


class Eddy2SPMMotionOututSpec(TraitedSpec):
    spm_motion_file = File(exists=True)


class Eddy2SPMMotion(SimpleInterface):
    input_spec = Eddy2SPMMotionInputSpec
    output_spec = Eddy2SPMMotionOututSpec

    def _run_interface(self, runtime):
        # Load the eddy motion params File
        eddy_motion = np.loadtxt(self.inputs.eddy_motion)
        spm_motion = eddy_motion[:, :6]
        spm_motion_file = fname_presuffix(self.inputs.eddy_motion, suffix="spm_rp.txt",
                                          use_ext=False, newpath=runtime.cwd)
        np.savetxt(spm_motion_file, spm_motion)
        self._results['spm_motion_file'] = spm_motion_file

        return runtime
