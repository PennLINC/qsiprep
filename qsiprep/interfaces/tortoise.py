#!python
"""
Wrappers for the TORTOISE programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import os.path as op
import subprocess
import logging
import numpy as np
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, InputMultiObject,
                                    OutputMultiObject, SimpleInterface)
from nipype.interfaces import ants
from nipype.utils.filemanip import split_filename
import nibabel as nb
from .fmap import get_distortion_grouping
from .gradients import write_concatenated_fsl_gradients
import nilearn.image as nim

LOGGER = logging.getLogger('nipype.interface')


class TORTOISEInputSpec(BaseInterfaceInputSpec):
    pass


class _GatherDRBUDDIInputsInputSpec(TORTOISEInputSpec):
    dwi_files = InputMultiObject(File(exists=True))
    bval_files = InputMultiObject(File(exists=True))
    bvec_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    b0_threshold = traits.CInt(100, usedefault=True)
    epi_fmaps = InputMultiObject(File(exists=True),
                                 desc='files from fmaps/ for distortion correction')
    raw_image_sdc = traits.Bool(True, usedefault=True)
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)


class _GatherDRBUDDIInputsOutputSpec(TORTOISEInputSpec):
    blip_up_image = File(exists=True)
    blip_up_bmat = File(exists=True)
    blip_down_image = File(exists=True)
    blip_down_bmat = File(exists=True)
    blip_up_json = File(exists=True)
    report = traits.Str()


class GatherDRBUDDIInputs(SimpleInterface):
    input_spec = _GatherDRBUDDIInputsInputSpec
    output_spec = _GatherDRBUDDIInputsOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.fieldmap_type == "rpe_series":
            self._results["blip_up_image"], self.results["blip_up_bmat"], \
                self._results["blip_down_image"], self.results["blip_down_bmat"] = \
                    split_into_up_and_down_niis(
                        dwi_files=self.inputs.dwi_files,
                        bval_files=self.inputs.bval_files,
                        bvec_files=self.inputs.bvec_files,
                        original_images=self.inputs.original_images,
                        prefix=op.join(runtime.cwd, "drbuddi"),
                        make_bmat=True
                    )


        return runtime

class _DRBUDDIInputSpec(TORTOISEInputSpec):
    blip_up_image = File(
        exists=True,
        help="Full path to the input UP NIFTI file to be corrected.",
        argstr="-d %s",
        mandatory=True)
    blip_up_json = File(
        exists=True,
        help="Phase encoding information will be read from this",
        argstr="-up_json %s",
        mandatory=True)
    blip_down_image = File(
        exists=True,
        help="Full path to the input DOWN NIFTI file to be corrected.",
        argstr="-d %s",
        mandatory=True)
    structural_image = InputMultiObject(
        File(exists=True),
        help="Path(s) to anatomical image files. Can provide more than one. NO T1W's!!"
    )
    nthreads=traits.Int(1, usedefault=True)


class _DRBUDDIOutputSpec(TraitedSpec):
    sdc_warps = OutputMultiObject(File(exists=True))
    undistorted_reference = File(exists=True)


class DRBUDDI(CommandLine):
    input_spec = _DRBUDDIInputSpec
    output_spec = _DRBUDDIOutputSpec


def drbuddi_boilerplate(fieldmap_type):
    desc = []
    if fieldmap_type in ("rpe_series", "epi"):
        desc.append("Data was collected with reversed phase-encode blips, resulting "
                    "in pairs of images with distortions going in opposite directions.")
        if fieldmap_type == "epi":
            desc.append("Here, b=0 reference images with reversed "
                        "phase encoding directions were used to estimate ")
        else:
            desc.append("Here, multiple DWI series were acquired with opposite phase encoding "
                        "directions. A b=0 image **and** the Fractional Anisotropy "
                        "images from both phase encoding diesctions were used together in "
                        "a multi-modal registration to estimate ")
        desc.append("the susceptibility-induced off-resonance "
                    "field [@drbuddi]. ")
    return " ".join(desc)


def split_into_up_and_down_niis(dwi_files, bval_files, bvec_files, original_images,
                                prefix, make_bmat=True):
    """Takes the concatenated output from pre_hmc_wf and split it into "up" and "down"
    decompressed nifti files with float32 datatypes."""
    group_names, group_assignments = get_distortion_grouping(original_images)

    if not len(set(group_names)) == 2:
        raise Exception("DRBUDDI requires exactly one blip up and one blip down")

    up_images = []
    up_bvals = []
    up_bvecs = []
    up_prefix = prefix + "_up_dwi"
    up_dwi_file = up_prefix + ".nii"
    up_bval_file = up_prefix + ".bval"
    up_bvec_file = up_prefix + ".bvec"
    up_bmat_file = up_prefix + ".bmtxt"
    down_images = []
    down_bvals = []
    down_bvecs = []
    down_prefix = prefix + "_down_dwi"
    down_dwi_file = down_prefix + ".nii"
    down_bval_file = down_prefix + ".bval"
    down_bvec_file = down_prefix + ".bvec"
    down_bmat_file = down_prefix + ".bmtxt"

    # We know up is first because we concatenated them ourselves
    up_group_name = group_assignments[0]
    for dwi_file, bval_file, bvec_file, distortion_group in \
            zip(dwi_files, bval_files, bvec_files, group_assignments):

        if distortion_group == up_group_name:
            up_images.append(dwi_file)
            up_bvals.append(bval_file)
            up_bvecs.append(bvec_file)
        else:
            down_images.append(dwi_file)
            down_bvals.append(bval_file)
            down_bvecs.append(bvec_file)

    # Write the 4d up image
    up_4d = nim.concat_imgs(up_images, dtype=np.float32, auto_resample=False)
    up_4d.to_filename(up_dwi_file)
    write_concatenated_fsl_gradients(up_bvals, up_bvecs, up_prefix)

    # Write the 4d down image
    down_4d = nim.concat_imgs(down_images, dtype=np.float32, auto_resample=False)
    down_4d.to_filename(down_dwi_file)
    write_concatenated_fsl_gradients(down_bvals, down_bvecs, down_prefix)

    # Send back FSL-style gradients
    if not make_bmat:
        return up_dwi_file, up_bval_file, up_bvec_file, \
            down_dwi_file, down_bval_file, down_bvec_file

    # Convert to bmatrix text file
    make_bmat(up_bval_file, up_bvec_file)
    make_bmat(down_bval_file, down_bvec_file)

    return up_dwi_file, up_bmat_file, down_dwi_file, down_bmat_file


def make_bmat(bvals, bvecs):
    pout = subprocess.run(
        ["FSLBVecsToTORTOISEBmatrix", op.abspath(bvals), op.abspath(bvecs)])
    return bvals.replace("bval", "bmtxt")

