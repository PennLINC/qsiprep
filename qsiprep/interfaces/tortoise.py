#!python
"""
Wrappers for the TORTOISE programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from curses import use_default_colors
import os
import os.path as op
import subprocess
import logging
import numpy as np
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, InputMultiObject,
                                    OutputMultiObject, SimpleInterface)
from nipype.interfaces import ants
from nipype.utils.filemanip import split_filename, fname_presuffix
import nilearn.image as nim
import nibabel as nb
from .fmap import get_distortion_grouping
from .images import to_lps
from .epi_fmap import get_best_b0_topup_inputs_from, safe_get_3d_image
from .gradients import write_concatenated_fsl_gradients
from .images import split_bvals_bvecs
import nilearn.image as nim
import pandas as pd

LOGGER = logging.getLogger('nipype.interface')

SLOPPY_DRBUDDI = \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{0.1\},cfs=\{100:8:4\},field_smoothing=\{9:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] " \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{0.25\},cfs=\{100:6:3\},field_smoothing=\{8:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] " \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{0.5\},cfs=\{100:4:2\},field_smoothing=\{7:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] " \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{1.25\},cfs=\{100:2:1\},field_smoothing=\{6:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] " \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{1.\},cfs=\{100:1:0\},field_smoothing=\{5:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] " \
    "--DRBUDDI_stage " \
        "\[learning_rate=\{1.\},cfs=\{20:1:0\},field_smoothing=\{4:0\}," \
        "metrics=\{MSJac:CC\},restrict_constrain=\{0:0\}\]"

class TORTOISEInputSpec(BaseInterfaceInputSpec):
    pass


class _GatherDRBUDDIInputsInputSpec(TORTOISEInputSpec):
    dwi_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    bval_files = traits.Either(
        InputMultiObject(File(exists=True)),
        File(exists=True))
    bvec_files = traits.Either(
        InputMultiObject(File(exists=True)),
        File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    b0_threshold = traits.CInt(100, usedefault=True)
    epi_fmaps = InputMultiObject(File(exists=True),
                                 desc='files from fmaps/ for distortion correction')
    raw_image_sdc = traits.Bool(True, usedefault=True)
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)
    dwi_series_pedir = traits.Enum(
        "i", "i-", "j", "j-", "k", "k-",
        mandatory=True)


class _GatherDRBUDDIInputsOutputSpec(TORTOISEInputSpec):
    blip_up_image = File(exists=True)
    blip_up_bmat = File(exists=True)
    blip_up_json = File(exists=True)
    blip_down_image = File(exists=True)
    blip_down_bmat = File(exists=True)
    blip_assignments = traits.List()
    report = traits.Str()


class GatherDRBUDDIInputs(SimpleInterface):
    input_spec = _GatherDRBUDDIInputsInputSpec
    output_spec = _GatherDRBUDDIInputsOutputSpec

    def _run_interface(self, runtime):

        # Write the metadata
        up_json = op.join(runtime.cwd, "blip_up.json")
        with open(up_json, "w") as up_jsonf:
            up_jsonf.write('{"PhaseEncodingDirection": "%s"}\n' % self.inputs.dwi_series_pedir)
        self._results["blip_up_json"] = up_json

        # Coerce the bvals and bvecs into lists of files
        if isinstance(self.inputs.bval_files, list) and len(self.inputs.bval_files) == 1:
            bval_files, bvec_files = split_bvals_bvecs(
                self.inputs.bval_files[0],
                self.inputs.bvec_files[0],
                deoblique=False,
                img_files=self.inputs.dwi_files,
                working_dir=runtime.cwd)
        else:
            bval_files, bvec_files = self.inputs.bval_files, self.inputs.bvec_files

        if self.inputs.fieldmap_type == "rpe_series":
            self._results["blip_assignments"], self._results["blip_up_image"], \
                self._results["blip_up_bmat"], self._results["blip_down_image"], \
                    self._results["blip_down_bmat"] = \
                        split_into_up_and_down_niis(
                            dwi_files=self.inputs.dwi_files,
                            bval_files=bval_files,
                            bvec_files=bvec_files,
                            original_images=self.inputs.original_files,
                            prefix=op.join(runtime.cwd, "drbuddi"),
                            make_bmat=True)

        elif self.inputs.fieldmap_type == 'epi':
            # Use the same function that was used to get images for TOPUP, but get the images
            # directly from the CSV
            _, _, _, b0_csv, _, _ = \
                get_best_b0_topup_inputs_from(
                    dwi_file=self.inputs.dwi_files,
                    bval_file=bval_files,
                    b0_threshold=self.inputs.b0_threshold,
                    cwd=runtime.cwd,
                    bids_origin_files=self.inputs.original_files,
                    epi_fmaps=self.inputs.epi_fmaps,
                    max_per_spec=True,
                    raw_image_sdc=self.inputs.raw_image_sdc)

            b0s_df = pd.read_csv(b0_csv)
            selected_images = b0s_df[b0s_df.selected_for_sdc].reset_index(drop=True)
            up_row = selected_images.loc[0]
            down_row = selected_images.loc[1]
            up_img = to_lps(
                safe_get_3d_image(up_row.bids_origin_file, up_row.original_volume))
            up_img.set_data_dtype(np.float32)
            down_img = to_lps(
                safe_get_3d_image(down_row.bids_origin_file, down_row.original_volume))
            down_img.set_data_dtype(np.float32)

            # Save the images
            blip_up_nii = op.join(runtime.cwd, "blip_up_b0.nii")
            blip_down_nii = op.join(runtime.cwd, "blip_down_b0.nii")
            up_img.to_filename(blip_up_nii)
            down_img.to_filename(blip_down_nii)
            self._results["blip_up_image"] = blip_up_nii
            self._results["blip_down_image"] = blip_down_nii
            self._results["blip_assignments"] = ["up"] * len(self.inputs.dwi_files)
            self._results["blip_up_bmat"] = write_dummy_bmtxt(blip_up_nii)
            self._results["blip_down_bmat"] = write_dummy_bmtxt(blip_down_nii)

        return runtime


def write_dummy_bmtxt(nii_file):
    new_fname = fname_presuffix(nii_file, suffix=".bmtxt", use_ext=False)
    img = nim.load_img(nii_file)
    nvols = 1 if img.ndim < 4 else img.ndim.shape[3]
    with open(new_fname, "w") as bmtxt_f:
        bmtxt_f.write("\n".join(["0 0 0 0 0 0"] * nvols) + "\n")
    return new_fname


class _DRBUDDIInputSpec(TORTOISEInputSpec):
    blip_up_image = File(
        exists=True,
        help="Full path to the input UP NIFTI file to be corrected.",
        argstr="-u %s",
        mandatory=True,
        copyfile=True)
    blip_up_bmat = File(
        exists=True,
        help="Full path to the input UP NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True)
    blip_up_json = File(
        exists=True,
        help="Phase encoding information will be read from this",
        argstr="--up_json %s",
        mandatory=True,
        copyfile=True)
    blip_down_image = File(
        exists=True,
        help="Full path to the input DOWN NIFTI file to be corrected.",
        argstr="-d %s",
        mandatory=True,
        copyfile=True)
    blip_down_bmat = File(
        exists=True,
        help="Full path to the input DOWN NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True)
    structural_image = InputMultiObject(
        File(exists=True, copyfile=False),
        help="Path(s) to anatomical image files. Can provide more than one. NO T1W's!!"
    )
    nthreads=traits.Int(1, usedefault=True, hash_files=False)
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)
    blip_assignments = traits.List()
    tensor_fit_bval_max = traits.Int(
        0,
        argstr="--DRBUDDI_DWI_bval_tensor_fitting %d",
        desc="Up to which b-value should be used for DRBUDDI's tensor fitting. "
             "Default: 0 , meaning use all b-values")
    disable_initial_rigid = traits.Bool(
        False,
        argstr="--DRBUDDI_disable_initial_rigid %d",
        desc="DRBUDDI performs an initial registration between the up and down data."
             "This registration starts with rigid, followed by a quick diffeomorphic "
             "and finalized by another rigid. This parameter, when set to 1 disables "
             "all these registrations. Default: False")
    start_with_diffeomorphic_for_rigid_reg = traits.Bool(
        False,
        argstr="--DRBUDDI_start_with_diffeomorphic_for_rigid_reg",
        desc="DRBUDDI performs an initial registration between the up and down data. "
             "This registration starts with rigid, followed by a quick diffeomorphic "
             "and finalized by another rigid. This parameter, when set to 1 disables "
             "the very initial rigid registration and starts with the quick diffemorphic. "
             "This is helpful with VERY DISTORTED data, for which the initial rigid "
             "registration is problematic. Default: False")
    estimate_learning_rate_per_iteration = traits.Bool(
        False,
        argstr="--DRBUDDI_estimate_LR_per_iteration %d",
        desc="Flat to estimate learning rate at every iteration. "
             "Makes DRBUDDI slower but better results. Default: False")
    sloppy = traits.Bool(
        False,
        argstr=SLOPPY_DRBUDDI,
        desc="use underpowered (sloppy) registration for speed")


class _DRBUDDIOutputSpec(TraitedSpec):
    # Direct outputs from DRBUDDI
    undistorted_reference = File(exists=True)
    bdown_to_bup_rigid_trans_h5 = File(exists=True)
    undistorted_reference = File(exists=True)
    blip_down_b0 = File(exists=True)
    blip_down_b0_corrected = File(exists=True)
    blip_down_b0_corrected_jac = File(exists=True)
    blip_down_b0_quad = File(exists=True)
    blip_up_b0 = File(exists=True)
    blip_up_b0_corrected = File(exists=True)
    blip_up_b0_corrected_jac = File(exists=True)
    blip_up_b0_quad = File(exists=True)
    deformation_finv = File(exists=True)
    deformation_minv = File(exists=True)
    blip_up_FA = File(exists=True)
    blip_down_FA = File(exists=True)


class DRBUDDI(CommandLine):
    input_spec = _DRBUDDIInputSpec
    output_spec = _DRBUDDIOutputSpec
    _cmd = "DRBUDDI"

    def _format_arg(self, name, spec, value):
        """Trick to get blip_down_bmat symlinked without an arg"""
        if name in ("blip_down_bmat", "blip_up_bmat"):
            return ""
        return super(DRBUDDI, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["undistorted_reference"] = op.abspath("b0_corrected_final.nii")
        outputs["blip_down_b0"] = op.abspath("blip_down_b0.nii")
        outputs["blip_down_b0_corrected"] = op.abspath("blip_down_b0_corrected.nii")
        outputs["blip_down_b0_corrected_jac"] = op.abspath("blip_down_b0_corrected_JAC.nii")
        outputs["blip_down_b0_quad"] = op.abspath("blip_down_b0_quad.nii")
        outputs["blip_up_b0"] = op.abspath("blip_up_b0.nii")
        outputs["blip_up_b0_corrected"] = op.abspath("blip_up_b0_corrected.nii")
        outputs["blip_up_b0_corrected_jac"] = op.abspath("blip_up_b0_corrected_JAC.nii")
        outputs["blip_up_b0_quad"] = op.abspath("blip_up_b0_quad.nii")
        outputs["deformation_finv"] = op.abspath("deformation_FINV.nii.gz")
        outputs["deformation_minv"] = op.abspath("deformation_MINV.nii.gz")

        # There will be an hdf5 transform file if there is an initial rigid
        if not self.inputs.disable_initial_rigid:
            outputs["bdown_to_bup_rigid_trans_h5"] = op.abspath("bdown_to_bup_rigidtrans.hdf5")

        # There will be FA images created if two DWI series were used as inputs
        if self.inputs.fieldmap_type == 'rpe_series':
            outputs["blip_up_FA"] = op.abspath("blip_up_FA.nii")
            outputs["blip_down_FA"] = op.abspath("blip_down_FA.nii")

        return outputs


class _DRBUDDIAggregateOutputsInputSpec(TORTOISEInputSpec):
    blip_assignments = traits.List()
    undistorted_reference = File(exists=True)
    bdown_to_bup_rigid_trans_h5 = File(exists=True)
    undistorted_reference = File(exists=True)
    blip_down_b0 = File(exists=True)
    blip_down_b0_corrected = File(exists=True)
    blip_down_b0_corrected_jac = File(exists=True)
    blip_down_b0_quad = File(exists=True)
    blip_up_b0 = File(exists=True)
    blip_up_b0_corrected = File(exists=True)
    blip_up_b0_corrected_jac = File(exists=True)
    blip_up_b0_quad = File(exists=True)
    deformation_finv = File(exists=True, desc="blip up to b0_corrected")
    deformation_minv = File(exists=True)
    blip_up_FA = File(exists=True)
    blip_down_FA = File(exists=True)
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)


class _DRBUDDIAggregateOutputsOutputSpec(TraitedSpec):
    # Aggregated outputs for convenience
    sdc_warps = OutputMultiObject(File(exists=True))
    sdc_scaling_images = OutputMultiObject(File(exists=True))
    visual_report = File(exists=True)


class DRBUDDIAggregateOutputs(SimpleInterface):
    input_spec = _DRBUDDIAggregateOutputsInputSpec
    output_spec = _DRBUDDIAggregateOutputsOutputSpec

    def _run_interface(self, runtime):

        # there may be 2 transforms for the blip down data. If so, compose them
        if isdefined(self.inputs.bdown_to_bup_rigid_trans_h5):
            # combine the rigid with displacement
            down_warp = op.join(runtime.cwd, "blip_down_composite.nii.gz")
            xfm = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_down_b0,
                transforms=[self.inputs.deformation_minv,
                            self.inputs.bdown_to_bup_rigid_trans_h5],
                reference_image=self.inputs.undistorted_reference,
                output_image=down_warp,
                print_out_composite_warp_file=True,
                interpolation='LanczosWindowedSinc'
            )
            xfm.terminal_output = 'allatonce'
            xfm.resource_monitor = False
            runtime = xfm.run().runtime
        else:
            down_warp = self.inputs.deformation_minv

        # Calculate the scaling images
        scaling_blip_up_file = op.join(runtime.cwd, "blip_up_scale.nii.gz")
        scaling_blip_down_file = op.join(runtime.cwd, "blip_down_scale.nii.gz")
        scaling_blip_up_img = nim.math_img(
            "a/b",
            a=self.inputs.undistorted_reference,
            b=self.inputs.blip_up_b0_corrected)
        scaling_blip_up_img.to_filename(scaling_blip_up_file)
        scaling_blip_down_img = nim.math_img(
            "a/b",
            a=self.inputs.undistorted_reference,
            b=self.inputs.blip_down_b0_corrected)
        scaling_blip_down_img.to_filename(scaling_blip_down_file)

        self._results["sdc_warps"] = [
            self.inputs.deformation_finv if blip_dir == "up" else
            down_warp for blip_dir in
            self.inputs.blip_assignments]
        self._results["sdc_scaling_images"] = [
            scaling_blip_up_file if blip_dir == "up" else
            scaling_blip_down_file for blip_dir in
            self.inputs.blip_assignments]

        # report_file = op.join(runtime.cwd, "drbuddi_report.svg")
        # up_ref, down_ref = self.inputs.blip_up_FA, self.inputs.blip_down_FA if \
        #     self.inputs.fieldmap_type == "rpe_series" else self.inputs.blip_up_b0_corrected, \
        #         self.inputs.blip_down_corrected

        return runtime


def plot_drbuddi(
    blip_up_orig_file, blip_up_transforms, blip_up_scaling_file,
    blip_down_orig_file, blip_down_transforms, blip_down_scaling_image,
    b0_corrected_file, cwd):

    def resample_to_reference(data_img, fname):
        resample = ants.ApplyTransforms(
            dimension=3,
            reference=b0_corrected_file
        )
    pass


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
    up_bmat_file = up_prefix + ".bmtxt"
    down_images = []
    down_bvals = []
    down_bvecs = []
    down_prefix = prefix + "_down_dwi"
    down_dwi_file = down_prefix + ".nii"
    down_bmat_file = down_prefix + ".bmtxt"

    # We know up is first because we concatenated them ourselves
    up_group_name = group_assignments[0]
    blip_assignments = []
    for dwi_file, bval_file, bvec_file, distortion_group in \
            zip(dwi_files, bval_files, bvec_files, group_assignments):

        if distortion_group == up_group_name:
            up_images.append(dwi_file)
            up_bvals.append(bval_file)
            up_bvecs.append(bvec_file)
            blip_assignments.append("up")
        else:
            down_images.append(dwi_file)
            down_bvals.append(bval_file)
            down_bvecs.append(bvec_file)
            blip_assignments.append("down")

    # Write the 4d up image
    up_4d = nim.concat_imgs(up_images, dtype=np.float32, auto_resample=False)
    up_4d.set_data_dtype(np.float32)
    up_4d.to_filename(up_dwi_file)
    up_bval_file, up_bvec_file = write_concatenated_fsl_gradients(
        up_bvals, up_bvecs, up_prefix)

    # Write the 4d down image
    down_4d = nim.concat_imgs(down_images, dtype=np.float32, auto_resample=False)
    down_4d.set_data_dtype(np.float32)
    down_4d.to_filename(down_dwi_file)
    down_bval_file, down_bvec_file = write_concatenated_fsl_gradients(
        down_bvals, down_bvecs, down_prefix)

    # Send back FSL-style gradients
    if not make_bmat:
        return blip_assignments, up_dwi_file, up_bval_file, up_bvec_file, \
            down_dwi_file, down_bval_file, down_bvec_file

    # Convert to bmatrix text file
    make_bmat_file(up_bval_file, up_bvec_file)
    make_bmat_file(down_bval_file, down_bvec_file)

    return blip_assignments, up_dwi_file, up_bmat_file, down_dwi_file, down_bmat_file


def make_bmat_file(bvals, bvecs):
    pout = subprocess.run(
        ["FSLBVecsToTORTOISEBmatrix", op.abspath(bvals), op.abspath(bvecs)])
    print(pout)
    return bvals.replace("bval", "bmtxt")
