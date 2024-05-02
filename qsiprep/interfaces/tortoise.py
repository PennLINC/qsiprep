#!python
"""
Wrappers for the TORTOISE programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import logging
import os
import os.path as op
import subprocess

import nibabel as nb
import nilearn.image as nim
import numpy as np
import pandas as pd
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.viz.utils import compose_view, cuts_from_bbox

from ..viz.utils import plot_denoise
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
)
from .epi_fmap import get_best_b0_topup_inputs_from, safe_get_3d_image
from .fmap import get_distortion_grouping
from .gradients import write_concatenated_fsl_gradients
from .images import split_bvals_bvecs, to_lps

LOGGER = logging.getLogger("nipype.interface")

SLOPPY_DRBUDDI = (
    "--DRBUDDI_stage "
    "\[learning_rate=\{0.3\},cfs=\{100:8:4\},field_smoothing=\{9:0\},"
    "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] "
)


class TORTOISEInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(desc="number of OMP threads")


class TORTOISECommandLine(CommandLine):
    """Support for TORTOISE commands that utilize OpenMP
    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.
    """

    input_spec = TORTOISEInputSpec
    _num_threads = None

    def __init__(self, **inputs):
        super(TORTOISECommandLine, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, "num_threads")
        if not self._num_threads:
            self._num_threads = os.environ.get("OMP_NUM_THREADS", None)
            if not self._num_threads:
                self._num_threads = os.environ.get("NSLOTS", None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({"OMP_NUM_THREADS": str(self.inputs.num_threads)})

    def run(self, **inputs):
        if "num_threads" in inputs:
            self.inputs.num_threads = inputs["num_threads"]
        self._num_threads_update()
        return super(TORTOISECommandLine, self).run(**inputs)


class _GatherDRBUDDIInputsInputSpec(TORTOISEInputSpec):
    dwi_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    bval_files = traits.Either(InputMultiObject(File(exists=True)), File(exists=True))
    bvec_files = traits.Either(InputMultiObject(File(exists=True)), File(exists=True))
    original_files = InputMultiObject(File(exists=True))
    b0_threshold = traits.CInt(100, usedefault=True)
    epi_fmaps = InputMultiObject(
        File(exists=True), desc="files from fmaps/ for distortion correction"
    )
    raw_image_sdc = traits.Bool(True, usedefault=True)
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)
    dwi_series_pedir = traits.Enum("i", "i-", "j", "j-", "k", "k-", mandatory=True)


class _GatherDRBUDDIInputsOutputSpec(TraitedSpec):
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
                working_dir=runtime.cwd,
            )
        else:
            bval_files, bvec_files = self.inputs.bval_files, self.inputs.bvec_files

        if self.inputs.fieldmap_type == "rpe_series":
            (
                self._results["blip_assignments"],
                self._results["blip_up_image"],
                self._results["blip_up_bmat"],
                self._results["blip_down_image"],
                self._results["blip_down_bmat"],
            ) = split_into_up_and_down_niis(
                dwi_files=self.inputs.dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                original_images=self.inputs.original_files,
                prefix=op.join(runtime.cwd, "drbuddi"),
                make_bmat=True,
            )

        elif self.inputs.fieldmap_type == "epi":
            # Use the same function that was used to get images for TOPUP, but get the images
            # directly from the CSV
            _, _, _, b0_csv, _, _ = get_best_b0_topup_inputs_from(
                dwi_file=self.inputs.dwi_files,
                bval_file=bval_files,
                b0_threshold=self.inputs.b0_threshold,
                cwd=runtime.cwd,
                bids_origin_files=self.inputs.original_files,
                epi_fmaps=self.inputs.epi_fmaps,
                max_per_spec=True,
                raw_image_sdc=self.inputs.raw_image_sdc,
            )

            b0s_df = pd.read_csv(b0_csv)
            selected_images = b0s_df[b0s_df.selected_for_sdc].reset_index(drop=True)
            up_row = selected_images.loc[0]
            down_row = selected_images.loc[1]
            up_img = to_lps(safe_get_3d_image(up_row.bids_origin_file, up_row.original_volume))
            up_img.set_data_dtype("float32")
            down_img = to_lps(
                safe_get_3d_image(down_row.bids_origin_file, down_row.original_volume)
            )
            down_img.set_data_dtype("float32")

            # Save the images
            blip_up_nii = op.join(runtime.cwd, "blip_up_b0.nii")
            blip_down_nii = op.join(runtime.cwd, "blip_down_b0.nii")
            up_img.to_filename(blip_up_nii)
            down_img.to_filename(blip_down_nii)
            self._results["blip_up_image"] = blip_up_nii
            self._results["blip_down_image"] = blip_down_nii
            self._results["blip_assignments"] = split_into_up_and_down_niis(
                dwi_files=self.inputs.dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                original_images=self.inputs.original_files,
                prefix=op.join(runtime.cwd, "drbuddi"),
                make_bmat=False,
                assignments_only=True,
            )
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
    num_threads = traits.Int(
        desc="number of OMP threads",
        argstr="--ncores %d",
        help="Number of cores to use in the CPU version. The default is 50% of system cores.",
        nohash=True,
    )
    blip_up_image = File(
        exists=True,
        help="Full path to the input UP NIFTI file to be corrected.",
        argstr="-u %s",
        mandatory=True,
        copyfile=True,
    )
    blip_up_bmat = File(
        exists=True,
        help="Full path to the input UP NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True,
    )
    blip_up_json = File(
        exists=True,
        help="Phase encoding information will be read from this",
        argstr="--up_json %s",
        mandatory=True,
        copyfile=True,
    )
    blip_down_image = File(
        exists=True,
        help="Full path to the input DOWN NIFTI file to be corrected.",
        argstr="-d %s",
        mandatory=True,
        copyfile=True,
    )
    blip_down_bmat = File(
        exists=True,
        help="Full path to the input DOWN NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True,
    )
    structural_image = InputMultiObject(
        File(exists=True, copyfile=False),
        argstr="-s %s",
        help="Path(s) to anatomical image files. Can provide more than one. NO T1W's!!",
    )
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)
    blip_assignments = traits.List()
    tensor_fit_bval_max = traits.Int(
        0,
        argstr="--DRBUDDI_DWI_bval_tensor_fitting %d",
        desc="Up to which b-value should be used for DRBUDDI's tensor fitting. "
        "Default: 0 , meaning use all b-values",
    )
    disable_initial_rigid = traits.Bool(
        False,
        argstr="--DRBUDDI_disable_initial_rigid %d",
        desc="DRBUDDI performs an initial registration between the up and down data."
        "This registration starts with rigid, followed by a quick diffeomorphic "
        "and finalized by another rigid. This parameter, when set to 1 disables "
        "all these registrations. Default: False",
    )
    start_with_diffeomorphic_for_rigid_reg = traits.Bool(
        False,
        argstr="--DRBUDDI_start_with_diffeomorphic_for_rigid_reg",
        desc="DRBUDDI performs an initial registration between the up and down data. "
        "This registration starts with rigid, followed by a quick diffeomorphic "
        "and finalized by another rigid. This parameter, when set to 1 disables "
        "the very initial rigid registration and starts with the quick diffemorphic. "
        "This is helpful with VERY DISTORTED data, for which the initial rigid "
        "registration is problematic. Default: False",
    )
    estimate_learning_rate_per_iteration = traits.Bool(
        False,
        argstr="--DRBUDDI_estimate_LR_per_iteration %d",
        desc="Flat to estimate learning rate at every iteration. "
        "Makes DRBUDDI slower but better results. Default: False",
    )
    sloppy = traits.Bool(
        False, argstr=SLOPPY_DRBUDDI, desc="use underpowered (sloppy) registration for speed"
    )
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr="--disable_itk_threads")


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
    structural_image = File(exists=True)


class DRBUDDI(TORTOISECommandLine):
    input_spec = _DRBUDDIInputSpec
    output_spec = _DRBUDDIOutputSpec
    _cmd = "DRBUDDI"

    def _format_arg(self, name, spec, value):
        """Trick to get blip_down_bmat symlinked without an arg"""
        if name in ("blip_down_bmat", "blip_up_bmat"):
            return ""
        if name == "structural_image":
            return "-s " + " ".join(value)
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
        if self.inputs.fieldmap_type == "rpe_series":
            outputs["blip_up_FA"] = op.abspath("blip_up_FA.nii")
            outputs["blip_down_FA"] = op.abspath("blip_down_FA.nii")

        # If there was a T2w
        if self.inputs.structural_image:
            outputs["structural_image"] = op.abspath("structural_used.nii")
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
    structural_image = File(exists=True)
    wm_seg = File(exists=True, desc="White matter segmentation image")


class _DRBUDDIAggregateOutputsOutputSpec(TraitedSpec):
    # Aggregated outputs for convenience
    sdc_warps = OutputMultiObject(File(exists=True))
    sdc_scaling_images = OutputMultiObject(File(exists=True))
    # Fieldmap outputs for the reports
    up_fa_corrected_image = File(exists=True)
    down_fa_corrected_image = File(exists=True)
    # The best image for coregistration to the corrected DWI
    b0_ref = File(exists=True)


class DRBUDDIAggregateOutputs(SimpleInterface):
    input_spec = _DRBUDDIAggregateOutputsInputSpec
    output_spec = _DRBUDDIAggregateOutputsOutputSpec

    def _run_interface(self, runtime):

        # If the structural image has been used, return that as the b0ref, otherwise
        # it's the b0_corrected_final
        self._results["b0_ref"] = (
            self.inputs.structural_image
            if isdefined(self.inputs.structural_image)
            else self.inputs.undistorted_reference
        )

        # there may be 2 transforms for the blip down data. If so, compose them
        if isdefined(self.inputs.bdown_to_bup_rigid_trans_h5):
            # combine the rigid with displacement
            down_warp = op.join(runtime.cwd, "blip_down_composite.nii.gz")
            xfm = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_down_b0,
                transforms=[self.inputs.deformation_minv, self.inputs.bdown_to_bup_rigid_trans_h5],
                reference_image=self.inputs.undistorted_reference,
                output_image=down_warp,
                print_out_composite_warp_file=True,
                interpolation="LanczosWindowedSinc",
            )
            xfm.terminal_output = "allatonce"
            xfm.resource_monitor = False
            _ = xfm.run()
        else:
            down_warp = self.inputs.deformation_minv

        # Calculate the scaling images
        scaling_blip_up_file = op.join(runtime.cwd, "blip_up_scale.nii.gz")
        scaling_blip_down_file = op.join(runtime.cwd, "blip_down_scale.nii.gz")
        scaling_blip_up_img = nim.math_img(
            "a/b", a=self.inputs.undistorted_reference, b=self.inputs.blip_up_b0_corrected
        )
        scaling_blip_up_img.to_filename(scaling_blip_up_file)
        scaling_blip_down_img = nim.math_img(
            "a/b", a=self.inputs.undistorted_reference, b=self.inputs.blip_down_b0_corrected
        )
        scaling_blip_down_img.to_filename(scaling_blip_down_file)

        self._results["sdc_warps"] = [
            self.inputs.deformation_finv if blip_dir == "up" else down_warp
            for blip_dir in self.inputs.blip_assignments
        ]
        self._results["sdc_scaling_images"] = [
            scaling_blip_up_file if blip_dir == "up" else scaling_blip_down_file
            for blip_dir in self.inputs.blip_assignments
        ]

        if self.inputs.fieldmap_type == "rpe_series":
            fa_up_warped = fname_presuffix(
                self.inputs.blip_up_FA, newpath=runtime.cwd, suffix="_corrected"
            )
            xfm_fa_up = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_up_FA,
                transforms=[self.inputs.deformation_finv],
                reference_image=self.inputs.undistorted_reference,
                output_image=fa_up_warped,
                interpolation="NearestNeighbor",
            )
            xfm_fa_up.terminal_output = "allatonce"
            xfm_fa_up.resource_monitor = False
            xfm_fa_up.run()

            fa_down_warped = fname_presuffix(
                self.inputs.blip_down_FA, newpath=runtime.cwd, suffix="_corrected"
            )
            xfm_fa_down = ants.ApplyTransforms(
                # input_image is ignored because print_out_composite_warp_file is True
                input_image=self.inputs.blip_down_FA,
                transforms=[self.inputs.deformation_minv, self.inputs.bdown_to_bup_rigid_trans_h5],
                reference_image=self.inputs.undistorted_reference,
                output_image=fa_down_warped,
                interpolation="NearestNeighbor",
            )
            xfm_fa_down.terminal_output = "allatonce"
            xfm_fa_down.resource_monitor = False
            xfm_fa_down.run()
            self._results["up_fa_corrected_image"] = fa_up_warped
            self._results["down_fa_corrected_image"] = fa_down_warped

        return runtime


class _GibbsInputSpec(TORTOISEInputSpec, SeriesPreprocReportInputSpec):
    """Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75)
    phase_encoding_dir nsh minW(optional) maxW(optional)"""

    in_file = traits.File(exists=True, mandatory=True, position=0, argstr="%s")
    out_file = traits.File(
        argstr="%s",
        position=1,
        name_source="in_file",
        name_template="%s_unrung.nii",
        use_extension=False,
    )
    kspace_coverage = traits.Float(mandatory=True, position=2, argstr="%.4f")
    phase_encoding_dir = traits.Enum(
        0, 1, mandatory=True, argstr="%d", position=3, desc="0: horizontal, 1:vertical"
    )
    nsh = traits.Int(argstr="%d", position=4)
    min_w = traits.Int()
    mask = File()
    num_threads = traits.Int(1, usedefault=True, nohash=True)


class _GibbsOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(exists=True)


class Gibbs(SeriesPreprocReport, TORTOISECommandLine):
    input_spec = _GibbsInputSpec
    output_spec = _GibbsOutputSpec
    _cmd = "Gibbs"

    def _get_plotting_images(self):
        input_dwi = nim.load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get("out_file")
        denoised_nii = nim.load_img(ref_name)
        return input_dwi, denoised_nii, None

    def _generate_report(self):
        """Generate a reportlet."""
        LOGGER.info("Generating denoising visual report")

        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
        image_intensities = np.array([img.mean() for img in image_data.T])
        lowb_index = int(np.argmax(image_intensities))
        highb_index = int(np.argmin(image_intensities))

        # Original images
        orig_lowb_nii = input_dwi.slicer[..., lowb_index]
        orig_highb_nii = input_dwi.slicer[..., highb_index]

        # Denoised images
        denoised_lowb_nii = denoised_nii.slicer[..., lowb_index]
        denoised_highb_nii = denoised_nii.slicer[..., highb_index]

        # Find spatial extent of the image
        contour_nii = mask_nii = None
        if isdefined(self.inputs.mask):
            contour_nii = nim.load_img(self.inputs.mask)
        else:
            mask_nii = nim.threshold_img(denoised_lowb_nii, 50)
        cuts = cuts_from_bbox(contour_nii or mask_nii, cuts=self._n_cuts)

        diff_lowb_nii = nb.Nifti1Image(
            orig_lowb_nii.get_fdata() - denoised_lowb_nii.get_fdata(),
            affine=denoised_lowb_nii.affine,
        )
        diff_highb_nii = nb.Nifti1Image(
            orig_highb_nii.get_fdata() - denoised_highb_nii.get_fdata(),
            affine=denoised_highb_nii.affine,
        )

        # Call composer
        compose_view(
            plot_denoise(
                denoised_lowb_nii,
                denoised_highb_nii,
                "moving-image",
                estimate_brightness=True,
                cuts=cuts,
                label="De-Gibbs",
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            plot_denoise(
                diff_lowb_nii,
                diff_highb_nii,
                "fixed-image",
                estimate_brightness=True,
                cuts=cuts,
                label="Estimated Ringing",
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            out_file=self._out_report,
        )

        self._calculate_nmse(input_dwi, denoised_nii)


class _TORTOISEConvertInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True, copyfile=True)
    bvec_file = File(exists=True, mandatory=True, copyfile=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)


class _TORTOISEConvertOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    mask_file = File(exists=True)
    bmtxt_file = File(exists=True)


class TORTOISEConvert(SimpleInterface):
    input_spec = _TORTOISEConvertInputSpec
    output_spec = _TORTOISEConvertOutputSpec

    def _run_interface(self, runtime):
        """Convert gzipped niftis and bval/bvec into TORTOISE format."""
        dwi_file = fname_presuffix(
            self.inputs.dwi_file, newpath=runtime.cwd, use_ext=False, suffix=".nii"
        )
        dwi_img = nim.load_img(self.inputs.dwi_file, dtype="float32")
        dwi_img.set_data_dtype("float32")
        dwi_img.to_filename(dwi_file)
        bmtxt_file = make_bmat_file(self.inputs.bval_file, self.inputs.bvec_file)

        if isdefined(self.inputs.mask_file):
            mask_file = fname_presuffix(
                self.inputs.mask_file, newpath=runtime.cwd, use_ext=False, suffix=".nii"
            )
            mask_img = nim.load_img(self.inputs.mask_file, dtype="float32")
            mask_img.set_data_dtype("float32")
            mask_img.to_filename(mask_file)
            self._results["mask_file"] = mask_file

        self._results["dwi_file"] = dwi_file
        self._results["bmtxt_file"] = bmtxt_file

        return runtime


class TORTOISEReconCommandLine(TORTOISECommandLine):
    _link_me = ["bmtxt_file"]

    # Most TORTOISE commandline programs don't offer an option for
    # what the output file should be called. The input file
    def _list_outputs(self):

        # If this is a special case where in_file is not defined and
        # there is no _suffix_map, do the normal version
        if not hasattr(self.inputs, "in_file"):
            raise Exception("TORTOISEReconCommandLine requires an in_file")

        outputs = self.output_spec().get()
        if not self._suffix_map:
            raise Exception("Compute classes need to have a _suffix_map")

        for trait_name, suffix in self._suffix_map.items():
            new_fname = op.basename(self.inputs.in_file.replace(".nii", suffix + ".nii"))
            if op.exists(new_fname):
                outputs[trait_name] = op.abspath(new_fname)
        return outputs

    def _format_arg(self, name, spec, value):
        if not hasattr(self, "_link_me"):
            raise Exception
        if name in self._link_me:
            return ""
        return super(TORTOISEReconCommandLine, self)._format_arg(name, spec, value)


class _TORTOISEEstimatorInputSpec(TORTOISEInputSpec):
    in_file = File(
        exists=False,
        mandatory=True,
        argstr="--input %s",
        desc="Full path to the input NIFTI DWI",
        copyfile=False,
    )
    bmtxt_file = File(
        exists=True, mandatory=True, desc="Full path to the input bmtxt file", copyfile=False
    )
    mask = File(exists=True, argstr="--mask %s", desc="Full path to the mask NIFTI image")
    bval_cutoff = traits.CInt(
        argstr="--bval_cutoff %d",
        desc="Maximum b-value volumes to use for tensor fitting. (Default: use all volumes)",
    )
    inclusion_file = File(exists=True, argstr="--inclusion %s")
    voxelwise_bmat_file = File(
        exists=True, desc="Use voxelwise Bmatrices for gradient non-linearity correction"
    )


class _EstimateTensorInputSpec(_TORTOISEEstimatorInputSpec):
    reg_mode = traits.Enum(
        "WLLS",
        "NLLS",
        "RESTORE",
        "DIAG",
        "N2",
        "NT2",
        argstr="--reg_mode %s",
        usedefault=True,
        desc="Regression mode. WLLS: Weighted linear least squares, "
        "NLLS: Nonlinear least squares, "
        "RESTORE: Robust NLLS, "
        "DIAG: Diagonal Only NLLS, "
        "N2: Full diffusion tensor + free water NLLS, "
        "NT2: One full parenchymal diffusion tensor + one full flow tensor",
    )
    free_water_diffusivity = traits.CInt(
        default_value=3000,
        argstr="--free_water_diffusivity %d",
        desc="Free water diffusivity in (mu m)^2/s for N2 fitting.",
    )
    write_cs = traits.Bool(
        default_value=True,
        usedefault=True,
        argstr="--write_CS %d",
        desc="Write the Chi-squred image?",
    )
    noise_file = File(
        exists=True,
        copyfile=False,
        desc="Use this image for weigthing and correction of interpolation artifacts.",
    )


class _EstimateTensorOutputSpec(TraitedSpec):
    dt_file = File(exists=True)
    am_file = File(exists=True)
    cs_file = File(exists=True)


class EstimateTensor(TORTOISEReconCommandLine):
    input_spec = _EstimateTensorInputSpec
    output_spec = _EstimateTensorOutputSpec
    _cmd = "EstimateTensor"
    _suffix_map = {"dt_file": "_L1_DT", "am_file": "_L1_AM"}
    _link_me = [
        "bmtxt",
    ]

    def _format_arg(self, name, spec, value):
        """Trick to get noise image or voxelwise bmat symlinked without an arg"""
        if name == "noise_file":
            return "--use_noise 1"
        if name == "voxelwise_bmat_file":
            return "--use_voxelwise_bmat 1"
        return super(EstimateTensor, self)._format_arg(name, spec, value)


class _TensorMapInputSpec(TORTOISEInputSpec):
    in_file = File(exists=True, mandatory=True, argstr="%s", position=1, copyfile=True)
    am_file = File(exists=True, copyfile=False)


class _TensorMapCmdline(TORTOISEReconCommandLine):
    _link_me = ["am_file"]


class _ComputeFAMapInputSpec(_TensorMapInputSpec):
    filter_outliers = traits.Bool(True, usedefault=True, argstr="%d", position=2)


class _ComputeFAMapOutputSpec(TraitedSpec):
    fa_file = File(exists=True)


class ComputeFAMap(_TensorMapCmdline):
    input_spec = _ComputeFAMapInputSpec
    output_spec = _ComputeFAMapOutputSpec
    _cmd = "ComputeFAMap"
    _suffix_map = {"fa_file": "_FA"}


class _ComputeRDMapOutputSpec(TraitedSpec):
    rd_file = File(exists=True)


class ComputeRDMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeRDMapOutputSpec
    _cmd = "ComputeRDMap"
    _suffix_map = {"rd_file": "_RD"}


class _ComputeADMapOutputSpec(TraitedSpec):
    ad_file = File(exists=True)


class ComputeADMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeADMapOutputSpec
    _cmd = "ComputeADMap"
    _suffix_map = {"ad_file": "_AD"}


class _ComputeLIMapOutputSpec(TraitedSpec):
    li_file = File(exists=True)


class ComputeLIMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeLIMapOutputSpec
    _cmd = "ComputeLIMap"
    _suffix_map = {"li_file": "_LI"}


class _EstimateMAPMRIInputSpec(_TORTOISEEstimatorInputSpec):
    dt_file = File(
        exists=True, argstr="--dti %s", requires=["a0_file"], desc="DTI image computed externally"
    )
    a0_file = File(exists=True, argstr="--A0 %s", desc="A0 image computed externally")
    map_order = traits.Int(
        default_value=4, usedefault=True, argstr="--map_order %d", desc="MAPMRI order"
    )
    big_delta = traits.CFloat(argstr="--big_delta %.7f", desc="Big Delta in seconds")
    small_delta = traits.CFloat(argstr="--small_delta %.7f", desc="Small Delta in seconds")


class _EstimateMAPMRIOutputSpec(TraitedSpec):
    coeffs_file = File(exists=True)
    uvec_file = File(exists=True)


class EstimateMAPMRI(TORTOISEReconCommandLine):
    input_spec = _EstimateMAPMRIInputSpec
    output_spec = _EstimateMAPMRIOutputSpec
    _cmd = "EstimateMAPMRI"
    _suffix_map = {"coeffs_file": "_mapmri", "uvec_file": "_uvec"}


class _ComputeMAPMRIInputSpec(TORTOISEInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="_mapmri.nii file",
        copyfile=False,
    )
    uvec_file = File(exists=True, mandatory=True, argstr="%s", position=2, copyfile=False)


class _ComputeMAPMRI_PAOutputSpec(TraitedSpec):
    pa_file = File(exists=True)
    path_file = File(exists=True)


class ComputeMAPMRI_PA(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_PAOutputSpec
    _cmd = "ComputeMAPMRI_PA"
    _suffix_map = {"pa_file": "_PA", "path_file": "_PAth"}


class _ComputeMAPMRI_RTOPOutputSpec(TraitedSpec):
    rtop_file = File(exists=True)
    rtap_file = File(exists=True)
    rtpp_file = File(exists=True)


class ComputeMAPMRI_RTOP(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_RTOPOutputSpec
    _cmd = "ComputeMAPMRI_RTOP"
    _suffix_map = {"rtap_file": "_RTAP", "rtop_file": "_RTOP", "rtpp_file": "_RTPP"}


class _ComputeMAPMRI_NGOutputSpec(TraitedSpec):
    ng_file = File(exists=True)
    ngpar_file = File(exists=True)
    ngperp_file = File(exists=True)


class ComputeMAPMRI_NG(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_NGOutputSpec
    _cmd = "ComputeMAPMRI_NG"
    _suffix_map = {"ng_file": "_NG", "ngpar_file": "_NGpar", "ngperp_file": "_NGperp"}


def split_into_up_and_down_niis(
    dwi_files,
    bval_files,
    bvec_files,
    original_images,
    prefix,
    make_bmat=True,
    assignments_only=False,
):
    """Takes the concatenated output from pre_hmc_wf and split it into "up" and "down"
    decompressed nifti files with float32 datatypes."""
    group_names, group_assignments = get_distortion_grouping(original_images)

    if not len(set(group_names)) == 2 and not assignments_only:
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
    for dwi_file, bval_file, bvec_file, distortion_group in zip(
        dwi_files, bval_files, bvec_files, group_assignments
    ):

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

    if assignments_only:
        return blip_assignments

    # Write the 4d up image
    up_4d = nim.concat_imgs(up_images, dtype="float32", auto_resample=False)
    up_4d.set_data_dtype("float32")
    up_4d.to_filename(up_dwi_file)
    up_bval_file, up_bvec_file = write_concatenated_fsl_gradients(up_bvals, up_bvecs, up_prefix)

    # Write the 4d down image
    down_4d = nim.concat_imgs(down_images, dtype="float32", auto_resample=False)
    down_4d.set_data_dtype("float32")
    down_4d.to_filename(down_dwi_file)
    down_bval_file, down_bvec_file = write_concatenated_fsl_gradients(
        down_bvals, down_bvecs, down_prefix
    )

    # Send back FSL-style gradients
    if not make_bmat:
        return (
            blip_assignments,
            up_dwi_file,
            up_bval_file,
            up_bvec_file,
            down_dwi_file,
            down_bval_file,
            down_bvec_file,
        )

    # Convert to bmatrix text file
    make_bmat_file(up_bval_file, up_bvec_file)
    make_bmat_file(down_bval_file, down_bvec_file)

    return blip_assignments, up_dwi_file, up_bmat_file, down_dwi_file, down_bmat_file


def make_bmat_file(bvals, bvecs):
    pout = subprocess.run(["FSLBVecsToTORTOISEBmatrix", op.abspath(bvals), op.abspath(bvecs)])
    print(pout)
    return bvals.replace("bval", "bmtxt")


def generate_drbuddi_boilerplate(fieldmap_type, t2w_sdc, with_topup=False):
    """Generate boilerplate that describes how DRBUDDI is being used."""

    desc = ["\n\nDRBUDDI [@drbuddi], part of the TORTOISE [@tortoisev3] software package,"]
    if not with_topup:
        # Until now there will have been no description of the SDC procedure.
        # Add extra details about the input data.
        desc.append(
            "was used to perform susceptibility distortion correction. "
            "Data was collected with reversed phase-encode blips, resulting "
            "in pairs of images with distortions going in opposite directions."
        )
    else:
        desc += ["was used to perform a second stage of distortion correction."]

    # Describe what's going on
    if fieldmap_type == "epi":
        desc.append(
            "DRBUDDI used b=0 reference images with reversed "
            "phase encoding directions to estimate"
        )
    else:
        desc.append(
            "DRBUDDI used multiple motion-corrected DWI series acquired "
            "with opposite phase encoding "
            "directions. A b=0 image **and** the Fractional Anisotropy "
            "images from both phase encoding diesctions were used together in "
            "a multi-modal registration to estimate"
        )
    desc.append("the susceptibility-induced off-resonance field.")

    if t2w_sdc:
        desc.append("A T2-weighted image was included in the multimodal registration.")
    desc.append(
        "Signal intensity was adjusted "
        "in the final interpolated images using a method similar to LSR.\n\n"
    )
    return " ".join(desc)
