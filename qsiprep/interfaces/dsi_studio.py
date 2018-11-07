#!python
from __future__ import print_function

from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec,
                                    CommandLine, File, traits, isdefined)

import os
from glob import glob
from nipype.interfaces.utility import Function


# Step 1 from DSI Studio, importing DICOM files or nifti
class DSIStudioCreateSrcInputSpec(CommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(
        desc="DWI Nifti file",
        argstr="--source=%s",
        exists=True,
        copyfile=False)
    input_dicom_dir = File(
        desc="Directory with DICOM data from only the dwi",
        exists=True,
        argstr="--source=%s")
    input_bvals_file = File(
        desc="Text file containing b values", exists=True, argstr="--bval=%s")
    input_bvecs_file = File(
        desc="Text file containing b vectors (FSL format)",
        exists=True,
        argstr="--bvec=%s")
    input_b_table_file = File(
        desc="Text file containing q-space sampling (DSI Studio format)",
        exists=True,
        argstr="--b_table=%s")
    recursive = traits.Bool(
        False,
        desc="Search for DICOM files recursively",
        argstr="--recursive=1")
    subject_id = traits.Str("data")
    output_src = File(
        name_template="%s.src.gz",
        desc="Output file (.src.gz)",
        argstr="--output=%s",
        name_source="subject_id",
        genfile=True)
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")


class DSIStudioCreateSrcOutputSpec(TraitedSpec):
    output_src = File(
        name_template="%s.src.gz",
        desc="Output file (.src.gz)",
        argstr="--output=%s",
        name_source="subject_id")


class DSIStudioCreateSrc(CommandLine):
    input_spec = DSIStudioCreateSrcInputSpec
    output_spec = DSIStudioCreateSrcOutputSpec
    _cmd = "dsi_studio --action=src "

    def _gen_filename(self):
        cwd = os.getcwd()
        return os.path.join(cwd, self.inputs.subject_id + ".src.gz")

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_src"] = self._gen_filename()
        return outputs


# Step 2 reonstruct ODFs
class DSIStudioReconstructionInputSpec(CommandLineInputSpec):
    input_src_file = File(
        desc="DSI Studio src file",
        mandatory=True,
        exists=True,
        copyfile=False,
        argstr="--source=%s")
    mask = File(
        desc="Volume to mask brain voxels",
        exists=True,
        copyfile=False,
        argstr="--mask=%s")
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")
    thread_count = traits.Int(1, usedefault=True, argstr="--thread_count=%d")
    output_odf = traits.Bool(
        False,
        usedefault=True,
        desc="Include full ODF's in output",
        argstr="--record_odf=1")
    odf_order = traits.Enum((8, 4, 5, 6, 10, 12, 16, 20),
                            usedefault=True,
                            desc="ODF tesselation order")
    check_btable = traits.Enum(
        (1, 0),
        usedefault=True,
        argstr="--check_btable=%d",
        desc="Check if btable matches nifti orientation (not foolproof)")
    num_fibers = traits.Int(
        3,
        usedefault=True,
        argstr="--num_fiber=%d",
        desc="number of fiber populations estimated at each voxel")
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")

    # Decomposition traits
    decomposition = traits.Bool(
        False, argstr="--decomposition=1", desc="Apply ODF Decomposition")
    decomp_fraction = traits.Float(
        0.05,
        desc="Decomposition Fraction",
        argstr="--param3=%.2f",
        requires=["decomposition"])
    decomp_m_value = traits.Int(
        10,
        desc="Decomposition m value",
        argstr="--param4=%d",
        requires=["decomposition"])

    # Deconvolution traits
    deconvolution = traits.Bool(
        False, argstr="--deconvolution=1", desc="Apply ODF Deconvolution")
    deconv_regularization = traits.Float(
        0.5,
        requires=["deconvolution"],
        desc="Deconvolution regularization parameter",
        argstr="--param2=%.2f")


class DSIStudioReconstructionOutputSpec(TraitedSpec):
    output_fib = File(desc="Output File", exists=True)


class DSIStudioGQIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    ratio_of_mean_diffusion_distance = traits.Float(
        1.25, usedefault=True, argstr="--param0=%.4f")


class DSIStudioDSIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    hamming_window_len = traits.Int(16, argstr="--param0=%d")


class DSIStudioReconstruction(CommandLine):
    input_spec = DSIStudioReconstructionInputSpec
    output_spec = DSIStudioReconstructionOutputSpec
    cmd = "dsi_studio --action=rec "


class DSIStudioDTIReconstruction(DSIStudioReconstruction):
    cmd = "dsi_studio --action=rec --method=1"
    input_spec = DSIStudioReconstructionInputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print("current dir", os.getcwd())
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        print("input src", self.inputs.input_src_file)
        print("split src name", srcname)
        target = os.path.join(os.getcwd(), srcname) + "*dti.fib.gz"
        print("search target", target)
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class DSIStudioGQIReconstruction(DSIStudioReconstruction):
    cmd = "dsi_studio --action=rec --method=4"
    input_spec = DSIStudioGQIReconstructionInputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print("current dir", os.getcwd())
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        print("input src", self.inputs.input_src_file)
        print("split src name", srcname)
        target = os.path.join(os.getcwd(), srcname) + \
            "*odf%d*f%d*gqi.%.2f.fib.gz" % (self.inputs.odf_order,
            self.inputs.num_fibers,
            self.inputs.ratio_of_mean_diffusion_distance)
        print("search target", target)
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class DSIStudioExportInputSpec(CommandLineInputSpec):
    input_file = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    to_export = traits.Str(mandatory=True, argstr="--export=%s")


class DSIStudioExportOutputSpec(CommandLineInputSpec):
    gfa_file = File(desc="Exported files")
    fa0_file = File(desc="Exported files")
    fa1_file = File(desc="Exported files")
    fa2_file = File(desc="Exported files")
    fa3_file = File(desc="Exported files")
    fa4_file = File(desc="Exported files")
    iso_file = File(desc="Exported files")
    image0_file = File(desc="Exported files")


class DSIStudioExport(CommandLine):
    input_spec = DSIStudioExportInputSpec
    output_spec = DSIStudioExportOutputSpec
    cmd = "dsi_studio --action=exp"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        to_expect = self.inputs.to_export.split(",")
        for expected in to_expect:
            print("searching for", expected, "in", os.getcwd())
            matches = glob("*" + expected + "*.nii.gz")
            print("found", matches)
            if len(matches) == 1:
                outputs[expected + "_file"] = os.path.abspath(matches[0])
            print(outputs)
        return outputs


def compute_mda(fib_file):
    import gzip
    from scipy.io.matlab import loadmat
    import re
    import numpy as np
    import os
    if fib_file.endswith(".gz"):
        with gzip.open(fib_file, "rb") as f:
            m = loadmat(f)
    elif fib_file.endswith(".fib") or fib_file.endswith(".mat"):
        m = loadmat(fib_file)
    else:
        raise ValueError("Unrecognized file type")
    odf_vars = [k for k in m.keys() if re.match("odf\\d+", k)]
    mdas = []
    msk = m["gfa"].squeeze() > 0
    for n in range(len(odf_vars)):
        varname = "odf%d" % n
        odfs = m[varname]
        odfs = odfs / odfs.sum(0)
        mu = np.power(odfs.min(0) / odfs.max(0), 2. / 3)
        mdas.append((1 - mu) / np.sqrt(1 + 2 * mu**2))
    # Was a mask used?
    mda = np.concatenate(mdas)
    mda = mda[np.isfinite(mda)]
    output = np.zeros_like(m['fa0'].squeeze())
    output[msk] = mda

    voxel_size = m['voxel_size'].squeeze()
    shape = m['dimension'].squeeze()
    output_aff = np.eye(4)
    output_aff[(0, 1, 2), (0, 1, 2)] = voxel_size

    import nibabel as nib
    nib.Nifti1Image(
        output.reshape(shape, order="F")[::-1, ::-1, :],
        output_aff).to_filename("MDA.nii.gz")
    import os
    return os.path.abspath("MDA.nii.gz")


mda_from_fib = Function(
    input_names=["fib_file"], output_names=["mda_file"], function=compute_mda)


class DSIStudioConnectivityMatrixInputSpec(CommandLineInputSpec):
    tract_file = File(exists=True, argstr="--tract=%s")
    input_fib = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    fiber_count = traits.Int(xor=["seed_count"], argstr="--fiber_count=%d")
    seed_count = traits.Int(xor=["fiber_count"], argstr="--seed_count=%d")
    seed_plan = traits.Enum((0, 1), argstr="--seed_plan=%d")
    initial_dir = traits.Enum((0, 1, 2), argstr="--initial_dir=%d")
    interpolation = traits.Enum((0, 1, 2), argstr="--interpolation=%d")
    # ROI related options
    seed_file = File(
        exists=True,
        desc="specify the seeding file. "
        "Supported file format includes text, Analyze, and "
        "nifti files.",
        argstr="--seed=%s")
    to_export = traits.Str(argstr="--export=%s")
    connectivity = traits.Str(argstr="--connectivity=%s")
    connectivity_type = traits.Str(argstr="--connectivity_type=%s")
    connectivity_value = traits.Str(argstr="--connectivity_value=%s")
    random_seed = traits.Bool(argstr="--random_seed=1")
    # Tracking options
    fa_threshold = traits.Float(argstr="--fa_threshold=%.2f")
    step_size = traits.CFloat(argstr="--step_size=%.2f")
    turning_angle = traits.CFloat(argstr="--turning_angle=%.2f")
    interpo_angle = traits.CFloat(argstr="--interpo_angle=%.2f")
    smoothing = traits.CFloat(argstr="--smoothing=%.2f")
    min_length = traits.CInt(argstr="--min_length=%d")
    max_length = traits.CInt(argstr="--max_length=%d")


class DSIStudioConnectivityMatrixOutputSpec(TraitedSpec):
    # What to write out
    connectivity_matrices = traits.List()


class DSIStudioConnectivityMatrix(CommandLine):
    input_spec = DSIStudioConnectivityMatrixInputSpec
    output_spec = DSIStudioConnectivityMatrixOutputSpec
    cmd = "dsi_studio --action=ana "

    def _list_outputs(self):
        outputs = self.output_spec().get()
        results = glob("*.connectivity.mat")
        assert len(results) > 1
        outputs["connectivity_matrices"] = [
            os.path.abspath(c) for c in results
        ]
        return outputs


class DSIStudioTrackingInputSpec(DSIStudioConnectivityMatrixInputSpec):
    roi = File(exists=True, argstr="--roi=%s")
    roi2 = File(exists=True, argstr="--roi2=%s")
    roa = File(exists=True, argstr="--roa=%s")
    end = File(exists=True, argstr="--end=%s")
    end2 = File(exists=True, argstr="--end2=%s")
    ter = File(exists=True, argstr="--ter=%s")
    method = traits.Enum((0, 4), argstr="--method=%d", usedefault=True)
    thread_count = traits.Int(1, argstr="--thread_count=%d", usedefault=True)
    output_trk = traits.Str(
        name_template="%s.trk.gz",
        desc="Output file (trk.gz)",
        argstr="--output=%s",
        name_source="input_fib")


class DSIStudioTrackingOutputSpec(TraitedSpec):
    output_trk = traits.Str(
        name_template="%s.trk.gz",
        desc="Output file (trk.gz)",
        argstr="--output=%s",
        name_source="input_fib")
    output_qa = File()
    output_gfa = File()
    connectivity_matrices = traits.List()


class DSIStudioTracking(CommandLine):
    input_spec = DSIStudioTrackingInputSpec
    output_spec = DSIStudioTrackingOutputSpec
    cmd = "dsi_studio --action=trk"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        results = glob("*src*fib*trk.gz")
        if len(results) == 1:
            trk_out = os.path.abspath(results[0])
            outputs["output_trk"] = trk_out
        conmat_results = glob("*.connectivity.mat")
        outputs["connectivity_matrices"] = [
            os.path.abspath(c) for c in conmat_results
        ]
        if isdefined(self.inputs.to_export):
            if "gfa" in self.inputs.to_export:
                outputs["output_gfa"] = trk_out + ".gfa.txt"
            if "qa" in self.inputs.to_export:
                outputs["output_qa"] = trk_out + ".qa.txt"
        return outputs
