#!python
import logging
import os
import os.path as op
from glob import glob
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix, split_filename

from .. import config

LOGGER = logging.getLogger("nipype.interface")
DSI_STUDIO_VERSION = "94b9c79"


class DSIStudioCommandLineInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(1, usedefault=True, argstr="--thread_count=%d", nohash=True)


class DSIStudioCreateSrcInputSpec(DSIStudioCommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(desc="DWI Nifti file", argstr="--source=%s")
    input_dicom_dir = File(
        desc="Directory with DICOM data from only the dwi", exists=True, argstr="--source=%s"
    )
    bvec_convention = traits.Enum(
        ("DIPY", "FSL"),
        usedefault=True,
        desc="Convention used for bvecs. FSL assumes LAS+ no matter image orientation",
    )
    input_bvals_file = File(desc="Text file containing b values", exists=True, argstr="--bval=%s")
    input_bvecs_file = File(
        desc="Text file containing b vectors (FSL format)", exists=True, argstr="--bvec=%s"
    )
    input_b_table_file = File(
        desc="Text file containing q-space sampling (DSI Studio format)",
        exists=True,
        argstr="--b_table=%s",
    )
    recursive = traits.Bool(
        False, desc="Search for DICOM files recursively", argstr="--recursive=1"
    )
    subject_id = traits.Str("data")
    output_src = File(desc="Output file (.src.gz)", argstr="--output=%s", genfile=True)
    grad_dev = File(
        desc="Gradient deviation file", exists=True, copyfile=True, position=-1, argstr="#%s"
    )


class DSIStudioCreateSrcOutputSpec(TraitedSpec):
    output_src = File(desc="Output file (.src.gz)", name_source="subject_id")


class DSIStudioCreateSrc(CommandLine):
    input_spec = DSIStudioCreateSrcInputSpec
    output_spec = DSIStudioCreateSrcOutputSpec
    _cmd = "dsi_studio --action=src "

    def _pre_run_hook(self, runtime):
        """As of QSIPrep > 0.17 DSI Studio changed from DIPY bvecs to FSL bvecs."""

        # b_table files and dicom directories are ok
        if isdefined(self.inputs.input_b_table_file) or isdefined(self.inputs.input_dicom_dir):
            return runtime

        if not (
            isdefined(self.inputs.input_bvals_file) and isdefined(self.inputs.input_bvecs_file)
        ):
            raise Exception(
                "without a b_table or dicom directory, both bvals and bvecs must be specified"
            )

        # If the bvecs are in DIPY format, convert them to a b_table.txt
        if self.inputs.bvec_convention == "DIPY":
            btable_file = self._gen_filename("output_src").replace(".src.gz", ".b_table.txt")
            btable_from_bvals_bvecs(
                self.inputs.input_bvals_file, self.inputs.input_bvecs_file, btable_file
            )
            self.inputs.input_b_table_file = btable_file
            self.inputs.input_bvals_file = traits.Undefined
            self.inputs.input_bvecs_file = traits.Undefined
            LOGGER.info("Converted DIPY LPS+ bval/bvec to DSI Studio b_table")
        return runtime

    def _gen_filename(self, name):
        if not name == "output_src":
            return None
        if isdefined(self.inputs.input_nifti_file):
            _, fname, ext = split_filename(self.inputs.input_nifti_file)
        elif isdefined(self.inputs.input_dicom_dir):
            fname = op.split(self.inputs.dicom_dir)[1]
        else:
            raise Exception("Need either an input dicom director or nifti")

        output = op.abspath(fname) + ".src.gz"
        return output

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_src"] = self._gen_filename("output_src")
        return outputs


class _DSIStudioQCOutputSpec(TraitedSpec):
    qc_txt = File(exists=True, desc="Text file with QC measures")


class DSIStudioQC(SimpleInterface):
    output_spec = _DSIStudioQCOutputSpec

    def _run_interface(self, runtime):
        # DSI Studio (0.12.2) action=qc has two modes, depending on wether the
        # input is a file (src.gz|nii.gz)|(fib.gz) or a directory. For
        # directories, the action will be run on a number of detected files
        # (which *cannot* be symbolic links for some reason).
        src_file = fname_presuffix(self.inputs.src_file, newpath=runtime.cwd)
        cmd = ["dsi_studio", "--action=qc", "--source=" + src_file]
        proc = Popen(cmd, cwd=runtime.cwd, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        if out:
            LOGGER.info(out.decode())
        if err:
            LOGGER.critical(err.decode())
        self._results["qc_txt"] = op.join(runtime.cwd, "qc.txt")
        return runtime


class _DSIStudioSrcQCInputSpec(DSIStudioCommandLineInputSpec):
    src_file = File(exists=True, copyfile=False, argstr="%s", desc="DSI Studio src[.gz] file")


class DSIStudioSrcQC(DSIStudioQC):
    input_spec = _DSIStudioSrcQCInputSpec
    ext = ".src.gz"


class _DSIStudioFibQCInputSpec(DSIStudioCommandLineInputSpec):
    src_file = File(exists=True, copyfile=False, argstr="%s", desc="DSI Studio fib[.gz] file")


class DSIStudioFibQC(DSIStudioQC):
    input_spec = _DSIStudioFibQCInputSpec
    ext = ".fib.gz"


# Step 2 reonstruct ODFs: This is needed still in preprocessing because it's used for QC
class DSIStudioReconstructionInputSpec(DSIStudioCommandLineInputSpec):
    input_src_file = File(
        desc="DSI Studio src file",
        mandatory=True,
        exists=True,
        copyfile=False,
        argstr="--source=%s",
    )
    mask = File(
        desc="Volume to mask brain voxels", exists=True, copyfile=False, argstr="--mask=%s"
    )
    grad_dev = File(
        desc="Gradient deviation file", exists=True, copyfile=True, position=-1, argstr="#%s"
    )
    thread_count = traits.Int(1, usedefault=True, argstr="--thread_count=%d", nohash=True)

    dti_no_high_b = traits.Bool(
        True,
        usedefault=True,
        argstr="--dti_no_high_b=%d",
        desc="specify whether the construction of DTI should ignore high b-value (b>1500)",
    )
    r2_weighted = traits.Bool(
        False,
        usedefault=True,
        argstr="--r2_weighted=%d",
        desc="specify whether GQI and QSDR uses r2-weighted to calculate SDF",
    )

    # Outputs
    output_odf = traits.Bool(
        True, usedefault=True, desc="Include full ODF's in output", argstr="--record_odf=1"
    )
    odf_order = traits.Enum(
        (8, 4, 5, 6, 10, 12, 16, 20), usedefault=True, desc="ODF tesselation order"
    )
    # Which scalars to include
    other_output = traits.Str(
        "all",
        argstr="--other_output=%s",
        desc="additional diffusion metrics to calculate",
        usedefault=True,
    )
    align_acpc = traits.Bool(
        False, usedefault=True, argstr="--align_acpc=%d", desc="rotate image volume to align ap-pc"
    )
    check_btable = traits.Enum(
        (0, 1),
        usedefault=True,
        argstr="--check_btable=%d",
        desc="Check if btable matches nifti orientation (not foolproof)",
    )

    num_fibers = traits.Int(
        3,
        usedefault=True,
        argstr="--num_fiber=%d",
        desc="number of fiber populations estimated at each voxel",
    )


class DSIStudioReconstructionOutputSpec(TraitedSpec):
    output_fib = File(desc="Output File", exists=True)


class DSIStudioGQIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    ratio_of_mean_diffusion_distance = traits.Float(1.25, usedefault=True, argstr="--param0=%.4f")


class DSIStudioGQIReconstruction(CommandLine):
    input_spec = DSIStudioGQIReconstructionInputSpec
    output_spec = DSIStudioReconstructionOutputSpec
    _cmd = "dsi_studio --action=rec --method=4"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        config.loggers.interface.info("current dir", os.getcwd())
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        config.loggers.interface.info(f"input src {self.inputs.input_src_file}")
        config.loggers.interface.info(f"split src name {srcname}")
        target = os.path.join(os.getcwd(), srcname) + "*gqi*.fib.gz"
        config.loggers.interface.info(f"search target: {target}")
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class _DSIStudioQCMergeInputSpec(BaseInterfaceInputSpec):
    src_qc = File(exists=True, mandatory=True)
    fib_qc = File(exists=True, mandatory=True)


class _DSIStudioQCMergeOutputSpec(TraitedSpec):
    qc_file = File(exists=True)


class DSIStudioMergeQC(SimpleInterface):
    input_spec = _DSIStudioQCMergeInputSpec
    output_spec = _DSIStudioQCMergeOutputSpec

    def _run_interface(self, runtime):
        output_csv = runtime.cwd + "/merged_qc.csv"
        src_qc = load_src_qc_file(self.inputs.src_qc)
        fib_qc = load_fib_qc_file(self.inputs.fib_qc)
        src_qc.update(fib_qc)
        qc_df = pd.DataFrame(src_qc)
        qc_df.to_csv(output_csv, index=False)
        self._results["qc_file"] = output_csv
        return runtime


class _DSIStudioBTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    bvec_convention = traits.Enum(
        ("DIPY", "FSL"),
        usedefault=True,
        desc="Convention used for bvecs. FSL assumes LAS+ no matter image orientation",
    )


class _DSIStudioBTableOutputSpec(TraitedSpec):
    btable_file = File(exists=True)


class DSIStudioBTable(SimpleInterface):
    input_spec = _DSIStudioBTableInputSpec
    output_spec = _DSIStudioBTableOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.bvec_convention != "DIPY":
            raise NotImplementedError("Only DIPY Bvecs supported for now")
        btab_file = op.join(runtime.cwd, "btable.txt")
        btable_from_bvals_bvecs(self.inputs.bval_file, self.inputs.bvec_file, btab_file)
        self._results["btable_file"] = btab_file
        return runtime


def load_src_qc_file(fname, prefix=""):
    with open(fname, "r") as qc_file:
        qc_data = qc_file.readlines()
    data = qc_data[1]
    parts = data.strip().split("\t")
    dwi_contrast = np.nan
    ndc_masked = np.nan
    if len(parts) == 7:
        _, dims, voxel_size, dirs, max_b, ndc, bad_slices = parts
    elif len(parts) == 8:
        _, dims, voxel_size, dirs, max_b, _, ndc, bad_slices = parts
    elif len(parts) == 9:
        _, dims, voxel_size, dirs, max_b, dwi_contrast, ndc, ndc_masked, bad_slices = parts
    else:
        raise Exception("Unknown QC File format")

    voxelsx, voxelsy, voxelsz = map(float, voxel_size.strip().split())
    dimx, dimy, dimz = map(float, dims.strip().split())
    n_dirs = float(dirs.split("/")[1])
    max_b = float(max_b)
    dwi_corr = float(ndc)
    n_bad_slices = float(bad_slices)
    ndc_masked = float(ndc_masked)
    dwi_contrast = float(dwi_contrast)
    data = {
        prefix + "dimension_x": [dimx],
        prefix + "dimension_y": [dimy],
        prefix + "dimension_z": [dimz],
        prefix + "voxel_size_x": [voxelsx],
        prefix + "voxel_size_y": [voxelsy],
        prefix + "voxel_size_z": [voxelsz],
        prefix + "max_b": [max_b],
        prefix + "neighbor_corr": [dwi_corr],
        prefix + "masked_neighbor_corr": [ndc_masked],
        prefix + "dwi_contrast": [dwi_contrast],
        prefix + "num_bad_slices": [n_bad_slices],
        prefix + "num_directions": [n_dirs],
    }
    return data


def load_fib_qc_file(fname):
    with open(fname, "r") as fibqc_f:
        lines = [line.strip().split() for line in fibqc_f]
    return {"coherence_index": [float(lines[1][-1])]}


def btable_from_bvals_bvecs(bval_file, bvec_file, output_file):
    """Create a b-table from DIPY-style bvals/bvecs.

    Assuming these come from qsiprep they will be in LPS+, which
    is the same convention as DSI Studio's btable.
    """
    bvals = np.loadtxt(bval_file).squeeze()
    bvecs = np.loadtxt(bvec_file).squeeze()
    if 3 not in bvecs.shape:
        raise Exception("uninterpretable bval/bvec files\n\t{}\n\t{}".format(bval_file, bvec_file))
    if not bvecs.shape[1] == 3:
        bvecs = bvecs.T

    if not bvecs.shape[0] == bvals.shape[0]:
        raise Exception("Bval/Bvec mismatch")

    rows = []
    for row in map(tuple, np.column_stack([bvals, bvecs])):
        rows.append("%d %.6f %.6f %.6f" % row)

    # Write the actual file:
    with open(output_file, "w") as btablef:
        btablef.write("\n".join(rows) + "\n")
