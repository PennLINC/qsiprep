# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import os.path as op
import shutil
from glob import glob
from pathlib import Path

import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import split_filename

from .bids import get_bids_params
from .dsi_studio import btable_from_bvals_bvecs
from .images import ConformDwi, to_lps
from .mrtrix import _convert_fsl_to_mrtrix

LOGGER = logging.getLogger("nipype.interface")


class QsiReconDWIIngressInputSpec(BaseInterfaceInputSpec):
    # DWI files
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    atlas_names = traits.List()


class QsiReconDWIIngressOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    space_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()
    dir_id = traits.Str()
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    btable_file = File(exists=True)
    confounds_file = File(exists=True)
    dwi_file = File(exists=True)
    local_bvec_file = File()
    mask_file = File()
    dwi_ref = File(exists=True)
    qc_file = File(exists=True)
    slice_qc_file = File(exists=True)


class QsiReconDWIIngress(SimpleInterface):
    input_spec = QsiReconDWIIngressInputSpec
    output_spec = QsiReconDWIIngressOutputSpec

    def _run_interface(self, runtime):
        params = get_bids_params(self.inputs.dwi_file)
        self._results = {key: val for key, val in list(params.items()) if val is not None}
        space = self._results.get("space_id")
        if space is None:
            raise Exception("Unable to detect space of %s" % self.inputs.dwi_file)

        # Find the additional files
        out_root, fname, _ = split_filename(self.inputs.dwi_file)
        self._results["bval_file"] = op.join(out_root, fname + ".bval")
        self._results["bvec_file"] = op.join(out_root, fname + ".bvec")
        self._get_if_exists("confounds_file", op.join(out_root, "*confounds.tsv"))
        self._get_if_exists("local_bvec_file", op.join(out_root, fname[:-3] + "bvec.nii*"))
        self._get_if_exists("b_file", op.join(out_root, fname + ".b"))
        self._get_if_exists("mask_file", op.join(out_root, fname[:-11] + "brain_mask.nii*"))
        self._get_if_exists("dwi_ref", op.join(out_root, fname[:-16] + "dwiref.nii*"))
        self._results["dwi_file"] = self.inputs.dwi_file

        # Image QC doesn't include space
        self._get_if_exists("qc_file", self._get_qc_filename(out_root, params, "ImageQC", "csv"))
        self._get_if_exists(
            "slice_qc_file", self._get_qc_filename(out_root, params, "SliceQC", "json")
        )

        # Get the anatomical data
        path_parts = out_root.split(op.sep)[:-1]  # remove "dwi"
        # Anat is above ses
        if path_parts[-1].startswith("ses"):
            path_parts.pop()
        return runtime

    def _get_if_exists(self, name, pattern, multi_ok=False):
        files = glob(pattern)
        if len(files) == 1:
            self._results[name] = files[0]
        if len(files) > 1 and multi_ok:
            self._results[name] = files[0]

    def _get_qc_filename(self, out_root, params, desc, suffix):
        used_keys = ["subject_id", "session_id", "acq_id", "dir_id", "run_id"]
        fname = "_".join([params[key] for key in used_keys if params[key]])
        return out_root + "/" + fname + "_desc-%s_dwi.%s" % (desc, suffix)


class _UKBioBankDWIIngressInputSpec(QsiReconDWIIngressInputSpec):
    dwi_file = File(exists=False, help="The name of what a BIDS dwi file may have been")
    data_dir = traits.Directory(
        exists=True, help="The UKB data directory for a subject. Must contain DTI/ and T1/"
    )


class UKBioBankDWIIngress(SimpleInterface):
    input_spec = _UKBioBankDWIIngressInputSpec
    output_spec = QsiReconDWIIngressOutputSpec

    def _run_interface(self, runtime):
        runpath = Path(runtime.cwd)

        # The UKB input files
        in_dir = Path(self.inputs.data_dir)
        dwi_dir = in_dir / "DTI" / "dMRI" / "dMRI"
        ukb_bval_file = dwi_dir / "bvals"
        ukb_bvec_file = dwi_dir / "bvecs"  # These are the same as eddy rotated
        ukb_dwi_file = dwi_dir / "data_ud.nii.gz"
        ukb_dwiref_file = dwi_dir / "dti_FA.nii.gz"

        # The bids_name is what the images will be renamed to
        bids_name = Path(self.inputs.dwi_file).name.replace(".nii.gz", "")
        dwi_file = str(runpath / (bids_name + ".nii.gz"))
        bval_file = str(runpath / (bids_name + ".bval"))
        bvec_file = str(runpath / (bids_name + ".bvec"))
        b_file = str(runpath / (bids_name + ".b"))
        btable_file = str(runpath / (bids_name + "btable.txt"))
        dwiref_file = str(runpath / (bids_name.replace("_dwi", "_dwiref") + ".nii.gz"))

        dwi_conform = ConformDwi(
            dwi_file=str(ukb_dwi_file), bval_file=str(ukb_bval_file), bvec_file=str(ukb_bvec_file)
        )

        result = dwi_conform.run()
        Path(result.outputs.dwi_file).rename(dwi_file)
        Path(result.outputs.bvec_file).rename(bvec_file)
        shutil.copyfile(result.outputs.bval_file, bval_file)
        # Reorient the dwi file to LPS+
        self._results["dwi_file"] = dwi_file
        self._results["bvec_file"] = bvec_file
        self._results["bval_file"] = bval_file

        # Create a btable_txt file for DSI Studio
        btable_from_bvals_bvecs(bval_file, bvec_file, btable_file)
        self._results["btable_file"] = btable_file

        # Create a mrtrix .b file
        _convert_fsl_to_mrtrix(bval_file, bvec_file, b_file)
        self._results["b_file"] = b_file

        # Create a dwi ref file
        to_lps(nb.load(ukb_dwiref_file)).to_filename(dwiref_file)
        self._results["dwi_ref"] = dwiref_file
        return runtime
