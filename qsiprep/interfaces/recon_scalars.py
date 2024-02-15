#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Classes that collect scalar images and metadata from Recon Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
import re
from pkg_resources import resource_filename as pkgr
from nipype import logging
from bids.layout import parse_file_entities
from .bids import _copy_any
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    InputMultiObject, traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface, isdefined, File
)

entity_order = [
    "model",
    "fit",
    "mdp"
]

class ReconScalarsInputSpec(BaseInterfaceInputSpec):
    source_file = File(exists=True, mandatory=True)
    workflow_name = traits.Str(mandatory=True)
    model_info = traits.Dict()
    model_name = traits.Str()


class ReconScalarsOutputSpec(TraitedSpec):
    scalar_info = traits.List()


class ReconScalars(SimpleInterface):
    input_spec = ReconScalarsInputSpec
    output_spec = ReconScalarsOutputSpec
    scalar_metadata = {}
    _ignore_traits = ("model_name", "workflow_name", "scalar_metadata",
                      "model_info", "source_file")

    def __init__(self, from_file=None, resource_monitor=None, **inputs):

        # Get self._results defined
        super().__init__(
            from_file=from_file, resource_monitor=resource_monitor, **inputs)

        # Check that the input_spec matches the scalar_metadata
        self._validate_scalar_metadata()

    def _validate_scalar_metadata(self):
        for input_key in self.inputs.editable_traits():
            if input_key in self._ignore_traits:
                continue
            if not input_key in self.scalar_metadata:
                raise Exception(f"No entry found for {input_key} in ``scalar_metadata`` in this class.")

            # Check that BIDS attributes are defined
            if "bids" not in self.scalar_metadata[input_key]:
                raise Exception(f"Missing BIDS metadata for {input_key}")

    def _run_interface(self, runtime):
        results = []
        inputs = self.inputs.get()

        # Get the BIDS info from the source file
        source_file_bids = parse_file_entities(self.inputs.source_file)
        del source_file_bids["extension"], source_file_bids["suffix"]

        file_traits = [name for name in self.inputs.editable_traits()
                       if name not in self._ignore_traits]

        for input_name in file_traits:
            if not isdefined(inputs[input_name]):
                continue
            result = self.scalar_metadata[input_name].copy()
            result["path"] = op.abspath(inputs[input_name])
            result["workflow_name"] = self.inputs.workflow_name
            result["variable_name"] = input_name
            result["source_file"] = self.inputs.source_file
            # Update the BIDS with the source file
            bids_overlap = set(source_file_bids.keys()).intersection(result['bids'].keys())
            if bids_overlap:
                raise Exception(f"BIDS fields for {input_name} conflict with source file BIDS {bids_overlap}")
            results.append(result)
        self._results["scalar_info"] = results
        return runtime


def get_output_name(base_dir, recon_scalar):

    entities = parse_file_entities(recon_scalar["source_file"])
    out_path = op.join(base_dir, "qsirecon-" + recon_scalar["workflow_name"])
    out_path = op.join(out_path, "sub-" + entities["subject"])

    if session in entities:
        out_path += "/ses-{session}".format(**entities)

    os.makedirs(out_path, exist_ok=True)

    _, source_fname, _ = split_filename(recon_scalar["source_file"])
    _, _, extension = split_filename(recon_scalar["path"])

    # It may be that the space has changed. Check if it has
    if "space" in recon_scalar["bids"]:
        source_fname = re.sub(
            "_space-[a-zA-Z0-9]+_",
            "_space-" + recon_scalar["bids"]["space"] + "_",
            source_fname)
    base_fname = op.join(out_path,source_fname)

    for entity_name in entity_order:
        if entity_name in recon_scalar["bids"]:
            base_fname += "_{entity}-{value}_".format(
                entity=entity_name,
                value=recon_scalar["bids"][entity_name])

    # Add the suffix
    suffix = recon_scalar["bids"].get("suffix", "dwimap")
    return f"{base_fname}_{suffix}.{extension}"


class _ReconScalarsDataSinkInputSpec(BaseInterfaceInputSpec):
    source_file = File()
    base_directory = File()
    resampled_files = InputMultiObject(File(exists=True))
    recon_scalars = InputMultiObject(traits.Any())


class ReconScalarsDataSink(SimpleInterface):
    input_spec = _ReconScalarsDataSinkInputSpec

    def _run_interface(self, runtime):

        for recon_scalar in self.inputs.recon_scalars:
            output_filename = get_output_name(recon_scalar)
            _copy_any(recon_scalar["path"], output_filename)

        return runtime


# Scalars produced in the TORTOISE recon workflow
tortoise_scalars = {
    "fa_file": {
        "desc": "Fractional Anisotropy from a tensor fit",
        "bids":{"mdp": "fa"}
    },
    "rd_file": {
        "desc": "Radial Diffusivity from a tensor fit",
        "bids":{"mdp": "rd"}
    },
    "ad_file": {
        "desc": "Apparent Diffusivity from a tensor fit",
        "bids":{"mdp": "ad"}
    },
    "li_file": {
        "desc": "LI from a tensor fit",
        "bids":{"mdp": "li"}
    },
    "am_file": {
        "desc": "A0 from a tensor fit",
        "bids":{"mfp": "AM"}
    },
    "pa_file": {
        "desc": "PA from MAPMRI",
        "bids":{"mdp": "PA"}
    },
    "path_file": {
        "desc": "PAth from MAPMRI",
        "bids":{"mdp": "PAth"}
    },
    "rtop_file": {
        "desc": "Return to origin probability from MAPMRI",
        "bids":{"mdp": "RTOP"}
    },
    "rtap_file": {
        "desc": "Return to axis probability from MAPMRI",
        "bids":{"mdp": "RTAP"}
    },
    "rtpp_file": {
        "desc": "Return to plane probability from MAPMRI",
        "bids":{"mdp": "RTPP"}
    },
    "ng_file": {
        "desc": "Non-Gaussianity from MAPMRI",
        "bids":{"mdp": "NG"}
    },
    "ngpar_file": {
        "desc": "Non-Gaussianity parallel from MAPMRI",
        "bids":{"mdp": "NGpar"}
    },
    "ngperp_file": {
        "desc": "Non-Gaussianity perpendicular from MAPMRI",
        "bids":{"mdp": "NGperp"}
    },
}

class _TORTOISEReconScalarInputSpec(ReconScalarsInputSpec):
    pass

for input_name in tortoise_scalars:
    _TORTOISEReconScalarInputSpec.add_class_trait(input_name, File(exists=True))

class TORTOISEReconScalars(ReconScalars):
    input_spec = _TORTOISEReconScalarInputSpec
    scalar_metadata = tortoise_scalars


# Scalars produced in the AMICO recon workflow
amico_scalars = {
    "icvf_image": {
        "desc": "Intracellular volume fraction from NODDI",
        "bids":{"mdp": "icvf"}
    },
    "isovf_image": {
        "desc": "Isotropic volume fraction from NODDI",
        "bids":{"mdp": "isovf"}
    },
    "od_image": {
        "desc": "OD from NODDI",
        "bids":{"mdp": "od"}
    }
}

class _AMICOReconScalarInputSpec(ReconScalarsInputSpec):
    pass

for input_name in amico_scalars:
    _AMICOReconScalarInputSpec.add_class_trait(input_name, File(exists=True))

class AMICOReconScalars(ReconScalars):
    input_spec = _AMICOReconScalarInputSpec
    scalar_metadata = amico_scalars


# Scalars produced by DSI Studio
dsistudio_scalars = {
    "qa_file": {
        "desc": "Fractional Anisotropy from a tensor fit",
        "bids":{"mdp": "qa"}
    },
    "dti_fa_file": {
        "desc": "Radial Diffusivity from a tensor fit",
        "bids":{"mdp": "qa", "fit": "DTI"}
    },
    "txx_file": {
        "desc": "Tensor fit txx",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "txy_file": {
        "desc": "Tensor fit txy",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "txz_file": {
        "desc": "Tensor fit txz",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "tyy_file": {
        "desc": "Tensor fit tyy",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "tyz_file": {
        "desc": "Tensor fit tyz",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "tzz_file": {
        "desc": "Tensor fit tzz",
        "bids":{"mfp": "txx", "fit": "DTI"}
    },
    "rd1_file": {
        "desc": "RD1",
        "bids":{"mdp": "rd1", "fit": "RDI"}
    },
    "rd2_file": {
        "desc": "RD2",
        "bids":{"mdp": "rd2", "fit": "RDI"}
    },
    "ha_file": {
        "desc": "HA"
    },
    "md_file": {
        "desc": "Mean Diffusivity"
    },
    "ad_file": {
        "desc": "AD"
    },
    "rd_file": {
        "desc": "Radial Diffusivity"
    },
    "gfa_file": {
        "desc": "Generalized Fractional Anisotropy",
        "bids": {"mdp": "gfa", "fit": "GQI"}
    },
    "iso_file": {
        "desc": "Isotropic Diffusion",
        "bids": {"mdp": "iso", "fit": "GQI"}
    },
    "rdi_file": {
        "desc": "RDI"
    },
    "nrdi02L_file": {
        "desc": "NRDI at 02L"
    },
    "nrdi04L_file": {
        "desc": "NRDI at 04L"
    },
    "nrdi06L_file": {
        "desc": "NRDI at 06L"
    },
}

class _DSIStudioReconScalarInputSpec(ReconScalarsInputSpec):
    pass

for input_name in dsistudio_scalars:
    _DSIStudioReconScalarInputSpec.add_class_trait(input_name, File(exists=True))

class DSIStudioReconScalars(ReconScalars):
    input_spec = _DSIStudioReconScalarInputSpec
    scalar_metadata = dsistudio_scalars


dipy_dki_scalars = {
    'dki_fa': {
        "desc": "DKI FA",
        "bids": {"mdp": "FA"}
    },
    'dki_md': {
        "desc": "DKI MD",
        "bids": {"mdp": "MD"}
    },
    'dki_rd': {
        "desc": "DKI RD",
        "bids": {"mdp": "RD"}
    },
    'dki_ad': {
        "desc": "DKI AD",
        "bids": {"mdp": "AD"}
    },
    'dki_kfa': {
        "desc": "DKI KFA",
        "bids": {"mdp": "KFA"}
    },
    'dki_mk': {
        "desc": "DKI MK",
        "bids": {"mdp": "MK"}
    },
    'dki_ak': {
        "desc": "DKI AK",
        "bids": {"mdp": "AK"}
    },
    'dki_rk': {
        "desc": "DKI RK",
        "bids": {"mdp": "RK"}
    },
    'dki_mkt': {
        "desc": "DKI MKT",
        "bids": {"mdp": "MKT"}
    }
}
class _DIPYDKIReconScalarInputSpec(ReconScalarsInputSpec):
    pass

for input_name in dipy_dki_scalars:
    _DIPYDKIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))

class DIPYDKIReconScalars(ReconScalars):
    input_spec = _DIPYDKIReconScalarInputSpec
    scalar_metadata = dipy_dki_scalars


# DIPY implementation of MAPMRI
dipy_mapmri_scalars = {
    "qiv_file": {
        "desc": "q-space inverse variance from MAPMRI"
    },
    "msd_file": {
        "desc": "mean square displacement from MAPMRI"
    },
    "lapnorm_file": {
        "desc": "Laplacian norm from regularized MAPMRI (MAPL)"
    },
    "rtop_file": {
        "desc": "Return to origin probability from MAPMRI"
    },
    "rtap_file": {
        "desc": "Return to axis probability from MAPMRI"
    },
    "rtpp_file": {
        "desc": "Return to plane probability from MAPMRI"
    },
    "ng_file": {
        "desc": "Non-Gaussianity from MAPMRI"
    },
    "ngpar_file": {
        "desc": "Non-Gaussianity parallel from MAPMRI"
    },
    "ngperp_file": {
        "desc": "Non-Gaussianity perpendicular from MAPMRI"
    },
}


class _DIPYMAPMRIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_mapmri_scalars:
    _DIPYMAPMRIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DIPYMAPMRIReconScalars(ReconScalars):
    input_spec = _DIPYMAPMRIReconScalarInputSpec
    scalar_metadata = dipy_mapmri_scalars


# Same as DIPY implementation of 3dSHORE, but with brainsuite bases
brainsuite_3dshore_scalars = dipy_mapmri_scalars.copy()
brainsuite_3dshore_scalars.update({
    "cnr_image": {
        "desc": "Contrast to noise ratio for 3dSHORE fit"
    },
    "alpha_image": {
        "desc": "alpha used when fitting in each voxel"
    },
    "r2_image": {
        "desc": "r^2 of the 3dSHORE fit"
    },
    "regularization_image": {
        "desc": "regularization of the 3dSHORE fit"
    },
})


class _BrainSuite3dSHOREReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in brainsuite_3dshore_scalars:
    _BrainSuite3dSHOREReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class BrainSuite3dSHOREReconScalars(ReconScalars):
    input_spec = _BrainSuite3dSHOREReconScalarInputSpec
    scalar_metadata = brainsuite_3dshore_scalars