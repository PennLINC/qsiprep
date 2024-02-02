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
from pkg_resources import resource_filename as pkgr
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface, isdefined, File
)


class ReconScalarsInputSpec(BaseInterfaceInputSpec):
    workflow_name = traits.Str(mandatory=True)
    source_file = traits.Str(mandatory=True)
    desc = traits.Str(
        desc="If doing multiple versions of the same model, use this as a desc-ID")


class ReconScalarsOutputSpec(TraitedSpec):
    scalar_info = traits.List()


class ReconScalars(SimpleInterface):
    input_spec = ReconScalarsInputSpec
    output_spec = ReconScalarsOutputSpec
    scalar_metadata = {}
    _ignore_traits = ("workflow_name", "scalar_metadata")
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

    def _run_interface(self, runtime):
        results = []
        inputs = self.inputs.get()

        file_traits = [name for name in self.inputs.editable_traits()
                       if name not in self._ignore_traits]
        for input_name in file_traits:
            if not isdefined(inputs[input_name]):
                continue
            result = self.scalar_metadata[input_name].copy()
            result["path"] = op.abspath(inputs[input_name])
            result["workflow_name"] = self.inputs.workflow_name
            result["variable_name"] = input_name
            if isdefined(self.inputs.desc) and self.inputs.desc:
                result["bids"]["desc"] = self.inputs.desc
            results.append(result)
        self._results["scalar_info"] = results
        return runtime


# Scalars produced in the TORTOISE recon workflow
tortoise_scalars = {
    "fa_file": {
        "desc": "Fractional Anisotropy from a tensor fit",
        "bids":{"model": "tensor", "param": "fa"}
    },
    "rd_file": {
        "desc": "Radial Diffusivity from a tensor fit",
        "bids":{"model": "tensor", "param": "rd"}
    },
    "ad_file": {
        "desc": "Apparent Diffusivity from a tensor fit",
        "bids":{"model": "tensor", "param": "ad"}
    },
    "li_file": {
        "desc": "LI from a tensor fit",
        "bids":{"model": "tensor", "param": "li"}
    },
    "am_file": {
        "desc": "A0 from a tensor fit",
        "bids":{"model": "tensor", "param": "AM", "suffix": "model"}
    },
    "pa_file": {
        "desc": "PA from MAPMRI",
        "bids":{"param": "PA", "suffix": "mdp", "model": "MAPMRI"}
    },
    "path_file": {
        "desc": "PAth from MAPMRI",
        "bids":{"param": "PAth", "suffix": "mdp", "model": "MAPMRI"}
    },
    "rtop_file": {
        "desc": "Return to origin probability from MAPMRI",
        "bids":{"param": "RTOP", "suffix": "mdp", "model": "MAPMRI"}
    },
    "rtap_file": {
        "desc": "Return to axis probability from MAPMRI",
        "bids":{"param": "RTAP", "suffix": "mdp", "model": "MAPMRI"}
    },
    "rtpp_file": {
        "desc": "Return to plane probability from MAPMRI",
        "bids":{"param": "RTPP", "suffix": "mdp", "model": "MAPMRI"}
    },
    "ng_file": {
        "desc": "Non-Gaussianity from MAPMRI",
        "bids":{"param": "NG", "suffix": "mdp", "model": "MAPMRI"}
    },
    "ngpar_file": {
        "desc": "Non-Gaussianity parallel from MAPMRI",
        "bids":{"param": "NGpar", "suffix": "mdp", "model": "MAPMRI"}
    },
    "ngperp_file": {
        "desc": "Non-Gaussianity perpendicular from MAPMRI",
        "bids":{"param": "NGperp", "suffix": "mdp", "model": "MAPMRI"}
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
        "bids":{"param": "icvf", "suffix": "mdp", "model": "NODDI"}
    },
    "isovf_image": {
        "desc": "Isotropic volume fraction from NODDI",
        "bids":{"param": "isovf", "suffix": "mdp", "model": "NODDI"}
    },
    "od_image": {
        "desc": "OD from NODDI",
        "bids":{"param": "od", "suffix": "mdp", "model": "NODDI"}
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
        "bids":{"param": "qa", "suffix": "mdp", "model": "GQI"}
    },
    "dti_fa_file": {
        "desc": "Radial Diffusivity from a tensor fit",

    },
    "txx_file": {
        "desc": "Tensor fit txx"
    },
    "txy_file": {
        "desc": "Tensor fit txy"
    },
    "txz_file": {
        "desc": "Tensor fit txz"
    },
    "tyy_file": {
        "desc": "Tensor fit tyy"
    },
    "tyz_file": {
        "desc": "Tensor fit tyz"
    },
    "tzz_file": {
        "desc": "Tensor fit tzz"
    },
    "rd1_file": {
        "desc": "RD1"
    },
    "rd2_file": {
        "desc": "RD2"
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
        "desc": "Generalized Fractional Anisotropy"
        "bids":{"param": "gfa", "suffix": "mdp", "model": "GQI"}
    },
    "iso_file": {
        "desc": "Isotropic Diffusion",
        "bids":{"param": "iso", "suffix": "mdp", "model": "GQI"}
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
        "desc": "DKI FA"
    },
    'dki_md': {
        "desc": "DKI MD"
    },
    'dki_rd': {
        "desc": "DKI RD"
    },
    'dki_ad': {
        "desc": "DKI AD"
    },
    'dki_kfa': {
        "desc": "DKI KFA"
    },
    'dki_mk': {
        "desc": "DKI MK"
    },
    'dki_ak': {
        "desc": "DKI AK"
    },
    'dki_rk': {
        "desc": "DKI RK"
    },
    'dki_mkt': {
        "desc": "DKI MKT"
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