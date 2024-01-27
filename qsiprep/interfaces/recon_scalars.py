#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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


class ReconScalarsOutputSpec(TraitedSpec):
    scalar_info = traits.List()


class ReconScalars(SimpleInterface):
    input_spec = ReconScalarsInputSpec
    output_spec = ReconScalarsOutputSpec
    scalar_metadata = {}
    _ignore_traits = ("workflow_name")

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
            result["variable_name"] = self.inputs.workflow_name + "_" + input_name
            results.append(result)
        self._results["scalar_info"] = results
        return runtime


# Scalars produced in the TORTOISE recon workflow
tortoise_scalars = {
    "fa_file": {
        "desc": "Fractional Anisotropy from a tensor fit"
    },
    "rd_file": {
        "desc": "Radial Diffusivity from a tensor fit"
    },
    "ad_file": {
        "desc": "Apparent Diffusivity from a tensor fit"
    },
    "li_file": {
        "desc": "LI from a tensor fit"
    },
    "am_file": {
        "desc": "A0 from a tensor fit"
    },
    "pa_file": {
        "desc": "PA from MAPMRI"
    },
    "path_file": {
        "desc": "PAth from MAPMRI"
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
        "desc": "Intracellular volume fraction from NODDI"
    },
    "isovf_image": {
        "desc": "Isotropic volume fraction from NODDI"
    },
    "od_image": {
        "desc": "OD from NODDI"
    }
}

class _AMICOReconScalarInputSpec(ReconScalarsInputSpec):
    pass

for input_name in tortoise_scalars:
    _AMICOReconScalarInputSpec.add_class_trait(input_name, File(exists=True))

class AMICOReconScalars(ReconScalars):
    input_spec = _AMICOReconScalarInputSpec
    scalar_metadata = amico_scalars
