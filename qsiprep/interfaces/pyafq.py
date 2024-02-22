#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interface for running a PyAFQ workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op

from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile

# from AFQ.definitions.mapping import ItkMap
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.interface")


class PyAFQInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    itk_file = File(exists=True, mandatory=True)
    kwargs = traits.Dict(exists=True, mandatory=True)
    tck_file = traits.Either(None, File(exists=True))


class PyAFQOutputSpec(TraitedSpec):
    afq_dir = traits.Directory()


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):

        # shim the expected inputs
        shim_dir = op.join(runtime.cwd, "study/subject")
        os.makedirs(shim_dir)
        bval_file = fname_presuffix(self.inputs.bval_file, newpath=shim_dir)
        bvec_file = fname_presuffix(self.inputs.bvec_file, newpath=shim_dir)
        dwi_file = fname_presuffix(self.inputs.dwi_file, newpath=shim_dir)
        mask_file = fname_presuffix(self.inputs.mask_file, newpath=shim_dir)
        itk_file = fname_presuffix(self.inputs.itk_file, newpath=shim_dir)
        os.symlink(self.inputs.bval_file, bval_file)
        os.symlink(self.inputs.bvec_file, bvec_file)
        os.symlink(self.inputs.dwi_file, dwi_file)
        os.symlink(self.inputs.mask_file, mask_file)
        os.symlink(self.inputs.itk_file, itk_file)

        kwargs = self.inputs.kwargs

        if self.inputs.tck_file and isdefined(self.inputs.tck_file):
            tck_file = fname_presuffix(self.inputs.tck_file, newpath=shim_dir)
            os.symlink(self.inputs.tck_file, tck_file)
        else:
            tck_file = None
        brain_mask_definition = ImageFile(path=mask_file)
        # itk_map = ItkMap(warp_path=itk_file)

        if tck_file is None:
            tck_file = kwargs["import_tract"]
        kwargs.pop("import_tract", None)
        if brain_mask_definition is None:
            brain_mask_definition = kwargs["brain_mask_definition"]
        kwargs.pop("brain_mask_definition", None)
        # if itk_map is None:  # Use pyAFQ internal mapping
        #     itk_map = kwargs['mapping_definition']
        # kwargs.pop('mapping_definition', None)

        if "parallel_segmentation" in kwargs:
            if (
                "n_jobs" not in kwargs["parallel_segmentation"]
                or kwargs["parallel_segmentation"]["n_jobs"] == -1
            ):
                kwargs["parallel_segmentation"]["n_jobs"] = self.inputs.kwargs["omp_nthreads"]
        else:
            kwargs["parallel_segmentation"] = {}
            kwargs["parallel_segmentation"]["n_jobs"] = self.inputs.kwargs["omp_nthreads"]

        output_dir = shim_dir + "/PYAFQ/"
        os.makedirs(output_dir, exist_ok=True)
        myafq = ParticipantAFQ(
            dwi_file,
            bval_file,
            bvec_file,
            output_dir,
            import_tract=tck_file,
            brain_mask_definition=brain_mask_definition,
            # mapping_definition=itk_map,
            **kwargs,
        )

        if "export" not in kwargs or kwargs["export"] == "all":
            myafq.export_all()
        else:
            myafq.export(kwargs["export"])

        self._results["afq_dir"] = output_dir

        return runtime
