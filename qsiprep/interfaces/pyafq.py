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
from AFQ.definitions.mask import MaskFile
from AFQ.definitions.mapping import ItkMap
from AFQ.utils.bin import parse_qsiprep_params_dict

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface
)


LOGGER = logging.getLogger('nipype.interface')


class PyAFQInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    itk_file = File(exists=True, mandatory=True)
    kwargs = traits.Dict(exists=True, mandatory=True)
    trk_file = traits.Either(None, File(exists=True))


class PyAFQOutputSpec(TraitedSpec):
    out_directory = File()


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):

        # shim the expected inputs
        shim_dir = op.join(runtime.cwd, "study/subject")
        os.makedirs(shim_dir)
        bval_file = fname_presuffix(self.inputs.bval_file,
                                    newpath=shim_dir)
        bvec_file = fname_presuffix(self.inputs.bvec_file,
                                    newpath=shim_dir)
        dwi_file = fname_presuffix(self.inputs.dwi_file,
                                   newpath=shim_dir)
        mask_file = fname_presuffix(self.inputs.mask_file,
                                    newpath=shim_dir)
        trk_file = fname_presuffix(self.inputs.trk_file,
                                   newpath=shim_dir)
        itk_file = fname_presuffix(self.inputs.itk_file,
                                   newpath=shim_dir)
        os.symlink(self.inputs.bval_file, bval_file)
        os.symlink(self.inputs.bvec_file, bvec_file)
        os.symlink(self.inputs.dwi_file, dwi_file)
        os.symlink(self.inputs.mask_file, mask_file)

        brain_mask_definition = MaskFile(path=mask_file)
        itk_map = ItkMap(warp_path=itk_file)
        output_dir = shim_dir + "/PYAFQ/"

        kwargs = parse_qsiprep_params_dict(self.inputs)

        myafq = ParticipantAFQ(
            dwi_file, bval_file, bvec_file, output_dir,
            import_tract=trk_file,
            brain_mask_definition=brain_mask_definition,
            mapping_definition=itk_map,
            **kwargs)

        self._results['profiles_csv'] = myafq.profiles

        return runtime
