#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Workflows for AMICO
~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
from pkg_resources import resource_filename as pkgr
import nibabel as nb
import numpy as np
import amico
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, isdefined
)

from .converters import get_dsi_studio_ODF_geometry, amplitudes_to_fibgz, amplitudes_to_sh_mif
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from ..interfaces.mrtrix import _convert_fsl_to_mrtrix

LOGGER = logging.getLogger('nipype.interface')
TAU_DEFAULT = 1. / (4 * np.pi**2)


class AmicoInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    big_delta = traits.Either(None, traits.Float(), usedefault=True)
    little_delta = traits.Either(None, traits.Float(), usedefault=True)
    b0_threshold = traits.CFloat(50, usedefault=True)
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)


class AmicoOutputSpec(TraitedSpec):
    fibgz = File()
    fod_sh_mif = File()
    extrapolated_dwi = File()
    extrapolated_bvals = File()
    extrapolated_bvecs = File()
    extrapolated_b = File()
    odf_amplitudes = File()
    odf_directions = File()


class AmicoReconInterface(SimpleInterface):
    input_spec = AmicoInputSpec
    output_spec = AmicoOutputSpec

    def _get_scheme(self, external_bvals=None, external_bvecs=None):
        little_delta = self.inputs.little_delta if isdefined(self.inputs.little_delta) else None
        big_delta = self.inputs.big_delta if isdefined(self.inputs.big_delta) else None
        bval_file = self.inputs.bval_file if external_bvals is None else external_bvals
        bvec_file = self.inputs.bvec_file if external_bvecs is None else external_bvecs
        gtab = gradient_table(bvals=np.loadtxt(bval_file),
                              bvecs=np.loadtxt(bvec_file).T,
                              b0_threshold=self.inputs.b0_threshold,
                              big_delta=big_delta,
                              small_delta=little_delta)
        return gtab

    def _save_scalar(self, data, suffix, runtime, ref_img):
        output_fname = fname_presuffix(self.inputs.dwi_file, suffix=suffix,
                                       newpath=runtime.cwd)
        nb.Nifti1Image(data, ref_img.affine, ref_img.header).to_filename(output_fname)
        return output_fname

    def _write_external_formats(self, runtime, fit_obj, mask_img, suffix):

        if not (self.inputs.write_fibgz or self.inputs.write_mif):
            return

        # Convert to amplitudes for other software
        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        hs = HemiSphere(x=x, y=y, z=z)
        odf_amplitudes = nb.Nifti1Image(fit_obj.odf(hs), mask_img.affine, mask_img.header)
        output_amps_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+"_amp.nii.gz",
                                           newpath=runtime.cwd, use_ext=False)
        output_dirs_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+"_dirs.npy",
                                           newpath=runtime.cwd, use_ext=False)
        odf_amplitudes.to_filename(output_amps_file)
        np.save(output_dirs_file, verts[:hemisphere])
        self._results['odf_amplitudes'] = output_amps_file
        self._results['odf_directions'] = output_dirs_file

        if self.inputs.write_fibgz:
            output_fib_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+".fib",
                                              newpath=runtime.cwd, use_ext=False)
            LOGGER.info("Writing DSI Studio fib file %s", output_fib_file)
            amplitudes_to_fibgz(odf_amplitudes, verts, faces, output_fib_file, mask_img,
                                num_fibers=5)
            self._results['fibgz'] = output_fib_file

        if self.inputs.write_mif:
            output_mif_file = fname_presuffix(self.inputs.dwi_file, suffix=suffix+".mif",
                                              newpath=runtime.cwd, use_ext=False)
            LOGGER.info("Writing sh mif file %s", output_mif_file)
            amplitudes_to_sh_mif(odf_amplitudes, verts, output_mif_file, runtime.cwd)
            self._results['fod_sh_mif'] = output_mif_file


class NODDIInputSpec(AmicoInputSpec):
    dPar = traits.Float(mandatory=True)
    dIso =  traits.Float(mandatory=True)
    isExvivo = traits.Bool(False, usedefault=True)


class NODDIOutputSpec(AmicoOutputSpec):
    directions_image = File()
    icvf_image = File()
    od_image = File()
    isovf_image = File()
    config_file = File()


class NODDI(AmicoReconInterface):
    input_spec = NODDIInputSpec
    output_spec = NODDIOutputSpec

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
        os.symlink(self.inputs.bval_file, bval_file)
        os.symlink(self.inputs.bvec_file, bvec_file)
        os.symlink(self.inputs.dwi_file, dwi_file)
        os.symlink(self.inputs.mask_file, mask_file)

        # Prevent a ton of deprecation warnings
        os.environ['KMP_WARNINGS'] = '0'
        # Set up the AMICO evaluation
        aeval = amico.Evaluation("study", "subject")
        scheme_file = amico.util.fsl2scheme(bval_file, bvec_file,
                                            flipAxes=[False, True, True])
        aeval.load_data(dwi_filename=dwi_file, scheme_filename=scheme_file,
                        mask_filename=mask_file, b0_thr=self.inputs.b0_threshold)
        LOGGER.info("Fitting NODDI Model.")
        aeval.set_model("NODDI")
        # set the parameters
        aeval.model.dPar = self.inputs.dPar
        aeval.model.dIso = self.inputs.dIso
        aeval.model.isExvivo = self.inputs.isExvivo
        aeval.generate_kernels()
        aeval.load_kernels()
        aeval.fit()

        # Write the results
        aeval.save_results()
        self._results['directions_image'] = shim_dir + "/AMICO/NODDI/FIT_dir.nii.gz"
        self._results['icvf_image'] = shim_dir + "/AMICO/NODDI/FIT_ICVF.nii.gz"
        self._results['od_image'] = shim_dir + "/AMICO/NODDI/FIT_OD.nii.gz"
        self._results['isovf_image'] = shim_dir + "/AMICO/NODDI/FIT_ISOVF.nii.gz"
        self._results['config_file'] = shim_dir + "/AMICO/NODDI/config.pickle"

        return runtime
