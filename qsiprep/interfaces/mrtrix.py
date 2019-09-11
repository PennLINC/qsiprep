#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import nibabel as nb
import numpy as np
import os.path as op

from tempfile import TemporaryDirectory
from time import time

from nipype import logging
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject, isdefined
)
from nipype.interfaces.ants.registration import RegistrationInputSpec
from .gradients import concatenate_bvecs, concatenate_bvals, GradientRotation
from dipy.core.gradients import gradient_table
from dipy.reconst.mapmri import MapmriModel
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from nipype.interfaces.mrtrix3 import (Generate5tt, ComputeTDI,
    ResponseSD, MRConvert)
from nipype.interfaces.mrtrix3.utils import Generate5ttInputSpec, Generate5ttOutputSpec
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec
from nipype.interfaces.mrtrix3.preprocess import ResponseSDInputSpec

LOGGER = logging.getLogger('nipype.interface')


class MRTrixGradientTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)


class MRTrixGradientTableOutputSpec(TraitedSpec):
    gradient_file = File(exists=True)


class MRTrixGradientTable(SimpleInterface):
    input_spec = MRTrixGradientTableInputSpec
    output_spec = MRTrixGradientTableOutputSpec

    def _run_interface(self, runtime):
        gtab_fname = fname_presuffix(self.inputs.bval_file, suffix=".b", newpath=runtime.cwd,
                                     use_ext=False)
        vecs = np.loadtxt(self.inputs.bvec_file)
        vals = np.loadtxt(self.inputs.bval_file)
        gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
        np.savetxt(gtab_fname, gtab, fmt=["%.8f", "%.8f", "%.8f", "%d"])
        self._results['gradient_file'] = gtab_fname
        return runtime


class MRTrixIngressInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    suffix = traits.Str("", usedefault=True)


class MRTrixIngressOutputSpec(TraitedSpec):
    mif_file = File()


class MRTrixIngress(SimpleInterface):
    input_spec = MRTrixIngressInputSpec
    output_spec = MRTrixIngressOutputSpec

    def _run_interface(self, runtime):
        output_mif = fname_presuffix(self.inputs.dwi_file, suffix=self.inputs.suffix + ".mif",
                                     newpath=runtime.cwd, use_ext=False)
        if isdefined(self.inputs.b_file):
            convert = MRConvert(in_file=self.inputs.dwi_file,
                                grad_file=self.inputs.b_file,
                                out_file=output_mif)
        elif isdefined(self.inputs.bval_file) and isdefined(self.inputs.bvec_file):
            convert = MRConvert(in_file=self.inputs.dwi_file,
                                in_bval=self.inputs.bval_file,
                                in_bvec=self.inputs.bvec_file,
                                out_file=output_mif)
        else:
            raise Exception("No valid mrtrix gradient files or fsl bval/bvec files specified")
        convert_run = convert.run()
        self._results['mif_file'] = convert_run.outputs.out_file

        return runtime


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='input DWI image')
    mask = File(
        exists=True,
        argstr='-mask %s',
        position=1,
        desc='mask image')
    extent = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        argstr='-extent %d,%d,%d',
        desc='set the window size of the denoising filter. (default = 5,5,5)')
    noise = File(
        argstr='-noise %s',
        name_template='%s_noise',
        name_source=['in_file'],
        keep_extension=True,
        desc='the output noise map')
    out_file = File(
        name_template='%s_denoised',
        name_source=['in_file'],
        keep_extension=True,
        argstr='%s',
        position=-1,
        desc='the output denoised DWI image')

class DWIDenoiseOutputSpec(TraitedSpec):
    noise = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)

class DWIDenoise(MRTrix3Base):
    """
    Denoise DWI data and estimate the noise level based on the optimal
    threshold for PCA.

    DWI data denoising and noise map estimation by exploiting data redundancy
    in the PCA domain using the prior knowledge that the eigenspectrum of
    random covariance matrices is described by the universal Marchenko Pastur
    distribution.

    Important note: image denoising must be performed as the first step of the
    image processing pipeline. The routine will fail if interpolation or
    smoothing has been applied to the data prior to denoising.

    Note that this function does not correct for non-Gaussian noise biases.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/dwidenoise.html>

    """

    _cmd = 'dwidenoise'
    input_spec = DWIDenoiseInputSpec
    output_spec = DWIDenoiseOutputSpec


class GenerateMasked5ttInputSpec(Generate5ttInputSpec):
    algorithm = traits.Enum(
        'fsl',
        'gif',
        'freesurfer',
        argstr='%s',
        position=0,
        mandatory=True,
        desc='tissue segmentation algorithm')
    in_file = File(
        exists=True, argstr='%s', mandatory=True, position=1, desc='input image')
    out_file = File(
        argstr='%s', genfile=True, position=2, desc='output image')
    mask = File(exists=True, argstr='-mask %s')


class GenerateMasked5tt(Generate5tt):
    input_spec = GenerateMasked5ttInputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            output = self.inputs.out_file
            if not isdefined(output):
                _ , fname, ext = split_filename(self.inputs.in_file)
                output = fname + '_5tt.mif'
            return output
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self._gen_filename('out_file'))
        return outputs


class Dwi2ResponseInputSpec(ResponseSDInputSpec):
    wm_file = File(
        argstr='%s',
        position=-3,
        genfile=True,
        desc='output WM response text file')
    gm_file = File(
        argstr='%s', genfile=True, position=-2, desc='output GM response text file')
    csf_file = File(
        argstr='%s', position=-1, genfile=True, desc='output CSF response text file')
    max_sh = InputMultiObject(
        traits.Int,
        argstr='-lmax %s',
        sep=',',
        desc=('maximum harmonic degree of response function - single value for '
              'single-shell response, list for multi-shell response'))

class Dwi2Response(ResponseSD):
    input_spec = Dwi2ResponseInputSpec

    def _format_arg(self, name, spec, val):
        if self.inputs.algorithm not in ('dhollander', 'msmt_5tt'):
            if name in ('gm_file', 'csf_file'):
                return ''
        return super(Dwi2Response, self)._format_arg(name, spec, val)

    def _gen_filename(self, name):
        if name in ('gm_file', 'csf_file', 'wm_file'):
            output = getattr(self.inputs, name)
            if not isdefined(output):
                _, fname, ext = split_filename(self.inputs.in_file)
                output = fname + "_" + name.split("_")[0] + '.txt'
            return output
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_file'] = op.abspath(self._gen_filename('wm_file'))
        if self.inputs.algorithm in ('dhollander', 'msmt_5tt'):
            outputs['gm_file'] = op.abspath(self._gen_filename('gm_file'))
            outputs['csf_file'] = op.abspath(self._gen_filename('csf_file'))
        return outputs


class EstimateFODInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum(
        'csd',
        'msmt_csd',
        argstr='%s',
        position=-8,
        mandatory=True,
        desc='FOD algorithm')
    in_file = File(
        exists=True,
        argstr='%s',
        position=-7,
        mandatory=True,
        desc='input DWI image')
    wm_txt = File(
        argstr='%s', position=-6, mandatory=True, desc='WM response text file')
    wm_odf = File(
        argstr='%s',
        position=-5,
        genfile=True,
        desc='output WM ODF')
    gm_txt = File(
        argstr='%s',
        position=-4,
        requires=['csf_txt'],
        desc='GM response text file')
    gm_odf = File(
        argstr='%s',
        position=-3,
        genfile=True,
        requires=['gm_txt'],
        desc='output GM ODF')
    csf_txt = File(
        argstr='%s', position=-2, desc='CSF response text file')
    csf_odf = File(
        argstr='%s',
        position=-1,
        genfile=True,
        requires=['csf_txt'],
        desc='output CSF ODF')
    mask_file = File(exists=True, argstr='-mask %s', desc='mask image')
    shell = traits.List(
        traits.Float,
        sep=',',
        argstr='-shell %s',
        desc='specify one or more dw gradient shells')
    max_sh = InputMultiObject(
        traits.Int,
        argstr='-lmax %s',
        sep=',',
        desc='maximum harmonic degree of response function - single value for single-shell '
             'response, list for multi-shell response')
    in_dirs = File(
        exists=True,
        argstr='-directions %s',
        desc=('specify the directions over which to apply the non-negativity '
              'constraint (by default, the built-in 300 direction set is '
              'used). These should be supplied as a text file containing the '
              '[ az el ] pairs for the directions.'))


class EstimateFODOutputSpec(TraitedSpec):
    wm_odf = File(desc='output WM ODF')
    gm_odf = File(desc='output GM ODF')
    csf_odf = File(desc='output CSF ODF')


class EstimateFOD(MRTrix3Base):
    """
    Estimate fibre orientation distributions from diffusion data using spherical deconvolution
    """

    _cmd = 'dwi2fod'
    input_spec = EstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_odf'] = op.abspath(self._gen_filename('wm_odf'))
        if self.inputs.algorithm == 'msmt_csd':
            outputs['gm_odf'] = op.abspath(self._gen_filename('gm_odf'))
            outputs['csf_odf'] = op.abspath(self._gen_filename('csf_odf'))
            print(outputs)
        return outputs

    def _format_arg(self, name, spec, value):
        if self.inputs.algorithm == 'csd':
            if name in ('gm_odf', 'gm_txt', 'csf_odf', 'csf_txt'):
                return ''
        return super(EstimateFOD, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name in ('gm_odf', 'gm_txt', 'wm_odf', 'wm_txt', 'csf_odf', 'csf_txt'):
            output = getattr(self.inputs, name)
            if not isdefined(output):
                _, fname, _ = split_filename(self.inputs.in_file)
                ext = '.txt' if name.endswith('txt') else '.mif'
                output = fname + "_" + name.split("_")[0] + ext
            return output
        return None


class SIFTInputSpec(MRTrix3BaseInputSpec):
    in_tracks = File(
        argstr='%s', exists=True, mandatory=True, position=-3, desc='input tck file')
    in_fod = File(
        argstr='%s', position=-2, exists=True, mandatory=True, desc='input FOD SH file')
    out_tracks = File(
        argstr='%s', position=-1, genfile=True, desc='')


class GlobalTractographyInputSpec(MRTrix3BaseInputSpec):
    dwi_file = File(
        argstr='%s', exists=True, mandatory=True, position=-3, desc='full dwi file')
    wm_txt = File(exists=True, mandatory=True, position=-2, desc='wm response function')
    out_tracks = File(
        argstr='%s', position=-1, genfile=True, desc='the globally-optimized streamlines')
    isotropic_response_txt = InputMultiObject(
        File(argstr='-riso %s', exists=True),
        desc='set one or more isotropic response functions')
    out_isotropic_fod = InputMultiObject(
        File(
            argstr='-fiso %s', requires=['isotropic_response_txt'], genfile=True,
            desc=' Predicted isotropic fractions of the tissues for which response '
                 'functions were provided with isotropic_response_txt. Typically, '
                 'these are CSF and GM.'))
