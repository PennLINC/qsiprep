#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import nibabel as nb
import numpy as np
import os.path as op
from scipy.io.matlab import savemat
from tempfile import TemporaryDirectory
from time import time
from copy import deepcopy

from nipype import logging
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject, isdefined, CommandLineInputSpec
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
from nipype.interfaces.mrtrix3.tracking import TractographyInputSpec, Tractography

LOGGER = logging.getLogger('nipype.interface')


class TckGenInputSpec(TractographyInputSpec):
    power = traits.CFloat(argstr='-power %f')
    select = traits.CInt(argstr='-select %d')
    select = traits.CInt(
        argstr='-select %d',
        desc=('set the desired number of tracks. The program will continue'
              ' to generate tracks until this number of tracks have been '
              'selected and written to the output file'))
    n_tracks = traits.Int(
        desc='NOT supported, do not use')


class TckGen(Tractography):
    input_spec = TckGenInputSpec


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
        _convert_fsl_to_mrtrix(self.inputs.bval_file, self.inputs.bvec_file, gtab_fname)
        self._results['gradient_file'] = gtab_fname
        return runtime


def _convert_fsl_to_mrtrix(bval_file, bvec_file, output_fname):
    vecs = np.loadtxt(bvec_file)
    vals = np.loadtxt(bval_file)
    gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
    np.savetxt(output_fname, gtab, fmt=["%.8f", "%.8f", "%.8f", "%d"])


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


class SIFT2InputSpec(MRTrix3BaseInputSpec):
    in_tracks = File(
        argstr='%s', exists=True, mandatory=True, position=-3, desc='input tck file')
    in_fod = File(
        argstr='%s', position=-2, exists=True, mandatory=True, desc='input FOD SH file')
    out_weights = File(
        argstr='%s', position=-1, genfile=True, desc='output text file containing the weighting'
        'factor for each streamline')
    act_file = File(
        exists=True,
        argstr='-act %s',
        desc=('use the Anatomically-Constrained Tractography framework during'
              ' tracking; provided image must be in the 5TT '
              '(five - tissue - type) format'))
    fd_scale_gm = traits.Bool(
        requires=['act_file'], argstr='-fd_scale_gm', desc='provide this option '
        '(in conjunction with -act) to heuristically downsize the fibre density estimates '
        'based on the presence of GM in the voxel. This can assist in reducing tissue interface '
        'effects when using a single-tissue deconvolution algorithm')
    no_dilate_lut = traits.Bool(
        argstr='-no_dilate_lut', desc='do NOT dilate FOD lobe lookup tables; only map '
        'streamlines to FOD lobes if the precise tangent lies within the angular spread of '
        'that lobe')
    make_null_lobes = traits.Bool(
        argstr='-make_null_lobes', desc='add an additional FOD lobe to each voxel, with zero '
        'integral, that covers all directions with zero / negative FOD amplitudes')
    remove_untracked = traits.Bool(
        argstr='-remove_untracked', desc='remove FOD lobes that do not have any streamline '
        'density attributed to them; this improves filtering slightly, at the expense of longer '
        'computation time (and you can no longer do quantitative comparisons between '
        'reconstructions if this is enabled)')
    fd_thresh = traits.Float(
        argstr='-fd_thresh %f', desc='fibre density threshold; exclude an FOD lobe from '
        'filtering processing if its integral is less than this amount (streamlines will still '
        'be mapped to it, but it will not contribute to the cost function or the filtering)')
    out_mu = traits.File(
        argstr='-out_mu %s', genfile=True, desc='output the final value of SIFT proportionality '
        'coefficient mu to a text file')


class SIFT2OutputSpec(TraitedSpec):
    out_mu = File(exists=True)
    out_weights = File(exists=True)


class SIFT2(MRTrix3Base):
    input_spec = SIFT2InputSpec
    output_spec = SIFT2OutputSpec
    _cmd = 'tcksift2'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_mu'] = op.abspath(self._gen_filename('out_mu'))
        outputs['out_weights'] = op.abspath(self._gen_filename('out_weights'))
        return outputs

    def _gen_filename(self, name):
        _, fname, _ = split_filename(self.inputs.in_fod)
        output = getattr(self.inputs, name)
        if name == 'out_mu':
            if not isdefined(output):
                output = fname + "_mu.txt"
            return output
        if name == 'out_weights':
            if not isdefined(output):
                output = fname + "_weights.csv"
            return output
        return None


class GlobalTractographyInputSpec(MRTrix3BaseInputSpec):
    dwi_file = File(
        argstr='%s', exists=True, mandatory=True, position=-3, desc='full dwi file (source)')
    wm_txt = File(exists=True, argstr='%s', mandatory=True, position=-2,
                  desc='wm response function (response)')
    mask = File(exists=True, argstr='-mask %s',
                desc='only reconstruct the tractogram within the specified brain mask image.')
    out_tracks = File(
        argstr='%s', position=-1, genfile=True,
        desc='the globally-optimized streamlines (tracks)')
    gm_txt = File(argstr='-riso %s', exists=True,
                  desc='gm isotropic response functions')
    csf_txt = File(argstr='-riso %s', exists=True,
                   desc='csf isotropic response functions')
    out_fod = File(
                argstr='-fod %s', genfile=True,
                desc='Predicted fibre orientation distribution function (fODF).This fODF is '
                'estimated as part of the global track optimization, and therefore incorporates '
                'the spatial regularization that it imposes. Internally, the fODF is '
                'represented as a discrete sum of apodized point spread functions (aPSF) '
                'oriented along the directions of all particles in the voxel, used to predict '
                'the DWI signal from the particle configuration')
    out_isotropic_fraction = File(
        argstr='-fiso %s', requires=['csf_txt'], genfile=True,
        desc=' Predicted isotropic fractions of the tissues for which response '
             'functions were provided with isotropic_response_txt. Typically, '
             'these are CSF and GM.')
    niter = traits.Int(1e9,
                       argstr='-niter %d',
                       desc='the number of iterations of the metropolis hastings optimizer. '
                       '(default = 10M)')
    out_residual_energy = File(
        argstr='-eext %s', genfile=True,
        desc=' Residual external energy in every voxel.')


class GlobalTractographyOutputSpec(TraitedSpec):
    wm_odf = File(
        exists=True,
        desc='Predicted fibre orientation distribution function (fODF).This fODF is '
        'estimated as part of the global track optimization, and therefore incorporates '
        'the spatial regularization that it imposes. Internally, the fODF is represented '
        'as a discrete sum of apodized point spread functions (aPSF) oriented along the '
        'directions of all particles in the voxel, used to predict the DWI signal from the '
        'particle configuration.')
    isotropic_fraction = File(exists=True,
                              desc='Predicted isotropic fractions of the tissues for '
                              'which response functions were provided with -riso. Typically, '
                              'these are CSF and GM.')
    residual_energy = File(exists=True, desc='Residual external energy in every voxel')
    tck_file = File(exists=True, desc='global tck file')


class GlobalTractography(MRTrix3Base):
    input_spec = GlobalTractographyInputSpec
    output_spec = GlobalTractographyOutputSpec
    _cmd = 'tckglobal'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_odf'] = op.abspath(self._gen_filename('out_fod'))
        outputs['isotropic_fraction'] = op.abspath(self._gen_filename('out_isotropic_fraction'))
        outputs['tck_file'] = op.abspath(self._gen_filename('out_tracks'))
        outputs['residual_energy'] = op.abspath(self._gen_filename('out_residual_energy'))
        return outputs

    def _gen_filename(self, name):
        output = getattr(self.inputs, name)
        _, fname, _ = split_filename(self.inputs.dwi_file)
        if name == 'out_isotropic_fraction':
            if not isdefined(output):
                output = fname + "_tckglobalISOfraction.mif"
            return output
        if name == 'out_fod':
            if not isdefined(output):
                output = fname + "_tckglobalFOD.mif"
            return output
        if name == 'out_tracks':
            if not isdefined(output):
                output = fname + "_tckglobal.tck"
            return output
        if name == 'out_residual_energy':
            if not isdefined(output):
                output = fname + "_residualEnergy.mif"
            return output
        return None


class BuildConnectomeInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-3,
        desc='input tractography')
    in_parc = File(
        exists=True, argstr='%s', position=-2, desc='parcellation file')
    out_file = File(
        'connectome.csv',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output file after processing')
    out_assignments = File(
        argstr='-out_assignments %s', desc='file with streamline assignments')
    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='number of threads. if zero, the number'
        ' of available cpus will be used',
        nohash=True)
    vox_lookup = traits.Bool(
        argstr='-assignment_voxel_lookup',
        desc='use a simple voxel lookup value at each streamline endpoint')
    search_radius = traits.Float(
        argstr='-assignment_radial_search %f',
        desc='perform a radial search from each streamline endpoint to locate '
        'the nearest node. Argument is the maximum radius in mm; if no node is'
        ' found within this radius, the streamline endpoint is not assigned to'
        ' any node.')
    search_reverse = traits.Float(
        argstr='-assignment_reverse_search %f',
        desc='traverse from each streamline endpoint inwards along the '
        'streamline, in search of the last node traversed by the streamline. '
        'Argument is the maximum traversal length in mm (set to 0 to allow '
        'search to continue to the streamline midpoint).')
    search_forward = traits.Float(
        argstr='-assignment_forward_search %f',
        desc='project the streamline forwards from the endpoint in search of a'
        'parcellation node voxel. Argument is the maximum traversal length in '
        'mm.')
    stat_edge = traits.Enum("sum", "mean", "min", "max", argstr='-stat_edge %s',
                            usedefault=True)
    length_scale = traits.Enum("None", "length", "invlength", argstr='%s')
    scale_invnodevol = traits.Bool(False, argstr="-scale_invnodevol")
    in_scalar = File(
        exists=True,
        argstr='-scale_file %s',
        desc='provide the associated image '
        'for the mean_scalar metric')
    in_weights = File(
        exists=True,
        argstr='-tck_weights_in %s',
        desc='specify a text scalar '
        'file containing the streamline weights')
    keep_unassigned = traits.Bool(
        argstr='-keep_unassigned',
        desc='By default, the program discards the'
        ' information regarding those streamlines that are not successfully '
        'assigned to a node pair. Set this option to keep these values (will '
        'be the first row/column in the output matrix)')
    zero_diagonal = traits.Bool(
        argstr='-zero_diagonal',
        desc='set all diagonal entries in the matrix '
        'to zero (these represent streamlines that connect to the same node at'
        ' both ends)')
    symmetric = traits.Bool(
        argstr='-symmetric',
        desc='Make matrices symmetric on output')


class BuildConnectomeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')
    out_assignments = File(exists=True, desc='streamline assignment csv')


class BuildConnectome(MRTrix3Base):
    """
    Generate a connectome matrix from a streamlines file and a node
    parcellation image
    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mat = mrt.BuildConnectome()
    >>> mat.inputs.in_file = 'tracks.tck'
    >>> mat.inputs.in_parc = 'aparc+aseg.nii'
    >>> mat.cmdline                               # doctest: +ELLIPSIS
    'tck2connectome tracks.tck aparc+aseg.nii connectome.csv'
    >>> mat.run()                                 # doctest: +SKIP
    """

    _cmd = 'tck2connectome'
    input_spec = BuildConnectomeInputSpec
    output_spec = BuildConnectomeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        if isdefined(self.inputs.out_assignments):
            outputs['out_assignments'] = op.abspath(self.inputs.out_assignments)
        return outputs

    def _format_arg(self, name, spec, val):
        if name == 'length_scale':
            if val == 'length':
                return '-scale_length'
            if val == 'invlength':
                return '-scale_invlength'
            return ''
        return super(BuildConnectome, self)._format_arg(name, spec, val)


class MRTrixAtlasGraphInputSpec(BuildConnectomeInputSpec):
    atlas_configs = traits.Dict(desc='atlas configs for atlases to run connectivity for',
                                mandatory=True)


class MRTrixAtlasGraphOutputSpec(TraitedSpec):
    connectivity_matfile = File(exists=True)
    commands = File()


class MRTrixAtlasGraph(SimpleInterface):
    """Produce one connectivity matrix per atlas based on DSI Studio tractography"""
    input_spec = MRTrixAtlasGraphInputSpec
    output_spec = MRTrixAtlasGraphOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()
        ifargs['nthreads'] = 1

        # Get number of parallel jobs
        num_threads = ifargs.pop('nthreads')
        atlas_configs = ifargs.pop('atlas_configs')
        del ifargs['in_parc']

        # flatten the atlas_configs
        args = [(atlas_name, atlas_config, self.inputs.in_file, ifargs) for atlas_name,
                atlas_config in atlas_configs.items()]

        if num_threads == 1:
            outputs = [_mrtrix_connectivity(arg) for arg in args]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                outputs = list(pool.map(_mrtrix_connectivity, args))

        commands = [out[0] for out in outputs]
        commands_file = op.join(runtime.cwd, "mrtrix_commands.txt")
        with open(commands_file, "w") as f:
            f.write("\n----------\n".join(commands))
        self._results['commands'] = commands_file

        matfile_list = [out[1] for out in outputs]
        merged_connectivity_file = op.join(runtime.cwd, "combined_connectivity.mat")
        _merge_conmats(matfile_list, args, merged_connectivity_file)
        self._results['connectivity_matfile'] = merged_connectivity_file

        return runtime


def _merge_conmats(matfile_list, recon_args, outfile):
    """Merge the many matfiles output by dsi studio and ensure they conform"""
    connectivity_values = {}

    for matfile, (atlas_name, atlas_config, tck_file, ifargs) in zip(matfile_list, recon_args):
        labels = np.array(atlas_config['node_ids']).astype(np.int)
        connectivity_values[atlas_name + "_region_ids"] = labels
        connectivity_values[atlas_name + "_region_labels"] = np.array(atlas_config['node_names'])
        measure_name = atlas_name + '_' + ifargs['stat_edge']
        if isdefined(ifargs['length_scale']):
            measure_name += "_" + ifargs['length_scale']
        if isdefined(ifargs['scale_invnodevol']):
            measure_name += "_invroiscale"
        connectivity_values[measure_name + "_connectivity"] = np.loadtxt(matfile)
        connectivity_values[measure_name + "_tck"] = tck_file
        connectivity_values[measure_name + "_image"] = atlas_config['dwi_resolution_mif']
    savemat(outfile, connectivity_values, do_compression=True)


def _mrtrix_connectivity(args):
    atlas_name, atlas_config, _, ifargs = args
    csv_name = 'atlas_{}_length_{}_roiscale_{}_stat_{}.csv'.format(
        atlas_name, ifargs['length_scale'], ifargs['scale_invnodevol'],
        ifargs['stat_edge']).replace("<undefined>", "None")
    ifargs = deepcopy(ifargs)
    ifargs['out_file'] = csv_name
    con = BuildConnectome(in_parc=atlas_config['dwi_resolution_mif'], **ifargs)
    con.terminal_output = 'allatonce'
    con.resource_monitor = False
    LOGGER.info(con.cmdline)
    run = con.run(cwd=os.getcwd())
    runtime = run.runtime

    return runtime.cmdline, run.outputs.out_file


class DWIBiasCorrectInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='input DWI image')
    in_mask = File(
        argstr='-mask %s',
        desc='input mask image for bias field estimation')
    use_ants = traits.Bool(
        argstr='-ants',
        mandatory=True,
        desc='use ANTS N4 to estimate the inhomogeneity field',
        xor=['use_fsl'])
    use_fsl = traits.Bool(
        argstr='-fsl',
        mandatory=True,
        desc='use FSL FAST to estimate the inhomogeneity field',
        xor=['use_ants'])
    bias = File(
        argstr='-bias %s',
        desc='bias field')
    out_file = File(
        name_template='%s_biascorr',
        name_source='in_file',
        keep_extension=True,
        argstr='%s',
        position=-1,
        desc='the output bias corrected DWI image',
        genfile=True)


class DWIBiasCorrectOutputSpec(TraitedSpec):
    bias = File(desc='the output bias field', exists=True)
    out_file = File(desc='the output bias corrected DWI image', exists=True)


class DWIBiasCorrect(MRTrix3Base):
    """
    Perform B1 field inhomogeneity correction for a DWI volume series.
    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/scripts/dwibiascorrect.html>
    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bias_correct = mrt.DWIBiasCorrect()
    >>> bias_correct.inputs.in_file = 'dwi.mif'
    >>> bias_correct.inputs.use_ants = True
    >>> bias_correct.cmdline
    'dwibiascorrect -ants dwi.mif dwi_biascorr.mif'
    >>> bias_correct.run()                             # doctest: +SKIP
    """

    _cmd = 'dwibiascorrect'
    input_spec = DWIBiasCorrectInputSpec
    output_spec = DWIBiasCorrectOutputSpec
