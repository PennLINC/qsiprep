# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRtrix3 Interfaces
~~~~~~~~~~~~~~~~~~


"""

import os

import nibabel as nb
import numpy as np
from nilearn.image import crop_img, load_img, threshold_img
from nipype import logging
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
from nipype.interfaces.mrtrix3 import MRConvert
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec
from nipype.utils.filemanip import fname_presuffix, which
from niworkflows.viz.utils import compose_view, cuts_from_bbox

from ..viz.utils import plot_denoise
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
    _to_magnitude,
)

LOGGER = logging.getLogger('nipype.interface')
RC3_ROOT = which('average_response')  # Only exists in RC3
if RC3_ROOT is not None:
    # Use the directory containing average_response
    RC3_ROOT = os.path.split(RC3_ROOT)[0]


class MRTrixGradientTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)


class MRTrixGradientTableOutputSpec(TraitedSpec):
    gradient_file = File(exists=True)


class MRTrixGradientTable(SimpleInterface):
    input_spec = MRTrixGradientTableInputSpec
    output_spec = MRTrixGradientTableOutputSpec

    def _run_interface(self, runtime):
        gtab_fname = fname_presuffix(
            self.inputs.bval_file, suffix='.b', newpath=runtime.cwd, use_ext=False
        )
        _convert_fsl_to_mrtrix(self.inputs.bval_file, self.inputs.bvec_file, gtab_fname)
        self._results['gradient_file'] = gtab_fname
        return runtime


def _convert_fsl_to_mrtrix(bval_file, bvec_file, output_fname):
    vecs = np.loadtxt(bvec_file)
    vals = np.loadtxt(bval_file)
    gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
    np.savetxt(output_fname, gtab, fmt=['%.8f', '%.8f', '%.8f', '%d'])


class MRTrixIngressInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    suffix = traits.Str('', usedefault=True)


class MRTrixIngressOutputSpec(TraitedSpec):
    mif_file = File()


class MRTrixIngress(SimpleInterface):
    input_spec = MRTrixIngressInputSpec
    output_spec = MRTrixIngressOutputSpec

    def _run_interface(self, runtime):
        output_mif = fname_presuffix(
            self.inputs.dwi_file,
            suffix=self.inputs.suffix + '.mif',
            newpath=runtime.cwd,
            use_ext=False,
        )
        if isdefined(self.inputs.b_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file, grad_file=self.inputs.b_file, out_file=output_mif
            )
        elif isdefined(self.inputs.bval_file) and isdefined(self.inputs.bvec_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file,
                in_bval=self.inputs.bval_file,
                in_bvec=self.inputs.bvec_file,
                out_file=output_mif,
            )
        else:
            raise Exception('No valid mrtrix gradient files or fsl bval/bvec files specified')
        convert_run = convert.run()
        self._results['mif_file'] = convert_run.outputs.out_file

        return runtime


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    mask = File(exists=True, argstr='-mask %s', position=1, desc='mask image')
    extent = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        argstr='-extent %d,%d,%d',
        desc='set the window size of the denoising filter. (default = 5,5,5)',
    )
    noise_image = File(
        argstr='-noise %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=-1,
        desc='the output denoised DWI image',
    )
    out_report = File(
        'dwidenoise_report.svg', usedefault=True, desc='filename for the visual report'
    )


class DWIDenoiseOutputSpec(SeriesPreprocReportOutputSpec):
    noise_image = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)


class DWIDenoise(SeriesPreprocReport, MRTrix3Base):
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

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        noise_name = outputs['noise_image']
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class DWIDenoise2InputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    mask = File(exists=True, desc='mask image')
    # The sliding-window kernel and the subsampling factor are properties of the
    # multi-resolution schedule rather than command-line options
    schedule = traits.Str(
        argstr='-schedule %s',
        desc='name of a bundled noise estimation schedule, or a path to a schedule file',
    )
    datatype = traits.Enum(
        'float32',
        'float64',
        argstr='-datatype %s',
        desc='eigendecomposition datatype',
    )
    decomposition = traits.Enum(
        'bdcsvd',
        'selfadjoint',
        argstr='-decomposition %s',
        desc='patch decomposition method',
    )
    estimator = traits.Enum(
        'exp1',
        'exp2',
        'med',
        'mrm2023',
        'tbme2022',
        argstr='-estimator %s',
        desc='noise level estimator',
    )
    noise_in = traits.Either(
        traits.Float,
        File(exists=True),
        argstr='-noise_in %s',
        xor=('fixed_rank',),
        desc='scalar noise level or pre-estimated noise map',
    )
    fixed_rank = traits.Int(
        argstr='-fixed_rank %d', xor=('noise_in',), desc='fixed input signal rank'
    )
    demodulate = traits.Enum(
        'none',
        'linear',
        'hann',
        'apc',
        argstr='-demodulate %s',
        desc='phase demodulation mode',
    )
    demod_axes = traits.Str(
        argstr='-demod_axes %s',
        desc='comma-separated FFT axes for phase demodulation',
    )
    demean = traits.Enum(
        'none',
        'volume_groups',
        'shells',
        'all',
        argstr='-demean %s',
        desc='demeaning method before PCA',
    )
    noise_dof = traits.Int(
        argstr='-noise_dof %d',
        desc='receive channels combined by sum-of-squares reconstruction of magnitude data',
    )
    vst_method = traits.Enum(
        'none',
        'linear',
        'foi',
        'koay',
        'mom',
        argstr='-vst_method %s',
        desc='variance-stabilising transform applied before PCA',
    )
    preserve_noise_bias = traits.Bool(
        argstr='-preserve_noise_bias',
        desc='retain the noise-floor bias in magnitude output instead of removing it',
    )
    debias_anchor = traits.Enum(
        'sample',
        'group_mean',
        argstr='-debias_anchor %s',
        desc='operating point at which the variance-stabilising inverse is evaluated',
    )
    preconditioned_input = File(
        argstr='-preconditioned_input %s',
        desc='export preconditioned PCA input',
    )
    preconditioned_output = File(
        argstr='-preconditioned_output %s',
        desc='export output before reversing preconditioning',
    )
    filter_method = traits.Enum(
        'optshrink',
        'optthresh',
        'truncate',
        argstr='-filter %s',
        desc='eigenvalue filtering method',
    )
    aggregator = traits.Enum(
        'exclusive',
        'gaussian',
        'invl0',
        'rank',
        'uniform',
        argstr='-aggregator %s',
        desc='overlapping-patch aggregation method',
    )
    noise_image = File(
        argstr='-noise_out %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map',
    )
    lamplus = File(argstr='-lamplus %s', desc='estimated upper noise eigenspectrum bound')
    rank_pcanonzero = File(argstr='-rank_pcanonzero %s', desc='non-zero PCA rank before denoising')
    rank_input = File(argstr='-rank_input %s', desc='estimated input rank per denoising patch')
    rank_output = File(
        argstr='-rank_output %s',
        desc='estimated output rank after patch aggregation',
    )
    variance_removed = File(
        argstr='-variance_removed %s',
        desc='fraction of variance removed by PCA',
    )
    eigenspectra = File(
        argstr='-eigenspectra %s',
        desc='matrix of eigenvalue spectra across patches',
    )
    residual_statistics = traits.Tuple(
        File(),
        File(),
        File(),
        argstr='-residual_statistics %s %s %s',
        desc='residual mean, variance, and maximum-absolute-value images',
    )
    max_dist = File(argstr='-max_dist %s', desc='maximum within-patch voxel distance')
    voxelcount = File(argstr='-voxelcount %s', desc='voxels contributing to each PCA')
    patchcount = File(argstr='-patchcount %s', desc='unique patches containing each voxel')
    sum_aggregation = File(
        argstr='-sum_aggregation %s',
        desc='sum of aggregation weights per voxel',
    )
    sum_optshrink = File(
        argstr='-sum_optshrink %s',
        desc='sum of optimal-shrinkage weights per patch',
    )
    grad_file = File(
        exists=True,
        argstr='-grad %s',
        xor=('bvec_file', 'bval_file'),
        desc='MRtrix-format diffusion gradient scheme',
    )
    bvec_file = File(
        exists=True,
        argstr='-fslgrad %s %s',
        requires=('bval_file',),
        xor=('grad_file',),
        desc='FSL-format diffusion gradient b-vector file',
    )
    bval_file = File(
        exists=True,
        requires=('bvec_file',),
        xor=('grad_file',),
        desc='FSL-format diffusion gradient b-value file',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=-1,
        desc='the output denoised DWI image',
    )
    out_report = File(
        'dwidenoise_report.svg',
        usedefault=True,
        desc='filename for the visual report',
    )


class DWIDenoise2OutputSpec(SeriesPreprocReportOutputSpec):
    noise_image = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)
    preconditioned_input = File(exists=True, desc='preconditioned PCA input')
    preconditioned_output = File(exists=True, desc='output before reversal of preconditioning')
    lamplus = File(exists=True, desc='estimated upper noise eigenspectrum bound')
    rank_pcanonzero = File(exists=True, desc='non-zero PCA rank before denoising')
    rank_input = File(exists=True, desc='estimated input rank per denoising patch')
    rank_output = File(exists=True, desc='estimated output rank after patch aggregation')
    variance_removed = File(exists=True, desc='fraction of variance removed by PCA')
    eigenspectra = File(exists=True, desc='matrix of eigenvalue spectra across patches')
    residual_statistics = traits.Tuple(
        File(exists=True), File(exists=True), File(exists=True), desc='residual statistic images'
    )
    max_dist = File(exists=True, desc='maximum within-patch voxel distance')
    voxelcount = File(exists=True, desc='voxels contributing to each PCA')
    patchcount = File(exists=True, desc='unique patches containing each voxel')
    sum_aggregation = File(exists=True, desc='sum of aggregation weights per voxel')
    sum_optshrink = File(exists=True, desc='sum of optimal-shrinkage weights per patch')


class DWIDenoise2(SeriesPreprocReport, MRTrix3Base):
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

    _cmd = 'dwidenoise2'
    input_spec = DWIDenoise2InputSpec
    output_spec = DWIDenoise2OutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'bvec_file':
            # -fslgrad takes both files, so format them here rather than passing a tuple
            # to a File trait, which nipype would try to shell-quote as a single value.
            return spec.argstr % (value, self.inputs.bval_file)
        return super()._format_arg(name, spec, value)

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        noise_name = outputs['noise_image']
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class DWIBiasCorrectInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    mask = File(argstr='-mask %s', desc='input mask image for bias field estimation')
    method = traits.Enum('ants', 'fsl', argstr='%s', position=1, usedefault=True)
    bias_image = File(
        argstr='-bias %s',
        name_source='in_file',
        name_template='%s_bias.nii.gz',
        keep_extension=False,
        desc='bias field',
    )
    out_file = File(
        name_source='in_file',
        keep_extension=False,
        argstr='%s',
        name_template='%s_N4.nii.gz',
        position=-1,
        desc='the output bias corrected DWI image',
    )
    # The argstr is a bare placeholder: DWIBiasCorrect._format_arg builds the real
    # flag, because 3.0.x spells these -ants.b and the development branch -ants_b.
    # The placeholder is load-bearing — nipype formats only traits that declare an
    # argstr, so removing it would silently emit nothing at all.
    ants_b = traits.Str(default_value='[150,3]', argstr='%s', usedefault=True)
    ants_c = traits.Str(default_value='[200x200,1e-6]', argstr='%s', usedefault=True)
    ants_s = traits.Str(default_value='4', argstr='%s')
    mrtrix_version = traits.Enum(
        'stable',
        'dev',
        usedefault=True,
        desc='which MRtrix3 installation this node will run against',
    )
    out_report = File('n4_report.svg', usedefault=True, desc='filename for the visual report')
    bzero_max = traits.Int(
        argstr='-config BZeroThreshold %d',
        desc='Maximum b-value that can be considered a b=0',
    )


class DWIBiasCorrectOutputSpec(SeriesPreprocReportOutputSpec):
    bias_image = File(desc='the output bias field', exists=True)
    out_file = File(desc='the output bias corrected DWI image', exists=True)


class DWIBiasCorrect(SeriesPreprocReport, MRTrix3Base):
    """
    Perform B1 field inhomogeneity correction for a DWI volume series.
    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/scripts/dwibiascorrect.html>
    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bias_correct = mrt.DWIBiasCorrect()
    >>> bias_correct.inputs.in_file = 'dwi.mif'
    >>> bias_correct.inputs.method = 'ants'
    >>> bias_correct.cmdline
    'dwibiascorrect ants dwi.mif dwi_biascorr.mif'
    >>> bias_correct.run()                             # doctest: +SKIP
    """

    _cmd = 'dwibiascorrect'
    input_spec = DWIBiasCorrectInputSpec
    output_spec = DWIBiasCorrectOutputSpec

    def _format_arg(self, name, spec, value):
        if name in ('ants_b', 'ants_c', 'ants_s'):
            separator = '_' if self.inputs.mrtrix_version == 'dev' else '.'
            return f'-ants{separator}{name[-1]} {value}'
        return super()._format_arg(name, spec, value)

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        noise_name = outputs['bias_image']
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class MRDeGibbsInputSpec(MRTrix3BaseInputSpec, SeriesPreprocReportInputSpec):
    out_report = File('degibbs_report.svg', usedefault=True, desc='filename for the visual report')
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    out_file = File(
        name_source='in_file',
        keep_extension=False,
        argstr='%s',
        name_template='%s_mrdegibbs.nii.gz',
        position=-1,
        desc="the output de-Gibbs'd DWI image",
    )
    mask = File(desc='input mask image for the visual report')
    nshifts = traits.Int(
        default=20, argstr='-nshifts %d', desc='discretization of subpixel spacing.'
    )
    axes = traits.Enum(
        '0,1',
        '0,2',
        '1,2',
        default='0,1',
        argstr='-axes %s',
        desc='select the slice axes (default: 0,1 - i.e. x-y)',
    )
    minw = traits.Int(
        default=1, argstr='-minW %d', desc='left border of window used for TV computation'
    )
    maxw = traits.Int(
        default=3, argstr='-maxW %d', desc='right border of window used for TV computation'
    )
    dimensionality = traits.Enum(
        2,
        3,
        argstr='-dimensionality %d',
        desc=(
            'dimensionality of the operation: 2 for the slice-wise method of Kellner et al., '
            '3 for the volume-wise extension of Bautista et al. Left unset, mrdegibbs '
            'defaults to 2. Requires --mrtrix-version dev; the released mrdegibbs does '
            'not accept this option.'
        ),
    )


class MRDeGibbsOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(desc="the output de-Gibbs'd DWI image")


class MRDeGibbs(SeriesPreprocReport, MRTrix3Base):
    input_spec = MRDeGibbsInputSpec
    output_spec = MRDeGibbsOutputSpec
    _cmd = 'mrdegibbs'

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        return input_dwi, denoised_nii, None

    def _generate_report(self):
        """Generate a reportlet."""
        LOGGER.info('Generating denoising visual report')

        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # mrdegibbs emits complex data when it is given complex data, and the report
        # always shows magnitude. This mirrors SeriesPreprocReport._generate_report,
        # which this method overrides.
        input_dwi = _to_magnitude(input_dwi)
        denoised_nii = _to_magnitude(denoised_nii)

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
        image_intensities = np.array([img.mean() for img in image_data.T])
        lowb_index = int(np.argmax(image_intensities))
        highb_index = int(np.argmin(image_intensities))

        # Original images
        orig_lowb_nii = input_dwi.slicer[..., lowb_index]
        orig_highb_nii = input_dwi.slicer[..., highb_index]

        # Denoised images
        denoised_lowb_nii = denoised_nii.slicer[..., lowb_index]
        denoised_highb_nii = denoised_nii.slicer[..., highb_index]

        # Find spatial extent of the image
        contour_nii = mask_nii = None
        if isdefined(self.inputs.mask):
            contour_nii = load_img(self.inputs.mask)
        else:
            mask_nii = threshold_img(denoised_lowb_nii, 50)

        ref_nii = contour_nii or mask_nii
        cuts = cuts_from_bbox(ref_nii, cuts=self._n_cuts)
        _, crop_offset = crop_img(ref_nii, return_offset=True)

        diff_lowb_nii = nb.Nifti1Image(
            orig_lowb_nii.get_fdata() - denoised_lowb_nii.get_fdata(),
            affine=denoised_lowb_nii.affine,
        )
        diff_highb_nii = nb.Nifti1Image(
            orig_highb_nii.get_fdata() - denoised_highb_nii.get_fdata(),
            affine=denoised_highb_nii.affine,
        )

        # Call composer
        compose_view(
            plot_denoise(
                denoised_lowb_nii,
                denoised_highb_nii,
                'moving-image',
                estimate_brightness=True,
                cuts=cuts,
                crop_offset=crop_offset,
                label='De-Gibbs',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            plot_denoise(
                diff_lowb_nii,
                diff_highb_nii,
                'fixed-image',
                estimate_brightness=True,
                cuts=cuts,
                crop_offset=crop_offset,
                label='Estimated Ringing',
                lowb_contour=None,
                highb_contour=None,
                compress=False,
            ),
            out_file=self._out_report,
        )

        self._calculate_nmse(input_dwi, denoised_nii)


class _ITKTransformConvertInputSpec(CommandLineInputSpec):
    in_transform = traits.File(exists=True, argstr='%s', mandatory=True, position=0)
    operation = traits.Enum(
        'itk_import', default='itk_import', usedefault=True, posision=1, argstr='%s'
    )
    out_transform = traits.File(
        argstr='%s',
        name_source='in_transform',
        name_template='%s.txt',
        keep_extension=False,
        position=-1,
    )


class _ITKTransformConvertOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ITKTransformConvert(CommandLine):
    _cmd = 'transformconvert'
    input_spec = _ITKTransformConvertInputSpec
    output_spec = _ITKTransformConvertOutputSpec


class _TransformHeaderInputSpec(CommandLineInputSpec):
    transform_file = traits.File(exists=True, position=0, mandatory=True, argstr='-linear %s')
    in_image = traits.File(exists=True, mandatory=True, position=1, argstr='%s')
    out_image = traits.File(
        argstr='%s',
        name_source='in_image',
        name_template='%s_hdrxform.nii.gz',
        keep_extension=False,
        position=-1,
    )


class _TransformHeaderOutputSpec(TraitedSpec):
    out_image = File(exists=True)


class TransformHeader(CommandLine):
    input_spec = _TransformHeaderInputSpec
    output_spec = _TransformHeaderOutputSpec
    _cmd = 'mrtransform -strides -1,-2,3'


class _PolarToComplexInputSpec(CommandLineInputSpec):
    mag_file = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    phase_file = traits.File(exists=True, mandatory=True, position=1, argstr='%s')
    out_file = traits.File(
        exists=False,
        name_source='mag_file',
        name_template='%s_complex.nii.gz',
        keep_extension=False,
        position=-1,
        argstr='-polar %s',
    )


class _PolarToComplexOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class PolarToComplex(CommandLine):
    """Convert a magnitude and phase image pair to a single complex image using mrcalc."""

    input_spec = _PolarToComplexInputSpec
    output_spec = _PolarToComplexOutputSpec

    _cmd = 'mrcalc'


class _ComplexToMagnitudeInputSpec(CommandLineInputSpec):
    complex_file = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    out_file = traits.File(
        exists=False,
        name_source='complex_file',
        name_template='%s_mag.nii.gz',
        keep_extension=False,
        position=-1,
        argstr='-abs %s',
    )


class _ComplexToMagnitudeOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ComplexToMagnitude(CommandLine):
    """Extract the magnitude portion of a complex image using mrcalc."""

    input_spec = _ComplexToMagnitudeInputSpec
    output_spec = _ComplexToMagnitudeOutputSpec

    _cmd = 'mrcalc'
