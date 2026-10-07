# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for svht_denoise.

`svht_denoise <https://github.com/rordenlab/svht_denoise>`_ denoises DWI series by
local PCA with the optimal singular value shrinkage of Gavish and Donoho, and removes
Gibbs ringing by local subvoxel shifts, with partial Fourier support. Unlike
``dwidenoise`` and ``dwidenoise2``, it is not restricted to non-commercial use.
"""

import nibabel as nb
import numpy as np
from nilearn.image import load_img
from nipype import logging
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    isdefined,
    traits,
)

from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
)
from .mrtrix import MRDeGibbs

LOGGER = logging.getLogger('nipype.interface')

# svht_denoise estimates the noise level from the volumes left after demeaning, and
# refuses to write a noise map from fewer than this
_MIN_NOISE_COLUMNS = 3


def _count_demeaned_shells(bvals):
    """Count the shells ``svht_denoise -demean y`` removes a mean from.

    Shells are clustered as svht_denoise (and MRtrix3) do: b <= 10 is b=0, and sorted
    neighbours less than 80 s/mm^2 apart share a shell. A shell of one volume is not
    demeaned.
    """
    bvals = np.sort(np.asarray(bvals, dtype=float))
    b0 = bvals <= 10
    sizes = [int(b0.sum())]
    weighted = bvals[~b0]
    if weighted.size:
        breaks = np.flatnonzero(np.diff(weighted) >= 80) + 1
        sizes += [len(shell) for shell in np.split(weighted, breaks)]
    return sum(size > 1 for size in sizes)


class _SVHTDenoiseInputSpec(CommandLineInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='input DWI series (the magnitude, if phase_file is given)',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=1,
        desc='the output denoised DWI series',
    )
    noise_image = File(
        argstr='-noise %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map',
    )
    phase_file = File(
        exists=True,
        argstr='-phase %s',
        desc=(
            'phase of the input series. The magnitude and phase are rotated onto the '
            'real axis before denoising, and the output is then signed.'
        ),
    )
    phase_units = traits.Enum(
        'auto',
        'radians',
        'degrees',
        'turns',
        argstr='-phaseunits %s',
        requires=['phase_file'],
        desc='units of phase_file',
    )
    extent = traits.Int(
        argstr='-extent %d',
        desc=(
            'patch size: odd, with extent**3 greater than the number of volumes. '
            'Defaults to the smallest such value.'
        ),
    )
    demean = traits.Bool(
        argstr='-demean %s',
        desc=(
            'remove the mean of each b-value shell before PCA and restore it after. '
            'True needs bval_file.'
        ),
    )
    bval_file = File(
        exists=True,
        argstr='-bval %s',
        desc='FSL b-values, which define the shells for demean. Only valid with demean.',
    )
    filter_method = traits.Enum(
        'optshrink',
        'optthresh',
        'truncate',
        argstr='-filter %s',
        desc='how components are kept',
    )
    shape = traits.Enum('sphere', 'cube', argstr='-shape %s', desc='patch shape')
    aggregator = traits.Enum(
        'gaussian',
        'uniform',
        'exclusive',
        argstr='-aggregator %s',
        desc='how overlapping patch estimates make the output',
    )
    aggregator_fwhm = traits.Float(
        argstr='-aggregator_fwhm %g',
        desc='Gaussian aggregator width, in units of the patch-centre spacing',
    )
    stride = traits.Enum(1, 2, argstr='-stride %d', desc='patch-centre spacing in voxels')
    vst = traits.Bool(
        argstr='-vst %s',
        desc='variance-stabilize magnitude data before denoising',
    )
    noise_dof = traits.Range(
        low=1,
        high=64,
        argstr='-noise_dof %d',
        desc='receive channels behind each magnitude (sum of squares)',
    )
    preserve_noise_bias = traits.Bool(
        argstr='-preserve_noise_bias',
        desc='invert the variance-stabilizing transform algebraically, keeping the noise floor',
    )
    nthreads = traits.Int(argstr='-nthreads %d', nohash=True, desc='number of threads')
    mask = File(desc='mask image for the visual report')
    out_report = File(
        'svht_denoise_report.svg', usedefault=True, desc='filename for the visual report'
    )


class _SVHTDenoiseOutputSpec(SeriesPreprocReportOutputSpec):
    noise_image = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI series', exists=True)


class SVHTDenoise(SeriesPreprocReport, CommandLine):
    """Denoise a DWI series by local PCA with optimal singular value shrinkage.

    The noise level is estimated with the Gavish-Donoho median estimator, and
    magnitude data are variance-stabilized first, so the Rician noise floor is
    removed rather than denoised.
    """

    _cmd = 'svht_denoise'
    input_spec = _SVHTDenoiseInputSpec
    output_spec = _SVHTDenoiseOutputSpec

    def _demean_is_possible(self):
        """Whether demeaning by shell leaves svht_denoise enough data to estimate the noise.

        Each shell svht_denoise demeans costs a column of the noise estimate, and it needs
        at least 3 to write a noise map. A short series can fall below that only because of
        demeaning, so demeaning is skipped there rather than failing a run that would
        succeed without it.
        """
        if not (self.inputs.demean and isdefined(self.inputs.bval_file)):
            return True
        bvals = np.loadtxt(self.inputs.bval_file, ndmin=1)
        return bvals.size - _count_demeaned_shells(bvals) >= _MIN_NOISE_COLUMNS

    def _run_interface(self, runtime):
        if not self._demean_is_possible():
            LOGGER.warning(
                'Not demeaning %s by shell: it would leave svht_denoise fewer than %d '
                'volumes to estimate the noise level from.',
                self.inputs.in_file,
                _MIN_NOISE_COLUMNS,
            )
        return super()._run_interface(runtime)

    def _format_arg(self, name, spec, value):
        if name in ('demean', 'bval_file') and not self._demean_is_possible():
            return ''
        if name in ('demean', 'vst'):
            # svht_denoise spells its switches y/n
            return spec.argstr % ('y' if value else 'n')
        if name == 'preserve_noise_bias':
            # A bare flag: present when true, absent when false
            return spec.argstr if value else ''
        return super()._format_arg(name, spec, value)

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        denoised_nii = load_img(outputs['out_file'])
        if isdefined(self.inputs.phase_file):
            # The output is the real-axis rotation of the complex data, which is signed.
            # Its absolute value is comparable to the input magnitude.
            denoised_nii = nb.Nifti1Image(
                np.abs(denoised_nii.get_fdata(dtype=np.float32)),
                denoised_nii.affine,
                denoised_nii.header,
            )
        noisenii = load_img(outputs['noise_image'])
        return input_dwi, denoised_nii, noisenii


class _SVHTDeGibbsInputSpec(CommandLineInputSpec, SeriesPreprocReportInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='input magnitude DWI series',
    )
    out_file = File(
        name_template='%s_unrung.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=1,
        desc="the output de-Gibbs'd DWI series",
    )
    # Unringing alone; denoising has its own node, so each step has its own report
    degibbs = traits.Enum('o', argstr='-degibbs %s', usedefault=True, desc='unringing only')
    partial_fourier = traits.Enum(
        0.875,
        0.75,
        argstr='-pF %g',
        desc=('partial Fourier factor along the second (y) voxel axis. Omit for full k-space.'),
    )
    nthreads = traits.Int(argstr='-nthreads %d', nohash=True, desc='number of threads')
    mask = File(desc='mask image for the visual report')
    out_report = File(
        'svht_degibbs_report.svg', usedefault=True, desc='filename for the visual report'
    )


class _SVHTDeGibbsOutputSpec(SeriesPreprocReportOutputSpec):
    out_file = File(desc="the output de-Gibbs'd DWI series", exists=True)


class SVHTDeGibbs(SeriesPreprocReport, CommandLine):
    """Remove Gibbs ringing by local subvoxel shifts, with partial Fourier support.

    Full k-space unringing matches ``mrdegibbs``; the partial Fourier factors
    (7/8 and 6/8) follow the RPG method of Lee et al. (2021).
    """

    _cmd = 'svht_denoise'
    input_spec = _SVHTDeGibbsInputSpec
    output_spec = _SVHTDeGibbsOutputSpec

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        denoised_nii = load_img(self._list_outputs()['out_file'])
        return input_dwi, denoised_nii, None

    # The same report as mrdegibbs: the unrung images and the ringing that was removed
    _generate_report = MRDeGibbs._generate_report
