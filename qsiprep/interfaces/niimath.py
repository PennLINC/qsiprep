# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype interfaces to niimath's fieldmap operations.

`niimath <https://github.com/rordenlab/niimath>`_ (Chris Rorden, BSD-2-Clause) is a
permissively licensed tool that provides the phase/fieldmap operations qsiprep needs:

* ``-romeo`` unwraps phase (used here in place of ``prelude``);
* ``-fugue`` applies a B0 fieldmap to an EPI to correct susceptibility distortion;
* ``-fmapprep`` builds a rad/s fieldmap from a wrapped phase difference;
* ``-skullstrip`` extracts the brain (AFNI 3dSkullStrip's surface method, no model,
  no template), used everywhere qsiprep used FSL BET.

Each op was checked on a known field and produces sensible fieldmaps/unwrapped phase.
"""

import os
import os.path as op

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class _RomeoUnwrapInputSpec(CommandLineInputSpec):
    phase_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='wrapped phase image, in radians (3D)',
    )
    magnitude_file = File(
        exists=True,
        mandatory=True,
        argstr='-romeo %s',
        position=1,
        desc='magnitude image guiding the unwrap (romeo requires it as a positional)',
    )
    mask_file = File(
        exists=True,
        argstr='-k %s',
        position=2,
        desc='brain mask restricting the unwrap (romeo -k); '
        'omit to let romeo compute a robustmask',
    )
    no_phase_rescale = traits.Bool(
        True,
        usedefault=True,
        argstr='-no-phase-rescale',
        position=3,
        desc='keep the input radian scaling instead of rescaling to [-pi, pi]; '
        'the input here is already in radians',
    )
    no_mask_out = traits.Bool(
        True,
        usedefault=True,
        argstr='-no-mask-out',
        position=4,
        desc='suppress the <out>_mask side output',
    )
    unwrapped_phase_file = File(
        argstr='%s',
        position=-1,
        name_source='phase_file',
        name_template='%s_unwrapped.nii.gz',
        keep_extension=False,
        hash_files=False,
        desc='unwrapped phase image',
    )


class _RomeoUnwrapOutputSpec(TraitedSpec):
    unwrapped_phase_file = File(exists=True, desc='unwrapped phase image, in radians')


class RomeoUnwrap(CommandLine):
    """Unwrap phase with niimath's ROMEO implementation (used in place of ``prelude``).

    ``-romeo`` must be the first operation because ``-no-phase-rescale`` re-reads
    the unscaled input; the input/output trait positions enforce that ordering.
    """

    input_spec = _RomeoUnwrapInputSpec
    output_spec = _RomeoUnwrapOutputSpec
    _cmd = 'niimath'


class _FugueInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='EPI image to unwarp',
    )
    fmap_file = File(
        exists=True,
        mandatory=True,
        argstr='-fugue %s',
        position=1,
        desc='B0 fieldmap in rad/s on the input grid',
    )
    dwell_time = traits.Float(
        mandatory=True,
        argstr='%g',
        position=2,
        desc='effective echo spacing in seconds (BIDS EffectiveEchoSpacing)',
    )
    unwarp_direction = traits.Enum(
        'x',
        'y',
        'z',
        'x-',
        'y-',
        'z-',
        'i',
        'j',
        'k',
        'i-',
        'j-',
        'k-',
        mandatory=True,
        argstr='%s',
        position=3,
        desc='phase-encoding axis, with an optional trailing "-"',
    )
    out_file = File(
        argstr='%s',
        position=-1,
        name_source='in_file',
        name_template='%s_unwarped.nii.gz',
        keep_extension=False,
        hash_files=False,
        desc='unwarped EPI',
    )


class _FugueOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='unwarped EPI')


class Fugue(CommandLine):
    """Apply a B0 fieldmap to an EPI with niimath to correct susceptibility distortion.

    The voxel shift along ``unwarp_direction`` is ``fmap/(2*pi)*dwell*N``.
    """

    input_spec = _FugueInputSpec
    output_spec = _FugueOutputSpec
    _cmd = 'niimath'


class _FieldmapPrepInputSpec(CommandLineInputSpec):
    phasediff_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='wrapped phase-difference image',
    )
    magnitude_file = File(
        exists=True,
        mandatory=True,
        argstr='-fmapprep %s',
        position=1,
        desc='brain-extracted magnitude image (supplies the mask)',
    )
    delta_te = traits.Float(
        mandatory=True,
        argstr='%g',
        position=2,
        desc='echo time difference EchoTime2 - EchoTime1, in milliseconds',
    )
    no_debranch = traits.Bool(
        False,
        usedefault=True,
        argstr='-no-debranch',
        position=3,
        desc='disable the 2*pi branch-outlier correction',
    )
    out_file = File(
        argstr='%s',
        position=-1,
        name_source='phasediff_file',
        name_template='%s_fieldmap.nii.gz',
        keep_extension=False,
        hash_files=False,
        desc='fieldmap in rad/s (0 outside the mask)',
    )


class _FieldmapPrepOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='fieldmap in rad/s')


class FieldmapPrep(CommandLine):
    """Build a rad/s B0 fieldmap from a wrapped phase difference with niimath.

    Runs ROMEO unwrapping, rad/s scaling by ``delta_te``, and a 2*pi branch-outlier
    cleanup (disable with ``no_debranch``).
    """

    input_spec = _FieldmapPrepInputSpec
    output_spec = _FieldmapPrepOutputSpec
    _cmd = 'niimath'


class _SkullStripInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='head image (scalar 3D; any modality)',
    )
    skullstrip = traits.Bool(
        True,
        usedefault=True,
        argstr='-skullstrip',
        position=1,
        desc='the operation; always on, here so it precedes -faithful and the output',
    )
    faithful = traits.Bool(
        False,
        usedefault=True,
        argstr='-faithful',
        position=2,
        desc='run the reference deformation kernel (about 1.8x slower, bit-reproducible '
        'with the pre-optimization release); the default kernel is the same algorithm',
    )
    out_file = File(
        argstr='%s',
        position=-1,
        name_source='in_file',
        name_template='%s_brain.nii.gz',
        keep_extension=False,
        hash_files=False,
        desc='brain-extracted image: in-mask voxels keep their intensities, the rest is '
        'set to the image minimum',
    )
    num_threads = traits.Int(desc='OpenMP threads for the surface node loop')


class _SkullStripOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='brain-extracted image')
    mask_file = File(exists=True, desc='binary brain mask (uint8) on the input grid')


class SkullStrip(CommandLine):
    """Extract the brain with ``niimath -skullstrip``.

    AFNI-style surface skull stripping: a surface expands from inside the head until it
    wraps the brain (the method of ``3dSkullStrip -no_use_edge``), with no template, mask or
    network. On a 1 mm T2w it takes about 1.5 s and 90 MB and its mask agrees with
    SynthStrip's at Dice 0.96; FSL BET, which it replaces, was at 0.96 too but under-covered
    the brain margin on T2w by ~18 %.

    niimath writes no mask file: it keeps in-brain intensities and fills the rest with the
    image minimum. The mask is recovered here as ``out > min(out)``, which loses only brain
    voxels that sit exactly at the image minimum (none in practice, since the minimum is
    air). The op is compiled in only when niimath is built with ``SKULLSTRIP=1``
    (qsiprep's image is); the release zips and the PyPI wheel refuse it.
    """

    input_spec = _SkullStripInputSpec
    output_spec = _SkullStripOutputSpec
    _cmd = 'niimath'

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        if isdefined(self.inputs.num_threads):
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})
        runtime = super()._run_interface(runtime, correct_return_codes)
        out_file = self._list_outputs()['out_file']
        if not op.exists(out_file):
            raise RuntimeError(
                'niimath -skullstrip produced no output; is this niimath built with '
                'SKULLSTRIP=1? ' + (runtime.stderr or '') + (runtime.stdout or '')
            )
        img = nb.load(out_file)
        data = np.asanyarray(img.dataobj)
        mask = (data > data.min()).astype('uint8')
        mask_file = fname_presuffix(
            self.inputs.in_file, suffix='_brain_mask.nii.gz', newpath=runtime.cwd, use_ext=False
        )
        mask_img = nb.Nifti1Image(mask, img.affine, img.header)
        mask_img.set_data_dtype('uint8')
        mask_img.to_filename(mask_file)
        self._mask_file = mask_file
        return runtime

    def _list_outputs(self):
        outputs = super()._list_outputs()
        mask_file = getattr(self, '_mask_file', None)
        if mask_file is None:
            mask_file = fname_presuffix(
                self.inputs.in_file,
                suffix='_brain_mask.nii.gz',
                newpath=os.getcwd(),
                use_ext=False,
            )
        outputs['mask_file'] = mask_file
        return outputs


def allineate_json_to_itk(json_file, out_file):
    """Write niimath -allineate's ``-savemat`` affine as an ITK text transform.

    The JSON holds ``fixed_to_moving``, a 4x4 world-space (RAS mm) matrix mapping fixed
    points to moving points: exactly ITK's convention, except that ITK works in LPS. Flipping
    the first two axes on both sides converts it. The result is a plain
    ``AffineTransform_double_3_3`` with a zero centre, usable as antsRegistration's
    ``initial_moving_transform``.
    """
    import json

    with open(json_file) as f:
        affine = json.load(f)
    ras = np.asarray(affine['fixed_to_moving'], dtype=np.float64).reshape(4, 4)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps = flip @ ras @ flip
    import SimpleITK as sitk

    from .itk import affine_from_matrix

    sitk.WriteTransform(affine_from_matrix(lps), out_file)
    return out_file


class _AllineateInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='moving image')
    reference = File(
        exists=True, mandatory=True, argstr='-allineate %s', position=1, desc='fixed image'
    )
    savemat = File(
        argstr='-savemat %s',
        position=2,
        name_source='in_file',
        name_template='%s_allineate.json',
        keep_extension=False,
        hash_files=False,
        desc='the fitted world-space affine, as niimath writes it',
    )
    out_file = File(
        argstr='%s',
        position=-1,
        name_source='in_file',
        name_template='%s_allineate.nii.gz',
        keep_extension=False,
        hash_files=False,
        desc='moving image resliced onto the fixed grid',
    )
    num_threads = traits.Int(desc='OpenMP threads')


class _AllineateOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    savemat = File(exists=True)
    out_transform = File(exists=True, desc='the affine as an ITK text transform (LPS)')


class Allineate(CommandLine):
    """Affine registration with ``niimath -allineate`` (its default "fast" engine).

    A 12-DOF multiresolution fit (8 -> 4 -> 2 mm, Hellinger + correlation ratio) adapted from
    AFNI 3dAllineate, in 1-2 s for a 1 mm head. Used here as the initializer of the ACPC
    registration: on a T1w rotated 25 degrees the registration started from it lands within
    0.2 degrees of the answer antsAI's full-resolution search gave in 55 s. The rigid-only
    ``-warp shr`` engine is not used: it is slower (36 s) and lands 4-7 degrees off on the same
    pair.
    """

    input_spec = _AllineateInputSpec
    output_spec = _AllineateOutputSpec
    _cmd = 'niimath'

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        if isdefined(self.inputs.num_threads):
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})
        runtime = super()._run_interface(runtime, correct_return_codes)
        outputs = self._list_outputs()
        allineate_json_to_itk(outputs['savemat'], outputs['out_transform'])
        return runtime

    def _list_outputs(self):
        outputs = super()._list_outputs()
        outputs['out_transform'] = fname_presuffix(
            self.inputs.in_file, suffix='_allineate.txt', newpath=os.getcwd(), use_ext=False
        )
        return outputs
