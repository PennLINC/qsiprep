# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype interfaces to niimath's fieldmap operations.

`niimath <https://github.com/rordenlab/niimath>`_ (Chris Rorden, BSD-2-Clause) is a
permissively licensed tool that provides the phase/fieldmap operations qsiprep needs:

* ``-romeo`` unwraps phase (used here in place of ``prelude``);
* ``-fugue`` applies a B0 fieldmap to an EPI to correct susceptibility distortion;
* ``-fmapprep`` builds a rad/s fieldmap from a wrapped phase difference.

Each op was checked on a known field and produces sensible fieldmaps/unwrapped phase.
"""

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


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
