"""Tests for the qsiprep.interfaces.niimath module.

The ``cmdline`` tests pin the command each interface builds and run anywhere.
The functional tests need the ``niimath`` binary and are skipped without it
(the fieldmap operations require a source build, not the fslmaths-only PyPI
wheel); they check that unwrapping and unwarping behave correctly on a known field.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.niimath import FieldmapPrep, Fugue, RomeoUnwrap

needs_niimath = pytest.mark.skipif(
    shutil.which('niimath') is None, reason='niimath binary is not installed'
)


def _write(path, data, zooms=(2.0, 2.0, 2.0)):
    nb.Nifti1Image(data.astype('float32'), np.diag([*zooms, 1.0])).to_filename(str(path))
    return str(path)


def _run(interface, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    return interface.run(cwd=str(work_dir))


# --------------------------------------------------------------- cmdline shape


def test_romeo_cmdline(tmp_path):
    """romeo runs first, keeps radian phase, takes the mask via -k, output last."""
    phase = _write(tmp_path / 'phase.nii.gz', np.zeros((4, 4, 4)))
    mag = _write(tmp_path / 'mag.nii.gz', np.ones((4, 4, 4)))
    mask = _write(tmp_path / 'mask.nii.gz', np.ones((4, 4, 4)))

    cmd = RomeoUnwrap(phase_file=phase, magnitude_file=mag, mask_file=mask).cmdline

    assert cmd.startswith('niimath ')
    # phase is the input, then -romeo <mag>, then -k <mask> (romeo must be first op)
    assert f'{phase} -romeo {mag} -k {mask}' in cmd
    assert '-no-phase-rescale' in cmd
    assert '-no-mask-out' in cmd
    assert cmd.split()[-1].endswith('_unwrapped.nii.gz')


def test_romeo_cmdline_without_mask(tmp_path):
    """The mask is optional (romeo falls back to its robustmask default)."""
    phase = _write(tmp_path / 'phase.nii.gz', np.zeros((4, 4, 4)))
    mag = _write(tmp_path / 'mag.nii.gz', np.ones((4, 4, 4)))

    cmd = RomeoUnwrap(phase_file=phase, magnitude_file=mag).cmdline

    assert '-k' not in cmd.split()
    assert f'{phase} -romeo {mag}' in cmd


def test_fugue_cmdline(tmp_path):
    """fugue takes <fmap> <dwell> <dir> as consecutive positionals."""
    epi = _write(tmp_path / 'epi.nii.gz', np.ones((4, 4, 4)))
    fmap = _write(tmp_path / 'fmap.nii.gz', np.zeros((4, 4, 4)))

    cmd = Fugue(in_file=epi, fmap_file=fmap, dwell_time=0.00055, unwarp_direction='y-').cmdline

    assert f'{epi} -fugue {fmap} 0.00055 y-' in cmd
    assert cmd.split()[-1].endswith('_unwarped.nii.gz')


def test_fieldmapprep_cmdline(tmp_path):
    """fmapprep takes the magnitude and the echo-time difference (ms)."""
    phase = _write(tmp_path / 'phasediff.nii.gz', np.zeros((4, 4, 4)))
    mag = _write(tmp_path / 'mag.nii.gz', np.ones((4, 4, 4)))

    cmd = FieldmapPrep(phasediff_file=phase, magnitude_file=mag, delta_te=2.46).cmdline

    assert f'{phase} -fmapprep {mag} 2.46' in cmd
    assert '-no-debranch' not in cmd  # off by default
    assert cmd.split()[-1].endswith('_fieldmap.nii.gz')


# --------------------------------------------------------------- functional


@needs_niimath
def test_romeo_unwraps_phase(tmp_path):
    """romeo recovers a smooth phase that wraps past +-pi."""
    shape = (32, 34, 28)
    x = np.linspace(-1, 1, shape[0])[:, None, None]
    y = np.linspace(-1, 1, shape[1])[None, :, None]
    z = np.linspace(-1, 1, shape[2])[None, None, :]
    true_phase = 6.0 * x + 4.0 * y + 2.0 * z + 3.0 * (x**2 - y**2)
    true_phase = np.broadcast_to(true_phase, shape).astype('float32')
    wrapped = np.arctan2(np.sin(true_phase), np.cos(true_phase))
    mag = np.broadcast_to(np.exp(-(x**2 + y**2 + z**2)) * 1000, shape).astype('float32')
    mask = (np.broadcast_to(x**2 + y**2 + z**2, shape) < 0.85).astype('float32')

    result = _run(
        RomeoUnwrap(
            phase_file=_write(tmp_path / 'phase.nii.gz', wrapped),
            magnitude_file=_write(tmp_path / 'mag.nii.gz', mag),
            mask_file=_write(tmp_path / 'mask.nii.gz', mask),
        ),
        tmp_path / 'w',
    )
    unwrapped = nb.load(result.outputs.unwrapped_phase_file).get_fdata()

    m = mask > 0
    # unwrapping is defined up to a global 2*pi offset
    resid = (unwrapped[m] - unwrapped[m].mean()) - (true_phase[m] - true_phase[m].mean())
    assert np.sqrt(np.mean(resid**2)) < 1e-2


@needs_niimath
def test_fugue_applies_and_preserves_grid(tmp_path):
    """fugue returns a same-grid image and actually shifts intensity."""
    shape = (24, 28, 20)
    y = np.linspace(-1, 1, shape[1])[None, :, None]
    epi = np.broadcast_to((1000 + 400 * np.sin(6 * y)), shape).astype('float32')
    field_rads = np.broadcast_to((2 * np.pi * 120 * y), shape).astype('float32')
    img = nb.load(_write(tmp_path / 'epi.nii.gz', epi))

    result = _run(
        Fugue(
            in_file=_write(tmp_path / 'epi.nii.gz', epi),
            fmap_file=_write(tmp_path / 'fmap.nii.gz', field_rads),
            dwell_time=0.0006,
            unwarp_direction='y',
        ),
        tmp_path / 'w',
    )
    out = nb.load(result.outputs.out_file)

    assert out.shape == img.shape
    assert np.allclose(out.affine, img.affine)
    # a 120 Hz field with this dwell/N produces a visible shift
    assert not np.allclose(out.get_fdata(), epi)
