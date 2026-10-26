"""Tests for the svht_denoise interfaces."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.svht import SVHTDeGibbs, SVHTDenoise, _count_demeaned_shells


def _write_image(path):
    nb.Nifti1Image(np.ones((4, 4, 4, 3), dtype=np.float32), np.eye(4)).to_filename(str(path))
    return str(path)


def test_svht_denoise_cmdline(tmp_path):
    in_file = _write_image(tmp_path / 'dwi.nii.gz')
    phase_file = _write_image(tmp_path / 'phase.nii.gz')

    denoise = SVHTDenoise(in_file=in_file, nthreads=2)
    assert denoise.cmdline == (
        f'svht_denoise {in_file} dwi_denoised.nii.gz -noise dwi_noise.nii.gz -nthreads 2'
    )

    denoise = SVHTDenoise(in_file=in_file, phase_file=phase_file, phase_units='radians', extent=7)
    cmdline = denoise.cmdline.split()
    assert cmdline[:3] == ['svht_denoise', in_file, 'dwi_denoised.nii.gz']
    assert {'-phase', phase_file, '-phaseunits', 'radians', '-extent', '7'} <= set(cmdline)


def _write_bvals(path, bvals):
    path.write_text(' '.join(str(b) for b in bvals) + '\n')
    return str(path)


def _options(cmdline):
    """Split a command line into its option tokens, after the input and output."""
    return cmdline.split()[3:]


def test_svht_denoise_cmdline_config_options(tmp_path):
    in_file = _write_image(tmp_path / 'dwi.nii.gz')
    bval_file = _write_bvals(tmp_path / 'dwi.bval', [0, 1000, 1000, 1000, 2000, 2000])

    denoise = SVHTDenoise(
        in_file=in_file,
        demean=True,
        bval_file=bval_file,
        filter_method='optthresh',
        shape='cube',
        aggregator='gaussian',
        aggregator_fwhm=1.5,
        stride=1,
        vst=True,
        noise_dof=4,
        preserve_noise_bias=True,
    )
    cmdline = ' '.join(_options(denoise.cmdline))
    for option in (
        '-demean y',
        f'-bval {bval_file}',
        '-filter optthresh',
        '-shape cube',
        '-aggregator gaussian',
        '-aggregator_fwhm 1.5',
        '-stride 1',
        '-vst y',
        '-noise_dof 4',
        '-preserve_noise_bias',
    ):
        assert option in cmdline

    # svht_denoise spells false as n, and a false bare flag is left out
    denoise = SVHTDenoise(in_file=in_file, demean=False, vst=False, preserve_noise_bias=False)
    cmdline = ' '.join(_options(denoise.cmdline))
    assert '-demean n' in cmdline
    assert '-vst n' in cmdline
    assert '-preserve_noise_bias' not in cmdline


@pytest.mark.parametrize(
    ('bvals', 'demeaned_shells'),
    [
        ([0, 1000, 1000, 2000, 2000], 2),
        # A shell of one volume is not demeaned
        ([0, 1000, 2000], 0),
        ([5, 0, 1000, 1000], 2),
        # Neighbours less than 80 s/mm^2 apart share a shell, even along a chain
        ([0, 995, 1000, 1005, 1080, 1085], 1),
        ([0, 0, 1000, 1005, 1100, 1105], 3),
        ([0, 0, 0], 1),
    ],
)
def test_count_demeaned_shells(bvals, demeaned_shells):
    assert _count_demeaned_shells(bvals) == demeaned_shells


def test_svht_denoise_skips_demean_that_leaves_too_few_volumes(tmp_path):
    """Test that demeaning is dropped where svht_denoise could not then estimate the noise.

    Two shells of two volumes leave two volumes after demeaning, and svht_denoise needs
    three to write a noise map. The same series denoises without demeaning.
    """
    in_file = _write_image(tmp_path / 'dwi.nii.gz')
    short = _write_bvals(tmp_path / 'short.bval', [0, 0, 1000, 1000])
    denoise = SVHTDenoise(in_file=in_file, demean=True, bval_file=short)
    assert '-demean' not in denoise.cmdline
    assert '-bval' not in denoise.cmdline

    enough = _write_bvals(tmp_path / 'enough.bval', [0, 1000, 1000, 2000, 2000])
    denoise = SVHTDenoise(in_file=in_file, demean=True, bval_file=enough)
    assert '-demean y' in denoise.cmdline
    assert f'-bval {enough}' in denoise.cmdline


def test_svht_degibbs_cmdline(tmp_path):
    in_file = _write_image(tmp_path / 'dwi.nii.gz')

    degibbs = SVHTDeGibbs(in_file=in_file, nthreads=2)
    # Unringing only: none of the denoising outputs are requested
    assert degibbs.cmdline == f'svht_denoise {in_file} dwi_unrung.nii.gz -degibbs o -nthreads 2'

    for factor in (0.875, 0.75):
        degibbs = SVHTDeGibbs(in_file=in_file, partial_fourier=factor)
        assert degibbs.cmdline.endswith(f'-degibbs o -pF {factor}')
