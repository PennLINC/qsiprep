"""Tests for the svht_denoise interfaces."""

import nibabel as nb
import numpy as np

from qsiprep.interfaces.svht import SVHTDeGibbs, SVHTDenoise


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


def test_svht_degibbs_cmdline(tmp_path):
    in_file = _write_image(tmp_path / 'dwi.nii.gz')

    degibbs = SVHTDeGibbs(in_file=in_file, nthreads=2)
    # Unringing only: none of the denoising outputs are requested
    assert degibbs.cmdline == f'svht_denoise {in_file} dwi_unrung.nii.gz -degibbs o -nthreads 2'

    for factor in (0.875, 0.75):
        degibbs = SVHTDeGibbs(in_file=in_file, partial_fourier=factor)
        assert degibbs.cmdline.endswith(f'-degibbs o -pF {factor}')
