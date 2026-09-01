"""Tests for the qsiprep.interfaces.mrtrix module."""

import os

import nibabel as nb
import numpy as np
import pytest
from traits.trait_errors import TraitError

from qsiprep.interfaces import mrtrix
from qsiprep.tests.utils import field_of_view


def test_dwidenoise(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.mrtrix.DWIDenoise."""
    tmpdir = tmp_path_factory.mktemp('test_dwidenoise')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    in_img = nb.load(in_file)

    interface = mrtrix.DWIDenoise(
        extent=(5, 5, 5),
        in_file=in_file,
        nthreads=1,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_file)
    denoised_img = nb.load(results.outputs.out_file)
    assert denoised_img.shape == in_img.shape

    assert os.path.isfile(results.outputs.noise_image)
    noise_img = nb.load(results.outputs.noise_image)
    assert noise_img.shape == in_img.shape[:3]
    assert noise_img.ndim == 3

    assert os.path.isfile(results.outputs.out_report)
    assert os.path.isfile(results.outputs.nmse_text)


def test_dwidenoise2(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.mrtrix.DWIDenoise2."""
    tmpdir = tmp_path_factory.mktemp('test_dwidenoise2')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    in_img = nb.load(in_file)

    interface = mrtrix.DWIDenoise2(
        in_file=in_file,
        nthreads=1,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_file)
    denoised_img = nb.load(results.outputs.out_file)
    assert denoised_img.shape == in_img.shape

    assert os.path.isfile(results.outputs.noise_image)
    noise_img = nb.load(results.outputs.noise_image)
    assert noise_img.ndim == 3
    # The schedule estimates noise on a subsampled grid, so the noise map is coarser
    # than the input but must cover the input's field of view.
    assert all(n <= i for n, i in zip(noise_img.shape, in_img.shape[:3], strict=True))
    overshoot = field_of_view(noise_img) - field_of_view(in_img)
    assert np.all(overshoot > -1e-4)
    assert np.all(overshoot < np.array(noise_img.header.get_zooms()[:3]))

    assert os.path.isfile(results.outputs.out_report)
    assert os.path.isfile(results.outputs.nmse_text)


@pytest.mark.parametrize(
    'kernel_option',
    ['shape', 'radius', 'extent', 'aspect_ratio', 'minvoxels', 'subsample', 'onepass'],
)
def test_dwidenoise2_has_no_kernel_options(tmp_path, kernel_option):
    """The kernel and subsampling come from the schedule, not from command-line options."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    with pytest.raises(TraitError, match='undefined'):
        mrtrix.DWIDenoise2(in_file=in_file, **{kernel_option: 1})


def test_dwidenoise2_passes_schedule(tmp_path):
    """Select a bundled noise estimation schedule by name."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    interface = mrtrix.DWIDenoise2(in_file=in_file, schedule='vlarge')

    assert '-schedule vlarge' in interface.cmdline


def test_dwidenoise2_formats_fslgrad(tmp_path):
    """Pass the bvec and bval files to dwidenoise2 as a single -fslgrad option."""
    in_file = tmp_path / 'dwi.nii.gz'
    bvec_file = tmp_path / 'dwi.bvec'
    bval_file = tmp_path / 'dwi.bval'
    for path in (in_file, bvec_file, bval_file):
        path.touch()

    interface = mrtrix.DWIDenoise2(
        in_file=in_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
    )

    assert f'-fslgrad {bvec_file} {bval_file}' in interface.cmdline


def test_dwibiascorrect_uses_underscore_ants_options(tmp_path):
    """Pass N4 options as -ants_b/-ants_c/-ants_s.

    MRtrix3's development branch renamed these from the dot-separated -ants.b form
    used by 3.0.x. Getting this wrong fails every bias-correction node at runtime,
    so the exact spelling is pinned here.
    """
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    interface = mrtrix.DWIBiasCorrect(method='ants', in_file=in_file, ants_s='4')
    cmdline = interface.cmdline
    assert '-ants_b [150,3]' in cmdline
    assert '-ants_c [200x200,1e-6]' in cmdline
    assert '-ants_s 4' in cmdline
    assert '-ants.' not in cmdline
