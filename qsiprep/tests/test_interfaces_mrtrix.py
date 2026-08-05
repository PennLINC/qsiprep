"""Tests for the qsiprep.interfaces.mrtrix module."""

import os

import nibabel as nb
import pytest
from traits.trait_errors import TraitError

from qsiprep.interfaces import mrtrix


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
    assert noise_img.shape == in_img.shape[:3]
    assert noise_img.ndim == 3

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
