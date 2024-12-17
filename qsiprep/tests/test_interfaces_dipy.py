"""Tests for the qsiprep.interfaces.dipy module."""

import os

import nibabel as nb

from qsiprep.interfaces import dipy


def test_patch2self(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.dipy.Patch2Self."""
    tmpdir = tmp_path_factory.mktemp('test_patch2self')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    bval_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.bval')
    in_img = nb.load(in_file)

    interface = dipy.Patch2Self(
        in_file=in_file,
        bval_file=bval_file,
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
