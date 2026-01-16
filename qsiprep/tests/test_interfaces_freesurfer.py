"""Tests for the qsiprep.interfaces.freesurfer module."""

import os
import shutil

import nibabel as nb
import numpy as np
import pytest
from nibabel.processing import resample_from_to

from qsiprep.interfaces import freesurfer


def _get_forrest_gump_t1w(datasets):
    in_dir = datasets['forrest_gump']
    return os.path.join(
        in_dir,
        'sub-01',
        'ses-forrestgump',
        'anat',
        'sub-01_ses-forrestgump_T1w.nii.gz',
    )


def _resample_to_64_cube(in_file, tmpdir):
    in_img = nb.load(in_file)
    new_shape = (64, 64, 64)

    direction = in_img.affine[:3, :3]
    voxel_sizes = np.sqrt((direction**2).sum(axis=0))
    direction_unit = direction / voxel_sizes
    new_voxel_sizes = voxel_sizes * (np.array(in_img.shape[:3]) / np.array(new_shape))

    new_affine = in_img.affine.copy()
    new_affine[:3, :3] = direction_unit * new_voxel_sizes

    resampled = resample_from_to(in_img, (new_shape, new_affine))
    out_file = os.path.join(tmpdir, 'sub-01_ses-forrestgump_T1w_64cube.nii.gz')
    resampled.to_filename(out_file)
    return out_file


def _gpu_available():
    return shutil.which('nvidia-smi') is not None


@pytest.mark.synthstrip
def test_synthstrip_interface(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.freesurfer.FixHeaderSynthStrip."""
    tmpdir = tmp_path_factory.mktemp('test_synthstrip')
    in_file = _resample_to_64_cube(_get_forrest_gump_t1w(datasets), tmpdir)
    in_img = nb.load(in_file)

    use_gpu = _gpu_available()
    interface = freesurfer.FixHeaderSynthStrip(input_image=in_file, gpu=use_gpu)
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_brain)
    assert os.path.isfile(results.outputs.out_brain_mask)

    out_img = nb.load(results.outputs.out_brain)
    mask_img = nb.load(results.outputs.out_brain_mask)
    assert out_img.shape == in_img.shape
    assert mask_img.shape == in_img.shape


@pytest.mark.synthseg
def test_synthseg_interface(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.freesurfer.SynthSeg."""
    tmpdir = tmp_path_factory.mktemp('test_synthseg')
    in_file = _resample_to_64_cube(_get_forrest_gump_t1w(datasets), tmpdir)

    use_gpu = _gpu_available()
    interface = freesurfer.SynthSeg(input_image=in_file, cpu=not use_gpu)
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_seg)
    assert os.path.isfile(results.outputs.out_post)
    assert os.path.isfile(results.outputs.out_qc)

    seg_img = nb.load(results.outputs.out_seg)
    post_img = nb.load(results.outputs.out_post)
    assert seg_img.shape == post_img.shape[:3]
