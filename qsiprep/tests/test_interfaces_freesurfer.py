"""Tests for the qsiprep.interfaces.freesurfer module."""

import os

import nibabel as nb
import pytest

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


@pytest.mark.synthstrip
def test_synthstrip_interface(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.freesurfer.FixHeaderSynthStrip."""
    tmpdir = tmp_path_factory.mktemp('test_synthstrip')
    in_file = _get_forrest_gump_t1w(datasets)

    assert os.path.isfile(in_file)
    in_img = nb.load(in_file)

    interface = freesurfer.FixHeaderSynthStrip(input_image=in_file)
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
    in_file = _get_forrest_gump_t1w(datasets)

    assert os.path.isfile(in_file)
    in_img = nb.load(in_file)

    interface = freesurfer.SynthSeg(input_image=in_file)
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_seg)
    assert os.path.isfile(results.outputs.out_post)
    assert os.path.isfile(results.outputs.out_qc)

    seg_img = nb.load(results.outputs.out_seg)
    post_img = nb.load(results.outputs.out_post)
    assert seg_img.shape == in_img.shape
    assert post_img.shape == in_img.shape
