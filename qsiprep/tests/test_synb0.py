"""Tests for SynB0 synthetic b=0 generation: interfaces and workflow wiring.

The U-Net itself is exercised only in the containers (it needs the torch
environment and the SynB0 distribution); here the ``Synb0Inference`` command
line and the surrounding numpy/graph logic are covered.
"""

import sys

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.freesurfer import SynthStrip, torch_script_command
from qsiprep.interfaces.synb0 import (
    NormalizeForSynb0,
    Synb0Inference,
    get_synb0_atlas,
    get_synb0_dir,
)
from qsiprep.workflows.fieldmap.synb0 import init_synb0_wf

WM_LABEL = 3


def _write_t1_and_dseg(tmp_path, wm_mean=480.0):
    rng = np.random.default_rng(seed=0)
    shape = (12, 12, 12)
    dseg = np.zeros(shape, dtype=np.int16)
    dseg[2:10, 2:10, 2:5] = 1  # CSF
    dseg[2:10, 2:10, 5:8] = 2  # GM
    dseg[2:10, 2:10, 8:10] = WM_LABEL
    t1 = np.zeros(shape, dtype=np.float32)
    t1[dseg == 1] = rng.normal(120.0, 5.0, size=(dseg == 1).sum())
    t1[dseg == 2] = rng.normal(300.0, 5.0, size=(dseg == 2).sum())
    t1[dseg == WM_LABEL] = rng.normal(wm_mean, 5.0, size=(dseg == WM_LABEL).sum())

    t1_file = str(tmp_path / 't1.nii.gz')
    dseg_file = str(tmp_path / 'dseg.nii.gz')
    nb.Nifti1Image(t1, np.eye(4)).to_filename(t1_file)
    nb.Nifti1Image(dseg, np.eye(4)).to_filename(dseg_file)
    return t1_file, dseg_file


def test_normalize_scales_wm_median_to_110(tmp_path):
    t1_file, dseg_file = _write_t1_and_dseg(tmp_path)

    result = NormalizeForSynb0(t1w_file=t1_file, dseg_file=dseg_file).run(cwd=str(tmp_path))

    out = nb.load(result.outputs.out_file).get_fdata()
    dseg = nb.load(dseg_file).get_fdata()
    assert np.isclose(np.median(out[dseg == WM_LABEL]), 110.0, atol=0.01)
    assert out.min() >= 0
    assert out.max() <= 255


def test_normalize_clips_to_uint8_range(tmp_path):
    # A WM median far below other tissue drives those intensities past 255
    t1_file, dseg_file = _write_t1_and_dseg(tmp_path, wm_mean=100.0)

    result = NormalizeForSynb0(t1w_file=t1_file, dseg_file=dseg_file).run(cwd=str(tmp_path))

    out = nb.load(result.outputs.out_file).get_fdata()
    assert out.max() == 255


def test_normalize_requires_wm_voxels(tmp_path):
    t1_file, dseg_file = _write_t1_and_dseg(tmp_path)

    interface = NormalizeForSynb0(t1w_file=t1_file, dseg_file=dseg_file, wm_label=9)
    with pytest.raises(ValueError, match='label 9'):
        interface.run(cwd=str(tmp_path))


def _synb0_distribution(tmp_path):
    """A fake SynB0 layout: atlases/ with the 2.5mm grid, model weights."""
    synb0_dir = tmp_path / 'synb0'
    (synb0_dir / 'atlases').mkdir(parents=True)
    (synb0_dir / 'dual_channel_unet').mkdir()
    atlas = synb0_dir / 'atlases' / 'mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)).to_filename(str(atlas))
    return synb0_dir, atlas


def test_synb0_locations_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('SYNB0_ATLASES', raising=False)
    assert get_synb0_dir() is None
    assert get_synb0_atlas() is None

    synb0_dir, atlas = _synb0_distribution(tmp_path)
    monkeypatch.setenv('SYNB0_ATLASES', str(synb0_dir / 'atlases'))
    assert get_synb0_dir() == str(synb0_dir)
    assert get_synb0_atlas() == str(atlas)


def test_inference_cmdline_uses_torch_python(tmp_path, monkeypatch):
    synb0_dir, atlas = _synb0_distribution(tmp_path)
    inputs = {'t1_file': str(atlas), 'b0_file': str(atlas), 'synb0_dir': str(synb0_dir)}

    # The command is fixed at construction so check_deps' which() probe on
    # _cmd.split()[0] sees the real interpreter
    monkeypatch.setenv('QSIPREP_TORCH_PYTHON', '/torchenv/bin/python')
    cmdline = Synb0Inference(**inputs).cmdline
    assert cmdline.startswith('/torchenv/bin/python ')
    assert 'synb0_runner.py' in cmdline
    assert f'--synb0-dir {synb0_dir}' in cmdline
    assert '--out b0_u_atlas.nii.gz' in cmdline

    # Without the variable the current interpreter runs the script
    monkeypatch.delenv('QSIPREP_TORCH_PYTHON')
    assert Synb0Inference(**inputs).cmdline.startswith(f'{sys.executable} ')


def test_synthstrip_cmd_uses_torch_python(tmp_path, monkeypatch):
    script = tmp_path / 'mri_synthstrip'
    script.write_text('#!/usr/bin/env python\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path), prepend=':')

    monkeypatch.setenv('QSIPREP_TORCH_PYTHON', '/torchenv/bin/python')
    assert SynthStrip().cmd == f'/torchenv/bin/python {script}'

    monkeypatch.delenv('QSIPREP_TORCH_PYTHON')
    assert SynthStrip().cmd == 'mri_synthstrip'


def test_torch_script_command_missing_script(monkeypatch, tmp_path):
    # An unresolvable script falls back to the bare name rather than a
    # half-built command line
    monkeypatch.setenv('PATH', str(tmp_path))
    monkeypatch.setenv('QSIPREP_TORCH_PYTHON', '/torchenv/bin/python')
    assert torch_script_command('mri_synthstrip') == 'mri_synthstrip'


def test_synb0_boilerplate_reflects_ignored_fieldmaps(monkeypatch):
    from qsiprep import config
    from qsiprep.interfaces.eddy import topup_boilerplate

    monkeypatch.setattr(config.workflow, 'ignore', [])
    assert 'No fieldmap was available' in topup_boilerplate('synb0', 'TOPUP')

    # Reaching SyNb0 with fieldmaps or RPE series present means the user
    # ignored them; the text must say so instead of claiming absence
    monkeypatch.setattr(config.workflow, 'ignore', ['fieldmaps', 'pepolar-dwis', 't2w'])
    text = topup_boilerplate('synb0', 'TOPUP')
    assert 'No fieldmap was available' not in text
    assert "excluded at the user's request" in text
    assert '--ignore fieldmaps pepolar-dwis' in text


def test_synb0_wf_builds_without_container(monkeypatch):
    monkeypatch.delenv('SYNB0_ATLASES', raising=False)

    wf = init_synb0_wf()

    for node_name in [
        'normalize_t1',
        'resample_t1_to_atlas',
        'resample_b0_to_atlas',
        'unet',
        'resample_to_native',
        'map_dseg_to_b0',
        'extract_wm',
        'acquired_synthetic_rpt',
        'unet_input_rpt',
    ]:
        assert wf.get_node(node_name) is not None
    assert wf.get_node('distorted_b0_coreg_wf.b0_to_anat') is not None
    # Atlas grid and model location stay unset outside the containers
    assert not wf.get_node('inputnode').inputs.atlas_image
    # The native-grid resample inverts the full linear chain on the fly
    assert wf.get_node('resample_to_native').inputs.invert_transform_flags == [True, True, True]


def test_synb0_wf_prefills_container_locations(tmp_path, monkeypatch):
    synb0_dir, atlas = _synb0_distribution(tmp_path)
    monkeypatch.setenv('SYNB0_ATLASES', str(synb0_dir / 'atlases'))

    wf = init_synb0_wf()

    assert wf.get_node('inputnode').inputs.atlas_image == str(atlas)
    assert wf.get_node('unet').inputs.synb0_dir == str(synb0_dir)
