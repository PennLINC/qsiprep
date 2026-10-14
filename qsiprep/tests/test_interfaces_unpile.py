"""Tests for the unpile interfaces: grid preparation and the field-to-warp conversion."""

import json
import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces.base import traits

from qsiprep.interfaces.unpile import (
    PrepareUnpileInputs,
    UnpileFieldToWarp,
    UnpileFit,
    pe_dir_on_grid,
    read_unpile_score,
)


def _affine(axcodes, zooms=(3.0, 3.0, 3.0)):
    """Build an affine whose voxel axes point along ``axcodes`` (e.g. 'LAS')."""
    signs = {'L': (0, -1), 'R': (0, 1), 'P': (1, -1), 'A': (1, 1), 'I': (2, -1), 'S': (2, 1)}
    affine = np.zeros((4, 4))
    for axis, code in enumerate(axcodes):
        world, sign = signs[code]
        affine[world, axis] = sign * zooms[axis]
    affine[:3, 3] = (-60.0, -90.0, -50.0)
    affine[3, 3] = 1.0
    return affine


def _write(path, data, affine):
    nb.Nifti1Image(np.asarray(data, dtype=np.float32), affine).to_filename(str(path))
    return str(path)


@pytest.mark.parametrize(
    ('raw', 'raw_pe', 'grid', 'expected'),
    [
        ('RAS', 'j-', 'LAS', 'j-'),  # same j axis after the x flip
        ('RAS', 'j-', 'LPS', 'j'),  # j flipped: the direction flips with it
        ('LAS', 'i', 'RAS', 'i-'),
        ('RAS', 'j', 'RAS', 'j'),
    ],
)
def test_pe_dir_on_grid_follows_the_voxel_axes(tmp_path, raw, raw_pe, grid, expected):
    raw_file = _write(tmp_path / 'raw.nii.gz', np.zeros((4, 5, 6)), _affine(raw))
    grid_file = _write(tmp_path / 'grid.nii.gz', np.zeros((4, 5, 6)), _affine(grid))
    assert pe_dir_on_grid(raw_file, raw_pe, grid_file) == expected


def test_prepare_keeps_the_native_grid_without_grid_mm(tmp_path):
    affine = _affine('LAS')
    epi = _write(tmp_path / 'epi.nii.gz', np.random.default_rng(0).random((10, 12, 8)), affine)
    mask = _write(tmp_path / 'mask.nii.gz', np.ones((10, 12, 8)), affine)
    # A reference at the anatomical's own 1 mm voxels on the same field of view
    fine = _affine('LAS', zooms=(1.0, 1.0, 1.0))
    reference = _write(tmp_path / 't2w.nii.gz', np.ones((30, 36, 24)), fine)

    result = PrepareUnpileInputs(
        epi=epi, mask=mask, reference=reference, raw_dwi_file=epi, pe_dir='j-'
    ).run(cwd=str(tmp_path))
    out = result.outputs
    assert out.resampled is False
    assert out.pe_dir == 'j-'
    assert out.acq_voxel_mm == pytest.approx(3.0)
    for name in ('epi', 'mask', 'reference'):
        img = nb.load(getattr(out, name))
        assert img.shape == (10, 12, 8)
        assert np.allclose(img.affine, affine)


def test_prepare_resamples_everything_onto_the_fine_grid(tmp_path):
    affine = _affine('LAS')
    epi = _write(tmp_path / 'epi.nii.gz', np.random.default_rng(0).random((10, 12, 8)), affine)
    mask = _write(tmp_path / 'mask.nii.gz', np.ones((10, 12, 8)), affine)
    reference = _write(tmp_path / 't2w.nii.gz', np.ones((10, 12, 8)), affine)
    fieldmap = _write(tmp_path / 'fmap.nii.gz', np.full((10, 12, 8), 20.0), affine)

    result = PrepareUnpileInputs(
        epi=epi,
        mask=mask,
        reference=reference,
        fieldmap=fieldmap,
        raw_dwi_file=epi,
        pe_dir='j-',
        grid_mm=2.0,
    ).run(cwd=str(tmp_path))
    out = result.outputs
    assert out.resampled is True
    assert out.acq_voxel_mm == pytest.approx(3.0)
    assert out.pe_dir == 'j-'
    for name in ('epi', 'mask', 'reference', 'fieldmap'):
        img = nb.load(getattr(out, name))
        assert np.allclose(img.header.get_zooms()[:3], 2.0)
        assert img.shape == (15, 18, 12)
    assert set(np.unique(np.asarray(nb.load(out.mask).dataobj))) <= {0.0, 1.0}
    inner = np.asarray(nb.load(out.fieldmap).dataobj)[2:-2, 2:-2, 2:-2]
    assert np.allclose(inner, 20.0, atol=1e-3)


def test_prepare_fine_grid_follows_the_epi_voxel(tmp_path):
    affine = _affine('LAS')
    epi = _write(tmp_path / 'epi.nii.gz', np.ones((6, 6, 6)), affine)
    mask = _write(tmp_path / 'mask.nii.gz', np.ones((6, 6, 6)), affine)
    result = PrepareUnpileInputs(
        epi=epi, mask=mask, reference=epi, raw_dwi_file=epi, pe_dir='j', fine_grid=True
    ).run(cwd=str(tmp_path))
    assert result.outputs.resampled is True
    assert np.allclose(nb.load(result.outputs.epi).header.get_zooms()[:3], 2.0)
    assert result.outputs.acq_voxel_mm == pytest.approx(3.0)


def test_prepare_ignores_a_grid_no_finer_than_the_epi(tmp_path):
    affine = _affine('LAS')
    epi = _write(tmp_path / 'epi.nii.gz', np.ones((6, 6, 6)), affine)
    mask = _write(tmp_path / 'mask.nii.gz', np.ones((6, 6, 6)), affine)
    result = PrepareUnpileInputs(
        epi=epi, mask=mask, reference=epi, raw_dwi_file=epi, pe_dir='j', grid_mm=3.0
    ).run(cwd=str(tmp_path))
    assert result.outputs.resampled is False


def test_field_to_warp_matches_the_topup_convention(tmp_path):
    """Test that a constant field becomes a constant shift along the acquired PE axis.

    ``field_Hz * TotalReadoutTime`` voxels along the phase-encoding axis, carried to
    world space (LPS) by the reference affine: 10 Hz x 0.05 s = 0.5 voxel = 1.5 mm
    toward -j of an LAS grid, i.e. toward P, which is +y in LPS.
    """
    affine = _affine('LAS')
    reference = _write(tmp_path / 'b0.nii.gz', np.ones((6, 8, 6)), affine)
    field = _write(tmp_path / 'field.nii.gz', np.full((6, 8, 6), 10.0), affine)

    result = UnpileFieldToWarp(
        field_hz=field, reference=reference, raw_dwi_file=reference, pe_dir='j-', readout_time=0.05
    ).run(cwd=str(tmp_path))
    out = result.outputs
    assert out.pe_axis == 1
    warp = nb.load(out.out_warp)
    assert warp.shape == (6, 8, 6, 1, 3)
    vectors = np.asarray(warp.dataobj).reshape(-1, 3)
    assert np.allclose(vectors, [0.0, 1.5, 0.0], atol=1e-6)
    assert np.allclose(np.asarray(nb.load(out.out_hz).dataobj), 10.0)


def test_field_to_warp_resamples_a_fine_field_onto_the_reference(tmp_path):
    affine = _affine('LAS')
    reference = _write(tmp_path / 'b0.nii.gz', np.ones((6, 8, 6)), affine)
    fine = _affine('LAS', zooms=(1.5, 1.5, 1.5))
    field = _write(tmp_path / 'field.nii.gz', np.full((12, 16, 12), 10.0), fine)

    result = UnpileFieldToWarp(
        field_hz=field, reference=reference, raw_dwi_file=reference, pe_dir='j', readout_time=0.05
    ).run(cwd=str(tmp_path))
    out_hz = nb.load(result.outputs.out_hz)
    assert out_hz.shape == (6, 8, 6)
    assert np.allclose(out_hz.affine, affine)
    assert np.allclose(np.asarray(out_hz.dataobj)[1:-1, 1:-1, 1:-1], 10.0, atol=1e-3)
    vectors = np.asarray(nb.load(result.outputs.out_warp).dataobj).reshape(-1, 3)
    # +j of an LAS grid is toward A, which is -y in LPS
    assert np.allclose(np.median(vectors, axis=0), [0.0, -1.5, 0.0], atol=1e-3)


def test_fit_command_line(tmp_path):
    affine = _affine('LAS')
    epi = _write(tmp_path / 'epi.nii.gz', np.ones((6, 8, 6)), affine)
    iface = UnpileFit(
        epi=epi,
        mask=epi,
        reference=epi,
        pe_dir='j-',
        total_readout_time=0.05,
        metric='lncc',
        pose=True,
        lam_mm=60,
        acq_voxel_mm=3,
        threads=4,
        fieldmap=epi,
        fieldmap_refine='pose,shim',
        levels='24,12,6',
        iters=20,
        reference_init_xfm=epi,
    )
    cmd = iface.cmdline
    assert cmd.startswith('unpile fit')
    for flag in (
        '--pe-dir j-',
        '--total-readout-time 0.05',
        '--metric lncc',
        '--pose',
        '--lam-mm 60',
        '--acq-voxel-mm 3',
        '--threads 4',
        '--fieldmap-refine pose,shim',
        '--levels 24,12,6',
        '--iters 20',
        f'--reference-init-xfm {epi}',
        '--out unpile',
    ):
        assert flag in cmd, flag
    outputs = iface._list_outputs()
    assert outputs['sdc_warp'].endswith('unpile/sdc_warp.nii.gz')
    assert outputs['field_hz'].endswith('unpile/field_hz.nii.gz')
    # optional outputs are only listed once the binary wrote them
    for key in ('pose_xfm', 'pose_composed_xfm', 'reference_posed', 'fieldmap_pose_xfm'):
        assert key not in outputs or outputs[key] is traits.Undefined


def test_read_unpile_score_drops_the_argument_echo(tmp_path):
    path = tmp_path / 'score.json'
    path.write_text(json.dumps({'args': ['fit'], 'field': {'corr': 0.9}, 'norm': 1.0}))
    assert read_unpile_score(str(path)) == {'field': {'corr': 0.9}, 'norm': 1.0}


@pytest.mark.skipif(shutil.which('unpile') is None, reason='the unpile binary is not installed')
def test_field_to_warp_reproduces_the_binary_warp(tmp_path):
    """Test that the Hz-to-warp conversion equals unpile's own ``sdc_warp.nii.gz``.

    A synthetic 'EPI' and a shifted copy of it as the reference: the fit's field
    is irrelevant here, only that the two warps built from it agree.
    """
    rng = np.random.default_rng(1)
    affine = _affine('LAS')
    shape = (16, 20, 12)
    blob = np.zeros(shape, dtype=np.float32)
    x, y, z = np.mgrid[: shape[0], : shape[1], : shape[2]]
    blob[(x - 8) ** 2 + (y - 10) ** 2 + (z - 6) ** 2 < 25] = 1.0
    epi = _write(tmp_path / 'epi.nii.gz', blob + 0.05 * rng.random(shape), affine)
    reference = _write(tmp_path / 'ref.nii.gz', np.roll(blob, 1, axis=1), affine)
    mask = _write(tmp_path / 'mask.nii.gz', np.ones(shape), affine)
    fit = UnpileFit(
        epi=epi,
        mask=mask,
        reference=reference,
        pe_dir='j-',
        total_readout_time=0.05,
        levels='24',
        iters=3,
        threads=1,
        out_dir=str(tmp_path / 'fit'),
    ).run(cwd=str(tmp_path))
    converted = UnpileFieldToWarp(
        field_hz=fit.outputs.field_hz,
        reference=epi,
        raw_dwi_file=epi,
        pe_dir='j-',
        readout_time=0.05,
    ).run(cwd=str(tmp_path))
    mine = np.asarray(nb.load(converted.outputs.out_warp).dataobj)
    theirs = np.asarray(nb.load(fit.outputs.sdc_warp).dataobj)
    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine, theirs, atol=1e-5)
