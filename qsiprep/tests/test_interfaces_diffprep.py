"""Tests for the TORTOISE DIFFPREP HMC + SDC backend.

Pure-Python behaviour -- command-line construction, Okan transform parsing and
workflow wiring -- is tested unconditionally.

Tests that exercise the real TORTOISE binaries are guarded with
``shutil.which`` and skip when those binaries are absent. They are *not*
permanently offline: CircleCI's ``unit_tests`` job runs pytest inside the
``pennlinc/qsiprep:test`` image, which ships the TORTOISE tools, so these
assertions run for real in CI. Full end-to-end runs of the backend live in
``test_cli.py`` behind the ``diffprep``/``diffprep_drbuddi`` integration markers.
"""

import os
import shutil

import numpy as np
import pytest
from nipype.interfaces.base import isdefined


def _require(*binaries):
    """Skip unless every named TORTOISE binary is on PATH."""
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        pytest.skip(f'{", ".join(missing)} required for this test')


def _write_dummy_nii(path, nvols=6):
    import nibabel as nb

    img = nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype='float32'), np.eye(4))
    img.to_filename(str(path))


def _write_fsl_gradients(tmp_path, bvals, bvecs, stem='grad'):
    """Write FSL-style .bval/.bvec files and return their paths."""
    bval_file = tmp_path / f'{stem}.bval'
    bvec_file = tmp_path / f'{stem}.bvec'
    bval_file.write_text(' '.join(str(b) for b in bvals) + '\n')
    bvec_file.write_text('\n'.join(' '.join(f'{v:.8f}' for v in row) for row in bvecs) + '\n')
    return bval_file, bvec_file


def _diffprep_siblings(tmp_path):
    dwi = tmp_path / 'dwi.nii'
    _write_dummy_nii(dwi)
    (tmp_path / 'dwi.bmtxt').write_text('0 0 0 0 0 0\n1000 0 0 0 0 0\n')
    (tmp_path / 'dwi.json').write_text('{"PhaseEncodingDirection": "j"}')
    return dwi, tmp_path / 'dwi.bmtxt', tmp_path / 'dwi.json'


# ---------------------------------------------------------------------------
# Command-line construction (pure Python)
# ---------------------------------------------------------------------------


def test_diffprep_cmdline_off(tmp_path):
    """DIFFPREP with epi_mode='off' drives TORTOISEProcess from --step import
    with all extra stages (including EPI) disabled."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    iface = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        correction_mode='quadratic',
        epi_mode='off',
    )
    cmd = iface.cmdline
    assert cmd.startswith('TORTOISEProcess')
    assert '-u ' in cmd
    assert str(dwi) in cmd
    assert '-c quadratic' in cmd
    assert '--step import' in cmd
    assert '--denoising off' in cmd
    assert '--gibbs 0' in cmd
    assert '--drift off' in cmd
    assert '--epi off' in cmd
    # The bmtxt/json siblings are found by stem, never passed as argstrs.
    assert 'dwi.bmtxt' not in cmd
    assert 'dwi.json' not in cmd


def test_diffprep_cmdline_t2wreg(tmp_path):
    """DIFFPREP with epi_mode='T2Wreg' emits --epi T2Wreg -s <structural>."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    t2w = tmp_path / 't2w.nii'
    _write_dummy_nii(t2w, nvols=1)

    iface = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        correction_mode='motion',
        epi_mode='T2Wreg',
        structural_image=str(t2w),
    )
    cmd = iface.cmdline
    assert '-c motion' in cmd
    assert '--epi T2Wreg' in cmd
    assert f'-s {t2w}' in cmd
    assert '--epi off' not in cmd


def test_diffprep_t2wreg_requires_structural(tmp_path):
    """epi_mode='T2Wreg' without a structural image is an error."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    iface = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        correction_mode='quadratic',
        epi_mode='T2Wreg',
    )
    with pytest.raises(ValueError, match='requires a structural_image'):
        _ = iface.cmdline


# ---------------------------------------------------------------------------
# Okan transform parsing (pure Python)
# ---------------------------------------------------------------------------


def test_diffprep_motion_params_basic(tmp_path):
    """``DIFFPREPMotionParams`` slices cols 0-5 from a 24-col TORTOISE
    transformations file and writes them as a whitespace-separated SPM file."""
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    n_volumes = 4
    rng = np.random.default_rng(0)
    full = rng.standard_normal((n_volumes, 24))
    # Use the bracket / comma serialization VNL VariableLengthVector emits.
    txt = '\n'.join('[' + ', '.join(f'{v:.6f}' for v in row) + ']' for row in full) + '\n'
    transforms_file = tmp_path / 'sub-1_dwi_moteddy_transformations.txt'
    transforms_file.write_text(txt)

    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    res = iface.run(cwd=str(tmp_path))

    spm = np.loadtxt(res.outputs.spm_motion_file)
    assert spm.shape == (n_volumes, 6)
    np.testing.assert_allclose(spm, full[:, :6], atol=1e-5)


def test_diffprep_motion_params_plain_whitespace(tmp_path):
    """Some VNL serializers omit brackets and just space-separate values."""
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    full = np.arange(24, dtype=float).reshape(1, 24)
    txt = ' '.join(f'{v}' for v in full[0]) + '\n'
    transforms_file = tmp_path / 'plain.txt'
    transforms_file.write_text(txt)

    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    res = iface.run(cwd=str(tmp_path))

    spm = np.loadtxt(res.outputs.spm_motion_file)
    assert spm.shape == (6,)
    np.testing.assert_allclose(spm, full[0, :6])


def test_diffprep_motion_params_rejects_short_rows(tmp_path):
    """A transforms file with fewer than 24 columns is rejected."""
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    transforms_file = tmp_path / 'short.txt'
    transforms_file.write_text('0 0 0 0 0 0\n')
    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    with pytest.raises(ValueError, match='24 columns'):
        iface.run(cwd=str(tmp_path))


# ---------------------------------------------------------------------------
# Real TORTOISE binaries (run in the qsiprep test image)
# ---------------------------------------------------------------------------


def test_bmtxt_fsl_roundtrip(tmp_path):
    """FSL gradients -> TORTOISE bmtxt -> FSL gradients must round-trip.

    This is the assertion the DIFFPREP backend depends on: ``DIFFPREPSplitOutputs``
    recovers bvals/bvecs from TORTOISE's rotated b-matrix via
    ``TORTOISEBmatrixToFSLBVecs``. Exercising both real binaries is what makes
    the gradient recovery trustworthy.
    """
    _require('FSLBVecsToTORTOISEBmatrix', 'TORTOISEBmatrixToFSLBVecs')
    from qsiprep.interfaces.tortoise import bmtxt_to_fsl, make_bmat_file

    bvals = [0, 1000, 1000, 2000]
    bvecs = [
        [0.0, 1.0, 0.0, np.sqrt(0.5)],  # x
        [0.0, 0.0, 1.0, np.sqrt(0.5)],  # y
        [0.0, 0.0, 0.0, 0.0],  # z
    ]
    bval_file, bvec_file = _write_fsl_gradients(tmp_path, bvals, bvecs)

    bmtxt = make_bmat_file(str(bval_file), str(bvec_file))
    assert os.path.exists(bmtxt)

    out_bval, out_bvec = bmtxt_to_fsl(bmtxt, str(tmp_path))
    rt_bvals = np.loadtxt(out_bval).reshape(-1)
    rt_bvecs = np.atleast_2d(np.loadtxt(out_bvec))
    # Accept either FSL layout (3 x N) or its transpose, so this asserts the
    # gradient values rather than the tool's row/column convention.
    if rt_bvecs.shape[0] != 3:
        rt_bvecs = rt_bvecs.T
    assert rt_bvecs.shape == (3, len(bvals))

    np.testing.assert_allclose(rt_bvals, bvals, atol=1.0)
    # Gradient sign is arbitrary; compare absolute directions.
    np.testing.assert_allclose(np.abs(rt_bvecs), np.abs(np.array(bvecs)), atol=1e-3)


def test_tortoise_convert_colocates_bmtxt(tmp_path):
    """TORTOISEConvert renames the DWI into cwd and co-locates a same-stemmed
    .bmtxt beside it, so TORTOISEProcess can pair them by basename."""
    _require('FSLBVecsToTORTOISEBmatrix')
    from qsiprep.interfaces.tortoise import TORTOISEConvert

    # The DWI stem deliberately differs from the gradient stem -- that mismatch
    # is exactly what the co-location fix addresses.
    dwi = tmp_path / 'sub-1_desc-preproc_dwi.nii.gz'
    _write_dummy_nii(dwi, nvols=2)
    bval_file, bvec_file = _write_fsl_gradients(
        tmp_path, [0, 1000], [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )

    run_dir = tmp_path / 'node'
    run_dir.mkdir()
    iface = TORTOISEConvert(dwi_file=str(dwi), bval_file=str(bval_file), bvec_file=str(bvec_file))
    res = iface.run(cwd=str(run_dir))

    out_dwi = res.outputs.dwi_file
    out_bmtxt = res.outputs.bmtxt_file
    assert out_dwi.endswith('.nii')
    assert os.path.splitext(out_dwi)[0] + '.bmtxt' == out_bmtxt
    assert os.path.exists(out_bmtxt)
    # One bmtxt row per volume
    assert len(np.loadtxt(out_bmtxt, ndmin=2)) == 2


def test_diffprep_split_outputs(tmp_path):
    """``DIFFPREPSplitOutputs`` splits the corrected 4D DWI + bmtxt into
    per-volume triples, finds the b=0s, and emits identity ITK affines."""
    _require('FSLBVecsToTORTOISEBmatrix', 'TORTOISEBmatrixToFSLBVecs')
    from qsiprep.interfaces.tortoise import DIFFPREPSplitOutputs, make_bmat_file

    bvals = [0, 1000, 1000]
    bvecs = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    bval_file, bvec_file = _write_fsl_gradients(tmp_path, bvals, bvecs)
    bmtxt = make_bmat_file(str(bval_file), str(bvec_file))

    corrected = tmp_path / 'corrected.nii'
    _write_dummy_nii(corrected, nvols=3)

    run_dir = tmp_path / 'split'
    run_dir.mkdir()
    iface = DIFFPREPSplitOutputs(
        corrected_dwi_file=str(corrected),
        corrected_bmtxt_file=str(bmtxt),
        b0_threshold=100,
    )
    res = iface.run(cwd=str(run_dir))

    assert len(res.outputs.dwi_files) == 3
    assert len(res.outputs.bval_files) == 3
    assert len(res.outputs.bvec_files) == 3
    assert res.outputs.b0_indices == [0]
    # DIFFPREP bakes the correction into the voxels, so downstream transforms
    # must be no-ops.
    assert len(res.outputs.forward_transforms) == 3
    for xfm in res.outputs.forward_transforms:
        text = open(xfm).read()
        assert 'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0' in text


# ---------------------------------------------------------------------------
# rpe_series split / recombine (pure Python)
# ---------------------------------------------------------------------------


def _make_original_with_sidecar(tmp_path, name, pe_dir):
    """Write a tiny original nii + BIDS sidecar for get_distortion_grouping."""
    import json as _json

    import nibabel as nb

    nii = tmp_path / f'{name}.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype='float32'), np.eye(4)).to_filename(str(nii))
    (tmp_path / f'{name}.json').write_text(
        _json.dumps({'PhaseEncodingDirection': pe_dir, 'TotalReadoutTime': 0.05})
    )
    return str(nii)


def _make_4d(path, values):
    """4D nii where volume i is a constant image of ``values[i]``."""
    import nibabel as nb

    data = np.zeros((2, 2, 2, len(values)), dtype='float32')
    for i, val in enumerate(values):
        data[..., i] = val
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))


def test_split_dwis_by_distortion_group(tmp_path):
    """SplitDWIsByDistortionGroup partitions the merged series by PE group,
    labels the first-appearing group '+' and the second '-', and preserves
    per-volume order within each group."""
    import nibabel as nb

    from qsiprep.interfaces.tortoise import SplitDWIsByDistortionGroup

    ap = _make_original_with_sidecar(tmp_path, 'sub-01_dir-AP_dwi', 'j')
    pa = _make_original_with_sidecar(tmp_path, 'sub-01_dir-PA_dwi', 'j-')
    # Volume 0 is AP -> AP is group 1 ("+"); PA is group 2 ("-").
    original_files = [ap, ap, ap, pa, pa, pa]

    merged = tmp_path / 'merged.nii.gz'
    _make_4d(merged, [0, 1, 2, 3, 4, 5])
    bval_file, bvec_file = _write_fsl_gradients(
        tmp_path,
        [0, 1000, 2000, 0, 1000, 2000],
        [[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
    )

    run_dir = tmp_path / 'split'
    run_dir.mkdir()
    res = SplitDWIsByDistortionGroup(
        dwi_file=str(merged),
        bval_file=str(bval_file),
        bvec_file=str(bvec_file),
        original_files=original_files,
        pe_axis='j',
    ).run(cwd=str(run_dir))

    assert res.outputs.group_assignments == [1, 1, 1, 2, 2, 2]
    assert res.outputs.group1_pe_dir == 'j'
    assert res.outputs.group2_pe_dir == 'j-'

    g1 = nb.load(res.outputs.group1_dwi_file)
    g2 = nb.load(res.outputs.group2_dwi_file)
    assert g1.shape[3] == 3
    assert g2.shape[3] == 3
    np.testing.assert_array_equal([g1.dataobj[0, 0, 0, i] for i in range(3)], [0, 1, 2])
    np.testing.assert_array_equal([g2.dataobj[0, 0, 0, i] for i in range(3)], [3, 4, 5])

    np.testing.assert_array_equal(np.loadtxt(res.outputs.group1_bval_file), [0, 1000, 2000])
    np.testing.assert_array_equal(np.loadtxt(res.outputs.group2_bval_file), [0, 1000, 2000])


def test_split_dwis_by_distortion_group_rejects_single_group(tmp_path):
    """A series with only one PE group is not a reverse-PE series."""
    from qsiprep.interfaces.tortoise import SplitDWIsByDistortionGroup

    ap = _make_original_with_sidecar(tmp_path, 'sub-01_dir-AP_dwi', 'j')
    merged = tmp_path / 'merged.nii.gz'
    _make_4d(merged, [0, 1])
    bval_file, bvec_file = _write_fsl_gradients(tmp_path, [0, 1000], [[0, 1], [0, 0], [0, 0]])

    run_dir = tmp_path / 'split'
    run_dir.mkdir()
    with pytest.raises(ValueError, match='exactly two'):
        SplitDWIsByDistortionGroup(
            dwi_file=str(merged),
            bval_file=str(bval_file),
            bvec_file=str(bvec_file),
            original_files=[ap, ap],
            pe_axis='j',
        ).run(cwd=str(run_dir))


def test_concatenate_diffprep_groups_preserves_original_order(tmp_path):
    """ConcatenateDIFFPREPGroups reconstructs the original (merged) volume order
    from two per-direction DIFFPREP outputs, even when groups interleave."""
    import nibabel as nb

    from qsiprep.interfaces.tortoise import ConcatenateDIFFPREPGroups

    g1 = tmp_path / 'g1.nii.gz'
    g2 = tmp_path / 'g2.nii.gz'
    _make_4d(g1, [10, 11, 12])
    _make_4d(g2, [20, 21])

    (tmp_path / 'g1.bmtxt').write_text('\n'.join(f'{b} 0 0 0 0 0' for b in (0, 1000, 2000)) + '\n')
    (tmp_path / 'g2.bmtxt').write_text('\n'.join(f'{b} 0 0 0 0 0' for b in (0, 1000)) + '\n')

    def _xf(path, n, base):
        rows = [' '.join(str(base + i * 100 + j) for j in range(24)) for i in range(n)]
        path.write_text('\n'.join(rows) + '\n')

    _xf(tmp_path / 'g1_xf.txt', 3, 0)
    _xf(tmp_path / 'g2_xf.txt', 2, 10000)

    # Interleaved: positions 0,1,3 -> g1[0,1,2]; positions 2,4 -> g2[0,1].
    assignments = [1, 1, 2, 1, 2]

    run_dir = tmp_path / 'recombine'
    run_dir.mkdir()
    res = ConcatenateDIFFPREPGroups(
        group1_dwi_file=str(g1),
        group1_bmtxt_file=str(tmp_path / 'g1.bmtxt'),
        group1_transformations_file=str(tmp_path / 'g1_xf.txt'),
        group2_dwi_file=str(g2),
        group2_bmtxt_file=str(tmp_path / 'g2.bmtxt'),
        group2_transformations_file=str(tmp_path / 'g2_xf.txt'),
        group_assignments=assignments,
    ).run(cwd=str(run_dir))

    out = nb.load(res.outputs.corrected_dwi_file)
    assert out.shape[3] == 5
    np.testing.assert_array_equal(
        [out.dataobj[0, 0, 0, i] for i in range(5)], [10, 11, 20, 12, 21]
    )

    bmat = np.atleast_2d(np.loadtxt(res.outputs.corrected_bmtxt_file))
    assert bmat.shape == (5, 6)
    np.testing.assert_array_equal(bmat[:, 0], [0, 1000, 0, 2000, 1000])

    xf = np.atleast_2d(np.loadtxt(res.outputs.transformations_file))
    assert xf.shape == (5, 24)
    np.testing.assert_array_equal(xf[:, 0], [0, 100, 10000, 200, 10100])


def test_concatenate_diffprep_groups_rejects_count_mismatch(tmp_path):
    """A per-group output whose volume count disagrees with the assignments is
    a wiring bug and must fail loudly rather than silently drop volumes."""
    from qsiprep.interfaces.tortoise import ConcatenateDIFFPREPGroups

    g1 = tmp_path / 'g1.nii.gz'
    g2 = tmp_path / 'g2.nii.gz'
    _make_4d(g1, [10, 11, 12])
    _make_4d(g2, [20, 21])
    (tmp_path / 'g1.bmtxt').write_text('\n'.join('0 0 0 0 0 0' for _ in range(3)) + '\n')
    (tmp_path / 'g2.bmtxt').write_text('\n'.join('0 0 0 0 0 0' for _ in range(2)) + '\n')
    (tmp_path / 'g1_xf.txt').write_text('\n'.join(' '.join(['0'] * 24) for _ in range(3)) + '\n')
    (tmp_path / 'g2_xf.txt').write_text('\n'.join(' '.join(['0'] * 24) for _ in range(2)) + '\n')

    run_dir = tmp_path / 'recombine'
    run_dir.mkdir()
    with pytest.raises(ValueError, match='volume count'):
        ConcatenateDIFFPREPGroups(
            group1_dwi_file=str(g1),
            group1_bmtxt_file=str(tmp_path / 'g1.bmtxt'),
            group1_transformations_file=str(tmp_path / 'g1_xf.txt'),
            group2_dwi_file=str(g2),
            group2_bmtxt_file=str(tmp_path / 'g2.bmtxt'),
            group2_transformations_file=str(tmp_path / 'g2_xf.txt'),
            # Only 4 assignments for a 5-volume pair.
            group_assignments=[1, 1, 1, 2],
        ).run(cwd=str(run_dir))


def test_write_bmat_tortoise(tmp_path):
    """WriteBmatTORTOISE emits one 6-float B-matrix row per volume, zeros at b=0."""
    from qsiprep.interfaces.tortoise import WriteBmatTORTOISE

    bval_file, bvec_file = _write_fsl_gradients(
        tmp_path, [0, 1000], [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    run_dir = tmp_path / 'bmat'
    run_dir.mkdir()
    res = WriteBmatTORTOISE(bval_file=str(bval_file), bvec_file=str(bvec_file)).run(
        cwd=str(run_dir)
    )
    bmat = np.atleast_2d(np.loadtxt(res.outputs.bmat_file))
    assert bmat.shape == (2, 6)
    np.testing.assert_array_equal(bmat[0], [0, 0, 0, 0, 0, 0])
    # g = (1, 0, 0), b = 1000 -> Byy-position (index 3) holds b*gx*gx = 1000.
    assert bmat[1, 0] == 1000


def test_merge_volumes_4d(tmp_path):
    """MergeVolumes4D stacks b0 + predicted volumes in order."""
    import nibabel as nb

    from qsiprep.interfaces.tortoise import MergeVolumes4D

    b0 = tmp_path / 'b0.nii.gz'
    nb.Nifti1Image(np.full((2, 2, 2), 5.0, dtype='float32'), np.eye(4)).to_filename(str(b0))
    preds = []
    for i, val in enumerate((7.0, 9.0)):
        p = tmp_path / f'pred{i}.nii.gz'
        nb.Nifti1Image(np.full((2, 2, 2), val, dtype='float32'), np.eye(4)).to_filename(str(p))
        preds.append(str(p))

    run_dir = tmp_path / 'merge'
    run_dir.mkdir()
    res = MergeVolumes4D(b0_image=str(b0), predicted_images=preds).run(cwd=str(run_dir))
    out = nb.load(res.outputs.merged_4d)
    assert out.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal([out.dataobj[0, 0, 0, i] for i in range(3)], [5, 7, 9])


def test_write_fsl_grad_files(tmp_path):
    """WriteFSLGradFiles writes (3, N) bvecs and a single-row bval."""
    from qsiprep.interfaces.tortoise import WriteFSLGradFiles

    bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bvals = np.array([0.0, 1000.0, 1000.0])
    run_dir = tmp_path / 'grad'
    run_dir.mkdir()
    res = WriteFSLGradFiles(bvecs=bvecs, bvals=bvals).run(cwd=str(run_dir))
    out_bvec = np.loadtxt(res.outputs.bvec_file)
    out_bval = np.loadtxt(res.outputs.bval_file)
    assert out_bvec.shape == (3, 3)
    np.testing.assert_allclose(out_bvec, bvecs.T)
    np.testing.assert_array_equal(out_bval, bvals)


def test_equally_distributed_directions_deterministic():
    """The prediction target set is fixed and shaped [1 b=0 + n b=bval]."""
    from qsiprep.interfaces.tortoise import equally_distributed_directions

    bvecs, bvals = equally_distributed_directions(n=8, bval=1000.0)
    assert bvecs.shape == (9, 3)
    assert bvals.shape == (9,)
    np.testing.assert_array_equal(bvecs[0], [0, 0, 0])
    assert bvals[0] == 0
    np.testing.assert_array_equal(bvals[1:], [1000.0] * 8)
    # Deterministic across calls (same seed).
    bvecs2, _ = equally_distributed_directions(n=8, bval=1000.0)
    np.testing.assert_array_equal(bvecs, bvecs2)


def test_stage_drbuddi_pair_distinct_uncompressed_stems(tmp_path):
    """StageDRBUDDIPair writes distinct-stemmed, decompressed blip_up/blip_down
    with matched sibling bmtxt (DRBUDDI segfaults otherwise)."""
    from qsiprep.interfaces.tortoise import StageDRBUDDIPair

    up = tmp_path / 'predicted_4d.nii.gz'
    down = tmp_path / 'predicted_4d_down.nii.gz'
    _write_dummy_nii(up, nvols=2)
    _write_dummy_nii(down, nvols=2)
    (tmp_path / 'up.bmtxt').write_text('0 0 0 0 0 0\n1000 0 0 0 0 0\n')
    (tmp_path / 'down.bmtxt').write_text('0 0 0 0 0 0\n1000 0 0 0 0 0\n')

    run_dir = tmp_path / 'stage'
    run_dir.mkdir()
    res = StageDRBUDDIPair(
        up_image=str(up),
        up_bmat=str(tmp_path / 'up.bmtxt'),
        down_image=str(down),
        down_bmat=str(tmp_path / 'down.bmtxt'),
    ).run(cwd=str(run_dir))

    assert res.outputs.up_image.endswith('blip_up.nii')
    assert res.outputs.down_image.endswith('blip_down.nii')
    assert res.outputs.up_bmat.endswith('blip_up.bmtxt')
    assert res.outputs.down_bmat.endswith('blip_down.bmtxt')
    # matched stems: <image-stem>.bmtxt
    assert os.path.splitext(res.outputs.up_image)[0] + '.bmtxt' == res.outputs.up_bmat


def test_split_corrected_by_group(tmp_path):
    """SplitCorrectedByGroup partitions per-volume corrected outputs into up/down
    lists with per-side b=0 positions and per-volume blip assignments."""
    from qsiprep.interfaces.tortoise import SplitCorrectedByGroup

    ap = _make_original_with_sidecar(tmp_path, 'sub-01_dir-AP_dwi', 'j')
    pa = _make_original_with_sidecar(tmp_path, 'sub-01_dir-PA_dwi', 'j-')
    # 4 volumes: AP(b0), AP(dwi), PA(b0), PA(dwi)
    original_files = [ap, ap, pa, pa]

    dwi_files, bval_files, bvec_files, transforms = [], [], [], []
    import nibabel as nb

    for i in range(4):
        d = tmp_path / f'vol{i}.nii.gz'
        nb.Nifti1Image(np.full((2, 2, 2), i, dtype='float32'), np.eye(4)).to_filename(str(d))
        dwi_files.append(str(d))
        bv = tmp_path / f'vol{i}.bval'
        bv.write_text('0\n' if i in (0, 2) else '1000\n')
        bval_files.append(str(bv))
        bvec = tmp_path / f'vol{i}.bvec'
        bvec.write_text('0\n0\n0\n' if i in (0, 2) else '1\n0\n0\n')
        bvec_files.append(str(bvec))
        xf = tmp_path / f'vol{i}.txt'
        xf.write_text('identity')
        transforms.append(str(xf))

    run_dir = tmp_path / 'splitcorr'
    run_dir.mkdir()
    res = SplitCorrectedByGroup(
        dwi_files=dwi_files,
        bval_files=bval_files,
        bvec_files=bvec_files,
        forward_transforms=transforms,
        b0_indices=[0, 2],
        original_files=original_files,
    ).run(cwd=str(run_dir))

    # nipype OutputMultiObject squeezes single-element lists to a scalar.
    def _aslist(x):
        return x if isinstance(x, list) else [x]

    assert res.outputs.blip_assignments == ['up', 'up', 'down', 'down']
    assert len(res.outputs.up_dwi_files) == 2
    assert len(res.outputs.down_dwi_files) == 2
    # each side's local b=0 is at position 0
    assert res.outputs.up_b0_indices == [0]
    assert res.outputs.down_b0_indices == [0]
    assert len(_aslist(res.outputs.up_b0_files)) == 1
    assert len(_aslist(res.outputs.down_b0_files)) == 1


def test_init_diffprep_hmc_wf_rpe_series_non_shelled_synthesizes(tmp_path):
    """A non-shelled reverse-PE series routes to the predicted-shell DRBUDDI path
    instead of the stock DRBUDDI workflow."""
    import json as _json

    config = _base_config()
    cfg = tmp_path / 'diffprep_config.json'
    cfg.write_text(_json.dumps({'rpe_series_shelled': False}))
    config.workflow.diffprep_config = str(cfg)
    try:
        wf = _build(
            _scan_groups('rpe_series', rpe_series=['/data/sub-01_dir-PA_dwi.nii.gz']),
            t2w_sdc=False,
            name='dp_rpe_syn',
        )
        # Synthesis nodes present; stock DRBUDDI workflow is NOT built.
        for node in (
            'split_corrected_by_group',
            'up_b0_mean',
            'down_b0_mean',
            'stage_drbuddi_pair',
            'drbuddi',
            'aggregate_drbuddi',
        ):
            assert wf.get_node(node) is not None, node
        assert wf.get_node('predict_up_shell') is not None
        assert wf.get_node('predict_down_shell') is not None
        assert wf.get_node('drbuddi_sdc_wf') is None
        # The per-direction DIFFPREP split still runs (non-shelled needs it too).
        assert wf.get_node('recombine_rpe_groups') is not None
        assert wf.get_node('outputnode').inputs.sdc_method == 'DRBUDDI (predicted shell)'

        # With a T2w available the synthesis path must still build: DRBUDDI takes
        # the structural, but aggregate_drbuddi.structural_image must NOT be
        # double-connected (once from drbuddi, once from inputnode) or nipype
        # raises at build time.
        wf_t2w = _build(
            _scan_groups('rpe_series', rpe_series=['/data/sub-01_dir-PA_dwi.nii.gz']),
            t2w_sdc=True,
            name='dp_rpe_syn_t2w',
        )
        assert wf_t2w.get_node('drbuddi') is not None
    finally:
        config.workflow.diffprep_config = None


def test_rpe_series_is_shelled(tmp_path):
    """The shelled/non-shelled detector distinguishes a DTI/HARDI shell from a
    CS-DSI q-space grid, and honours the config override."""
    from qsiprep.workflows.dwi.diffprep import _rpe_series_is_shelled

    ap = tmp_path / 'ap_dwi.nii.gz'
    pa = tmp_path / 'pa_dwi.nii.gz'
    scan_groups = {
        'dwi_series': [str(ap)],
        'fieldmap_info': {'suffix': 'rpe_series', 'rpe_series': [str(pa)]},
    }

    # Shelled: a single b=1000 shell with plenty of directions.
    (tmp_path / 'ap_dwi.bval').write_text(' '.join(['0'] + ['1000'] * 12) + '\n')
    (tmp_path / 'pa_dwi.bval').write_text(' '.join(['0'] + ['1000'] * 12) + '\n')
    assert _rpe_series_is_shelled(scan_groups, 100) is True

    # Non-shelled: a CS-DSI-like grid -- many distinct b-values, none forming a
    # populous low-b shell.
    grid = list(range(200, 3000, 150))
    (tmp_path / 'ap_dwi.bval').write_text(' '.join(map(str, [0] + grid)) + '\n')
    (tmp_path / 'pa_dwi.bval').write_text(' '.join(map(str, [0] + grid)) + '\n')
    assert _rpe_series_is_shelled(scan_groups, 100) is False

    # Override wins over auto-detection either way.
    assert _rpe_series_is_shelled(scan_groups, 100, override=True) is True

    # Regression: a real CS-DSI HASC55 scheme has a dense low-b cluster (8
    # volumes near b=1195 when both PE directions are pooled) that a bare
    # min-shell-dirs count would mis-read as shelled. The grid guard (18 distinct
    # shells) plus per-side evaluation must still classify it non-shelled.
    hasc55 = (
        '5 5 3395 3400 2595 4395 3795 2795 1995 4190 3600 3395 2795 1595 5 3790 '
        '4390 800 3400 3990 1195 3590 2195 4190 4000 2790 5000 5 1795 1795 4195 '
        '3395 1195 2795 595 3590 3395 1990 2795 4195 5 3390 3600 4395 4985 4195 '
        '3390 3990 3400 2590 3590 995 2790 5000 2395 2000 1795 2190 1195 1195 '
        '2595 3790 5'
    )
    (tmp_path / 'ap_dwi.bval').write_text(hasc55 + '\n')
    (tmp_path / 'pa_dwi.bval').write_text(hasc55 + '\n')
    assert _rpe_series_is_shelled(scan_groups, 100) is False

    # Unreadable b-values default to shelled (safe stock DRBUDDI path).
    missing = {
        'dwi_series': ['/nonexistent/ap_dwi.nii.gz'],
        'fieldmap_info': {'suffix': 'rpe_series', 'rpe_series': ['/nonexistent/pa_dwi.nii.gz']},
    }
    assert _rpe_series_is_shelled(missing, 100) is True


# ---------------------------------------------------------------------------
# Dispatch + workflow wiring (pure Python)
# ---------------------------------------------------------------------------


def test_diffprep_order_helper():
    from qsiprep.workflows.dwi.base import _diffprep_order

    assert _diffprep_order('diffprep_motion') == 'motion'
    assert _diffprep_order('diffprep_quadratic') == 'quadratic'
    assert _diffprep_order('diffprep_cubic') == 'cubic'


def _base_config():
    from qsiprep import config

    config.nipype.omp_nthreads = 1
    config.workflow.diffprep_config = None
    config.workflow.b0_threshold = 100
    config.workflow.pepolar_method = 'drbuddi'
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.execution.sloppy = False
    return config


def _scan_groups(suffix=None, **extra):
    fieldmap_info = {'suffix': suffix}
    fieldmap_info.update(extra)
    return {
        'dwi_series': ['/data/sub-01_dwi.nii.gz'],
        'fieldmap_info': fieldmap_info,
        'dwi_series_pedir': 'j',
    }


def _build(scan_groups, t2w_sdc, name='dp'):
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    return init_diffprep_hmc_wf(
        scan_groups=scan_groups,
        source_file='/data/sub-01_dwi.nii.gz',
        t2w_sdc=t2w_sdc,
        correction_mode='quadratic',
        dwi_metadata={'PhaseEncodingDirection': 'j'},
        name=name,
    )


def test_init_diffprep_hmc_wf_contract_hmc_only():
    """No fieldmap + no T2w -> HMC-only, sdc_method='None', full contract."""
    _base_config()
    wf = _build(_scan_groups(None), t2w_sdc=False)

    outputnode = wf.get_node('outputnode')
    required = {
        'dwi_files_to_transform',
        'bvec_files_to_transform',
        'bval_files',
        'b0_indices',
        'to_dwi_ref_affines',
        'to_dwi_ref_warps',
        'b0_template',
        'b0_template_mask',
        'cnr_map',
        'slice_quality',
        'sdc_method',
        'motion_params',
        'pre_sdc_template',
    }
    assert required.issubset(set(outputnode.inputs.copyable_trait_names()))
    assert outputnode.inputs.sdc_method == 'None'
    assert wf.get_node('diffprep').inputs.epi_mode == 'off'
    for node in ('tortoise_convert', 'diffprep', 'split_outputs', 'b0_ref_for_coreg'):
        assert wf.get_node(node) is not None


def test_init_diffprep_hmc_wf_t2wreg():
    """No fieldmap + T2w -> TORTOISE T2Wreg baked in (sdc_method='T2Wreg')."""
    _base_config()
    wf = _build(_scan_groups(None), t2w_sdc=True)
    assert wf.get_node('diffprep').inputs.epi_mode == 'T2Wreg'
    assert wf.get_node('outputnode').inputs.sdc_method == 'T2Wreg'


def test_init_diffprep_hmc_wf_syn_without_t2w():
    """Fieldmap-less SyN request with no T2w falls back to init_sdc_wf, and the
    DIFFPREP call leaves TORTOISE's own EPI stage off."""
    _base_config()
    wf = _build(_scan_groups('syn'), t2w_sdc=False)
    assert wf.get_node('diffprep').inputs.epi_mode == 'off'
    assert wf.get_node('sdc_wf') is not None


def test_cnr_model_label_is_bids_valid():
    """The ``model`` entity names the signal model and must be alphanumeric.

    ``diffprep_quadratic`` could not be parsed back -- ``_`` is the BIDS entity
    separator -- and DIFFPREP emits no CNR of its own, so the diffprep backends
    report the MAPMRI model the CNR is actually derived from. Every other
    backend must be left exactly as it was.
    """
    import re

    from qsiprep.workflows.dwi.derivatives import _cnr_model_label

    for unchanged in ('3dSHORE', 'eddy', 'tensor', 'none'):
        assert _cnr_model_label(unchanged) == unchanged

    for diffprep_model in ('diffprep_motion', 'diffprep_quadratic', 'diffprep_cubic'):
        assert _cnr_model_label(diffprep_model) == 'MAPMRI'

    entity = re.compile(r'^[a-zA-Z0-9]+$')
    for model in ('3dSHORE', 'eddy', 'tensor', 'none', 'diffprep_quadratic'):
        assert entity.match(_cnr_model_label(model)), model


def test_cnr_description_flags_in_sample_bias():
    """The diffprep CNR is an in-sample fit; the sidecar must say so."""
    from qsiprep.workflows.dwi.derivatives import _cnr_description

    baseline = _cnr_description('3dSHORE')
    assert baseline == 'Contrast-to-noise ratio map for the HMC step.'

    diffprep_desc = _cnr_description('diffprep_quadratic')
    assert 'MAPMRI' in diffprep_desc
    assert 'in-sample' in diffprep_desc
    assert 'not quantitatively comparable' in diffprep_desc


def test_init_diffprep_hmc_wf_cnr_is_computed_not_placeholder():
    """cnr_map must come from CalculateCNR on the MAPMRI synthesis, not zeros."""
    _base_config()
    wf = _build(_scan_groups(None), t2w_sdc=False, name='dp_cnr')

    node = wf.get_node('calculate_cnr')
    assert node is not None
    # Same three inputs SliceQC consumes, so no extra model fit is needed.
    assert wf.get_node('synth_dwis') is not None
    assert wf.get_node('split_outputs') is not None

    # cnr_map is fed by calculate_cnr.cnr_image
    edge = wf._graph.get_edge_data(node, wf.get_node('outputnode'))
    assert edge is not None
    assert ('cnr_image', 'cnr_map') in edge['connect']


def test_init_diffprep_hmc_wf_honours_sloppy():
    """--sloppy must take TORTOISE's expensive second pass out, via --niter 0.

    Without it a DIFFPREP node can burn >1h on CI-sized data (emitting no output
    while it does, which trips no_output_timeout).

    It must do so with ``--niter 0`` and NOT by clearing ``is_human_brain``:
    that flag reaches the same ``iterative`` gate but also makes DIFFPREP's
    auto-masking look for a ``<stem>_noise.nii`` and changes structural-target
    masking on the T2Wreg path.
    """
    config = _base_config()

    wf = _build(_scan_groups(None), t2w_sdc=False, name='dp_notsloppy')
    node = wf.get_node('diffprep')
    assert node.inputs.is_human_brain is True
    assert not isdefined(node.inputs.niter)
    # a production run gets exactly the correction the user asked for
    assert node.inputs.correction_mode == 'quadratic'

    config.execution.sloppy = True
    try:
        wf = _build(_scan_groups(None), t2w_sdc=False, name='dp_sloppy')
        node = wf.get_node('diffprep')
        assert node.inputs.niter == 0
        # --niter 0 only bites on high-b data, so the always-run first pass is
        # bounded by dropping the 24-parameter quadratic fit to rigid.
        assert node.inputs.correction_mode == 'motion'
        # sloppy must not silently redefine what the data *is*
        assert node.inputs.is_human_brain is True
    finally:
        config.execution.sloppy = False


def test_init_diffprep_hmc_wf_rpe_series_runs_per_direction(tmp_path):
    """rpe_series must run DIFFPREP once per phase-encoding direction.

    A single DIFFPREP run models one phase axis / one b=0 reference for the
    whole file, so the concatenated opposing-PE series would be silently
    mis-corrected. The backend re-splits the merged series into its two PE
    groups, corrects each on its own, and recombines before handing the flat
    list to the stock DRBUDDI path.
    """
    _base_config()
    # DRBUDDI's GatherDRBUDDIInputs validates ``epi_fmaps`` (the rpe series) as
    # existing files at build time, so point it at a real (tiny) nii.
    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe, nvols=2)
    wf = _build(
        _scan_groups('rpe_series', rpe_series=[str(rpe)]),
        t2w_sdc=False,
        name='dp_rpe',
    )

    # Per-direction stage present; the single-run DIFFPREP nodes are NOT built.
    for node in (
        'split_rpe_groups',
        'tortoise_convert_g1',
        'diffprep_g1',
        'tortoise_convert_g2',
        'diffprep_g2',
        'recombine_rpe_groups',
    ):
        assert wf.get_node(node) is not None, node
    assert wf.get_node('diffprep') is None
    assert wf.get_node('tortoise_convert') is None

    # The recombined triple feeds the shared downstream split (drop-in for a
    # single DIFFPREP node), and DRBUDDI is wired for SDC.
    recombine = wf.get_node('recombine_rpe_groups')
    split_outputs = wf.get_node('split_outputs')
    edge = wf._graph.get_edge_data(recombine, split_outputs)
    assert edge is not None
    assert ('corrected_dwi_file', 'corrected_dwi_file') in edge['connect']
    assert wf.get_node('drbuddi_sdc_wf') is not None


def test_init_diffprep_hmc_wf_rpe_series_pe_axis(tmp_path):
    """The split node is told the phase-encoding axis of the series.

    DIFFPREP is sign-agnostic on the axis, but the split still labels the
    first-appearing group '+' and the second '-' so provenance is explicit.
    """
    _base_config()
    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe, nvols=2)
    wf = _build(
        _scan_groups('rpe_series', rpe_series=[str(rpe)]),
        t2w_sdc=False,
        name='dp_rpe_axis',
    )
    split = wf.get_node('split_rpe_groups')
    assert split.inputs.pe_axis == 'j'


def test_drbuddi_sloppy_skips_rigid_diffeo_loop(tmp_path):
    """--sloppy should also cheapen DRBUDDI's initial registration.

    This covers shared code (``init_drbuddi_wf`` is used by the FSL backend
    too), but DRBUDDI is the SDC half of the DIFFPREP backend. ``sloppy``
    already replaces the diffeomorphic schedule via --DRBUDDI_stage, which has
    no effect on Step1; this additionally skips Step1's rigid+diffeo+rigid loop.

    Deliberately NOT --DRBUDDI_disable_initial_rigid: that suppresses
    ``bdown_to_bup_rigid_trans_h5``, which DRBUDDIAggregateOutputs dereferences
    unguarded on the rpe_series FA branch.
    """
    from qsiprep.interfaces.tortoise import DRBUDDI

    up = tmp_path / 'up.nii'
    down = tmp_path / 'down.nii'
    _write_dummy_nii(up, nvols=1)
    _write_dummy_nii(down, nvols=1)
    up_json = tmp_path / 'up.json'
    up_json.write_text('{"PhaseEncodingDirection": "j"}')

    common = {
        'fieldmap_type': 'rpe_series',
        'blip_up_image': str(up),
        'blip_down_image': str(down),
        'blip_up_json': str(up_json),
    }
    flag = '--DRBUDDI_start_with_diffeomorphic_for_rigid_reg'

    sloppy_cmd = DRBUDDI(
        sloppy=True, start_with_diffeomorphic_for_rigid_reg=True, **common
    ).cmdline
    assert flag in sloppy_cmd
    assert '--DRBUDDI_stage' in sloppy_cmd
    # the destructive lever stays off so the rigid transform is still produced
    assert '--DRBUDDI_disable_initial_rigid' not in sloppy_cmd

    prod_cmd = DRBUDDI(
        sloppy=False, start_with_diffeomorphic_for_rigid_reg=False, **common
    ).cmdline
    assert flag not in prod_cmd
    assert '--DRBUDDI_stage' not in prod_cmd


def test_init_diffprep_hmc_wf_topup_rejected():
    """DIFFPREP cannot use eddy-internal TOPUP; ask for DRBUDDI instead."""
    config = _base_config()
    config.workflow.pepolar_method = 'TOPUP'
    try:
        with pytest.raises(Exception, match='TOPUP'):
            _build(_scan_groups('epi', epi=['/data/sub-01_epi.nii.gz']), False)
    finally:
        config.workflow.pepolar_method = 'drbuddi'
