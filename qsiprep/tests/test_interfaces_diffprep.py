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


def test_init_diffprep_hmc_wf_topup_rejected():
    """DIFFPREP cannot use eddy-internal TOPUP; ask for DRBUDDI instead."""
    config = _base_config()
    config.workflow.pepolar_method = 'TOPUP'
    try:
        with pytest.raises(Exception, match='TOPUP'):
            _build(_scan_groups('rpe_series', rpe_series=['/data/sub-01_dwi.nii.gz']), False)
    finally:
        config.workflow.pepolar_method = 'drbuddi'
