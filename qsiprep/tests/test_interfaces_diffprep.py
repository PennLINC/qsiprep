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


def test_tortoise_use_cuda_swaps_the_binary(tmp_path):
    """use_cuda picks the <cmd>_cuda build, mirroring ExtendedEddy._use_cuda.

    TORTOISE ships fixed ``_cuda`` suffixes, so unlike FSL (eddy_cuda11.0) no
    PATH scan is needed. Default must stay on CPU.
    """
    from qsiprep.interfaces.tortoise import DIFFPREP, DRBUDDI

    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    common = {'dwi_file': str(dwi), 'bmtxt_file': str(bmtxt), 'json_file': str(json_file)}

    assert DIFFPREP(**common).cmd == 'TORTOISEProcess'
    assert DIFFPREP(use_cuda=False, **common).cmd == 'TORTOISEProcess'
    assert DIFFPREP(use_cuda=True, **common).cmd == 'TORTOISEProcess_cuda'

    assert DRBUDDI().cmd == 'DRBUDDI'
    assert DRBUDDI(use_cuda=False).cmd == 'DRBUDDI'
    assert DRBUDDI(use_cuda=True).cmd == 'DRBUDDI_cuda'


def test_use_cuda_is_not_passed_on_the_command_line(tmp_path):
    """use_cuda selects the executable; it is not a TORTOISEProcess flag."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    cmd = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        correction_mode='quadratic',
        use_cuda=True,
    ).cmdline
    assert cmd.startswith('TORTOISEProcess_cuda')
    # Not `'use_cuda' not in cmd` -- pytest's tmp_path embeds the test name.
    assert '--use_cuda' not in cmd
    assert not any(tok == 'use_cuda' for tok in cmd.split())


def test_diffprep_config_use_cuda_default_and_override(tmp_path):
    """``use_cuda`` gets no default: only a user-set value counts as intent.

    The shipped diffprep_params.json sets it to False, a user file may set it
    either way, and a user file without the key must leave it absent so
    gpu_enabled() does not mistake a default for a choice.
    """
    import json as _json

    from qsiprep.workflows.dwi.diffprep import _load_diffprep_config

    assert _load_diffprep_config(None)['use_cuda'] is False

    cfg = tmp_path / 'cuda_cfg.json'
    cfg.write_text(_json.dumps({'use_cuda': True}))
    assert _load_diffprep_config(str(cfg))['use_cuda'] is True

    cfg_absent = tmp_path / 'no_cuda_key.json'
    cfg_absent.write_text(_json.dumps({'b0_id': 0}))
    assert 'use_cuda' not in _load_diffprep_config(str(cfg_absent))


def test_diffprep_wf_honours_use_cuda(tmp_path):
    """A --diffprep-config asking for CUDA reaches the DIFFPREP node."""
    import json as _json

    config = _base_config()
    cfg = tmp_path / 'wf_cuda_cfg.json'
    cfg.write_text(_json.dumps({'use_cuda': True}))
    orig = config.workflow.diffprep_config
    try:
        config.workflow.diffprep_config = str(cfg)
        wf = _build(_make_unit(), t2w_sdc=False, name='cuda_on')
        assert wf.get_node('diffprep').interface.cmd == 'TORTOISEProcess_cuda'

        config.workflow.diffprep_config = None
        wf_cpu = _build(_make_unit(), t2w_sdc=False, name='cuda_off')
        assert wf_cpu.get_node('diffprep').interface.cmd == 'TORTOISEProcess'
    finally:
        config.workflow.diffprep_config = orig


def test_diffprep_correction_mode_defaults_to_quadratic():
    """``correction_mode`` is a --diffprep-config key, defaulting to quadratic.

    The CLI exposes one ``--hmc-model tortoise`` rather than a value per mode,
    so the config JSON is the only way to reach ``motion`` or ``cubic``.
    """
    from qsiprep.workflows.dwi.diffprep import _load_diffprep_config

    assert _load_diffprep_config(None)['correction_mode'] == 'quadratic'


def test_diffprep_wf_honours_correction_mode(tmp_path):
    """A correction_mode in --diffprep-config reaches the DIFFPREP node."""
    import json as _json

    config = _base_config()
    cfg = tmp_path / 'cubic_cfg.json'
    cfg.write_text(_json.dumps({'correction_mode': 'cubic'}))
    orig = config.workflow.diffprep_config
    try:
        config.workflow.diffprep_config = str(cfg)
        wf = _build(_make_unit(), t2w_sdc=False, name='mode_cubic')
        assert wf.get_node('diffprep').inputs.correction_mode == 'cubic'

        config.workflow.diffprep_config = None
        wf_default = _build(_make_unit(), t2w_sdc=False, name='mode_default')
        assert wf_default.get_node('diffprep').inputs.correction_mode == 'quadratic'
    finally:
        config.workflow.diffprep_config = orig


def test_diffprep_boilerplate_describes_the_configured_mode(tmp_path):
    """The methods section must describe the mode that actually ran.

    ``correction_mode`` is selectable through --diffprep-config, so boilerplate
    that hardcodes "quadratic eddy currents" would misreport a motion-only or
    cubic run.
    """
    import json as _json

    config = _base_config()
    cfg = tmp_path / 'boilerplate_cfg.json'
    cfg.write_text(_json.dumps({'correction_mode': 'motion'}))
    orig = config.workflow.diffprep_config
    try:
        config.workflow.diffprep_config = str(cfg)
        wf = _build(_make_unit(), t2w_sdc=False, name='boiler_motion')
        assert 'rigid head motion only' in wf.__desc__
        assert 'quadratic eddy' not in wf.__desc__

        config.workflow.diffprep_config = None
        wf_quad = _build(_make_unit(), t2w_sdc=False, name='boiler_quad')
        assert 'quadratic eddy currents' in wf_quad.__desc__
    finally:
        config.workflow.diffprep_config = orig


def _stage_diffprep_outputs(tmp_path, t2wreg):
    """Recreate the file tree TORTOISEProcess leaves behind in a node's cwd.

    The motion+eddy step writes ``<stem>_temp_proc/<stem>_proc_moteddy.*``,
    and when ``-s`` is
    given the StructuralAlignment + FinalData stages additionally write
    ``<stem>_TORTOISE_final.nii`` and its own reoriented ``.bmtxt`` in the cwd.
    """
    dwi, bmtxt, json_file = _diffprep_siblings(tmp_path)
    temp_proc = tmp_path / 'dwi_temp_proc'
    temp_proc.mkdir()
    _write_dummy_nii(temp_proc / 'dwi_proc_moteddy.nii')
    (temp_proc / 'dwi_proc_moteddy.bmtxt').write_text('0 0 0 0 0 0\n1000 0 0 0 0 0\n')
    (temp_proc / 'dwi_proc_moteddy_transformations.txt').write_text('')
    if t2wreg:
        # The EPI stage's displacement field plus the before/after b=0 pair.
        _write_dummy_nii(temp_proc / 'deformation_FINV.nii.gz', nvols=3)
        _write_dummy_nii(temp_proc / 'blip_up_b0.nii', nvols=1)
        _write_dummy_nii(temp_proc / 'blip_up_b0_corrected.nii', nvols=1)
        _write_dummy_nii(temp_proc / 'structural_used.nii', nvols=1)
        # StructuralAlignment + FinalData still run (--step only sets the START
        # step), so their output exists -- the interface must ignore it.
        _write_dummy_nii(tmp_path / 'dwi_TORTOISE_final.nii')
        (tmp_path / 'dwi_TORTOISE_final.bmtxt').write_text('0 0 0 0 0 0\n0 0 1000 0 0 0\n')
    return dwi, bmtxt, json_file


def test_diffprep_outputs_off_uses_moteddy(tmp_path, monkeypatch):
    """epi_mode='off': both the image and the bmatrix come from motion+eddy."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _stage_diffprep_outputs(tmp_path, t2wreg=False)
    monkeypatch.chdir(tmp_path)

    outputs = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        epi_mode='off',
    )._list_outputs()

    assert outputs['corrected_dwi_file'].endswith('dwi_temp_proc/dwi_proc_moteddy.nii')
    assert outputs['corrected_bmtxt_file'].endswith('dwi_temp_proc/dwi_proc_moteddy.bmtxt')


def test_diffprep_outputs_t2wreg_stays_in_native_space(tmp_path, monkeypatch):
    """epi_mode='T2Wreg' must emit the PRE-EPI image plus the EPI warp.

    Passing ``-s`` makes TORTOISE run StructuralAlignment and FinalData, which
    resample the DWIs into the structural's frame -- pulling coregistration and
    ACPC alignment forward into HMC, where qsiprep does not want them (measured on
    real data: ~12 deg of the alignment happened inside DIFFPREP, leaving
    qsiprep's own b0->anat step with 1.7 deg). Take the native-grid motion+eddy
    image and hand the EPI displacement field out as a transform instead, which is
    the contract the DRBUDDI branch already uses.
    """
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _stage_diffprep_outputs(tmp_path, t2wreg=True)
    t2w = tmp_path / 't2w.nii'
    _write_dummy_nii(t2w, nvols=1)
    monkeypatch.chdir(tmp_path)

    outputs = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        epi_mode='T2Wreg',
        structural_image=str(t2w),
    )._list_outputs()

    # Native grid, and the bmatrix that goes with it.
    assert outputs['corrected_dwi_file'].endswith('dwi_temp_proc/dwi_proc_moteddy.nii')
    assert outputs['corrected_bmtxt_file'].endswith('dwi_temp_proc/dwi_proc_moteddy.bmtxt')
    # The reoriented FinalData output exists but must NOT be used.
    assert 'TORTOISE_final' not in outputs['corrected_dwi_file']
    assert 'TORTOISE_final' not in outputs['corrected_bmtxt_file']
    # SDC travels as a transform.
    assert outputs['sdc_warp'].endswith('dwi_temp_proc/deformation_FINV.nii.gz')
    # Named so nipype's remove_unnecessary_outputs keeps the SDC report inputs.
    assert outputs['b0_up_image'].endswith('blip_up_b0.nii')
    assert outputs['b0_up_corrected_image'].endswith('blip_up_b0_corrected.nii')
    assert outputs['structural_image'].endswith('structural_used.nii')


def test_diffprep_epi_off_emits_no_warp(tmp_path, monkeypatch):
    """Without the EPI stage there is no displacement field to hand downstream."""
    from nipype.interfaces.base import isdefined

    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _stage_diffprep_outputs(tmp_path, t2wreg=False)
    monkeypatch.chdir(tmp_path)

    outputs = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        epi_mode='off',
    )._list_outputs()

    assert outputs['corrected_dwi_file'].endswith('dwi_proc_moteddy.nii')
    assert not isdefined(outputs['sdc_warp'])


def test_diffprep_t2wreg_missing_warp(tmp_path, monkeypatch):
    """A T2Wreg run without its displacement field is an error, not a silent skip.

    Falling through would produce a run with no susceptibility correction at all
    while still reporting sdc_method='T2Wreg'.
    """
    from qsiprep.interfaces.tortoise import DIFFPREP

    dwi, bmtxt, json_file = _stage_diffprep_outputs(tmp_path, t2wreg=True)
    t2w = tmp_path / 't2w.nii'
    _write_dummy_nii(t2w, nvols=1)
    (tmp_path / 'dwi_temp_proc' / 'deformation_FINV.nii.gz').unlink()
    monkeypatch.chdir(tmp_path)

    iface = DIFFPREP(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        json_file=str(json_file),
        epi_mode='T2Wreg',
        structural_image=str(t2w),
    )
    with pytest.raises(FileNotFoundError, match='displacement field'):
        iface._list_outputs()


def test_diffprep_motion_params_basic(tmp_path):
    """``DIFFPREPMotionParams`` slices cols 0-5 from a 24-col TORTOISE
    transformations file, converts them from LPS to RAS+ (negating the x/y
    translation and rotation components), and writes them as a
    whitespace-separated SPM file."""
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
    expected = full[:, :6].copy()
    expected[:, [0, 1, 3, 4]] *= -1.0
    np.testing.assert_allclose(spm, expected, atol=1e-5)


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
    expected = full[0, :6].copy()
    expected[[0, 1, 3, 4]] *= -1.0
    np.testing.assert_allclose(spm, expected)


def test_diffprep_motion_params_rejects_short_rows(tmp_path):
    """A transforms file with fewer than 24 columns is rejected."""
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    transforms_file = tmp_path / 'short.txt'
    transforms_file.write_text('0 0 0 0 0 0\n')
    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    with pytest.raises(ValueError, match='24 columns'):
        iface.run(cwd=str(tmp_path))


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
    # deoblique=True routes the gradients through mrtrix's mrinfo -dwgrad.
    _require('FSLBVecsToTORTOISEBmatrix', 'TORTOISEBmatrixToFSLBVecs', 'mrinfo')
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


def test_diffprep_split_outputs_deobliques_gradients(tmp_path):
    """Gradients go through qsiprep's mrtrix RAS+ conversion, not raw FSL bvecs.

    ``TORTOISEBmatrixToFSLBVecs`` emits FSL-convention (voxel-frame) bvecs, which
    depend on the image's orientation and obliquity. ``split_bvals_bvecs`` with
    ``deoblique=True`` runs them through ``mrinfo -dwgrad -fslgrad`` -- the same
    tested conversion ``SplitDWIsFSL(deoblique_bvecs=True)`` gives the eddy
    backend.

    Assert the interface's output matches that conversion rather than the raw
    bvecs. On an oblique grid the two differ, so flipping the flag back fails
    here; DIFFPREP's own grid is normally axis-aligned (TORTOISE resamples onto
    the ACPC structural), which is why such a regression would otherwise be
    invisible.
    """
    import nibabel as nb

    _require('FSLBVecsToTORTOISEBmatrix', 'TORTOISEBmatrixToFSLBVecs', 'mrinfo')
    from qsiprep.interfaces.images import split_bvals_bvecs
    from qsiprep.interfaces.tortoise import DIFFPREPSplitOutputs, bmtxt_to_fsl, make_bmat_file

    bvals = [0, 1000, 1000]
    bvecs = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    bval_file, bvec_file = _write_fsl_gradients(tmp_path, bvals, bvecs)
    bmtxt = make_bmat_file(str(bval_file), str(bvec_file))

    # A deliberately oblique grid (~10 degrees about z), where the two differ.
    theta = np.deg2rad(10.0)
    affine = np.eye(4)
    affine[:3, :3] = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    corrected = tmp_path / 'oblique.nii'
    nb.Nifti1Image(np.zeros((4, 4, 4, 3), dtype='float32'), affine).to_filename(str(corrected))

    run_dir = tmp_path / 'split_oblique'
    run_dir.mkdir()
    res = DIFFPREPSplitOutputs(
        corrected_dwi_file=str(corrected),
        corrected_bmtxt_file=str(bmtxt),
        b0_threshold=100,
    ).run(cwd=str(run_dir))
    emitted = np.array([np.loadtxt(f) for f in res.outputs.bvec_files])

    # Reproduce both candidate conversions from the same bmatrix, against the
    # per-volume images the interface itself wrote.
    ref_dir = tmp_path / 'reference'
    (ref_dir / 'on').mkdir(parents=True)
    (ref_dir / 'off').mkdir(parents=True)
    ref_bval, ref_bvec = bmtxt_to_fsl(str(bmtxt), str(ref_dir))
    imgs = list(res.outputs.dwi_files)
    _, deob_files = split_bvals_bvecs(
        ref_bval, ref_bvec, imgs, deoblique=True, working_dir=str(ref_dir / 'on')
    )
    _, raw_files = split_bvals_bvecs(
        ref_bval, ref_bvec, imgs, deoblique=False, working_dir=str(ref_dir / 'off')
    )
    deobliqued = np.array([np.loadtxt(f) for f in deob_files])
    raw = np.array([np.loadtxt(f) for f in raw_files])

    assert np.allclose(emitted, deobliqued, atol=1e-5), (
        f'gradients are not the mrtrix RAS+ ones:\n{emitted}\nvs\n{deobliqued}'
    )
    # And on this grid that is a real difference, so the assertion has teeth.
    assert not np.allclose(deobliqued, raw, atol=1e-3), (
        'oblique fixture failed to distinguish the two conversions'
    )


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


@pytest.fixture
def t2w_gate_config():
    """Pin every config knob _t2w_available_for_sdc reads, and restore after."""
    from qsiprep import config

    saved = {
        k: getattr(config.workflow, k, None)
        for k in ('anat_modality', 'pepolar_method', 'hmc_model')
    }
    config.workflow.anat_modality = 't1w'
    config.workflow.pepolar_method = 'TOPUP'
    config.workflow.hmc_model = 'eddy'
    try:
        yield config
    finally:
        for k, v in saved.items():
            setattr(config.workflow, k, v)


T2W_SUBJECT = {'t2w': ['/data/sub-01_T2w.nii.gz']}


def test_t2w_available_for_sdc_requires_anat_processing(t2w_gate_config):
    """With --anat-modality none there is no anatomical workflow, so t2w_unfatsat
    is never produced; requesting T2w SDC then leaves the DRBUDDI structural / the
    extended report's t2w_n4 with an empty input (the CI failure)."""
    from qsiprep.grouping.methods import selection_for_config
    from qsiprep.workflows.base import _t2w_available_for_sdc

    selection = selection_for_config('eddy', 'drbuddi')
    assert _t2w_available_for_sdc(T2W_SUBJECT, selection) is True
    assert _t2w_available_for_sdc({}, selection) is False

    t2w_gate_config.workflow.anat_modality = 'none'
    assert _t2w_available_for_sdc(T2W_SUBJECT, selection) is False


def test_t2w_available_for_sdc_requires_a_consuming_method(t2w_gate_config):
    """t2w_unfatsat only exists when a selected method has a consuming stage.

    ``additional_t2ws`` -- the only thing that makes init_anat_preproc_wf build its
    T2w branch -- was gated on the PEPOLAR tool alone. DIFFPREP's T2Wreg path
    consumes the T2w without any PEPOLAR data and is not gated on that choice, so
    a plain ``--hmc-method tortoise --ignore fieldmaps`` run requested T2w SDC
    while nothing produced the image, and DIFFPREP died with
    ``epi_mode="T2Wreg" requires a structural_image``.
    """
    from qsiprep.grouping.methods import selection_for_config
    from qsiprep.workflows.base import _t2w_available_for_sdc, _t2w_sdc_enabled

    # eddy + TOPUP: nothing consumes a T2w, so do not claim T2w SDC.
    selection = selection_for_config('eddy', 'topup')
    assert _t2w_sdc_enabled(selection) is False
    assert _t2w_available_for_sdc(T2W_SUBJECT, selection) is False

    # DRBUDDI consumes it as its multimodal --structural.
    selection = selection_for_config('eddy', 'topup+drbuddi')
    assert _t2w_sdc_enabled(selection) is True
    assert _t2w_available_for_sdc(T2W_SUBJECT, selection) is True

    # The regression: DIFFPREP consumes it via --epi T2Wreg regardless of
    # the PEPOLAR tool choice.
    selection = selection_for_config('tortoise', 'auto')
    assert _t2w_sdc_enabled(selection) is True, 'tortoise'
    assert _t2w_available_for_sdc(T2W_SUBJECT, selection) is True, 'tortoise'


def test_extended_pepolar_report_t2w_n4_gets_input():
    """The node that failed in CI: with a T2w, the report's t2w_n4 must be fed
    from inputnode.t2w_image; without one it uses the t1w-seg branch instead."""
    _base_config()
    from qsiprep.workflows.fieldmap.pepolar import init_extended_pepolar_report_wf

    wf = init_extended_pepolar_report_wf(segment_t2w=True)
    t2w_n4 = wf.get_node('t2w_n4')
    assert t2w_n4 is not None
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), t2w_n4)
    assert edge is not None
    assert ('t2w_image', 'input_image') in edge['connect']

    wf_no_t2w = init_extended_pepolar_report_wf(segment_t2w=False)
    assert wf_no_t2w.get_node('t2w_n4') is None
    assert wf_no_t2w.get_node('map_seg') is not None


def _base_config():
    from qsiprep import config

    config.nipype.omp_nthreads = 1
    config.workflow.diffprep_config = None
    config.workflow.b0_threshold = 100
    config.workflow.pepolar_method = 'drbuddi'
    # The legacy keys drive these tests; clear the axis keys so a selection
    # left behind by another test cannot shadow them.
    config.workflow.hmc_method = None
    config.workflow.sdc_method = None
    config.workflow.shoreline_model = None
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.gpu = None  # --gpu not given, so legacy use_cuda keys apply
    config.execution.sloppy = False
    return config


def _make_unit(suffix=None, **extra):
    """Build the PreprocUnit for a DIFFPREP test case.

    ``suffix`` mirrors the old scan-group vocabulary: ``None`` fieldmap-less,
    ``'rpe_series'`` a reverse-PE partner series, ``'epi'`` a dedicated epi
    fieldmap, or a GRE suffix (``'phasediff'``/``'fieldmap'``).
    """
    from qsiprep.grouping.models import CorrectionMethod
    from qsiprep.tests.preproc_factory import make_preproc_unit

    dwi = '/data/sub-01_dwi.nii.gz'
    if suffix == 'rpe_series':
        partners = list(extra['rpe_series'])
        return make_preproc_unit(
            [dwi, *partners],
            method=CorrectionMethod.PEPOLAR,
            pe_dirs={dwi: 'j', **dict.fromkeys(partners, 'j-')},
        )
    if suffix == 'epi':
        return make_preproc_unit(
            [dwi],
            method=CorrectionMethod.PEPOLAR,
            pe_dir='j',
            estimation_sources=[dwi, *extra['epi']],
        )
    if suffix in ('phasediff', 'fieldmap'):
        return make_preproc_unit(
            [dwi],
            method=CorrectionMethod.PHASEDIFF
            if suffix == 'phasediff'
            else CorrectionMethod.DIRECT,
            pe_dir='j',
            estimation_sources=list(extra.get('sources', [])),
        )
    # None (or the retired 'syn'): fieldmap-less.
    return make_preproc_unit([dwi], method=None, pe_dir='j')


def _build(unit, t2w_sdc, name='dp'):
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    return init_diffprep_hmc_wf(
        unit=unit,
        source_file='/data/sub-01_dwi.nii.gz',
        t2w_sdc=t2w_sdc,
        name=name,
    )


def test_init_diffprep_hmc_wf_contract_hmc_only():
    """No fieldmap + no T2w -> HMC-only, sdc_method='None', full contract."""
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=False)

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
    """No fieldmap + T2w -> TORTOISE T2Wreg (sdc_method='T2Wreg')."""
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=True)
    assert wf.get_node('diffprep').inputs.epi_mode == 'T2Wreg'
    assert wf.get_node('outputnode').inputs.sdc_method == 'T2Wreg'


def _connect_fields(wf, src, dst):
    edge = wf._graph.get_edge_data(wf.get_node(src), wf.get_node(dst))
    return [] if edge is None else list(edge['connect'])


def test_t2wreg_sdc_travels_as_a_warp_not_baked_in():
    """The EPI field must reach to_dwi_ref_warps so qsiprep resamples once.

    Taking TORTOISE's FinalData instead would bake in its StructuralAlignment,
    performing coregistration and ACPC inside HMC. Routing the displacement field
    to ``to_dwi_ref_warps`` puts this branch on the same footing as DRBUDDI: the
    correction composes with HMC and coregistration in a single resampling.
    """
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=True, name='t2wreg_warp')

    out_fields = [dst for _, dst in _connect_fields(wf, 'diffprep', 'outputnode')]
    assert 'to_dwi_ref_warps' in out_fields
    # The report needs the before/after pair and the structural actually used.
    for field in ('b0_up_image', 'b0_up_corrected_image', 't2w_image'):
        assert field in out_fields, field


def test_t2wreg_coregistration_uses_the_corrected_b0():
    """Coregistration must see the undistorted b=0.

    DIFFPREP's output is pre-EPI on this path, so the warp has to be applied to
    the b=0 before it becomes the coregistration reference -- otherwise the
    dwi->anat registration is computed on distorted data.
    """
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=True, name='t2wreg_coreg')

    apply_sdc = wf.get_node('apply_sdc_to_b0')
    assert apply_sdc is not None, 'expected the b=0 to be unwarped before coregistration'
    assert ('b0_average', 'input_image') in _connect_fields(wf, 'extract_b0s', 'apply_sdc_to_b0')
    assert 'transforms' in [dst for _, dst in _connect_fields(wf, 'diffprep', 'apply_sdc_to_b0')]
    assert ('output_image', 'inputnode.b0_template') in _connect_fields(
        wf, 'apply_sdc_to_b0', 'b0_ref_for_coreg'
    )
    # The raw b=0 must NOT also feed the reference, or the corrected one is moot.
    assert ('b0_average', 'inputnode.b0_template') not in _connect_fields(
        wf, 'extract_b0s', 'b0_ref_for_coreg'
    )


def test_non_t2wreg_coregistration_uses_the_b0_directly():
    """Without an in-TORTOISE EPI stage there is nothing to unwarp first."""
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=False, name='no_t2wreg_coreg')
    assert wf.get_node('apply_sdc_to_b0') is None
    assert ('b0_average', 'inputnode.b0_template') in _connect_fields(
        wf, 'extract_b0s', 'b0_ref_for_coreg'
    )


def test_init_diffprep_hmc_wf_fieldmapless_without_t2w():
    """Fieldmap-less with no T2w is HMC only: no SDC node, TORTOISE's EPI stage off."""
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=False)
    assert wf.get_node('diffprep').inputs.epi_mode == 'off'
    assert wf.get_node('sdc_wf') is None


def test_cnr_model_label_is_bids_valid():
    """The ``model`` entity names the signal model and must be alphanumeric.

    DIFFPREP emits no CNR of its own, so the ``tortoise`` backend reports the
    MAPMRI model the CNR is actually derived from rather than its own name.
    Every other backend must be left exactly as it was.
    """
    import re

    from qsiprep.workflows.dwi.derivatives import _cnr_model_label

    for unchanged in ('3dSHORE', 'eddy', 'tensor', 'none'):
        assert _cnr_model_label(unchanged) == unchanged

    assert _cnr_model_label('tortoise') == 'MAPMRI'

    entity = re.compile(r'^[a-zA-Z0-9]+$')
    for model in ('3dSHORE', 'eddy', 'tensor', 'none', 'tortoise'):
        assert entity.match(_cnr_model_label(model)), model


def test_cnr_description_flags_in_sample_bias():
    """The diffprep CNR is an in-sample fit; the sidecar must say so."""
    from qsiprep.workflows.dwi.derivatives import _cnr_description

    baseline = _cnr_description('3dSHORE')
    assert baseline == 'Contrast-to-noise ratio map for the HMC step.'

    diffprep_desc = _cnr_description('tortoise')
    assert 'MAPMRI' in diffprep_desc
    assert 'in-sample' in diffprep_desc
    assert 'not quantitatively comparable' in diffprep_desc


def test_init_diffprep_hmc_wf_cnr_is_computed_not_placeholder():
    """cnr_map must come from CalculateCNR on the MAPMRI synthesis, not zeros."""
    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=False, name='dp_cnr')

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

    wf = _build(_make_unit(None), t2w_sdc=False, name='dp_notsloppy')
    node = wf.get_node('diffprep')
    assert node.inputs.is_human_brain is True
    assert not isdefined(node.inputs.niter)
    # a production run gets exactly the correction the user asked for
    assert node.inputs.correction_mode == 'quadratic'

    config.execution.sloppy = True
    try:
        wf = _build(_make_unit(None), t2w_sdc=False, name='dp_sloppy')
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
        _make_unit('rpe_series', rpe_series=[str(rpe)]),
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
        _make_unit('rpe_series', rpe_series=[str(rpe)]),
        t2w_sdc=False,
        name='dp_rpe_axis',
    )
    split = wf.get_node('split_rpe_groups')
    assert split.inputs.pe_axis == 'j'


def test_drbuddi_never_sends_parser_disabled_flags(tmp_path):
    """Two DRBUDDI options are disabled in TORTOISE's parser; never send them.

    ``--DRBUDDI_start_with_diffeomorphic_for_rigid_reg`` and
    ``--DRBUDDI_disable_initial_rigid`` are both commented out of
    DRBUDDI_parserBase.cxx, along with their getters, so neither can change the
    registration. Worse, DRBUDDI rejects an unrecognised parameter by printing
    "Unknown command line parameter" and then exiting 0 -- nipype records that
    as success and the run dies afterwards collecting the
    bdown_to_bup_rigidtrans.hdf5 that was never written.

    ``sloppy`` must therefore cheapen DRBUDDI through --DRBUDDI_stage alone.
    (``--DRBUDDI_disable_initial_rigid`` would also suppress
    ``bdown_to_bup_rigid_trans_h5``, which DRBUDDIAggregateOutputs dereferences
    unguarded on the rpe_series FA branch.)
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
    disabled = (
        '--DRBUDDI_start_with_diffeomorphic_for_rigid_reg',
        '--DRBUDDI_disable_initial_rigid',
    )

    for sloppy in (True, False):
        cmd = DRBUDDI(sloppy=sloppy, **common).cmdline
        for flag in disabled:
            assert flag not in cmd, f'{flag} is rejected by DRBUDDI (sloppy={sloppy})'

    # sloppy still has to do its job, just through the stage schedule
    assert '--DRBUDDI_stage' in DRBUDDI(sloppy=True, **common).cmdline
    assert '--DRBUDDI_stage' not in DRBUDDI(sloppy=False, **common).cmdline

    # ...and the workflow must not set either trait, under sloppy or not
    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe)
    groups = _make_unit('rpe_series', rpe_series=[str(rpe)])
    config = _base_config()
    try:
        for sloppy in (True, False):
            config.execution.sloppy = sloppy
            wf = _build(groups, t2w_sdc=False, name=f'no_disabled_flags_{sloppy}')
            node = wf.get_node('drbuddi_sdc_wf.drbuddi')
            assert not isdefined(node.inputs.start_with_diffeomorphic_for_rigid_reg)
            assert not isdefined(node.inputs.disable_initial_rigid)
    finally:
        config.execution.sloppy = False


def test_init_diffprep_hmc_wf_pepolar_always_uses_drbuddi(tmp_path):
    """TORTOISE corrects PEPOLAR with DRBUDDI regardless of --pepolar-method.

    The builder no longer rejects TOPUP itself; backend feasibility is owned by
    the grouping validation / config layer, not the workflow builders.
    """
    epi = tmp_path / 'sub-01_epi.nii.gz'
    _write_dummy_nii(epi)
    config = _base_config()
    config.workflow.pepolar_method = 'TOPUP'
    try:
        wf = _build(_make_unit('epi', epi=[str(epi)]), False)
        assert wf.get_node('drbuddi_sdc_wf') is not None
    finally:
        config.workflow.pepolar_method = 'drbuddi'


def test_sloppy_epi_working_res_only_under_sloppy():
    """The working-grid override is emitted for --sloppy and nowhere else.

    A stock (unpatched) TORTOISE rejects --epi_working_res, so a normal run must
    not emit it. 2.5mm is a deliberate speed choice for smoke tests, not a
    validated registration resolution.
    """
    from qsiprep.interfaces.tortoise import SLOPPY_EPI_WORKING_RES, sloppy_epi_working_res

    config = _base_config()
    try:
        config.execution.sloppy = False
        assert sloppy_epi_working_res() == {}

        config.execution.sloppy = True
        assert sloppy_epi_working_res() == {'epi_working_res': SLOPPY_EPI_WORKING_RES}
    finally:
        config.execution.sloppy = False


def test_drbuddi_epi_working_res_on_the_command_line(tmp_path):
    """The trait renders as --epi_working_res, and is absent when unset."""
    from qsiprep.interfaces.tortoise import DRBUDDI

    for name in ('up.nii', 'up.bmtxt', 'up.json', 'down.nii'):
        (tmp_path / name).write_text('')
    kwargs = {
        'blip_up_image': str(tmp_path / 'up.nii'),
        'blip_up_bmat': str(tmp_path / 'up.bmtxt'),
        'blip_up_json': str(tmp_path / 'up.json'),
        'blip_down_image': str(tmp_path / 'down.nii'),
        'fieldmap_type': 'rpe_series',
    }
    assert '--epi_working_res' not in DRBUDDI(**kwargs).cmdline
    assert '--epi_working_res 2.5' in DRBUDDI(epi_working_res=2.5, **kwargs).cmdline


def test_sloppy_reaches_the_drbuddi_node(tmp_path):
    """--sloppy propagates all the way to the DRBUDDI node's command line."""
    from qsiprep.interfaces.tortoise import SLOPPY_EPI_WORKING_RES

    # GatherDRBUDDIInputs takes epi_fmaps as a File trait, so the reverse-PE
    # series has to exist on disk for the workflow to build.
    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe)
    groups = _make_unit('rpe_series', rpe_series=[str(rpe)])

    config = _base_config()
    try:
        config.execution.sloppy = True
        wf = _build(groups, t2w_sdc=False, name='sloppy_on')
        node = wf.get_node('drbuddi_sdc_wf.drbuddi')
        assert node.inputs.epi_working_res == SLOPPY_EPI_WORKING_RES

        config.execution.sloppy = False
        wf_off = _build(groups, t2w_sdc=False, name='sloppy_off')
        node_off = wf_off.get_node('drbuddi_sdc_wf.drbuddi')
        assert not isdefined(node_off.inputs.epi_working_res)
    finally:
        config.execution.sloppy = False


def test_t2wreg_is_recognised_as_sdc_for_reporting():
    """The reportlet gate must recognise T2Wreg, which carries no fieldmap.

    Before this, ``fieldmap_type is None`` fell through both gates in
    ``init_dwi_preproc_wf`` and T2Wreg silently produced no SDC figure, while the
    identical correction tagged ``syn`` did produce one.
    """
    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.workflows.dwi.base import _doing_t2wreg

    config = _base_config()
    try:
        config.workflow.hmc_model = 'tortoise'
        t2w = ['/data/sub-01_T2w.nii.gz']
        fieldmapless = make_preproc_unit(['/data/sub-01_dwi.nii.gz'], anat_files=t2w)
        assert _doing_t2wreg(fieldmapless, '/path/to/T2w.nii.gz') is True

        # No T2w -> no T2Wreg -> nothing to show.
        assert _doing_t2wreg(_make_unit(None), '') is False
        # A measured fieldmap goes through its own SDC reports instead.
        rpe = _make_unit('rpe_series', rpe_series=['/data/sub-01_dir-PA_dwi.nii.gz'])
        assert _doing_t2wreg(rpe, '/path/to/T2w.nii.gz') is False
        epi = _make_unit('epi', epi=['/data/sub-01_epi.nii.gz'])
        assert _doing_t2wreg(epi, '/path/to/T2w.nii.gz') is False

        # Other methods do not run T2Wreg at all.
        config.workflow.hmc_model = 'eddy'
        fieldmapless = make_preproc_unit(['/data/sub-01_dwi.nii.gz'], anat_files=t2w)
        assert _doing_t2wreg(fieldmapless, '/path/to/T2w.nii.gz') is False
    finally:
        config.workflow.hmc_model = 'eddy'


def test_t2wreg_reportlet_desc_is_registered_in_the_report_spec():
    """A desc absent from reports-spec.yml is written to disk but never shown."""
    import yaml

    from qsiprep.data import load as load_data

    spec = yaml.safe_load(load_data('reports-spec.yml').read_text())
    descs = set()
    for section in spec['sections']:
        for r in section.get('reportlets', []):
            bids = r.get('bids')
            if not isinstance(bids, dict):
                continue
            desc = bids.get('desc')
            # some entries carry a list of descs
            descs.update(desc if isinstance(desc, list) else [desc])
    assert 'sdcT2w' in descs


def test_non_shelled_rpe_series_uses_the_stock_drbuddi_path(tmp_path):
    """Non-shelled reverse-PE series no longer auto-synthesize.

    The old behaviour routed CS-DSI data to a qsiprep-side predicted-shell
    workflow, on the theory that DRBUDDI cannot tensor-fit a usable [b0, FA] from
    a q-space grid. Measurement contradicted that: the plain-tensor FA resolves
    corpus callosum, internal capsule and corona radiata on real HASC55 data and
    lands within ~0.002 correlation of a synthesized-shell target for half the
    runtime. Synthesis is now opt-in via --diffprep-config.
    """
    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe)

    _base_config()
    wf = _build(
        _make_unit('rpe_series', rpe_series=[str(rpe)]),
        t2w_sdc=False,
        name='dp_nonshelled',
    )
    # Stock DRBUDDI workflow, not the removed synthesis path.
    assert wf.get_node('drbuddi_sdc_wf') is not None
    assert wf.get_node('predict_up_shell') is None
    assert wf.get_node('predict_down_shell') is None
    # The per-direction DIFFPREP split still runs -- that is independent of
    # how DRBUDDI's target is built.
    assert wf.get_node('recombine_rpe_groups') is not None


def test_drbuddi_synth_shell_is_opt_in(tmp_path):
    """The synthesis flags reach DRBUDDI only when explicitly configured.

    A stock (unpatched) TORTOISE rejects --DRBUDDI_synth_shell_bval, so absence
    must mean the flag is never emitted -- not emitted as 0.
    """
    import json as _json

    rpe = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    _write_dummy_nii(rpe)
    groups = _make_unit('rpe_series', rpe_series=[str(rpe)])

    config = _base_config()
    try:
        # default: off, and no trait set at all
        wf = _build(groups, t2w_sdc=False, name='dp_synth_off')
        node = wf.get_node('drbuddi_sdc_wf.drbuddi')
        # cmdline is not buildable here -- blip_up_image arrives by connection
        # at runtime. Flag rendering is covered by test_drbuddi_synth_shell_cmdline.
        assert not isdefined(node.inputs.synth_shell_bval)
        assert not isdefined(node.inputs.synth_shell_ndirs)

        # opt-in
        cfg = tmp_path / 'synth_on.json'
        cfg.write_text(_json.dumps({'drbuddi_synth_shell_bval': 1000}))
        config.workflow.diffprep_config = str(cfg)
        wf_on = _build(groups, t2w_sdc=False, name='dp_synth_on')
        node_on = wf_on.get_node('drbuddi_sdc_wf.drbuddi')
        assert node_on.inputs.synth_shell_bval == 1000
        assert node_on.inputs.synth_shell_ndirs == 30
    finally:
        config.workflow.diffprep_config = None


def test_drbuddi_synth_shell_cmdline(tmp_path):
    """The traits render as the TORTOISE flags."""
    from qsiprep.interfaces.tortoise import DRBUDDI

    for name in ('up.nii', 'up.bmtxt', 'up.json', 'down.nii'):
        (tmp_path / name).write_text('')
    kwargs = {
        'blip_up_image': str(tmp_path / 'up.nii'),
        'blip_up_bmat': str(tmp_path / 'up.bmtxt'),
        'blip_up_json': str(tmp_path / 'up.json'),
        'blip_down_image': str(tmp_path / 'down.nii'),
        'fieldmap_type': 'rpe_series',
    }
    assert '--DRBUDDI_synth_shell_bval' not in DRBUDDI(**kwargs).cmdline
    cmd = DRBUDDI(synth_shell_bval=1000, synth_shell_ndirs=30, **kwargs).cmdline
    assert '--DRBUDDI_synth_shell_bval 1000' in cmd
    assert '--DRBUDDI_synth_shell_ndirs 30' in cmd


def test_diffprep_node_declares_its_threads():
    """DIFFPREP should declare its threads, consistent with the other TORTOISE nodes.

    Caveat, measured rather than assumed: OMP_NUM_THREADS does NOT actually bound
    TORTOISEProcess (~1893% CPU at OMP_NUM_THREADS=4 versus ~2071% unconstrained
    on a 24-core host), and neither does ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS.
    This keeps the declaration consistent and correct for tools that do honour it;
    it does not make the node's CPU use match n_procs.
    """
    config = _base_config()
    config.nipype.omp_nthreads = 7
    wf = _build(_make_unit(None), t2w_sdc=False, name='threads_declared')
    node = wf.get_node('diffprep')
    assert node.inputs.num_threads == 7
    # nipype's own accounting must match what the tool is allowed to use
    assert node.n_procs == 7


def test_diffprep_interface_exports_omp_num_threads_to_the_subprocess():
    """The env var goes on inputs.environ (passed to the child), not os.environ."""
    from qsiprep.interfaces.tortoise import DIFFPREP

    assert DIFFPREP(num_threads=5).inputs.environ.get('OMP_NUM_THREADS') == '5'


def test_rpe_series_diffprep_nodes_also_declare_threads(tmp_path):
    """The per-PE-direction DIFFPREP nodes share diffprep_kwargs, so they inherit it."""
    import nibabel as nb
    import numpy as np

    config = _base_config()
    config.nipype.omp_nthreads = 6
    # the rpe path needs the partner series to exist
    partner = tmp_path / 'sub-01_dir-PA_dwi.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4, 6), dtype='float32'), np.eye(4)).to_filename(str(partner))
    rpe = _make_unit('rpe_series', rpe_series=[str(partner)])

    wf = _build(rpe, t2w_sdc=False, name='rpe_threads')
    group_nodes = [n for n in wf._get_all_nodes() if n.name.startswith('diffprep_g')]
    assert group_nodes, 'no per-group DIFFPREP nodes found'
    for node in group_nodes:
        assert node.inputs.num_threads == 6


def test_diffprep_passes_ncores_to_tortoise():
    """--ncores is the only knob that actually bounds TORTOISEProcess.

    num_threads only sets OMP_NUM_THREADS, which TORTOISE overrides via
    omp_set_num_threads().
    """
    config = _base_config()
    config.nipype.omp_nthreads = 8
    wf = _build(_make_unit(None), t2w_sdc=False, name='ncores_wired')
    node = wf.get_node('diffprep')
    assert node.inputs.ncores == 8
    # nipype's accounting and the process's real budget must agree
    assert node.n_procs == 8
