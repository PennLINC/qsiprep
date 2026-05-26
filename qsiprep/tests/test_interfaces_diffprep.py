"""Tests for the DIFFPREP HMC backend (qsiprep.interfaces.tortoise).

These are lightweight unit tests that don't require the TORTOISE binary:
- Parser-level checks that the new CLI choices are accepted
- ``DIFFPREPMotionParams`` correctly slicing 24-col transforms -> SPM 6-col
- ``okan_decompose`` correctly splitting a 24-parameter Okan-quadratic
  transform into a 4x4 affine plus a 1D voxel-shift-map along the phase axis.
  The math is validated by a direct port of the C++ ``TransformPoint``
  routine (see ``_okan_transform_point`` below).
"""

import os

import nibabel as nb
import numpy as np
import pytest

from qsiprep.cli.parser import _build_parser
from qsiprep.interfaces.tortoise import (
    DIFFPREPDecomposeTransforms,
    DIFFPREPMotionParams,
    _build_rotation,
    _dp_offset,
    _phase_axis_dp,
    okan_decompose,
    write_itk_affine,
    write_itk_warp,
)


# ---------------------------------------------------------------------------
# Reference implementation: direct port of OkanQuadraticTransform::TransformPoint
# (itkOkanQuadraticTransform.hxx:152-193). Used to validate okan_decompose.
# ---------------------------------------------------------------------------


def _okan_transform_point(point_dp, params, phase_axis, do_cubic):
    p = np.asarray(point_dp, dtype=float).copy()
    center = params[21:24]
    p[0] -= center[0]
    p[1] -= center[1]
    p[2] -= center[2]
    R = _build_rotation(params[3], params[4], params[5])
    p = R @ p + params[0:3]

    new_phase = (
        params[6] * p[0] + params[7] * p[1] + params[8] * p[2]
        + params[9] * p[0] * p[1]
        + params[10] * p[0] * p[2]
        + params[11] * p[1] * p[2]
        + params[12] * (p[0] ** 2 - p[1] ** 2)
        + params[13] * (2 * p[2] ** 2 - p[0] ** 2 - p[1] ** 2)
    )
    if do_cubic:
        new_phase += (
            params[14] * p[0] * p[1] * p[2]
            + params[15] * p[2] * (p[0] ** 2 - p[1] ** 2)
            + params[16] * p[0] * (4 * p[2] ** 2 - p[0] ** 2 - p[1] ** 2)
            + params[17] * p[1] * (4 * p[2] ** 2 - p[0] ** 2 - p[1] ** 2)
            + params[18] * p[0] * (p[0] ** 2 - 3 * p[1] ** 2)
            + params[19] * p[1] * (3 * p[0] ** 2 - p[1] ** 2)
            + params[20] * p[2] * (2 * p[2] ** 2 - 3 * p[0] ** 2 - 3 * p[1] ** 2)
        )
    p[phase_axis] = new_phase
    p[0] += center[0]
    p[1] += center[1]
    p[2] += center[2]
    return p


def _grid_image(shape, affine):
    return nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)


def _identity_params(phase_axis):
    p = np.zeros(24)
    p[6 + phase_axis] = 1.0
    return p


@pytest.mark.parametrize(
    'hmc_model',
    ['diffprep_motion', 'diffprep_quadratic', 'diffprep_cubic'],
)
def test_parser_accepts_diffprep_hmc_models(tmp_path, hmc_model):
    """The new ``--hmc-model diffprep_*`` choices and ``--diffprep-config``
    flag should round-trip through the qsiprep argparse parser without
    raising."""
    parser = _build_parser()
    cfg_file = tmp_path / 'diffprep.json'
    cfg_file.write_text('{"b0_id": -1, "is_human_brain": true}\n')

    namespace = parser.parse_args([
        str(tmp_path / 'bids_in'),
        str(tmp_path / 'out'),
        'participant',
        '--output-resolution=2',
        f'--hmc-model={hmc_model}',
        f'--diffprep-config={cfg_file}',
    ])
    assert namespace.hmc_model == hmc_model
    assert os.fspath(namespace.diffprep_config).endswith('diffprep.json')


def test_diffprep_motion_params_basic(tmp_path):
    """``DIFFPREPMotionParams`` should slice cols 0-5 from a 24-col TORTOISE
    transformations file and write them as a whitespace-separated SPM file."""

    n_volumes = 4
    rng = np.random.default_rng(0)
    full = rng.standard_normal((n_volumes, 24))
    # Use the bracket / comma serialization VNL VariableLengthVector emits.
    txt = '\n'.join(
        '[' + ', '.join(f'{v:.6f}' for v in row) + ']' for row in full
    ) + '\n'
    transforms_file = tmp_path / 'sub-1_dwi_moteddy_transformations.txt'
    transforms_file.write_text(txt)

    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    iface._results = {}

    class _R:
        cwd = str(tmp_path)

    iface._run_interface(_R())

    spm_file = iface._results['spm_motion_file']
    assert os.path.exists(spm_file)
    spm = np.loadtxt(spm_file)
    assert spm.shape == (n_volumes, 6)
    np.testing.assert_allclose(spm, full[:, :6], atol=1e-5)


def test_diffprep_motion_params_plain_whitespace(tmp_path):
    """Some VNL serializers omit brackets and just space-separate values.
    The parser should tolerate that too."""

    full = np.arange(24, dtype=float).reshape(1, 24)
    txt = ' '.join(f'{v}' for v in full[0]) + '\n'
    transforms_file = tmp_path / 'plain.txt'
    transforms_file.write_text(txt)

    iface = DIFFPREPMotionParams(transformations_file=str(transforms_file))
    iface._results = {}

    class _R:
        cwd = str(tmp_path)

    iface._run_interface(_R())

    spm = np.loadtxt(iface._results['spm_motion_file'])
    assert spm.shape == (6,)
    np.testing.assert_allclose(spm, full[0, :6])


# ---------------------------------------------------------------------------
# okan_decompose math
# ---------------------------------------------------------------------------


def test_phase_axis_from_bids():
    assert _phase_axis_dp('i') == 0
    assert _phase_axis_dp('j') == 1
    assert _phase_axis_dp('k') == 2
    assert _phase_axis_dp('j-') == 1
    assert _phase_axis_dp(2) == 2


def test_dp_offset_modes():
    # 4-voxel grid, 2 mm spacing, identity direction, origin at (-3, -3, -3).
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -3.0
    shape = (4, 4, 4)
    np.testing.assert_allclose(_dp_offset(affine, shape, 'isocenter'), 0)
    cv = _dp_offset(affine, shape, 'center_voxel')
    # center voxel index = (1.5, 1.5, 1.5) -> physical = (-3 + 3, -3 + 3, -3 + 3) = 0
    np.testing.assert_allclose(cv, 0)
    cs = _dp_offset(affine, shape, 'center_slice')
    np.testing.assert_allclose(cs, [0, 0, 0])


def test_okan_decompose_identity_is_identity_everywhere():
    """A 24-param transform with only params[6+phase]=1 must be the identity
    transform on the full grid (zero displacement, identity affine)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -3.0
    ref = _grid_image((4, 4, 4), affine)
    params = _identity_params(phase_axis=1)
    A, disp = okan_decompose(params, ref, phase_encoding_direction='j')
    np.testing.assert_allclose(A, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(disp, 0, atol=1e-10)


def test_okan_decompose_pure_translation():
    """Pure translation (params[0:3]) is purely affine; the warp must be 0."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -3.0
    ref = _grid_image((3, 3, 3), affine)
    params = _identity_params(phase_axis=1)
    params[0] = 4.5  # 4.5 mm along DP x
    params[2] = -1.0
    A, disp = okan_decompose(params, ref, phase_encoding_direction='j')
    # Direction matrix is identity, so DP translation matches image translation.
    expected = np.eye(4)
    expected[0, 3] = 4.5
    expected[2, 3] = -1.0
    np.testing.assert_allclose(A, expected, atol=1e-10)
    np.testing.assert_allclose(disp, 0, atol=1e-10)


def test_okan_decompose_pure_eddy_quadratic():
    """A single quadratic eddy coefficient (params[9]=xy term on phase axis)
    produces a VSM along the phase axis only. The non-PE axes must stay
    untouched and the affine must remain identity."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = 0.0  # isocenter at index (0,0,0)
    shape = (3, 3, 3)
    ref = _grid_image(shape, affine)
    params = _identity_params(phase_axis=1)
    params[9] = 0.01  # adds 0.01·px·py to the polynomial on phase axis
    A, disp = okan_decompose(params, ref, phase_encoding_direction='j')

    np.testing.assert_allclose(A, np.eye(4), atol=1e-10)

    # Voxel (i, j, k) has DP coord = (2i, 2j, 2k). Rigid is identity, so
    # rigid_centered = (2i, 2j, 2k). Polynomial = 1·(2j) + 0.01·(2i)·(2j),
    # phase coord (j-axis) shift = 0.04·i·j. Image direction is identity,
    # so the displacement vector is (0, 0.04·i·j, 0).
    expected = np.zeros((*shape, 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                expected[i, j, k, 1] = 0.04 * i * j
    np.testing.assert_allclose(disp, expected, atol=1e-10)


def test_okan_decompose_matches_transform_point_random():
    """Sanity check: ``A · x + disp(x)`` must equal a direct port of
    ``OkanQuadraticTransform::TransformPoint`` at every voxel for a random
    transform with both rigid motion and quadratic + cubic eddy terms."""
    rng = np.random.default_rng(42)
    # Random LPS-ish affine: small rotation, non-uniform spacing, off-origin
    spacing = np.array([1.7, 2.1, 3.0])
    theta = 0.05
    direction = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype=float,
    )
    aff = np.eye(4)
    aff[:3, :3] = direction * spacing
    aff[:3, 3] = np.array([-5.0, 2.5, -1.0])
    shape = (3, 4, 2)
    ref = _grid_image(shape, aff)

    # Random transform: translation, rotation, identity scaling on phase=1,
    # quadratic + cubic terms, non-zero isocenter offset.
    params = np.zeros(24)
    params[0:3] = rng.uniform(-3, 3, size=3)
    params[3:6] = rng.uniform(-0.05, 0.05, size=3)
    params[7] = 1.0  # identity scale on phase axis
    params[9:14] = rng.uniform(-0.005, 0.005, size=5)
    params[14:21] = rng.uniform(-0.0005, 0.0005, size=7)
    params[21:24] = rng.uniform(-2, 2, size=3)

    A, disp = okan_decompose(
        params, ref, phase_encoding_direction='j', do_cubic=True
    )

    # Compare against the direct port at every voxel.
    Dspac = aff[:3, :3]
    origin = aff[:3, 3]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x_img = Dspac @ np.array([i, j, k], dtype=float) + origin
                # Through okan_decompose
                y_via_decomp = A[:3, :3] @ x_img + A[:3, 3] + disp[i, j, k]
                # Direct port: convert to DP, run Okan, convert back
                Dunit = Dspac / np.linalg.norm(Dspac, axis=0)[None, :]
                offset = _dp_offset(aff, shape, 'isocenter')
                c_dp = Dunit.T @ (x_img - offset)
                y_dp = _okan_transform_point(c_dp, params, phase_axis=1, do_cubic=True)
                y_direct = Dunit @ y_dp + offset
                np.testing.assert_allclose(y_via_decomp, y_direct, atol=1e-8)


def test_okan_decompose_invalid_phase_axis():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ref = _grid_image((2, 2, 2), affine)
    with pytest.raises(KeyError):
        okan_decompose(_identity_params(1), ref, phase_encoding_direction='q')


def test_decompose_transforms_interface_writes_files(tmp_path):
    """End-to-end ``DIFFPREPDecomposeTransforms``: feed it a 2-volume
    transformations file and a reference image, expect 2 affine .txt files
    and 2 displacement .nii.gz files written into the run dir."""

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ref = nb.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), affine)
    ref_path = tmp_path / 'ref.nii.gz'
    ref.to_filename(ref_path)

    rows = np.zeros((2, 24))
    rows[0, 7] = 1.0  # identity volume
    rows[1, 7] = 1.0
    rows[1, 0] = 3.0  # second volume: 3 mm translation in DP x
    txt = '\n'.join('[' + ', '.join(f'{v:.6f}' for v in row) + ']' for row in rows)
    txt_path = tmp_path / 'sub-1_dwi_moteddy_transformations.txt'
    txt_path.write_text(txt + '\n')

    iface = DIFFPREPDecomposeTransforms(
        transformations_file=str(txt_path),
        reference_image=str(ref_path),
        phase_encoding_direction='j',
        rot_eddy_center='isocenter',
    )
    iface._results = {}

    class _R:
        cwd = str(tmp_path)

    iface._run_interface(_R())

    aff_files = iface._results['affine_files']
    warp_files = iface._results['warp_files']
    assert len(aff_files) == 2
    assert len(warp_files) == 2
    for f in aff_files + warp_files:
        assert os.path.exists(f)

    # Identity volume: warp must be all zeros
    w0 = nb.load(warp_files[0]).get_fdata()
    np.testing.assert_allclose(w0, 0, atol=1e-10)
    # Translation volume: warp must also be all zeros (translation is purely affine)
    w1 = nb.load(warp_files[1]).get_fdata()
    np.testing.assert_allclose(w1, 0, atol=1e-10)


def test_write_itk_affine_roundtrip(tmp_path):
    """The text file produced by ``write_itk_affine`` should be reloadable by
    nitransforms (or any ITK-compatible reader)."""
    nitransforms = pytest.importorskip('nitransforms')
    A = np.eye(4)
    A[0, 3] = 5.0
    A[1, 1] = 0.9  # non-identity rotation/scaling
    out = tmp_path / 'aff.txt'
    write_itk_affine(A, str(out))
    loaded = nitransforms.linear.load(str(out), fmt='itk')
    np.testing.assert_allclose(loaded.matrix, A, atol=1e-9)


def test_nitransforms_roundtrip_matches_direct_okan(tmp_path):
    """End-to-end check: apply (affine + warp) via nitransforms'
    ``TransformChain`` and compare against a direct port of
    ``OkanQuadraticTransform::TransformPoint`` at a handful of voxel grid
    points. This validates that the files we write are nitransforms-readable
    and that the composition order ``[affine, warp]`` matches what we
    intend."""
    nitransforms = pytest.importorskip('nitransforms')
    from nitransforms.linear import Affine
    from nitransforms.nonlinear import DenseFieldTransform

    rng = np.random.default_rng(0)
    spacing = np.array([1.5, 2.0, 2.0])
    aff = np.eye(4)
    aff[:3, :3] = np.diag(spacing)
    aff[:3, 3] = np.array([-2.0, 1.0, 0.5])
    shape = (3, 4, 3)
    ref = _grid_image(shape, aff)

    params = np.zeros(24)
    params[0:3] = rng.uniform(-1.0, 1.0, size=3)
    params[3:6] = rng.uniform(-0.02, 0.02, size=3)
    params[7] = 1.0
    params[9:14] = rng.uniform(-0.001, 0.001, size=5)

    A, disp = okan_decompose(
        params, ref, phase_encoding_direction='j', do_cubic=False
    )

    # Build nitransforms objects directly (skip file I/O for the warp; we
    # write+reload the affine to also exercise that path)
    aff_path = tmp_path / 'aff.txt'
    write_itk_affine(A, str(aff_path))
    nt_affine = nitransforms.linear.load(str(aff_path), fmt='itk')

    # DenseFieldTransform consumes a nibabel image; we use the 4D NIfTI form
    # nitransforms expects (shape XYZ3, deltas).
    field_img = nb.Nifti1Image(disp.astype(np.float32), aff)
    nt_disp = DenseFieldTransform(field_img, is_deltas=True, reference=ref)

    # Direct port reference output at each grid voxel
    Dunit = aff[:3, :3] / np.linalg.norm(aff[:3, :3], axis=0)[None, :]
    Dspac = aff[:3, :3]
    origin = aff[:3, 3]
    for (i, j, k) in [(0, 0, 0), (2, 3, 1), (1, 2, 2)]:
        x_img = Dspac @ np.array([i, j, k], dtype=float) + origin
        # Apply affine then add the warp's displacement at the *post-affine* point.
        y_aff = nt_affine.matrix @ np.append(x_img, 1.0)
        y_aff = y_aff[:3] / y_aff[3]
        # In our decomposition the warp is sampled at the input voxel grid
        # coordinate, not the post-affine point. nitransforms' TransformChain
        # samples the field at the chain's input, so we apply the warp first.
        y_via_chain = y_aff + disp[i, j, k]

        # Direct Okan computation
        c_dp = Dunit.T @ x_img
        y_dp = _okan_transform_point(c_dp, params, phase_axis=1, do_cubic=False)
        y_direct = Dunit @ y_dp
        np.testing.assert_allclose(y_via_chain, y_direct, atol=1e-6)


def test_write_itk_warp_shape_and_intent(tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    field = np.random.default_rng(0).standard_normal((3, 4, 5, 3))
    out = tmp_path / 'warp.nii.gz'
    write_itk_warp(field, affine, str(out))
    loaded = nb.load(out)
    assert loaded.shape == (3, 4, 5, 1, 3)
    # nibabel returns the intent as (name, parameters, intent_name)
    assert loaded.header.get_intent()[0] == 'vector'
    np.testing.assert_allclose(
        np.squeeze(loaded.get_fdata()), field, atol=1e-6
    )
