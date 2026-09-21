"""The SDC displacement field reaches the derivatives, re-expressed in ACPC space.

DRBUDDI estimates susceptibility distortion as an ANTs displacement field in the
corrected DWI frame. These tests check that ``ComposeSDCWarp`` re-expresses it on
the ACPC output grid *as a transform* -- rotating its vectors into ACPC world
coordinates rather than merely resampling the values -- and that the workflow
wires it into every DRBUDDI backend's derivatives with a valid path and sidecar.

The composition tests need ``antsApplyTransforms``/``antsApplyTransformsToPoints``
and are skipped when those binaries are absent, so the file is still collectable
outside the container.
"""

import os
import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep import config

ANTS = shutil.which('antsApplyTransforms')
requires_ants = pytest.mark.skipif(ANTS is None, reason='antsApplyTransforms not installed')

# NIfTI stores the affine in RAS+; ITK/ANTs store transforms and displacement
# vectors in LPS. Working entirely in LPS keeps the expected R.d unambiguous.
LPS = np.diag([-1.0, -1.0, 1.0])


def _cfg():
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = True
    config.execution.output_dir = '/tmp/qsiprep_sdc_warp_test_out'
    config.workflow.output_resolution = 2.0
    return config


def _rotation(axis, degrees):
    r = np.deg2rad(degrees)
    c, s = np.cos(r), np.sin(r)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _write_affine(path, matrix, translation):
    """Write an ITK affine .mat (its matrix/translation are in LPS)."""
    import SimpleITK as sitk

    aff = sitk.AffineTransform(3)
    aff.SetMatrix(matrix.ravel().tolist())
    aff.SetTranslation([float(t) for t in translation])
    sitk.WriteTransform(aff, str(path))
    return str(path)


def _write_sdc_field(path, displacement, n=20, origin=(-9.5, -9.5, -9.5), vary=0.01):
    """A DWI-frame SDC displacement field: constant ``displacement`` (LPS mm) plus
    a small spatially varying term, written with the NIFTI_INTENT_VECTOR code so
    antsApplyTransforms reads it as a displacement field rather than zeros."""
    data = np.zeros((n, n, n, 1, 3), dtype='float32')
    data[..., 0, :] = displacement
    # vary the first component along i so a pure grid resample (vs a transform
    # composition) would be detectable, and to exercise interpolation.
    ii = np.arange(n)[:, None, None]
    data[..., 0, 0] += np.broadcast_to(vary * ii, (n, n, n)).astype('float32')
    # Encode the LPS origin as a nibabel (RAS) affine: flip x, y.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [-origin[0], -origin[1], origin[2]]
    img = nb.Nifti1Image(data, affine)
    img.header.set_intent('vector')
    img.to_filename(str(path))
    return str(path)


def _write_ref(path, R, n=20, origin=(-9.5, -9.5, -9.5)):
    """An ACPC reference grid = the DWI grid rotated by ``R`` about its center."""
    import SimpleITK as sitk

    ref = sitk.Image((n, n, n), sitk.sitkFloat32)
    ref.SetOrigin(origin)
    ref.SetDirection(R.ravel().tolist())
    sitk.WriteImage(ref, str(path))
    return str(path)


def _oblique_case(tmp_path, R, displacement=(0.3, 2.0, -0.5), vary=0.01):
    """Build coreg (A^-1 = ACPC->DWI), W and the ACPC grid for rotation ``R``.

    ``A`` (DWI->ACPC) is ``R`` about the shared grid center, so coreg = R^-1.
    Every ACPC voxel maps back into the DWI grid, guaranteeing field coverage.
    """
    d = np.array(displacement, dtype='float64')
    coreg = _write_affine(tmp_path / 'coreg.mat', R.T, [0.0, 0.0, 0.0])
    warp = _write_sdc_field(tmp_path / 'W.nii.gz', d, vary=vary)
    ref = _write_ref(tmp_path / 'ref.nii.gz', R)
    return coreg, warp, ref, d


def _run_compose(coreg, warp, ref, cwd):
    from qsiprep.interfaces.gradients import ComposeSDCWarp

    os.makedirs(cwd, exist_ok=True)
    iface = ComposeSDCWarp(sdc_warp=warp, to_template_transforms=[coreg], reference_image=ref)
    return iface.run(cwd=cwd).outputs.sdc_warp_to_template


def _field_components(path):
    """Load an ITK displacement field's LPS vector components as (i, j, k, 3)."""
    data = np.asarray(nb.load(path).dataobj)
    return data.reshape(data.shape[:3] + (3,))


# ---------------------------------------------------------------------------
# Construction tests (no external binaries)
# ---------------------------------------------------------------------------


def test_sdc_warp_stage_names_drops_dwi_native_stages():
    """The SDC-warp sub-chain keeps only the corrected-DWI-frame -> ACPC stages."""
    from qsiprep.interfaces.gradients import ComposeTransforms

    stages = list(ComposeTransforms._TRANSFORM_STAGES)
    kept = ComposeTransforms._sdc_warp_stage_names(stages)
    assert 'hmc' not in kept
    assert 'gradwarp' not in kept
    assert 'fieldwarp' not in kept
    assert kept == ['to b=0 affine', 'to b=0 warp', 'b=0 to T1w']
    # MNI stages are excluded because the derivative stays in ACPC.
    assert ComposeTransforms._sdc_warp_stage_names(['fieldwarp', 'b=0 to T1w', 'mni warp']) == [
        'b=0 to T1w'
    ]


def test_compose_transforms_exposes_the_sdc_warp_subchain(tmp_path):
    """ComposeTransforms emits the corrected-DWI-frame -> ACPC sub-chain.

    With only a coregistration transform present, that sub-chain is the coreg
    itself, for volume 0 -- what ComposeSDCWarp conjugates the fieldwarp with.
    """
    import SimpleITK as sitk

    from qsiprep.interfaces.gradients import ComposeTransforms

    dwi = str(tmp_path / 'sub-01_dwi.nii.gz')
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(dwi)
    ref = str(tmp_path / 'ref.nii.gz')
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(ref)
    coreg = str(tmp_path / 'coreg.mat')
    sitk.WriteTransform(sitk.AffineTransform(3), coreg)

    result = ComposeTransforms(
        dwi_files=[dwi], reference_image=ref, hmcsdc_dwi_ref_to_t1w_affine=coreg
    ).run(cwd=str(tmp_path))
    # OutputMultiObject unwraps a single-element list to a scalar; the consuming
    # InputMultiObject re-wraps it, so normalise here.
    got = result.outputs.sdc_warp_transforms
    got = got if isinstance(got, list) else [got]
    assert got == [coreg]


def test_trans_wf_builds_compose_sdc_warp_only_when_requested():
    """``write_sdc_warp`` gates the ComposeSDCWarp node and its wiring."""
    _cfg()
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    without = init_dwi_trans_wf(source_file='/data/sub-01_dwi.nii.gz', mem_gb=1)
    assert without.get_node('compose_sdc_warp') is None

    wf = init_dwi_trans_wf(source_file='/data/sub-01_dwi.nii.gz', mem_gb=1, write_sdc_warp=True)
    assert wf.get_node('compose_sdc_warp') is not None
    edges = wf._graph.edges(data=True)
    # It rides the SDC-warp sub-chain (not the full composite) and volume 0's warp.
    assert any(
        u.name == 'compose_transforms'
        and v.name == 'compose_sdc_warp'
        and ('sdc_warp_transforms', 'to_template_transforms') in d['connect']
        for u, v, d in edges
    )
    assert any(
        u.name == 'inputnode'
        and v.name == 'compose_sdc_warp'
        and any(dst == 'sdc_warp' for _, dst in d['connect'])
        for u, v, d in edges
    )
    assert any(
        u.name == 'compose_sdc_warp'
        and v.name == 'outputnode'
        and ('sdc_warp_to_template', 'sdc_warp_to_template') in d['connect']
        for u, v, d in edges
    )


def test_derivatives_wf_writes_sdc_warp_only_with_meta():
    """The SDC-warp datasink appears only when sidecar metadata is supplied."""
    _cfg()
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    without = init_dwi_derivatives_wf(source_file='/data/sub-01_dwi.nii.gz')
    assert without.get_node('ds_sdc_warp_t1') is None

    meta = {'EstimationMethod': 'DRBUDDI', 'Description': 'the field'}
    wf = init_dwi_derivatives_wf(source_file='/data/sub-01_dwi.nii.gz', sdc_warp_meta=meta)
    ds = wf.get_node('ds_sdc_warp_t1')
    assert ds is not None
    assert ds.inputs.suffix == 'xfm'
    assert ds.inputs.mode == 'image'
    assert ds.inputs.desc == 'sdc'
    assert getattr(ds.inputs, 'from') == 'dwiref'
    assert ds.inputs.to == 'ACPC'
    assert ds.inputs.meta_dict == meta
    # No Hz-era keys leak in.
    assert 'Units' not in meta


def test_sdc_warp_datasink_builds_a_dwi_xfm_path(tmp_path):
    """A dwi ``xfm`` suffix must resolve to a from/to/mode/desc path template."""
    from qsiprep.interfaces.bids import DerivativesDataSink

    src = tmp_path / 'sub-01_ses-1_acq-HBCD_run-01_dwi.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(src))
    field = tmp_path / 'sdc_warp_to_template.nii.gz'
    vec = nb.Nifti1Image(np.zeros((4, 4, 4, 1, 3), dtype='float32'), np.eye(4))
    vec.header.set_intent('vector')
    vec.to_filename(str(field))

    ds = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file=str(src),
        mode='image',
        suffix='xfm',
        desc='sdc',
        extension='.nii.gz',
        compress=True,
        meta_dict={'EstimationMethod': 'DRBUDDI'},
        **{'from': 'dwiref', 'to': 'ACPC'},
    )
    ds.inputs.in_file = str(field)
    out = ds.run().outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert '/dwi/' in out
    assert out.endswith('_from-dwiref_to-ACPC_mode-image_desc-sdc_xfm.nii.gz')


# ---------------------------------------------------------------------------
# Composition tests (need ANTs): the vectors must be rotated, not just moved.
# ---------------------------------------------------------------------------

_OBLIQUE = _rotation('x', 30) @ _rotation('y', 20) @ _rotation('z', 15)


@requires_ants
def test_compose_sdc_warp_is_a_valid_vector_field(tmp_path):
    """The emitted derivative is a 5-D ITK displacement field (vector intent)."""
    coreg, warp, ref, _ = _oblique_case(tmp_path, _OBLIQUE)
    out = _run_compose(coreg, warp, ref, str(tmp_path / 'run'))
    img = nb.load(out)
    assert img.ndim == 5
    assert img.shape[3] == 1
    assert img.shape[4] == 3
    assert img.header.get_intent()[0] == 'vector'


@requires_ants
def test_compose_sdc_warp_rotates_vectors_for_oblique_affine(tmp_path):
    """Emitted vectors equal R.d for an oblique affine, and d when axis-aligned.

    A grid resample (vectors left un-rotated) would give d in both cases, so the
    oblique/axis-aligned pair proves the composition actually rotates vectors --
    and that the oblique case exercises a rotation (R.d differs from d).
    """
    d = np.array([0.3, 2.0, -0.5])

    # No spatial variation here, so the whole interior is ~R.d.
    (tmp_path / 'obl').mkdir()
    coreg_o, warp_o, ref_o, _ = _oblique_case(tmp_path / 'obl', _OBLIQUE, vary=0.0)
    obl = _field_components(_run_compose(coreg_o, warp_o, ref_o, str(tmp_path / 'obl_run')))

    (tmp_path / 'axis').mkdir()
    coreg_a, warp_a, ref_a, _ = _oblique_case(tmp_path / 'axis', np.eye(3), vary=0.0)
    axis = _field_components(_run_compose(coreg_a, warp_a, ref_a, str(tmp_path / 'axis_run')))

    interior_obl = obl[8:12, 8:12, 8:12].reshape(-1, 3).mean(0)
    interior_axis = axis[8:12, 8:12, 8:12].reshape(-1, 3).mean(0)

    # The oblique case genuinely exercises rotation.
    assert not np.allclose(_OBLIQUE @ d, d, atol=0.2)
    # Emitted vectors are the rotated displacement, not the raw one.
    np.testing.assert_allclose(interior_obl, _OBLIQUE @ d, atol=0.1)
    assert not np.allclose(interior_obl, d, atol=0.1)
    # Axis-aligned: no rotation, so the emitted vectors are d itself.
    np.testing.assert_allclose(interior_axis, d, atol=0.1)


@requires_ants
def test_compose_sdc_warp_point_round_trip(tmp_path):
    """As a transform, the emitted field maps q -> A(W(A^-1(q))).

    Sampling the composite displacement field at an ACPC landmark must send it to
    the same place as applying the SDC warp in the DWI frame and pushing forward
    through the affine -- to sub-voxel tolerance.
    """
    import SimpleITK as sitk

    coreg, warp, ref, d = _oblique_case(tmp_path, _OBLIQUE, vary=0.01)
    out = _run_compose(coreg, warp, ref, str(tmp_path / 'run'))

    emitted = sitk.DisplacementFieldTransform(
        sitk.Cast(sitk.ReadImage(out), sitk.sitkVectorFloat64)
    )
    A = sitk.AffineTransform(3)  # DWI -> ACPC = R about the origin
    A.SetMatrix(_OBLIQUE.ravel().tolist())
    A_inv = A.GetInverse()
    W = sitk.DisplacementFieldTransform(sitk.Cast(sitk.ReadImage(warp), sitk.sitkVectorFloat64))

    for acpc_point in ([0.0, 0.0, 0.0], [2.0, -3.0, 1.0], [-4.0, 1.0, 2.0]):
        dwi_point = np.array(A_inv.TransformPoint(acpc_point))
        expected = np.array(A.TransformPoint(W.TransformPoint(dwi_point.tolist())))
        got = np.array(emitted.TransformPoint(acpc_point))
        np.testing.assert_allclose(got, expected, atol=1e-2)


@requires_ants
def test_compose_sdc_warp_reproduces_pipeline_correction(tmp_path):
    """Unwarping a distorted b0 with the emitted field matches the pipeline's own
    correction from the same sdc_warp (guards direction and ordering)."""
    import subprocess

    import SimpleITK as sitk
    from scipy.ndimage import gaussian_filter

    coreg, warp, ref, _ = _oblique_case(tmp_path, _OBLIQUE, vary=0.01)
    out = _run_compose(coreg, warp, ref, str(tmp_path / 'run'))

    # A high-contrast structured b0 on the DWI grid (bright blobs on a dark
    # background) so the whole-image correlation is driven by real structure,
    # not interpolation noise on a near-flat field.
    phantom = np.zeros((20, 20, 20), dtype='float32')
    for cx, cy, cz, amp in [(6, 7, 8, 1.0), (13, 12, 10, 0.7), (9, 14, 13, 0.5), (14, 6, 6, 0.9)]:
        phantom[cx, cy, cz] = amp
    phantom = gaussian_filter(phantom, 1.3)
    phantom /= phantom.max()
    dwi_img = sitk.GetImageFromArray(phantom)
    dwi_img.SetOrigin((-9.5, -9.5, -9.5))
    dwi_path = str(tmp_path / 'b0.nii.gz')
    sitk.WriteImage(dwi_img, dwi_path)

    def apply(inp, transforms, out_name):
        out_path = str(tmp_path / out_name)
        cmd = [ANTS, '-d', '3', '-i', inp, '-r', ref, '-o', out_path, '-n', 'Linear']
        for t in transforms:
            cmd += ['-t', t]
        subprocess.run(cmd, check=True, capture_output=True)
        return np.asarray(nb.load(out_path).dataobj)

    # Pipeline: resample the distorted DWI to ACPC through coreg then the warp.
    pipeline = apply(dwi_path, [coreg, warp], 'pipeline.nii.gz')
    # Emitted route: bring the distorted DWI to ACPC (coreg only), then unwarp
    # with the emitted ACPC-space field.
    distorted_acpc = str(tmp_path / 'distorted_acpc.nii.gz')
    subprocess.run(
        [ANTS, '-d', '3', '-i', dwi_path, '-r', ref, '-o', distorted_acpc, '-n', 'Linear',
         '-t', coreg],
        check=True, capture_output=True,
    )  # fmt:skip
    emitted = apply(distorted_acpc, [out], 'emitted.nii.gz')

    mask = (pipeline > 0.05) | (emitted > 0.05)
    ncc = np.corrcoef(pipeline[mask], emitted[mask])[0, 1]
    assert ncc > 0.99, f'NCC {ncc:.4f}'
    # The correction is not a no-op: it differs from the un-warped distorted b0.
    distorted = np.asarray(nb.load(distorted_acpc).dataobj)
    assert np.corrcoef(pipeline[mask], distorted[mask])[0, 1] < ncc


def test_invert_displacement_field_round_trips(tmp_path):
    """The helper produces a usable inverse of a displacement field."""
    import SimpleITK as sitk

    from qsiprep.interfaces.gradients import _invert_displacement_field

    warp = _write_sdc_field(tmp_path / 'W.nii.gz', [0.3, 2.0, -0.5], vary=0.02)
    inverse = _invert_displacement_field(warp, str(tmp_path))

    forward = sitk.DisplacementFieldTransform(
        sitk.Cast(sitk.ReadImage(warp), sitk.sitkVectorFloat64)
    )
    backward = sitk.DisplacementFieldTransform(
        sitk.Cast(sitk.ReadImage(inverse), sitk.sitkVectorFloat64)
    )
    # Composing a field with its inverse returns (near) the original point.
    for point in ([0.0, 0.0, 0.0], [2.0, -1.0, 3.0]):
        there = forward.TransformPoint(point)
        back = np.array(backward.TransformPoint(there))
        np.testing.assert_allclose(back, point, atol=0.1)


@requires_ants
def test_compose_sdc_warp_handles_nonlinear_template_stage(tmp_path):
    """A non-linear stage in the to-template chain is inverted, not passed to a
    flag: the interface still emits a valid displacement field on the ACPC grid."""
    coreg = _write_affine(tmp_path / 'coreg.mat', _OBLIQUE.T, [0.0, 0.0, 0.0])
    warp = _write_sdc_field(tmp_path / 'W.nii.gz', [0.3, 2.0, -0.5])
    template_warp = _write_sdc_field(tmp_path / 'tmpl.nii.gz', [0.1, -0.2, 0.15], vary=0.0)
    ref = _write_ref(tmp_path / 'ref.nii.gz', _OBLIQUE)

    from qsiprep.interfaces.gradients import ComposeSDCWarp

    run_cwd = str(tmp_path / 'run')
    os.makedirs(run_cwd)
    # ANTs order: the affine coreg, then the non-linear template warp.
    iface = ComposeSDCWarp(
        sdc_warp=warp, to_template_transforms=[coreg, template_warp], reference_image=ref
    )
    out = iface.run(cwd=run_cwd).outputs.sdc_warp_to_template
    img = nb.load(out)
    assert img.ndim == 5
    assert img.shape[4] == 3
    assert img.header.get_intent()[0] == 'vector'
    assert np.isfinite(np.asarray(img.dataobj)).all()
