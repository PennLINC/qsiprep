"""The SDC displacement reaches the derivatives, re-expressed in ACPC space.

Susceptibility distortion is corrected with a displacement field in the corrected
DWI frame. These tests check that ``ComposeSDCWarp`` re-expresses it on the ACPC
output grid *as a transform* -- rotating its vectors into ACPC world coordinates
rather than merely resampling the values -- and that every SDC method writes it
as a ``space-ACPC_desc-sdc_displacement`` map with a valid path and sidecar.

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
    iface = ComposeSDCWarp(sdc_warps=warp, to_template_transforms=[coreg], reference_image=ref)
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


def _tiny_dwi(path, nvols=4):
    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype='int16'), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    np.savetxt(stem + '.bval', np.array([0] + [1000] * (nvols - 1))[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def test_sdc_warp_source_emits_for_every_standalone_warp_method(tmp_path):
    """GRE and SyN write a standalone warp, so both emit the derivative."""
    from qsiplan.models import CorrectionMethod

    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.utils.sdc import sdc_warp_source

    dwi = _tiny_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    for method, label in (
        (CorrectionMethod.PHASEDIFF, 'GRE fieldmap'),
        (CorrectionMethod.NIPREPS_SYN, 'SyN (fieldmap-less)'),
    ):
        unit = make_preproc_unit([dwi], method=method)
        assert sdc_warp_source(unit, t2w_sdc=False) == ('fieldwarp', label), method
    # No susceptibility correction -> nothing is emitted.
    assert sdc_warp_source(make_preproc_unit([dwi]), t2w_sdc=False) == (None, None)


@pytest.mark.parametrize(
    ('hmc', 'method', 't2w_sdc', 'expected'),
    [
        # DIFFPREP's T2Wreg writes a warp, whether it targets a T2w or SynB0.
        ('tortoise', 'T2WREG', True, ('fieldwarp', 'TORTOISE T2Wreg')),
        ('tortoise', 'SYNB0', False, ('fieldwarp', 'TORTOISE T2Wreg (SynB0)')),
        # A T2w target without a usable T2w (--anat-modality none): T2Wreg is skipped.
        ('tortoise', 'T2WREG', False, (None, None)),
        # Only DIFFPREP runs T2Wreg; eddy and SHORELine leave these uncorrected.
        ('eddy', 'T2WREG', True, (None, None)),
        ('shoreline', 'T2WREG', True, (None, None)),
        # On eddy, SynB0 feeds TOPUP instead.
        ('eddy', 'SYNB0', False, ('topup', 'TOPUP (SynB0)')),
    ],
)
def test_sdc_warp_source_follows_the_t2wreg_stage(
    tmp_path, monkeypatch, hmc, method, t2w_sdc, expected
):
    """The T2Wreg decision comes from the plan's stage, not the estimation method."""
    from qsiplan.models import CorrectionMethod

    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.utils.sdc import sdc_warp_source

    monkeypatch.setattr(config.workflow, 'hmc_method', hmc)
    # The default method selection: other tests leave an --sdc-method behind.
    monkeypatch.setattr(config.workflow, 'sdc_method', None)
    dwi = _tiny_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    # SynB0 synthesizes its b=0 from the T1w; T2Wreg registers to the T2w.
    anat = str(tmp_path / ('sub-01_T1w.nii.gz' if method == 'SYNB0' else 'sub-01_T2w.nii.gz'))
    unit = make_preproc_unit(
        [dwi],
        method=getattr(CorrectionMethod, method),
        estimation_sources=[anat] if method == 'SYNB0' else None,
        anat_files=[anat],
    )
    assert sdc_warp_source(unit, t2w_sdc=t2w_sdc) == expected


def _reverse_pe_unit(tmp_path, dwi_writer=None):
    """A PEPOLAR unit of an AP/PA DWI pair, planned under the configured methods."""
    from qsiplan.models import CorrectionMethod

    from qsiprep.tests.preproc_factory import make_preproc_unit

    write = dwi_writer or _tiny_dwi
    ap = str(write(tmp_path / 'sub-01_dir-AP_dwi.nii.gz'))
    pa = str(write(tmp_path / 'sub-01_dir-PA_dwi.nii.gz'))
    unit = make_preproc_unit(
        [ap, pa], method=CorrectionMethod.PEPOLAR, pe_dirs={ap: 'j-', pa: 'j'}
    )
    return unit, ap


@pytest.mark.parametrize(
    ('sdc_method', 'expected'),
    [
        ('topup', ('topup', 'TOPUP')),
        ('drbuddi', ('fieldwarp', 'DRBUDDI')),
        # DRBUDDI refines what eddy corrected with TOPUP: the total needs both.
        ('topup+drbuddi', ('topup+drbuddi', 'TOPUP+DRBUDDI')),
    ],
)
def test_sdc_warp_source_separates_topup_drbuddi(tmp_path, monkeypatch, sdc_method, expected):
    """DRBUDDI's fieldwarp is the whole correction alone, only a residual after TOPUP."""
    from qsiprep.utils.sdc import sdc_warp_source

    monkeypatch.setattr(config.workflow, 'hmc_method', 'eddy')
    monkeypatch.setattr(config.workflow, 'sdc_method', sdc_method)
    unit, _ = _reverse_pe_unit(tmp_path)
    assert sdc_warp_source(unit, t2w_sdc=False) == expected


def test_trans_wf_builds_compose_sdc_warp_only_when_requested():
    """``sdc_warp_source`` gates the ComposeSDCWarp node and its wiring."""
    _cfg()
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    without = init_dwi_trans_wf(source_file='/data/sub-01_dwi.nii.gz', mem_gb=1)
    assert without.get_node('compose_sdc_warp') is None

    wf = init_dwi_trans_wf(
        source_file='/data/sub-01_dwi.nii.gz', mem_gb=1, sdc_warp_source='fieldwarp'
    )
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
        and any(dst == 'sdc_warps' for _, dst in d['connect'])
        for u, v, d in edges
    )
    assert any(
        u.name == 'compose_sdc_warp'
        and v.name == 'outputnode'
        and ('sdc_warp_to_template', 'sdc_warp_to_template') in d['connect']
        for u, v, d in edges
    )


def test_trans_wf_takes_volume_0_warp_from_a_single_path_or_a_list():
    """GRE's init_sdc_wf hands over one warp path; the others a per-volume list.

    Indexing a bare path would take its first character. The connection function
    is rebuilt from source here exactly as nipype does at run time.
    """
    _cfg()
    from nipype.utils.functions import create_function_from_source

    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(
        source_file='/data/sub-01_dwi.nii.gz', mem_gb=1, sdc_warp_source='fieldwarp'
    )
    (source,) = [
        src
        for u, v, d in wf._graph.edges(data=True)
        if u.name == 'inputnode' and v.name == 'compose_sdc_warp'
        for src, dst in d['connect']
        if dst == 'sdc_warps'
    ]
    port, func_source, _ = source
    assert port == 'fieldwarps'
    first_warp = create_function_from_source(func_source)
    assert first_warp('/work/vsm2dfm/fmap_antswarp.nii.gz') == (
        '/work/vsm2dfm/fmap_antswarp.nii.gz'
    )
    assert first_warp(['/work/finv.nii.gz', '/work/minv.nii.gz']) == '/work/finv.nii.gz'


def test_trans_wf_topup_builds_hz_to_warp_chain():
    """TOPUP has no standalone warp, so the field is turned into one first."""
    _cfg()
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        mem_gb=1,
        sdc_warp_source='topup',
        sdc_pe_dir='j',
        sdc_readout_time=0.05,
    )
    # Hz field -> ANTs warp -> the same conjugation node.
    hz_to_warp = wf.get_node('hz_to_warp')
    assert hz_to_warp is not None
    assert hz_to_warp.inputs.pe_dir == 'j'
    assert hz_to_warp.inputs.readout_time == 0.05
    edges = wf._graph.edges(data=True)
    assert any(
        u.name == 'inputnode'
        and v.name == 'hz_to_warp'
        and ('fieldmap_hz', 'in_file') in d['connect']
        for u, v, d in edges
    )
    assert any(
        u.name == 'hz_to_warp'
        and v.name == 'compose_sdc_warp'
        and ('out_file', 'sdc_warps') in d['connect']
        for u, v, d in edges
    )
    # TOPUP does not feed a standalone fieldwarp into the conjugation.
    assert not any(
        u.name == 'inputnode'
        and v.name == 'compose_sdc_warp'
        and any(dst == 'sdc_warps' for _, dst in d['connect'])
        for u, v, d in edges
    )
    assert wf.get_node('compose_sdc_refinement') is None


def test_trans_wf_topup_drbuddi_builds_total_and_refinement():
    """The total runs DRBUDDI's refinement, then TOPUP; the refinement is also kept."""
    _cfg()
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        mem_gb=1,
        sdc_warp_source='topup+drbuddi',
        sdc_pe_dir='j',
        sdc_readout_time=0.05,
    )
    edges = [(u.name, v.name, d['connect']) for u, v, d in wf._graph.edges(data=True)]

    def ports(src, dst):
        """(source port, destination port) pairs from ``src`` to ``dst``."""
        return {
            (s[0] if isinstance(s, tuple) else s, t)
            for u, v, connect in edges
            if (u, v) == (src, dst)
            for s, t in connect
        }

    # ComposeSDCWarp applies its warps to a point in list order: DRBUDDI's
    # refinement (in1) first, then the rebuilt TOPUP field (in2).
    assert ('fieldwarps', 'in1') in ports('inputnode', 'sdc_warp_chain')
    assert ('out_file', 'in2') in ports('hz_to_warp', 'sdc_warp_chain')
    assert ('out', 'sdc_warps') in ports('sdc_warp_chain', 'compose_sdc_warp')
    # The refinement is DRBUDDI's fieldwarp alone, on the same ACPC chain.
    assert ('fieldwarps', 'sdc_warps') in ports('inputnode', 'compose_sdc_refinement')
    assert not ports('hz_to_warp', 'compose_sdc_refinement')
    assert ('sdc_warp_transforms', 'to_template_transforms') in ports(
        'compose_transforms', 'compose_sdc_refinement'
    )
    assert ('sdc_warp_to_template', 'sdc_refinement_to_template') in ports(
        'compose_sdc_refinement', 'outputnode'
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
    # A map of the correction on the ACPC grid, not a from/to transform.
    assert ds.inputs.space == 'ACPC'
    assert ds.inputs.desc == 'sdc'
    assert ds.inputs.suffix == 'displacement'
    assert wf.get_node('ds_sdc_warp_t1_sidecar').inputs.meta == meta
    edges = [(u.name, v.name, d['connect']) for u, v, d in wf._graph.edges(data=True)]
    assert ('ds_sdc_warp_t1_sidecar', 'ds_sdc_warp_t1', [('meta', 'meta_dict')]) in edges
    assert any(
        (u, v) == ('inputnode', 'ds_sdc_warp_t1_sidecar')
        and ('sdc_transform_files', 'transform_files') in c
        for u, v, c in edges
    )
    # The TOPUP+DRBUDDI refinement has its own datasink, only when asked for.
    assert wf.get_node('ds_sdc_refinement_t1') is None

    refinement_meta = {'EstimationMethod': 'DRBUDDI', 'Description': 'the refinement'}
    wf = init_dwi_derivatives_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        sdc_warp_meta=meta,
        sdc_refinement_meta=refinement_meta,
    )
    ds = wf.get_node('ds_sdc_refinement_t1')
    assert ds.inputs.desc == 'sdcrefinement'
    assert ds.inputs.suffix == 'displacement'
    assert wf.get_node('ds_sdc_refinement_t1_sidecar').inputs.meta == refinement_meta
    assert wf.get_node('ds_sdc_warp_t1').inputs.desc == 'sdc'


@pytest.mark.parametrize(
    ('transform_files', 'expected'),
    [
        # One written transform (a distortion-group dwiref): a plain BIDS URI.
        (
            '/out/sub-01/dwi/sub-01_from-distortiongroup_to-ACPC_mode-image_desc-coreg_xfm.mat',
            'bids::sub-01/dwi/sub-01_from-distortiongroup_to-ACPC_mode-image_desc-coreg_xfm.mat',
        ),
        # A linear subject-level dwiref: both hops, in the order they apply, as
        # DerivativesDataSink and Merge may hand them over (possibly nested).
        (
            [
                [
                    '/out/sub-01/dwi/sub-01_from-distortiongroup_to-subject_mode-image_desc-coreg_xfm.mat'
                ],
                '/out/sub-01/dwi/sub-01_from-subject_to-ACPC_mode-image_desc-coreg_xfm.mat',
            ],
            [
                'bids::sub-01/dwi/sub-01_from-distortiongroup_to-subject_mode-image_desc-coreg_xfm.mat',
                'bids::sub-01/dwi/sub-01_from-subject_to-ACPC_mode-image_desc-coreg_xfm.mat',
            ],
        ),
    ],
)
def test_sdc_sidecar_names_the_transforms_into_acpc(transform_files, expected):
    """BEP014's TransformFile: the written transforms that carried the map into ACPC."""
    from qsiprep.utils.sdc import sdc_displacement_sidecar

    meta = {'EstimationMethod': 'DRBUDDI', 'Units': 'mm'}
    got = sdc_displacement_sidecar(meta, '/out', transform_files)
    assert got['TransformFile'] == expected
    assert got['Units'] == 'mm'
    assert 'TransformFile' not in meta  # the input metadata is not modified


def test_sdc_sidecar_leaves_out_an_unwritten_chain():
    """No transform files (a nonlinear subject dwiref): no half-chain TransformFile."""
    from qsiprep.utils.sdc import sdc_displacement_sidecar

    assert 'TransformFile' not in sdc_displacement_sidecar({'Units': 'mm'}, '/out')


def test_connect_sdc_transform_files_keeps_the_order_they_apply():
    """The helper feeds the finalize workflow every written hop, first hop first."""
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe

    from qsiprep.workflows.base import connect_sdc_transform_files

    workflow = pe.Workflow(name='subject_wf')
    first, second = (
        pe.Node(niu.IdentityInterface(fields=['out_file']), name=name)
        for name in ('ds_distortiongroup_to_subject', 'ds_subject_to_acpc')
    )
    finalize = pe.Workflow(name='dwi_finalize_wf')
    finalize.add_nodes([pe.Node(niu.IdentityInterface(['sdc_transform_files']), 'inputnode')])

    connect_sdc_transform_files(workflow, [first, second], finalize, 'sub_01_dwi')

    edges = {(u.name, v.name): d['connect'] for u, v, d in workflow._graph.edges(data=True)}
    chain = 'sdc_transform_files_sub_01_dwi'
    assert edges[('ds_distortiongroup_to_subject', chain)] == [('out_file', 'in1')]
    assert edges[('ds_subject_to_acpc', chain)] == [('out_file', 'in2')]
    assert edges[(chain, 'dwi_finalize_wf')] == [('out', 'inputnode.sdc_transform_files')]


@pytest.mark.parametrize('sdc_method', ['topup', 'topup+drbuddi'])
def test_finalize_writes_the_refinement_only_for_topup_drbuddi(tmp_path, monkeypatch, sdc_method):
    """TOPUP+DRBUDDI gets the total and the refinement, each with its own figure."""
    from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    for section, key, value in [
        (config.execution, 'output_dir', str(tmp_path / 'out')),
        (config.execution, 'sloppy', False),
        (config.workflow, 'hmc_method', 'eddy'),
        (config.workflow, 'sdc_method', sdc_method),
        (config.workflow, 'output_resolution', 1.2),
        (config.workflow, 'dwiref_definition', 'distortion-group'),
        (config.nipype, 'omp_nthreads', 1),
    ]:
        monkeypatch.setattr(section, key, value)
    unit, source = _reverse_pe_unit(tmp_path, dwi_writer=write_dwi_with_gradients)

    wf = init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=source,
        output_prefix='sub-01',
        do_biascorr=False,
        write_derivatives=True,
    )
    derivatives = wf.get_node('dwi_derivatives_wf')
    total = derivatives.get_node('ds_sdc_warp_t1_sidecar').inputs.meta
    refinement = derivatives.get_node('ds_sdc_refinement_t1_sidecar')
    assert total['VectorConvention'] == 'LPS'
    # The written DWI-to-ACPC transforms reach the maps' sidecars.
    assert any(
        (u.name, v.name) == ('inputnode', 'dwi_derivatives_wf')
        and ('sdc_transform_files', 'inputnode.sdc_transform_files') in d['connect']
        for u, v, d in wf._graph.edges(data=True)
    )

    if sdc_method == 'topup':
        assert total['EstimationMethod'] == 'TOPUP'
        assert total['Units'] == 'mm'
        assert refinement is None
        assert wf.get_node('sdcrefinement_plot') is None
        return

    assert total['EstimationMethod'] == 'TOPUP+DRBUDDI'
    assert total['Units'] == refinement.inputs.meta['Units'] == 'mm'
    assert refinement.inputs.meta['VectorConvention'] == 'LPS'
    assert 'desc-sdcrefinement' in total['Description']
    assert refinement.inputs.meta['EstimationMethod'] == 'DRBUDDI'
    assert wf.get_node('sdcwarp_plot').inputs.title == (
        'SDC displacement field, TOPUP+DRBUDDI (ACPC space)'
    )
    assert wf.get_node('sdcrefinement_plot').inputs.title == (
        'DRBUDDI refinement of the TOPUP field (ACPC space)'
    )
    assert wf.get_node('ds_report_sdcrefinement').inputs.desc == 'sdcrefinement'


_HZ, _TRT = 10.0, 0.05  # a uniform field: every voxel shifts _HZ * _TRT voxels
_LAS = np.diag([-2.0, 3.0, 4.0, 1.0])  # eddy's grid; voxel sizes i=2, j=3, k=4 mm


def _topup_warp_vector(tmp_path, affine, pe_dir):
    """The single LPS displacement ``_hz_to_warp`` writes for a uniform field."""
    from qsiprep.workflows.dwi.resampling import _hz_to_warp

    hz_path = str(tmp_path / 'hz.nii.gz')
    nb.Nifti1Image(np.full((6, 6, 6), _HZ, dtype='float32'), affine).to_filename(hz_path)
    warp = nb.load(_hz_to_warp(hz_path, _TRT, pe_dir, newpath=str(tmp_path)))
    assert warp.shape == (6, 6, 6, 1, 3)
    assert warp.header.get_intent()[0] == 'vector'
    np.testing.assert_allclose(warp.affine, affine)
    field = np.asarray(warp.dataobj).reshape(-1, 3)
    np.testing.assert_allclose(field, field[:1].repeat(len(field), 0))
    return field[0]


@pytest.mark.parametrize(
    ('pe_dir', 'expected_lps'),
    [
        # TOPUP shifts along the voxel axis its acqp row names, on the LAS+ grid
        # it ran on: +i is Left (LPS +x), +j is Anterior (LPS -y), +k is Superior.
        ('i', (_HZ * _TRT * 2.0, 0.0, 0.0)),
        ('i-', (-_HZ * _TRT * 2.0, 0.0, 0.0)),
        ('j', (0.0, -_HZ * _TRT * 3.0, 0.0)),
        ('j-', (0.0, _HZ * _TRT * 3.0, 0.0)),
        ('k', (0.0, 0.0, _HZ * _TRT * 4.0)),
    ],
)
def test_topup_hz_to_warp_follows_the_grid_axes(tmp_path, pe_dir, expected_lps):
    """field(Hz) * readout voxels along the PE voxel axis, in world mm via the affine."""
    np.testing.assert_allclose(_topup_warp_vector(tmp_path, _LAS, pe_dir), expected_lps, atol=1e-6)


def test_topup_hz_to_warp_matches_fugue_where_fugue_is_right(tmp_path):
    """On LAS+ the j axis is Anterior, as ``FUGUEvsm2ANTSwarp`` hard-codes.

    That j-axis result matched DRBUDDI's blip-up warp on real reverse-PE data
    (slope 1.02, r 0.965), so it anchors the sign. FUGUEvsm2ANTSwarp's i axis
    is hard-coded to Right and would disagree on this grid; ``_hz_to_warp`` does not.
    """
    from qsiprep.interfaces.niworkflows import FUGUEvsm2ANTSwarp

    vsm_path = str(tmp_path / 'vsm.nii.gz')
    nb.Nifti1Image(np.full((6, 6, 6), _HZ * _TRT, dtype='float32'), _LAS).to_filename(vsm_path)
    fugue = FUGUEvsm2ANTSwarp(in_file=vsm_path, pe_dir='j').run(cwd=str(tmp_path))
    fugue_vector = np.asarray(nb.load(fugue.outputs.out_file).dataobj).reshape(-1, 3)[0]

    (tmp_path / 'ours').mkdir()
    ours = _topup_warp_vector(tmp_path / 'ours', _LAS, 'j')
    np.testing.assert_allclose(ours, fugue_vector, atol=1e-6)


def test_topup_hz_to_warp_follows_an_oblique_grid(tmp_path):
    """An oblique grid carries the shift along its rotated PE column, not a world axis."""
    affine = np.eye(4)
    affine[:3, :3] = _rotation('z', 30) @ np.diag([2.0, 3.0, 4.0])
    step_ras = affine[:3, 1]  # one voxel along j, in RAS mm
    expected = _HZ * _TRT * step_ras * np.array([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(_topup_warp_vector(tmp_path, affine, 'j'), expected, atol=1e-6)


@pytest.mark.parametrize(
    ('node', 'desc'), [('ds_sdc_warp_t1', 'sdc'), ('ds_sdc_refinement_t1', 'sdcrefinement')]
)
def test_sdc_displacement_datasinks_write_space_acpc_maps(tmp_path, node, desc):
    """The workflow's own datasinks resolve to ``space-ACPC_desc-<desc>_displacement``."""
    _cfg()
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    src = tmp_path / 'sub-01_ses-1_acq-HBCD_run-01_dwi.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(src))
    field = tmp_path / 'sdc_warp_to_template.nii.gz'
    vec = nb.Nifti1Image(np.zeros((4, 4, 4, 1, 3), dtype='float32'), np.eye(4))
    vec.header.set_intent('vector')
    vec.to_filename(str(field))

    meta = {'EstimationMethod': 'DRBUDDI', 'Units': 'mm'}
    wf = init_dwi_derivatives_wf(
        source_file=str(src), sdc_warp_meta=meta, sdc_refinement_meta=meta
    )
    datasink = wf.get_node(node).interface
    datasink.inputs.base_directory = str(tmp_path / 'out')
    datasink.inputs.in_file = str(field)
    out = datasink.run().outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert out.endswith(
        f'/dwi/sub-01_ses-1_acq-HBCD_run-01_space-ACPC_desc-{desc}_displacement.nii.gz'
    )


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


def _write_linear_field(path, displacement_at, n=20, origin=(-9.5, -9.5, -9.5)):
    """A displacement field whose LPS vector at LPS point ``p`` is ``displacement_at(p)``.

    Linear functions are reproduced exactly by trilinear interpolation, so a
    point test can compare ANTs against SimpleITK without interpolation error.
    """
    grid = np.stack(np.meshgrid(*[np.arange(n) + o for o in origin], indexing='ij'), axis=-1)
    data = np.asarray(displacement_at(grid), dtype='float32')[:, :, :, np.newaxis, :]
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [-origin[0], -origin[1], origin[2]]
    img = nb.Nifti1Image(data, affine)
    img.header.set_intent('vector')
    img.to_filename(str(path))
    return str(path)


@requires_ants
def test_compose_sdc_warp_applies_several_warps_in_order(tmp_path):
    """Two warps map q -> A(W2(W1(A^-1(q)))), and that order is observable.

    For TOPUP+DRBUDDI, W1 is DRBUDDI's refinement and W2 TOPUP's field. Each
    displaces along one axis by an amount that depends on the other, so the two
    orders give different points.
    """
    import SimpleITK as sitk

    from qsiprep.interfaces.gradients import ComposeSDCWarp

    def refinement(p):
        return np.stack([0 * p[..., 0], 1.0 + 0.1 * p[..., 0], 0 * p[..., 0]], axis=-1)

    def topup(p):
        return np.stack([0.8 + 0.1 * p[..., 1], 0 * p[..., 1], 0 * p[..., 1]], axis=-1)

    w1 = _write_linear_field(tmp_path / 'refinement.nii.gz', refinement)
    w2 = _write_linear_field(tmp_path / 'topup.nii.gz', topup)
    coreg = _write_affine(tmp_path / 'coreg.mat', _OBLIQUE.T, [0.0, 0.0, 0.0])
    ref = _write_ref(tmp_path / 'ref.nii.gz', _OBLIQUE)
    (tmp_path / 'run').mkdir()
    out = (
        ComposeSDCWarp(sdc_warps=[w1, w2], to_template_transforms=[coreg], reference_image=ref)
        .run(cwd=str(tmp_path / 'run'))
        .outputs.sdc_warp_to_template
    )

    def field(path):
        return sitk.DisplacementFieldTransform(
            sitk.Cast(sitk.ReadImage(path), sitk.sitkVectorFloat64)
        )

    emitted, W1, W2 = field(out), field(w1), field(w2)
    A = sitk.AffineTransform(3)  # DWI -> ACPC
    A.SetMatrix(_OBLIQUE.ravel().tolist())
    A_inv = A.GetInverse()

    orders_differ = False
    for acpc_point in ([0.0, 0.0, 0.0], [2.0, -3.0, 1.0], [-4.0, 1.0, 2.0]):
        dwi_point = A_inv.TransformPoint(acpc_point)
        expected = np.array(A.TransformPoint(W2.TransformPoint(W1.TransformPoint(dwi_point))))
        reversed_order = np.array(
            A.TransformPoint(W1.TransformPoint(W2.TransformPoint(dwi_point)))
        )
        got = np.array(emitted.TransformPoint(acpc_point))
        np.testing.assert_allclose(got, expected, atol=1e-2)
        orders_differ |= not np.allclose(expected, reversed_order, atol=0.05)
    assert orders_differ


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


def test_sdc_warp_glyph_field_shows_the_inverse(tmp_path):
    """The glyph field is the INVERSE (point-transport) direction, not the raw field.

    Slicer displays where seed points travel, which follows the inverse of an
    image-resampling displacement field. For a uniform +d warp the inverse is -d,
    so the RAS glyph must be the opposite of the naive LPS->RAS of the raw field --
    this is exactly the sign that was wrong before.
    """
    from qsiprep.viz.utils import sdc_warp_glyph_field

    d_lps = np.array([0.0, 2.0, -1.0])  # uniform displacement, ITK-LPS mm
    warp = _write_sdc_field(tmp_path / 'w.nii.gz', d_lps, n=18, vary=0.0)

    disp_ras, mag, _ = sdc_warp_glyph_field(warp)
    interior = disp_ras[6:12, 6:12, 6:12].reshape(-1, 3).mean(0)

    raw_ras = d_lps * np.array([-1.0, -1.0, 1.0])  # naive (wrong) direction
    # inverse of a uniform +d is -d; in RAS that is -(raw_ras)
    np.testing.assert_allclose(interior, -raw_ras, atol=0.2)
    assert np.dot(interior, raw_ras) < 0  # opposite of the raw field
    np.testing.assert_allclose(mag[6:12, 6:12, 6:12].mean(), np.linalg.norm(d_lps), atol=0.2)


def test_sdc_warp_display_planes_contain_the_ped(tmp_path):
    """The two display planes contain the PE axis; the perpendicular one is skipped."""
    from qsiprep.viz.utils import sdc_warp_display_planes

    disp = np.zeros((10, 10, 10, 3))
    disp[..., 1] = 2.0  # RAS-y (A-P) dominant -> the PE axis
    disp[..., 2] = 0.4  # a little RAS-z (S-I) tilt
    affine = np.diag([-2.0, -2.0, 2.0, 1.0])  # voxel axis a -> RAS axis a
    ped_ras, slice_axes, _vox = sdc_warp_display_planes(disp, affine)
    assert ped_ras == 1  # A-P is the phase-encode axis
    # Slice normals are the two non-PE axes, so their planes contain the PE axis;
    # slicing along the PE axis itself (a perpendicular plane) is excluded.
    assert 1 not in slice_axes
    assert set(slice_axes) == {0, 2}
    assert slice_axes[0] == 0  # most-informative first: the least-displaced normal leads


def test_sdc_warp_plot_builds_a_valid_svg(tmp_path):
    """The reportlet interface renders an SVG from a warp + ACPC b=0 (no ANTs)."""
    from qsiprep.interfaces.reports import SDCWarpPlot

    warp = _write_sdc_field(tmp_path / 'w.nii.gz', [0.0, 2.0, -0.5], n=20, vary=0.03)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [9.5, 9.5, -9.5]  # match _write_sdc_field's grid
    b0 = tmp_path / 'b0.nii.gz'
    nb.Nifti1Image(
        np.random.default_rng(0).random((20, 20, 20)).astype('float32'), affine
    ).to_filename(str(b0))

    out = SDCWarpPlot(warp_file=warp, b0_ref=str(b0)).run(cwd=str(tmp_path)).outputs.out_file
    assert out.endswith('.svg')
    assert os.path.getsize(out) > 0
    assert '<svg' in open(out).read(4096)

    # A sub-millimetre field (a DRBUDDI refinement) renders too, with its own title.
    (tmp_path / 'small').mkdir()
    small = _write_sdc_field(tmp_path / 'small' / 'w.nii.gz', [0.0, 0.2, 0.0], n=20, vary=0.01)
    out = (
        SDCWarpPlot(warp_file=small, b0_ref=str(b0), title='A refinement')
        .run(cwd=str(tmp_path / 'small'))
        .outputs.out_file
    )
    assert '<svg' in open(out).read(4096)


@pytest.mark.parametrize(
    ('p99', 'expected'),
    [
        # Large enough to fill the arrow spacing: true length, the 0.5 mm floor.
        (6.0, (0.5, 6.0, 1.0)),
        # Sub-millimetre: a lower floor, its own color scale, lengthened arrows.
        (0.4, (0.04, 0.4, 16.0)),
    ],
)
def test_sdc_warp_glyph_scale_adapts_to_the_field(p99, expected):
    """Small fields get visible arrows; large ones stay at true length."""
    from qsiprep.viz.utils import sdc_warp_glyph_scale

    mag = np.full(1000, p99)  # every voxel at the 99th percentile
    affine = np.diag([2.0, 2.0, 2.0, 1.0])  # step 4 -> arrows 8 mm apart
    np.testing.assert_allclose(sdc_warp_glyph_scale(mag, affine, step=4), expected)


def test_sdc_warp_glyph_scale_handles_a_still_field():
    from qsiprep.viz.utils import sdc_warp_glyph_scale

    assert sdc_warp_glyph_scale(np.zeros(100), np.eye(4), step=4) == (0.5, 1.0, 1.0)


def test_invert_displacement_field_round_trips(tmp_path):
    """The helper produces a usable inverse of a displacement field."""
    import SimpleITK as sitk

    from qsiprep.utils.misc import invert_displacement_field

    warp = _write_sdc_field(tmp_path / 'W.nii.gz', [0.3, 2.0, -0.5], vary=0.02)
    inverse = invert_displacement_field(warp)

    forward = sitk.DisplacementFieldTransform(
        sitk.Cast(sitk.ReadImage(warp), sitk.sitkVectorFloat64)
    )
    backward = sitk.DisplacementFieldTransform(inverse)
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
        sdc_warps=warp, to_template_transforms=[coreg, template_warp], reference_image=ref
    )
    out = iface.run(cwd=run_cwd).outputs.sdc_warp_to_template
    img = nb.load(out)
    assert img.ndim == 5
    assert img.shape[4] == 3
    assert img.header.get_intent()[0] == 'vector'
    assert np.isfinite(np.asarray(img.dataobj)).all()
