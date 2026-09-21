"""Construction tests for Jacobian weighting inside the resampling workflow.

Asserts the graph, not the numbers: that the weighting node exists, that it is
fed from the right sources, and that ``--no-jacobian-weighting`` removes it.
The numeric correctness of the weights lives in
``test_interfaces_jacobian.py`` and ``test_jacobian_conservation.py``.
"""

import json
import os

import pytest
from bids.layout.writing import build_path
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.data import load as load_data
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.fixture(autouse=True)
def _reset_config():
    saved = (config.workflow.jacobian_weighting, config.workflow.output_resolution)
    config.workflow.jacobian_weighting = True
    config.workflow.output_resolution = 2.0
    # config.nipype.init() normally resolves this; it is not run in these bare
    # construction tests, so init_modelfree_qc_wf's DSIStudioGQIReconstruction
    # node (which requires an int thread_count) fails to build without it.
    config.nipype.omp_nthreads = 1
    yield
    config.workflow.jacobian_weighting, config.workflow.output_resolution = saved


def _trans_wf():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    return init_dwi_trans_wf(source_file='/data/sub-1_dwi.nii.gz', mem_gb=1)


def _edges(workflow):
    # key=repr: some ``connect`` entries pair a plain field name with a
    # function-wrapped one (e.g. gradwarp_field's ``_listify``), whose first
    # element is itself a tuple. Sorting those against plain strings raises
    # ``TypeError: '<' not supported between instances of 'tuple' and 'str'``;
    # repr() gives every entry a consistent, comparable sort key.
    return {
        (src.name, dst.name, tuple(sorted(data['connect'], key=repr)))
        for src, dst, data in workflow._graph.edges(data=True)
    }


def _source_name(source):
    """The forwarded field's own name, whether plain or function-wrapped.

    A functor connection (e.g. ``(('gradwarp_field', _listify), 'gradwarp_field')``)
    stores the upstream field name as the first element of a tuple rather than
    as a bare string.
    """
    return source[0] if isinstance(source, tuple) else source


def test_compose_jacobian_node_exists():
    assert _trans_wf().get_node('compose_jacobian') is not None


def test_compose_jacobian_feeds_the_weighting_node():
    edges = _edges(_trans_wf())
    assert any(
        src == 'compose_jacobian'
        and dst == 'scale_dwis'
        and ('jacobian_weight_images', 'jacobian_weight_images') in connect
        for src, dst, connect in edges
    )


def test_scale_dwis_outputs_reach_outputnode():
    """The single-node replacement for the old dedup/resample/floor pipeline.

    ``scale_dwis`` (``ApplyJacobianWeights``) transports, floors and multiplies
    the weight maps itself, parallelizing the per-unique-map ``ants
    ApplyTransforms`` calls internally with a ``ThreadPoolExecutor`` (see the
    interface's own docstring) rather than fanning out across separate nipype
    nodes -- so its own outputs feed the workflow outputnode directly.
    """
    edges = _edges(_trans_wf())

    def _has(src, dst, pair):
        return any(s == src and d == dst and pair in connect for s, d, connect in edges)

    assert _has(
        'compose_jacobian', 'scale_dwis', ('jacobian_weight_images', 'jacobian_weight_images')
    )
    assert _has('scale_dwis', 'outputnode', ('resampled_weight_images', 'jacobian_weights'))
    assert _has('scale_dwis', 'outputnode', ('weight_index', 'jacobian_weight_index'))


def test_scale_dwis_num_threads_is_set_from_omp_nthreads():
    """The threading knob is wired the same way as its neighbouring nodes.

    ``ApplyJacobianWeights`` parallelizes its own per-unique-map ``ants
    ApplyTransforms`` calls with a ``num_threads`` trait, mirroring
    ``ComposeTransforms`` (``gradients.py``); the workflow must actually set it
    from ``config.nipype.omp_nthreads`` rather than leaving it at the
    interface's serial default.
    """
    config.nipype.omp_nthreads = 4
    wf = _trans_wf()
    scale_dwis = wf.get_node('scale_dwis')
    assert scale_dwis.inputs.num_threads == 4


def test_compose_jacobian_consumes_gradwarp_and_fieldwarps():
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if src == 'inputnode' and dst == 'compose_jacobian'
        for pair in connect
    }
    sources = {_source_name(source) for source, _ in forwarded}
    assert 'fieldwarps' in sources
    assert 'gradwarp_field' in sources
    assert 'dwi_files' in sources
    assert 'b0_ref_image' in sources
    assert 'dwi_mask' in sources
    assert 'ec_jacobian_images' in sources


def test_no_jacobian_weighting_removes_the_node():
    config.workflow.jacobian_weighting = False
    assert _trans_wf().get_node('compose_jacobian') is None


def test_no_jacobian_weighting_leaves_weights_unconnected():
    """With weighting off, scale_dwis must get nothing and pass data through."""
    config.workflow.jacobian_weighting = False
    edges = _edges(_trans_wf())
    assert not any(
        ('jacobian_weight_images', 'jacobian_weight_images') in connect
        for _, _, connect in edges
    )


def test_hmc_xforms_never_reach_the_jacobian_node():
    """HMC is excluded by policy; a wire here would silently modulate by it."""
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if dst == 'compose_jacobian'
        for pair in connect
    }
    assert not any('hmc' in _source_name(source) for source, _ in forwarded)


def test_coreg_and_template_transforms_never_reach_the_jacobian_node():
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if dst == 'compose_jacobian'
        for pair in connect
    }
    excluded = ('itk_b0_to_t1', 'intramodal', 't1_2_mni')
    for source, _ in forwarded:
        assert not any(name in _source_name(source) for name in excluded), source


def test_source_name_detects_a_functor_wrapped_forbidden_field():
    """Pin the normalization itself, not just today's wiring.

    ``gradwarp_field`` is already threaded through a functor wrapper
    (``(('gradwarp_field', _listify), 'gradwarp_field')`` in
    ``resampling.py``), so wrapping a field to adapt its shape is an
    established pattern here -- exactly how a forbidden field (e.g.
    ``hmc_xforms``) could plausibly be wired in without tripping a check that
    only looks at the raw, unnormalized ``source`` half of the pair. This
    test fails if ``_source_name`` (or the exclusion checks above) stop
    unwrapping functor connections before the membership test.
    """
    forwarded = {(('hmc_xforms', lambda x: x), 'hmc_affines')}
    assert any('hmc' in _source_name(source) for source, _ in forwarded)


@pytest.mark.parametrize(
    ('method', 'sources'),
    [
        pytest.param(
            CorrectionMethod.PEPOLAR,
            ['/data/sub-1/dwi/sub-1_dwi.nii.gz', '/data/sub-1/fmap/sub-1_epi.nii.gz'],
            marks=pytest.mark.xfail(
                reason=(
                    "pre-existing bug, unrelated to jacobian weighting: "
                    "init_sdc_wf's PEPOLAR branch calls "
                    'init_pepolar_unwarp_wf(omp_nthreads=...) '
                    '(qsiprep/workflows/fieldmap/base.py:166), but '
                    'init_pepolar_unwarp_wf takes no such parameter '
                    '(qsiprep/workflows/fieldmap/pepolar.py:27)'
                ),
                strict=True,
            ),
        ),
        (
            CorrectionMethod.PHASEDIFF,
            [
                '/data/sub-1/fmap/sub-1_phasediff.nii.gz',
                '/data/sub-1/fmap/sub-1_magnitude1.nii.gz',
            ],
        ),
        (CorrectionMethod.NIPREPS_SYN, ['/data/sub-1/dwi/sub-1_dwi.nii.gz']),
    ],
)
def test_each_sdc_branch_emits_a_warp_for_weighting(method, sources, monkeypatch):
    """Every SDC branch must reach ``to_dwi_ref_warps``.

    ``to_dwi_ref_warps`` becomes ``fieldwarps``, which is the single input
    ComposeJacobianWeights derives the SDC determinant from. A branch that
    stops populating it loses weighting silently rather than loudly.
    """
    from qsiprep.workflows.fieldmap import init_sdc_wf

    # init_phdiff_wf's N4BiasFieldCorrection needs an integer thread count;
    # config.nipype.init() normally resolves this, but it is not run here.
    config.nipype.omp_nthreads = 1
    # PEPOLAR/phasediff both check only that FSLDIR is set, not its contents.
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')
    unit = make_preproc_unit(
        ['/data/sub-1/dwi/sub-1_dwi.nii.gz'],
        method=method,
        pe_dir='j',
        estimation_sources=sources,
    )
    workflow = init_sdc_wf(unit)
    outputnode = workflow.get_node('outputnode')
    assert 'out_warp' in outputnode.outputs.copyable_trait_names()

    connected = {
        pair
        for _, dst, data in workflow._graph.edges(data=True)
        if dst.name == 'outputnode'
        for pair in data['connect']
    }
    assert any(target == 'out_warp' for _, target in connected), (
        f'{method} produced no out_warp, so nothing would reach fieldwarps '
        'and this backend would be silently unweighted.'
    )


def test_sdc_unwarp_wf_has_no_dead_jacobian_node():
    """The SDC-warp-only Jacobian is not what the weighting needs.

    It was computed and discarded for years. ComposeJacobianWeights derives the
    determinant of the *composed* gradwarp-and-SDC warp instead, so leaving
    this node in place would be a second, subtly-wrong source of truth.

    ``init_sdc_unwarp_wf`` hard-requires FSL (it raises if ``FSLDIR`` is
    unset), which is present in the qsiprep container but not in every dev
    environment, hence the skip guard below.
    """
    if not os.environ.get('FSLDIR'):
        pytest.skip('FSLDIR is not set; init_sdc_unwarp_wf requires FSL')

    from qsiprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf

    workflow = init_sdc_unwarp_wf()
    assert workflow.get_node('jac_dfm') is None
    assert 'out_jacobian' not in workflow.get_node('outputnode').outputs.copyable_trait_names()


def test_sdc_scaling_images_channel_is_gone():
    """DRBUDDI's ratio images are replaced by the analytic determinant.

    The channel existed only for DRBUDDI, whose warps already arrive as
    ``fieldwarps`` -- so ComposeJacobianWeights derives its determinant the
    same way as every other backend's, and the bespoke plumbing is dead.
    """
    import subprocess

    # Excludes this file itself: its grep invocation and assertion message
    # necessarily contain the literal string being searched for, which would
    # otherwise make this test self-matching and permanently red.
    hits = subprocess.run(
        [
            'grep',
            '-rn',
            '--include=*.py',
            '--exclude=test_workflows_jacobian.py',
            'sdc_scaling_images',
            'qsiprep/',
        ],
        capture_output=True,
        text=True,
    ).stdout
    assert hits == '', f'sdc_scaling_images still referenced:\n{hits}'


def test_drbuddi_aggregate_has_no_scaling_output():
    from qsiprep.interfaces.tortoise import DRBUDDIAggregateOutputs

    outputs = DRBUDDIAggregateOutputs().output_spec().copyable_trait_names()
    assert 'sdc_scaling_images' not in outputs


def _patterns():
    return json.loads(load_data('io_spec.json').read_text())['default_path_patterns']


def test_jacobian_derivative_path_renders():
    """Assert the rendered path, not the datasink inputs.

    Entity-level checks are blind to a pattern that silently drops an entity or
    collides with another derivative's name.
    """
    # strict=True: with strict=False an entity the pattern does not support is
    # silently dropped, so the test would pass while the real filename lost it.
    out = build_path(
        {
            'subject': '01',
            'datatype': 'dwi',
            'space': 'ACPC',
            'desc': 'jacobian',
            'suffix': 'dwimap',
            'extension': '.nii.gz',
        },
        _patterns(),
        strict=True,
    )
    assert out == 'sub-01/dwi/sub-01_space-ACPC_desc-jacobian_dwimap.nii.gz'


def test_jacobian_derivative_does_not_collide_with_preproc_dwi():
    preproc = build_path(
        {
            'subject': '01', 'datatype': 'dwi', 'space': 'ACPC', 'desc': 'preproc',
            'suffix': 'dwi', 'extension': '.nii.gz',
        },
        _patterns(),
        strict=False,
    )
    jacobian = build_path(
        {
            'subject': '01', 'datatype': 'dwi', 'space': 'ACPC', 'desc': 'jacobian',
            'suffix': 'dwimap', 'extension': '.nii.gz',
        },
        _patterns(),
        strict=False,
    )
    assert preproc != jacobian
    assert not jacobian.endswith('_dwi.nii.gz')


def test_jacobian_sidecar_index_is_zero_based_and_full_length():
    from qsiprep.interfaces.jacobian import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0, 0, 1, 0], applied=['gradwarp', 'sdc'],
        unmodulated=[], reason=None,
    )
    assert sidecar['JacobianWeightIndex'] == [0, 0, 1, 0]
    assert sidecar['AppliedCorrections'] == ['gradwarp', 'sdc']
    assert sidecar['UnmodulatedCorrections'] == []
    assert 'UnmodulatedReason' not in sidecar


def test_jacobian_sidecar_records_a_gap():
    from qsiprep.interfaces.jacobian import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0, 0], applied=['gradwarp', 'sdc'],
        unmodulated=['eddy-current'],
        reason='TORTOISE correction_mode=cubic is not supported',
    )
    assert sidecar['UnmodulatedCorrections'] == ['eddy-current']
    assert 'cubic' in sidecar['UnmodulatedReason']


def test_jacobian_sidecar_collapsed_case_is_written_in_full():
    """All-zeros rather than omitted, so consumers need no special case."""
    from qsiprep.interfaces.jacobian import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0] * 5, applied=['sdc'], unmodulated=[], reason=None
    )
    assert sidecar['JacobianWeightIndex'] == [0, 0, 0, 0, 0]


# --- stack_jacobian / ds_jacobian wiring ------------------------------------
#
# A dropped connection here produces no error at construction or run time --
# just a silently absent derivative (see the task-12 report). These tests
# therefore check edges, not just node presence, mirroring
# ``test_tsnr_is_wired_into_derivatives`` (qsiprep/tests/test_tsnr.py) and the
# ``_finalize_wf``/``_finalize_cfg`` setup in test_workflows_gradwarp.py.


def _derivatives_wf(tmp_path):
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    config.execution.output_dir = str(tmp_path)
    config.workflow.hmc_method = 'tortoise'
    config.workflow.write_local_bvecs = False
    return init_dwi_derivatives_wf('/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz')


def _derivatives_edges(workflow):
    return {(u.name, v.name): d['connect'] for u, v, d in workflow._graph.edges(data=True)}


def test_stack_jacobian_and_sink_exist_when_weighting_on(tmp_path):
    wf = _derivatives_wf(tmp_path)
    assert wf.get_node('stack_jacobian') is not None
    assert wf.get_node('ds_jacobian') is not None


def test_stack_jacobian_and_sink_absent_when_weighting_off(tmp_path):
    config.workflow.jacobian_weighting = False
    wf = _derivatives_wf(tmp_path)
    assert wf.get_node('stack_jacobian') is None
    assert wf.get_node('ds_jacobian') is None


def test_stack_jacobian_connects_to_the_sink(tmp_path):
    """A missing edge here is the silent-absence failure mode; a missing node
    is not the only way to lose the file."""
    wf = _derivatives_wf(tmp_path)
    edges = _derivatives_edges(wf)
    assert set(edges[('stack_jacobian', 'ds_jacobian')]) == {
        ('out_file', 'in_file'),
        ('meta_dict', 'meta_dict'),
    }


def test_stack_jacobian_receives_weights_from_inputnode(tmp_path):
    wf = _derivatives_wf(tmp_path)
    edges = _derivatives_edges(wf)
    assert set(edges[('inputnode', 'stack_jacobian')]) == {
        ('jacobian_weights', 'weight_images'),
        ('jacobian_weight_index', 'weight_index'),
    }


def _finalize_wf(tmp_path, write_derivatives=True):
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.execution.output_dir = str(tmp_path)
    config.execution.sloppy = False
    config.workflow.sdc_method = 'topup'
    config.workflow.intramodal_template_iters = 0
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi])
    return init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=dwi,
        output_prefix='sub-01',
        write_derivatives=write_derivatives,
    )


def test_jacobian_weights_reach_the_derivatives_workflow_through_finalize(tmp_path):
    """The full inter-workflow chain: trans_wf -> finalize outputnode -> derivatives_wf.

    This is the shape of the boundary Task 10 and Task 11 each missed once in
    ``workflows/base.py`` (a dropped edge between two outputnodes, no error,
    just an absent derivative). ``init_dwi_trans_wf``/``init_dwi_derivatives_wf``
    are only ever called from inside ``init_dwi_finalize_wf`` for this
    channel, so this is the boundary that matters for it, not ``base.py``.
    """
    wf = _finalize_wf(tmp_path)
    trans_wf = wf.get_node('transform_dwis_t1')
    outputnode = wf.get_node('outputnode')
    deriv_wf = wf.get_node('dwi_derivatives_wf')
    assert deriv_wf.get_node('stack_jacobian') is not None

    trans_to_out = wf._graph.get_edge_data(trans_wf, outputnode)
    assert trans_to_out is not None
    assert ('outputnode.jacobian_weights', 'jacobian_weights') in trans_to_out['connect']
    assert (
        'outputnode.jacobian_weight_index',
        'jacobian_weight_index',
    ) in trans_to_out['connect']

    out_to_deriv = wf._graph.get_edge_data(outputnode, deriv_wf)
    assert out_to_deriv is not None
    assert ('jacobian_weights', 'inputnode.jacobian_weights') in out_to_deriv['connect']
    assert (
        'jacobian_weight_index',
        'inputnode.jacobian_weight_index',
    ) in out_to_deriv['connect']


def test_jacobian_weights_do_not_reach_finalize_when_weighting_off(tmp_path):
    config.workflow.jacobian_weighting = False
    wf = _finalize_wf(tmp_path)
    trans_wf = wf.get_node('transform_dwis_t1')
    outputnode = wf.get_node('outputnode')
    deriv_wf = wf.get_node('dwi_derivatives_wf')
    assert deriv_wf.get_node('stack_jacobian') is None

    edge = wf._graph.get_edge_data(trans_wf, outputnode)
    connect = edge['connect'] if edge is not None else []
    assert not any('jacobian_weights' in str(pair) for pair in connect)



# --- regression: zero real weight maps at run time --------------------------
#
# 679be72 split ApplyJacobianWeights into a dedup Node feeding a
# resample_jacobian_weights MapNode: when a run applies no distortion
# correction at all, ComposeJacobianWeights leaves jacobian_weight_images
# Undefined by design (a unity map must never be synthesized -- see
# jacobian.py's ComposeJacobianWeights docstring), and nipype collapses a
# defined empty list back to Undefined on a MapNode's own iterfield, so
# MapNode._check_iterfield raised at run time. That is not an edge case:
# --hmc-method eddy with TOPUP and no gradwarp is exactly this configuration.
# Reverting to a single ApplyJacobianWeights Node (this task) restores the
# early-return no-op for free, because a plain Node has no iterfield to
# collapse. These tests pin that directly.


def test_scale_dwis_is_a_plain_node_not_a_mapnode():
    """No iterfield exists to collapse an empty/Undefined weight list into.

    This is the structural reason the zero-corrections crash from 679be72
    cannot recur: a MapNode's iterfield input collapses a defined empty list
    back to Undefined and then raises; a plain Node has no such input.
    """
    from nipype.pipeline.engine import MapNode, Node

    scale_dwis = _trans_wf().get_node('scale_dwis')
    assert isinstance(scale_dwis, Node)
    assert not isinstance(scale_dwis, MapNode)


def test_zero_distortion_corrections_produces_no_weights_and_does_not_raise(tmp_path):
    """The exact regression 679be72 introduced, exercised end to end.

    A run with no gradwarp, no SDC warp and no EC Jacobian leaves
    ``ComposeJacobianWeights.jacobian_weight_images`` Undefined
    (``test_interfaces_jacobian.py``'s
    ``test_compose_weights_with_no_fields_is_undefined`` pins that half).
    Feeding that Undefined output straight into ``ApplyJacobianWeights`` --
    exactly how ``compose_jacobian -> scale_dwis`` are wired in
    ``resampling.py`` -- must pass the DWIs through unmodified rather than
    raising, with no ANTs binary required since nothing is ever resampled.
    """
    import nibabel as nb
    import numpy as np
    from nipype.interfaces.base import isdefined

    from qsiprep.interfaces.fmap import ApplyJacobianWeights
    from qsiprep.interfaces.jacobian import ComposeJacobianWeights

    dwi_files = []
    for i in range(3):
        path = tmp_path / f'd{i}.nii.gz'
        nb.Nifti1Image(np.ones((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(path))
        dwi_files.append(str(path))
    mask = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), dtype='int16'), np.eye(4)).to_filename(str(mask))

    compose_result = ComposeJacobianWeights(
        dwi_files=dwi_files,
        b0_ref_image=dwi_files[0],
        mask=str(mask),
    ).run()
    assert not isdefined(compose_result.outputs.jacobian_weight_images)

    apply_result = ApplyJacobianWeights(
        dwi_files=dwi_files,
        jacobian_weight_images=compose_result.outputs.jacobian_weight_images,
        reference_image=dwi_files[0],
    ).run()

    assert apply_result.outputs.scaled_images == dwi_files
