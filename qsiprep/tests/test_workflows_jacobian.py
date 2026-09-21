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
    """``compose_jacobian`` feeds the dedup step, not the multiply step directly.

    Resampling was split out of what used to be a single ``scale_dwis`` node
    into a dedup step (``dedupe_jacobian_weights``), a parallel resampling
    ``MapNode`` (``resample_jacobian_weights``), a per-map floor ``MapNode``
    (``floor_jacobian_weights``), and a multiply-only ``scale_dwis``. Per-volume
    weight images now reach the dedup step first; see
    ``test_jacobian_weights_reach_scale_dwis_through_the_parallel_pipeline``
    for the rest of the chain.
    """
    edges = _edges(_trans_wf())
    assert any(
        src == 'compose_jacobian'
        and dst == 'dedupe_jacobian_weights'
        and ('jacobian_weight_images', 'jacobian_weight_images') in connect
        for src, dst, connect in edges
    )


def test_jacobian_weights_reach_scale_dwis_through_the_parallel_pipeline():
    """The full replacement chain for the old single-node ``scale_dwis``.

    Each edge here used to be internal Python state inside one interface
    (``ApplyJacobianWeights``); asserting them individually is what makes a
    dropped connection in the split visible at construction time instead of
    as a silently-absent or silently-unweighted derivative.
    """
    edges = _edges(_trans_wf())

    def _has(src, dst, pair):
        return any(s == src and d == dst and pair in connect for s, d, connect in edges)

    assert _has(
        'compose_jacobian',
        'dedupe_jacobian_weights',
        ('jacobian_weight_images', 'jacobian_weight_images'),
    )
    assert _has(
        'dedupe_jacobian_weights',
        'resample_jacobian_weights',
        ('unique_weight_images', 'input_image'),
    )
    assert _has(
        'dedupe_jacobian_weights',
        'resample_jacobian_weights',
        ('transforms', 'transforms'),
    )
    assert _has(
        'resample_jacobian_weights',
        'floor_jacobian_weights',
        ('output_image', 'weight_image'),
    )
    assert _has(
        'floor_jacobian_weights', 'scale_dwis', ('weight_image', 'resampled_weight_images')
    )
    assert _has('dedupe_jacobian_weights', 'scale_dwis', ('weight_index', 'weight_index'))
    assert _has('floor_jacobian_weights', 'outputnode', ('weight_image', 'jacobian_weights'))
    assert _has(
        'dedupe_jacobian_weights', 'outputnode', ('weight_index', 'jacobian_weight_index')
    )


def test_jacobian_weight_resampling_is_a_parallel_mapnode():
    """The whole point of the split: resampling must not still be a serial node.

    ``dwi_transform`` (this same workflow, constructed a few lines above the
    Jacobian block in ``resampling.py``) is the existing example of the
    pattern this follows -- a ``MapNode`` with an ``iterfield`` that Nipype's
    plugins (e.g. MultiProc) fan out across worker slots. If
    ``resample_jacobian_weights`` were a plain ``Node`` wrapping a Python
    for-loop, as ``scale_dwis`` used to be, this test would fail.
    """
    from nipype.pipeline.engine import MapNode

    wf = _trans_wf()
    resample_node = wf.get_node('resample_jacobian_weights')
    assert isinstance(resample_node, MapNode)
    assert resample_node.iterfield == ['input_image']

    floor_node = wf.get_node('floor_jacobian_weights')
    assert isinstance(floor_node, MapNode)
    assert floor_node.iterfield == ['weight_image']


def test_resample_jacobian_weights_mapnode_expands_per_unique_map(tmp_path):
    """MapNode fan-out, demonstrated directly: N inputs become N independent subnodes.

    This is the mechanism that makes the resampling actually run in parallel
    under a concurrent plugin (e.g. MultiProc), rather than merely looking
    parallel in the graph: each unique map becomes its own subnode that the
    plugin can schedule onto a separate worker slot. Exercised on a bare
    MapNode built the same way as ``resample_jacobian_weights`` rather than on
    the full workflow, since ``unique_weight_images`` is only populated at
    runtime by the upstream dedup node.
    """
    import nibabel as nb
    import numpy as np
    from nipype.interfaces import ants
    from nipype.pipeline import engine as pe

    node = pe.MapNode(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', dimension=3),
        name='resample_jacobian_weights',
        iterfield=['input_image'],
    )
    paths = []
    for letter in 'abcd':
        path = tmp_path / f'{letter}.nii.gz'
        nb.Nifti1Image(np.ones((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(path))
        paths.append(str(path))
    node.inputs.input_image = paths

    subnodes = list(node._make_nodes())

    assert len(subnodes) == 4
    assert {index for index, _ in subnodes} == {0, 1, 2, 3}


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
    wf = _trans_wf()
    assert wf.get_node('compose_jacobian') is None
    # The whole parallel resampling pipeline is gated the same way
    # ``compose_jacobian`` always was -- none of it should exist when the
    # feature is off, not just its entry point.
    assert wf.get_node('dedupe_jacobian_weights') is None
    assert wf.get_node('resample_jacobian_weights') is None
    assert wf.get_node('floor_jacobian_weights') is None


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


# --- known limitation: zero real weight maps at run time -------------------
#
# Flagged, not fixed -- see the mapnode-report for this task. Reported rather
# than improvised around, per this task's explicit instruction to stop when
# preserving the output contracts conflicts with the split.


def test_zero_unique_weights_is_a_defined_empty_list(tmp_path):
    """DeduplicateJacobianWeights' own contract: correct, on its own.

    This is not the bug -- an empty, *defined* list is exactly what
    ``resample_jacobian_weights``' MapNode needs to not choke on
    ``isdefined()``, and it is what
    ``test_deduplicate_jacobian_weights_passes_through_without_weights``
    (``test_interfaces_fmap.py``) already pins. The problem is one level
    further down: see
    ``test_a_defined_empty_iterfield_still_crashes_the_mapnode`` below.
    """
    import nibabel as nb
    import numpy as np

    from qsiprep.interfaces.fmap import DeduplicateJacobianWeights

    dwi = tmp_path / 'd0.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(dwi))

    result = DeduplicateJacobianWeights(dwi_files=[str(dwi)]).run()
    assert result.outputs.unique_weight_images == []


def test_a_defined_empty_iterfield_still_crashes_the_mapnode(tmp_path, monkeypatch):
    """The real conflict: nipype cannot run a MapNode over zero items.

    ``ComposeJacobianWeights`` leaves ``jacobian_weight_images`` Undefined
    when there is nothing to modulate (no gradwarp, no SDC, no eddy-current
    Jacobian) -- a real, tested, silent no-op
    (``test_compose_weights_with_no_fields_is_undefined`` in
    ``test_interfaces_jacobian.py``; ``derivatives.py``: "A unity map is
    never synthesized for that case"). ``DeduplicateJacobianWeights`` turns
    that into a defined ``[]`` (see the test above), which is the only way to
    avoid ``MapNode._check_iterfield`` treating an *Undefined* input as an
    error. But nipype's own ``MultiObject.validate`` collapses ANY defined
    empty list back to ``Undefined`` the moment it lands on a MapNode's own
    dynamically-created iterfield trait (this happens for every MapNode,
    regardless of the wrapped interface or the value's origin) -- so the
    crash still happens, just one hop later, inside
    ``resample_jacobian_weights`` itself, in a real run with no real
    distortion corrections.

    A placeholder map was considered to keep the MapNode non-empty and
    rejected: ``StackJacobianWeights`` gates solely on
    ``isdefined(weight_images)``, so a discarded-looking placeholder would
    still make it write a real jacobian derivative file for a run that
    applied no weighting at all -- exactly what "A unity map is never
    synthesized" forbids. This test pins the underlying nipype behaviour
    directly (not the full qsiprep workflow, which would need a real ANTs
    binary to execute) so the conflict stays verifiable rather than merely
    asserted in prose.
    """
    from nipype.interfaces import ants
    from nipype.pipeline import engine as pe

    # A crash writes a crashfile to the current directory by default;
    # keep it out of the repo.
    monkeypatch.chdir(tmp_path)

    node = pe.MapNode(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', dimension=3),
        name='resample_jacobian_weights',
        iterfield=['input_image'],
        base_dir=str(tmp_path),
    )
    node.inputs.input_image = []

    with pytest.raises(ValueError, match='was not set but it is listed in iterfields'):
        node.run()
