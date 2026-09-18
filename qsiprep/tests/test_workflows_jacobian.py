"""Construction tests for Jacobian weighting inside the resampling workflow.

Asserts the graph, not the numbers: that the weighting node exists, that it is
fed from the right sources, and that ``--no-jacobian-weighting`` removes it.
The numeric correctness of the weights lives in
``test_interfaces_jacobian.py`` and ``test_jacobian_conservation.py``.
"""

import os

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
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
