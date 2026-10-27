"""The ``hmcOptimization`` derivative under ``--distortion-group-merge``.

Regression: with ``--hmc-method shoreline`` (3dSHORE, more than one iteration)
the distortion-group merge workflow built a ``ds_optimization`` datasink in
its ``dwi_derivatives_wf`` but nothing fed it. The per-unit finalize
workflows return before their derivatives under a merge, and ``base.py`` never
forwarded the units' ``hmc_optimization_data`` into the merge, so a dataset
whose series merge into one output (AP+PA pairs, say) crashed at run time with
``DerivativesDataSink requires a value for input 'in_file'`` on
``*_final_merge_wf.dwi_derivatives_wf.ds_optimization``.

The merge workflow now stacks the per-unit iteration summaries
(:class:`~qsiprep.interfaces.shoreline.MergeIterationSummaries`) and sinks the
result; ``base.py`` feeds it from each unit's ``dwi_preproc_wf``.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from nipype.interfaces import utility as niu
from nipype.interfaces.base import isdefined

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.tests.test_workflows_native import _cfg, _StubLayout, _write_dwi


def undefined_mandatory_inputs(workflow, external='inputnode'):
    """List the ``(node, input)`` pairs nothing will ever set.

    A node input counts as fed when a connection reaches it from a real node,
    from the workflow's own ``external`` identity node (whose fields the caller
    is documented to connect), from an identity field holding a value, or from
    an identity field that is itself fed by one of those. A connection from an
    identity field that dead-ends -- the sub-workflow ``inputnode`` field
    nobody connected -- is what nipype drops when it expands the graph, leaving
    the mandatory input undefined at run time.
    """
    graph = workflow._create_flat_graph()
    external_name = f'{workflow.name}.{external}'

    def is_identity(node):
        return isinstance(node.interface, niu.IdentityInterface)

    def is_fed(node, field, seen):
        for src, _, data in graph.in_edges(node, data=True):
            for src_field, dst_field in data['connect']:
                if dst_field != field:
                    continue
                if isinstance(src_field, tuple):
                    src_field = src_field[0]
                if not is_identity(src) or src.fullname == external_name:
                    return True
                if isdefined(getattr(src.inputs, src_field)):
                    return True
                if (src.fullname, src_field) in seen:
                    continue
                seen.add((src.fullname, src_field))
                if is_fed(src, src_field, seen):
                    return True
        return False

    missing = []
    for node in graph.nodes():
        if is_identity(node):
            continue
        for name in node.inputs.traits(mandatory=True):
            if isdefined(getattr(node.inputs, name)):
                continue
            if not is_fed(node, name, set()):
                missing.append((node.fullname, name))
    return sorted(missing)


def _merge_wf(tmp_path, hmc_method, name='merge_wf'):
    from qsiplan.plan import OutputAssembly

    from qsiprep.workflows.dwi.distortion_group_merge import init_distortion_group_merge_wf

    cfg = _cfg(hmc_method=hmc_method, layout=_StubLayout())
    cfg.execution.output_dir = str(tmp_path / 'out')
    if hmc_method == 'shoreline':
        # The configuration that builds ds_optimization: see
        # qsiprep.workflows.dwi.derivatives.writes_hmc_optimization.
        cfg.workflow.shoreline_model = '3dshore'
        cfg.workflow.shoreline_iters = 2
    unit_a = make_preproc_unit([_write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')])
    unit_b = make_preproc_unit([_write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')])
    assembly = OutputAssembly(
        output_group='sub-01',
        input_runs=(unit_a.output_name, unit_b.output_name),
        strategy='concat',
        output_name='sub-01',
    )
    inputs_list = [unit_a.output_name, unit_b.output_name]
    return inputs_list, init_distortion_group_merge_wf(
        merging_strategy='concat',
        inputs_list=inputs_list,
        source_file='sub-01_dwi.nii.gz',
        output_prefix='sub-01',
        name=name,
        assembly=assembly,
        units=[unit_a, unit_b],
    )


def test_merge_wf_feeds_every_datasink_under_shoreline(tmp_path):
    """Test that the merged-group derivatives sink the stacked iteration summaries.

    Under the SHORELine configuration that writes ``hmcOptimization`` the merge
    workflow builds ``ds_optimization`` and feeds it. No datasink -- nothing in
    the graph -- is left with a mandatory input the caller cannot satisfy
    through ``inputnode``.
    """
    inputs_list, wf = _merge_wf(tmp_path, 'shoreline')

    assert undefined_mandatory_inputs(wf) == []

    sink = wf.get_node('dwi_derivatives_wf').get_node('ds_optimization')
    assert sink is not None
    assert sink.inputs.suffix == 'hmcOptimization'
    merger = wf.get_node('merge_iteration_summaries')
    assert merger is not None
    # One row-block per unit, named after the unit so rows stay attributable.
    assert merger.inputs.input_names == inputs_list
    # The merge workflow exposes one hmc_optimization_data input per unit, like
    # its other per-unit inputs, and routes them through the stacker.
    per_unit = sorted(
        field
        for field in wf.inputs.inputnode.copyable_trait_names()
        if field.endswith('_hmc_optimization_data')
    )
    assert len(per_unit) == 2
    graph = wf._create_flat_graph()
    stacker = next(n for n in graph.nodes() if n.name == 'merge_iteration_summaries')
    sources = {
        src.name
        for src, _, data in graph.in_edges(stacker, data=True)
        for _, dst in data['connect']
        if dst == 'iteration_summary_files'
    }
    assert sources == {'merge_hmc_optimization'}
    sink_flat = next(n for n in graph.nodes() if n.name == 'ds_optimization')
    fed_from = {
        (src.name, src_field)
        for src, _, data in graph.in_edges(sink_flat, data=True)
        for src_field, dst in data['connect']
        if dst == 'in_file'
    }
    assert fed_from == {('inputnode', 'hmc_optimization_data')}


def test_merge_wf_builds_no_optimization_sink_under_eddy(tmp_path):
    """Test that eddy runs neither build nor try to feed ``ds_optimization``.

    eddy's ``hmc_optimization_data`` is its outlier map, not an iteration
    table, so neither the sink nor the stacker exists and the graph is still
    fully fed.
    """
    _, wf = _merge_wf(tmp_path, 'eddy', name='eddy_merge_wf')

    assert undefined_mandatory_inputs(wf) == []
    assert wf.get_node('dwi_derivatives_wf').get_node('ds_optimization') is None
    assert wf.get_node('merge_iteration_summaries') is None
    assert wf.get_node('merge_hmc_optimization') is None


def test_undefined_mandatory_inputs_sees_through_dead_identity_fields():
    """Test the checker on the exact shape of the original bug.

    A sink fed from a sub-workflow ``inputnode`` field that nobody connects is
    reported; the same sink fed from the top-level ``inputnode`` is not.
    """
    import nipype.pipeline.engine as pe

    from qsiprep.interfaces import DerivativesDataSink

    def build(connect_outer):
        inner = pe.Workflow(name='inner')
        inner_in = pe.Node(niu.IdentityInterface(fields=['data']), name='inputnode')
        sink = pe.Node(
            DerivativesDataSink(source_file='sub-01_dwi.nii.gz', suffix='x'), name='ds_x'
        )
        inner.connect([(inner_in, sink, [('data', 'in_file')])])
        outer = pe.Workflow(name='outer')
        outer_in = pe.Node(niu.IdentityInterface(fields=['data']), name='inputnode')
        outer.add_nodes([outer_in, inner])
        if connect_outer:
            outer.connect([(outer_in, inner, [('data', 'inputnode.data')])])
        return outer

    assert undefined_mandatory_inputs(build(connect_outer=False)) == [
        ('outer.inner.ds_x', 'in_file')
    ]
    assert undefined_mandatory_inputs(build(connect_outer=True)) == []


def test_subject_workflow_forwards_hmc_optimization_into_the_merge():
    """Test that ``base.py`` feeds the merge's per-unit optimization inputs.

    Source check: the subject-level workflow needs a BIDS layout to build. The
    data must come from ``dwi_preproc_wf``: ``init_dwi_finalize_wf`` only
    forwards ``hmc_optimization_data`` to its outputnode when it writes
    derivatives, which under a merge it does not.
    """
    from qsiprep.workflows import base

    src = inspect.getsource(base)
    assert "hmc_optimization_name = f'inputnode.{output_wfname}_hmc_optimization_data'" in src
    assert "('outputnode.hmc_optimization_data', hmc_optimization_name)" in src
    preproc_block = src[src.index('(dwi_preproc_wf, final_merge_wf, [') :]
    preproc_block = preproc_block[: preproc_block.index('])')]
    assert 'hmc_optimization_name' in preproc_block


def test_merge_iteration_summaries_stacks_per_unit_tables(tmp_path, monkeypatch):
    """Test that the stacker keeps every row, labelled by its unit, in input order."""
    from qsiprep.interfaces.shoreline import MergeIterationSummaries

    monkeypatch.chdir(tmp_path)

    def write_summary(path, n_rows, offset):
        df = pd.DataFrame(
            {
                'trans_x': np.arange(n_rows) + offset,
                'iter_num': [0] * (n_rows // 2) + [1] * (n_rows - n_rows // 2),
                'iter_name': [''] * n_rows,
            }
        )
        df.to_csv(path, index=False)
        return str(path)

    ap = write_summary(tmp_path / 'ap.csv', 4, 0)
    pa = write_summary(tmp_path / 'pa.csv', 2, 100)

    result = MergeIterationSummaries(
        iteration_summary_files=[ap, pa],
        input_names=['sub-01-dir-AP', 'sub-01-dir-PA'],
    ).run()
    merged = pd.read_csv(result.outputs.iteration_summary_file)
    assert list(merged.columns) == ['input_name', 'trans_x', 'iter_num', 'iter_name']
    assert merged['input_name'].tolist() == ['sub-01-dir-AP'] * 4 + ['sub-01-dir-PA'] * 2
    assert merged['trans_x'].tolist() == [0, 1, 2, 3, 100, 101]
    assert merged['iter_num'].tolist() == [0, 0, 1, 1, 0, 1]

    # Without names the units are numbered in input order.
    (tmp_path / 'unnamed').mkdir()
    monkeypatch.chdir(tmp_path / 'unnamed')
    unnamed = MergeIterationSummaries(iteration_summary_files=[ap, pa]).run()
    assert pd.read_csv(unnamed.outputs.iteration_summary_file)['input_name'].tolist() == (
        [0] * 4 + [1] * 2
    )

    with pytest.raises(ValueError, match='input_names'):
        MergeIterationSummaries(iteration_summary_files=[ap, pa], input_names=['only-one']).run()


def test_writes_hmc_optimization_is_the_shoreline_gate():
    """Test the shared gate both ``ds_optimization`` builders consult."""
    from qsiprep.workflows.dwi.derivatives import writes_hmc_optimization

    config.workflow.hmc_method = 'shoreline'
    config.workflow.shoreline_model = '3dshore'
    config.workflow.shoreline_iters = 2
    assert writes_hmc_optimization()
    config.workflow.shoreline_iters = 1
    assert not writes_hmc_optimization()
    config.workflow.shoreline_iters = 2
    config.workflow.shoreline_model = 'none'
    assert not writes_hmc_optimization()
    config.workflow.shoreline_model = '3dshore'
    config.workflow.hmc_method = 'eddy'
    assert not writes_hmc_optimization()
