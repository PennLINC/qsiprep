"""Wiring for DSI Studio QC warnings: from DSIStudioQC to the HTML report.

DSI Studio can fail without saying so. QC must not fail a run, so a failed
QC stage becomes n/a in image_qc.tsv, and its reason travels
DSIStudioQC.warning -> DSIStudioMergeQC -> merged_qc.csv -> SeriesQC, which
writes a reportlet that ds_report_qc_warnings sinks as desc-qcwarnings. A
dropped edge anywhere in that chain produces no error, only a missing warning,
so each hop is checked here.
"""

import inspect

import pytest

from qsiprep import config
from qsiprep.data import load as load_data
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.fixture(autouse=True)
def _reset_config():
    saved = (
        config.workflow.jacobian_weighting,
        config.workflow.output_resolution,
        config.workflow.sdc_method,
        config.workflow.dwiref_construction_iters,
        config.execution.output_dir,
        config.execution.sloppy,
        config.nipype.omp_nthreads,
    )
    config.workflow.jacobian_weighting = True
    config.workflow.output_resolution = 2.0
    # config.nipype.init() is not run in construction tests, and
    # DSIStudioGQIReconstruction needs an int thread_count to build.
    config.nipype.omp_nthreads = 1
    yield
    (
        config.workflow.jacobian_weighting,
        config.workflow.output_resolution,
        config.workflow.sdc_method,
        config.workflow.dwiref_construction_iters,
        config.execution.output_dir,
        config.execution.sloppy,
        config.nipype.omp_nthreads,
    ) = saved


def _edge(wf, source, target):
    edge = wf._graph.get_edge_data(wf.get_node(source), wf.get_node(target))
    assert edge is not None, f'{source} does not reach {target}'
    return edge['connect']


def test_qc_workflow_carries_both_warnings_to_the_merge():
    from qsiprep.workflows.dwi.qc import init_modelfree_qc_wf

    wf = init_modelfree_qc_wf()

    assert ('warning', 'src_qc_warning') in _edge(wf, 'raw_src_qc', 'merged_qc')
    assert ('warning', 'fib_qc_warning') in _edge(wf, 'raw_fib_qc', 'merged_qc')


def test_finalize_sinks_the_qc_warnings_reportlet(tmp_path):
    from qsiprep.interfaces.bids import DerivativesMaybeDataSink
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.execution.output_dir = str(tmp_path)
    config.execution.sloppy = False
    config.workflow.sdc_method = 'topup'
    config.workflow.dwiref_construction_iters = 0
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_dwi_finalize_wf(
        unit=make_preproc_unit([dwi]),
        name='dwi_finalize_wf',
        source_file=dwi,
        output_prefix='sub-01',
    )

    assert ('qc_warnings_report', 'in_file') in _edge(wf, 'series_qc', 'ds_report_qc_warnings')
    sink = wf.get_node('ds_report_qc_warnings')
    # The Maybe sink writes nothing when the report is Undefined, which is
    # what keeps the report quiet when QC succeeded.
    assert isinstance(sink.interface, DerivativesMaybeDataSink)
    assert sink.inputs.datatype == 'figures'
    assert sink.inputs.desc == 'qcwarnings'
    assert sink.inputs.suffix == 'dwi'


def test_distortion_group_merge_sinks_the_qc_warnings_reportlet():
    """Source check: building this workflow needs a newer QSIPlan than the dev
    environment has, and it has a SeriesQC of its own that must be wired too.
    """
    from qsiprep.workflows.dwi import distortion_group_merge

    src = inspect.getsource(distortion_group_merge)
    assert "(series_qc, ds_report_qc_warnings, [('qc_warnings_report', 'in_file')])" in src
    assert "desc='qcwarnings'" in src


def test_report_spec_lists_the_qc_warnings_reportlet():
    """Without a spec entry the reportlet is written but never shown."""
    import yaml

    spec = yaml.safe_load(load_data.readable('reports-spec.yml').read_text())
    diffusion = next(section for section in spec['sections'] if section['name'] == 'Diffusion')
    bids = [reportlet['bids'] for reportlet in diffusion['reportlets']]
    assert {'datatype': 'figures', 'desc': 'qcwarnings', 'suffix': 'dwi'} in bids
