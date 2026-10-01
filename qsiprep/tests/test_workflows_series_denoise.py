"""Each raw DWI series is conformed and denoised once, at the subject level.

A series curated into several outputs (a virtual acquisition) used to be
denoised once per output. These tests pin the shape of the lifted design: one
``init_dwi_series_denoise_wf`` per distinct file, collected per unit in
``unit.dwi_files`` order, and split by polarity inside ``pre_hmc`` by position.
"""

import numpy as np
import pytest
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.workflows.base import connect_series_to_unit, init_series_denoise_wfs
from qsiprep.workflows.dwi.merge import (
    SERIES_FIELDS,
    SERIES_LIST_FIELDS,
    init_dwi_series_denoise_wf,
)
from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf

METADATA = {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}


def _write_dwi(path, nvols=6):
    """Write a tiny valid 4D DWI (with .bval/.bvec) so header reads succeed."""
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


@pytest.fixture(autouse=True)
def _cfg():
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.execution.layout = None
    config.workflow.hmc_method = 'eddy'
    config.workflow.sdc_method = 'topup'
    config.workflow.shoreline_model = None
    config.workflow.b0_threshold = 100
    config.workflow.dwi_biascorrect = 'n4'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwidenoise_window = 5
    config.workflow.unringing_method = 'none'
    config.workflow.ignore = []
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'


def test_series_denoise_wf_conforms_then_denoises(tmp_path):
    src = _write_dwi(tmp_path / 'sub-01_run-1_dwi.nii.gz')
    wf = init_dwi_series_denoise_wf(src, metadata=METADATA, orientation='LAS')

    assert wf.name == 'dwi_denoise_run_1_dwi_wf'
    conform = wf.get_node('conform_dwi')
    assert conform.inputs.orientation == 'LAS'
    assert conform.inputs.dwi_file == src
    assert wf.get_node('denoise_wf') is not None
    assert wf.get_node('conform_phase') is None
    # The denoiser is sized from this file's own volume count.
    denoiser = wf.get_node('denoise_wf').get_node('denoiser')
    assert denoiser.inputs.extent == (5, 5, 5)
    assert set(wf.get_node('outputnode').inputs.get()) == {
        series_field for series_field, _ in SERIES_FIELDS
    }


def test_series_denoise_wf_conforms_the_phase_companion(tmp_path):
    src = _write_dwi(tmp_path / 'sub-01_part-mag_dwi.nii.gz')
    phase = _write_dwi(tmp_path / 'sub-01_part-phase_dwi.nii.gz')

    wf = init_dwi_series_denoise_wf(src, metadata=METADATA, phase_file=phase)
    assert wf.get_node('conform_phase').inputs.dwi_file == phase

    config.workflow.ignore = ['phase']
    wf = init_dwi_series_denoise_wf(src, metadata=METADATA, phase_file=phase)
    assert wf.get_node('conform_phase') is None


def test_series_denoise_wfs_are_built_once_per_file(tmp_path):
    """A file in two units (a virtual acquisition) gets a single workflow."""
    shared = _write_dwi(tmp_path / 'sub-01_acq-shared_dwi.nii.gz')
    only_a = _write_dwi(tmp_path / 'sub-01_acq-A_dwi.nii.gz')
    only_b = _write_dwi(tmp_path / 'sub-01_acq-B_dwi.nii.gz')
    unit_a = make_preproc_unit([shared, only_a], output_name='sub-01_acq-A')
    unit_b = make_preproc_unit([only_b, shared], output_name='sub-01_acq-B')

    series_wfs = init_series_denoise_wfs([unit_a, unit_b], orientation='LAS')

    assert set(series_wfs) == {shared, only_a, only_b}
    assert len({id(wf) for wf in series_wfs.values()}) == 3
    assert all(
        wf.get_node('conform_dwi').inputs.orientation == 'LAS' for wf in series_wfs.values()
    )


def test_connect_series_to_unit_collects_in_unit_order(tmp_path):
    shared = _write_dwi(tmp_path / 'sub-01_acq-shared_dwi.nii.gz')
    only_b = _write_dwi(tmp_path / 'sub-01_acq-B_dwi.nii.gz')
    unit = make_preproc_unit([only_b, shared], output_name='sub-01_acq-B')
    series_wfs = init_series_denoise_wfs([unit], orientation='LPS')

    parent = pe.Workflow(name='parent')
    stub_preproc = pe.Workflow(name='dwi_preproc_wf')
    stub_preproc.add_nodes(
        [pe.Node(niu.IdentityInterface(fields=list(SERIES_LIST_FIELDS)), name='inputnode')]
    )
    connect_series_to_unit(parent, series_wfs, unit, stub_preproc, 'acq_B')

    for series_field, list_field in SERIES_FIELDS:
        collect = parent.get_node(f'collect_{list_field}_acq_B')
        assert collect is not None
        assert collect.interface._numinputs == len(unit.dwi_files)
        # in1 is the unit's first file (only_b), in2 the shared one.
        for index, path in enumerate(unit.dwi_files, start=1):
            edge = parent._graph.get_edge_data(series_wfs[path], collect)
            assert (f'outputnode.{series_field}', f'in{index}') in edge['connect']
        edge = parent._graph.get_edge_data(collect, stub_preproc)
        assert ('out', f'inputnode.{list_field}') in edge['connect']


def test_pre_hmc_exposes_the_series_list_inputs():
    wf = init_dwi_pre_hmc_wf(
        make_preproc_unit(['/data/sub-01_dwi.nii.gz']),
        orientation='LAS',
        source_file='/data/sub-01_dwi.nii.gz',
        do_biascorr=True,
    )
    assert set(wf.get_node('inputnode').inputs.get()) == set(SERIES_LIST_FIELDS)
    # The single-polarity path hands the lists straight to the merge workflow.
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('merge_dwis_wf'))
    assert set(edge['connect']) == {(f, f'inputnode.{f}') for f in SERIES_LIST_FIELDS}


def test_pre_hmc_rpe_selects_each_polarity_by_position():
    main = '/data/sub-01_dir-AP_dwi.nii.gz'
    partner = '/data/sub-01_dir-PA_dwi.nii.gz'
    unit = make_preproc_unit(
        [partner, main],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
    )
    wf = init_dwi_pre_hmc_wf(unit, orientation='LAS', source_file=main, do_biascorr=True)

    # unit.dwi_files is (partner, main): PA is position 0, AP position 1.
    assert wf.get_node('select_plus').inputs.indices == [1]
    assert wf.get_node('select_minus').inputs.indices == [0]
    for selector, merge in (('select_plus', 'merge_plus'), ('select_minus', 'merge_minus')):
        edge = wf._graph.get_edge_data(wf.get_node(selector), wf.get_node(merge))
        assert set(edge['connect']) == {(f, f'inputnode.{f}') for f in SERIES_LIST_FIELDS}

    # The edges must survive flattening. Nipype empties an edge's connection
    # list as it expands a sub-workflow, so two edges built from one list
    # object leave the second polarity with nothing feeding it.
    flat = wf._create_flat_graph()
    for selector, merge in (('select_plus', 'merge_plus'), ('select_minus', 'merge_minus')):
        (merge_inputnode,) = [
            node for node in flat.nodes() if node.fullname.endswith(f'{merge}.inputnode')
        ]
        fed = {
            (source.name, dest_field)
            for source, _, data in flat.in_edges(merge_inputnode, data=True)
            for _, dest_field in data['connect']
        }
        assert fed == {(selector, field) for field in SERIES_LIST_FIELDS}


def test_bidirectional_pre_hmc_runs_from_raw_series_to_one_merged_file(tmp_path):
    """Test that a reverse-PE unit executes from raw files to the merged series.

    Building the graph is not enough to show the per-series results reach both
    polarities, so this runs it: conform each series once, collect them for
    the unit, split by polarity, merge each polarity, and concatenate the two.
    Denoising is off and the DSI Studio QC is dropped, so only Python runs.
    """
    import os.path as op

    import nibabel as nb

    config.workflow.denoise_method = 'none'

    def write_series(name):
        path = tmp_path / name
        data = np.random.default_rng(0).integers(50, 100, (6, 6, 6, 4)).astype(np.int16)
        nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
        stem = str(path).split('.nii')[0]
        np.savetxt(stem + '.bval', np.array([[0, 1000, 1000, 1000]]), fmt='%d')
        np.savetxt(stem + '.bvec', np.eye(3, 4, k=1), fmt='%.1f')
        return str(path)

    ap = write_series('sub-01_dir-AP_dwi.nii.gz')
    pa = write_series('sub-01_dir-PA_dwi.nii.gz')
    unit = make_preproc_unit(
        [ap, pa], method=CorrectionMethod.PEPOLAR, pe_dirs={ap: 'j', pa: 'j-'}
    )

    parent = pe.Workflow(name='parent', base_dir=str(tmp_path / 'work'))
    series_wfs = init_series_denoise_wfs([unit], orientation='LAS')
    pre_hmc_wf = init_dwi_pre_hmc_wf(unit, orientation='LAS', source_file=ap, do_biascorr=False)
    pre_hmc_wf.remove_nodes([node for node in pre_hmc_wf._graph.nodes() if 'qc' in node.name])
    connect_series_to_unit(parent, series_wfs, unit, pre_hmc_wf, 'rpe')
    parent.config['execution'] = {
        'stop_on_first_crash': True,
        'crashdump_dir': str(tmp_path / 'crash'),
    }

    graph = parent.run(plugin='Linear')

    results = {node.fullname.split('pre_hmc_wf.')[-1]: node.result.outputs for node in graph}
    merged = results['rpe_concat']
    assert nb.load(merged.out_dwi).shape == (6, 6, 6, 8)
    origins = [op.basename(path) for path in merged.original_images]
    assert origins == ['sub-01_dir-AP_dwi.nii.gz'] * 4 + ['sub-01_dir-PA_dwi.nii.gz'] * 4
    # Each polarity merged its own series, and only that one.
    assert {op.basename(f) for f in results['merge_plus.merge_dwis'].original_images} == {
        'sub-01_dir-AP_dwi.nii.gz'
    }
    assert {op.basename(f) for f in results['merge_minus.merge_dwis'].original_images} == {
        'sub-01_dir-PA_dwi.nii.gz'
    }
