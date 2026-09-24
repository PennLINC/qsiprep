"""Tests for the qsiprep.workflows.dwi.merge module."""

import json
import os
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nipype.interfaces import io as nio
from nipype.interfaces.base import isdefined
from nipype.pipeline import engine as pe

from qsiprep import config
from qsiprep.interfaces import mrtrix
from qsiprep.interfaces.dipy import Patch2Self
from qsiprep.tests.utils import field_of_view
from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf


def _use_dwidenoise2_config(monkeypatch, tmp_path, settings):
    """Point config.workflow.dwidenoise2_config at a JSON file holding ``settings``.

    ``None`` clears the setting.
    """
    path = None
    if settings is not None:
        path = tmp_path / 'dwidenoise2.json'
        path.write_text(json.dumps(settings))
    monkeypatch.setattr(config.workflow, 'dwidenoise2_config', path)


@pytest.mark.parametrize('use_phase', [False, True])
def test_dwidenoise_workflow_uses_dwidenoise(monkeypatch, use_phase):
    """Test that a DWIDenoise node, not Patch2Self, is built when ``dwidenoise`` is requested."""
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise')
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=use_phase,
    )
    denoiser = workflow.get_node('denoiser')

    assert isinstance(denoiser.interface, mrtrix.DWIDenoise)
    assert denoiser.inputs.extent == (5, 5, 5)
    assert denoiser.inputs.nthreads == 1


def test_dwidenoise_workflow_resolves_auto_window(monkeypatch):
    """Test that the default ``auto`` window resolves into a cuboid extent for dwidenoise."""
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise')
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 'auto')
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
    )
    denoiser = workflow.get_node('denoiser')

    # cbrt(30) rounded up to the closest odd integer
    assert denoiser.inputs.extent == (5, 5, 5)


def test_dwidenoise2_workflow_ignores_denoise_window(monkeypatch):
    """Test that dwidenoise2 leaves the kernel to its schedule rather than the requested window.

    dwidenoise2 sizes its patches per iteration from its multi-resolution schedule and
    exposes no kernel options, so ``--dwidenoise-window`` cannot apply to it.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise2')
    monkeypatch.setattr(config.workflow, 'dwidenoise2_config', None)
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
    )
    denoiser = workflow.get_node('denoiser')

    for removed in ('shape', 'radius', 'extent', 'subsample'):
        assert not hasattr(denoiser.inputs, removed)
    # No schedule is requested either, so dwidenoise2 uses its bundled default
    assert not isdefined(denoiser.inputs.schedule)


def test_dwidenoise2_config_reaches_workflow(monkeypatch, tmp_path):
    """Test that the --dwidenoise2-config settings are forwarded to the workflow node."""
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise2')
    _use_dwidenoise2_config(
        monkeypatch, tmp_path, {'demodulate': 'hann', 'decomposition': 'bdcsvd'}
    )
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        # demodulation is only valid for complex-valued data
        use_phase=True,
    )
    denoiser = workflow.get_node('denoiser')

    assert denoiser.inputs.demodulate == 'hann'
    assert denoiser.inputs.decomposition == 'bdcsvd'
    # Parameters that weren't requested are left at the dwidenoise2 defaults
    assert not isdefined(denoiser.inputs.estimator)
    assert not isdefined(denoiser.inputs.schedule)


@pytest.mark.parametrize('demodulate', ['linear', 'hann', 'apc'])
def test_dwidenoise2_rejects_demodulation_without_phase(monkeypatch, tmp_path, demodulate):
    """Test that phase demodulation is rejected unless phase data are available.

    ``dwidenoise2`` errors out partway through a run when asked to demodulate
    magnitude-only data, so the workflow rejects the request up front instead.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise2')
    _use_dwidenoise2_config(monkeypatch, tmp_path, {'demodulate': demodulate})
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    kwargs = {
        'source_file': 'sub-01_dwi.nii.gz',
        'partial_fourier': 1.0,
        'phase_encoding_direction': 'j',
        'n_volumes': 30,
    }

    with pytest.raises(ValueError, match='magnitude-only data'):
        init_dwi_denoising_wf(use_phase=False, **kwargs)

    # The same request is fine once phase data are available
    workflow = init_dwi_denoising_wf(use_phase=True, **kwargs)
    assert workflow.get_node('denoiser').inputs.demodulate == demodulate


def test_dwidenoise2_config_schedule_reaches_workflow(monkeypatch, tmp_path):
    rows = [
        {'spatial_subsample': 4, 'update_noise': True},
        {'spatial_subsample': [1, 1, 1], 'kernel': 'rank'},
    ]
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise2')
    _use_dwidenoise2_config(monkeypatch, tmp_path, {'schedule': rows})
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 'auto')
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
    )
    denoiser = workflow.get_node('denoiser')

    assert denoiser.inputs.schedule == [
        {'spatial_subsample': 4, 'update_noise': True},
        {'spatial_subsample': (1, 1, 1), 'kernel': 'rank'},
    ]


def _run_denoising_wf(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    use_phase,
    dwidenoise_window='auto',
    unringing_method='none',
    mrtrix_version='dev',
    dwidenoise2_settings=None,
):
    """Build and execute a denoising workflow on the nibs DWI series.

    Bias correction and b=0 harmonization are disabled; unringing is off unless
    ``unringing_method`` says otherwise.

    Returns
    -------
    nodes : dict
        The executed nodes, keyed by node name.
    sink_dir : :obj:`pathlib.Path`
        Directory holding the files that reached the workflow's ``outputnode``.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', denoise_method)
    _use_dwidenoise2_config(monkeypatch, tmp_path, dwidenoise2_settings)
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', dwidenoise_window)
    monkeypatch.setattr(config.workflow, 'unringing_method', unringing_method)
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    # config.workflow.init() mutates os.environ['PATH'] directly, which monkeypatch
    # would not undo. Setting PATH through monkeypatch first registers it for
    # restoration at teardown.
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    config.workflow.init()
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    metadata = json.loads(Path(nibs_dwi['json_file']).read_text())

    denoise_wf = init_dwi_denoising_wf(
        source_file=nibs_dwi['dwi_file'],
        partial_fourier=metadata['PartialFourier'],
        phase_encoding_direction=metadata['PhaseEncodingDirection'].replace('-', ''),
        n_volumes=nb.load(nibs_dwi['dwi_file']).shape[3],
        use_phase=use_phase,
    )
    denoise_wf.inputs.inputnode.dwi_file = nibs_dwi['dwi_file']
    denoise_wf.inputs.inputnode.bval_file = nibs_dwi['bval_file']
    denoise_wf.inputs.inputnode.bvec_file = nibs_dwi['bvec_file']
    if use_phase:
        denoise_wf.inputs.inputnode.dwi_phase_file = nibs_dwi['phase_file']

    # nipype prunes IdentityInterface nodes out of the execution graph, so ``outputnode``
    # can't be inspected directly. Routing its outputs to a DataSink both survives the
    # pruning and checks that the workflow really connects them.
    sink_dir = tmp_path / 'sink'
    sink = pe.Node(
        nio.DataSink(base_directory=str(sink_dir), parameterization=False),
        name='sink',
    )
    workflow = pe.Workflow(name='denoise_test_wf', base_dir=str(tmp_path))
    workflow.connect([
        (denoise_wf, sink, [
            ('outputnode.dwi_file', 'dwi_file'),
            ('outputnode.noise_image', 'noise_image'),
            ('outputnode.confounds', 'confounds'),
        ]),
    ])  # fmt:skip

    # nipype raises if any node fails, so a returned graph means every node ran
    graph = workflow.run(plugin='Linear')

    return {node.name: node for node in graph.nodes}, sink_dir


def _sink_output(sink_dir, field):
    """Return the single file the DataSink wrote for ``field``."""
    matches = sorted((sink_dir / field).glob('*'))
    assert len(matches) == 1, f'expected one {field} file, found {matches}'
    return matches[0]


def _assert_denoiser_is_not_masked(nodes):
    """Check that the denoiser processes the full FOV rather than a masked subset."""
    assert 'quick_mask' not in nodes
    assert not isdefined(nodes['denoiser'].inputs.mask)


def _assert_denoising_outputs(nodes, sink_dir, raw_file):
    """Check the files produced by an executed denoising workflow."""
    raw_img = nb.load(raw_file)
    denoiser_outputs = nodes['denoiser'].result.outputs

    denoised_img = nb.load(_sink_output(sink_dir, 'dwi_file'))
    assert denoised_img.shape == raw_img.shape
    assert np.allclose(denoised_img.affine, raw_img.affine)
    assert np.all(np.isfinite(denoised_img.get_fdata()))
    # The workflow always returns magnitude data, even when it denoises complex data
    assert not np.issubdtype(denoised_img.header.get_data_dtype(), np.complexfloating)

    noise_img = nb.load(_sink_output(sink_dir, 'noise_image'))
    assert noise_img.ndim == 3
    assert np.allclose(field_of_view(noise_img), field_of_view(raw_img), rtol=0.05)
    noise_data = noise_img.get_fdata()
    finite = np.isfinite(noise_data)
    assert finite.any()
    assert np.all(noise_data[finite] >= 0)

    assert len(pd.read_csv(_sink_output(sink_dir, 'confounds'))) == raw_img.shape[3]

    assert os.path.isfile(denoiser_outputs.out_report)


# The single row of dwidenoise2's bundled "legacy" schedule
_LEGACY_SCHEDULE_ROW = {
    'spatial_subsample': 1,
    'temporal_subsample': 1,
    'partitions': 1,
    'smooth_noise': False,
    'update_noise': True,
    'kernel': 'cuboid=1x',
}


@pytest.mark.parametrize(
    (
        'denoise_method',
        'dwidenoise2_settings',
        'dwidenoise_window',
        'interface',
        'expected_inputs',
    ),
    [
        pytest.param(
            'dwidenoise',
            None,
            5,
            mrtrix.DWIDenoise,
            {'extent': (5, 5, 5)},
            id='dwidenoise_window5',
        ),
        pytest.param(
            'dwidenoise',
            None,
            'auto',
            mrtrix.DWIDenoise,
            {'extent': (5, 5, 5)},
            id='dwidenoise_auto',
        ),
        # Every option is left at its default, so the bundled 'default' schedule sizes the
        # kernel and the mrm2023 estimator is used
        pytest.param(
            'dwidenoise2', None, 'auto', mrtrix.DWIDenoise2, {}, id='dwidenoise2_default'
        ),
        pytest.param(
            'dwidenoise2',
            {'decomposition': 'selfadjoint'},
            'auto',
            mrtrix.DWIDenoise2,
            {'decomposition': 'selfadjoint'},
            id='dwidenoise2_selfadjoint',
        ),
        pytest.param(
            'dwidenoise2',
            {'filter_method': 'optthresh'},
            'auto',
            mrtrix.DWIDenoise2,
            {'filter_method': 'optthresh'},
            id='dwidenoise2_optthresh',
        ),
        pytest.param(
            'dwidenoise2',
            {'estimator': 'exp2'},
            'auto',
            mrtrix.DWIDenoise2,
            {'estimator': 'exp2'},
            id='dwidenoise2_exp2',
        ),
        # The bundled "legacy" schedule written out inline, so that dwidenoise2 runs with a
        # schedule file that QSIPrep generated
        pytest.param(
            'dwidenoise2',
            {'schedule': [_LEGACY_SCHEDULE_ROW]},
            'auto',
            mrtrix.DWIDenoise2,
            {'schedule': [_LEGACY_SCHEDULE_ROW]},
            id='dwidenoise2_inline_schedule',
        ),
        pytest.param('patch2self', None, 'auto', Patch2Self, {}, id='patch2self'),
    ],
)
def test_denoising_wf_magnitude(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    dwidenoise2_settings,
    dwidenoise_window,
    interface,
    expected_inputs,
):
    """Test denoising magnitude-only DWI data with each supported method."""
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method=denoise_method,
        use_phase=False,
        dwidenoise_window=dwidenoise_window,
        dwidenoise2_settings=dwidenoise2_settings,
    )
    denoiser = nodes['denoiser']
    assert isinstance(denoiser.interface, interface)
    for name, value in expected_inputs.items():
        assert getattr(denoiser.inputs, name) == value

    # Magnitude-only data never goes through the complex-valued path
    assert 'combine_complex' not in nodes
    assert 'split_complex' not in nodes

    _assert_denoiser_is_not_masked(nodes)
    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])


@pytest.mark.parametrize(
    ('denoise_method', 'dwidenoise2_settings', 'interface', 'expected_inputs'),
    [
        pytest.param(
            'dwidenoise', None, mrtrix.DWIDenoise, {'extent': (5, 5, 5)}, id='dwidenoise'
        ),
        pytest.param('dwidenoise2', None, mrtrix.DWIDenoise2, {}, id='dwidenoise2'),
        pytest.param(
            'dwidenoise2',
            {'demodulate': 'hann'},
            mrtrix.DWIDenoise2,
            {'demodulate': 'hann'},
            id='dwidenoise2_demodulate',
        ),
        pytest.param('patch2self', None, Patch2Self, {}, id='patch2self_ignores_phase'),
    ],
)
def test_denoising_wf_complex(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    dwidenoise2_settings,
    interface,
    expected_inputs,
):
    """Test denoising DWI data when phase data are available.

    Only the dwidenoise variants combine the magnitude and phase data into a
    complex-valued series. ``patch2self`` ignores the phase data and denoises the
    magnitude data alone.
    """
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method=denoise_method,
        use_phase=True,
        dwidenoise2_settings=dwidenoise2_settings,
    )
    denoiser = nodes['denoiser']
    assert isinstance(denoiser.interface, interface)
    for name, value in expected_inputs.items():
        assert getattr(denoiser.inputs, name) == value

    uses_complex = denoise_method.startswith('dwidenoise')
    assert ('combine_complex' in nodes) is uses_complex
    assert ('split_complex' in nodes) is uses_complex
    if uses_complex:
        complex_img = nb.load(nodes['combine_complex'].result.outputs.out_file)
        assert np.issubdtype(complex_img.header.get_data_dtype(), np.complexfloating)

    _assert_denoiser_is_not_masked(nodes)
    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])


def _connections(workflow):
    """Map (source node name, destination node name) to the connected field pairs."""
    return {
        (src.name, dest.name): set(data['connect'])
        for src, dest, data in workflow._graph.edges(data=True)
    }


def _build_denoising_wf(
    monkeypatch,
    denoise_method,
    unringing_method,
    use_phase,
    mrtrix_version='dev',
):
    """Build (without running) a denoising workflow with the given configuration.

    ``mrtrix_version`` defaults to ``'dev'`` because the complex-path tests in this
    module were written against the development branch, where mrdegibbs reads and
    writes complex data.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', denoise_method)
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', unringing_method)
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    return init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=use_phase,
    )


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_complex_data_stay_complex_through_mrdegibbs(monkeypatch, denoise_method):
    """Test that mrdegibbs gets complex-valued denoised data, split to magnitude after it.

    mrdegibbs is built on the Fourier shift theorem, so it works better on complex
    data; MRtrix3's development branch reads and writes it.
    """
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'mrdegibbs', use_phase=True)
    connections = _connections(workflow)

    assert connections[('combine_complex', 'denoiser')] == {('out_file', 'in_file')}
    assert connections[('denoiser', 'degibbser')] == {('out_file', 'in_file')}
    assert connections[('degibbser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'outputnode')] == {('out_file', 'dwi_file')}
    # The split happens once, after unringing, not before it
    assert ('denoiser', 'split_complex') not in connections


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_stable_mrtrix_splits_before_mrdegibbs(monkeypatch, denoise_method):
    """Test that data are reduced to magnitude before mrdegibbs under a released MRtrix3.

    3.0.x mrdegibbs cannot read complex data, so handing it complex input would fail
    at runtime. This is the behavior QSIPrep had before complex unringing existed.
    """
    workflow = _build_denoising_wf(
        monkeypatch, denoise_method, 'mrdegibbs', use_phase=True, mrtrix_version='stable'
    )
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'degibbser')] == {('out_file', 'in_file')}
    assert connections[('degibbser', 'outputnode')] == {('out_file', 'dwi_file')}
    # The split happens once, before unringing, not after it
    assert ('degibbser', 'split_complex') not in connections


def test_stable_mrdegibbs_says_what_dev_would_buy(monkeypatch, caplog):
    """Test that the user is told complex unringing exists, only where it is actionable.

    The message belongs at workflow-build time rather than parse time: use_phase is a
    per-scan property the parser cannot know. Assert visibility at the real default
    log level (25, see ``execution.log_level``) rather than lowering the threshold to
    INFO, so this proves a real run would actually show the message.
    """
    caplog.set_level(25, logger='nipype.workflow')
    _build_denoising_wf(
        monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True, mrtrix_version='stable'
    )

    assert '--mrtrix-version dev' in caplog.text


@pytest.mark.parametrize(
    ('unringing_method', 'mrtrix_version'),
    [('rpg', 'stable'), ('none', 'stable'), ('mrdegibbs', 'dev')],
)
def test_no_advice_when_mrdegibbs_is_not_running(
    monkeypatch, caplog, unringing_method, mrtrix_version
):
    """Test that no advice is given where it would not apply.

    rpg is magnitude-only regardless of version; with none, unringing does not run at
    all; and with dev + mrdegibbs, complex data are already carried through unringing,
    so there is nothing dev would additionally buy.
    """
    caplog.set_level(25, logger='nipype.workflow')
    _build_denoising_wf(
        monkeypatch,
        'dwidenoise',
        unringing_method,
        use_phase=True,
        mrtrix_version=mrtrix_version,
    )

    assert '--mrtrix-version dev' not in caplog.text


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_rpg_unringing_gets_magnitude(monkeypatch, denoise_method):
    """Test that data are split to magnitude before rpg unringing (TORTOISE, magnitude-only)."""
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'rpg', use_phase=True)
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'degibbser')] == {('out_file', 'in_file')}
    assert ('degibbser', 'split_complex') not in connections


@pytest.mark.parametrize('unringing_method', ['mrdegibbs', 'rpg', 'none'])
def test_patch2self_never_goes_complex(monkeypatch, unringing_method):
    """Test that patch2self runs stay in the magnitude domain, whatever the unringing."""
    workflow = _build_denoising_wf(monkeypatch, 'patch2self', unringing_method, use_phase=True)
    node_names = {node.name for node in workflow._get_all_nodes()}

    assert 'combine_complex' not in node_names
    assert 'split_complex' not in node_names


@pytest.mark.parametrize('unringing_method', ['mrdegibbs', 'rpg', 'none'])
def test_magnitude_only_input_never_goes_complex(monkeypatch, unringing_method):
    """Test that magnitude-only runs stay magnitude even with a complex-capable denoiser."""
    workflow = _build_denoising_wf(monkeypatch, 'dwidenoise', unringing_method, use_phase=False)
    node_names = {node.name for node in workflow._get_all_nodes()}

    assert 'combine_complex' not in node_names
    assert 'split_complex' not in node_names


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_split_follows_the_denoiser_without_unringing(monkeypatch, denoise_method):
    """Test that data are split to magnitude right after denoising when no unringing runs."""
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'none', use_phase=True)
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'outputnode')] == {('out_file', 'dwi_file')}


def test_boilerplate_describes_where_the_split_happens(monkeypatch):
    """Test that the boilerplate says unringing ran on complex data, with the split after it."""
    complex_degibbs = _build_denoising_wf(monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True)
    assert 'complex-valued' in complex_degibbs.__desc__
    assert 'After denoising, the complex-valued data were split' not in complex_degibbs.__desc__
    # The split is described after unringing is described
    assert complex_degibbs.__desc__.index('Gibbs ringing') < complex_degibbs.__desc__.index(
        'split back into magnitude'
    )

    # rpg is magnitude-only, so the split is still described right after denoising
    complex_rpg = _build_denoising_wf(monkeypatch, 'dwidenoise', 'rpg', use_phase=True)
    assert complex_rpg.__desc__.index('split back into magnitude') < complex_rpg.__desc__.index(
        'Gibbs ringing'
    )


def test_boilerplate_says_magnitude_under_stable_mrtrix(monkeypatch):
    """Test that the boilerplate describes what actually ran under a released MRtrix3.

    Released mrdegibbs sees magnitude data only.

    The denoising step still describes combining magnitude and phase into a
    complex-valued file -- that is unaffected by --mrtrix-version, only mrdegibbs is
    gated -- so the assertion targets the unringing-specific phrasing rather than
    the substring "complex-valued data" wholesale.
    """
    workflow = _build_denoising_wf(
        monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True, mrtrix_version='stable'
    )

    assert 'Gibbs ringing was removed from the magnitude data' in workflow.__desc__
    assert 'Gibbs ringing was removed from the complex-valued data' not in workflow.__desc__
    assert workflow.__desc__.index('split back into magnitude') < workflow.__desc__.index(
        'Gibbs ringing'
    )


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_denoising_wf_complex_mrdegibbs(monkeypatch, tmp_path, nibs_dwi, denoise_method):
    """Test that mrdegibbs runs on complex-valued data and returns magnitude.

    This is the only test that proves the MRtrix3 in the image really accepts and
    emits complex data; the graph-shape tests only check the wiring.
    """
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method=denoise_method,
        use_phase=True,
        unringing_method='mrdegibbs',
    )

    degibbser = nodes['degibbser']
    degibbs_in = nb.load(degibbser.inputs.in_file)
    assert np.issubdtype(degibbs_in.header.get_data_dtype(), np.complexfloating)

    degibbs_out = nb.load(degibbser.result.outputs.out_file)
    assert np.issubdtype(degibbs_out.header.get_data_dtype(), np.complexfloating)
    assert degibbs_out.shape == degibbs_in.shape

    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])


def test_denoising_wf_stable_mrdegibbs(monkeypatch, tmp_path, nibs_dwi):
    """Test that the released mrdegibbs unrings magnitude data and keeps the result real.

    The graph-shape tests check that the split precedes unringing; this one checks
    that the binary the reordered PATH selects actually accepts what it is given.
    """
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method='dwidenoise',
        use_phase=True,
        unringing_method='mrdegibbs',
        mrtrix_version='stable',
    )

    degibbser = nodes['degibbser']
    degibbs_in = nb.load(degibbser.inputs.in_file)
    assert not np.issubdtype(degibbs_in.header.get_data_dtype(), np.complexfloating)

    degibbs_out = nb.load(degibbser.result.outputs.out_file)
    assert not np.issubdtype(degibbs_out.header.get_data_dtype(), np.complexfloating)
    assert degibbs_out.shape == degibbs_in.shape

    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])


def test_no_bias_correction_plumbing_survives_in_the_merge_stack(monkeypatch):
    """Test that no bias-correction plumbing survives in the merge stack.

    The pre-0.17 ("legacy") bias-correction path is gone, plumbing included.

    Deleting only the ``biascorr`` node would leave ``bias_image``/``bias_images``
    output traits and ``Merge`` nodes fed with undefined values, which fails at
    runtime rather than at construction -- so assert on the traits too.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', 'none')
    monkeypatch.setattr(config.workflow, 'dwidenoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    # No do_biascorr argument: its absence is the point.
    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
    )

    assert not [name for name in workflow.list_node_names() if 'bias' in name]
    outputs = workflow.get_node('outputnode').outputs.copyable_trait_names()
    assert 'bias_image' not in outputs
