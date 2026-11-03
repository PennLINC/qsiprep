"""Tests for writing the preprocessed DWI in the anat and dwiref spaces."""

import pytest

from qsiprep import config
from qsiprep.tests.test_multires_merge_integration import (
    N_VOLUMES,
    _shelled_gradients,
    _write_two_pair_dataset,
)
from qsiprep.tests.test_output_spaces_naming import (
    ANAT_BASE,
    DWI_BASE,
    FIGURE_BASE,
    assert_no_collisions,
    collect_datasink_entities,
    collect_figure_entities,
    render_datasink_paths,
)


@pytest.fixture(autouse=True)
def _reset_execution_config():
    """Drop the execution state parse_args leaves on the config singleton."""

    def reset():
        config.execution._layout = None
        config.execution.bids_database_dir = None
        config.execution.session_label = None

    reset()
    yield
    reset()


def _build_finalize(tmp_path, output_spaces, write_derivatives=True, make_dwiref=False):
    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.tests.test_output_spaces_naming import _write_dwi
    from qsiprep.utils.spaces import dwi_specs_of, parse_output_spaces
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.workflow.output_spaces = output_spaces
    config.nipype.omp_nthreads = 1
    config.workflow.hmc_method = 'tortoise'
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.b0_threshold = 100
    config.workflow.intramodal_template_iters = 0
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    dwi_specs = dwi_specs_of(parse_output_spaces(output_spaces))

    src = _write_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_dwi_finalize_wf(
        unit=make_preproc_unit([src]),
        name='dwi_finalize_wf',
        source_file=src,
        output_prefix='sub-01',
        dwi_specs=dwi_specs,
        write_derivatives=write_derivatives,
        make_dwiref=make_dwiref,
    )
    return wf


def _edges_into(wf, node_name):
    node = wf.get_node(node_name)
    assert node is not None, f'{node_name} is missing'
    return {
        (src.name, source, dest)
        for src, _, data in wf._graph.in_edges(node, data=True)
        for source, dest in data['connect']
    }


def test_anat_dwi_sinks_name_their_space(tmp_path):
    """Test that an anat spec writes space-anat beside the ACPC outputs."""
    wf = _build_finalize(tmp_path, ['acpc:res-2mm', 'anat:res-2mm'])
    found = collect_datasink_entities(wf, full_names=True)
    anat = {name: e for name, e in found.items() if name.startswith('dwi_derivatives_wf_anat')}
    assert anat
    assert {e.get('space') for e in anat.values()} == {'anat'}
    acpc = {name: e for name, e in found.items() if name.startswith('dwi_derivatives_wf_acpc')}
    assert {e.get('space') for e in acpc.values()} == {'ACPC'}

    data_sinks = {name: e for name, e in found.items() if e.get('datatype') != 'figures'}
    paths = render_datasink_paths(data_sinks, DWI_BASE)
    assert_no_collisions(paths)
    assert (
        'sub-01/ses-1/dwi/sub-01_ses-1_space-anat_res-2mm_desc-preproc_dwi.nii.gz'
        in paths.values()
    )


def test_anat_resampling_ends_with_the_acpc_to_anat_rigid(tmp_path):
    """Test that an anat spec appends acpc_inv_transform to the chain, and only it."""
    wf = _build_finalize(tmp_path, ['acpc:res-2mm', 'anat:res-2mm'])
    anat_edges = _edges_into(wf, 'dwi_trans_wf_anat_res2mm')
    assert ('inputnode', 'acpc_inv_transform', 'inputnode.trailing_transform') in anat_edges
    acpc_edges = _edges_into(wf, 'dwi_trans_wf_acpc_res2mm')
    assert not any(dest == 'inputnode.trailing_transform' for _, _, dest in acpc_edges)

    trans_wf = wf.get_node('dwi_trans_wf_anat_res2mm')
    for node_name in ('compose_transforms', 'scale_dwis'):
        edges = _edges_into(trans_wf, node_name)
        assert ('inputnode', 'trailing_transform', 'trailing_transform') in edges, node_name


@pytest.mark.parametrize(
    ('make_dwiref', 'compose_drop', 'transport_drop'),
    [
        (
            False,
            ['to b=0 affine', 'to b=0 warp', 'b=0 to T1w'],
            ['to b=0 affine', 'to b=0 warp', 'b=0 to T1w'],
        ),
        (True, ['b=0 to T1w'], ['b=0 to T1w']),
    ],
)
def test_dwiref_drops_the_coregistration(tmp_path, make_dwiref, compose_drop, transport_drop):
    """Test that a dwiref output stops in its reference frame, at either level."""
    wf = _build_finalize(tmp_path, ['acpc:res-2mm', 'dwiref:res-2mm'], make_dwiref=make_dwiref)
    trans_wf = wf.get_node('dwi_trans_wf_dwiref_res2mm')
    assert trans_wf.get_node('compose_transforms').inputs.drop_stages == compose_drop
    assert trans_wf.get_node('scale_dwis').inputs.drop_stages == transport_drop
    acpc_wf = wf.get_node('dwi_trans_wf_acpc_res2mm')
    assert acpc_wf.get_node('compose_transforms').inputs.drop_stages == []

    level = 'subject' if make_dwiref else 'distortiongroup'
    sink = wf.get_node('dwi_derivatives_wf_dwiref_res2mm').get_node('ds_dwi_t1')
    assert sink.inputs.space == level


def test_acpc_only_finalize_is_unchanged(tmp_path):
    """Test that an ACPC-only build keeps the whole chain and its node names."""
    wf = _build_finalize(tmp_path, ['acpc:res-2mm'])
    trans_wf = wf.get_node('dwi_trans_wf')
    assert trans_wf.get_node('compose_transforms').inputs.drop_stages == []
    assert not any(
        dest == 'inputnode.trailing_transform' for _, _, dest in _edges_into(wf, 'dwi_trans_wf')
    )
    assert wf.get_node('dwi_mask_to_acpc') is None
    figures = collect_figure_entities(wf)
    assert all('space' not in entities for entities in figures.values())


@pytest.mark.parametrize(
    ('token', 'make_dwiref', 'transform'),
    [
        ('anat:res-2mm', False, 'acpc_transform'),
        ('dwiref:res-2mm', False, 'itk_b0_to_t1'),
        ('dwiref:res-2mm', True, 'dwiref_to_t1_affine'),
    ],
)
def test_dice_is_computed_in_acpc_for_every_space(tmp_path, token, make_dwiref, transform):
    """Test that an output outside ACPC carries its mask into ACPC for the Dice (D4)."""
    from qsiprep.utils.spaces import parse_space_token

    wf = _build_finalize(tmp_path, ['acpc:res-2mm', token], make_dwiref=make_dwiref)
    (spec,) = parse_space_token(token)
    suffix = f'_{spec.space}_res2mm'
    edges = _edges_into(wf, f'dwi_mask_to_acpc{suffix}')
    assert ('inputnode', 't1_mask', 'reference_image') in edges
    assert any(src == 'inputnode' and dest == 'transforms' for src, _, dest in edges)
    sources = {source for src, source, dest in edges if dest == 'transforms'}
    names = {source[0] if isinstance(source, tuple) else source for source in sources}
    assert names == {transform}
    dice_edges = _edges_into(wf.get_node(f't1_dice_calc{suffix}'), 'inputnode')
    assert dice_edges == set()  # fed from the parent workflow
    parent = _edges_into(wf, f't1_dice_calc{suffix}')
    assert (f'dwi_mask_to_acpc{suffix}', 'output_image', 'inputnode.dwi_mask') in parent


def test_per_space_figures_do_not_collide(tmp_path):
    """Test that two spaces at one resolution write distinct reportlets."""
    wf = _build_finalize(tmp_path, ['acpc:res-2mm', 'anat:res-2mm'])
    found = collect_figure_entities(wf)
    for node_name in found:
        node = wf.get_node(node_name)
        space = getattr(node.interface.inputs, 'space', None)
        if node_name.split('.')[-1] not in {'ds_report_gradients', 'ds_report_sdcwarp'}:
            assert space in {'ACPC', 'anat'}, node_name
            found[node_name]['space'] = space
    paths = render_datasink_paths(found, FIGURE_BASE, include_figures=True)
    assert_no_collisions(paths)


def test_res_native_writes_no_res_entity(tmp_path):
    """Test that res-native outputs go unlabeled, and still report their grid."""
    wf = _build_finalize(tmp_path, ['acpc:res-native'])
    found = collect_datasink_entities(wf, full_names=True)
    grid_sinks = {name: e for name, e in found.items() if e.get('space') == 'ACPC'}
    assert grid_sinks
    assert all('res' not in e for e in grid_sinks.values())
    grid_metadata = wf.get_node('grid_metadata')
    assert grid_metadata is not None
    assert not isinstance(grid_metadata.inputs.label, str)


def test_native_anatomicals_are_written_without_space():
    """Test that anat and dwiref write the anatomical frame's images with no space- or res-."""
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    for token in ('anat:res-2mm', 'dwiref'):
        specs = parse_output_spaces(['acpc:res-2mm', token, 'MNI152NLin2009cAsym'])
        wf = init_anat_derivatives_wf(output_spaces=specs)
        found = collect_datasink_entities(wf, full_names=True)
        native_names = ('ds_anat_preproc', 'ds_anat_mask', 'ds_anat_dseg')
        native = {name: found[name] for name in native_names}
        assert all('space' not in e and 'res' not in e for e in native.values())

        paths = render_datasink_paths(
            {name: {**e, 'space': None} for name, e in found.items() if 'from' not in e},
            ANAT_BASE,
            include_figures=True,
        )
        assert paths['ds_anat_preproc'] == 'sub-01/anat/sub-01_desc-preproc_T1w.nii.gz'
        assert paths['ds_anat_mask'] == 'sub-01/anat/sub-01_desc-brain_mask.nii.gz'
        assert paths['ds_anat_dseg'] == 'sub-01/anat/sub-01_dseg.nii.gz'


def test_acpc_only_writes_no_native_anatomicals():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    wf = init_anat_derivatives_wf(output_spaces=parse_output_spaces(['acpc:res-2mm']))
    assert not any(name.startswith('ds_anat_') for name in wf.list_node_names())


def _anat_preproc_wf(tmp_path, output_spaces):
    from qsiprep.tests.test_workflows_native import _build_anat_preproc_wf

    return _build_anat_preproc_wf(tmp_path, output_spaces)


def test_written_acpc_anatomicals_move_but_internal_ones_stay(tmp_path):
    """Test that only the written ACPC anatomicals move to the anatomical's voxel size (D9).

    The ACPC anatomicals on the outputnode are what DWI processing registers to,
    so they stay on the anchor template's grid, which keeps estimation unchanged.
    """
    wf = _anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym'])
    outputnode_edges = _edges_into(wf, 'outputnode')
    assert ('rigid_acpc_resample_head', 'output_image', 't1_preproc') in outputnode_edges
    assert ('rigid_acpc_resample_brain', 'output_image', 't1_brain') in outputnode_edges
    head_edges = _edges_into(wf, 'rigid_acpc_resample_head')
    assert ('anchor_lps_wf', 'outputnode.template_lps', 'reference_image') in head_edges

    derivatives_edges = _edges_into(wf, 'anat_derivatives_wf')
    assert ('written_acpc_preproc', 'output_image', 'inputnode.t1_preproc') in derivatives_edges
    assert ('written_acpc_mask', 'output_image', 'inputnode.t1_mask') in derivatives_edges
    written_edges = _edges_into(wf, 'written_acpc_preproc')
    assert ('acpc_anat_grid', 'out_file', 'reference_image') in written_edges
    assert ('anat_template_wf', 'outputnode.bias_corrected', 'input_image') in written_edges
    grid_edges = _edges_into(wf, 'acpc_anat_grid')
    assert ('anchor_lps_wf', 'outputnode.template_lps', 'fixed_image') in grid_edges
    assert ('anat_template_wf', 'outputnode.bias_corrected', 'zooms_image') in grid_edges


def test_a_templateflow_label_keeps_its_grid(tmp_path):
    """Test that MNI152NLin6Asym:res-2 is still written on TemplateFlow's res-2 grid."""
    wf = _anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin6Asym:res-2'])
    derivatives_wf = wf.get_node('anat_derivatives_wf')
    assert derivatives_wf.get_node('MNI152NLin6Asymres2_anat_grid') is None
    edges = _edges_into(derivatives_wf, 'resample_MNI152NLin6Asymres2_preproc')
    assert ('select_MNI152NLin6Asymres2_template_lps', 'out', 'reference_image') in edges


def test_anatomical_frame_images_are_exposed(tmp_path):
    wf = _anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'anat:res-2mm'])
    edges = _edges_into(wf, 'outputnode')
    assert ('anat_template_wf', 'outputnode.bias_corrected', 'anat_preproc') in edges
    assert ('synthstrip_anat_wf', 'outputnode.brain_mask', 'anat_mask') in edges
    assert ('anat_aseg_to_dseg', 'out_file', 'anat_dseg') in edges


def test_synb0_without_acpc_writes_the_distortion_group_frame():
    """Test that SynB0's side output goes on the b=0 grid when no acpc space is requested."""
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from qsiprep.workflows.dwi.util import add_synb0_outputs

    def _build(output_spaces):
        config.workflow.output_spaces = output_spaces
        synb0_wf = Workflow(name='synb0_wf')
        synb0_wf.add_nodes(
            [
                pe.Node(
                    niu.IdentityInterface(
                        fields=[
                            'acquired_synthetic_report',
                            'unet_input_report',
                            'synthetic_b0',
                            'synthetic_b0_acpc',
                            'qc_file',
                        ]
                    ),
                    name='outputnode',
                )
            ]
        )
        workflow = Workflow(name='parent_wf')
        add_synb0_outputs(workflow, synb0_wf, '/data/sub-01/dwi/sub-01_dwi.nii.gz')
        sink = workflow.get_node('ds_synb0_dwiref')
        edge = workflow._graph.get_edge_data(synb0_wf, sink)
        return sink, edge['connect']

    sink, connect = _build(['anat:res-2mm'])
    assert sink.inputs.space == 'distortiongroup'
    assert str(sink.inputs.res) == '<undefined>'
    assert connect == [('outputnode.synthetic_b0', 'in_file')]

    sink, connect = _build(['anat:res-2mm', 'acpc:res-1p5mm'])
    assert sink.inputs.space == 'ACPC'
    assert sink.inputs.res == '1p5mm'
    assert connect == [('outputnode.synthetic_b0_acpc', 'in_file')]


# ---------------------------------------------------------------------------
# The integrated subject workflow
# ---------------------------------------------------------------------------


def _write_two_output_dataset(root):
    """Write two AP/PA pairs that stay two outputs: one unit each, nothing to merge."""
    from qsiprep.tests.utils import build_test_dataset

    skeleton = {
        '01': [
            {
                'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
                'dwi': [
                    {
                        'acq': acq,
                        'dir': direction,
                        'suffix': 'dwi',
                        'metadata': {
                            'MultipartID': acq,
                            'PhaseEncodingDirection': polarity,
                            'TotalReadoutTime': readout,
                        },
                    }
                    for acq, readout in (('a', 0.05), ('b', 0.07))
                    for direction, polarity in (('AP', 'j-'), ('PA', 'j'))
                ],
            }
        ],
    }
    return build_test_dataset(
        root, skeleton, extra_files=_shelled_gradients(), n_volumes=N_VOLUMES
    )


def _subject_wf(tmp_path, *args, dataset=None):
    from qsiprep.cli.parser import parse_args
    from qsiprep.workflows.base import init_single_subject_wf

    bids_dir = dataset or _write_two_pair_dataset(tmp_path / 'bids')
    work_dir = tmp_path / 'work'
    config.from_dict({'bids_dir': str(bids_dir), 'work_dir': str(work_dir)}, init=True)
    parse_args(
        [
            str(bids_dir),
            str(tmp_path / 'out'),
            'participant',
            '--participant-label',
            '01',
            '--hmc-method',
            'tortoise',
            '--work-dir',
            str(work_dir),
            '--skip-bids-validation',
            *args,
        ]
    )
    return init_single_subject_wf('01', [])


@pytest.fixture
def acq_b_is_coarser(monkeypatch):
    """Report the acq-b runs as 2.5 mm, the acq-a runs as 1 mm.

    qsiplan refuses to stack runs on different grids within one correction unit, so
    the voxel sizes are changed where QSIPrep reads them rather than on disk.
    """
    from qsiprep.utils import spaces

    def fake_zooms(path):
        return (2.5, 2.5, 2.5) if 'acq-b' in str(path) else (1.0, 1.0, 1.0)

    monkeypatch.setattr(spaces, 'canonical_zooms', fake_zooms)


def test_the_default_refuses_an_output_that_mixes_voxel_sizes(tmp_path, acq_b_is_coarser):
    """Test that acpc:res-nativemin fails when one output merges runs of two voxel sizes."""
    from qsiprep.utils.spaces import OutputSpacesError

    with pytest.raises(OutputSpacesError) as excinfo:
        _subject_wf(tmp_path, '--distortion-group-merge', 'concat')
    message = str(excinfo.value)
    assert 'combines runs with different voxel sizes' in message
    assert 'sub-01_acq-b_dir-AP_dwi.nii.gz: 2.5 mm isotropic' in message
    assert 'acpc:res-nativemin is ambiguous' in message
    assert 'acpc:res-1mm' in message


def test_a_physical_size_builds_an_output_that_mixes_voxel_sizes(tmp_path, acq_b_is_coarser):
    wf = _subject_wf(
        tmp_path,
        '--distortion-group-merge',
        'concat',
        '--output-spaces',
        'acpc:res-2mm',
    )
    assert any('final_merge_wf' in name for name in wf.list_node_names())


def test_unmerged_outputs_resolve_their_own_native_size(tmp_path, acq_b_is_coarser):
    """Test that each output's nativemin comes from its own runs."""
    wf = _subject_wf(tmp_path, dataset=_write_two_output_dataset(tmp_path / 'bids'))
    grids = {name.split('.')[0] for name in wf.list_node_names() if 'output_grid_acpc' in name}
    assert grids == {'output_grid_acpc_1mm_wf', 'output_grid_acpc_2p5mm_wf'}
    for unit, label in (('acq_a', '1mm'), ('acq_b', '2p5mm')):
        edges = _edges_into(wf, f'dwi_sampling_grids_sub_01_{unit}')
        assert (f'output_grid_acpc_{label}_wf', 'outputnode.grid_image', 'in1') in edges


def test_anat_and_acpc_each_get_a_merge_workflow(tmp_path):
    """Test that merging builds one merge workflow per DWI spec, each in its space."""
    wf = _subject_wf(
        tmp_path,
        '--distortion-group-merge',
        'concat',
        '--output-spaces',
        'acpc:res-2mm',
        'anat:res-2mm',
    )
    merge_wfs = sorted(
        {name.split('.')[0] for name in wf.list_node_names() if 'final_merge_wf' in name}
    )
    assert merge_wfs == [
        'sub_01_final_merge_wf_acpc_res2mm',
        'sub_01_final_merge_wf_anat_res2mm',
    ]
    spaces = {
        name: wf.get_node(name).get_node('dwi_derivatives_wf').get_node('ds_dwi_t1').inputs.space
        for name in merge_wfs
    }
    assert spaces == {
        'sub_01_final_merge_wf_acpc_res2mm': 'ACPC',
        'sub_01_final_merge_wf_anat_res2mm': 'anat',
    }
    anat_merge = 'sub_01_final_merge_wf_anat_res2mm'
    edges = _edges_into(wf, anat_merge)
    assert ('anat_preproc_wf', 'outputnode.acpc_transform', 'inputnode.to_acpc_transform') in edges
    assert ('output_grid_anat_2mm', 'out_file', 'inputnode.dwi_sampling_grid') in edges


def test_dwiref_cannot_merge_distortion_groups(tmp_path):
    """Test D11: merged distortion groups have no shared dwiref frame."""
    from qsiprep.utils.spaces import OutputSpacesError

    with pytest.raises(OutputSpacesError) as excinfo:
        _subject_wf(
            tmp_path,
            '--distortion-group-merge',
            'concat',
            '--dwiref-definition',
            'distortion-group',
            '--output-spaces',
            'dwiref',
        )
    message = str(excinfo.value)
    assert 'sub-01' in message
    assert '--dwiref-definition subject' in message
    assert '--distortion-group-merge none' in message


def test_dwiref_merges_at_the_subject_level(tmp_path):
    wf = _subject_wf(
        tmp_path,
        '--distortion-group-merge',
        'concat',
        '--dwiref-definition',
        'subject',
        '--output-spaces',
        'dwiref:res-2mm',
    )
    merge_wfs = {name.split('.')[0] for name in wf.list_node_names() if 'final_merge_wf' in name}
    assert merge_wfs == {'sub_01_final_merge_wf'}
    merge_wf = wf.get_node('sub_01_final_merge_wf')
    assert merge_wf.get_node('dwi_derivatives_wf').get_node('ds_dwi_t1').inputs.space == 'subject'
    edges = _edges_into(wf, 'sub_01_final_merge_wf')
    assert ('dwiref_wf', 'outputnode.dwiref_to_t1_affine', 'inputnode.to_acpc_transform') in edges
    assert ('output_grid_subject_2mm', 'out_file', 'inputnode.dwi_sampling_grid') in edges
    grid_edges = _edges_into(wf, 'output_grid_subject_2mm')
    assert ('dwiref_wf', 'outputnode.dwiref', 'fixed_image') in grid_edges
    # Cropped to the brain, through the template's own coregistration, inverted
    assert ('acpc_mask_to_subject_dwiref', 'output_image', 'fov_mask') in grid_edges
    mask_edges = _edges_into(wf, 'acpc_mask_to_subject_dwiref')
    assert ('anat_preproc_wf', 'outputnode.t1_mask', 'input_image') in mask_edges
    assert ('dwiref_wf', 'outputnode.dwiref', 'reference_image') in mask_edges
    assert any(
        src == 'dwiref_wf' and source[0] == 'outputnode.dwiref_to_t1_affine'
        for src, source, dest in mask_edges
        if dest == 'transforms'
    )
    assert wf.get_node('acpc_mask_to_subject_dwiref').inputs.invert_transform_flags == [True]


def test_dwiref_at_the_distortion_group_level_grids_each_unit(tmp_path):
    """Test that each unit's dwiref grid is cut from that unit's own b=0 reference."""
    wf = _subject_wf(
        tmp_path,
        '--dwiref-definition',
        'distortion-group',
        '--output-spaces',
        'dwiref:res-2mm',
        dataset=_write_two_output_dataset(tmp_path / 'bids'),
    )
    grids = sorted(
        name for name in wf.list_node_names() if name.startswith('output_grid_distortiongroup')
    )
    assert grids == [
        'output_grid_distortiongroup_2mm_sub_01_acq_a',
        'output_grid_distortiongroup_2mm_sub_01_acq_b',
    ]
    for unit in ('acq_a', 'acq_b'):
        edges = _edges_into(wf, f'output_grid_distortiongroup_2mm_sub_01_{unit}')
        assert (f'dwi_preproc_{unit}_wf', 'outputnode.b0_ref_image', 'fixed_image') in edges
        assert (f'dwi_preproc_{unit}_wf', 'outputnode.dwi_mask', 'fov_mask') in edges
        sink = (
            wf.get_node(f'dwi_finalize_{unit}_wf')
            .get_node('dwi_derivatives_wf')
            .get_node('ds_dwi_t1')
        )
        assert sink.inputs.space == 'distortiongroup'


def test_anat_needs_an_anatomical(tmp_path):
    from qsiprep.utils.spaces import OutputSpacesError

    with pytest.raises(OutputSpacesError, match='anat-modality is none'):
        _subject_wf(tmp_path, '--anat-modality', 'none', '--output-spaces', 'anat')


def test_dwiref_writes_the_native_anatomicals(tmp_path):
    wf = _subject_wf(
        tmp_path,
        '--output-spaces',
        'acpc:res-2mm',
        'dwiref:res-2mm',
        dataset=_write_two_output_dataset(tmp_path / 'bids'),
    )
    derivatives_wf = wf.get_node('anat_preproc_wf').get_node('anat_derivatives_wf')
    assert derivatives_wf.get_node('ds_anat_preproc') is not None


@pytest.mark.parametrize(
    ('hmc_method', 'sdc_method'),
    [('eddy', 'topup'), ('tortoise', 'drbuddi'), ('shoreline', 'topup')],
)
def test_dropping_the_coregistration_lands_in_the_reference_frame(
    tmp_path,
    monkeypatch,
    hmc_method,
    sdc_method,
):
    """Test that each backend's b=0 reference is the image its coregistration moves.

    The resampling chain ends with the b=0-to-anatomical coregistration, whose moving
    image is the backend's b=0 template. A dwiref output drops that stage, so its
    chain ends in the frame of that same template; it is the written
    space-distortiongroup reference only if the two are one backend output.
    """
    from qsiprep.tests.test_workflows_native import SRC, _cfg, _rpe_unit, _StubLayout
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')
    cfg = _cfg(hmc_method=hmc_method, sdc_method=sdc_method, layout=_StubLayout())
    cfg.workflow.anat_modality = 't1w'
    cfg.workflow.dwi2anat_dof = 6
    cfg.workflow.hmc_transform = 'Affine'
    cfg.workflow.diffprep_config = None
    cfg.workflow.tortoise_gpu_cpu_ratio = None
    cfg.workflow.gpu = None
    cfg.workflow.shoreline_model = 'tensor' if hmc_method == 'shoreline' else None
    cfg.workflow.impute_slice_threshold = 0
    wf = init_dwi_preproc_wf(
        _rpe_unit(tmp_path),
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=SRC,
    )
    outputnode_edges = _edges_into(wf, 'outputnode')
    sources = {dest: (src, source) for src, source, dest in outputnode_edges}
    (backend,) = {sources[field][0] for field in ('b0_ref_image', 'dwi_files')}
    # The volumes and the transforms that bring them into the b=0 template's frame,
    # and the template itself, are each the backend's own output for that frame.
    assert sources['b0_ref_image'] == (backend, 'outputnode.b0_template')
    assert sources['dwi_files'] == (backend, 'outputnode.dwi_files_to_transform')
    assert sources['hmc_xforms'] == (backend, 'outputnode.to_dwi_ref_affines')
    assert sources['fieldwarps'] == (backend, 'outputnode.to_dwi_ref_warps')
    coreg_edges = _edges_into(wf, 'b0_anat_coreg')
    moving = {(src, source) for src, source, dest in coreg_edges if dest.endswith('ref_b0_brain')}
    assert moving == {(backend, 'outputnode.b0_template')}


def test_subject_dwiref_grids_of_two_sizes_share_one_mask(tmp_path):
    """Test that two subject-level dwiref voxel sizes build, sharing the template's mask.

    The mask does not depend on the voxel size, so it is built once; building it per
    grid gave two nodes one name.
    """
    wf = _subject_wf(
        tmp_path,
        '--distortion-group-merge',
        'concat',
        '--dwiref-definition',
        'subject',
        '--output-spaces',
        'dwiref:res-2mm',
        'dwiref:res-1p5mm',
    )
    for label in ('2mm', '1p5mm'):
        edges = _edges_into(wf, f'output_grid_subject_{label}')
        assert ('acpc_mask_to_subject_dwiref', 'output_image', 'fov_mask') in edges


def test_subject_dwiref_default_with_two_native_sizes(tmp_path, acq_b_is_coarser):
    """Test the default dwiref when unmerged outputs differ in native voxel size."""
    wf = _subject_wf(
        tmp_path,
        '--dwiref-definition',
        'subject',
        '--output-spaces',
        'dwiref',
        dataset=_write_two_output_dataset(tmp_path / 'bids'),
    )
    grids = {name for name in wf.list_node_names() if name.startswith('output_grid_subject')}
    assert grids == {'output_grid_subject_1mm', 'output_grid_subject_2p5mm'}


def test_two_dwiref_specs_of_one_size_share_a_unit_grid(tmp_path):
    """Test that res-native and res-nativemin on isotropic runs share each unit's grid."""
    wf = _subject_wf(
        tmp_path,
        '--dwiref-definition',
        'distortion-group',
        '--output-spaces',
        'dwiref:res-native',
        'dwiref:res-nativemin',
        dataset=_write_two_output_dataset(tmp_path / 'bids'),
    )
    for unit in ('acq_a', 'acq_b'):
        grid = f'output_grid_distortiongroup_1mm_sub_01_{unit}'
        edges = _edges_into(wf, f'dwi_sampling_grids_sub_01_{unit}')
        assert edges == {(grid, 'out_file', 'in1'), (grid, 'out_file', 'in2')}
