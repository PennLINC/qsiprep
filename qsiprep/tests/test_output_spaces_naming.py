"""Guards the derivative names a single-acpc run produces.

QSIRecon consumes these filenames, so a single `acpc` space must keep producing
exactly what QSIPrep produced before --output-spaces existed.
"""

import numpy as np
import pytest

from qsiprep import config
from qsiprep.interfaces.bids import DerivativesDataSink

# node name -> the entities that end up in its filename
EXPECTED_ANAT_ENTITIES = {
    'ds_t1_preproc': {'space': 'ACPC', 'desc': 'preproc'},
    'ds_t1_mask': {'space': 'ACPC', 'desc': 'brain', 'suffix': 'mask'},
    'ds_t1_seg': {'space': 'ACPC', 'suffix': 'dseg'},
    'ds_t1_aseg': {'space': 'ACPC', 'desc': 'aseg', 'suffix': 'dseg'},
    'ds_t1_mni_warp': {'from': 'ACPC', 'to': 'MNI152NLin2009cAsym', 'suffix': 'xfm'},
    'ds_t1_mni_inv_warp': {'from': 'MNI152NLin2009cAsym', 'to': 'ACPC', 'suffix': 'xfm'},
    'ds_t1_template_acpc_transforms': {'from': 'anat', 'to': 'ACPC', 'suffix': 'xfm'},
    'ds_t1_template_acpc_inv_transforms': {'from': 'ACPC', 'to': 'anat', 'suffix': 'xfm'},
    'ds_t1_template_transforms': {'from': 'orig', 'to': 'anat', 'suffix': 'xfm'},
}

# node name -> the entities that end up in its filename, for a single-acpc run
# with the default (non-3dSHORE) hmc_model. Built from what
# init_dwi_derivatives_wf('/data/.../dwi.nii.gz') actually emits today.
EXPECTED_DWI_ENTITIES = {
    'ds_dwi_t1': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwi',
        'extension': '.nii.gz',
    },
    'ds_bvals_t1': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwi',
        'extension': '.bval',
    },
    'ds_bvecs_t1': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwi',
        'extension': '.bvec',
    },
    'ds_t1_b0_ref': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwiref',
        'extension': '.nii.gz',
    },
    'ds_dwi_mask_t1': {
        'space': 'ACPC',
        'desc': 'brain',
        'suffix': 'mask',
        'extension': '.nii.gz',
    },
    'ds_cnr_map_t1': {
        'space': 'ACPC',
        'suffix': 'dwimap',
        'model': 'MAPMRI',
        'extension': '.nii.gz',
    },
    'ds_gradient_table_t1': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwi',
        'extension': '.b',
    },
    'ds_btable_t1': {
        'space': 'ACPC',
        'desc': 'preproc',
        'suffix': 'dwi',
        'extension': '.b_table.txt',
    },
    'ds_tsnr': {'space': 'ACPC', 'suffix': 'dwimap', 'extension': '.nii.gz'},
}


def collect_datasink_entities(workflow, full_names=False):
    """Map each DerivativesDataSink node name to the entities it will write.

    ``full_names`` keys on the fully qualified node name. The default short name
    is convenient to assert against, but two sub-workflows (one per ACPC
    resolution) each contain a node called ``ds_dwi_t1``, so short names silently
    collapse them -- never use the default when counting sinks.
    """
    found = {}
    for name in workflow.list_node_names():
        node = workflow.get_node(name)
        if node is None or not isinstance(node.interface, DerivativesDataSink):
            continue
        entities = {}
        for key in (
            'space',
            'desc',
            'suffix',
            'res',
            'cohort',
            'from',
            'to',
            'mode',
            'model',
            'extension',
            'datatype',
        ):
            value = node.inputs.trait_get().get(key)
            if value is not None and str(value) != '<undefined>':
                entities[key] = value
        found[name if full_names else name.split('.')[-1]] = entities
    return found


@pytest.fixture
def single_acpc_config():
    config.workflow.output_spaces = ['acpc:res-2mm', 'MNI152NLin2009cAsym']
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    return config


def test_single_acpc_anat_derivative_names(single_acpc_config):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    for node_name, expected in EXPECTED_ANAT_ENTITIES.items():
        assert node_name in found, f'{node_name} disappeared from the derivatives workflow'
        for key, value in expected.items():
            assert found[node_name].get(key) == value, (
                f'{node_name}: {key} is {found[node_name].get(key)!r}, expected {value!r}'
            )


def test_acpc_anatomicals_write_no_res_entity(single_acpc_config):
    """Test that ACPC anatomical derivatives carry no res- entity.

    They are resampled onto the AC-PC anchor template's grid, not a DWI grid,
    so the DWI-grid res- rule does not apply to them.
    """
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    acpc_nodes = {n: e for n, e in found.items() if e.get('space') == 'ACPC'}
    assert acpc_nodes, 'expected some ACPC-space derivatives'
    for node_name, entities in acpc_nodes.items():
        assert 'res' not in entities, f'{node_name} gained a res- entity'


@pytest.fixture
def dwi_config():
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    config.workflow.hmc_method = 'tortoise'
    config.workflow.write_local_bvecs = False
    return config


def _single_acpc_resolution():
    from qsiprep.utils.spaces import parse_output_spaces

    return parse_output_spaces(['acpc:res-2mm'])[0].resolution


def test_single_acpc_dwi_derivative_names(dwi_config):
    """Test the entities of QSIRecon's primary input, the preprocessed ACPC DWI.

    Every output on a DWI grid carries its res- entity, even with a single ACPC
    resolution, so QSIRecon can select a resolution by name.
    """
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    wf = init_dwi_derivatives_wf(
        source_file='/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz',
        resolution=_single_acpc_resolution(),
    )
    found = collect_datasink_entities(wf)

    assert found, 'expected some DerivativesDataSink nodes in the dwi derivatives workflow'
    for node_name, expected in EXPECTED_DWI_ENTITIES.items():
        assert node_name in found, f'{node_name} disappeared from the derivatives workflow'
        for key, value in expected.items():
            assert found[node_name].get(key) == value, (
                f'{node_name}: {key} is {found[node_name].get(key)!r}, expected {value!r}'
            )

    acpc_nodes = {n: e for n, e in found.items() if e.get('space') == 'ACPC'}
    assert acpc_nodes, 'expected some ACPC-space dwi derivatives'
    for node_name, entities in acpc_nodes.items():
        assert entities.get('res') == '2mm', f'{node_name} has no res-2mm entity'


def test_single_acpc_finalize_writes_res_on_every_dwi_grid_output(tmp_path):
    """Test that a single ACPC resolution labels every ACPC output of finalize."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'])
    found = collect_datasink_entities(wf, full_names=True)
    acpc_sinks = {n: e for n, e in found.items() if e.get('space') == 'ACPC'}
    assert acpc_sinks
    unlabelled = sorted(n for n, e in acpc_sinks.items() if e.get('res') != '2mm')
    assert not unlabelled, f'ACPC outputs without res-2mm: {unlabelled}'


def test_synb0_dwiref_carries_the_first_acpc_resolution():
    """Test that SynB0's ACPC side output is labelled with its grid's resolution.

    It is resampled onto the first ACPC grid, so it is a DWI-grid output.
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from qsiprep.workflows.dwi.util import add_synb0_outputs

    config.workflow.output_spaces = ['acpc:res-1p5mm', 'acpc:res-2mm']
    synb0_wf = Workflow(name='synb0_wf')
    synb0_wf.add_nodes(
        [
            pe.Node(
                niu.IdentityInterface(
                    fields=[
                        'acquired_synthetic_report',
                        'unet_input_report',
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
    assert workflow.get_node('ds_synb0_dwiref').inputs.res == '1p5mm'


def _write_dwi(path, nvols=6):
    """Write a tiny valid 4D DWI (with .bval/.bvec) so merge nodes can build."""
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def _build_finalize(tmp_path, output_spaces, write_derivatives=True):
    """Build a finalize workflow. Mirrors the fixture style in test_workflows_native."""
    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.workflow.output_spaces = output_spaces
    config.nipype.omp_nthreads = 1
    config.workflow.hmc_method = 'tortoise'
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.b0_threshold = 100
    config.workflow.intramodal_template_iters = 0
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    specs = parse_output_spaces(output_spaces)
    acpc_specs = [s for s in specs if not s.standard]

    src = _write_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_dwi_finalize_wf(
        unit=make_preproc_unit([src]),
        name='dwi_finalize_wf',
        source_file=src,
        output_prefix='sub-01',
        acpc_specs=acpc_specs,
        write_derivatives=write_derivatives,
    )
    return wf, acpc_specs


def test_single_acpc_builds_one_trans_wf(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'])
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    assert prefixes == {'dwi_trans_wf'}


def test_two_acpc_resolutions_build_two_trans_wfs(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    assert prefixes == {'dwi_trans_wf_res2mm', 'dwi_trans_wf_res1p5mm'}


def _grid_selections(wf, dest_field):
    """Resolve every ``dwi_sampling_grids`` connection the way nipype resolves it.

    ``Workflow.connect`` stores everything after the function as the argument
    *tuple* (``source[2:]``), so ``(field, func, [index])`` passes the list, not
    the index -- ``_select_grid`` then raises "list indices must be integers or
    slices, not list" at hash time. Checking node names alone never sees that, so
    run the stored connection through nipype's own evaluator.
    """
    from nipype.pipeline.engine.utils import evaluate_connect_function

    grids = ['grid_res2mm.nii.gz', 'grid_res1p5mm.nii.gz']
    inputnode = wf.get_node('inputnode')
    selected = {}
    for _, node, data in wf._graph.out_edges(inputnode, data=True):
        for source, dest in data['connect']:
            if not isinstance(source, tuple) or dest != dest_field:
                continue
            selected[node.name] = evaluate_connect_function(source[1], source[2], grids)
    return selected


def test_each_trans_wf_is_handed_its_own_output_grid(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    selected = _grid_selections(wf, 'inputnode.output_grid')
    assert selected == {
        'dwi_trans_wf_res2mm': 'grid_res2mm.nii.gz',
        'dwi_trans_wf_res1p5mm': 'grid_res1p5mm.nii.gz',
    }


def test_grid_metadata_reads_its_own_output_grid(tmp_path):
    """Test that each grid_metadata node reads its own output grid.

    The sidecar's Resolution key comes from this node, so a mis-indexed grid
    would report one resolution's voxel size on another's derivatives.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    selected = _grid_selections(wf, 'grid_file')
    assert selected == {
        'grid_metadata_res2mm': 'grid_res2mm.nii.gz',
        'grid_metadata_res1p5mm': 'grid_res1p5mm.nii.gz',
    }


def test_single_acpc_trans_wf_is_handed_the_only_grid(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'])
    assert _grid_selections(wf, 'inputnode.output_grid') == {'dwi_trans_wf': 'grid_res2mm.nii.gz'}


def test_two_acpc_resolutions_write_a_res_entity(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = collect_datasink_entities(wf)
    acpc_sinks = [e for e in found.values() if e.get('space') == 'ACPC']
    assert acpc_sinks
    assert {e.get('res') for e in acpc_sinks} == {'2mm', '1p5mm'}


def test_single_native_acpc_records_its_resolution(tmp_path):
    """res-nativemax is resolved from the DWI headers at run time.

    Its res- entity is symbolic (res-nativemax), so the JSON sidecar is the only
    place it says what nativemax turned out to be. It must be written anyway.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-nativemax'])
    names = wf.list_node_names()
    assert any(n.endswith('grid_metadata') for n in names), names
    edges = {
        (u.name, v.name): d['connect']
        for u, v, d in wf._graph.edges(data=True)
        if u.name == 'grid_metadata'
    }
    assert edges, 'grid_metadata is not wired into the unit sidecar'


def test_single_fixed_acpc_writes_no_resolution_metadata(tmp_path):
    """A res-2mm run says what it is in its own name; no sidecar needed."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'])
    assert not any(n.endswith('grid_metadata') for n in wf.list_node_names())


def test_one_normalization_per_standard_space():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    targets = {e.get('to') for e in found.values() if e.get('from') == 'ACPC'}
    assert 'MNI152NLin2009cAsym' in targets
    assert 'MNI152NLin6Asym' in targets


def test_standard_space_anatomicals_are_written():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym:res-1'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    mni = [e for e in found.values() if e.get('space') == 'MNI152NLin6Asym']
    assert any(e.get('desc') == 'preproc' for e in mni)
    assert any(e.get('suffix') == 'mask' for e in mni)
    assert all(e.get('res') == '1' for e in mni)


def test_bare_standard_space_writes_no_res_entity():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    mni = [e for e in found.values() if e.get('space') == 'MNI152NLin6Asym']
    assert mni
    assert all('res' not in e for e in mni)


def test_cohort_is_a_separate_entity_on_space_but_inline_on_transforms():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-3'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    images = [e for e in found.values() if e.get('space') == 'MNIInfant']
    assert images
    assert all(e.get('cohort') == '3' for e in images)

    transforms = [e for e in found.values() if e.get('from') == 'ACPC' and 'to' in e]
    assert any(e['to'] == 'MNIInfant+3' for e in transforms)


# ---------------------------------------------------------------------------
# Path-level guards.
#
# Everything above checks the entities a node is *handed*. That is blind to the
# failure that matters: two sinks can be handed different entities and still land
# on one filename, because ``data/io_spec.json`` decides which entities survive
# into the path. So render the collected entities through the very same
# build_path machinery DerivativesDataSink uses, and assert on filenames.
# ---------------------------------------------------------------------------

ANAT_BASE = {'subject': '01', 'datatype': 'anat', 'suffix': 'T1w'}
DWI_BASE = {'subject': '01', 'session': '1', 'datatype': 'dwi', 'suffix': 'dwi'}

# What a single-acpc run wrote before --output-spaces existed. QSIRecon reads these.
HISTORICAL_ANAT_ACPC_PATHS = {
    'ds_t1_preproc': 'sub-01/anat/sub-01_space-ACPC_desc-preproc_T1w.nii.gz',
    'ds_t1_mask': 'sub-01/anat/sub-01_space-ACPC_desc-brain_mask.nii.gz',
    'ds_t1_seg': 'sub-01/anat/sub-01_space-ACPC_dseg.nii.gz',
    'ds_t1_aseg': 'sub-01/anat/sub-01_space-ACPC_desc-aseg_dseg.nii.gz',
}

# What a single acpc:res-2mm run writes. QSIRecon selects among ACPC resolutions
# by the res- entity.
SINGLE_ACPC_DWI_PATHS = {
    'ds_dwi_t1': 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwi.nii.gz',
    'ds_bvals_t1': 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwi.bval',
    'ds_bvecs_t1': 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwi.bvec',
    'ds_gradient_table_t1': (
        'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwi.b'
    ),
    'ds_btable_t1': (
        'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwi.b_table.txt'
    ),
    'ds_t1_b0_ref': (
        'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-preproc_dwiref.nii.gz'
    ),
    'ds_dwi_mask_t1': ('sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_desc-brain_mask.nii.gz'),
    'ds_cnr_map_t1': (
        'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_model-MAPMRI_dwimap.nii.gz'
    ),
    'ds_tsnr': 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_res-2mm_dwimap.nii.gz',
}


def render_datasink_paths(found, base, include_figures=False):
    """Render collected sink entities into the relative paths they will be written to.

    Uses ``DerivativesDataSink``'s own path patterns, so a pattern that silently
    drops ``res-`` or ``cohort-`` shows up here as two sinks sharing one path.

    Transform sinks are skipped: they name themselves with ``from-``/``to-`` and
    carry the cohort inline in the ``to-`` label, so they have no ``space-``.

    ``include_figures`` opts in to reportlet sinks, which have no ``space-`` either
    and must name their own ``extension``.
    """
    from bids.layout.writing import build_path

    from qsiprep.interfaces.bids import qsiprep_spec

    patterns = qsiprep_spec['default_path_patterns']
    paths = {}
    for node_name, entities in found.items():
        if 'space' not in entities and not include_figures:
            continue
        merged = {**base, **entities}
        # A sink that does not name an extension inherits it from its in_file at
        # run time; try the ones qsiprep actually writes until a pattern matches.
        extensions = [merged['extension']] if 'extension' in merged else ['.nii.gz', '.tsv']
        path = None
        for extension in extensions:
            path = build_path({**merged, 'extension': extension}, patterns)
            if path is not None:
                break
        assert path is not None, f'{node_name}: no path pattern matched {merged}'
        paths[node_name] = path
    return paths


def assert_no_collisions(paths):
    """Every sink must own its filename: N sinks, N distinct paths."""
    assert len(set(paths.values())) == len(paths), (
        f'{len(paths)} sinks collapsed onto {len(set(paths.values()))} paths: {paths}'
    )


def test_single_acpc_anat_paths_are_the_historical_ones(single_acpc_config):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    paths = render_datasink_paths(collect_datasink_entities(wf, full_names=True), ANAT_BASE)

    assert_no_collisions(paths)
    for node_name, expected in HISTORICAL_ANAT_ACPC_PATHS.items():
        assert paths.get(node_name) == expected, (
            f'{node_name} writes {paths.get(node_name)!r}, expected {expected!r}'
        )
    acpc = {n: p for n, p in paths.items() if '_space-ACPC' in p}
    assert acpc
    assert all('_res-' not in p for p in acpc.values())


def test_single_acpc_dwi_paths(dwi_config):
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    wf = init_dwi_derivatives_wf(
        source_file='/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz',
        resolution=_single_acpc_resolution(),
    )
    paths = render_datasink_paths(collect_datasink_entities(wf, full_names=True), DWI_BASE)

    assert_no_collisions(paths)
    for node_name, expected in SINGLE_ACPC_DWI_PATHS.items():
        assert paths.get(node_name) == expected, (
            f'{node_name} writes {paths.get(node_name)!r}, expected {expected!r}'
        )
    assert all('_res-2mm' in p for p in paths.values())


def test_two_acpc_resolutions_write_distinct_paths(tmp_path):
    """Test that two ACPC resolutions write two files.

    Without a res- entity in the patterns, the second resampling pass simply
    overwrites the first.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    paths = render_datasink_paths(collect_datasink_entities(wf, full_names=True), DWI_BASE)

    assert paths
    assert_no_collisions(paths)

    preproc = sorted(p for p in paths.values() if p.endswith('desc-preproc_dwi.nii.gz'))
    assert len(preproc) == 2, preproc
    assert any('_res-2mm' in p for p in preproc)
    assert any('_res-1p5mm' in p for p in preproc)


def test_two_cohorts_write_distinct_anat_paths():
    """Two cohorts of one template must not clobber each other."""
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    specs = parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-2', 'MNIInfant:cohort-3'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    paths = render_datasink_paths(collect_datasink_entities(wf, full_names=True), ANAT_BASE)

    assert_no_collisions(paths)
    infant = sorted(p for p in paths.values() if '_space-MNIInfant' in p)
    assert len(infant) == 6, infant  # preproc/mask/dseg x 2 cohorts
    assert sum('_cohort-2' in p for p in infant) == 3
    assert sum('_cohort-3' in p for p in infant) == 3


def test_two_standard_resolutions_write_distinct_anat_paths():
    """The TemplateFlow res- label has to reach the filename too."""
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    paths = render_datasink_paths(collect_datasink_entities(wf, full_names=True), ANAT_BASE)

    assert_no_collisions(paths)
    std = sorted(p for p in paths.values() if '_space-MNI152NLin2009cAsym' in p)
    assert len(std) == 6, std
    assert sum('_res-1' in p for p in std) == 3
    assert sum('_res-2' in p for p in std) == 3


def test_resolution_does_not_collide_with_the_unit_sidecar(tmp_path):
    """The resolved voxel size and the unit sidecar must not race for one path.

    niworkflows writes ``<stem before first '.'>.json`` beside any sink carrying
    metadata, and the preproc dwi/bval/bvec/b/b_table sinks all share one stem --
    the stem ds_merged_sidecar writes the full unit sidecar to.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-nativemin'])
    derivatives_wf = wf.get_node('dwi_derivatives_wf')
    assert derivatives_wf is not None
    # meta_dict is fed by a connection, not a static input, so check the edges:
    # asserting on sink.inputs.meta_dict passes even when the wiring is there.
    fed_meta_dict = {
        node.name
        for _, node, data in derivatives_wf._graph.edges(data=True)
        for _, field in data['connect']
        if field == 'meta_dict'
    }
    # These five share the stem ..._desc-preproc_dwi, which is the path
    # ds_merged_sidecar writes the unit sidecar to.
    colliding = fed_meta_dict & {
        'ds_dwi_t1',
        'ds_bvals_t1',
        'ds_bvecs_t1',
        'ds_gradient_table_t1',
        'ds_btable_t1',
    }
    assert not colliding, (
        f'{sorted(colliding)} write a sidecar that collides with ds_merged_sidecar'
    )


def test_resolved_voxel_size_reaches_the_unit_sidecar(tmp_path):
    """res-native* is only reported in the sidecar, so it must still be written."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-nativemin'])
    grid_metadata = wf.get_node('grid_metadata')
    merged_sidecar = wf.get_node('merged_sidecar')
    assert grid_metadata is not None
    edge = wf._graph.get_edge_data(grid_metadata, merged_sidecar)
    assert edge is not None
    assert ('meta_dict', 'extra_data') in edge['connect']


FIGURE_BASE = {'subject': '01', 'datatype': 'figures', 'suffix': 'dwi'}


def collect_figure_entities(wf):
    """Every figures-datatype sink in the workflow, by fully qualified node name."""
    from nipype.interfaces.base import Undefined, isdefined

    found = {}
    for node_name in wf.list_node_names():
        node = wf.get_node(node_name)
        interface = getattr(node, 'interface', None)
        inputs = getattr(interface, 'inputs', None)
        if inputs is None or getattr(inputs, 'datatype', Undefined) != 'figures':
            continue
        entities = {'datatype': 'figures', 'desc': inputs.desc, 'extension': '.svg'}
        if isdefined(getattr(inputs, 'res', Undefined)):
            entities['res'] = inputs.res
        found[node_name] = entities
    return found


def test_two_acpc_resolutions_write_distinct_figure_paths(tmp_path):
    """Two resolutions must not overwrite each other's reportlets."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = collect_figure_entities(wf)
    assert found, 'expected figure sinks in the finalize workflow'
    paths = render_datasink_paths(found, FIGURE_BASE, include_figures=True)
    assert_no_collisions(paths)


def test_merged_groups_build_every_resolution(tmp_path):
    """Test that a merged unit resamples every ACPC resolution.

    Each resolution now feeds its own merge workflow, so none is dropped.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'], write_derivatives=False)
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    assert prefixes == {'dwi_trans_wf_res2mm', 'dwi_trans_wf_res1p5mm'}


def test_finalize_outputnode_carries_every_acpc_resolution(tmp_path):
    """Test that the merge fields carry one slot per ACPC resolution.

    The merge path reads this outputnode, and it used to expose only index 0.
    """
    wf, acpc_specs = _build_finalize(
        tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'], write_derivatives=False
    )
    outputnode = wf.get_node('outputnode')
    for field in ('dwi_t1', 'bvals_t1', 'bvecs_t1', 't1_b0_ref', 'cnr_map_t1'):
        merge = wf.get_node(f'merge_out_{field}')
        assert merge is not None, f'{field} is not merged across resolutions'
        assert merge.interface._numinputs == len(acpc_specs)
        filled = {
            name
            for _, _, data in wf._graph.in_edges(merge, data=True)
            for _, name in data['connect']
        }
        assert filled == {f'in{i}' for i in range(1, len(acpc_specs) + 1)}, (
            f'merge_out_{field} slots not all connected: {sorted(filled)}'
        )
        edge = wf._graph.get_edge_data(merge, outputnode)
        assert edge is not None
        assert ('out', field) in edge['connect']


def test_single_acpc_finalize_outputnode_is_a_one_element_list(tmp_path):
    """Test that a single resolution takes the same shape, so consumers never branch."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'], write_derivatives=False)
    merge = wf.get_node('merge_out_dwi_t1')
    assert merge is not None
    assert merge.interface._numinputs == 1


MERGED_DWI_BASE = {'subject': '01', 'datatype': 'dwi', 'suffix': 'dwi'}


def test_merged_resolutions_write_distinct_paths(tmp_path):
    """Test that two merged outputs in one directory do not share a filename."""
    from qsiprep.tests.test_workflows_native import _merge_wf
    from qsiprep.utils.spaces import parse_output_spaces

    specs = parse_output_spaces(['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = {}
    for index, spec in enumerate(specs):
        wf = _merge_wf(
            tmp_path / f'r{index}',
            resolution=spec.resolution,
            write_shared_outputs=(index == 0),
            name=f'merge_wf_res{spec.resolution.label}',
        )
        collected = {
            **collect_datasink_entities(wf, full_names=True),
            **collect_figure_entities(wf),
        }
        for node_name, entities in collected.items():
            found[f'{spec.resolution.label}.{node_name}'] = entities

    # A sink that stops being emitted must fail here, not vanish quietly.
    short_names = {key.split('.')[-1] for key in found}
    expected = {'ds_series_qc', 'ds_report_qc_warnings', 'ds_merged_sidecar', 'ds_report_b0_mask'}
    assert expected <= short_names, f'sink inventory shrank: {sorted(short_names)}'

    # include_figures also lets through the space-less ds_series_qc.
    paths = render_datasink_paths(found, MERGED_DWI_BASE, include_figures=True)
    assert_no_collisions(paths)


def test_single_merged_resolution_writes_its_res_entity(tmp_path):
    """Test that a one-resolution merged run labels its outputs with res-.

    base.py always hands the merge workflow its resolution, as finalize does for
    unmerged outputs. Only the sampling-scheme figure, which does not depend on
    the grid, goes unlabelled.
    """
    from qsiprep.tests.test_workflows_native import _merge_wf

    wf = _merge_wf(tmp_path, resolution=_single_acpc_resolution())
    found = {
        **collect_datasink_entities(wf, full_names=True),
        **collect_figure_entities(wf),
    }
    paths = render_datasink_paths(found, MERGED_DWI_BASE, include_figures=True)
    grid_paths = {n: p for n, p in paths.items() if not n.endswith('ds_report_gradients')}
    assert grid_paths
    assert all('_res-2mm' in path for path in grid_paths.values()), grid_paths
