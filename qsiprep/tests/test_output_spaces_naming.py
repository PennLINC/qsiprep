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
    'ds_t1_b0_ref': {'space': 'ACPC', 'suffix': 'dwiref', 'extension': '.nii.gz'},
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


def collect_datasink_entities(workflow):
    """Map each DerivativesDataSink node name to the entities it will write."""
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
        found[name.split('.')[-1]] = entities
    return found


@pytest.fixture
def single_acpc_config():
    config.workflow.output_spaces = ['acpc:res-2mm', 'MNI152NLin2009cAsym']
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    return config


@pytest.mark.xfail(reason='init_anat_derivatives_wf gains output_spaces in Task 13', strict=False)
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


@pytest.mark.xfail(reason='init_anat_derivatives_wf gains output_spaces in Task 13', strict=False)
def test_single_acpc_writes_no_res_entity(single_acpc_config):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    acpc_nodes = {n: e for n, e in found.items() if e.get('space') == 'ACPC'}
    assert acpc_nodes, 'expected some ACPC-space derivatives'
    for node_name, entities in acpc_nodes.items():
        assert 'res' not in entities, f'{node_name} gained a res- entity on a single-acpc run'


@pytest.fixture
def dwi_config():
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    config.workflow.hmc_model = 'tortoise'
    config.workflow.write_local_bvecs = False
    return config


def test_single_acpc_dwi_derivative_names(dwi_config):
    """QSIRecon's primary input is the preprocessed DWI -- this must never rename it.

    Unlike the anatomical tests above, ``init_dwi_derivatives_wf`` still takes only
    ``source_file`` today, and Task 12's added resolution parameter must default to
    keeping this one-argument call form working. So this test is NOT xfail: it must
    pass now and keep passing.
    """
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    wf = init_dwi_derivatives_wf(source_file='/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz')
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
        assert 'res' not in entities, f'{node_name} gained a res- entity on a single-acpc run'


def _write_dwi(path, nvols=6):
    """Write a tiny valid 4D DWI (with .bval/.bvec) so merge nodes can build."""
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def _build_finalize(tmp_path, output_spaces):
    """Build a finalize workflow. Mirrors the fixture style in test_workflows_native."""
    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.workflow.output_spaces = output_spaces
    config.nipype.omp_nthreads = 1
    config.workflow.hmc_model = 'tortoise'
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


def test_two_acpc_resolutions_write_a_res_entity(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = collect_datasink_entities(wf)
    acpc_sinks = [e for e in found.values() if e.get('space') == 'ACPC']
    assert acpc_sinks
    assert {e.get('res') for e in acpc_sinks} == {'2mm', '1p5mm'}
