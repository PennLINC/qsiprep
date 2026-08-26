"""Guards the derivative names a single-acpc run produces.

QSIRecon consumes these filenames, so a single `acpc` space must keep producing
exactly what QSIPrep produced before --output-spaces existed.
"""

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
}


def collect_datasink_entities(workflow):
    """Map each DerivativesDataSink node name to the entities it will write."""
    found = {}
    for name in workflow.list_node_names():
        node = workflow.get_node(name)
        if node is None or not isinstance(node.interface, DerivativesDataSink):
            continue
        entities = {}
        for key in ('space', 'desc', 'suffix', 'res', 'cohort', 'from', 'to', 'mode'):
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
