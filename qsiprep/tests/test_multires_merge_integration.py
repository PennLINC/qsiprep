"""Build the integrated subject workflow for multi-resolution distortion-group merging.

Every other test of this path builds one sub-workflow or reads source text. This one
builds ``init_single_subject_wf`` itself, which is where each finalize output's slot
is handed to its merge workflow -- the wiring most likely to be wrong and least
likely to be noticed.
"""

import numpy as np
import pytest

N_VOLUMES = 20


@pytest.fixture(autouse=True)
def _reset_execution_config():
    """Drop the execution state parse_args leaves on the config singleton."""
    from qsiprep import config

    def reset():
        config.execution._layout = None
        config.execution.bids_database_dir = None
        config.execution.session_label = None

    reset()
    yield
    reset()


def _shelled_gradients():
    """Return one b=0 and a single b=1000 shell, so eddy's shell check would also pass."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(3, N_VOLUMES))
    directions /= np.linalg.norm(directions, axis=0)
    directions[:, 0] = 0
    bvals = ' '.join(['0'] + ['1000'] * (N_VOLUMES - 1))
    bvecs = '\n'.join(' '.join(f'{value:.6f}' for value in row) for row in directions)
    return {
        'sub-01/dwi/sub-01_dwi.bval': bvals + '\n',
        'sub-01/dwi/sub-01_dwi.bvec': bvecs + '\n',
    }


def _write_two_pair_dataset(root):
    """Write two AP/PA pairs that qsiplan corrects separately and merges into one output.

    The pairs differ in readout time, so they are separate blip groups. Under
    TORTOISE, DRBUDDI corrects one blip pair at a time, so each pair becomes its
    own correction unit; the shared MultipartID makes them one output.
    """
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
                            'MultipartID': 'combined',
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


def test_two_resolutions_build_two_merge_workflows(tmp_path):
    """Test that base.py hands each merge workflow its own resolution's slot."""
    from qsiprep import config
    from qsiprep.cli.parser import parse_args
    from qsiprep.workflows.base import init_single_subject_wf

    bids_dir = _write_two_pair_dataset(tmp_path / 'bids')
    work_dir = tmp_path / 'work'
    config.from_dict({'bids_dir': str(bids_dir), 'work_dir': str(work_dir)}, init=True)
    parse_args(
        [
            str(bids_dir),
            str(tmp_path / 'out'),
            'participant',
            '--participant-label',
            '01',
            '--output-spaces',
            'acpc:res-2mm',
            '--distortion-group-merge',
            'concat',
            '--hmc-method',
            'tortoise',
            '--work-dir',
            str(work_dir),
            '--skip-bids-validation',
        ]
    )
    # Set after parsing: until multiple resolutions are allowed with merging at the
    # CLI, the workflows are exercised by setting the config directly.
    config.workflow.output_spaces = ['acpc:res-2mm', 'acpc:res-1p5mm']

    wf = init_single_subject_wf('01', [])

    merge_wfs = sorted(
        {name.split('.')[0] for name in wf.list_node_names() if 'final_merge_wf' in name}
    )
    assert merge_wfs == ['sub_01_final_merge_wf_res1p5mm', 'sub_01_final_merge_wf_res2mm']

    # Slot i of every unit's finalize outputs reaches merge workflow i, and both
    # units feed both merge workflows.
    slots = {}
    for merge_name in merge_wfs:
        merge_wf = wf.get_node(merge_name)
        for source_node, _, data in wf._graph.in_edges(merge_wf, data=True):
            for source, dest in data['connect']:
                if isinstance(source, tuple) and dest.endswith('_image'):
                    slots.setdefault(merge_name, {})[source_node.name] = source[2]
    assert slots == {
        'sub_01_final_merge_wf_res2mm': {
            'dwi_finalize_acq_a_wf': (0,),
            'dwi_finalize_acq_b_wf': (0,),
        },
        'sub_01_final_merge_wf_res1p5mm': {
            'dwi_finalize_acq_a_wf': (1,),
            'dwi_finalize_acq_b_wf': (1,),
        },
    }
