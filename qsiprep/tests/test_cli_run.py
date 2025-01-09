"""Tests for the command line interface"""

import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.cli.parser import parse_args


def gen_layout(bids_dir, database_dir=None):
    """Generate a BIDSLayout object."""
    import re

    from bids.layout import BIDSLayout, BIDSLayoutIndexer

    _indexer = BIDSLayoutIndexer(
        validate=False,
        ignore=(
            'code',
            'stimuli',
            'sourcedata',
            'models',
            'derivatives',
            re.compile(r'^\.'),
            re.compile(r'sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|eeg|ieeg|meg|micr|perf)'),
        ),
    )

    layout_kwargs = {'indexer': _indexer}

    if database_dir:
        layout_kwargs['database_path'] = database_dir

    layout = BIDSLayout(bids_dir, **layout_kwargs)
    return layout


longitudinal = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'dwi': [
                {
                    'direction': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'direction': 'PA',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
            ],
        },
        {
            'session': '02',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'dwi': [
                {
                    'direction': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'direction': 'PA',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
            ],
        },
    ],
}


@pytest.mark.parametrize(
    ('name', 'skeleton'),
    [
        ('longitudinal', longitudinal),
    ],
)
def test_processing_list(tmpdir, name, skeleton):
    from qsiprep import config

    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    parse_args(
        [
            bids_dir,
            str(tmpdir / 'out'),
            'participant',
            '--participant_label',
            '01',
            '--subject-anatomical-reference',
            'sessionwise',
        ],
    )
    assert config.execution.processing_list == ['0\t01\t01', '1\t01\t02']
