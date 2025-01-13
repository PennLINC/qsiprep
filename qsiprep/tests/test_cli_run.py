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


long = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'dir': 'PA',
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
                    'dir': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'dir': 'PA',
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

long2 = {
    '01': [
        {
            'session': 'full',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'dir': 'PA',
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
            'session': 'diffonly',
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '01',
                    'suffix': 'dwi',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'dir': 'PA',
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
    ('name', 'skeleton', 'reference', 'expected'),
    [
        ('longitudinal', long, 'sessionwise', [['01', ['01']], ['01', ['02']]]),
        ('longitudinal', long, 'unbiased', [['01', ['01', '02']]]),
        ('longitudinal', long, 'first', [['01', ['01', '02']]]),
        ('longitudinal2', long2, 'sessionwise', [['01', ['diffonly']], ['01', ['full']]]),
        ('longitudinal2', long2, 'unbiased', [['01', ['diffonly', 'full']]]),
        ('longitudinal2', long2, 'first', [['01', ['diffonly', 'full']]]),
    ],
)
def test_processing_list(tmpdir, name, skeleton, reference, expected):
    from qsiprep import config

    full_name = f'{name}_{reference}'

    bids_dir = str(tmpdir / full_name)
    generate_bids_skeleton(bids_dir, skeleton)
    parse_args(
        [
            bids_dir,
            str(tmpdir / 'out'),
            'participant',
            '--participant-label',
            '01',
            '--subject-anatomical-reference',
            reference,
            '--output-resolution',
            '2',
            '--skip-bids-validation',
        ],
    )
    assert config.execution.processing_list == expected
