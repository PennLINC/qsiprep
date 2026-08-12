"""Tests for the command line interface"""

import pytest
from niworkflows.utils.testing import generate_bids_skeleton


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
        ('long', long, 'sessionwise', [['01', ['01']], ['01', ['02']]]),
        ('long', long, 'unbiased', [['01', ['01', '02']]]),
        ('long', long, 'first-lex', [['01', ['01', '02']]]),
        ('long2', long2, 'sessionwise', [['01', ['diffonly']], ['01', ['full']]]),
        ('long2', long2, 'unbiased', [['01', ['diffonly', 'full']]]),
        ('long2', long2, 'first-lex', [['01', ['diffonly', 'full']]]),
    ],
)
def _test_processing_list(tmpdir, name, skeleton, reference, expected):
    """Test qsiprep.cli.parser.parse_args.

    Unfortunately, parse_args isn't overwriting all of the Config object
    each time, so bad layouts are lingering across tests.
    I will re-enable this once I figure it out.
    """
    from qsiprep import config
    from qsiprep.cli.parser import parse_args

    full_name = f'{name}_{reference}'

    bids_dir = tmpdir / full_name
    generate_bids_skeleton(str(bids_dir), skeleton)

    config.from_dict({'bids_dir': str(bids_dir)}, init=True)

    parse_args(
        [
            str(bids_dir),
            str(tmpdir / f'out_{full_name}'),
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
    assert config.execution.processing_list == expected, config


@pytest.mark.parametrize(
    ('name', 'skeleton', 'sessions', 'n_anats'),
    [
        ('long', long, ['01', '02'], [1, 1, 2]),
        ('long2', long2, ['diffonly', 'full'], [0, 1, 1]),
    ],
)
def test_collect_data(tmpdir, name, skeleton, sessions, n_anats):
    """Test qsiprep.utils.bids.collect_data."""
    import pprint

    from qsiprep.utils.bids import collect_data

    bids_dir = tmpdir / name

    generate_bids_skeleton(str(bids_dir), skeleton)
    participant_label = '01'

    subj_data = collect_data(
        bids_dir=str(bids_dir),
        participant_label=participant_label,
        session_id=sessions[0],
        filters=None,
        bids_validate=False,
        ignore=[],
    )[0]
    assert len(subj_data['t1w']) == n_anats[0], pprint.pformat(subj_data)

    subj_data = collect_data(
        bids_dir=str(bids_dir),
        participant_label=participant_label,
        session_id=sessions[1],
        filters=None,
        bids_validate=False,
        ignore=[],
    )[0]
    assert len(subj_data['t1w']) == n_anats[1], pprint.pformat(subj_data)

    subj_data = collect_data(
        bids_dir=str(bids_dir),
        participant_label=participant_label,
        session_id=sessions,
        filters=None,
        bids_validate=False,
        ignore=['t2w'],
    )[0]
    assert len(subj_data['t1w']) == n_anats[2], pprint.pformat(subj_data)
    assert len(subj_data['t2w']) == 0, pprint.pformat(subj_data)


@pytest.fixture
def minimal_args(tmp_path):
    """Return the arguments every qsiprep call needs, for parser-level tests."""
    bids_dir = tmp_path / 'bids'
    bids_dir.mkdir()
    return [str(bids_dir), str(tmp_path / 'out'), 'participant', '--output-resolution', '2']


def test_dwi_only_warns_and_sets_anat_modality(minimal_args, capsys):
    """--dwi-only is deprecated, but still selects --anat-modality none."""
    from qsiprep.cli.parser import _build_parser, _resolve_deprecated_dwi_only

    parser = _build_parser()
    opts = parser.parse_args([*minimal_args, '--dwi-only'])

    assert '--dwi-only' in capsys.readouterr().err

    _resolve_deprecated_dwi_only(opts, parser)

    assert opts.anat_modality == 'none'
    # The deprecated attribute must not reach the config object
    assert not hasattr(opts, 'dwi_only')


def test_dwi_only_is_order_independent(minimal_args):
    """--dwi-only and an explicit --anat-modality none agree in either order."""
    from qsiprep.cli.parser import _build_parser, _resolve_deprecated_dwi_only

    for extra_args in (
        ['--dwi-only', '--anat-modality', 'none'],
        ['--anat-modality', 'none', '--dwi-only'],
    ):
        parser = _build_parser()
        opts = parser.parse_args(minimal_args + extra_args)
        _resolve_deprecated_dwi_only(opts, parser)
        assert opts.anat_modality == 'none'


@pytest.mark.parametrize(
    'extra_args',
    [
        ['--dwi-only', '--anat-modality', 'T2w'],
        ['--anat-modality', 'T2w', '--dwi-only'],
    ],
)
def test_dwi_only_conflicts_with_anat_modality(minimal_args, extra_args):
    """Combining --dwi-only with a contradictory --anat-modality is an error."""
    from qsiprep.cli.parser import _build_parser, _resolve_deprecated_dwi_only

    parser = _build_parser()
    opts = parser.parse_args(minimal_args + extra_args)

    with pytest.raises(SystemExit):
        _resolve_deprecated_dwi_only(opts, parser)


def test_anat_modality_none_is_not_deprecated(minimal_args, capsys):
    """The replacement option is silent and untouched by the deprecation shim."""
    from qsiprep.cli.parser import _build_parser, _resolve_deprecated_dwi_only

    parser = _build_parser()
    opts = parser.parse_args([*minimal_args, '--anat-modality', 'none'])

    assert capsys.readouterr().err == ''

    _resolve_deprecated_dwi_only(opts, parser)

    assert opts.anat_modality == 'none'
