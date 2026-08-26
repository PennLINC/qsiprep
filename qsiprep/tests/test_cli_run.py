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
    ('reference', 'requested', 'expected'),
    [
        ('sessionwise', 'auto', 'session'),
        ('unbiased', 'auto', 'root'),
        ('first-lex', 'auto', 'root'),
        ('sessionwise', 'root', 'root'),
        ('unbiased', 'session', 'session'),
        ('unbiased', 'subject', 'subject'),
    ],
)
def test_report_output_level(tmpdir, reference, requested, expected):
    """Test that --report-output-level=auto is resolved from the anatomical reference."""
    from qsiprep import config
    from qsiprep.cli.parser import parse_args

    full_name = f'report_output_level_{reference}_{requested}'

    bids_dir = tmpdir / full_name
    generate_bids_skeleton(str(bids_dir), long)

    work_dir = tmpdir / f'work_{full_name}'
    config.from_dict({'bids_dir': str(bids_dir), 'work_dir': str(work_dir)}, init=True)

    parse_args(
        [
            str(bids_dir),
            str(tmpdir / f'out_{full_name}'),
            'participant',
            '--participant-label',
            '01',
            '--subject-anatomical-reference',
            reference,
            '--report-output-level',
            requested,
            '--output-resolution',
            '2',
            '--work-dir',
            str(work_dir),
            '--skip-bids-validation',
        ],
    )
    assert config.execution.report_output_level == expected


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


def _dest(option):
    """Turn an option string back into the namespace attribute it sets."""
    return option.lstrip('-').replace('-', '_')


# (deprecated flag, the option it enables, the value that option is set to)
FORWARDED_FLAGS = [
    ('--dwi-only', '--anat-modality', 'none'),
    ('--longitudinal', '--subject-anatomical-reference', 'unbiased'),
    ('--dwi-no-biascorr', '--b1-biascorrect-stage', 'none'),
]


@pytest.mark.parametrize(('flag', 'option', 'value'), FORWARDED_FLAGS)
def test_forwarded_flag_warns_and_enables_its_replacement(
    minimal_args, capsys, flag, option, value
):
    """A deprecated flag warns, names its replacement, and turns it on."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args([*minimal_args, flag])

    warning = capsys.readouterr().err
    assert flag in warning
    assert 'deprecated' in warning
    assert f'{option} {value}' in warning

    assert getattr(opts, _dest(option)) == value
    # The deprecated flag itself must not reach the config object
    assert not hasattr(opts, _dest(flag))


@pytest.mark.parametrize(('flag', 'option', 'value'), FORWARDED_FLAGS)
def test_forwarded_flag_agrees_with_an_explicit_replacement(minimal_args, flag, option, value):
    """Asking for the same thing twice is not a conflict, in either order."""
    from qsiprep.cli.parser import _build_parser

    for extra_args in ([flag, option, value], [option, value, flag]):
        opts = _build_parser().parse_args(minimal_args + extra_args)
        assert getattr(opts, _dest(option)) == value


@pytest.mark.parametrize(('flag', 'option', 'value'), FORWARDED_FLAGS)
def test_forwarded_flag_conflicting_with_its_replacement_is_an_error(
    minimal_args, capsys, flag, option, value
):
    """Silently picking a winner would hide half of what the user asked for."""
    from qsiprep.cli.parser import _build_parser

    # A value the flag does not forward to
    other = {
        'anat_modality': 'T2w',
        'subject_anatomical_reference': 'sessionwise',
        'b1_biascorrect_stage': 'legacy',
    }[_dest(option)]

    for extra_args in ([flag, option, other], [option, other, flag]):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(minimal_args + extra_args)
        assert 'conflicts with' in capsys.readouterr().err


@pytest.mark.parametrize(('flag', 'option', 'value'), FORWARDED_FLAGS)
def test_replacement_option_is_not_deprecated(minimal_args, capsys, flag, option, value):
    """The replacement option is silent and takes effect."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args([*minimal_args, option, value])

    assert capsys.readouterr().err == ''
    assert getattr(opts, _dest(option)) == value


def test_prefer_dedicated_fmaps_warns_and_is_ignored(minimal_args, capsys):
    """The flag is gone from the workflow, so it only warns."""
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args([*minimal_args, '--prefer-dedicated-fmaps'])

    warning = capsys.readouterr().err
    assert '--prefer-dedicated-fmaps' in warning
    assert 'no effect' in warning
    assert 'B0FieldIdentifier' in warning
    assert not hasattr(opts, 'prefer_dedicated_fmaps')


@pytest.mark.parametrize('value', ['iterative', 'first'])
def test_b0_motion_corr_to_warns_but_still_works(minimal_args, capsys, value):
    """Deprecated, but it still selects the SHORELine b=0 alignment strategy."""
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args([*minimal_args, '--b0-motion-corr-to', value])

    warning = capsys.readouterr().err
    assert '--b0-motion-corr-to' in warning
    assert 'iterative' in warning
    assert opts.b0_motion_corr_to == value


def test_b0_motion_corr_to_is_silent_by_default(minimal_args, capsys):
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args(minimal_args)

    assert capsys.readouterr().err == ''
    assert opts.b0_motion_corr_to == 'iterative'


@pytest.mark.parametrize('value', ['Rigid', 'Affine'])
def test_b0_to_t1w_transform_forwards_its_value(minimal_args, capsys, value):
    """The renamed option keeps working, and sets the new one."""
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args([*minimal_args, '--b0-to-t1w-transform', value])

    warning = capsys.readouterr().err
    assert '--b0-to-t1w-transform' in warning
    assert '--b0-to-anat-transform' in warning
    assert opts.b0_to_anat_transform == value
    assert not hasattr(opts, 'b0_to_t1w_transform')


@pytest.mark.parametrize('value', ['Rigid', 'Affine'])
def test_b0_to_anat_transform_is_not_deprecated(minimal_args, capsys, value):
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args([*minimal_args, '--b0-to-anat-transform', value])

    assert capsys.readouterr().err == ''
    assert opts.b0_to_anat_transform == value


def test_b0_to_anat_transform_defaults_to_rigid(minimal_args):
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args(minimal_args)
    assert opts.b0_to_anat_transform == 'Rigid'


def test_b0_transform_options_are_mutually_exclusive(minimal_args, capsys):
    """Both name the same setting, so giving both is ambiguous."""
    from qsiprep.cli.parser import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [*minimal_args, '--b0-to-anat-transform', 'Rigid', '--b0-to-t1w-transform', 'Affine']
        )
    assert 'not allowed with' in capsys.readouterr().err


def test_ignore_accepts_shims_and_fov(minimal_args):
    """The grouping honors both; they must be reachable from the CLI."""
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args([*minimal_args, '--ignore', 'shims', 'fov'])
    assert opts.ignore == ['shims', 'fov']


# --- The method axes (--hmc-method/--sdc-method) and their deprecated aliases ---


def _parse(minimal_args, *extra):
    from qsiprep.cli.parser import _build_parser

    return _build_parser().parse_args([*minimal_args, *extra])


def test_method_axis_defaults(minimal_args, capsys):
    opts = _parse(minimal_args)
    assert capsys.readouterr().err == ''
    assert opts.hmc_method == 'eddy'
    assert opts.shoreline_model is None
    assert opts.sdc_method == 'topup'
    # Legacy vocabulary is back-filled for unconverted readers.
    assert opts.hmc_model == 'eddy'
    assert opts.pepolar_method == 'TOPUP'


def test_hmc_method_shoreline_gets_model_and_drbuddi(minimal_args):
    opts = _parse(minimal_args, '--hmc-method', 'shoreline')
    assert opts.shoreline_model == '3dshore'
    assert opts.sdc_method == 'drbuddi'
    assert opts.hmc_model == '3dSHORE'
    # Report gating still reads pepolar_method; SHORELine keeps the legacy
    # default so today's graphs are unchanged.
    assert opts.pepolar_method == 'TOPUP'


def test_hmc_method_tortoise_auto_resolves_drbuddi(minimal_args):
    """The legacy TOPUP default never produced a working DIFFPREP run."""
    opts = _parse(minimal_args, '--hmc-method', 'tortoise')
    assert opts.sdc_method == 'drbuddi'
    assert opts.hmc_model == 'tortoise'
    assert opts.pepolar_method == 'DRBUDDI'


@pytest.mark.parametrize(
    ('legacy', 'hmc_method', 'shoreline_model', 'hmc_model'),
    [
        ('eddy', 'eddy', None, 'eddy'),
        ('tortoise', 'tortoise', None, 'tortoise'),
        ('3dSHORE', 'shoreline', '3dshore', '3dSHORE'),
        ('tensor', 'shoreline', 'tensor', 'tensor'),
        ('none', 'shoreline', 'none', 'none'),
    ],
)
def test_hmc_model_alias_maps_and_warns(
    minimal_args, capsys, legacy, hmc_method, shoreline_model, hmc_model
):
    opts = _parse(minimal_args, '--hmc-model', legacy)
    warning = capsys.readouterr().err
    assert '--hmc-model' in warning
    assert 'deprecated' in warning
    assert opts.hmc_method == hmc_method
    assert opts.shoreline_model == shoreline_model
    assert opts.hmc_model == hmc_model


def test_hmc_model_conflicts_with_hmc_method(minimal_args, capsys):
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-model', 'eddy', '--hmc-method', 'eddy')
    assert 'not allowed with' in capsys.readouterr().err


def test_pepolar_method_alias_maps_and_warns(minimal_args, capsys):
    opts = _parse(minimal_args, '--pepolar-method', 'TOPUP+DRBUDDI')
    warning = capsys.readouterr().err
    assert '--pepolar-method' in warning
    assert 'deprecated' in warning
    assert opts.sdc_method == 'topup+drbuddi'
    assert opts.pepolar_method == 'TOPUP+DRBUDDI'


def test_pepolar_method_conflicts_with_sdc_method(minimal_args, capsys):
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--pepolar-method', 'TOPUP', '--sdc-method', 'topup')
    assert 'not allowed with' in capsys.readouterr().err


@pytest.mark.parametrize('hmc_method', ['shoreline', 'tortoise'])
@pytest.mark.parametrize('sdc_method', ['topup', 'topup+drbuddi'])
def test_explicit_topup_requires_eddy(minimal_args, capsys, hmc_method, sdc_method):
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-method', hmc_method, '--sdc-method', sdc_method)
    assert 'requires --hmc-method eddy' in capsys.readouterr().err


def test_legacy_topup_with_tortoise_is_an_error(minimal_args, capsys):
    """This pairing used to parse and then hard-fail mid-run."""
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-model', 'tortoise', '--pepolar-method', 'TOPUP')
    assert 'requires --hmc-method eddy' in capsys.readouterr().err


def test_shoreline_model_requires_shoreline_method(minimal_args, capsys):
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--shoreline-model', 'tensor')
    assert 'requires --hmc-method shoreline' in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-model', '3dSHORE', '--shoreline-model', 'tensor')
    assert 'requires --hmc-method shoreline' in capsys.readouterr().err

    opts = _parse(minimal_args, '--hmc-method', 'shoreline', '--shoreline-model', 'tensor')
    assert opts.shoreline_model == 'tensor'
    assert opts.hmc_model == 'tensor'


def test_force_t2wreg_parses(minimal_args):
    assert _parse(minimal_args).force == []
    assert _parse(minimal_args, '--force', 't2wreg').force == ['t2wreg']


def test_shoreline_selection_warns_of_removal(minimal_args, capsys):
    _parse(minimal_args, '--hmc-method', 'shoreline')
    assert 'scheduled for removal' in capsys.readouterr().err
