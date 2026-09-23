"""Tests for qsiprep.utils.misc."""

import json
import logging

import numpy as np
import pytest

from qsiprep.cli.parser import _build_parser
from qsiprep.utils.misc import (
    describe_dwidenoise2,
    format_dwidenoise2_schedule,
    load_dwidenoise2_config,
    parse_denoise_method,
    safe_unit_vector,
)


def test_safe_unit_vector_zero_magnitude_substitutes_x_axis():
    result = safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_safe_unit_vector_normalizes_nonzero_vector():
    result = safe_unit_vector(np.array([0.0, 3.0, 0.0]))
    assert np.allclose(result, np.array([0.0, 1.0, 0.0]))
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_safe_unit_vector_no_nan_on_zero():
    result = safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert not np.any(np.isnan(result))


def test_safe_unit_vector_warns_on_zero_magnitude(caplog):
    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert any('zero-magnitude' in record.message for record in caplog.records)


def test_average_bvec_no_nan_with_zero_magnitude_pair():
    from qsiprep.interfaces.dwi_merge import average_bvec

    # Antipodal vectors average to a zero-magnitude vector, which the old
    # normalization turned into NaN. The guard must keep the result finite.
    bvec1 = np.array([1.0, 0.0, 0.0])
    bvec2 = np.array([-1.0, 0.0, 0.0])
    averaged, _ = average_bvec(bvec1, bvec2)
    assert not np.any(np.isnan(averaged))


def test_angle_between_finite_for_zero_vector():
    from qsiprep.interfaces.dwi_merge import angle_between

    angle = angle_between(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(angle)


def test_parse_denoise_method_parameters():
    method, parameters = parse_denoise_method(
        'dwidenoise2;demodulate:hann;decomposition:bdcsvd;'
        'preserve_noise_bias:true;noise_dof:8;schedule:vlarge',
        use_phase=True,
    )

    assert method == 'dwidenoise2'
    assert parameters == {
        'demodulate': 'hann',
        'decomposition': 'bdcsvd',
        'preserve_noise_bias': True,
        'noise_dof': 8,
        'schedule': 'vlarge',
    }


@pytest.mark.parametrize(
    'spec',
    [
        'unknown',
        'patch2self;decomposition:bdcsvd',
        'dwidenoise;decomposition:bdcsvd',
        'dwidenoise2;decomposition',
        'dwidenoise2;unknown:value',
        'dwidenoise2;decomposition:bdcsvd;decomposition:selfadjoint',
        'dwidenoise2;decomposition:invalid',
        'dwidenoise2;preserve_noise_bias:maybe',
        # The kernel and subsampling are set by the schedule, not by command-line options
        'dwidenoise2;extent:1,2',
        'dwidenoise2;shape:sphere',
        'dwidenoise2;radius:2.5',
        'dwidenoise2;subsample:2',
        'dwidenoise2;onepass:true',
        # dwidenoise2 renamed its demodulation and estimator choices
        'dwidenoise2;demodulate:nonlinear',
        'dwidenoise2;estimator:MRM2023',
    ],
)
def test_parse_denoise_method_rejects_invalid_specs(spec):
    with pytest.raises(ValueError, match='.'):
        parse_denoise_method(spec, use_phase=True)


@pytest.mark.parametrize('demodulate', ['linear', 'hann', 'apc'])
def test_parse_denoise_method_rejects_demodulation_without_phase(demodulate):
    """Reject phase demodulation of magnitude-only data, which dwidenoise2 cannot do."""
    spec = f'dwidenoise2;demodulate:{demodulate}'
    with pytest.raises(ValueError, match='magnitude-only data'):
        parse_denoise_method(spec, use_phase=False)

    assert parse_denoise_method(spec, use_phase=True) == (
        'dwidenoise2',
        {'demodulate': demodulate},
    )

    # The CLI validates the specification before it knows whether phase data exist, so an
    # unknown phase state skips the check rather than guessing
    assert parse_denoise_method(spec) == ('dwidenoise2', {'demodulate': demodulate})


def test_denoise_parameters_match_interface():
    """Every allowlisted dwidenoise2 parameter must be a trait on DWIDenoise2InputSpec."""
    from qsiprep.interfaces.mrtrix import DWIDenoise2
    from qsiprep.utils.misc import _DWIDENOISE_PARAMETERS

    trait_names = set(DWIDenoise2.input_spec().trait_names())
    missing = sorted(_DWIDENOISE_PARAMETERS - trait_names)
    assert not missing


def test_denoise_method_cli_parameter(tmp_path):
    spec = 'dwidenoise2;demodulate:apc;decomposition:bdcsvd'
    opts = _build_parser().parse_args(
        [
            str(tmp_path),
            str(tmp_path / 'out'),
            'participant',
            '--output-resolution',
            '2',
            '--denoise-method',
            spec,
        ]
    )

    assert opts.denoise_method == spec


def test_denoise_method_cli_rejects_invalid_parameter(tmp_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                str(tmp_path),
                str(tmp_path / 'out'),
                'participant',
                '--output-resolution',
                '2',
                '--denoise-method',
                'dwidenoise;decomposition:invalid',
            ]
        )


def test_describe_dwidenoise2_covers_defaults():
    """Describe the methods that run by default, not only the requested parameters."""
    description = describe_dwidenoise2({}, complex_data=False)

    # The software, MP-PCA and the noise mapping paper are always applicable
    for citation in (
        '@dwidenoise2software',
        '@dwidenoise1',
        '@dwidenoise2',
        '@cordero2019complex',
    ):
        assert citation in description

    # ...as are the defaults: the mrm2023 estimator, Gaussian aggregation over overlapping
    # patches, and the nonlinear variance-stabilizing transform magnitude data require
    assert '@olesen2023' in description
    assert '@manjon2013' in description
    assert '@foi2011' in description
    assert '@ma2020' in description

    # Nothing that did not run should be cited
    for citation in ('@pizzolato2020', '@patron2024', '@gavish2014', '@zhu2022', '@koay2006'):
        assert citation not in description


def test_describe_dwidenoise2_demodulation_is_complex_only():
    """Only describe phase demodulation when there are phase data to demodulate."""
    parameters = {'demodulate': 'apc'}

    assert '@pizzolato2020' in describe_dwidenoise2(parameters, complex_data=True)
    assert '@pizzolato2020' not in describe_dwidenoise2(parameters, complex_data=False)

    # Complex data are Gaussian, so they need no nonlinear variance-stabilizing transform
    # and carry no noise-floor bias
    complex_description = describe_dwidenoise2(parameters, complex_data=True)
    assert '@foi2011' not in complex_description
    assert 'noise-floor bias' not in complex_description


@pytest.mark.parametrize(
    ('parameters', 'expected', 'unexpected'),
    [
        ({'demodulate': 'hann'}, '@patron2024', '@pizzolato2020'),
        ({'demodulate': 'linear'}, '@cordero2019complex', '@pizzolato2020'),
        ({'estimator': 'tbme2022'}, '@zhu2022', '@olesen2023'),
        ({'estimator': 'med'}, '@gavish2014', '@olesen2023'),
        ({'aggregator': 'exclusive'}, 'solely from the patch', '@manjon2013'),
    ],
)
def test_describe_dwidenoise2_conditional_citations(parameters, expected, unexpected):
    """Follow the conditions dwidenoise2 attaches to each citation in its own help."""
    description = describe_dwidenoise2(parameters, complex_data=True)

    assert expected in description
    assert unexpected not in description


def test_describe_dwidenoise2_filter_follows_fixed_rank():
    """Describe hard truncation when the rank is given rather than estimated."""
    description = describe_dwidenoise2({'fixed_rank': 12}, complex_data=True)

    assert 'hard truncation' in description
    assert 'signal rank was fixed at 12' in description
    # The rank was not estimated, so no estimator applies
    assert '@olesen2023' not in description


@pytest.mark.parametrize(
    ('denoise_method', 'window', 'expected'),
    [
        # dwidenoise2 has no kernel options at all, so a requested window silently does nothing
        ('dwidenoise2', 5, 'not used when --denoise-method=dwidenoise2'),
        ('none', 5, 'not used when --denoise-method=none'),
        # dwidenoise is the only method that takes a window
        ('dwidenoise', 5, None),
        # 'auto' is the default, so an unused value is not a sign of a misunderstanding
        ('dwidenoise2', 'auto', None),
        ('patch2self', 'auto', None),
    ],
)
def test_check_denoise_window_warns_when_unused(caplog, denoise_method, window, expected):
    """Warn when --dwidenoise-window cannot affect the selected denoising method."""
    from qsiprep.cli.parser import check_denoise_window

    with caplog.at_level(logging.WARNING, logger='cli'):
        check_denoise_window(denoise_method, window)

    messages = ' '.join(record.message for record in caplog.records)
    if expected is None:
        assert not messages
    else:
        assert expected in messages


def test_check_denoise_window_errors_for_patch2self(caplog):
    """patch2self never had a window, so an explicit one is reported as an error."""
    from qsiprep.cli.parser import check_denoise_window

    with caplog.at_level(logging.ERROR, logger='cli'):
        check_denoise_window('patch2self', 5)

    assert any(record.levelname == 'ERROR' for record in caplog.records)


def test_denoise_window_help_mentions_dwidenoise2():
    """Say in the help text that dwidenoise2 ignores the window."""
    parser = _build_parser()
    action = next(a for a in parser._actions if '--dwidenoise-window' in a.option_strings)

    assert 'dwidenoise2' in action.help
    assert 'schedule' in action.help


def _dwidenoise2_json(tmp_path, text=None, **settings):
    """Write a --dwidenoise2-config file and return its path."""
    path = tmp_path / 'dwidenoise2.json'
    path.write_text(json.dumps(settings) if text is None else text)
    return str(path)


def test_load_dwidenoise2_config_accepts_empty_file(tmp_path):
    assert load_dwidenoise2_config(_dwidenoise2_json(tmp_path)) == {}


def test_load_dwidenoise2_config_full(tmp_path):
    rows = [
        {'spatial_subsample': 8, 'kernel': 'aspect=2.0', 'update_noise': True},
        {'spatial_subsample': [4, 4, 2], 'kernel': 'rmse=0.02', 'temporal_subsample': 0.5},
        {'spatial_subsample': 2, 'kernel': 'rank', 'update_noise': False, 'partitions': 2},
    ]
    path = _dwidenoise2_json(
        tmp_path,
        demodulate='linear',
        decomposition='selfadjoint',
        demod_axes=[0, 1],
        noise_dof=4,
        preserve_noise_bias=True,
        schedule=rows,
    )

    params = load_dwidenoise2_config(path)

    assert params == {
        'demodulate': 'linear',
        'decomposition': 'selfadjoint',
        'demod_axes': '0,1',
        'noise_dof': 4,
        'preserve_noise_bias': True,
        'schedule': [
            {'spatial_subsample': 8, 'kernel': 'aspect=2.0', 'update_noise': True},
            {'spatial_subsample': (4, 4, 2), 'kernel': 'rmse=0.02', 'temporal_subsample': 0.5},
            {'spatial_subsample': 2, 'kernel': 'rank', 'update_noise': False, 'partitions': 2},
        ],
    }


@pytest.mark.parametrize(
    'settings',
    [
        # A single estimating row
        {'schedule': [{'kernel': 'cuboid=1x', 'update_noise': True}]},
        # A single non-estimating row with a scalar noise level
        {'noise_in': 0.5, 'schedule': [{'spatial_subsample': 1}]},
        {'schedule': [{'update_noise': True, 'temporal_subsample': 1.0}]},
        {'schedule': [{'update_noise': True}, {'kernel': 'rmse=0.999'}]},
        {'fixed_rank': 12, 'schedule': [{'kernel': 'rank_fixed', 'update_noise': True}]},
        # dwidenoise2 supplies its bundled fixedrank schedule
        {'fixed_rank': 12},
        {'fixed_rank': 12, 'aggregator': 'exclusive'},
        # dwidenoise2 runs a single pass at subsample 1 for exclusive aggregation
        {'vst_method': 'none', 'aggregator': 'exclusive'},
        {'aggregator': 'exclusive', 'schedule': [{'update_noise': True, 'spatial_subsample': 1}]},
        {'schedule': [{'update_noise': True, 'max_partition_size': 'none', 'partitions': 3}]},
        {'schedule': [{'update_noise': True, 'max_partition_size': 384}]},
        {'schedule': [{'update_noise': True, 'kernel': 'voxels=1e2'}]},
        {'schedule': [{'update_noise': True, 'kernel': 'cuboid=5,5,3'}]},
        {'schedule': [{'update_noise': True, 'kernel': 'cuboid'}]},
        {'schedule': [{'update_noise': True, 'kernel': 'aspect_ratio=2'}]},
        {'schedule': [{'update_noise': True, 'kernel': 'radius=2.5'}]},
    ],
)
def test_load_dwidenoise2_config_accepts_boundaries(tmp_path, settings):
    load_dwidenoise2_config(_dwidenoise2_json(tmp_path, **settings))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        # Top-level keys and values
        ({'unknown': 1}, 'unknown key'),
        ({'grad_file': 'x'}, 'unknown key'),
        ({'schedule_name': 'vlarge'}, 'unknown key'),
        ({'decomposition': 'invalid'}, 'decomposition'),
        ({'estimator': 'MRM2023'}, 'estimator'),
        ({'preserve_noise_bias': 'true'}, 'preserve_noise_bias'),
        ({'fixed_rank': 0}, 'fixed_rank'),
        ({'fixed_rank': True}, 'fixed_rank'),
        ({'fixed_rank': 2.0}, 'fixed_rank'),
        ({'noise_dof': 0}, 'noise_dof'),
        ({'noise_in': -1}, 'noise_in'),
        ({'noise_in': 'noise.nii.gz'}, 'noise_in'),
        ({'noise_in': True}, 'noise_in'),
        ({'demod_axes': '0,1'}, 'demod_axes'),
        ({'demod_axes': []}, 'demod_axes'),
        ({'demod_axes': [0, -1]}, 'demod_axes'),
        ({'demod_axes': [0, True]}, 'demod_axes'),
        ({'fixed_rank': 3, 'noise_in': 1.0}, 'fixed_rank'),
        # Schedule structure
        ({'schedule': []}, 'non-empty list'),
        ({'schedule': 'vlarge'}, 'non-empty list'),
        ({'schedule': ['aspect=2.0']}, 'row 1'),
        ({'schedule': [{'update_noise': True, 'extent': 3}]}, 'unknown column'),
        # Schedule cell values
        ({'schedule': [{'update_noise': True, 'spatial_subsample': 0}]}, 'spatial_subsample'),
        ({'schedule': [{'update_noise': True, 'spatial_subsample': 2.0}]}, 'spatial_subsample'),
        ({'schedule': [{'update_noise': True, 'spatial_subsample': [2, 2]}]}, 'spatial_subsample'),
        (
            {'schedule': [{'update_noise': True, 'spatial_subsample': [1, True, 1]}]},
            'spatial_subsample',
        ),
        ({'schedule': [{'update_noise': 'true'}]}, 'update_noise'),
        ({'schedule': [{'update_noise': True, 'smooth_noise': 1}]}, 'smooth_noise'),
        ({'schedule': [{'update_noise': True, 'temporal_subsample': 0}]}, 'temporal_subsample'),
        ({'schedule': [{'update_noise': True, 'temporal_subsample': True}]}, 'temporal_subsample'),
        ({'schedule': [{'update_noise': True, 'partitions': 1.0}]}, 'partitions'),
        ({'schedule': [{'update_noise': True, 'max_partition_size': 0}]}, 'max_partition_size'),
        (
            {'schedule': [{'update_noise': True, 'max_partition_size': 'all'}]},
            'max_partition_size',
        ),
        ({'schedule': [{'update_noise': True, 'kernel': 'aspect=-1'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'aspect=nan'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'voxels=inf'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'radius=0'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'cuboid=0'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'cuboid=2,2'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'cuboid=2.5'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'cuboid=x'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'rank=2'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'sphere'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 2}]}, 'kernel'),
        ({'schedule': [{'update_noise': True}, {'kernel': 'rmse=1'}]}, 'kernel'),
        # Rule 1: partitions and max_partition_size together
        (
            {'schedule': [{'update_noise': True, 'partitions': 2, 'max_partition_size': 10}]},
            'partitions',
        ),
        # Rule 2: smoothing a row that does not estimate
        (
            {'schedule': [{'smooth_noise': True, 'update_noise': False}, {}]},
            'smooth_noise',
        ),
        # Rule 3: the first row has no rank density yet
        ({'schedule': [{'kernel': 'rmse=0.02'}, {}]}, 'first'),
        ({'schedule': [{'kernel': 'rank'}, {}]}, 'first'),
        # Rule 4: a non-final row must estimate
        ({'schedule': [{'update_noise': False}, {'update_noise': True}]}, 'update_noise'),
        # Rule 5: the reconstruction row may not be smoothed
        ({'schedule': [{}, {'update_noise': True, 'smooth_noise': True}]}, 'smooth_noise'),
        # Rule 7: the reconstruction row uses all volumes
        ({'schedule': [{}, {'temporal_subsample': 0.5}]}, 'row 2 .*temporal_subsample'),
        # Only ASCII digits, which dwidenoise2 can parse
        ({'schedule': [{'update_noise': True, 'kernel': 'aspect=٢'}]}, 'kernel'),
        ({'schedule': [{'update_noise': True, 'kernel': 'cuboid=²'}]}, 'kernel'),
        # Integers too large for a float
        ({'noise_in': 10**400}, 'noise_in'),
        ({'schedule': [{'update_noise': True, 'temporal_subsample': 10**400}]}, 'temporal'),
        # Rule 8: something must set the noise level
        ({'schedule': [{}]}, 'noise level'),
        ({'schedule': [{'update_noise': False}]}, 'noise level'),
        # Rule 9: fixed_rank and rank_fixed go together
        ({'fixed_rank': 3, 'schedule': [{'update_noise': True}]}, 'rank_fixed'),
        (
            {'fixed_rank': 3, 'schedule': [{'kernel': 'rank_fixed'}, {'kernel': 'rank_fixed'}]},
            'fixed_rank',
        ),
        (
            {'schedule': [{'update_noise': True}, {'kernel': 'rank_fixed'}]},
            'row 2 sets "kernel" "rank_fixed"',
        ),
        # Rule 10: no transform means no iterations and no noise seed
        ({'vst_method': 'none', 'schedule': [{}, {}]}, 'vst_method'),
        ({'vst_method': 'none', 'noise_in': 1.0}, 'vst_method'),
        # Rule 11: exclusive aggregation needs unit subsampling on the last row
        ({'aggregator': 'exclusive'}, 'exclusive'),
        (
            {'aggregator': 'exclusive', 'schedule': [{'update_noise': True}]},
            'exclusive',
        ),
        (
            {
                'aggregator': 'exclusive',
                'schedule': [{'update_noise': True, 'spatial_subsample': [1, 1, 2]}],
            },
            'exclusive',
        ),
    ],
)
def test_load_dwidenoise2_config_rejects(tmp_path, settings, message):
    path = _dwidenoise2_json(tmp_path, **settings)
    with pytest.raises(ValueError, match=message) as excinfo:
        load_dwidenoise2_config(path)
    assert path in str(excinfo.value)


@pytest.mark.parametrize(
    'text',
    [
        '{"decomposition": "bdcsvd", "decomposition": "selfadjoint"}',
        '{"schedule": [{"update_noise": true, "kernel": "rank", "kernel": "cuboid"}]}',
    ],
)
def test_load_dwidenoise2_config_rejects_duplicate_keys(tmp_path, text):
    with pytest.raises(ValueError, match='duplicate key'):
        load_dwidenoise2_config(_dwidenoise2_json(tmp_path, text=text))


@pytest.mark.parametrize(
    'text',
    [
        '{"noise_in": NaN}',
        '{"noise_in": Infinity}',
        '{"schedule": [{"update_noise": true, "temporal_subsample": NaN}]}',
    ],
)
def test_load_dwidenoise2_config_rejects_non_finite_numbers(tmp_path, text):
    with pytest.raises(ValueError, match='.'):
        load_dwidenoise2_config(_dwidenoise2_json(tmp_path, text=text))


@pytest.mark.parametrize('text', ['[]', '{', '"schedule"'])
def test_load_dwidenoise2_config_rejects_malformed_files(tmp_path, text):
    with pytest.raises(ValueError, match='dwidenoise2 configuration file'):
        load_dwidenoise2_config(_dwidenoise2_json(tmp_path, text=text))


def test_load_dwidenoise2_config_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        load_dwidenoise2_config(str(tmp_path / 'missing.json'))


def _schedule_table(text):
    """Split schedule file text into rows of cells, dropping comment lines."""
    return [line.split() for line in text.splitlines() if not line.startswith('#')]


def test_format_dwidenoise2_schedule_single_row():
    text = format_dwidenoise2_schedule([{'kernel': 'cuboid=1x', 'update_noise': True}])

    assert text.startswith('#')
    assert text.endswith('\n')
    assert _schedule_table(text) == [['kernel', 'update_noise'], ['cuboid=1x', 'true']]


def test_format_dwidenoise2_schedule_fills_omitted_cells():
    rows = [
        {'spatial_subsample': (4, 4, 2), 'kernel': 'aspect=2.0'},
        {'temporal_subsample': 0.333333, 'max_partition_size': 384},
        {'spatial_subsample': 2, 'kernel': 'rank'},
    ]

    assert _schedule_table(format_dwidenoise2_schedule(rows)) == [
        [
            'spatial_subsample',
            'kernel',
            'update_noise',
            'temporal_subsample',
            'max_partition_size',
        ],
        ['4,4,2', 'aspect=2.0', 'true', '1.0', 'none'],
        ['2', 'aspect=2.0', 'true', '0.333333', '384'],
        # dwidenoise2 resolves an omitted update_noise to false on the last row
        ['2', 'rank', 'false', '1.0', 'none'],
    ]


def test_format_dwidenoise2_schedule_empty_rows_keep_a_header():
    """An empty header would make dwidenoise2 reject the file."""
    assert _schedule_table(format_dwidenoise2_schedule([{}, {}])) == [
        ['update_noise'],
        ['true'],
        ['false'],
    ]


def test_format_dwidenoise2_schedule_accepts_list_triplets():
    """nipype may hand the interface lists where the loader produced tuples."""
    text = format_dwidenoise2_schedule([{'spatial_subsample': [1, 1, 2], 'update_noise': True}])

    assert _schedule_table(text)[1] == ['1,1,2', 'true']


@pytest.mark.parametrize(
    ('schedule', 'expected'),
    [
        (None, 'following its default schedule'),
        (
            [{'update_noise': True}],
            'following a custom 1-iteration schedule provided in the QSIPrep configuration file',
        ),
        (
            [{}, {}, {'kernel': 'rank'}],
            'following a custom 3-iteration schedule provided in the QSIPrep configuration file',
        ),
    ],
)
def test_describe_dwidenoise2_schedule(schedule, expected):
    parameters = {} if schedule is None else {'schedule': schedule}

    assert expected in describe_dwidenoise2(parameters, complex_data=False)
