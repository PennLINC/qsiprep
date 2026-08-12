"""Tests for qsiprep.utils.misc."""

import logging

import numpy as np
import pytest

from qsiprep.cli.parser import _build_parser
from qsiprep.utils.misc import describe_dwidenoise2, parse_denoise_method, safe_unit_vector


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
    """Warn when --dwi-denoise-window cannot affect the selected denoising method."""
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
    action = next(a for a in parser._actions if '--dwi-denoise-window' in a.option_strings)

    assert 'dwidenoise2' in action.help
    assert 'schedule' in action.help
