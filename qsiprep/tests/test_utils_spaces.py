"""Tests for the --output-spaces grammar."""

import pytest

from qsiprep.utils.spaces import (
    OutputSpacesError,
    parse_output_spaces,
    parse_space_token,
)


def test_acpc_isotropic_mm():
    (spec,) = parse_space_token('acpc:res-2mm')
    assert spec.space == 'acpc'
    assert spec.standard is False
    assert spec.cohort is None
    assert spec.resolution.kind == 'mm'
    assert spec.resolution.zooms == (2.0, 2.0, 2.0)
    assert spec.resolution.label == '2mm'
    assert str(spec) == 'acpc:res-2mm'


def test_acpc_decimal_uses_p():
    (spec,) = parse_space_token('acpc:res-1p5mm')
    assert spec.resolution.zooms == (1.5, 1.5, 1.5)
    assert spec.resolution.label == '1p5mm'
    assert str(spec) == 'acpc:res-1p5mm'


@pytest.mark.parametrize('strategy', ['min', 'max'])
def test_acpc_native(strategy):
    (spec,) = parse_space_token(f'acpc:res-native{strategy}')
    assert spec.resolution.kind == 'native'
    assert spec.resolution.strategy == strategy
    assert spec.resolution.zooms is None
    assert spec.needs_native_resolution is True


def test_acpc_bare_number_is_rejected():
    with pytest.raises(OutputSpacesError, match='acpc:res-2mm'):
        parse_space_token('acpc:res-2')


def test_acpc_anisotropic_is_rejected():
    with pytest.raises(OutputSpacesError, match='isotropic'):
        parse_space_token('acpc:res-2x2x3mm')


def test_acpc_requires_a_resolution():
    with pytest.raises(OutputSpacesError, match='res-'):
        parse_space_token('acpc')


def test_acpc_rejects_cohort():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('acpc:res-2mm:cohort-1')


def test_standard_space_bare():
    (spec,) = parse_space_token('MNI152NLin2009cAsym')
    assert spec.standard is True
    assert spec.resolution is None
    assert str(spec) == 'MNI152NLin2009cAsym'


def test_standard_space_templateflow_label():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-2')
    assert spec.resolution.kind == 'label'
    assert spec.resolution.label == '2'


def test_standard_space_custom_mm():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-1p5mm')
    assert spec.resolution.kind == 'mm'
    assert spec.resolution.zooms == (1.5, 1.5, 1.5)


def test_standard_space_anisotropic_allowed():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-6x6x3mm')
    assert spec.resolution.zooms == (6.0, 6.0, 3.0)


def test_repeated_res_expands():
    specs = parse_space_token('MNI152NLin2009cAsym:res-1:res-3mm')
    assert [s.resolution.label for s in specs] == ['1', '3mm']


def test_unknown_resolution_label_is_rejected():
    with pytest.raises(OutputSpacesError, match='res-9'):
        parse_space_token('MNI152NLin2009cAsym:res-9')


def test_native_rejected_on_standard_space():
    with pytest.raises(OutputSpacesError, match='native'):
        parse_space_token('MNI152NLin2009cAsym:res-nativemax')


def test_unknown_space_is_rejected():
    with pytest.raises(OutputSpacesError, match='NotATemplate'):
        parse_space_token('NotATemplate')


def test_unknown_key_is_rejected():
    with pytest.raises(OutputSpacesError, match='den'):
        parse_space_token('MNI152NLin2009cAsym:den-32k')


def test_cohort_template_requires_a_cohort():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('MNIInfant')


def test_cohort_template_accepts_a_label():
    (spec,) = parse_space_token('MNIInfant:cohort-3')
    assert spec.cohort == '3'
    assert spec.fullname == 'MNIInfant+3'
    assert spec.needs_cohort_resolution is False


def test_cohort_auto_is_deferred():
    (spec,) = parse_space_token('MNIInfant:cohort-auto')
    assert spec.cohort == 'auto'
    assert spec.needs_cohort_resolution is True
    assert spec.fullname == 'MNIInfant'
    assert str(spec) == 'MNIInfant:cohort-auto'


def test_cohort_auto_rejected_without_an_age_table():
    with pytest.raises(OutputSpacesError, match='cohort-1'):
        parse_space_token('MNIPediatricAsym:cohort-auto')


def test_invalid_cohort_is_rejected():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('MNIInfant:cohort-99')


def test_with_cohort_replaces_auto():
    (spec,) = parse_space_token('MNIInfant:cohort-auto')
    resolved = spec.with_cohort('3')
    assert resolved.cohort == '3'
    assert resolved.fullname == 'MNIInfant+3'
    assert spec.cohort == 'auto'  # original untouched


def test_parse_output_spaces_requires_acpc():
    with pytest.raises(OutputSpacesError, match='acpc'):
        parse_output_spaces(['MNI152NLin2009cAsym'])


def test_parse_output_spaces_allows_multiple_acpc():
    specs = parse_output_spaces(['acpc:res-2mm', 'acpc:res-1p5mm'])
    assert [s.resolution.label for s in specs] == ['2mm', '1p5mm']


def test_parse_output_spaces_deduplicates_preserving_order():
    specs = parse_output_spaces(
        ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'acpc:res-2mm']
    )
    assert [str(s) for s in specs] == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def _min_args(tmp_path, *extra):
    bids = tmp_path / 'bids'
    (bids / 'sub-01' / 'anat').mkdir(parents=True)
    (bids / 'dataset_description.json').write_text(
        '{"Name": "t", "BIDSVersion": "1.8.0", "DatasetType": "raw"}'
    )
    return [str(bids), str(tmp_path / 'out'), 'participant', *extra]


def test_parser_accepts_output_spaces(tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args(
        _min_args(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNI152NLin2009cAsym')
    )
    assert opts.output_spaces == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def test_parser_expands_repeated_res(tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args(
        _min_args(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2')
    )
    assert opts.output_spaces == [
        'acpc:res-2mm',
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-2',
    ]


def test_parser_rejects_a_bad_token(tmp_path, capsys):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_min_args(tmp_path, '--output-spaces', 'acpc:res-2'))
    assert 'acpc:res-2mm' in capsys.readouterr().err


def test_parser_accumulates_repeated_flags(tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args(
        _min_args(
            tmp_path,
            '--output-spaces', 'acpc:res-2mm',
            '--output-spaces', 'MNI152NLin2009cAsym',
        )
    )
    assert opts.output_spaces == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def _parse(tmp_path, *extra):
    from qsiprep.cli.parser import _build_parser

    return _build_parser().parse_args(_min_args(tmp_path, *extra))


def test_output_resolution_forwards(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '2')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def test_output_resolution_decimal_forwards(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '1.5')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces[0] == 'acpc:res-1p5mm'


def test_output_resolution_forwards_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '2', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-auto']


def test_skip_normalization_drops_standard_spaces(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(
        tmp_path, '--output-resolution', '2', '--skip-anat-based-spatial-normalization'
    )
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm']


def test_old_and_new_together_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(
        tmp_path, '--output-resolution', '2', '--output-spaces', 'acpc:res-2mm'
    )
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)


def test_infant_adds_the_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'acpc:res-2mm', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-auto']


def test_infant_does_not_duplicate_an_explicit_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNIInfant:cohort-3', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-3']


def test_missing_acpc_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'MNI152NLin2009cAsym')
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)


def test_nothing_given_at_all_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path)
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)
