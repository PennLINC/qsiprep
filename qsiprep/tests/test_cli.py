"""Command-line interface tests."""

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from nipype import config as nipype_config

from qsiprep.cli import run
from qsiprep.cli.parser import parse_args
from qsiprep.cli.workflow import build_boilerplate, build_workflow
from qsiprep.reports.core import generate_reports
from qsiprep.tests.utils import check_generated_files, download_test_data, get_test_data_path
from qsiprep.utils.bids import write_bidsignore, write_derivative_description

nipype_config.enable_debug_mode()
nipype_config.update_config({'execution': {'remove_unnecessary_outputs': False}})

DEFAULT_NUM_CPUS = 4


def _forrest_gump_dataset(data_dir, working_dir, test_name):
    """Copy forrest_gump with ``TotalReadoutTime`` rescaled to its 5 mm DWI grid.

    The DWI was downsampled from 2 mm, but the sidecar kept the 2 mm readout
    time, which applies the GRE field 2.5x too strongly and makes eddy diverge.
    Drop this once the dataset is replaced.
    """
    source = download_test_data('forrest_gump', data_dir)
    dataset_dir = os.path.join(working_dir, f'{test_name}_bids')
    if not os.path.isdir(dataset_dir):
        shutil.copytree(source, dataset_dir)
    sidecar = Path(dataset_dir) / 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.json'
    metadata = json.loads(sidecar.read_text())
    metadata['TotalReadoutTime'] = 0.0188758  # 0.0471895 * 2 mm / 5 mm
    sidecar.write_text(json.dumps(metadata, indent=4))
    return dataset_dir


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda(data_dir, output_dir, working_dir):
    """Run the CUDA test on reverse-PE series data.

    Was in CUDATest.sh.
    XXX: Not called in CircleCI.

    This tests the following features:

    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Input data: DSDTI BIDS data (data/drbuddi_rpe_series).
    """
    TEST_NAME = 'cuda'

    from qsiprep.tests.trxscan_fixtures import fixture_dir

    dataset_dir = str(fixture_dir('rpe', data_dir))
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)
    test_data_path = get_test_data_path()
    eddy_config = os.path.join(test_data_path, 'eddy_config.json')

    parameters = [
        dataset_dir,
        out_dir,
        'participant',
        f'-w={work_dir}',
        '--sloppy',
        '--anat-modality=none',
        '--denoise-method=none',
        '--dwi-biascorrect=none',
        '--sdc-method=drbuddi',
        f'--eddy-config={eddy_config}',
        '--output-resolution=5',
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=False)


def test_parser_accepts_tortoise(tmp_path):
    """Test that the parser accepts ``tortoise`` for --hmc-method.

    ``tortoise`` is the single --hmc-method value for the DIFFPREP backend.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [
            str(bids),
            str(out),
            'participant',
            '--hmc-method',
            'tortoise',
            '--output-resolution',
            '2',
        ]
    )
    assert opts.hmc_method == 'tortoise'


def test_parser_rejects_removed_diffprep_hmc_models(tmp_path):
    """Test that the parser rejects the removed per-mode DIFFPREP --hmc-method values.

    The per-mode values were replaced by "tortoise" + --diffprep-config.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    for removed in ('diffprep_motion', 'diffprep_quadratic', 'diffprep_cubic'):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    str(bids),
                    str(out),
                    'participant',
                    '--hmc-method',
                    removed,
                    '--output-resolution',
                    '2',
                ]
            )


@pytest.mark.parametrize('forced', ['gradwarp1D', 'gradwarp3D'])
def test_parser_accepts_force_gradwarp_and_gradient_file(tmp_path, forced):
    """Test that the parser accepts --force gradwarp{1,3}D and --gradient-file.

    --force gradwarp{1,3}D and --gradient-file land on the namespace under
    those dests.
    """
    from qsiprep.cli.parser import _build_parser
    from qsiprep.tests.gradient_fixtures import write_siemens_grad

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    opts = parser.parse_args(
        [
            str(bids),
            str(out),
            'participant',
            '--force',
            forced,
            '--gradient-file',
            str(coeff),
            '--output-resolution',
            '2',
        ]
    )
    assert opts.force == [forced]
    assert opts.gradient_file == coeff


def test_parser_accepts_ignore_gradwarp(tmp_path):
    """Test that --ignore accepts 'gradwarp'.

    'gradwarp' extends the existing --ignore choices.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [str(bids), str(out), 'participant', '--ignore', 'gradwarp', '--output-resolution', '2']
    )
    assert opts.ignore == ['gradwarp']


def test_parser_accepts_ignore_jacobian(tmp_path):
    """Test that --ignore accepts 'jacobian'.

    'jacobian' is the --ignore off-switch for Jacobian weighting.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [str(bids), str(out), 'participant', '--ignore', 'jacobian', '--output-resolution', '2']
    )
    assert opts.ignore == ['jacobian']


def test_parser_accepts_force_jacobian_and_rejects_the_pair(tmp_path):
    """Test that --force jacobian is accepted, but not together with --ignore jacobian.

    --force jacobian modulates the T2Wreg field; it cannot combine with --ignore jacobian.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    base = [str(bids), str(out), 'participant', '--output-resolution', '2']
    opts = parser.parse_args([*base, '--force', 'jacobian'])
    assert opts.force == ['jacobian']
    with pytest.raises(SystemExit):
        parser.parse_args([*base, '--ignore', 'jacobian', '--force', 'jacobian'])


def test_parser_rejects_removed_jacobian_weighting_flag(tmp_path):
    """Test that the parser rejects the removed --no-jacobian-weighting flag.

    --no-jacobian-weighting was replaced by --ignore jacobian.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(bids),
                str(out),
                'participant',
                '--output-resolution',
                '2',
                '--no-jacobian-weighting',
            ]
        )


def test_repeated_force_accumulates(tmp_path):
    """Test that repeated --force options accumulate.

    action='store' would keep only the last occurrence, so
    "--force gradwarp1D --force gradwarp3D" would reach the validator as a
    single value and silently apply 3D instead of being rejected.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [
            str(bids),
            str(out),
            'participant',
            '--force',
            'gradwarp1D',
            '--force',
            'gradwarp3D',
            '--output-resolution',
            '2',
        ]
    )
    assert opts.force == ['gradwarp1D', 'gradwarp3D']


def test_repeated_force_does_not_leak_between_parses(tmp_path):
    """Test that --force values do not leak between parses.

    action='extend' appends to whatever is on the namespace, so a shared
    mutable default would carry one parse's values into the next.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    base = [str(bids), str(out), 'participant', '--output-resolution', '2']

    assert parser.parse_args([*base, '--force', 'gradwarp1D']).force == ['gradwarp1D']
    assert parser.parse_args([*base, '--force', 'gradwarp3D']).force == ['gradwarp3D']
    assert parser.parse_args(base).force == []


@pytest.mark.parametrize('flag', ['--force', '--ignore'])
def test_parser_rejects_the_old_gradients_value(tmp_path, flag):
    """Test that the parser rejects the old 'gradients' value.

    The pre-rename spelling must fail loudly rather than be silently ignored.
    """
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        parser.parse_args(
            [str(bids), str(out), 'participant', flag, 'gradients', '--output-resolution', '2']
        )


def test_parser_rejects_unknown_force_value(tmp_path):
    """Test that --force accepts only its documented values."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(bids),
                str(out),
                'participant',
                '--force',
                'bogus',
                '--output-resolution',
                '2',
            ]
        )


def test_validate_diffprep_config_missing(tmp_path):
    from qsiprep.utils.misc import validate_diffprep_config

    with pytest.raises(ValueError, match='does not exist'):
        validate_diffprep_config(str(tmp_path / 'nope.json'))


def test_validate_diffprep_config_default_is_valid():
    from qsiprep.data import load as load_data
    from qsiprep.utils.misc import validate_diffprep_config

    validate_diffprep_config(str(load_data('diffprep_params.json')))


def test_validate_diffprep_config_rejects_bad_correction_mode(tmp_path):
    """Test that an invalid correction mode is rejected at parse time.

    A typo must fail at parse time, not deep inside workflow construction.
    """
    import json

    from qsiprep.utils.misc import validate_diffprep_config

    cfg = tmp_path / 'bad_mode.json'
    cfg.write_text(json.dumps({'correction_mode': 'quadratik'}))
    with pytest.raises(ValueError, match='correction_mode'):
        validate_diffprep_config(str(cfg))


def test_validate_diffprep_config_accepts_each_correction_mode(tmp_path):
    import json

    from qsiprep.utils.misc import validate_diffprep_config

    for mode in ('motion', 'quadratic', 'cubic'):
        cfg = tmp_path / f'{mode}.json'
        cfg.write_text(json.dumps({'correction_mode': mode}))
        validate_diffprep_config(str(cfg))


_SHORELINE_DEFAULTS = {'model': '3dshore', 'iters': 2, 'transform': 'Affine'}


def test_load_shoreline_config_defaults_match_the_shipped_file():
    import json

    from qsiprep.data import load as load_data
    from qsiprep.utils.misc import load_shoreline_config

    shipped = load_data('shoreline_params.json')
    assert json.loads(shipped.read_text()) == _SHORELINE_DEFAULTS
    assert load_shoreline_config(None) == _SHORELINE_DEFAULTS
    assert load_shoreline_config(str(shipped)) == _SHORELINE_DEFAULTS


def test_load_shoreline_config_merges_a_partial_file(tmp_path):
    import json

    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'tensor.json'
    cfg.write_text(json.dumps({'model': 'tensor'}))
    assert load_shoreline_config(str(cfg)) == {**_SHORELINE_DEFAULTS, 'model': 'tensor'}


def test_load_shoreline_config_missing(tmp_path):
    from qsiprep.utils.misc import load_shoreline_config

    with pytest.raises(ValueError, match='does not exist'):
        load_shoreline_config(str(tmp_path / 'nope.json'))


@pytest.mark.parametrize(
    ('contents', 'match'),
    [
        ('[1, 2]', 'must contain a JSON object'),
        ('{"model": ', 'not valid JSON'),
        # A typo must fail loudly rather than silently fall back to a default.
        ('{"iter": 3}', 'unknown key'),
        # Values are case-sensitive: the legacy --hmc-model spelling is not accepted.
        ('{"model": "3dSHORE"}', 'model='),
        ('{"transform": "affine"}', 'transform='),
        ('{"iters": 0}', 'iters='),
        ('{"iters": true}', 'iters='),
        ('{"iters": "2"}', 'iters='),
        ('{"iters": 1.5}', 'iters='),
    ],
)
def test_load_shoreline_config_rejects_bad_files(tmp_path, contents, match):
    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'bad.json'
    cfg.write_text(contents)
    with pytest.raises(ValueError, match=match):
        load_shoreline_config(str(cfg))


def test_load_shoreline_config_names_the_file_on_decode_errors(tmp_path):
    """Test that a file that is not UTF-8 produces an error naming the file."""
    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'latin1.json'
    cfg.write_bytes(b'{"model": "\xff"}')
    with pytest.raises(ValueError, match=r'SHORELine configuration file .* is not valid JSON'):
        load_shoreline_config(str(cfg))


def test_load_shoreline_config_names_the_file_on_read_errors(tmp_path):
    """Test that an unreadable path (here a directory) raises ValueError, not a bare OSError."""
    from qsiprep.utils.misc import load_shoreline_config

    with pytest.raises(ValueError, match=r'SHORELine configuration file .* could not be read'):
        load_shoreline_config(str(tmp_path))


def test_load_shoreline_config_model_none_ignores_iters(tmp_path):
    import json

    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'none.json'
    cfg.write_text(json.dumps({'model': 'none', 'iters': 0}))
    assert load_shoreline_config(str(cfg)) == {**_SHORELINE_DEFAULTS, 'model': 'none', 'iters': 0}


def test_load_shoreline_config_legacy_model_override(tmp_path):
    """Test that the deprecated --hmc-model alias overrides the configured model.

    The deprecated --hmc-model alias supplies the model; the file may still set the rest.
    """
    import json

    from qsiprep.utils.misc import load_shoreline_config

    assert load_shoreline_config(None, model='tensor') == {
        **_SHORELINE_DEFAULTS,
        'model': 'tensor',
    }

    iters_only = tmp_path / 'iters.json'
    iters_only.write_text(json.dumps({'iters': 3}))
    assert load_shoreline_config(str(iters_only), model='none') == {
        **_SHORELINE_DEFAULTS,
        'model': 'none',
        'iters': 3,
    }

    with_model = tmp_path / 'with_model.json'
    with_model.write_text(json.dumps({'model': '3dshore'}))
    with pytest.raises(ValueError, match='conflicts'):
        load_shoreline_config(str(with_model), model='tensor')


@pytest.mark.parametrize('forced', ['gradwarp1D', 'gradwarp3D'])
def test_validate_gradient_flags_force_and_ignore_conflict(tmp_path, forced):
    from qsiprep.tests.gradient_fixtures import write_siemens_grad
    from qsiprep.utils.misc import validate_gradient_flags

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    with pytest.raises(ValueError, match='contradictory'):
        validate_gradient_flags(str(coeff), force=[forced], ignore=['gradwarp'])


def test_validate_gradient_flags_rejects_both_forced_dimensionalities(tmp_path):
    """Test that forcing both gradwarp dimensionalities is rejected.

    --force takes a list of values, so argparse cannot make the two
    dimensionalities mutually exclusive; the validator does it instead.
    """
    from qsiprep.tests.gradient_fixtures import write_siemens_grad
    from qsiprep.utils.misc import validate_gradient_flags

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    with pytest.raises(ValueError, match='mutually exclusive'):
        validate_gradient_flags(str(coeff), force=['gradwarp3D', 'gradwarp1D'], ignore=[])


@pytest.mark.parametrize('forced', ['gradwarp1D', 'gradwarp3D'])
def test_validate_gradient_flags_accepts_a_repeated_identical_dimensionality(tmp_path, forced):
    """Test that a repeated identical gradwarp dimensionality is accepted.

    "--force gradwarp1D gradwarp1D" names one dimensionality, not two.
    """
    from qsiprep.tests.gradient_fixtures import write_siemens_grad
    from qsiprep.utils.misc import validate_gradient_flags

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    validate_gradient_flags(str(coeff), force=[forced, forced], ignore=[])


@pytest.mark.parametrize('forced', ['gradwarp1D', 'gradwarp3D'])
def test_validate_gradient_flags_force_requires_gradient_file(forced):
    from qsiprep.utils.misc import validate_gradient_flags

    with pytest.raises(ValueError, match='requires --gradient-file'):
        validate_gradient_flags(None, force=[forced], ignore=[])


def test_validate_gradient_flags_ignores_unrelated_force_values():
    """Test that unrelated --force values are ignored.

    --force sdc-anat-reference has nothing to do with --gradient-file.
    """
    from qsiprep.utils.misc import validate_gradient_flags

    validate_gradient_flags(None, force=['sdc-anat-reference'], ignore=[])


def test_validate_gradient_flags_rejects_unknown_extension(tmp_path):
    """Test that an unknown gradient file extension is rejected.

    TORTOISE only warns and silently disables correction. Silently producing
    uncorrected output is the wrong default for a batch pipeline.
    """
    from qsiprep.utils.misc import validate_gradient_flags

    bogus = tmp_path / 'coeff.txt'
    bogus.write_text('not a coefficient file')
    with pytest.raises(ValueError, match='gradient-file'):
        validate_gradient_flags(str(bogus), force=[], ignore=[])


@pytest.mark.parametrize('extension', ['.grad', '.dat', '.gc', '.nii', '.nii.gz'])
def test_validate_gradient_flags_accepts_every_tortoise_extension(tmp_path, extension):
    from qsiprep.utils.misc import validate_gradient_flags

    path = tmp_path / f'coeff{extension}'
    validate_gradient_flags(str(path), force=[], ignore=[])


def test_validate_gradient_flags_default_is_a_noop():
    """Test that nothing is raised when no gradient flags are given.

    No flags at all: the feature is off and nothing is raised.
    """
    from qsiprep.utils.misc import validate_gradient_flags

    validate_gradient_flags(None, force=[], ignore=[])


def test_validate_gradient_flags_warns_when_ignored_gradient_file_is_unused(tmp_path, caplog):
    from qsiprep.tests.gradient_fixtures import write_siemens_grad
    from qsiprep.utils.misc import validate_gradient_flags

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    with caplog.at_level('WARNING', logger='cli'):
        validate_gradient_flags(str(coeff), force=[], ignore=['gradwarp'])

    assert 'unused' in caplog.text.lower()


# ─── TRXScan truth-scored integration tests ──────────────────────────────────
#
# Each runs qsiprep on a simulated fixture (qsiprep/tests/trxscan_fixtures.py) and scores the
# output against the simulator's ground truth (qsiprep/tests/truth_scoring.py) instead of
# comparing file-name manifests. Thresholds come from measured runs of qsiprep 26.1 on these
# fixtures with --sloppy, with margin; a sign error, a wrong readout time or a failed
# registration fails them by a wide gap, a few percent of accuracy does not.

TRXSCAN_COMMON = [
    '--sloppy',
    '--denoise-method=none',
    '--dwi-biascorrect=none',
    '--output-resolution=3',
]


def _assert_clean_run(out_dir):
    """Fail on what the HTML report would only show: crash files and a non-empty Errors section."""
    crashes = sorted(Path(out_dir).glob('sub-*/log/*/crash-*.txt'))
    assert not crashes, 'qsiprep wrote crash files: ' + ', '.join(c.name for c in crashes[:5])
    reports = sorted(Path(out_dir).glob('sub-*.html'))
    assert reports, f'no subject HTML report under {out_dir}'
    for report in reports:
        assert 'No errors to report!' in report.read_text(), (
            f'{report.name} lists errors in its Errors section'
        )


def _assert_gradient_tables_consistent(out_dir):
    """Assert that every QC stage found its gradient table the most coherent one.

    cs_dmri's ``gradient_table_ratio`` compares fiber-chain lengths across the 24 axis
    permutations and flips of the bvecs and is 1 when the table as used wins. The phantom's
    fiber field is anatomically structured, so any other value means a bvec was misread
    (raw stage) or misrotated or misreoriented by preprocessing (later stages).
    """
    import pandas as pd

    tables = sorted(Path(out_dir).glob('sub-*/**/dwi/*desc-image_qc.tsv'))
    assert tables, f'no desc-image_qc.tsv under {out_dir}'
    for table in tables:
        row = pd.read_csv(table, sep='\t', na_values='n/a').iloc[0]
        columns = [c for c in row.index if c.endswith('gradient_table_ratio')]
        assert columns, f'{table.name} has no gradient_table_ratio column'
        for column in columns:
            value = row[column]
            assert pd.notna(value), f'{column} is n/a in {table.name}: cs_dmri QC did not run'
            assert value <= 1 + 1e-9, (
                f'{column} = {value:.3g} in {table.name}, expected 1: another axis permutation '
                'or flip of the bvecs gives a more coherent fiber field'
            )


def test_gradient_table_check_reads_every_stage(tmp_path):
    dwi = tmp_path / 'sub-01' / 'dwi'
    dwi.mkdir(parents=True)
    table = dwi / 'sub-01_space-ACPC_desc-image_qc.tsv'
    table.write_text('raw_gradient_table_ratio\tt1_gradient_table_ratio\n1.0\t1.0\n')
    _assert_gradient_tables_consistent(tmp_path)
    table.write_text('raw_gradient_table_ratio\tt1_gradient_table_ratio\n1.0\t1.38\n')
    with pytest.raises(AssertionError, match='t1_gradient_table_ratio = 1.38'):
        _assert_gradient_tables_consistent(tmp_path)
    table.write_text('raw_gradient_table_ratio\tt1_gradient_table_ratio\nn/a\t1.0\n')
    with pytest.raises(AssertionError, match='raw_gradient_table_ratio is n/a'):
        _assert_gradient_tables_consistent(tmp_path)


def _trxscan_run(test_name, fixture, extra, data_dir, output_dir, working_dir):
    from qsiprep.tests.truth_scoring import score_run
    from qsiprep.tests.trxscan_fixtures import fixture_dir

    dataset_dir = str(fixture_dir(fixture, data_dir))
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    # eddy with a fixed seed (--initrand), 1000 hyperparameter voxels and 3 iterations: the
    # 100-voxel, 2-iteration config the old smoke runs used picks its voxels at random, and
    # on one CI run that halved every motion estimate (correlations 0.95 -> 0.4).
    eddy_config = os.path.join(get_test_data_path(), 'eddy_config_trxscan.json')
    parameters = [
        dataset_dir,
        out_dir,
        'participant',
        f'-w={work_dir}',
        f'--eddy-config={eddy_config}',
    ]
    parameters += TRXSCAN_COMMON + list(extra)
    _run_and_generate(test_name, parameters, test_main=False, check_outputs=False)
    _assert_clean_run(out_dir)
    _assert_gradient_tables_consistent(out_dir)
    score = score_run(dataset_dir, out_dir)
    # Kept with the derivatives (a CI artifact) and printed, so the numbers are findable
    # whether or not an assertion fires.
    with open(os.path.join(out_dir, 'truth_score.json'), 'w') as f:
        json.dump(score, f, indent=1, default=float)
    print('TRUTH SCORE', test_name, json.dumps(score, default=float))
    return score


def _expect(score, path, lo=None, hi=None, note=None):
    """Assert ``score[path...]`` lies in ``[lo, hi]`` with a message that reads on its own.

    CircleCI's Tests tab shows the assertion message and nothing else, so it names the
    quantity, its value and the bound rather than dumping a dict.
    """
    value = score
    for key in path:
        value = value[key]
    name = '.'.join(str(k) for k in path)
    bounds = ' and '.join(
        s
        for s in (
            f'>= {lo:.3g}' if lo is not None else '',
            f'<= {hi:.3g}' if hi is not None else '',
        )
        if s
    )
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    assert ok, f'{name} = {value:.3g}, expected {bounds}' + (f' ({note})' if note else '')


def _assert_topup_quality(score, coreg_deg=1.0):
    """Assert what a correct TOPUP + eddy + coregistration run looks like on these fixtures."""
    _expect(score, ('sdc', 'slope'), 0.85, 1.15, 'estimated / true PE displacement')
    _expect(score, ('sdc', 'corr'), lo=0.95)
    _expect(
        score,
        ('sdc', 'rms_residual'),
        hi=0.3 * score['sdc']['rms_truth'],
        note=f'30% of the {score["sdc"]["rms_truth"]:.2f} mm rms true displacement',
    )
    _expect(
        score,
        ('b0_corrected_vs_clean',),
        lo=score['b0_uncorrected_vs_clean'] + 0.05,
        note=f'uncorrected b0 scores {score["b0_uncorrected_vs_clean"]:.3f}; must beat it by 0.05',
    )
    _expect(score, ('coreg_error', 'rotation_deg'), hi=coreg_deg)
    _expect(score, ('coreg_error', 'translation_mm'), hi=1.5)


@pytest.mark.integration
@pytest.mark.trxscan_rpe_topup
def test_trxscan_rpe_topup(data_dir, output_dir, working_dir):
    """Score a reverse-PE pair through TOPUP and eddy against the simulator's truth.

    The correction, the coregistration and the absence of spurious motion on a static object.
    """
    score = _trxscan_run(
        'trxscan_rpe_topup', 'rpe', ['--sdc-method=topup'], data_dir, output_dir, working_dir
    )
    _assert_topup_quality(score)
    _expect(score, ('fd_mean_mm',), hi=0.1, note='the object does not move')


@pytest.mark.integration
@pytest.mark.trxscan_rpe_drbuddi
def test_trxscan_rpe_drbuddi(data_dir, output_dir, working_dir):
    """Score the same pair through DRBUDDI, with the T2w.

    The field bounds are loose on purpose: the single coarse stage --sloppy runs recovers 0.43
    of the field (its metrics are MSJac and CC on the blips; DRBUDDI's default stages at the
    same 2.5 mm reach 0.80, but with the T2w they land the corrected b0 4-5 degrees off frame on
    this fixture, see SLOPPY_DRBUDDI). The coregistration is scored tightly: it starts from
    DRBUDDI's undistorted b0 now, not from the T2w as DRBUDDI's rigid had placed it, which was
    5 degrees off here.
    """
    score = _trxscan_run(
        'trxscan_rpe_drbuddi', 'rpe', ['--sdc-method=drbuddi'], data_dir, output_dir, working_dir
    )
    _expect(score, ('sdc', 'corr'), lo=0.75, note='right pattern and sign')
    _expect(score, ('sdc', 'slope'), 0.3, 1.2, 'the sloppy single stage recovers ~0.43')
    _expect(
        score,
        ('b0_corrected_vs_clean',),
        lo=score['b0_uncorrected_vs_clean'] + 0.05,
        note=f'uncorrected b0 scores {score["b0_uncorrected_vs_clean"]:.3f}',
    )
    _expect(score, ('coreg_error', 'rotation_deg'), hi=1.0)
    _expect(score, ('coreg_error', 'translation_mm'), hi=1.5)


@pytest.mark.integration
@pytest.mark.trxscan_epi_topup
def test_trxscan_epi_topup(data_dir, output_dir, working_dir):
    """Score one series plus a reverse-PE epi fieldmap, both under one B0FieldIdentifier."""
    score = _trxscan_run(
        'trxscan_epi_topup', 'epi', ['--sdc-method=topup'], data_dir, output_dir, working_dir
    )
    _assert_topup_quality(score)


@pytest.mark.integration
@pytest.mark.trxscan_phasediff
def test_trxscan_phasediff(data_dir, output_dir, working_dir):
    """Score a GRE phasediff fieldmap with the subject moved between the DWI, T1w and fieldmap.

    Asserts what holds today: the exported field has the right sign and most of the magnitude,
    and b0->T1w coregistration recovers the recorded movement. The image is NOT asserted to
    improve: the fieldmap-to-b0 registration leaves ~2 deg / 2.5 mm of error on this fixture
    and the applied warp recovers under half the field (see the TRXScan report).
    """
    score = _trxscan_run('trxscan_phasediff', 'phasediff', [], data_dir, output_dir, working_dir)
    _expect(score, ('sdc', 'corr'), lo=0.7)
    _expect(score, ('sdc', 'slope'), 0.5, 1.2)
    assert score['coreg_error']['truth'] == 'movement', 'scored against the recorded movement'
    _expect(
        score, ('coreg_error', 'rotation_deg'), hi=1.0, note='vs the recorded 5.4 deg movement'
    )
    _expect(score, ('coreg_error', 'translation_mm'), hi=2.0)


@pytest.mark.integration
@pytest.mark.trxscan_gnl
def test_trxscan_gnl(data_dir, output_dir, working_dir):
    """Score strong gradient nonlinearity with --gradient-file.

    The geometry is restored and the written gradient deviation matches the truth.
    """
    from qsiprep.tests.trxscan_fixtures import fixture_dir

    coeff = next(
        Path(fixture_dir('gnl', data_dir)).glob(
            'derivatives/trxscan/sub-*/dwi/*_desc-gnlcoeff_dwi.grad'
        )
    )
    score = _trxscan_run(
        'trxscan_gnl',
        'gnl',
        ['--sdc-method=topup', f'--gradient-file={coeff}'],
        data_dir,
        output_dir,
        working_dir,
    )
    _assert_topup_quality(score)
    _expect(score, ('gnl_graddev', 'corr'), lo=0.99)
    _expect(score, ('gnl_graddev', 'slope'), 0.95, 1.05)
    _expect(
        score,
        ('gnl_graddev', 'rms_residual'),
        hi=0.1 * score['gnl_graddev']['rms_truth_dev'],
        note='10% of the true gradient deviation',
    )


@pytest.mark.integration
@pytest.mark.trxscan_motion
def test_trxscan_motion(data_dir, output_dir, working_dir):
    """Score eddy's motion parameters against the poses the simulator applied (5 mm / 2.8 deg)."""
    score = _trxscan_run('trxscan_motion', 'motion', [], data_dir, output_dir, working_dir)
    # Same axis, same sign. eddy's estimates vary with its thread count and the sloppy
    # settings: the 5 mm trans_y component scored 0.74 on CircleCI against 0.85 locally.
    for axis in ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_z'):
        _expect(score, ('motion', axis, 'corr'), lo=0.6, note='eddy vs applied, same axis')
    # eddy under --sloppy recovers the shape of the trace (corr 0.83-0.95 on CircleCI) but
    # only a fraction of its amplitude, and the fraction moves between runs: rot_x 0.33 on
    # CircleCI against 0.5 locally. The bound keeps "a fraction", not "most of it".
    for axis in ('trans_y', 'rot_x'):  # the large components
        _expect(score, ('motion', axis, 'amplitude_ratio'), 0.25, 1.3, 'eddy / applied amplitude')


@pytest.mark.integration
@pytest.mark.trxscan_offsets
def test_trxscan_offsets(data_dir, output_dir, working_dir):
    """Score a subject who moved between the T1w and the DWI, and between the AP and PA series.

    The coregistration must recover the recorded movement and TOPUP must still correct the pair.
    """
    score = _trxscan_run(
        'trxscan_offsets', 'offsets', ['--sdc-method=topup'], data_dir, output_dir, working_dir
    )
    assert score['coreg_error']['truth'] == 'movement', 'scored against the recorded movement'
    _expect(score, ('coreg_error', 'rotation_deg'), hi=1.5)
    _expect(score, ('coreg_error', 'translation_mm'), hi=3.0)
    _expect(score, ('sdc', 'corr'), lo=0.95)
    _expect(score, ('sdc', 'slope'), 0.75, 1.15, 'TOPUP with the PA series moved')
    _expect(
        score,
        ('b0_corrected_vs_clean',),
        lo=score['b0_uncorrected_vs_clean'] + 0.05,
        note=f'uncorrected b0 scores {score["b0_uncorrected_vs_clean"]:.3f}',
    )


@pytest.mark.integration
@pytest.mark.trxscan_diffprep
def test_trxscan_diffprep(data_dir, output_dir, working_dir):
    """Score TORTOISE DIFFPREP motion/eddy correction plus DRBUDDI on the reverse-PE pair."""
    score = _trxscan_run(
        'trxscan_diffprep',
        'rpe',
        ['--hmc-method=tortoise', '--sdc-method=drbuddi'],
        data_dir,
        output_dir,
        working_dir,
    )
    _expect(score, ('sdc', 'corr'), lo=0.75, note='right pattern and sign')
    _expect(score, ('sdc', 'slope'), 0.3, 1.2)
    _expect(
        score,
        ('b0_corrected_vs_clean',),
        lo=score['b0_uncorrected_vs_clean'] + 0.05,
        note=f'uncorrected b0 scores {score["b0_uncorrected_vs_clean"]:.3f}',
    )
    _expect(score, ('coreg_error', 'rotation_deg'), hi=1.0)
    _expect(score, ('coreg_error', 'translation_mm'), hi=1.5)


@pytest.mark.integration
@pytest.mark.trxscan_t2wreg
def test_trxscan_t2wreg(data_dir, output_dir, working_dir):
    """Score DIFFPREP's T2Wreg correction: one series, no fieldmap, the subject's T2w.

    With ``--hmc-method tortoise`` and no fieldmap, qsiprep lets DIFFPREP register the EPI to
    the T2w (``--epi T2Wreg``) instead of running SyN. The field it exports has the right
    pattern. What it does to the image is a known defect, asserted as an expected failure so
    the Tests tab shows it and the day it passes is noticed: TORTOISE's rigid placement of
    the T2w lands ~3 degrees off the b0 on this fixture (the same failure as DRBUDDI's
    structural registration), so the corrected b0 scores below the uncorrected one (0.49 vs
    0.69) and the DWI reaches ACPC space 3.5 deg / 4 mm off.
    """
    score = _trxscan_run(
        'trxscan_t2wreg', 't2wreg', ['--hmc-method=tortoise'], data_dir, output_dir, working_dir
    )
    _expect(score, ('sdc', 'corr'), lo=0.7, note='right pattern and sign')
    _expect(score, ('sdc', 'slope'), 0.4, 1.2)
    _expect(score, ('fd_mean_mm',), hi=0.5, note='the object does not move')
    try:
        _expect(score, ('coreg_error', 'rotation_deg'), hi=1.0)
        _expect(score, ('coreg_error', 'translation_mm'), hi=1.5)
        _expect(
            score,
            ('b0_corrected_vs_clean',),
            lo=score['b0_uncorrected_vs_clean'],
            note=f'uncorrected b0 scores {score["b0_uncorrected_vs_clean"]:.3f}',
        )
    except AssertionError as exc:
        pytest.xfail(f'known: T2Wreg places the T2w off the b0 frame on this fixture -- {exc}')
    pytest.fail('T2Wreg now lands the frame: drop the xfail in this test and tighten it')


def _check_arg_specified(argname, arglist):
    for arg in arglist:
        if arg.startswith(argname):
            return True
    return False


def _update_resources(parameters):
    """Set the number of CPUs used for testing.

    We should use all the available CPUs for testing.

    Sometimes a test will set a specific amount of cpus. In that
    case, the number should be kept. Otherwise, try to read the
    env variable (specified in each job in config.yml). If
    this variable doesn't work, just set it to 4.
    """
    # CircleCI exports CIRCLE_CPUS (see .circleci/continue_config.yml); CIRCLECPUS is the old name
    nthreads = int(os.environ.get('CIRCLE_CPUS', os.environ.get('CIRCLECPUS', DEFAULT_NUM_CPUS)))
    if not _check_arg_specified('--nthreads', parameters):
        parameters.append(f'--nthreads={nthreads}')
    if not _check_arg_specified('--omp-nthreads', parameters):
        parameters.append(f'--omp-nthreads={nthreads}')
    return parameters


def _run_and_generate(test_name, parameters, test_main=False, check_outputs=True):
    from qsiprep import config

    # TODO: Add --clean-workdir param to CLI
    parameters.append('--stop-on-first-crash')
    parameters.append('--notrack')
    parameters.append('-vv')

    # Update resource parameters
    parameters = _update_resources(parameters)

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ['qsiprep'] + parameters
        with patch.object(sys, 'argv', argv):
            with pytest.raises(SystemExit) as e:
                run.main()

            assert e.value.code == 0
    else:
        parse_args(parameters)
        config_file = config.execution.work_dir / f'config-{config.execution.run_uuid}.toml'
        config.loggers.cli.warning(f'Saving config file to {config_file}')
        config.to_filename(config_file)

        retval = build_workflow(config_file, retval={})
        qsiprep_wf = retval['workflow']
        build_boilerplate(str(config_file), qsiprep_wf)
        config.loggers.workflow.log(
            15,
            '\n'.join(['config:'] + [f'\t\t{s}' for s in config.dumps().splitlines()]),
        )

        qsiprep_wf.run(**config.nipype.get_plugin())

        boiler_file = config.execution.output_dir / 'logs' / 'CITATION.md'
        if boiler_file.exists():
            if config.environment.exec_env in (
                'apptainer',
                'singularity',
                'docker',
            ):
                boiler_file = Path('<OUTPUT_PATH>') / boiler_file.relative_to(
                    config.execution.output_dir
                )
            config.loggers.workflow.log(
                25,
                'Works derived from this QSIPrep execution should include the '
                f'boilerplate text found in {boiler_file}.',
            )

        write_derivative_description(config.execution.bids_dir, config.execution.output_dir)
        failed_reports = generate_reports(
            processing_list=config.execution.processing_list,
            subject_anatomical_reference=config.workflow.subject_anatomical_reference,
            report_output_level=config.execution.report_output_level,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
        )
        assert not failed_reports
        write_derivative_description(
            config.execution.bids_dir,
            config.execution.output_dir,
            # dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.output_dir)

    if check_outputs:
        output_list_file = os.path.join(get_test_data_path(), f'{test_name}_outputs.txt')
        optional_outputs_list = os.path.join(
            get_test_data_path(), f'{test_name}_optional_outputs.txt'
        )
        if not os.path.isfile(optional_outputs_list):
            optional_outputs_list = None

        check_generated_files(config.execution.output_dir, output_list_file, optional_outputs_list)
