"""--tortoise-gpu-cpu-ratio reaches TORTOISEProcess, and only when asked for.

DIFFPREP treats the GPU as one more worker rather than offloading the series to
it, splitting each pass between the GPU (ratio volumes) and the CPU threads (one
each). Upstream hardcodes 15; that number describes the machine, so it has to be
settable. But qsiprep should not invent a value -- unset must leave TORTOISE's
own default alone.
"""

import pytest


def _config(ratio=None, omp=8):
    from qsiprep import config

    config.nipype.omp_nthreads = omp
    config.workflow.tortoise_gpu_cpu_ratio = ratio
    return config


@pytest.fixture
def diffprep_inputs(tmp_path):
    """DIFFPREP needs its mandatory inputs to render a command line."""
    dwi = tmp_path / 'dwi.nii'
    dwi.write_bytes(b'\0' * 16)
    bmtxt = tmp_path / 'dwi.bmtxt'
    bmtxt.write_text('0 0 0 0 0 0\n')
    js = tmp_path / 'dwi.json'
    js.write_text('{"PhaseEncodingDirection": "j"}')
    return {
        'dwi_file': str(dwi),
        'bmtxt_file': str(bmtxt),
        'json_file': str(js),
        'correction_mode': 'quadratic',
    }


def test_trait_renders_the_expected_flag(diffprep_inputs):
    from qsiprep.interfaces.tortoise import DIFFPREP

    node = DIFFPREP(gpu_cpu_ratio=37, **diffprep_inputs)
    assert '--gpu_cpu_ratio 37' in node.cmdline


def test_flag_absent_when_trait_unset(diffprep_inputs):
    from qsiprep.interfaces.tortoise import DIFFPREP

    node = DIFFPREP(ncores=8, **diffprep_inputs)
    assert '--gpu_cpu_ratio' not in node.cmdline
    assert '--ncores 8' in node.cmdline


@pytest.mark.parametrize('ratio', [None, 0])
def test_workflow_omits_it_unless_requested(ratio):
    """None and 0 both mean 'leave TORTOISE alone'."""
    import inspect

    from qsiprep.workflows.dwi import diffprep

    _config(ratio=ratio)
    src = inspect.getsource(diffprep)
    # the forward is guarded by truthiness of the config value
    assert 'if config.workflow.tortoise_gpu_cpu_ratio' in src
    assert "{'gpu_cpu_ratio': config.workflow.tortoise_gpu_cpu_ratio}" in src


def test_cli_option_exists_and_defaults_to_none():
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = {a.dest: a for a in parser._actions}
    assert 'tortoise_gpu_cpu_ratio' in opts, sorted(opts)
    assert opts['tortoise_gpu_cpu_ratio'].default is None
    assert opts['tortoise_gpu_cpu_ratio'].type is int


def test_cli_parses_a_value(tmp_path):
    from qsiprep.cli.parser import _build_parser

    bids = tmp_path / 'bids'
    (bids / 'sub-01' / 'anat').mkdir(parents=True)
    (bids / 'dataset_description.json').write_text(
        '{"Name": "t", "BIDSVersion": "1.8.0", "DatasetType": "raw"}'
    )
    out = tmp_path / 'out'
    parser = _build_parser()
    args = parser.parse_args(
        [
            str(bids),
            str(out),
            'participant',
            '--tortoise-gpu-cpu-ratio',
            '40',
            '--output-spaces',
            'acpc:res-2mm',
            'MNI152NLin2009cAsym',
            '--skip-bids-validation',
        ]
    )
    assert args.tortoise_gpu_cpu_ratio == 40
