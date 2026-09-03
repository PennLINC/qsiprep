"""Settings must survive the trip through the config file.

The CLI parses arguments in one process, writes ``qsiprep.toml``, and builds
the workflow in another process that reads that file back.
"""

import functools
from pathlib import Path

import pytest
from toml import loads

from qsiprep import config


@pytest.fixture
def _restore_config():
    saved = {k: getattr(config.workflow, k) for k in config.workflow._paths}
    yield
    for k, v in saved.items():
        setattr(config.workflow, k, v)


@pytest.mark.usefixtures('_restore_config')
def test_gradient_file_survives_the_config_round_trip(tmp_path):
    """A Path outside ``_paths`` is dumped as its repr, not as a path.

    ``--gradient-file`` is parsed into a Path, so before it was listed in
    ``workflow._paths`` the subprocess that builds the workflow read back the
    literal string ``"PosixPath('/path/to/coeffs.grad')"`` and every
    ``File(exists=True)`` input built from it failed to validate.
    """
    coeff = tmp_path / 'coeffs.grad'
    coeff.write_text('')
    config.workflow.gradient_file = coeff

    dumped = loads(config.dumps())['workflow']['gradient_file']
    assert dumped == str(coeff)

    config.workflow.gradient_file = None
    config.workflow.load({'gradient_file': dumped}, init=False)
    assert config.workflow.gradient_file == coeff
    assert Path(config.workflow.gradient_file).exists()


def test_path_valued_options_are_declared_in_their_section(tmp_path):
    """Every CLI option parsed into a Path is listed in its section's ``_paths``."""
    from qsiprep.cli.parser import _build_parser

    sections = (config.execution, config.workflow, config.nipype, config.seeds)
    undeclared = []
    for action in _build_parser()._actions:
        parse = action.type
        if not isinstance(parse, functools.partial):
            continue
        if parse.func.__name__ not in ('_path_exists', '_is_file'):
            continue
        for section in sections:
            if hasattr(section, action.dest) and action.dest not in section._paths:
                undeclared.append(f'{section.__name__}.{action.dest}')
    assert undeclared == []
