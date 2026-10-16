"""Tests for the latest-release and flagged-version checks.

Ported from fMRIPrep's ``fmriprep/cli/tests/test_version.py`` and the banner tests in
``fmriprep/cli/tests/test_parser.py``.
"""

from datetime import UTC, datetime
from os import getenv, geteuid
from pathlib import Path

import pytest
from packaging.version import Version

from qsiprep import config
from qsiprep.cli import version as _version
from qsiprep.cli.parser import _build_parser
from qsiprep.cli.version import DATE_FMT, check_latest, is_flagged, requests


class MockResponse:
    """Mocks the requests module so that PyPI is not actually queried."""

    status_code = 200
    _json = {'releases': {'1.0.0': None, '1.0.1': None, '1.1.0': None, '1.1.1rc1': None}}

    def __init__(self, code=200, json=None):
        """Allow setting different response codes."""
        self.status_code = code
        if json is not None:
            self._json = json

    def json(self):
        """Redefine the response object."""
        return self._json


def test_check_latest1(tmp_path, monkeypatch):
    """Test latest version check."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, 'get', mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / '.cache' / 'qsiprep' / 'latest'
    assert not cachefile.exists()

    # First check actually fetches from pypi
    v = check_latest()
    assert cachefile.exists()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')
    assert cachefile.read_text().split('|') == [
        str(v),
        datetime.now(tz=UTC).strftime(DATE_FMT),
    ]

    # Second check - test the cache file is read
    cachefile.write_text('|'.join(('1.0.0', cachefile.read_text().split('|')[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.0.0')

    # Third check - forced outdating of cache
    cachefile.write_text('2.0.0|20180121')
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')

    # Mock timeouts
    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout

    monkeypatch.setattr(requests, 'get', mock_get)

    cachefile.write_text('|'.join(('1.0.0', cachefile.read_text().split('|')[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.0.0')

    cachefile.write_text('2.0.0|20180121')
    v = check_latest()
    assert v is None

    cachefile.unlink()
    v = check_latest()
    assert v is None


@pytest.mark.parametrize(
    ('result', 'code', 'json'),
    [
        (None, 404, None),
        (None, 200, {'releases': {'1.0.0rc1': None}}),
        (Version('1.1.0'), 200, None),
        (Version('1.0.0'), 200, {'releases': {'1.0.0': None}}),
    ],
)
def test_check_latest2(tmp_path, monkeypatch, result, code, json):
    """Test latest version check with varying server responses."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, 'get', mock_get)

    v = check_latest()
    if result is None:
        assert v is None
    else:
        assert isinstance(v, Version)
        assert v == result


@pytest.mark.parametrize(
    'bad_cache',
    [
        '3laj#r???d|3akajdf#',
        '2.0.0|3akajdf#',
        '|'.join(('2.0.0', datetime.now(tz=UTC).strftime(DATE_FMT), '')),
        '',
    ],
)
def test_check_latest3(tmp_path, monkeypatch, bad_cache):
    """Test latest version check when the cache file is corrupted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, 'get', mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / '.cache' / 'qsiprep' / 'latest'
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    assert not cachefile.exists()

    cachefile.write_text(bad_cache)
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')


@pytest.mark.parametrize(
    ('result', 'version', 'code', 'json'),
    [
        (False, '1.2.1', 200, {'flagged': {'1.0.0': None}}),
        (True, '1.2.1', 200, {'flagged': {'1.2.1': None}}),
        (True, '1.2.1', 200, {'flagged': {'1.2.1': 'FATAL Bug!'}}),
        (False, '1.2.1', 404, {'flagged': {'1.0.0': None}}),
        (False, '1.2.1', 200, {'flagged': []}),
        (False, '1.2.1', 200, {'flagged': {}}),
    ],
)
def test_is_flagged(monkeypatch, result, version, code, json):
    """Test that the flagged-versions check is correct."""
    monkeypatch.setattr(_version, '__version__', version)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, 'get', mock_get)

    val, reason = is_flagged()
    assert val is result

    test_reason = None
    if val:
        test_reason = json.get('flagged', {}).get(version, None)

    if test_reason is not None:
        assert reason == test_reason
    else:
        assert reason is None


def test_versions_json_is_valid():
    """The flagged-versions file that is_flagged downloads must parse and map to reasons."""
    import json

    versions_file = Path(__file__).parents[2] / '.versions.json'
    if not versions_file.is_file():
        pytest.skip('.versions.json is not available outside a source checkout')

    flagged = json.loads(versions_file.read_text())['flagged']
    assert isinstance(flagged, dict)
    for flagged_version, reason in flagged.items():
        Version(flagged_version)
        assert reason is None or isinstance(reason, str)


def test_readonly(tmp_path, monkeypatch):
    """Test behavior when $HOME/.cache/qsiprep/latest can't be written out."""
    home_path = Path('/home/readonly') if getenv('TEST_READONLY_FILESYSTEM') else tmp_path
    monkeypatch.setenv('HOME', str(home_path))
    cachedir = home_path / '.cache'

    if getenv('TEST_READONLY_FILESYSTEM') is None:
        if geteuid() == 0:
            pytest.skip('Cannot mock being unable to create directories as root')
        cachedir.mkdir(mode=0o555, exist_ok=True)

    # Make sure creating the folder will raise the exception.
    with pytest.raises(OSError, match=r'Read-only file system|Permission denied'):
        (cachedir / 'qsiprep').mkdir(parents=True)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, 'get', mock_get)

    # Should not raise
    check_latest()


@pytest.mark.parametrize(('current', 'latest'), [('1.0.0', '1.3.2'), ('1.3.2', '1.3.2')])
def test_get_parser_update(monkeypatch, capsys, current, latest):
    """Make sure the out-of-date banner is shown."""
    expectation = Version(current) < Version(latest)

    monkeypatch.setattr(config.environment, 'version', current)
    monkeypatch.setattr(_version, 'check_latest', lambda: Version(latest))

    _build_parser()
    captured = capsys.readouterr().err

    msg = f"""\
You are using QSIPrep-{current}, and a newer version of QSIPrep is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://qsiprep.readthedocs.io/en/latest/troubleshooting.html#version-warnings"""

    assert (msg in captured) is expectation


@pytest.mark.parametrize('flagged', [(True, None), (True, 'random reason'), (False, None)])
def test_get_parser_flagged(monkeypatch, capsys, flagged):
    """Make sure the flagged-version banner is shown."""
    monkeypatch.setattr(_version, 'is_flagged', lambda: flagged)

    _build_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert f'reason: {flagged[1] or "unknown"}' in captured
