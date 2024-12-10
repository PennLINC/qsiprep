"""Fixtures for the CircleCI tests."""

import os

import pytest


def pytest_addoption(parser):
    """Collect pytest parameters for running tests."""
    parser.addoption('--working_dir', action='store', default='/tmp')
    parser.addoption('--data_dir', action='store')
    parser.addoption('--output_dir', action='store')


# Set up the commandline options as fixtures
@pytest.fixture(scope='session')
def data_dir(request):
    """Grab data directory."""
    return request.config.getoption('--data_dir')


@pytest.fixture(scope='session')
def working_dir(request):
    """Grab working directory."""
    workdir = request.config.getoption('--working_dir')
    os.makedirs(workdir, exist_ok=True)
    return workdir


@pytest.fixture(scope='session')
def output_dir(request):
    """Grab output directory."""
    outdir = request.config.getoption('--output_dir')
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope='session')
def datasets(data_dir):
    """Locate downloaded datasets."""
    dsets = {}
    dsets['forrest_gump'] = os.path.join(data_dir, 'forrest_gump')
    return dsets
