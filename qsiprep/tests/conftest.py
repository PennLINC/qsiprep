"""Fixtures for the CircleCI tests."""

import base64
import os

import pytest


def pytest_addoption(parser):
    """Collect pytest parameters for running tests."""
    parser.addoption("--working_dir", action="store", default="/tmp")
    parser.addoption("--data_dir", action="store")
    parser.addoption("--output_dir", action="store")


# Set up the commandline options as fixtures
@pytest.fixture(scope="session")
def data_dir(request):
    """Grab data directory."""
    return request.config.getoption("--data_dir")


@pytest.fixture(scope="session")
def working_dir(request):
    """Grab working directory."""
    workdir = request.config.getoption("--working_dir")
    os.makedirs(workdir, exist_ok=True)
    return workdir


@pytest.fixture(scope="session")
def output_dir(request):
    """Grab output directory."""
    outdir = request.config.getoption("--output_dir")
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def datasets(data_dir):
    """Locate downloaded datasets."""
    return {
        "examples_pasl_multipld": os.path.join(data_dir, "examples_pasl_multipld"),
        "examples_pcasl_multipld": os.path.join(data_dir, "examples_pcasl_multipld"),
        "examples_pcasl_singlepld": os.path.join(data_dir, "examples_pcasl_singlepld"),
        "qtab": os.path.join(data_dir, "qtab"),
        "test_001": os.path.join(data_dir, "test_001"),
        "test_002": os.path.join(data_dir, "test_002"),
        "test_003": os.path.join(data_dir, "test_003"),
    }
