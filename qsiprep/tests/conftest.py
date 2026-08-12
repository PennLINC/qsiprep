"""Fixtures for the CircleCI tests."""

import os
from pathlib import Path

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
    dsets['nibs'] = os.path.join(data_dir, 'nibs')
    return dsets


@pytest.fixture(scope='session')
def nibs_dwi(data_dir):
    """Locate the nibs DWI series used to test the denoising workflow.

    The series is small and has both magnitude and phase data, so it can
    exercise the complex-valued denoising paths without a long runtime.

    Tests using this fixture are skipped when the dataset is unavailable, which keeps
    them runnable outside of the container.
    """
    if not data_dir:
        pytest.skip('--data_dir was not provided')

    dwi_dir = Path(data_dir) / 'nibs' / 'sub-22449' / 'ses-01' / 'dwi'
    stem = 'sub-22449_ses-01_dir-AP'
    files = {
        'dwi_file': dwi_dir / f'{stem}_part-mag_dwi.nii.gz',
        'phase_file': dwi_dir / f'{stem}_part-phase_dwi.nii.gz',
        'bval_file': dwi_dir / f'{stem}_dwi.bval',
        'bvec_file': dwi_dir / f'{stem}_dwi.bvec',
        'json_file': dwi_dir / f'{stem}_part-mag_dwi.json',
    }
    missing = sorted(str(f) for f in files.values() if not f.is_file())
    if missing:
        pytest.skip(f'nibs dataset is unavailable; missing {missing}')

    return {key: str(value) for key, value in files.items()}
