# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" py.test configuration file """
import os
from tempfile import mkdtemp
from datetime import datetime as dt
import pytest
from qsiprep.niworkflows.data.getters import get_template, get_ds003_downsampled

filepath = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.realpath(os.path.join(filepath, 'data'))


def _run_interface_mock(objekt, runtime):
    runtime.returncode = 0
    runtime.endTime = dt.isoformat(dt.utcnow())

    objekt._out_report = os.path.abspath(objekt.inputs.out_report)
    objekt._post_run_hook(runtime)
    objekt._generate_report()
    return runtime


def pytest_runtest_setup(item):
    """Change to temporal directory"""
    os.chdir(mkdtemp())


@pytest.fixture
def mni_dir():
    return get_template('MNI152Lin')


@pytest.fixture
def oasis_dir():
    return get_template('OASIS')


@pytest.fixture
def reference():
    return str(get_template('MNI152Lin') / 'tpl-MNI152Lin_space-MNI_res-02_T1w.nii.gz')


@pytest.fixture
def reference_mask():
    return str(get_template('MNI152Lin') / 'tpl-MNI152Lin_space-MNI_res-02_brainmask.nii.gz')


@pytest.fixture
def moving():
    return os.path.join(get_ds003_downsampled(), 'sub-01/anat/sub-01_T1w.nii.gz')


@pytest.fixture
def nthreads():
    from multiprocessing import cpu_count
    # Tests are linear, so don't worry about leaving space for a control thread
    return min(int(os.getenv('CIRCLE_NPROCS', '8')), cpu_count())
