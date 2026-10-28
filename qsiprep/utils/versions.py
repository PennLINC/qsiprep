"""Versions of the external tools named in the methods boilerplate.

Each lookup returns ``None`` when the tool is absent or does not report a
version, so the boilerplate leaves the version out rather than state a wrong one.
"""

import functools
import glob
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .. import config


def _command_output(command, *args):
    executable = shutil.which(command)
    if executable is None:
        return ''
    try:
        proc = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ''
    return proc.stdout + proc.stderr


def _search(pattern, text):
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def versioned(name, version):
    """Return ``name`` followed by ``version``, or ``name`` alone when it is unknown."""
    return f'{name} {version}' if version else name


@functools.cache
def eddy_version():
    """Return the version of the installed eddy.

    A conda/pixi FSL has no ``$FSLDIR/etc/fslversion`` (what nipype reads), and
    its packages are versioned one by one, so eddy's own package version is read
    from the environment's ``conda-meta`` records.
    """
    from nipype.interfaces import fsl

    fsldir = os.getenv('FSLDIR')
    if not fsldir:
        return None
    for record in sorted(glob.glob(os.path.join(fsldir, 'conda-meta', 'fsl-eddy-[0-9]*.json'))):
        # pixi writes these records owner-only, so a non-root run (Apptainer,
        # docker --user) on an image that did not open them up cannot read them.
        try:
            with open(record) as fobj:
                return json.load(fobj).get('version')
        except OSError:
            continue
    return fsl.Info.version()


@functools.cache
def fsl_release():
    """Return the FSL release the FSL packages were pinned to (e.g. ``6.0.7.23``).

    ``scripts/update_fsl_pins.py`` records it in ``qsiprep/data/fsl_release.json``.
    """
    from ..data import load as load_data

    try:
        return json.loads(load_data('fsl_release.json').read_text()).get('release')
    except (OSError, ValueError):
        return None


def mrtrix3_version(dev=False):
    """Return the version of the MRtrix3 installation a command runs from.

    Commands resolve through the installation ``--mrtrix-version`` selects, except
    those only the development branch has, which always run from it.
    """
    if dev:
        return os.getenv('MRTRIX3_DEV_VERSION')
    return config.environment.mrtrix3_version


@functools.cache
def tortoise_version():
    """Return the version TORTOISE records in its ``settings/version.txt`` (e.g. ``V4.2.0``)."""
    for command in ('TORTOISEProcess', 'DRBUDDI'):
        executable = shutil.which(command)
        if executable is None:
            continue
        version_file = Path(executable).resolve().parent.parent / 'settings' / 'version.txt'
        try:
            return _search(r'TORTOISE\s+(\S+)', version_file.read_text())
        except OSError:
            return None
    return None


@functools.cache
def niimath_version():
    """Return niimath's version (e.g. ``v1.0.20260924``)."""
    return _search(r'^(v\S+)', _command_output('niimath', '--version'))


@functools.cache
def svht_denoise_version():
    """Return svht_denoise's version (e.g. ``0.1.20261001``)."""
    return _search(r'svht_denoise (\S+)', _command_output('svht_denoise', '--version'))


def python_package_version(name):
    """Return an installed Python package's version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
