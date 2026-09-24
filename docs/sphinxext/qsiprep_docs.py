"""Helpers for the ``.. workflow::`` directives in the documentation.

The workflow builders read their settings from :mod:`qsiprep.config`, which is
normally filled by the command-line parser, and some of them read NIfTI headers
of their input files. Importing this module loads the parser defaults into the
config and writes a tiny fake dataset to a temporary directory, so a directive
can build a workflow with two lines::

    from qsiprep_docs import example_unit
    wf = init_fsl_hmc_wf(example_unit('pepolar'), source_file=..., t2w_sdc=False)

``example_unit`` returns a :class:`~qsiplan.adapters.PreprocUnit` for one of
these layouts:

``single``
    One DWI series and no fieldmap.
``pepolar``
    Two DWI series with opposite phase encoding, correcting each other.
``epi``
    One DWI series and a reverse phase-encoded ``epi`` fieldmap.
``phasediff``
    One DWI series and a phase-difference GRE fieldmap.
"""

import argparse
import atexit
import json
import os
import shutil
import tempfile

import nibabel as nb
import numpy as np

from qsiprep import config
from qsiprep.cli.parser import _build_parser

_ROOT = tempfile.mkdtemp(prefix='qsiprep_docs_')
atexit.register(shutil.rmtree, _ROOT, True)

# Some builders refuse to construct without FSL. Only the graph is drawn here,
# so a placeholder location is enough.
os.environ.setdefault('FSLDIR', _ROOT)
os.environ.setdefault('FSLOUTPUTTYPE', 'NIFTI_GZ')

ANATOMICAL_TEMPLATE = 'MNI152NLin2009cAsym'


def configure(**overrides):
    """Load the parser defaults into :mod:`qsiprep.config`, then apply overrides.

    Overrides are ``section__setting`` keyword arguments, for example
    ``workflow__hmc_method='tortoise'``.
    """
    defaults = {
        action.dest: action.default
        for action in _build_parser()._actions
        if action.default is not argparse.SUPPRESS and action.dest not in ('help', 'version')
    }
    config.from_dict(defaults, init=False)
    config.workflow.output_resolution = 2.0
    config.workflow.anatomical_template = ANATOMICAL_TEMPLATE
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.execution.output_dir = _ROOT
    config.execution.work_dir = _ROOT
    if overrides.get('workflow__hmc_method') == 'shoreline':
        # The parser resolves these from --shoreline-config; mirror its defaults.
        config.workflow.shoreline_model = '3dshore'
        config.workflow.shoreline_iters = 2
        config.workflow.hmc_transform = 'Affine'
    for key, value in overrides.items():
        section, setting = key.split('__', 1)
        setattr(getattr(config, section), setting, value)


def _write_image(path, n_volumes=1):
    shape = (2, 2, 2) if n_volumes == 1 else (2, 2, 2, n_volumes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)).to_filename(path)


def _write_dwi(path, pe_dir, n_volumes=4):
    _write_image(path, n_volumes)
    stem = path[: -len('.nii.gz')]
    bvals = [0] + [1000] * (n_volumes - 1)
    with open(stem + '.bval', 'w') as fobj:
        fobj.write(' '.join(str(b) for b in bvals) + '\n')
    with open(stem + '.bvec', 'w') as fobj:
        for row in ([0] + [1] * (n_volumes - 1), [0] * n_volumes, [0] * n_volumes):
            fobj.write(' '.join(str(v) for v in row) + '\n')
    with open(stem + '.json', 'w') as fobj:
        json.dump({'PhaseEncodingDirection': pe_dir, 'TotalReadoutTime': 0.05}, fobj)


_SUBJECT = os.path.join(_ROOT, 'bids', 'sub-01')
AP = os.path.join(_SUBJECT, 'dwi', 'sub-01_dir-AP_dwi.nii.gz')
PA = os.path.join(_SUBJECT, 'dwi', 'sub-01_dir-PA_dwi.nii.gz')
EPI = os.path.join(_SUBJECT, 'fmap', 'sub-01_dir-PA_epi.nii.gz')
PHASEDIFF = os.path.join(_SUBJECT, 'fmap', 'sub-01_phasediff.nii.gz')
MAGNITUDE1 = os.path.join(_SUBJECT, 'fmap', 'sub-01_magnitude1.nii.gz')
MAGNITUDE2 = os.path.join(_SUBJECT, 'fmap', 'sub-01_magnitude2.nii.gz')
T1W = os.path.join(_SUBJECT, 'anat', 'sub-01_T1w.nii.gz')

_write_dwi(AP, 'j-')
_write_dwi(PA, 'j')
_write_image(EPI)
for _path in (PHASEDIFF, MAGNITUDE1, MAGNITUDE2, T1W):
    _write_image(_path)

configure()


def example_unit(kind='single'):
    """Return a :class:`~qsiplan.adapters.PreprocUnit` for one of the fake layouts."""
    from qsiplan.models import CorrectionMethod

    from qsiprep.tests.preproc_factory import make_preproc_unit

    if kind == 'single':
        return make_preproc_unit([AP], pe_dir='j-')
    if kind == 'pepolar':
        return make_preproc_unit(
            [AP, PA],
            method=CorrectionMethod.PEPOLAR,
            pe_dirs={AP: 'j-', PA: 'j'},
        )
    if kind == 'epi':
        return make_preproc_unit(
            [AP],
            method=CorrectionMethod.PEPOLAR,
            pe_dir='j-',
            estimation_sources=[AP, EPI],
            pe_dirs={EPI: 'j'},
        )
    if kind == 'phasediff':
        return make_preproc_unit(
            [AP],
            method=CorrectionMethod.PHASEDIFF,
            pe_dir='j-',
            estimation_sources=[PHASEDIFF, MAGNITUDE1, MAGNITUDE2],
        )
    raise ValueError(f'unknown example unit {kind!r}')
