# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'The PennBBL Developers'
__copyright__ = 'Copyright 2019, PennBBL, Perelman School of Medicine, University of Pennsylvania'
__credits__ = ['Matt Cieslak', 'Clint Greene', 'The FMRIPREP Authors']
__license__ = '3-clause BSD'
__maintainer__ = 'Matt Cieslak'
__email__ = 'crn.pennbbl@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/pennbbl/qsiprep'
__packagename__ = 'qsiprep'
__description__ = ("qsiprep builds workflows for preprocessing and reconstructing q-space images")
__longdesc__ = """\
qsiprep borrows heavily from FMRIPREP to build workflows for preprocessing q-space images
such as Diffusion Spectrum Images (DSI), multi-shell HARDI and compressed sensing DSI (CS-DSI).
It utilizes Dipy and ANTs to implement a novel high-b-value head motion correction approach
using q-space methods such as 3dSHORE to iteratively generate head motion target images for each
gradient direction and strength.

Since qsiprep uses the FMRIPREP workflow-building strategy, it can also generate methods
boilerplate and quality-check figures.

Users can also reconstruct orientation distribution functions (ODFs), fiber orientation
distributions (FODs) and perform tractography, estimate anisotropy scalars and connectivity
estimation using a combination of Dipy, MRTrix and DSI Studio using a JSON-based pipeline
specification.

[Documentation `qsiprep.org <https://qsiprep.readthedocs.io>`_]
"""

DOWNLOAD_URL = (
    'https://github.com/pennbbl/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))


SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

REQUIRES = [
    'numpy',
    'lockfile',
    'future',
    'scikit-learn',
    'matplotlib>=2.2.0',
    'nilearn',
    'sklearn',
    'dipy',
    'nibabel>=2.2.1',
    'pandas',
    'grabbit==0.2.3',
    'pybids==0.6.5',
    'nitime',
    'nipype>=1.1.3',
    'statsmodels',
    'seaborn',
    'indexed_gzip>=0.8.2',
    'scikit-image',
    'versioneer',
    'pyyaml',
]

LINKS_REQUIRES = [
]

TESTS_REQUIRES = [
    "mock",
    "codecov",
    "pytest",
]

EXTRA_REQUIRES = {
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme',
        'sphinx-argparse',
        'pydotplus',
        'pydot>=1.2.3',
        'packaging',
        'nbsphinx',
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'datalad': ['datalad'],
    'resmon': ['psutil>=5.4.0'],
    # 'sentry': ['raven'],
}
EXTRA_REQUIRES['docs'] = EXTRA_REQUIRES['doc']

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]
