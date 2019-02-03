#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
These pipelines are developed by the Poldrack lab at Stanford University
(https://poldracklab.stanford.edu/) for use at
the Center for Reproducible Neuroscience (http://reproducibility.stanford.edu/),
as well as for open-source software distribution.
"""
import logging

from .__about__ import (
    __version__, __packagename__, __author__, __copyright__,
    __credits__, __license__, __maintainer__, __email__, __status__,
    __description__, __longdesc__)


__all__ = [
    '__version__',
    '__packagename__',
    '__author__',
    '__copyright__',
    '__credits__',
    '__license__',
    '__maintainer__',
    '__email__',
    '__status__',
    '__description__',
    '__longdesc__',
    'NIWORKFLOWS_LOG',
]

NIWORKFLOWS_LOG = logging.getLogger(__packagename__)
NIWORKFLOWS_LOG.setLevel(logging.INFO)

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
