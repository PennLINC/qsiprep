#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

"""

from .__about__ import (  # noqa
    __version__,
    __copyright__,
    __credits__,
    __packagename__,
)

import warnings

# cmp is not used by qsiprep, so ignore nipype-generated warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', r'cmp not installed')
warnings.filterwarnings('ignore', r'Enable tracemalloc')
warnings.filterwarnings('ignore', r"can't resolve package from __spec__ or __package__")
warnings.filterwarnings('ignore', category=ResourceWarning)
warnings.filterwarnings('ignore', r'Using or importing the ABCs from')
