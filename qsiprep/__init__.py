#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

"""

import warnings

from .__about__ import __copyright__, __credits__, __packagename__, __version__  # noqa

# cmp is not used by qsiprep, so ignore nipype-generated warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', r'cmp not installed')
warnings.filterwarnings('ignore', r'Enable tracemalloc')
warnings.filterwarnings('ignore', r"can't resolve package from __spec__ or __package__")
warnings.filterwarnings('ignore', category=ResourceWarning)
warnings.filterwarnings('ignore', r'Using or importing the ABCs from')
