# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Top-module metadata."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"

import warnings

# cmp is not used by qsiprep, so ignore nipype-generated warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", r"cmp not installed")
warnings.filterwarnings("ignore", r"Enable tracemalloc")
warnings.filterwarnings("ignore", r"can't resolve package from __spec__ or __package__")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", r"Using or importing the ABCs from")
