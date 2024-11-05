# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Utilities and mocks for testing and documentation building."""
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

from toml import loads

from qsiprep.data import load as load_data
from qsiprep.utils import doc


@contextmanager
def mock_config():
    """Create a mock config for documentation and testing purposes."""
    from qsiprep import config

    _old_fs = os.getenv("FREESURFER_HOME")
    if not _old_fs:
        os.environ["FREESURFER_HOME"] = mkdtemp()

    filename = load_data("tests/config.toml").resolve()
    if not filename.exists():
        base_path = os.path.dirname(filename)
        raise FileNotFoundError(
            f"File not found: {filename}\nFiles in {base_path}:\n{os.listdir(base_path)}"
        )

    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != "environment":
            section = getattr(config, sectionname)
            section.load(configs, init=False)

    config.nipype.omp_nthreads = 1
    config.nipype.init()
    config.loggers.init()

    config.execution.work_dir = Path(mkdtemp())
    config.execution.fmri_dir = Path(doc.download_example_data(out_dir=mkdtemp()))
    config.execution.output_dir = Path(mkdtemp())
    config.execution.bids_database_dir = None
    config.execution._layout = None
    config.execution.init()

    yield

    shutil.rmtree(config.execution.work_dir)
    shutil.rmtree(config.execution.output_dir)

    if not _old_fs:
        del os.environ["FREESURFER_HOME"]
