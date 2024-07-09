# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Edited from fMRIPrep to represent QSIPrep arguments
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
r"""
A Python module to maintain unique, run-wide *QSIPrep* settings.

This module implements the memory structures to keep a consistent, singleton config.
Settings are passed across processes via filesystem, and a copy of the settings for
each run and subject is left under
``<qsiprep_dir>/sub-<participant_id>/log/<run_unique_id>/qsiprep.toml``.
Settings are stored using :abbr:`ToML (Tom's Markup Language)`.
The module has a :py:func:`~qsiprep.config.to_filename` function to allow writing out
the settings to hard disk in *ToML* format, which looks like:

.. literalinclude:: ../qsiprep/data/tests/config.toml
   :language: toml
   :name: qsiprep.toml
   :caption: **Example file representation of QSIPrep settings**.

This config file is used to pass the settings across processes,
using the :py:func:`~qsiprep.config.load` function.

Configuration sections
----------------------
.. autoclass:: environment
   :members:
.. autoclass:: execution
   :members:
.. autoclass:: workflow
   :members:
.. autoclass:: nipype
   :members:

Usage
-----
A config file is used to pass settings and collect information as the execution
graph is built across processes.

.. code-block:: Python

    from qsiprep import config
    config_file = config.execution.work_dir / '.qsiprep.toml'
    config.to_filename(config_file)
    # Call build_workflow(config_file, retval) in a subprocess
    with Manager() as mgr:
        from .workflow import build_workflow
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()
    config.load(config_file)
    # Access configs from any code section as:
    value = config.section.setting

Logging
-------
.. autoclass:: loggers
   :members:

Other responsibilities
----------------------
The :py:mod:`config` is responsible for other conveniency actions.

  * Switching Python's :obj:`multiprocessing` to *forkserver* mode.
  * Set up a filter for warnings as early as possible.
  * Automated I/O magic operations. Some conversions need to happen in the
    store/load processes (e.g., from/to :obj:`~pathlib.Path` \<-\> :obj:`str`,
    :py:class:`~bids.layout.BIDSLayout`, etc.)

"""

import os
from multiprocessing import set_start_method

from templateflow.conf import TF_LAYOUT

# Disable NiPype etelemetry always
_disable_et = bool(os.getenv("NO_ET") is not None or os.getenv("NIPYPE_NO_ET") is not None)
os.environ["NIPYPE_NO_ET"] = "1"
os.environ["NO_ET"] = "1"

CONFIG_FILENAME = "qsiprep.toml"

try:
    set_start_method("forkserver")
except RuntimeError:
    pass  # context has been already set
finally:
    # Defer all custom import for after initializing the forkserver and
    # ignoring the most annoying warnings
    import random
    import sys
    from pathlib import Path
    from time import strftime
    from uuid import uuid4

    from nipype import __version__ as _nipype_ver
    from templateflow import __version__ as _tf_ver

    from . import __version__

if not hasattr(sys, "_is_pytest_session"):
    sys._is_pytest_session = False  # Trick to avoid sklearn's FutureWarnings
# Disable all warnings in main and children processes only on production versions
if not any(
    (
        "+" in __version__,
        __version__.endswith(".dirty"),
        os.getenv("QSIPREP_DEV", "0").lower() in ("1", "on", "true", "y", "yes"),
    )
):
    from ._warnings import logging

    os.environ["PYTHONWARNINGS"] = "ignore"
elif os.getenv("QSIPREP_WARNINGS", "0").lower() in ("1", "on", "true", "y", "yes"):
    # allow disabling warnings on development versions
    # https://github.com/nipreps/fmriprep/pull/2080#discussion_r409118765
    from ._warnings import logging
else:
    import logging

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG

DEFAULT_MEMORY_MIN_GB = 0.01

# Ping NiPype eTelemetry once if env var was not set
# workers on the pool will have the env variable set from the master process
if not _disable_et:
    # Just get so analytics track one hit
    from contextlib import suppress

    from requests import ConnectionError, ReadTimeout
    from requests import get as _get_url

    with suppress((ConnectionError, ReadTimeout)):
        _get_url("https://rig.mit.edu/et/projects/nipy/nipype", timeout=0.05)

# Execution environment
_exec_env = os.name
_docker_ver = None
# special variable set in the container
if os.getenv("IS_DOCKER_8395080871"):
    _exec_env = "singularity"
    _cgroup = Path("/proc/1/cgroup")
    if _cgroup.exists() and "docker" in _cgroup.read_text():
        _docker_ver = os.getenv("DOCKER_VERSION_8395080871")
        _exec_env = "qsiprep-docker" if _docker_ver else "docker"
    del _cgroup

_fs_license = os.getenv("FS_LICENSE")
if not _fs_license and os.getenv("FREESURFER_HOME"):
    _fs_home = os.getenv("FREESURFER_HOME")
    if _fs_home and (Path(_fs_home) / "license.txt").is_file():
        _fs_license = str(Path(_fs_home) / "license.txt")
    del _fs_home

_templateflow_home = Path(
    os.getenv("TEMPLATEFLOW_HOME", os.path.join(os.getenv("HOME"), ".cache", "templateflow"))
)

try:
    from psutil import virtual_memory

    _free_mem_at_start = round(virtual_memory().available / 1024**3, 1)
except ImportError:
    _free_mem_at_start = None

_oc_limit = "n/a"
_oc_policy = "n/a"
try:
    # Memory policy may have a large effect on types of errors experienced
    _proc_oc_path = Path("/proc/sys/vm/overcommit_memory")
    if _proc_oc_path.exists():
        _oc_policy = {"0": "heuristic", "1": "always", "2": "never"}.get(
            _proc_oc_path.read_text().strip(), "unknown"
        )
        if _oc_policy != "never":
            _proc_oc_kbytes = Path("/proc/sys/vm/overcommit_kbytes")
            if _proc_oc_kbytes.exists():
                _oc_limit = _proc_oc_kbytes.read_text().strip()
            if _oc_limit in ("0", "n/a") and Path("/proc/sys/vm/overcommit_ratio").exists():
                _oc_limit = "{}%".format(Path("/proc/sys/vm/overcommit_ratio").read_text().strip())
except Exception:  # noqa: S110, BLE001
    pass


# Debug modes are names that influence the exposure of internal details to
# the user, either through additional derivatives or increased verbosity
DEBUG_MODES = ("fieldmaps", "pdb")


class _Config:
    """An abstract class forbidding instantiation."""

    _paths = ()

    def __init__(self):
        """Avert instantiation."""
        raise RuntimeError("Configuration type is not instantiable.")

    @classmethod
    def load(cls, settings, init=True, ignore=None):
        """Store settings from a dictionary."""
        ignore = ignore or {}
        for k, v in settings.items():
            if k in ignore or v is None:
                continue
            if k in cls._paths:
                if isinstance(v, list | tuple):
                    setattr(cls, k, [Path(val).absolute() for val in v])
                elif isinstance(v, dict):
                    setattr(cls, k, {key: Path(val).absolute() for key, val in v.items()})
                else:
                    setattr(cls, k, Path(v).absolute())
            elif hasattr(cls, k):
                setattr(cls, k, v)

        if init:
            try:
                cls.init()
            except AttributeError:
                pass

    @classmethod
    def get(cls):
        """Return defined settings."""
        from niworkflows.utils.spaces import Reference, SpatialReferences

        out = {}
        for k, v in cls.__dict__.items():
            if k.startswith("_") or v is None:
                continue
            if callable(getattr(cls, k)):
                continue
            if k in cls._paths:
                if isinstance(v, list | tuple):
                    v = [str(val) for val in v]
                elif isinstance(v, dict):
                    v = {key: str(val) for key, val in v.items()}
                else:
                    v = str(v)
            if isinstance(v, SpatialReferences):
                v = " ".join(str(s) for s in v.references) or None
            if isinstance(v, Reference):
                v = str(v) or None
            out[k] = v
        return out


class environment(_Config):
    """
    Read-only options regarding the platform and environment.

    Crawls runtime descriptive settings (e.g., default FreeSurfer license,
    execution environment, nipype and *QSIPrep* versions, etc.).
    The ``environment`` section is not loaded in from file,
    only written out when settings are exported.
    This config section is useful when reporting issues,
    and these variables are tracked whenever the user does not
    opt-out using the ``--notrack`` argument.

    """

    cpu_count = os.cpu_count()
    """Number of available CPUs."""
    exec_docker_version = _docker_ver
    """Version of Docker Engine."""
    exec_env = _exec_env
    """A string representing the execution platform."""
    free_mem = _free_mem_at_start
    """Free memory at start."""
    overcommit_policy = _oc_policy
    """Linux's kernel virtual memory overcommit policy."""
    overcommit_limit = _oc_limit
    """Linux's kernel virtual memory overcommit limits."""
    nipype_version = _nipype_ver
    """Nipype's current version."""
    templateflow_version = _tf_ver
    """The TemplateFlow client version installed."""
    version = __version__
    """*QSPrep*'s version."""


class nipype(_Config):
    """Nipype settings."""

    crashfile_format = "txt"
    """The file format for crashfiles, either text (txt) or pickle (pklz)."""
    get_linked_libs = False
    """Run NiPype's tool to enlist linked libraries for every interface."""
    memory_gb = None
    """Estimation in GB of the RAM this workflow can allocate at any given time."""
    nprocs = os.cpu_count()
    """Number of processes (compute tasks) that can be run in parallel (multiprocessing only)."""
    omp_nthreads = None
    """Number of CPUs a single process can access for multithreaded execution."""
    plugin = "MultiProc"
    """NiPype's execution plugin."""
    plugin_args = {
        "maxtasksperchild": 1,
        "raise_insufficient": False,
    }
    """Settings for NiPype's execution plugin."""
    remove_unnecessary_outputs = True
    """Clean up unused outputs after running"""
    resource_monitor = False
    """Enable resource monitor."""
    stop_on_first_crash = True
    """Whether the workflow should stop or continue after the first error."""

    @classmethod
    def get_plugin(cls):
        """Format a dictionary for Nipype consumption."""
        out = {
            "plugin": cls.plugin,
            "plugin_args": cls.plugin_args,
        }
        if cls.plugin in ("MultiProc", "LegacyMultiProc"):
            out["plugin_args"]["n_procs"] = int(cls.nprocs)
            if cls.memory_gb:
                out["plugin_args"]["memory_gb"] = float(cls.memory_gb)
        return out

    @classmethod
    def init(cls):
        """Set NiPype configurations."""
        from nipype import config as ncfg

        # Configure resource_monitor
        if cls.resource_monitor:
            ncfg.update_config(
                {
                    "monitoring": {
                        "enabled": cls.resource_monitor,
                        "sample_frequency": "0.5",
                        "summary_append": True,
                    }
                }
            )
            ncfg.enable_resource_monitor()

        # Nipype config (logs and execution)
        ncfg.update_config(
            {
                "execution": {
                    "crashdump_dir": str(execution.log_dir),
                    "crashfile_format": cls.crashfile_format,
                    "get_linked_libs": cls.get_linked_libs,
                    "remove_unnecessary_outputs": cls.remove_unnecessary_outputs,
                    "stop_on_first_crash": cls.stop_on_first_crash,
                    "check_version": False,  # disable future telemetry
                }
            }
        )

        if cls.omp_nthreads is None:
            cls.omp_nthreads = min(cls.nprocs - 1 if cls.nprocs > 1 else os.cpu_count(), 8)


class execution(_Config):
    """Configure run-level settings."""

    bids_dir = None
    """An existing path to the dataset, which must be BIDS-compliant."""
    derivatives = {}
    """Path(s) to search for pre-computed derivatives"""
    bids_database_dir = None
    """Path to the directory containing SQLite database indices for the input BIDS dataset."""
    bids_description_hash = None
    """Checksum (SHA256) of the ``dataset_description.json`` of the BIDS dataset."""
    bids_filters = None
    """A dictionary of BIDS selection filters."""
    boilerplate_only = False
    """Only generate a boilerplate."""
    sloppy = False
    """Run in sloppy mode (meaning, suboptimal parameters that minimize run-time)."""
    debug = []
    """Debug mode(s)."""
    fs_license_file = _fs_license
    """An existing file containing a FreeSurfer license."""
    fs_subjects_dir = None
    """FreeSurfer's subjects directory."""
    interactive_reports_only = False
    """Only build the interactive reports, based on the output directory."""
    layout = None
    """A :py:class:`~bids.layout.BIDSLayout` object, see :py:func:`init`."""
    log_dir = None
    """The path to a directory that contains execution logs."""
    log_level = 25
    """Output verbosity."""
    low_mem = None
    """Utilize uncompressed NIfTIs and other tricks to minimize memory allocation."""
    # md_only_boilerplate = False
    # """Do not convert boilerplate from MarkDown to LaTex and HTML."""
    notrack = False
    """Do not collect telemetry information for *QSIPrep*."""
    output_dir = None
    """Folder where derivatives will be stored."""
    output_layout = None
    """Layout of derivatives within output_dir."""
    # output_spaces = None
    # """List of (non)standard spaces designated (with the ``--output-spaces`` flag of
    # the command line) as spatial references for outputs."""
    reports_only = False
    """Only build the reports, based on the reportlets found in a cached working directory."""
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
    """Unique identifier of this particular run."""
    skip_odf_reports = False
    """Disable ODF recon reports."""
    participant_label = None
    """List of participant identifiers that are to be preprocessed."""
    qsiprep_dir = None
    """Root of QSIPrep BIDS Derivatives dataset. Depends on output_layout."""
    qsirecon_dir = None
    """Root of QSIRecon BIDS Derivatives dataset."""
    recon_input = None
    """Directory containing QSIPrep derivatives to run through recon workflows."""
    recon_only = False
    """Run only recon workflows."""
    reportlets_dir = None
    """Path where reportlets are written."""
    run_preproc_and_recon = False
    """Will both preproc and recon be run in a single call?"""
    skip_anat_based_spatial_normalization = False
    """Should we skip normalizing the anatomical data to a template?"""
    task_id = None
    """Select a particular task from all available in the dataset."""
    templateflow_home = _templateflow_home
    """The root folder of the TemplateFlow client."""
    work_dir = Path("work").absolute()
    """Path to a working directory where intermediate results will be available."""
    write_graph = False
    """Write out the computational graph corresponding to the planned preprocessing."""
    dataset_links = {}
    """A dictionary of dataset links to be used to track Sources in sidecars."""

    _layout = None

    _paths = (
        "bids_dir",
        "bids_database_dir",
        "dataset_links",
        "eddy_config",
        "fs_license_file",
        "fs_subjects_dir",
        "freesurfer_input",
        "layout",
        "log_dir",
        "output_dir",
        "qsiprep_dir",
        "qsirecon_dir",
        "recon_input",
        "reportlets_dir",
        "templateflow_home",
        "work_dir",
    )

    @classmethod
    def init(cls):
        """Create a new BIDS Layout accessible with :attr:`~execution.layout`."""
        if cls.fs_license_file and Path(cls.fs_license_file).is_file():
            os.environ["FS_LICENSE"] = str(cls.fs_license_file)

        if cls._layout is None:
            import re

            from bids.layout import BIDSLayout
            from bids.layout.index import BIDSLayoutIndexer

            _db_path = cls.bids_database_dir or (cls.work_dir / cls.run_uuid / "bids_db")
            _db_path.mkdir(exist_ok=True, parents=True)

            # Recommended after PyBIDS 12.1
            ignore_patterns = [
                "code",
                "stimuli",
                "sourcedata",
                "models",
                re.compile(r"\/\.\w+|^\.\w+"),  # hidden files
                re.compile(r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|func|eeg|ieeg|meg|perf)"),
            ]
            if cls.participant_label and cls.bids_database_dir is None:
                # Ignore any subjects who aren't the requested ones.
                # This is only done if the database is written out to a run-specific folder.
                ignore_patterns.append(
                    re.compile(r"sub-(?!(" + "|".join(cls.participant_label) + r")(\b|_))")
                )

            _indexer = BIDSLayoutIndexer(
                validate=False,
                ignore=ignore_patterns,
            )
            cls._layout = BIDSLayout(
                str(cls.bids_dir),
                database_path=_db_path,
                reset_database=cls.bids_database_dir is None,
                indexer=_indexer,
            )
            cls.bids_database_dir = _db_path
        cls.layout = cls._layout
        if cls.bids_filters:
            from bids.layout import Query

            def _process_value(value):
                """Convert string with "Query" in it to Query object."""
                if isinstance(value, list):
                    return [_process_value(val) for val in value]
                else:
                    return (
                        getattr(Query, value[7:-4])
                        if not isinstance(value, Query) and "Query" in value
                        else value
                    )

            # unserialize pybids Query enum values
            for acq, filters in cls.bids_filters.items():
                for k, v in filters.items():
                    cls.bids_filters[acq][k] = _process_value(v)

        dataset_links = {
            "raw": cls.bids_dir,
            "templateflow": Path(TF_LAYOUT.root),
        }
        for deriv_name, deriv_path in cls.derivatives.items():
            dataset_links[deriv_name] = deriv_path
        cls.dataset_links = dataset_links

        if "all" in cls.debug:
            cls.debug = list(DEBUG_MODES)


# These variables are not necessary anymore
del _fs_license
del _exec_env
del _nipype_ver
del _templateflow_home
del _tf_ver
del _free_mem_at_start
del _oc_limit
del _oc_policy


class workflow(_Config):
    """Configure the particular execution graph of this workflow."""

    anat_modality = None
    """Modality to use as the anatomical reference. Images of this
    contrast will be skull stripped and segmented for use in the
    visual reports and reconstruction. If --infant, T2w is forced."""
    anat_only = False
    """Execute the anatomical preprocessing only."""
    anatomical_template = None
    """Keeps the :py:class:`~niworkflows.utils.spaces.SpatialReferences`
    instance keeping standard and nonstandard spaces."""
    b0_threshold = None
    """Any value in the .bval file less than this will be considered a b=0 image."""
    b0_motion_corr_to = None
    """Perform SHORELine's initial b=0-based registration to first volume? 
    Or make a template? Either 'iterative' or 'first'"""
    b0_to_t1w_transform = None
    """Transformation model for intramodal registration."""
    b1_biascorrect_stage = None
    """The stage of processing at which to apply B1 bias correction. Either "final" (after
    resampling), "none" (skipped entirely) or "legacy" (before concatenation)."""
    cifti_output = None
    """Generate HCP Grayordinates, accepts either ``'91k'`` (default) or ``'170k'``."""
    denoise_after_combining = False
    """Run ``dwidenoise`` after combining dwis, but before motion correction."""
    denoise_method = None
    """Image-based denoising method. Either "dwidenoise" (MRtrix), "patch2self" (DIPY)
    or "none"."""
    distortion_group_merge = None
    """How to combine images across distortion groups (concatenate, average or none)."""
    do_reconall = True
    """Run FreeSurfer's surface reconstruction (ignored)."""
    dwi_denoise_window = None
    """Window size in voxels for image-based denoising, integer or "auto"."""
    dwi_no_biascorr = None
    """DEPRECATED: see --b1-biascorrect-stage."""
    dwi_only = False
    """DEPRECATED: True if anat_modality is 'none'."""
    eddy_config = None
    """Configuration for running Eddy."""
    fmap_bspline = None
    """Regularize fieldmaps with a field of B-Spline basis."""
    fmap_demean = None
    """Remove the mean from fieldmaps."""
    force_syn = None
    """Run *fieldmap-less* susceptibility-derived distortions estimation."""
    hmc_model = None
    """Model used to generate target images for hmc."""
    hmc_transform = None
    """Transformation to be used in SHORELine."""
    ignore = None
    """Ignore particular steps for *QSIPrep*."""
    infant = False
    """Configure pipelines specifically for infant brains"""
    intramodal_template_iters = None
    """Number of iterations for intramodal template construction."""
    intramodal_template_transform = None
    """Transformation used for building the intramodal template."""
    longitudinal = False
    """Run FreeSurfer ``recon-all`` with the ``-logitudinal`` flag."""
    no_b0_harmonization = False
    """Skip re-scaling dwi scans to have matching b=0 intensities."""
    output_resolution = None
    """Isotropic voxel size for outputs."""
    pepolar_method = None
    """SDC method to be used for PEPOLAR fieldmaps."""
    recon_input_pipeline = None
    """Specifies which pipeline was used to preprocess data in ``recon_input``"""
    recon_spec = None
    """Recon workflow specification."""
    separate_all_dwis = False
    """Process all dwis separately - do not attempt concatenation."""
    shoreline_iters = None
    """How many iterations to run SHORELine."""
    unringing_method = None
    """Method for Gibbs-ringing removal. Either "none", "mrdegibbs" or "rpg"."""
    use_syn_sdc = None
    """Run *fieldmap-less* susceptibility-derived distortions estimation
    in the absence of any alternatives."""


class loggers:
    """Keep loggers easily accessible (see :py:func:`init`)."""

    _fmt = "%(asctime)s,%(msecs)d %(name)-2s %(levelname)-2s:\n\t %(message)s"
    _datefmt = "%y%m%d-%H:%M:%S"

    default = logging.getLogger()
    """The root logger."""
    cli = logging.getLogger("cli")
    """Command-line interface logging."""
    workflow = logging.getLogger("nipype.workflow")
    """NiPype's workflow logger."""
    interface = logging.getLogger("nipype.interface")
    """NiPype's interface logger."""
    utils = logging.getLogger("nipype.utils")
    """NiPype's utils logger."""

    @classmethod
    def init(cls):
        """
        Set the log level, initialize all loggers into :py:class:`loggers`.

            * Add new logger levels (25: IMPORTANT, and 15: VERBOSE).
            * Add a new sub-logger (``cli``).
            * Logger configuration.

        """
        from nipype import config as ncfg

        if not cls.cli.hasHandlers():
            _handler = logging.StreamHandler(stream=sys.stdout)
            _handler.setFormatter(logging.Formatter(fmt=cls._fmt, datefmt=cls._datefmt))
            cls.cli.addHandler(_handler)
        cls.default.setLevel(execution.log_level)
        cls.cli.setLevel(execution.log_level)
        cls.interface.setLevel(execution.log_level)
        cls.workflow.setLevel(execution.log_level)
        cls.utils.setLevel(execution.log_level)
        ncfg.update_config(
            {"logging": {"log_directory": str(execution.log_dir), "log_to_file": True}}
        )


class seeds(_Config):
    """Initialize the PRNG and track random seed assignments"""

    _random_seed = None
    master = None
    """Master random seed to initialize the Pseudorandom Number Generator (PRNG)"""
    ants = None
    """Seed used for antsRegistration, antsAI, antsMotionCorr"""
    numpy = None
    """Seed used by NumPy"""

    @classmethod
    def init(cls):
        if cls._random_seed is not None:
            cls.master = cls._random_seed
        if cls.master is None:
            cls.master = random.randint(1, 65536)
        random.seed(cls.master)  # initialize the PRNG
        # functions to set program specific seeds
        cls.ants = _set_ants_seed()
        cls.numpy = _set_numpy_seed()


def _set_ants_seed():
    """Fix random seed for antsRegistration, antsAI, antsMotionCorr"""
    val = random.randint(1, 65536)
    os.environ["ANTS_RANDOM_SEED"] = str(val)
    return val


def _set_numpy_seed():
    """NumPy's random seed is independent from Python's `random` module"""
    import numpy as np

    val = random.randint(1, 65536)
    np.random.seed(val)
    return val


def from_dict(settings, init=True, ignore=None):
    """Read settings from a flat dictionary.

    Arguments
    ---------
    setting : dict
        Settings to apply to any configuration
    init : `bool` or :py:class:`~collections.abc.Container`
        Initialize all, none, or a subset of configurations.
    ignore : :py:class:`~collections.abc.Container`
        Collection of keys in ``setting`` to ignore
    """

    # Accept global True/False or container of configs to initialize
    def initialize(x):
        return init if init in (True, False) else x in init

    nipype.load(settings, init=initialize("nipype"), ignore=ignore)
    execution.load(settings, init=initialize("execution"), ignore=ignore)
    workflow.load(settings, init=initialize("workflow"), ignore=ignore)
    seeds.load(settings, init=initialize("seeds"), ignore=ignore)

    loggers.init()


def load(filename, skip=None, init=True):
    """Load settings from file.

    Arguments
    ---------
    filename : :py:class:`os.PathLike`
        TOML file containing QSIPrep configuration.
    skip : dict or None
        Sets of values to ignore during load, keyed by section name
    init : `bool` or :py:class:`~collections.abc.Container`
        Initialize all, none, or a subset of configurations.
    """
    from toml import loads

    skip = skip or {}

    # Accept global True/False or container of configs to initialize
    def initialize(x):
        return init if init in (True, False) else x in init

    filename = Path(filename)
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != "environment":
            section = getattr(sys.modules[__name__], sectionname)
            ignore = skip.get(sectionname)
            section.load(configs, ignore=ignore, init=initialize(sectionname))
    init_spaces()


def get(flat=False):
    """Get config as a dict."""
    settings = {
        "environment": environment.get(),
        "execution": execution.get(),
        "workflow": workflow.get(),
        "nipype": nipype.get(),
        "seeds": seeds.get(),
    }
    if not flat:
        return settings

    return {
        ".".join((section, k)): v
        for section, configs in settings.items()
        for k, v in configs.items()
    }


def dumps():
    """Format config into toml."""
    from toml import dumps

    return dumps(get())


def to_filename(filename):
    """Write settings to file."""
    filename = Path(filename)
    filename.write_text(dumps())


def init_spaces(checkpoint=True):
    """Initialize the :attr:`~workflow.spaces` setting."""
    from niworkflows.utils.spaces import Reference, SpatialReferences

    # spaces = execution.output_spaces or SpatialReferences()
    spaces = SpatialReferences()
    if not isinstance(spaces, SpatialReferences):
        spaces = SpatialReferences(
            [ref for s in spaces.split(" ") for ref in Reference.from_string(s)]
        )

    if checkpoint and not spaces.is_cached():
        spaces.checkpoint()

    # Add the default standard space if not already present (required by several sub-workflows)
    if "MNI152NLin2009cAsym" not in spaces.get_spaces(nonstandard=False, dim=(3,)):
        spaces.add(Reference("MNI152NLin2009cAsym", {}))

    # Ensure user-defined spatial references for outputs are correctly parsed.
    # Certain options require normalization to a space not explicitly defined by users.
    # These spaces will not be included in the final outputs.
    cifti_output = workflow.cifti_output
    if cifti_output:
        # CIFTI grayordinates to corresponding FSL-MNI resolutions.
        vol_res = "2" if cifti_output == "91k" else "1"
        spaces.add(Reference("MNI152NLin6Asym", {"res": vol_res}))

    # Make the SpatialReferences object available
    workflow.spaces = spaces
