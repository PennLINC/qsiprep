# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Changes made to parse QSIPrep cli arguments
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
"""Parser."""

import sys

from .. import config


def _build_parser(**kwargs):
    """Build parser object.

    ``kwargs`` are passed to ``argparse.ArgumentParser`` (mainly useful for debugging).
    """
    from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path

    from packaging.version import Version

    deprecations = {
        # parser attribute name: (replacement flag, version slated to be removed in)
        'dwi_only': ('--anat-modality none', '0.23.0'),
        'prefer_dedicated_fmaps': (None, '0.23.0'),
        'dwi_no_biascorr': ('--b1-biascorrect-stage none', '0.23.0'),
        'b0_motion_corr_to': (None, '0.23.0'),
        'b0_to_t1w_transform': ('--b0-t0-anat-transform', '0.23.0'),
        'longitudinal': ('--subject-anatomical-reference unbiased', '0.24.0'),
    }

    class DeprecatedAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            new_opt, rem_vers = deprecations.get(self.dest, (None, None))
            msg = (
                f'{self.option_strings} has been deprecated and will be removed in '
                f'{rem_vers or "a later version"}.'
            )
            if new_opt:
                msg += f' Please use `{new_opt}` instead.'
            print(msg, file=sys.stderr)
            delattr(namespace, self.dest)

    class ToDict(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            d = {}
            for spec in values:
                try:
                    name, loc = spec.split('=')
                    loc = Path(loc)
                except ValueError:
                    loc = Path(spec)
                    name = loc.name

                if name in d:
                    raise ValueError(f'Received duplicate derivative name: {name}')

                d[name] = loc
            setattr(namespace, self.dest, d)

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f'Path does not exist: <{path}>.')
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f'Path should point to a file (or symlink of file): <{path}>.')
        return path

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _int_or_auto(value, parser):
        """Ensure an argument is an integer or 'auto'."""
        if value.lower() == 'auto':
            return value
        try:
            value = int(value)
        except ValueError as exc:
            raise parser.error('Argument must be an integer or "auto".') from exc

        if value < 1:
            raise parser.error('Argument must be greater than zero.')

        if value % 2 == 0:
            raise parser.error('Argument must be an odd integer.')

        return value

    def _to_gb(value):
        scale = {'G': 1, 'T': 10**3, 'M': 1e-3, 'K': 1e-6, 'B': 1e-9}
        digits = ''.join([c for c in value if c.isdigit()])
        units = value[len(digits) :] or 'M'
        return int(digits) * scale[units[0]]

    def _drop_sub(value):
        return value[4:] if value.startswith('sub-') else value

    def _drop_ses(value):
        return value[4:] if value.startswith('ses-') else value

    def _process_value(value):
        import bids

        if value is None:
            return bids.layout.Query.NONE
        elif value == '*':
            return bids.layout.Query.ANY
        else:
            return value

    def _filter_pybids_none_any(dct):
        d = {}
        for k, v in dct.items():
            if isinstance(v, list):
                d[k] = [_process_value(val) for val in v]
            else:
                d[k] = _process_value(v)
        return d

    def _bids_filter(value, parser):
        from json import JSONDecodeError, loads

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError as e:
                    raise parser.error(f'JSON syntax error in: <{value}>.') from e
            else:
                raise parser.error(f'Path does not exist: <{value}>.')

    verstr = f'QSIPrep v{config.environment.version}'
    currentv = Version(config.environment.version)
    is_release = not any((currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease))

    parser = ArgumentParser(
        description=f'{verstr}: q-Space Image Preprocessing workflows',
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    IntOrAuto = partial(_int_or_auto, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'bids_dir',
        action='store',
        type=PathExists,
        help='The root folder of a BIDS valid dataset (sub-XXXXX folders should '
        'be found at the top level in this folder).',
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help='The output path for the outcomes of preprocessing and visual reports',
    )
    parser.add_argument(
        'analysis_level',
        choices=['participant'],
        help='Processing stage to be run, only "participant" in the case of QSIPrep (for now).',
    )

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument(
        '--skip-bids-validation',
        action='store_true',
        default=False,
        help='Assume the input dataset is BIDS compliant and skip the validation',
    )
    g_bids.add_argument(
        '--participant-label',
        action='store',
        nargs='+',
        type=_drop_sub,
        help='A space delimited list of participant identifiers or a single '
        'identifier (the sub- prefix can be removed)',
    )
    g_bids.add_argument(
        '--session-id',
        action='store',
        nargs='+',
        type=_drop_ses,
        default=None,
        help='A space delimited list of session identifiers or a single '
        'identifier (the ses- prefix can be removed)',
    )

    g_bids.add_argument(
        '--bids-filter-file',
        dest='bids_filters',
        action='store',
        type=BIDSFilter,
        metavar='FILE',
        help='A JSON file describing custom BIDS input filters using PyBIDS. '
        'For further details, please check out '
        'https://fmriprep.readthedocs.io/en/'
        f'{currentv.base_version if is_release else "latest"}/faq.html#'
        'how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep',
    )
    g_bids.add_argument(
        '--bids-database-dir',
        metavar='PATH',
        type=Path,
        help='Path to a PyBIDS database folder, for faster indexing (especially '
        'useful for large datasets). Will be created if not present.',
    )

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument(
        '--nprocs',
        '--nthreads',
        '--n-cpus',
        dest='nprocs',
        action='store',
        type=PositiveInt,
        help='Maximum number of threads across all processes',
    )
    g_perfm.add_argument(
        '--omp-nthreads',
        action='store',
        type=PositiveInt,
        help='Maximum number of threads per-process',
    )
    g_perfm.add_argument(
        '--mem',
        '--mem-mb',
        dest='memory_gb',
        action='store',
        type=_to_gb,
        metavar='MEMORY_MB',
        help='Upper bound memory limit for QSIPrep processes',
    )
    g_perfm.add_argument(
        '--low-mem',
        action='store_true',
        help='Attempt to reduce memory usage (will increase disk usage in working directory)',
    )
    g_perfm.add_argument(
        '--use-plugin',
        '--nipype-plugin-file',
        action='store',
        metavar='FILE',
        type=IsFile,
        help='Nipype plugin configuration file',
    )
    g_perfm.add_argument(
        '--sloppy',
        action='store_true',
        default=False,
        help='Use low-quality tools for speed - TESTING ONLY',
    )

    g_subset = parser.add_argument_group('Options for performing only a subset of the workflow')
    g_subset.add_argument('--anat-only', action='store_true', help='Run anatomical workflows only')
    g_subset.add_argument(
        '--dwi-only',
        action='store_true',
        help='ignore anatomical (T1w/T2w) data and process DWIs only',
    )
    g_subset.add_argument(
        '--boilerplate-only',
        '--boilerplate',
        action='store_true',
        default=False,
        help='Generate boilerplate only',
    )
    g_subset.add_argument(
        '--reports-only',
        action='store_true',
        default=False,
        help="Only generate reports, don't run workflows. This will only rerun report "
        'aggregation, not reportlet generation for specific nodes.',
    )

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore',
        required=False,
        action='store',
        nargs='+',
        default=[],
        choices=['fieldmaps', 'sbref', 't2w', 'flair', 'fmap-jacobian', 'phase'],
        help='Ignore selected aspects of the input dataset to disable corresponding '
        'parts of the workflow (a space delimited list)',
    )
    g_conf.add_argument(
        '--infant',
        action='store_true',
        help='Configure pipelines to process infant brains. '
        'If using this parameter, the anatomical-template will be changed to MNIInfant. '
        "The appropriate MNIInfant cohort will be selected based on the participant's age.",
    )
    g_conf.add_argument(
        '--longitudinal',
        action=DeprecatedAction,
        help='Treat dataset as longitudinal - may increase runtime',
    )
    g_conf.add_argument(
        '--subject-anatomical-reference',
        choices=['first-alphabetically', 'unbiased', 'sessionwise'],
        default='first-alphabetically',
        help='How to define subject-specific anatomical space. '
        'sessionwise will produce one anatomical space per session. '
        'The others combine anatomical data across sessions to define '
        'one anatomical space per subject.',
    )
    g_conf.add_argument(
        '--skip-anat-based-spatial-normalization',
        action='store_true',
        default=False,
        help='skip running the anat-based normalization to template space. '
        'Default is to run the normalization.',
    )
    g_conf.add_argument(
        '--anat-modality',
        choices=['T1w', 'T2w', 'none'],
        default='T1w',
        help='Modality to use as the anatomical reference. Images of this '
        'contrast will be skull stripped and segmented for use in the '
        'visual reports. If --infant, T2w is forced.',
    )
    g_conf.add_argument(
        '--b0-threshold',
        action='store',
        type=int,
        default=100,
        help='any value in the .bval file less than this will be considered '
        'a b=0 image. Current default threshold = 100; this threshold can be '
        'lowered or increased. Note, setting this too high can result in inaccurate results.',
    )
    g_conf.add_argument(
        '--dwi-denoise-window',
        action='store',
        type=IntOrAuto,
        default='auto',
        help=(
            'Window size in voxels for image-based denoising: odd integer or "auto". '
            'Any non-"auto" value must be an odd, positive integer. '
            'If using the "dwidenoise" denoising method, '
            'the "auto" option will calculate a window size '
            'based on the number of volumes according to the method described by the '
            'dwidenoise documentation. '
            'If using the "patch2self" denoising method, this argument will not be used.'
        ),
    )
    g_conf.add_argument(
        '--denoise-method',
        action='store',
        choices=['dwidenoise', 'patch2self', 'none'],
        default='dwidenoise',
        help='Image-based denoising method. Either "dwidenoise" (MRtrix), '
        '"patch2self" (DIPY) or "none". (default: dwidenoise)',
    )
    g_conf.add_argument(
        '--unringing-method',
        action='store',
        choices=['none', 'mrdegibbs', 'rpg'],
        help='Method for Gibbs-ringing removal.\n - none: no action\n - mrdegibbs: '
        'use mrdegibbs from mrtrix3\n - rpg: Gibbs from TORTOISE, suggested for partial'
        ' Fourier acquisitions (default: none).',
    )
    g_conf.add_argument(
        '--dwi-no-biascorr',
        action='store_true',
        help='DEPRECATED: see --b1-biascorrect-stage',
    )
    g_conf.add_argument(
        '--b1-biascorrect-stage',
        action='store',
        choices=['final', 'none', 'legacy'],
        default='final',
        help="Which stage to apply B1 bias correction. The default 'final' will "
        'apply it after all the data has been resampled to its final space. '
        "'none' will skip B1 bias correction and 'legacy' will behave consistent "
        'with qsiprep < 0.17.',
    )
    g_conf.add_argument(
        '--no-b0-harmonization',
        action='store_true',
        help='skip re-scaling dwi scans to have matching b=0 intensities',
    )
    g_conf.add_argument(
        '--denoise-after-combining',
        action='store_true',
        help='run ``dwidenoise`` after combining dwis, but before motion correction',
    )
    g_conf.add_argument(
        '--separate-all-dwis',
        action='store_true',
        help="don't attempt to combine dwis from multiple runs. Each will be "
        'processed separately.',
    )
    g_conf.add_argument(
        '--distortion-group-merge',
        action='store',
        choices=['concat', 'average', 'none'],
        default='none',
        help='How to combine images across distorted groups.\n'
        ' - concatenate: append images in the 4th dimension\n '
        ' - average: if a whole sequence was duplicated in both PE\n'
        '            directions, average the corrected images of the same\n'
        '            q-space coordinate\n'
        ' - none: Default. Keep distorted groups separate',
    )
    g_conf.add_argument(
        '--anatomical-template',
        required=False,
        action='store',
        choices=['MNI152NLin2009cAsym'],
        default='MNI152NLin2009cAsym',
        help='volume template space (default: MNI152NLin2009cAsym)',
    )
    g_conf.add_argument(
        '--output-resolution',
        action='store',
        required=True,
        type=float,
        help='the isotropic voxel size in mm the data will be resampled to '
        'after preprocessing. If set to a lower value than the original voxel '
        'size, your data will be upsampled using BSpline interpolation.',
    )

    g_coreg = parser.add_argument_group('Options for dwi-to-Anatomical coregistration')
    g_coreg.add_argument(
        '--b0-to-t1w-transform',
        action='store',
        default='Rigid',
        choices=['Rigid', 'Affine'],
        help='Degrees of freedom when registering b0 to anatomical images. '
        '6 degrees (rotation and translation) are used by default.',
    )
    g_coreg.add_argument(
        '--intramodal-template-iters',
        action='store',
        default=0,
        type=int,
        help='Number of iterations for finding the midpoint image '
        'from the b0 templates from all groups. Has no effect if there '
        'is only one group. If 0, all b0 templates are directly registered '
        'to the t1w image.',
    )
    g_coreg.add_argument(
        '--intramodal-template-transform',
        default='BSplineSyN',
        choices=['Rigid', 'Affine', 'BSplineSyN', 'SyN'],
        action='store',
        help='Transformation used for building the intramodal template.',
    )

    # FreeSurfer options
    g_fs = parser.add_argument_group('Specific options for FreeSurfer preprocessing')
    g_fs.add_argument(
        '--fs-license-file',
        metavar='PATH',
        type=Path,
        help='Path to FreeSurfer license key file. Get it (for free) by registering '
        'at https://surfer.nmr.mgh.harvard.edu/registration.html',
    )

    g_moco = parser.add_argument_group('Specific options for motion correction and coregistration')
    g_moco.add_argument(
        '--b0-motion-corr-to',
        action='store',
        default='iterative',
        choices=['iterative', 'first'],
        help='align to the "first" b0 volume or do an "iterative" registration'
        ' of all b0 images to their midpoint image (default: iterative)',
    )
    g_moco.add_argument(
        '--hmc-transform',
        action='store',
        default='Affine',
        choices=['Affine', 'Rigid'],
        help='transformation to be optimized during head motion correction (default: affine)',
    )
    g_moco.add_argument(
        '--hmc-model',
        action='store',
        default='eddy',
        choices=['none', '3dSHORE', 'eddy', 'tensor'],
        help='model used to generate target images for hmc. If "none" the '
        'non-b0 images will be warped using the same transform as their '
        'nearest b0 image. If "3dSHORE", SHORELine will be used. if "tensor", '
        'SHORELine iterations with a tensor model will be used',
    )
    g_moco.add_argument(
        '--eddy-config',
        action='store',
        help='path to a json file with settings for the call to eddy. If no '
        'json is specified, a default one will be used. The current default '
        'json can be found here: '
        'https://github.com/PennLINC/qsiprep/blob/master/qsiprep/data/eddy_params.json',
    )
    g_moco.add_argument(
        '--shoreline-iters',
        action='store',
        type=int,
        default=2,
        help='number of SHORELine iterations. (default: 2)',
    )

    # Fieldmap options
    g_fmap = parser.add_argument_group('Specific options for handling fieldmaps')
    g_fmap.add_argument(
        '--pepolar-method',
        action='store',
        default='TOPUP',
        choices=['TOPUP', 'DRBUDDI', 'TOPUP+DRBUDDI'],
        help='select which SDC method to use for PEPOLAR fieldmaps (default: TOPUP)',
    )
    g_fmap.add_argument(
        '--fmap-bspline',
        action='store_true',
        default=False,
        help='Fit a B-Spline field using least-squares (experimental)',
    )
    g_fmap.add_argument(
        '--fmap-no-demean',
        action='store_false',
        default=True,
        help='Do not remove median (within mask) from fieldmap',
    )

    # SyN-unwarp options
    g_syn = parser.add_argument_group('Specific options for SyN distortion correction')
    g_syn.add_argument(
        '--use-syn-sdc',
        nargs='?',
        choices=['warn', 'error'],
        action='store',
        const='error',
        default=False,
        help='Use fieldmap-less distortion correction based on anatomical image; '
        'if unable, error (default) or warn based on optional argument.',
    )
    g_syn.add_argument(
        '--force-syn',
        action='store_true',
        default=False,
        help='EXPERIMENTAL/TEMPORARY: Use SyN correction in addition to '
        'fieldmap correction, if available',
    )

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('--version', action='version', version=verstr)
    g_other.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='Increases log verbosity for each occurrence, debug level is -vvv',
    )
    g_other.add_argument(
        '-w',
        '--work-dir',
        action='store',
        type=Path,
        default=Path('work').absolute(),
        help='Path where intermediate results should be stored',
    )
    g_other.add_argument(
        '--resource-monitor',
        action='store_true',
        default=False,
        help="Enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        '--config-file',
        action='store',
        metavar='FILE',
        help='Use pre-generated configuration file. Values in file will be overridden '
        'by command-line arguments.',
    )
    g_other.add_argument(
        '--write-graph',
        action='store_true',
        default=False,
        help='Write workflow graph.',
    )
    g_other.add_argument(
        '--stop-on-first-crash',
        action='store_true',
        default=False,
        help='Force stopping on first crash, even if a work directory was specified.',
    )
    g_other.add_argument(
        '--notrack',
        action='store_true',
        default=False,
        help='Opt-out of sending tracking information of this run to '
        'the QSIPrep developers. This information helps to '
        'improve QSIPrep and provides an indicator of real '
        'world usage crucial for obtaining funding.',
    )
    g_other.add_argument(
        '--debug',
        action='store',
        nargs='+',
        choices=config.DEBUG_MODES + ('all',),
        help="Debug mode(s) to enable. 'all' is alias for all available modes.",
    )
    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging

    from bids.layout import Query

    # from niworkflows.utils.spaces import Reference, SpatialReferences

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    # Change anatomical_template based on infant parameter
    opts.anatomical_template = 'MNI152NLin2009cAsym'
    if opts.infant:
        config.loggers.cli.info(
            'Infant processing mode enabled. '
            "Inferring the subject's age and selecting the appropriate MNIInfant cohort."
        )
        opts.anatomical_template = 'MNIInfant'
        if opts.subject_anatomical_reference != 'sessionwise':
            config.loggers.cli.error(
                'Infant processing requires --subject-anatomical-reference sessionwise'
            )

    if opts.config_file:
        skip = {} if opts.reports_only else {'execution': ('run_uuid',)}
        config.load(opts.config_file, skip=skip, init=False)
        config.loggers.cli.info(f'Loaded previous configuration file {opts.config_file}')

    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts), init=['nipype'])

    if not config.execution.notrack:
        import importlib.util

        if importlib.util.find_spec('sentry_sdk') is None:
            config.execution.notrack = True
            config.loggers.cli.warning('Telemetry disabled because sentry_sdk is not installed.')
        else:
            config.loggers.cli.info(
                'Telemetry system to collect crashes and errors is enabled '
                '- thanks for your feedback! Use option ``--notrack`` to opt out.'
            )

    # Initialize --output-spaces if not defined
    # if config.execution.output_spaces is None:
    #     config.execution.output_spaces = SpatialReferences(
    #         [Reference("MNI152NLin2009cAsym", {"res": "native"})]
    #     )

    # Retrieve logging level
    build_log = config.loggers.cli

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        import yaml

        with open(opts.use_plugin) as f:
            plugin_settings = yaml.safe_load(f)
        _plugin = plugin_settings.get('plugin')
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get('plugin_args', {})
            config.nipype.nprocs = opts.nprocs or config.nipype.plugin_args.get(
                'n_procs', config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        build_log.warning(
            f'Per-process threads (--omp-nthreads={config.nipype.omp_nthreads}) exceed '
            f'total threads (--nthreads/--n-cpus={config.nipype.nprocs})'
        )

    # Validate the tricky options here
    if config.workflow.dwi_denoise_window != 'auto':
        if config.workflow.denoise_method == 'patch2self':
            config.loggers.cli.error(
                'The --dwi-denoise-window option is not used when --denoise-method=patch2self'
            )
        elif config.workflow.denoise_method == 'none':
            config.loggers.cli.warning(
                'The --dwi-denoise-window option is not used when --denoise-method=none'
            )

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Update the config with an empty dict to trigger initialization of all config
    # sections (we used `init=False` above).
    # This must be done after cleaning the work directory, or we could delete an
    # open SQLite database
    config.from_dict({})

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        rec_path = output_dir / 'derivatives' / f'qsiprep-{version.split("+")[0]}'
        parser.error(
            'The selected output folder is the same as the input BIDS folder. '
            f'Please modify the output path (suggestion: {rec_path}).'
        )

    if bids_dir in work_dir.parents:
        parser.error(
            'The selected working directory is a subdirectory of the input BIDS folder. '
            'Please modify the output path.'
        )

    # Validate inputs
    if not opts.skip_bids_validation:
        from ..utils.bids import validate_input_dir

        build_log.info(
            'Making sure the input data is BIDS compliant (warnings can be ignored in most cases).'
        )
        validate_input_dir(
            config.environment.exec_env,
            opts.bids_dir,
            opts.participant_label,
        )

    # Setup directories
    config.execution.log_dir = config.execution.output_dir / 'logs'
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            'One or more participant labels were not found in the BIDS directory: '
            f'{", ".join(missing_subjects)}.'
        )

    # Determine which sessions to process and group them
    processing_groups = []

    # Determine any session filters
    session_filters = config.execution.session_id or []
    # if config.execution.bids_filters is not None:
    #     for _, filters in config.execution.bids_filters:
    #         ses_filter = filters.get("session")
    #         if isinstance(ses_filter, str):
    #             session_filters.append(ses_filter)
    #         elif isinstance(ses_filter, list):
    #             session_filters.extend(ses_filter)

    # Examine the available sessions for each participant
    for subject_id in participant_label:
        # Find sessions with DWI data
        sessions = config.execution.layout.get_sessions(
            subject=subject_id,
            session=session_filters or Query.OPTIONAL,
            suffix=['dwi'],
        )

        # If there are no sessions, there is only one option:
        if not sessions:
            if config.workflow.subject_anatomical_reference == 'sessionwise':
                config.loggers.workflow.warning(
                    f'Subject {subject_id} had no sessions, '
                    'but --subject-anatomical-reference was set to "sessionwise". '
                    'Outputs will NOT appear in a session directory for '
                    f'{subject_id}.',
                )

            processing_groups.append([subject_id, []])
            continue

        if config.workflow.subject_anatomical_reference == 'sessionwise':
            for session in sessions:
                processing_groups.append([subject_id, [session]])
        else:
            # We can now use sessions that have anatomical data, but no DWI
            sessions = config.execution.layout.get_sessions(
                subject=subject_id,
                session=session_filters or Query.OPTIONAL,
                suffix=['dwi', 'T1w', 'T2w'],
            )
            processing_groups.append([subject_id, sessions])

    # Make a nicely formatted message showing what we will process
    def pretty_group(group_num, processing_group):
        participant_label, ses_labels = processing_group
        if ses_labels:
            session_txt = ', '.join(map(str, ses_labels))
        else:
            session_txt = 'No session level'

        return f'{group_num}\t{participant_label}\t{session_txt}'

    processing_msg = '\nGroup\tSubject\tSessions\n' + '\n'.join(
        [pretty_group(gnum, group) for gnum, group in enumerate(processing_groups)]
    )
    config.loggers.workflow.info(processing_msg)

    config.execution.participant_label = sorted(participant_label)
    config.execution.processing_list = processing_groups
