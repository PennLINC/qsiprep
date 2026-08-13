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
from ..utils.gpu import GPU_ALIASES, GPU_TASKS
from ..utils.misc import parse_denoise_method

B0_TO_ANAT_TRANSFORM_DEFAULT = 'Rigid'
"""Default for ``--b0-to-anat-transform``.

Applied after parsing rather than by argparse, because the option is declared with
``default=SUPPRESS`` so that its mutual exclusion with the deprecated
``--b0-to-t1w-transform`` is checked reliably.
"""


def _build_parser(**kwargs):
    """Build parser object.

    ``kwargs`` are passed to ``argparse.ArgumentParser`` (mainly useful for debugging).
    """
    from argparse import (
        SUPPRESS,
        Action,
        ArgumentDefaultsHelpFormatter,
        ArgumentParser,
    )
    from functools import partial
    from pathlib import Path

    from packaging.version import Version

    # Deprecated options: {option string: (version it is removed in, what happens instead)}
    deprecations = {
        '--dwi-only': ('27.0.0', 'Enabling `--anat-modality none` instead.'),
        '--dwi-no-biascorr': ('27.0.0', 'Enabling `--b1-biascorrect-stage none` instead.'),
        '--longitudinal': (
            '27.0.0',
            'Enabling `--subject-anatomical-reference unbiased` instead.',
        ),
        '--prefer-dedicated-fmaps': (
            '27.0.0',
            'It has no effect. Which fieldmap is applied to which DWI series is determined '
            'by the fieldmaps\' "B0FieldIdentifier"/"B0FieldSource" (or "IntendedFor") '
            'metadata.',
        ),
        '--b0-motion-corr-to': (
            '27.0.0',
            'Later versions will always use the "iterative" approach.',
        ),
        '--b0-to-t1w-transform': ('27.0.0', 'Please use `--b0-to-anat-transform` instead.'),
    }

    # Deprecated flags that enable their replacement automatically:
    # {option string: (replacement option, its namespace attribute, the value it is set to)}
    forwarded_deprecations = {
        '--dwi-only': ('--anat-modality', 'anat_modality', 'none'),
        '--dwi-no-biascorr': ('--b1-biascorrect-stage', 'b1_biascorrect_stage', 'none'),
        '--longitudinal': (
            '--subject-anatomical-reference',
            'subject_anatomical_reference',
            'unbiased',
        ),
    }

    def _warn_deprecated(option_string):
        removed_in, detail = deprecations[option_string]
        print(
            f'{option_string} has been deprecated and will be removed in {removed_in}. {detail}',
            file=sys.stderr,
        )

    class DeprecatedAction(Action):
        """Warn that a deprecated option is ignored, and keep it out of the namespace.

        Declared with ``default=SUPPRESS`` so the dest never reaches the config object.
        """

        def __init__(self, option_strings, dest, nargs=0, **kwargs):
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            _warn_deprecated(option_string or self.option_strings[0])

    class DeprecatedForwardAction(Action):
        """Warn about a deprecated flag, and record that its replacement must be enabled.

        The replacement is applied after the whole command line has been read, so the
        outcome does not depend on the order the options were given in.
        """

        def __init__(self, option_strings, dest, nargs=0, **kwargs):
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            option_string = option_string or self.option_strings[0]
            _warn_deprecated(option_string)
            pending = getattr(namespace, '_forwarded_deprecations', [])
            namespace._forwarded_deprecations = [*pending, option_string]

    class DeprecatedStoreAction(Action):
        """Warn about a deprecated option, then store its value like ``store`` would."""

        def __call__(self, parser, namespace, values, option_string=None):
            _warn_deprecated(option_string or self.option_strings[0])
            setattr(namespace, self.dest, values)

    class DeprecationForwardingParser(ArgumentParser):
        """Enables the replacements for any deprecated options that were given."""

        def parse_known_args(self, args=None, namespace=None):
            namespace, extras = super().parse_known_args(args, namespace)

            for option_string in getattr(namespace, '_forwarded_deprecations', []):
                replacement, dest, value = forwarded_deprecations[option_string]
                current = getattr(namespace, dest)
                if current not in (self.get_default(dest), value):
                    self.error(
                        f'{option_string} enables `{replacement} {value}`, which conflicts '
                        f'with the requested `{replacement} {current}`.'
                    )
                setattr(namespace, dest, value)
            if hasattr(namespace, '_forwarded_deprecations'):
                del namespace._forwarded_deprecations

            # --b0-to-t1w-transform was renamed; the two are mutually exclusive, so at
            # most one of them is set here.
            if hasattr(namespace, 'b0_to_t1w_transform'):
                namespace.b0_to_anat_transform = namespace.b0_to_t1w_transform
                del namespace.b0_to_t1w_transform
            if not hasattr(namespace, 'b0_to_anat_transform'):
                namespace.b0_to_anat_transform = B0_TO_ANAT_TRANSFORM_DEFAULT

            return namespace, extras

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
        """Ensure an argument is an odd integer >= 3 or 'auto'."""
        if value.lower() == 'auto':
            return 'auto'
        try:
            value = int(value)
        except ValueError as exc:
            raise parser.error('Argument must be an integer or "auto".') from exc

        if value < 3:
            raise parser.error('Argument must be an odd integer >= 3.')

        if value % 2 == 0:
            raise parser.error('Argument must be an odd integer >= 3.')

        return value

    def _denoise_method(value, parser):
        try:
            parse_denoise_method(value)
        except ValueError as exc:
            parser.error(f'Invalid --denoise-method specification: {exc}')
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

    parser = DeprecationForwardingParser(
        description=f'{verstr}: q-Space Image Preprocessing workflows',
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    IntOrAuto = partial(_int_or_auto, parser=parser)
    DenoiseMethod = partial(_denoise_method, parser=parser)
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
        action=DeprecatedForwardAction,
        default=SUPPRESS,
        help='DEPRECATED: this flag now enables `--anat-modality none`. Use that instead.',
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
    g_subset.add_argument(
        '--report-output-level',
        action='store',
        choices=['auto', 'root', 'subject', 'session'],
        default='auto',
        help='Where should the HTML reports be written? '
        '"root" will write them to the output directory. '
        '"subject" will write them into each subject\'s directory. '
        '"session" will write them into each session\'s directory. '
        'The default is "auto", which is "session" when '
        '--subject-anatomical-reference is "sessionwise" and "root" otherwise. '
        'Reports that cover more than one session, or data without a session level, '
        'are written to the subject level instead of the session level, with a warning.',
    )

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore',
        required=False,
        action='store',
        nargs='+',
        default=[],
        choices=['fieldmaps', 't2w', 'phase'],
        help=(
            'Ignore selected aspects of the input dataset to disable corresponding '
            'parts of the workflow (a space delimited list). '
            '"fieldmaps" will completely disable susceptibility distortion correction, '
            'whether using field maps or reverse phase-encoded dMRI runs.'
        ),
    )
    g_conf.add_argument(
        '--gpu',
        required=False,
        action='store',
        nargs='+',
        # None (not given) is distinct from ["none"] (explicitly off): when the
        # flag is absent, a legacy "use_cuda" in --eddy-config/--diffprep-config
        # still decides, so those runs do not silently drop to CPU.
        default=None,
        choices=sorted(GPU_TASKS) + list(GPU_ALIASES),
        help=(
            'Run selected tasks on the GPU (a space delimited list). GPU memory is '
            'usually the binding constraint rather than the pipeline, so tasks are '
            'selected individually: an 8 GB card typically runs "eddy", "diffprep" '
            'and "drbuddi" but not "synthstrip" or "synthseg". "all" enables every '
            'task, "none" (the default) disables all of them. The GPU must also be '
            'exposed to the container ("docker run --gpus all" / '
            '"apptainer run --nv"). NOTE: GPU builds are not numerically identical '
            'to their CPU counterparts, so this changes results, not just runtime. '
            'When given, this overrides "use_cuda" in --eddy-config / '
            '--diffprep-config; when omitted entirely, those keys still apply.'
        ),
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
        action=DeprecatedForwardAction,
        default=SUPPRESS,
        help=(
            'DEPRECATED: this flag now enables `--subject-anatomical-reference unbiased`. '
            'Use that instead.'
        ),
    )
    g_conf.add_argument(
        '--subject-anatomical-reference',
        choices=['first-lex', 'unbiased', 'sessionwise', 'first-alphabetically'],
        default='first-lex',
        help=(
            'How to define subject-specific anatomical space. '
            'sessionwise will produce one anatomical space per session. '
            'The others combine anatomical data across sessions to define '
            'one anatomical space per subject. '
            'The "first-alphabetically" option is deprecated in favor of "first-lex".'
        ),
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
            'This argument only applies to the "dwidenoise" denoising method, '
            'where the "auto" option will calculate a window size '
            'based on the number of volumes according to the method described by the '
            'dwidenoise documentation. '
            'It is not used by the "patch2self" or "dwidenoise2" methods: dwidenoise2 sizes '
            'its patches per iteration from its multi-resolution schedule, which is selected '
            'with "dwidenoise2;schedule:<name>" instead.'
        ),
    )
    g_conf.add_argument(
        '--denoise-method',
        action='store',
        type=DenoiseMethod,
        default='dwidenoise',
        help=(
            'Image-based denoising method: "dwidenoise" (MRtrix), "dwidenoise2", '
            '"patch2self" (DIPY), or "none".\n'
            'dwidenoise2 parameters may follow the method as semicolon-delimited '
            'name:value pairs, for example '
            '"dwidenoise2;demodulate:linear;decomposition:bdcsvd".'
        ),
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
        action=DeprecatedForwardAction,
        default=SUPPRESS,
        help='DEPRECATED: this flag now enables `--b1-biascorrect-stage none`. Use that instead.',
    )
    g_conf.add_argument(
        '--anat-biascorrect',
        action='store',
        choices=['n4', 'auto', 'none'],
        default='n4',
        help=(
            'Whether to run N4 bias field correction on ANATOMICAL images. '
            'Note this is separate from --b1-biascorrect-stage, which only governs '
            'the DWIs. '
            '"n4" (default) always runs it; scanner-side intensity normalization '
            '(e.g. Siemens NORM) does not remove the need for it. '
            '"none" never runs it. '
            '"auto" skips it when the BIDS ImageType metadata contains "NORM", '
            'which is how Siemens and others flag console-applied normalization.'
        ),
    )
    g_conf.add_argument(
        '--b1-biascorrect-stage',
        action='store',
        choices=['final', 'none', 'legacy'],
        default='final',
        help=(
            'Which stage to apply B1 bias correction. '
            'The default "final" will apply it after all the data has been resampled '
            'to its final space. '
            '"none" will skip B1 bias correction and '
            '"legacy" will behave consistent with qsiprep < 0.17. '
            'For prescan-normalized data, we recommend using "none", '
            'as bias correction may introduce artifacts on normalized data.'
        ),
    )
    g_conf.add_argument(
        '--no-b0-harmonization',
        action='store_true',
        help='skip re-scaling dwi scans to have matching b=0 intensities',
    )
    g_conf.add_argument(
        '--denoise-after-combining',
        action='store_true',
        help='run denoising after combining dwis, but before motion correction',
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
        help="""\
How to combine images across distorted groups.
 - concat: append images in the 4th dimension
 - average: if a whole sequence was duplicated in both PE
            directions, average the corrected images of the same
            q-space coordinate
 - none: Default. Keep distorted groups separate
""",
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
    # Both are declared with default=SUPPRESS so that "was this given?" is just
    # hasattr. argparse's own mutual-exclusion check compares the parsed value against
    # the default by identity, which would miss `--b0-to-anat-transform Rigid` when
    # 'Rigid' happens to be interned; against SUPPRESS it always fires. The default is
    # applied in DeprecationForwardingParser instead.
    g_b0_to_anat = g_coreg.add_mutually_exclusive_group()
    g_b0_to_anat.add_argument(
        '--b0-to-anat-transform',
        action='store',
        default=SUPPRESS,
        choices=['Rigid', 'Affine'],
        help='Degrees of freedom when registering b0 to anatomical images: '
        '6 (Rigid, rotation and translation) or 12 (Affine). '
        f'(default: {B0_TO_ANAT_TRANSFORM_DEFAULT})',
    )
    g_b0_to_anat.add_argument(
        '--b0-to-t1w-transform',
        action=DeprecatedStoreAction,
        default=SUPPRESS,
        choices=['Rigid', 'Affine'],
        help='DEPRECATED: renamed to `--b0-to-anat-transform`, which this option now sets. '
        'Use that instead.',
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
        action=DeprecatedStoreAction,
        default='iterative',
        choices=['iterative', 'first'],
        help='DEPRECATED: align to the "first" b0 volume or do an "iterative" registration '
        'of all b0 images to their midpoint image. '
        'Later versions will always use "iterative".',
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
        choices=[
            'none',
            '3dSHORE',
            'eddy',
            'tensor',
            'diffprep_motion',
            'diffprep_quadratic',
            'diffprep_cubic',
        ],
        help='model used to generate target images for hmc. If "none" the '
        'non-b0 images will be warped using the same transform as their '
        'nearest b0 image. If "3dSHORE", SHORELine will be used. if "tensor", '
        'SHORELine iterations with a tensor model will be used. The '
        '"diffprep_*" options run TORTOISE DIFFPREP: "diffprep_motion" '
        'corrects rigid head motion only, "diffprep_quadratic" adds '
        '24-parameter quadratic eddy-current correction (recommended for '
        'non-shelled / CS-DSI schemes), "diffprep_cubic" adds cubic eddy '
        'correction. DIFFPREP works on arbitrary q-space (no shells '
        'required).',
    )
    g_moco.add_argument(
        '--eddy-config',
        action='store',
        help='path to a json file with settings for the call to eddy. If no '
        'json is specified, a default one will be used. The current default '
        'json can be found here: '
        'https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/eddy_params.json',
    )
    g_moco.add_argument(
        '--diffprep-config',
        action='store',
        help='path to a json file with settings for the call to TORTOISE '
        'DIFFPREP (used only when --hmc-model is one of the diffprep_* '
        'options). If no json is specified, a default one will be used. The '
        'current default can be found here: '
        'https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/diffprep_params.json',
    )
    g_moco.add_argument(
        '--tortoise-gpu-cpu-ratio',
        action='store',
        type=int,
        default=None,
        help=(
            'How many volumes DIFFPREP gives the GPU per pass, against one per CPU '
            'thread, during motion and eddy correction. TORTOISE does not move the '
            'series onto the GPU the way eddy_cuda does: it treats the GPU as one '
            'more worker, so the number of passes is '
            'ceil(nvolumes / (ngpus * ratio + omp-nthreads - ngpus)). '
            'It describes the machine, not the data -- roughly how many volumes the '
            'GPU gets through while one CPU core does one. Only worth setting when '
            'the GPU is fast relative to the core count, since its influence falls '
            'as --omp-nthreads rises (about 68%% of volumes at 8 cores, 19%% at 64). '
            'Requires the patched TORTOISE. Unset leaves TORTOISE at its default of 15.'
        ),
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
        '--prefer-dedicated-fmaps',
        action=DeprecatedAction,
        default=SUPPRESS,
        help='DEPRECATED: this flag has no effect. Which fieldmap is applied to which DWI '
        'series is determined by the fieldmaps\' "B0FieldIdentifier"/"B0FieldSource" '
        '(or "IntendedFor") metadata.',
    )
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


def check_denoise_window(denoise_method, dwi_denoise_window):
    """Report a ``--dwi-denoise-window`` that the selected denoising method will ignore.

    Only ``dwidenoise`` takes a window size. Leaving the others to silently ignore it would
    hide a request that never took effect.
    """
    if dwi_denoise_window == 'auto':
        # The default, so an unused value is not a sign that anything was misunderstood
        return

    if denoise_method == 'patch2self':
        config.loggers.cli.error(
            'The --dwi-denoise-window option is not used when --denoise-method=patch2self'
        )
    elif denoise_method == 'dwidenoise2':
        config.loggers.cli.warning(
            'The --dwi-denoise-window option is not used when --denoise-method=dwidenoise2. '
            'dwidenoise2 sizes its patches per iteration from its multi-resolution schedule, '
            'which can be selected with "dwidenoise2;schedule:<name>" instead.'
        )
    elif denoise_method == 'none':
        config.loggers.cli.warning(
            'The --dwi-denoise-window option is not used when --denoise-method=none'
        )


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging

    from bids.layout import Query

    # from niworkflows.utils.spaces import Reference, SpatialReferences

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    # Warn about deprecated options
    if opts.subject_anatomical_reference == 'first-alphabetically':
        config.loggers.cli.warning(
            '--subject-anatomical-reference=first-alphabetically has been deprecated '
            'and will be removed in a later version. '
            'Please use --subject-anatomical-reference=first-lex instead.'
        )
        opts.subject_anatomical_reference = 'first-lex'

    # Reports follow the anatomical processing level unless the user asked for a specific one
    if opts.report_output_level == 'auto':
        opts.report_output_level = (
            'session' if opts.subject_anatomical_reference == 'sessionwise' else 'root'
        )

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

    if opts.eddy_config:
        from ..utils.misc import validate_eddy_config

        validate_eddy_config(opts.eddy_config)

    if opts.diffprep_config:
        from ..utils.misc import validate_diffprep_config

        validate_diffprep_config(opts.diffprep_config)

    if opts.gpu:
        from ..utils.gpu import check_gpu_available

        # Raises if no CUDA device is visible or a GPU build is missing. Doing
        # this here costs seconds; discovering it inside a node costs the whole
        # anatomical workflow.
        check_gpu_available(opts.gpu)

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
    denoise_method, denoise_params = parse_denoise_method(config.workflow.denoise_method)
    check_denoise_window(denoise_method, config.workflow.dwi_denoise_window)
    if (
        config.workflow.denoise_after_combining
        and denoise_params.get('demodulate', 'none') != 'none'
    ):
        # Temporary workaround for a bug in dwidenoise2: the concatenated series
        # cannot be denoised with phase data.
        parser.error(
            '--denoise-after-combining cannot be used with phase demodulation '
            f'("demodulate:{denoise_params["demodulate"]}"). '
            'Remove the demodulate parameter and use "--ignore phase" to denoise '
            'the magnitude data only.'
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
