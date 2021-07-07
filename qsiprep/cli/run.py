#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSI workflow
=====
"""
import warnings
import os
import re
import os.path as op
from pathlib import Path
import logging
import sys
import gc
import uuid
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count
from time import strftime
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.addLevelName(25,
                     'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def check_deps(workflow):
    from nipype.utils.filemanip import which
    return sorted((node.interface.__class__.__name__, node.interface._cmd)
                  for node in workflow._get_all_nodes()
                  if (hasattr(node.interface, '_cmd')
                      and which(node.interface._cmd.split()[0]) is None))


def _filter_pybids_none_any(dct):
    import bids
    return {
        k: bids.layout.Query.NONE
        if v is None
        else (bids.layout.Query.ANY if v == "*" else v)
        for k, v in dct.items()
    }


def _bids_filter(value):
    from json import loads
    from bids.layout import Query

    if value and Path(value).exists():
        try:
            filters = loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
        except Exception as e:
            raise Exception("Unable to parse BIDS filter file. Check that it is "
                            "valid JSON.")
    else:
        raise Exception("Unable to load BIDS filter file " + value)

    # unserialize pybids Query enum values
    for acq, _filters in filters.items():
        filters[acq] = {
            k: getattr(Query, v[7:-4])
            if not isinstance(v, Query) and "Query" in v
            else v
            for k, v in _filters.items()
        }
    return filters


def get_parser():
    """Build parser object"""
    from ..__about__ import __version__

    verstr = 'qsiprep v{}'.format(__version__)

    parser = ArgumentParser(
        description='qsiprep: q-Space Image Preprocessing workflows',
        formatter_class=ArgumentDefaultsHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir',
                        type=Path,
                        action='store',
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders '
                        'should be found at the top level in this folder).')
    parser.add_argument('output_dir',
                        action='store',
                        type=Path,
                        help='the output path for the outcomes of preprocessing and visual'
                        ' reports')
    parser.add_argument('analysis_level',
                        choices=['participant'],
                        action='store',
                        help='processing stage to be run, only "participant" in the case of '
                        'qsiprep (see BIDS-Apps specification).')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--skip_bids_validation', '--skip-bids-validation', action='store_true',
                        default=False,
                        help='assume the input dataset is BIDS compliant and skip the validation')
    g_bids.add_argument(
        '--participant_label',
        '--participant-label',
        action='store',
        nargs='+',
        help='a space delimited list of participant identifiers or a single '
        'identifier (the sub- prefix can be removed)')
    g_bids.add_argument('--acquisition_type', '--acquisition_type', action='store',
                        help='select a specific acquisition type to be processed')
    g_bids.add_argument('--bids-database-dir', '--bids_database_dir',
                        help="path to a saved BIDS database directory",
                        type=Path,
                        action='store')
    g_bids.add_argument(
        "--bids-filter-file",
        dest="bids_filters",
        action="store",
        type=_bids_filter,
        metavar="FILE",
        help="a JSON file describing custom BIDS input filters using PyBIDS. "
        "For further details, please check out "
        "https://fmriprep.readthedocs.io/en/latest/faq.html#"
        "how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep")

    # arguments for reconstructing QSI data
    g_ireports = parser.add_argument_group('Options for interactive report outputs')
    g_ireports.add_argument(
        '--interactive-reports-only', '--interactive_reports_only',
        action='store_true',
        default=False,
        help='create interactive report json files on already preprocessed data.'
    )

    # arguments for reconstructing QSI data
    g_recon = parser.add_argument_group('Options for reconstructing qsiprep outputs')
    g_recon.add_argument(
        '--recon-only', '--recon_only',
        action='store_true',
        default=False,
        help='run only reconstruction, assumes preprocessing has already completed.'
    )
    g_recon.add_argument(
        '--recon-spec', '--recon_spec',
        action='store',
        type=str,
        help='json file specifying a reconstruction pipeline to be run after preprocessing'
    )
    g_recon.add_argument(
        '--recon-input', '--recon_input',
        action='store',
        type=os.path.abspath,
        help='use this directory as inputs to qsirecon. This option skips qsiprep.'
    )

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument(
        '--nthreads',
        '--n_cpus',
        '--n-cpus',
        action='store',
        type=int,
        help='maximum number of threads across all processes')
    g_perfm.add_argument(
        '--omp-nthreads', '--omp_nthreads',
        action='store',
        type=int,
        default=0,
        help='maximum number of threads per-process')
    g_perfm.add_argument(
        '--mem_mb',
        '--mem-mb',
        action='store',
        default=0,
        type=int,
        help='upper bound memory limit for qsiprep processes')
    g_perfm.add_argument(
        '--low-mem', '--low_mem',
        action='store_true',
        help='attempt to reduce memory usage (will increase disk usage '
        'in working directory)')
    g_perfm.add_argument(
        '--use-plugin', '--use_plugin',
        action='store',
        default=None,
        help='nipype plugin configuration file')
    g_perfm.add_argument(
        '--anat-only', '--anat_only',
        action='store_true',
        help='run anatomical workflows only')
    g_perfm.add_argument(
        '--dwi-only', '--dwi_only',
        action='store_true',
        help='ignore anatomical (T1w/T2w) data and process DWIs only')
    g_perfm.add_argument(
        '--infant',
        action='store_true',
        help='configure pipelines to process infant brains')
    g_perfm.add_argument(
        '--boilerplate', action='store_true', help='generate boilerplate only')
    g_perfm.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurence, debug level is -vvv")

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore',
        required=False,
        action='store',
        nargs="+",
        default=[],
        choices=['fieldmaps', 'sbref'],
        help='ignore selected aspects of the input dataset to disable '
        'corresponding parts of the workflow (a space delimited list)')
    g_conf.add_argument(
        '--longitudinal',
        action='store_true',
        help='treat dataset as longitudinal - may increase runtime')
    g_conf.add_argument(
        '--b0-threshold', '--b0_threshold',
        action='store',
        type=int,
        default=100,
        help='any value in the .bval file less than this will be considered '
        'a b=0 image. Current default threshold = 100; this threshold can be '
        'lowered or increased. Note, setting this too high can result in inaccurate results.')
    g_conf.add_argument(
        '--dwi_denoise_window', '--dwi-denoise-window',
        action='store',
        default='auto',
        help='window size in voxels for image-based denoising, integer or "auto".'
             'If "auto", 5 will be used for dwidenoise and auto-configured for '
             'patch2self based on the number of b>0 images.')
    g_conf.add_argument(
        '--denoise-method', '--denoise_method',
        action='store',
        choices=['dwidenoise', 'patch2self', 'none'],
        default='dwidenoise',
        help='Image-based denoising method. Either "dwidenoise" (MRtrix), '
             '"patch2self" (DIPY) or none. (default: dwidenoise)')
    g_conf.add_argument(
        '--unringing-method', '--unringing_method',
        action='store',
        choices=['none', 'mrdegibbs'],
        help='Method for Gibbs-ringing removal.\n - none: no action\n - mrdegibbs: '
             'use mrdegibbs from mrtrix3 (default: none).')
    g_conf.add_argument(
        '--dwi-no-biascorr', '--dwi_no_biascorr',
        action='store_true',
        help='skip b0-based dwi spatial bias correction')
    g_conf.add_argument(
        '--no-b0-harmonization', '--no_b0_harmonization',
        action='store_true',
        help='skip re-scaling dwi scans to have matching b=0 intensities')
    g_conf.add_argument(
        '--denoise-after-combining', '--denoise_after_combining',
        action='store_true',
        help='run ``dwidenoise`` after combining dwis. Requires '
             '``--combine-all-dwis``')
    g_conf.add_argument(
        '--separate_all_dwis', '--separate-all-dwis',
        action='store_true',
        help="don't attempt to combine dwis from multiple runs. Each will be "
        'processed separately.')
    g_conf.add_argument(
        '--distortion-group-merge', '--distortion_group_merge',
        action='store',
        choices=['concat', 'average', 'none'],
        default='none',
        help='How to combine images across distorted groups.\n'
        ' - concatenate: append images in the 4th dimension\n '
        ' - average: if a whole sequence was duplicated in both PE\n'
        '            directions, average the corrected images of the same\n'
        '            q-space coordinate\n'
        ' - none: Default. Keep distorted groups separate')
    g_conf.add_argument(
        '--write-local-bvecs', '--write_local_bvecs',
        action='store_true',
        default=False,
        help='write a series of voxelwise bvecs, relevant if '
        'writing preprocessed dwis to template space')
    g_conf.add_argument(
        '--output-space', '--output_space',
        action='store',
        choices=['T1w', 'template'],
        nargs='+',
        default=['T1w'],
        help='volume and surface spaces to resample dwis into\n'
        ' - T1w: subject anatomical volume\n'
        ' - template: deprecated. Will be ignored\n')
    g_conf.add_argument(
        '--template',
        required=False,
        action='store',
        choices=['MNI152NLin2009cAsym'],
        default='MNI152NLin2009cAsym',
        help='volume template space (default: MNI152NLin2009cAsym)')
    g_conf.add_argument(
        '--output-resolution', '--output_resolution',
        action='store',
        # required when not recon-only (which can be specified in sysargs 2 ways)
        required=not any(rcn in sys.argv for rcn in ['--recon-only', '--recon_only']),
        type=float,
        help='the isotropic voxel size in mm the data will be resampled to '
        'after preprocessing. If set to a lower value than the original voxel '
        'size, your data will be upsampled using BSpline interpolation.')

    g_coreg = parser.add_argument_group('Options for dwi-to-T1w coregistration')
    g_coreg.add_argument(
        '--b0-to-t1w-transform', '--b0_to_t1w_transform',
        action='store',
        default="Rigid",
        choices=["Rigid", "Affine"],
        help='Degrees of freedom when registering b0 to T1w images. '
        '6 degrees (rotation and translation) are used by default.')
    g_coreg.add_argument(
        '--intramodal-template-iters', '--intramodal_template_iters',
        action='store',
        default=0,
        type=int,
        help='Number of iterations for finding the midpoint image '
        'from the b0 templates from all groups. Has no effect if there '
        'is only one group. If 0, all b0 templates are directly registered '
        'to the t1w image.')
    g_coreg.add_argument(
        '--intramodal-template-transform', '--intramodal_template_transform',
        default='BSplineSyN',
        choices=['Rigid', 'Affine', 'BSplineSyN', 'SyN'],
        action='store',
        help='Transformation used for building the intramodal template.')

    g_moco = parser.add_argument_group(
        'Specific options for motion correction and coregistration')
    g_moco.add_argument(
        '--b0-motion-corr-to', '--bo_motion_corr_to',
        action='store',
        default='iterative',
        choices=['iterative', 'first'],
        help='align to the "first" b0 volume or do an "iterative" registration'
        ' of all b0 images to their midpoint image (default: iterative)')
    g_moco.add_argument(
        '--hmc-transform', '--hmc_transform',
        action='store',
        default='Affine',
        choices=['Affine', 'Rigid'],
        help='transformation to be optimized during head motion correction '
        '(default: affine)')
    g_moco.add_argument(
        '--hmc_model', '--hmc-model',
        action='store',
        default='eddy',
        choices=['none', '3dSHORE', 'eddy'],
        help='model used to generate target images for hmc. If "none" the '
        'non-b0 images will be warped using the same transform as their '
        'nearest b0 image. If "3dSHORE", SHORELine will be used. If '
        '"eddy_ingress", the dwis are assumed to have been run through '
        'fsls eddy. ')
    g_moco.add_argument(
        '--eddy-config', '--eddy_config',
        action='store',
        help='path to a json file with settings for the call to eddy. If no '
        'json is specified, a default one will be used. The current default '
        'json can be found here: '
        'https://github.com/PennBBL/qsiprep/blob/master/qsiprep/data/eddy_params.json')
    g_moco.add_argument(
        '--shoreline_iters', '--shoreline-iters',
        action='store',
        type=int,
        default=2,
        help='number of SHORELine iterations. (default: 2)')
    g_moco.add_argument(
        '--impute-slice-threshold', '--impute_slice_threshold',
        action='store',
        default=0,
        type=float,
        help='impute data in slices that are this many SDs from expected. '
        'If 0 (default), no slices will be imputed')

    # ANTs options
    g_ants = parser.add_argument_group(
        'Specific options for ANTs registrations')
    g_ants.add_argument(
        '--skull-strip-template', '--skull_strip_template',
        action='store',
        default='OASIS',
        choices=['OASIS', 'NKI'],
        help='select ANTs skull-stripping template (default: OASIS)')
    g_ants.add_argument(
        '--skull-strip-fixed-seed', '--skull_strip_fixed_seed',
        action='store_true',
        help='do not use a random seed for skull-stripping - will ensure '
        'run-to-run replicability when used with --omp-nthreads 1')
    g_ants.add_argument(
        '--skip-t1-based-spatial-normalization', '--skip_t1_based_spatial_normalization',
        action='store_true',
        default=False,
        help='skip running the t1w-based normalization to template space. '
        'Default is to run the normalization.')

    # FreeSurfer options
    g_fs = parser.add_argument_group('Specific options for FreeSurfer preprocessing')
    g_fs.add_argument(
        '--fs-license-file', '--fs_license_file', metavar='PATH', type=Path,
        help='Path to FreeSurfer license key file. Get it (for free) by registering '
        'at https://surfer.nmr.mgh.harvard.edu/registration.html')
    g_fs.add_argument(
        '--do-reconall', '--do_reconall', action='store_true',
        help='Run the FreeSurfer recon-all pipeline')

    # Fieldmap options
    g_fmap = parser.add_argument_group(
        'Specific options for handling fieldmaps')
    g_fmap.add_argument(
        '--prefer_dedicated_fmaps', '--prefer-dedicated-fmaps',
        action='store_true',
        default=False,
        help='forces unwarping to use files from the fmap directory instead '
        'of using an RPEdir scan from the same session.')
    g_fmap.add_argument(
        '--fmap-bspline', '--fmap_bspline',
        action='store_true',
        default=False,
        help='fit a B-Spline field using least-squares (experimental)')
    g_fmap.add_argument(
        '--fmap-no-demean', '--fmap_no_demean',
        action='store_false',
        default=True,
        help='do not remove median (within mask) from fieldmap (default: True)')

    # SyN-unwarp options
    g_syn = parser.add_argument_group(
        'Specific options for SyN distortion correction')
    g_syn.add_argument(
        '--use-syn-sdc', '--use_syn_sdc',
        action='store_true',
        default=False,
        help='EXPERIMENTAL: Use fieldmap-free distortion correction. To use '
        'this option, "template" must be passed to --output-space')
    g_syn.add_argument(
        '--force-syn', '--force_syn',
        action='store_true',
        default=False,
        help='EXPERIMENTAL/TEMPORARY: Use SyN correction in addition to '
        'fieldmap correction, if available')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument(
        '-w',
        '--work-dir', '--work_dir',
        type=Path,
        action='store',
        help='path where intermediate results should be stored')
    g_other.add_argument(
        '--resource-monitor', '--resource_monitor',
        action='store_true',
        default=False,
        help='enable Nipype\'s resource monitoring to keep track of memory '
        'and CPU usage')
    g_other.add_argument(
        '--reports-only', '--reports_only',
        action='store_true',
        default=False,
        help='only generate reports, don\'t run workflows. This will only '
        'rerun report aggregation, not reportlet generation for specific '
        'nodes.')
    g_other.add_argument(
        '--run-uuid', '--run_uuid',
        action='store',
        default=None,
        help='Specify UUID of previous run, to include error logs in report. '
        'No effect without --reports-only.')
    g_other.add_argument(
        '--write-graph', '--write_graph',
        action='store_true',
        default=False,
        help='Write workflow graph.')
    g_other.add_argument(
        '--stop-on-first-crash', '--stop_on_first_crash',
        action='store_true',
        default=False,
        help='Force stopping on first crash, even if a work directory'
        ' was specified.')
    g_other.add_argument(
        '--notrack',
        action='store_true',
        default=False,
        help='Opt-out of sending tracking information of this run to '
        'the qsiprep developers. This information helps to '
        'improve qsiprep and provides an indicator of real '
        'world usage crucial for obtaining funding.')
    g_other.add_argument(
        '--sloppy',
        action='store_true',
        default=False,
        help='Use low-quality tools for speed - TESTING ONLY')

    return parser


def validate_bids(opts):
    """Validate bids unless opts say otherwise"""
    from ..utils.bids import validate_input_dir

    # Validate inputs
    if not (opts.recon_only or opts.skip_bids_validation):
        print("Making sure the input data is BIDS compliant (warnings can be ignored in most "
              "cases).")
        validate_input_dir(os.name, opts.bids_dir, opts.participant_label)
        return True
    return False


def set_freesurfer_license(opts):
    """Set FS_LICENSE environment variable"""
    # FreeSurfer license
    # if qsiprep's current directory has a license.txt file, this will use that
    default_license = str(Path(os.getenv('FREESURFER_HOME', '')) / 'license.txt')
    # Precedence: --fs-license-file, $FS_LICENSE, default_license
    license_file = opts.fs_license_file or Path(os.getenv('FS_LICENSE', default_license))
    if not license_file.exists():
        raise RuntimeError("""\
ERROR: a valid license file is required for FreeSurfer to run. fMRIPrep looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html""")
    os.environ['FS_LICENSE'] = str(license_file.resolve())
    return os.environ['FS_LICENSE']


def main():
    """Entry point"""
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    from ..viz.reports import generate_reports
    from ..utils.bids import write_derivative_description

    try:
        set_start_method('forkserver')
    except RuntimeError:
        pass

    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    exec_env = os.name

    # special variable set in the container
    if os.getenv('IS_DOCKER_8395080871'):
        exec_env = 'singularity'
        cgroup = Path('/proc/1/cgroup')
        if cgroup.exists() and 'docker' in cgroup.read_text():
            exec_env = 'docker'
            if os.getenv('DOCKER_VERSION_8395080871'):
                exec_env = 'qsiprep-docker'

    sentry_sdk = None
    if not opts.notrack:
        import sentry_sdk
        from ..utils.sentry import sentry_setup
        sentry_setup(opts, exec_env)

    # Check input files and directories
    validate_bids(opts)
    set_freesurfer_license(opts)

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    errno = 0
    mode = "qsirecon" if opts.recon_only else "qsiprep"
    if mode == "qsirecon":
        logger.info("running qsirecon")
        building_func = build_recon_workflow
    else:
        logger.info("running qsiprep")
        building_func = build_qsiprep_workflow

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=building_func, args=(opts, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        qsiprep_wf = retval['workflow']
        plugin_settings = retval['plugin_settings']
        bids_dir = retval['bids_dir']
        output_dir = retval['output_dir']
        work_dir = retval['work_dir']
        subject_list = retval['subject_list']
        run_uuid = retval['run_uuid']
        retcode = retval['return_code']

    if qsiprep_wf is None:
        sys.exit(1)

    if opts.write_graph:
        qsiprep_wf.write_graph(
            graph2use="colored", format='svg', simple_form=True)

    if opts.reports_only:
        sys.exit(int(retcode > 0))

    if opts.boilerplate:
        sys.exit(int(retcode > 0))

    # Check workflow for missing commands
    missing = check_deps(qsiprep_wf)
    if missing:
        print("Cannot run {}. Missing dependencies:".format(mode))
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if not opts.notrack:
        from ..utils.sentry import start_ping
        start_ping(run_uuid, len(subject_list))

    errno = 1
    try:
        qsiprep_wf.run(**plugin_settings)
    except Exception as e:
        if not opts.notrack:
            from ..utils.sentry import process_crashfile
            crashfolders = [Path(output_dir) / mode / 'sub-{}'.format(s) / 'log' / run_uuid
                            for s in subject_list]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob('crash*.*'):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        logger.critical('QSIPrep failed: %s', e)
        raise
    else:
        errno = 0
        logger.log(25, 'QSI{} finished without errors'.format(mode[3:]))
        if not opts.notrack:
            sentry_sdk.capture_message('QSI{} finished without errors'.format(mode[3:]),
                                       level='info')
    # Generate reports phase
    errno += generate_reports(subject_list, output_dir, work_dir, run_uuid,
                              pipeline_mode=mode)
    write_derivative_description(bids_dir, str(Path(output_dir) / mode))

    # If we were recon-only, then we're done
    if mode == "qsirecon" or opts.recon_spec is None:
        logger.info("No additional workflows to run.")
        sys.exit(int(errno > 0))

    # Run an additional workflow if preproc + recon are requested
    opts.recon_input = output_dir + "/qsiprep"
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_recon_workflow, args=(opts, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        qsirecon_post_wf = retval['workflow']
        plugin_settings = retval['plugin_settings']
        bids_dir = retval['bids_dir']
        output_dir = retval['output_dir']
        work_dir = retval['work_dir']
        subject_list = retval['subject_list']
        run_uuid = retval['run_uuid']
        retcode = retval['return_code']

    if qsirecon_post_wf is None:
        sys.exit(1)

    if opts.write_graph:
        qsirecon_post_wf.write_graph(
            graph2use="colored", format='svg', simple_form=True)

    if opts.reports_only:
        sys.exit(int(retcode > 0))

    if opts.boilerplate:
        sys.exit(int(retcode > 0))

    # Check workflow for missing commands
    missing = check_deps(qsirecon_post_wf)
    if missing:
        print("Cannot run qsiprep. Missing dependencies:")
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)

    # Clean up master process before running workflow, which may create forks
    gc.collect()
    try:
        qsirecon_post_wf.run(**plugin_settings)
    except Exception as e:
        if not opts.notrack:
            from ..utils.sentry import process_crashfile
            crashfolders = [Path(output_dir) / 'qsirecon' / 'sub-{}'.format(s) / 'log' / run_uuid
                            for s in subject_list]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob('crash*.*'):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        logger.critical('QSIRecon failed: %s', e)
        raise
    else:
        errno += 0
        logger.log(25, 'QSIPrep finished without errors')
        if not opts.notrack:
            sentry_sdk.capture_message('QSIPostRecon finished without errors',
                                       level='info')
    errno += generate_reports(subject_list, output_dir, work_dir, run_uuid,
                              pipeline_mode='qsirecon')
    sys.exit(int(errno > 0))


def build_qsiprep_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution
    graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows qsiprep to enforce
    a hard-limited memory-scope.

    """
    from subprocess import check_call, CalledProcessError, TimeoutExpired
    from pkg_resources import resource_filename as pkgrf
    from bids import BIDSLayout
    from nipype import logging, config as ncfg
    from ..__about__ import __version__
    from ..workflows.base import init_qsiprep_wf
    from ..utils.bids import collect_participants
    from ..viz.reports import generate_reports

    logger = logging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running qsiprep version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    bids_dir = opts.bids_dir.resolve()
    output_dir = opts.output_dir.resolve()
    work_dir = Path(opts.work_dir or 'work')  # Set work/ as default

    retval['return_code'] = 1
    retval['workflow'] = None
    retval['bids_dir'] = str(bids_dir)
    retval['work_dir'] = str(work_dir)
    retval['output_dir'] = str(output_dir)

    if output_dir == bids_dir:
        logger.error(
            'The selected output folder is the same as the input BIDS folder. '
            'Please modify the output path (suggestion: %s).',
            bids_dir / 'derivatives' / ('qsiprep-%s' % __version__.split('+')[0]))
        retval['return_code'] = 1
        return retval

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    retval['run_uuid'] = run_uuid

    _db_path = opts.bids_database_dir or (
        work_dir / run_uuid / "bids_db")
    _db_path.mkdir(exist_ok=True, parents=True)

    # First check that bids_dir looks like a BIDS folder
    layout = BIDSLayout(
                str(bids_dir),
                validate=False,
                database_path=_db_path,
                reset_database=opts.bids_database_dir is None,
                ignore=(
                    "code",
                    "stimuli",
                    "sourcedata",
                    "models",
                    re.compile(r"^\.")))

    subject_list = collect_participants(
        layout, participant_label=opts.participant_label)
    retval['subject_list'] = subject_list

    # Deprecated output space
    output_spaces = opts.output_space or []
    if 'template' in output_spaces:
        logger.warning("Using 'template' as an output space is no longer supported. "
                       "Spatial normalization should be done during reconstruction.")
        output_spaces = ["T1w"]

    force_spatial_normalization = not opts.skip_t1_based_spatial_normalization
    if not force_spatial_normalization and (opts.use_syn_sdc or opts.force_syn):
        msg = [
            'SyN SDC correction requires T1 to MNI registration.',
            'Adding T1w-based normalization'
        ]
        force_spatial_normalization = True
        logger.warning(' '.join(msg))

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nthreads

    if opts.mem_mb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        logger.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)
    retval['plugin_settings'] = plugin_settings
    logger.info('Running with omp_nthreads=%d, nthreads=%d', omp_nthreads, nthreads)

    # Set up directories
    log_dir = output_dir / 'qsiprep' / 'logs'
    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': str(log_dir),
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir': str(log_dir),
            'crashfile_format': 'txt',
            'get_linked_libs': False,
            'stop_on_first_crash':
            opts.stop_on_first_crash or opts.work_dir is None,
        },
        'monitoring': {
            'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Called with reports only
    if opts.reports_only:
        logger.log(25, 'Running --reports-only on participants %s',
                   ', '.join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
            retval['run_uuid'] = run_uuid
        retval['return_code'] = generate_reports(subject_list, output_dir,
                                                 work_dir, run_uuid)
        return retval

    # Build main workflow
    logger.log(
        25,
        INIT_MSG(
            version=__version__,
            bids_dir=bids_dir,
            subject_list=subject_list,
            uuid=run_uuid))

    retval['workflow'] = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid=run_uuid,
        work_dir=work_dir,
        output_dir=str(output_dir),
        ignore=opts.ignore,
        hires=False,
        freesurfer=opts.do_reconall,
        bids_filters=opts.bids_filters,
        debug=opts.sloppy,
        low_mem=opts.low_mem,
        dwi_only=opts.dwi_only,
        infant_mode=opts.infant,
        anat_only=opts.anat_only,
        longitudinal=opts.longitudinal,
        b0_threshold=opts.b0_threshold,
        denoise_method=opts.denoise_method,
        combine_all_dwis=not opts.separate_all_dwis,
        distortion_group_merge=opts.distortion_group_merge,
        dwi_denoise_window=opts.dwi_denoise_window,
        unringing_method=opts.unringing_method,
        dwi_no_biascorr=opts.dwi_no_biascorr,
        no_b0_harmonization=opts.no_b0_harmonization,
        denoise_before_combining=not opts.denoise_after_combining,
        write_local_bvecs=opts.write_local_bvecs,
        omp_nthreads=omp_nthreads,
        skull_strip_template=opts.skull_strip_template,
        skull_strip_fixed_seed=opts.skull_strip_fixed_seed,
        force_spatial_normalization=force_spatial_normalization,
        output_spaces=output_spaces,
        output_resolution=opts.output_resolution,
        template=opts.template,
        bids_dir=bids_dir,
        motion_corr_to=opts.b0_motion_corr_to,
        hmc_transform=opts.hmc_transform,
        hmc_model=opts.hmc_model,
        eddy_config=opts.eddy_config,
        shoreline_iters=opts.shoreline_iters,
        impute_slice_threshold=opts.impute_slice_threshold,
        b0_to_t1w_transform=opts.b0_to_t1w_transform,
        intramodal_template_iters=opts.intramodal_template_iters,
        intramodal_template_transform=opts.intramodal_template_transform,
        prefer_dedicated_fmaps=opts.prefer_dedicated_fmaps,
        fmap_bspline=opts.fmap_bspline,
        fmap_demean=opts.fmap_no_demean,
        use_syn=opts.use_syn_sdc,
        force_syn=opts.force_syn
    )
    retval['return_code'] = 0

    logs_path = Path(output_dir) / 'qsiprep' / 'logs'
    boilerplate = retval['workflow'].visit_desc()
    (logs_path / 'CITATION.md').write_text(boilerplate)
    logger.log(
        25, 'Works derived from this qsiprep execution should '
        'include the following boilerplate:\n\n%s', boilerplate)

    # Generate HTML file resolving citations
    cmd = [
        'pandoc', '-s', '--bibliography',
        pkgrf('qsiprep', 'data/boilerplate.bib'), '--filter',
        'pandoc-citeproc',
        str(logs_path / 'CITATION.md'), '-o',
        str(logs_path / 'CITATION.html')
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.html file:\n%s',
                       ' '.join(cmd))

    # Generate LaTex file resolving citations
    cmd = [
        'pandoc', '-s', '--bibliography',
        pkgrf('qsiprep', 'data/boilerplate.bib'), '--natbib',
        str(logs_path / 'CITATION.md'), '-o',
        str(logs_path / 'CITATION.tex')
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.tex file:\n%s',
                       ' '.join(cmd))
    return retval


def build_recon_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution
    graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows qsiprep to enforce
    a hard-limited memory-scope.

    """
    from subprocess import check_call, CalledProcessError, TimeoutExpired
    from pkg_resources import resource_filename as pkgrf
    from bids import BIDSLayout
    from nipype import logging, config as ncfg
    from ..__about__ import __version__
    from ..workflows.recon import init_qsirecon_wf
    from ..utils.bids import collect_participants

    logger = logging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running qsirecon version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    # Set up directories
    output_dir = op.abspath(opts.output_dir)
    log_dir = op.join(output_dir, 'qsirecon', 'logs')
    work_dir = Path(opts.work_dir or 'work')  # Set work/ as default

    _db_path = opts.bids_database_dir or (
        work_dir / run_uuid / "bids_db")
    _db_path.mkdir(exist_ok=True, parents=True)

    # First check that bids_dir looks like a BIDS folder
    bids_dir = opts.bids_dir.resolve()
    layout = BIDSLayout(
                str(bids_dir),
                validate=False,
                database_path=_db_path,
                reset_database=opts.bids_database_dir is None,
                ignore=(
                    "code",
                    "stimuli",
                    "sourcedata",
                    "models",
                    re.compile(r"^\.")))
    subject_list = collect_participants(
        layout, participant_label=opts.participant_label)
    retval['subject_list'] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nthreads

    if opts.mem_mb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        logger.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)

    # Check and create output and working directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': log_dir,
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir':
            log_dir,
            'crashfile_format':
            'txt',
            'get_linked_libs':
            False,
            'stop_on_first_crash':
            opts.stop_on_first_crash or opts.work_dir is None,
        },
        'monitoring': {
            'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    retval['return_code'] = 0
    retval['plugin_settings'] = plugin_settings
    retval['bids_dir'] = bids_dir
    retval['output_dir'] = output_dir
    retval['work_dir'] = work_dir
    retval['subject_list'] = subject_list
    retval['run_uuid'] = run_uuid
    retval['workflow'] = None

    # Build main workflow
    logger.log(
        25,
        INIT_MSG(
            version=__version__,
            bids_dir=bids_dir,
            subject_list=subject_list,
            uuid=run_uuid))

    retval['workflow'] = init_qsirecon_wf(
        subject_list=subject_list,
        run_uuid=run_uuid,
        work_dir=work_dir,
        output_dir=output_dir,
        recon_input=opts.recon_input,
        recon_spec=opts.recon_spec,
        low_mem=opts.low_mem,
        omp_nthreads=omp_nthreads,
        sloppy=opts.sloppy
    )
    retval['return_code'] = 0

    logs_path = Path(output_dir) / 'qsirecon' / 'logs'
    boilerplate = retval['workflow'].visit_desc()
    (logs_path / 'CITATION.md').write_text(boilerplate)
    logger.log(
        25, 'Works derived from this qsiprep execution should '
        'include the following boilerplate:\n\n%s', boilerplate)

    # Generate HTML file resolving citations
    cmd = [
        'pandoc', '-s', '--bibliography',
        pkgrf('qsiprep', 'data/boilerplate.bib'), '--filter',
        'pandoc-citeproc',
        str(logs_path / 'CITATION.md'), '-o',
        str(logs_path / 'CITATION.html')
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.html file:\n%s',
                       ' '.join(cmd))

    # Generate LaTex file resolving citations
    cmd = [
        'pandoc', '-s', '--bibliography',
        pkgrf('qsiprep', 'data/boilerplate.bib'), '--natbib',
        str(logs_path / 'CITATION.md'), '-o',
        str(logs_path / 'CITATION.tex')
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.tex file:\n%s',
                       ' '.join(cmd))
    return retval


if __name__ == '__main__':
    raise RuntimeError(
        "qsiprep/cli/run.py should not be run directly;\n"
        "Please `pip install` qsiprep and use the `qsiprep` command")
