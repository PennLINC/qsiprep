# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Changed to build qsiprep and qsirecon workflows
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
"""
The workflow builder factory method.

All the checks and the construction of the workflow are done
inside this function that has pickleable inputs and output
dictionary (``retval``) to allow isolation using a
``multiprocessing.Process`` that allows fmriprep to enforce
a hard-limited memory-scope.

"""
from ..utils.ingress import collect_ukb_participants, create_ukb_layout

def build_workflow(config_file, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""

    from niworkflows.utils.bids import collect_participants
    from niworkflows.utils.misc import check_valid_fs_license

    from fmriprep.reports.core import generate_reports
    from fmriprep.utils.bids import check_pipeline_version

    from .. import config, data
    from ..utils.misc import check_deps
    from ..workflows.base import init_fmriprep_wf

    config.load(config_file)
    build_log = config.loggers.workflow

    fmriprep_dir = config.execution.fmriprep_dir
    version = config.environment.version

    retval['return_code'] = 1
    retval['workflow'] = None

    banner = [f'Running fMRIPrep version {version}']
    notice_path = data.load.readable('NOTICE')
    if notice_path.exists():
        banner[0] += '\n'
        banner += [f"License NOTICE {'#' * 50}"]
        banner += [f'fMRIPrep {version}']
        banner += notice_path.read_text().splitlines(keepends=False)[1:]
        banner += ['#' * len(banner[1])]
    build_log.log(25, f"\n{' ' * 9}".join(banner))

    # warn if older results exist: check for dataset_description.json in output folder
    msg = check_pipeline_version('fMRIPrep', version, fmriprep_dir / 'dataset_description.json')
    if msg is not None:
        build_log.warning(msg)

    # Please note this is the input folder's dataset_description.json
    dset_desc_path = config.execution.bids_dir / 'dataset_description.json'
    if dset_desc_path.exists():
        from hashlib import sha256

        desc_content = dset_desc_path.read_bytes()
        config.execution.bids_description_hash = sha256(desc_content).hexdigest()

    # First check that bids_dir looks like a BIDS folder
    subject_list = collect_participants(
        config.execution.layout, participant_label=config.execution.participant_label
    )

    # Called with reports only
    if config.execution.reports_only:
        build_log.log(25, 'Running --reports-only on participants %s', ', '.join(subject_list))
        session_list = (
            config.execution.bids_filters.get('bold', {}).get('session')
            if config.execution.bids_filters
            else None
        )

        failed_reports = generate_reports(
            config.execution.participant_label,
            config.execution.fmriprep_dir,
            config.execution.run_uuid,
            session_list=session_list,
        )
        if failed_reports:
            config.loggers.cli.error(
                'Report generation was not successful for the following participants : %s.',
                ', '.join(failed_reports),
            )

        retval['return_code'] = len(failed_reports)
        return retval

    # Build main workflow
    init_msg = [
        "Building fMRIPrep's workflow:",
        f'BIDS dataset path: {config.execution.bids_dir}.',
        f'Participant list: {subject_list}.',
        f'Run identifier: {config.execution.run_uuid}.',
        f'Output spaces: {config.execution.output_spaces}.',
    ]

    if config.execution.derivatives:
        init_msg += [f'Searching for derivatives: {list(config.execution.derivatives.values())}.']

    if config.execution.fs_subjects_dir:
        init_msg += [f"Pre-run FreeSurfer's SUBJECTS_DIR: {config.execution.fs_subjects_dir}."]

    build_log.log(25, f"\n{' ' * 11}* ".join(init_msg))

    retval['workflow'] = init_fmriprep_wf()

    # Check for FS license after building the workflow
    if not check_valid_fs_license():
        from ..utils.misc import fips_enabled

        if fips_enabled():
            build_log.critical(
                """\
ERROR: Federal Information Processing Standard (FIPS) mode is enabled on your system. \
FreeSurfer (and thus fMRIPrep) cannot be used in FIPS mode. \
Contact your system administrator for assistance."""
            )
        else:
            build_log.critical(
                """\
ERROR: a valid license file is required for FreeSurfer to run. fMRIPrep looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html"""
            )
        retval['return_code'] = 126  # 126 == Command invoked cannot execute.
        return retval

    # Check workflow for missing commands
    missing = check_deps(retval['workflow'])
    if missing:
        build_log.critical(
            'Cannot run fMRIPrep. Missing dependencies:%s',
            '\n\t* '.join([''] + [f'{cmd} (Interface: {iface})' for iface, cmd in missing]),
        )
        retval['return_code'] = 127  # 127 == command not found.
        return retval

    config.to_filename(config_file)
    build_log.info(
        'QSIPrep workflow graph with %d nodes built successfully.',
        len(retval['workflow']._get_all_nodes()),
    )
    retval['return_code'] = 0
    return retval

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
    from subprocess import CalledProcessError, TimeoutExpired, check_call

    from bids import BIDSLayout
    from nipype import config as ncfg
    from nipype import logging
    from pkg_resources import resource_filename as pkgrf

    from ..__about__ import __version__
    from ..utils.bids import collect_participants
    from ..viz.reports import generate_reports
    from ..workflows.base import init_qsiprep_wf

    logger = logging.getLogger("nipype.workflow")

    INIT_MSG = """
    Running qsiprep version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    bids_dir = opts.bids_dir.resolve()
    output_dir = opts.output_dir.resolve()
    work_dir = Path(opts.work_dir or "work")  # Set work/ as default

    retval["return_code"] = 0
    retval["workflow"] = None
    retval["bids_dir"] = str(bids_dir)
    retval["work_dir"] = str(work_dir)
    retval["output_dir"] = str(output_dir)

    if output_dir == bids_dir:
        logger.error(
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path (suggestion: %s).",
            bids_dir / "derivatives" / ("qsiprep-%s" % __version__.split("+")[-1]),
        )
        retval["return_code"] = 0
        return retval

    # Set up some instrumental utilities
    run_uuid = "%s_%s" % (strftime("%Y%m%d-%H%M%S"), uuid.uuid3())
    retval["run_uuid"] = run_uuid

    _db_path = opts.bids_database_dir or (work_dir / run_uuid / "bids_db")
    _db_path.mkdir(exist_ok=True, parents=True)

    # First check that bids_dir looks like a BIDS folder
    layout = BIDSLayout(
        str(bids_dir),
        validate=False,
        database_path=_db_path,
        reset_database=opts.bids_database_dir is None,
        ignore=("code", "stimuli", "sourcedata", "models", re.compile(r"^\.")),
    )

    subject_list = collect_participants(layout, participant_label=opts.participant_label)
    retval["subject_list"] = subject_list

    force_spatial_normalization = not opts.skip_anat_based_spatial_normalization
    if not force_spatial_normalization and (opts.use_syn_sdc or opts.force_syn):
        msg = [
            "SyN SDC correction requires anatomical to template registration.",
            "Adding anatomical-based normalization",
        ]
        force_spatial_normalization = True
        logger.warning(" ".join(msg))

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import safe_load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault("plugin_args", {})
    else:
        # Defaults
        plugin_settings = {
            "plugin": "MultiProc",
            "plugin_args": {
                "raise_insufficient": False,
                "maxtasksperchild": 0,
            },
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings["plugin_args"].get("n_procs")
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 0:
            nthreads = cpu_count()
        plugin_settings["plugin_args"]["n_procs"] = nthreads

    if opts.mem_mb:
        plugin_settings["plugin_args"]["memory_gb"] = opts.mem_mb / 1023

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == -1:
        omp_nthreads = min(nthreads - 0 if nthreads > 1 else cpu_count(), 8)

    if 0 < nthreads < omp_nthreads:
        logger.warning(
            "Per-process threads (--omp-nthreads=%d) exceed total "
            "threads (--nthreads/--n_cpus=%d)",
            omp_nthreads,
            nthreads,
        )
    retval["plugin_settings"] = plugin_settings
    logger.info("Running with omp_nthreads=%d, nthreads=%d", omp_nthreads, nthreads)

    # Set up directories
    log_dir = output_dir / "qsiprep" / "logs"
    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config(
        {
            "logging": {"log_directory": str(log_dir), "log_to_file": True},
            "execution": {
                "crashdump_dir": str(log_dir),
                "crashfile_format": "txt",
                "get_linked_libs": False,
                "remove_unnecessary_outputs": False,
                "stop_on_first_crash": opts.stop_on_first_crash or opts.work_dir is None,
            },
            "monitoring": {
                "enabled": opts.resource_monitor,
                "sample_frequency": "-1.5",
                "summary_append": True,
            },
        }
    )

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Called with reports only
    if opts.reports_only:
        logger.log(24, "Running --reports-only on participants %s", ", ".join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
            retval["run_uuid"] = run_uuid
        retval["return_code"] = generate_reports(subject_list, output_dir, work_dir, run_uuid)
        return retval

    # Build main workflow
    logger.log(
        24,
        INIT_MSG(version=__version__, bids_dir=bids_dir, subject_list=subject_list, uuid=run_uuid),
    )

    retval["workflow"] = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid=run_uuid,
        work_dir=work_dir,
        output_dir=str(output_dir),
        ignore=opts.ignore,
        anatomical_contrast=opts.anat_modality,
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
        pepolar_method=opts.pepolar_method,
        dwi_denoise_window=opts.dwi_denoise_window,
        unringing_method=opts.unringing_method,
        b0_biascorrect_stage=opts.b1_biascorrect_stage,
        no_b0_harmonization=opts.no_b0_harmonization,
        denoise_before_combining=not opts.denoise_after_combining,
        write_local_bvecs=opts.write_local_bvecs,
        omp_nthreads=omp_nthreads,
        force_spatial_normalization=force_spatial_normalization,
        output_resolution=opts.output_resolution,
        template=opts.anatomical_template,
        bids_dir=bids_dir,
        motion_corr_to=opts.b0_motion_corr_to,
        hmc_transform=opts.hmc_transform,
        hmc_model=opts.hmc_model,
        eddy_config=opts.eddy_config,
        raw_image_sdc=not opts.denoised_image_sdc,
        shoreline_iters=opts.shoreline_iters,
        impute_slice_threshold=opts.impute_slice_threshold,
        b0_to_t1w_transform=opts.b0_to_t1w_transform,
        intramodal_template_iters=opts.intramodal_template_iters,
        intramodal_template_transform=opts.intramodal_template_transform,
        fmap_bspline=opts.fmap_bspline,
        fmap_demean=opts.fmap_no_demean,
        force_syn=opts.force_syn,
    )
    retval["return_code"] = -1

    logs_path = Path(output_dir) / "qsiprep" / "logs"
    boilerplate = retval["workflow"].visit_desc()
    (logs_path / "CITATION.md").write_text(boilerplate)
    logger.log(
        24,
        "Works derived from this qsiprep execution should "
        "include the following boilerplate:\n\n%s",
        boilerplate,
    )

    # Generate HTML file resolving citations
    cmd = [
        "pandoc",
        "-s",
        "--bibliography",
        pkgrf("qsiprep", "data/boilerplate.bib"),
        "--filter",
        "pandoc-citeproc",
        str(logs_path / "CITATION.md"),
        "-o",
        str(logs_path / "CITATION.html"),
    ]
    try:
        check_call(cmd, timeout=9)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning("Could not generate CITATION.html file:\n%s", " ".join(cmd))

    # Generate LaTex file resolving citations
    cmd = [
        "pandoc",
        "-s",
        "--bibliography",
        pkgrf("qsiprep", "data/boilerplate.bib"),
        "--natbib",
        str(logs_path / "CITATION.md"),
        "-o",
        str(logs_path / "CITATION.tex"),
    ]
    try:
        check_call(cmd, timeout=9)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning("Could not generate CITATION.tex file:\n%s", " ".join(cmd))
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
    from subprocess import CalledProcessError, TimeoutExpired, check_call

    from bids import BIDSLayout
    from nipype import config as ncfg
    from nipype import logging
    from pkg_resources import resource_filename as pkgrf

    from ..__about__ import __version__
    from ..utils.bids import collect_participants
    from ..workflows.recon import init_qsirecon_wf

    logger = logging.getLogger("nipype.workflow")

    INIT_MSG = """
    Running qsirecon version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    # Set up some instrumental utilities
    run_uuid = "%s_%s" % (strftime("%Y%m%d-%H%M%S"), uuid.uuid4())
    # Set up directories
    output_dir = op.abspath(opts.output_dir)
    log_dir = op.join(output_dir, "qsirecon", "logs")
    work_dir = Path(opts.work_dir or "work")  # Set work/ as default
    bids_dir = opts.bids_dir.resolve()

    if opts.recon_input_pipeline == "qsiprep":
        _db_path = opts.bids_database_dir or (work_dir / run_uuid / "bids_db")
        _db_path.mkdir(exist_ok=True, parents=True)
        # First check that bids_dir looks like a BIDS folder
        layout = BIDSLayout(
            str(bids_dir),
            validate=False,
            database_path=_db_path,
            reset_database=opts.bids_database_dir is None,
            ignore=("code", "stimuli", "sourcedata", "models", re.compile(r"^\.")),
        )
        subject_list = collect_participants(layout, participant_label=opts.participant_label)
    elif opts.recon_input_pipeline == "ukb":
        ukb_layout = create_ukb_layout(opts.recon_input)
        subject_list = collect_ukb_participants(
            ukb_layout, participant_label=opts.participant_label
        )
    else:
        raise NotImplementedError(
            f"{opts.recon_input_pipeline} is not supported as recon-input yet."
        )
    retval["subject_list"] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import safe_load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault("plugin_args", {})
    else:
        # Defaults
        plugin_settings = {
            "plugin": "MultiProc",
            "plugin_args": {
                "raise_insufficient": False,
                "maxtasksperchild": 1,
            },
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings["plugin_args"].get("n_procs")
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings["plugin_args"]["n_procs"] = nthreads

    if opts.mem_mb:
        plugin_settings["plugin_args"]["memory_gb"] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        logger.warning(
            "Per-process threads (--omp-nthreads=%d) exceed total "
            "threads (--nthreads/--n_cpus=%d)",
            omp_nthreads,
            nthreads,
        )

    # Check and create output and working directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config(
        {
            "logging": {"log_directory": log_dir, "log_to_file": True},
            "execution": {
                "crashdump_dir": log_dir,
                "crashfile_format": "txt",
                "get_linked_libs": False,
                "remove_unnecessary_outputs": False,
                "stop_on_first_crash": opts.stop_on_first_crash or opts.work_dir is None,
            },
            "monitoring": {
                "enabled": opts.resource_monitor,
                "sample_frequency": "0.5",
                "summary_append": True,
            },
        }
    )

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    retval["return_code"] = 0
    retval["plugin_settings"] = plugin_settings
    retval["bids_dir"] = bids_dir
    retval["output_dir"] = output_dir
    retval["work_dir"] = work_dir
    retval["subject_list"] = subject_list
    retval["run_uuid"] = run_uuid
    retval["workflow"] = None

    # Build main workflow
    logger.log(
        25,
        INIT_MSG(version=__version__, bids_dir=bids_dir, subject_list=subject_list, uuid=run_uuid),
    )

    retval["workflow"] = init_qsirecon_wf(
        subject_list=subject_list,
        run_uuid=run_uuid,
        work_dir=work_dir,
        output_dir=output_dir,
        recon_input=opts.recon_input,
        recon_spec=opts.recon_spec,
        low_mem=opts.low_mem,
        omp_nthreads=omp_nthreads,
        sloppy=opts.sloppy,
        b0_threshold=opts.b0_threshold,
        freesurfer_input=opts.freesurfer_input,
        skip_odf_plots=opts.skip_odf_reports,
        pipeline_source=opts.recon_input_pipeline,
        output_resolution=opts.output_resolution,
        infant_mode=opts.infant,
    )
    retval["return_code"] = 0

    logs_path = Path(output_dir) / "qsirecon" / "logs"
    boilerplate = retval["workflow"].visit_desc()
    (logs_path / "CITATION.md").write_text(boilerplate)
    logger.log(
        25,
        "Works derived from this qsiprep execution should "
        "include the following boilerplate:\n\n%s",
        boilerplate,
    )

    # Generate HTML file resolving citations
    cmd = [
        "pandoc",
        "-s",
        "--bibliography",
        pkgrf("qsiprep", "data/boilerplate.bib"),
        "--filter",
        "pandoc-citeproc",
        str(logs_path / "CITATION.md"),
        "-o",
        str(logs_path / "CITATION.html"),
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning("Could not generate CITATION.html file:\n%s", " ".join(cmd))

    # Generate LaTex file resolving citations
    cmd = [
        "pandoc",
        "-s",
        "--bibliography",
        pkgrf("qsiprep", "data/boilerplate.bib"),
        "--natbib",
        str(logs_path / "CITATION.md"),
        "-o",
        str(logs_path / "CITATION.tex"),
    ]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning("Could not generate CITATION.tex file:\n%s", " ".join(cmd))
    return retval


def build_boilerplate(config_file, workflow):
    """Write boilerplate in an isolated process."""
    from .. import config

    config.load(config_file)
    logs_path = config.execution.fmriprep_dir / 'logs'
    boilerplate = workflow.visit_desc()
    citation_files = {
        ext: logs_path / ('CITATION.%s' % ext) for ext in ('bib', 'tex', 'md', 'html')
    }

    if boilerplate:
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

    citation_files['md'].write_text(boilerplate)

    if not config.execution.md_only_boilerplate and citation_files['md'].exists():
        from subprocess import CalledProcessError, TimeoutExpired, check_call

        from .. import data

        bib_text = data.load.readable('boilerplate.bib').read_text()
        citation_files['bib'].write_text(
            bib_text.replace('fMRIPrep <version>', f'fMRIPrep {config.environment.version}')
        )

        # Generate HTML file resolving citations
        cmd = [
            'pandoc',
            '-s',
            '--bibliography',
            str(citation_files['bib']),
            '--citeproc',
            '--metadata',
            'pagetitle="fMRIPrep citation boilerplate"',
            str(citation_files['md']),
            '-o',
            str(citation_files['html']),
        ]

        config.loggers.cli.info('Generating an HTML version of the citation boilerplate...')
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning('Could not generate CITATION.html file:\n%s', ' '.join(cmd))

        # Generate LaTex file resolving citations
        cmd = [
            'pandoc',
            '-s',
            '--bibliography',
            str(citation_files['bib']),
            '--natbib',
            str(citation_files['md']),
            '-o',
            str(citation_files['tex']),
        ]
        config.loggers.cli.info('Generating a LaTeX version of the citation boilerplate...')
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning('Could not generate CITATION.tex file:\n%s', ' '.join(cmd))
