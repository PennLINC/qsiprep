#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSI workflow
=====
"""
import gc
import logging
import os
import os.path as op
import re
import sys
import uuid
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
from time import strftime

from ..utils.ingress import collect_ukb_participants, create_ukb_layout

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG
logger = logging.getLogger("cli")


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning("Captured warning (%s): %s", category, message)


def check_deps(workflow):
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, "_cmd") and which(node.interface._cmd.split()[0]) is None)
    )


def _filter_pybids_none_any(dct):
    import bids

    return {
        k: bids.layout.Query.NONE if v is None else (bids.layout.Query.ANY if v == "*" else v)
        for k, v in dct.items()
    }


def _bids_filter(value):
    from json import loads

    from bids.layout import Query

    if value and Path(value).exists():
        try:
            filters = loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
        except Exception:
            raise Exception("Unable to parse BIDS filter file. Check that it is " "valid JSON.")
    else:
        raise Exception("Unable to load BIDS filter file " + value)

    # unserialize pybids Query enum values
    for acq, _filters in filters.items():
        filters[acq] = {
            k: getattr(Query, v[7:-4]) if not isinstance(v, Query) and "Query" in v else v
            for k, v in _filters.items()
        }
    return filters


def get_parser():
    """Build parser object"""
    from ..__about__ import __version__

    verstr = "qsiprep v{}".format(__version__)

    parser = ArgumentParser(
        description="qsiprep: q-Space Image Preprocessing workflows",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )



    return parser



def main():
    """Entry point"""
    from multiprocessing import Manager, Process, set_start_method

    from nipype import logging as nlogging

    from ..utils.bids import write_derivative_description
    from ..viz.reports import generate_reports

    try:
        set_start_method("forkserver")
    except RuntimeError:
        pass

    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    exec_env = os.name

    # special variable set in the container
    if os.getenv("IS_DOCKER_8395080871"):
        exec_env = "singularity"
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text():
            exec_env = "docker"
            if os.getenv("DOCKER_VERSION_8395080871"):
                exec_env = "qsiprep-docker"

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
    nlogging.getLogger("nipype.workflow").setLevel(log_level)
    nlogging.getLogger("nipype.interface").setLevel(log_level)
    nlogging.getLogger("nipype.utils").setLevel(log_level)

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

        qsiprep_wf = retval["workflow"]
        plugin_settings = retval["plugin_settings"]
        bids_dir = retval["bids_dir"]
        output_dir = retval["output_dir"]
        work_dir = retval["work_dir"]
        subject_list = retval["subject_list"]
        run_uuid = retval["run_uuid"]
        retcode = retval["return_code"]

    if qsiprep_wf is None:
        sys.exit(1)

    if opts.write_graph:
        qsiprep_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

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

            crashfolders = [
                Path(output_dir) / mode / "sub-{}".format(s) / "log" / run_uuid
                for s in subject_list
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        logger.critical("QSIPrep failed: %s", e)
        raise
    else:
        errno = 0
        logger.log(25, "QSI{} finished without errors".format(mode[3:]))
        if not opts.notrack:
            sentry_sdk.capture_message(
                "QSI{} finished without errors".format(mode[3:]), level="info"
            )
    # Generate reports phase
    errno += generate_reports(subject_list, output_dir, work_dir, run_uuid, pipeline_mode=mode)
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

        qsirecon_post_wf = retval["workflow"]
        plugin_settings = retval["plugin_settings"]
        bids_dir = retval["bids_dir"]
        output_dir = retval["output_dir"]
        work_dir = retval["work_dir"]
        subject_list = retval["subject_list"]
        run_uuid = retval["run_uuid"]
        retcode = retval["return_code"]

    if qsirecon_post_wf is None:
        sys.exit(1)

    if opts.write_graph:
        qsirecon_post_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

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

            crashfolders = [
                Path(output_dir) / "qsirecon" / "sub-{}".format(s) / "log" / run_uuid
                for s in subject_list
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        logger.critical("QSIRecon failed: %s", e)
        raise
    else:
        errno += 0
        logger.log(25, "QSIPrep finished without errors")
        if not opts.notrack:
            sentry_sdk.capture_message("QSIPostRecon finished without errors", level="info")
    errno += generate_reports(
        subject_list, output_dir, work_dir, run_uuid, pipeline_mode="qsirecon"
    )
    sys.exit(int(errno > 0))





if __name__ == "__main__":
    raise RuntimeError(
        "qsiprep/cli/run.py should not be run directly;\n"
        "Please `pip install` qsiprep and use the `qsiprep` command"
    )
