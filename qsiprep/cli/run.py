# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Changed to run qsiprep/qsirecon
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
"""q-Space Image preprocessing and reconstruction workflows."""

from .. import config


def main():
    """Entry point."""
    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE
    from pathlib import Path

    from ..utils.bids import write_bidsignore, write_derivative_description
    from .parser import parse_args
    from .workflow import build_workflow

    parse_args()

    if "pdb" in config.execution.debug:
        from qsiprep.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = "Linear"

    sentry_sdk = None
    if not config.execution.notrack and not config.execution.debug:
        import sentry_sdk

        from ..utils.sentry import sentry_setup

        sentry_setup()

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / config.execution.run_uuid / "config.toml"
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config.to_filename(config_file)

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    if "pdb" not in config.execution.debug:
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), "auto", retval))
            p.start()
            p.join()
            retval = dict(retval.items())  # Convert to base dictionary

            if p.exitcode:
                retval["return_code"] = p.exitcode

    else:
        retval = build_workflow(str(config_file), "auto", {})

    exitcode = retval.get("return_code", 0)
    qsiprep_wf = retval.get("workflow", None)
    exec_mode = retval.get("exec_mode", "QSIPrep")
    output_dir = (
        config.execution.qsiprep_dir if exec_mode == "QSIPrep" else config.execution.output_dir
    )

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of QSIPrep).
    config.load(config_file)

    if config.execution.reports_only:
        sys.exit(int(exitcode > 0))

    if qsiprep_wf and config.execution.write_graph:
        qsiprep_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

    exitcode = exitcode or (qsiprep_wf is None) * EX_SOFTWARE
    if exitcode != 0:
        sys.exit(exitcode)

    # Generate boilerplate
    with Manager() as mgr:
        from .workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), qsiprep_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(exitcode > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if sentry_sdk is not None:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("run_uuid", config.execution.run_uuid)
            scope.set_tag("npart", len(config.execution.participant_label))
        sentry_sdk.add_breadcrumb(message="QSIPrep started", level="info")
        sentry_sdk.capture_message("QSIPrep started", level="info")

    config.loggers.workflow.log(
        15,
        "\n".join([f"{exec_mode} config:"] + ["\t\t%s" % s for s in config.dumps().splitlines()]),
    )
    config.loggers.workflow.log(25, f"{exec_mode} started!")
    errno = 1  # Default is error exit unless otherwise set
    try:
        qsiprep_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        if not config.execution.notrack:
            from ..utils.sentry import process_crashfile

            crashfolders = [
                output_dir / f"sub-{s}" / "log" / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if sentry_sdk is not None and "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        config.loggers.workflow.critical("%s failed: %s", exec_mode, e)
        raise
    else:
        config.loggers.workflow.log(25, "QSIPrep finished successfully!")
        if sentry_sdk is not None:
            success_message = "QSIPrep finished without errors"
            sentry_sdk.add_breadcrumb(message=success_message, level="info")
            sentry_sdk.capture_message(success_message, level="info")

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = output_dir / "logs" / "CITATION.md"
        if boiler_file.exists():
            if config.environment.exec_env in (
                "singularity",
                "docker",
                "qsiprep-docker",
            ):
                boiler_file = Path("<OUTPUT_PATH>") / boiler_file.relative_to(
                    config.execution.output_dir
                )
            config.loggers.workflow.log(
                25,
                "Works derived from this QSIPrep execution should include the "
                f"boilerplate text found in {boiler_file}.",
            )

        errno = 0
    finally:

        from ..viz.reports import generate_reports

        # Generate reports phase
        # session_list = (
        #     config.execution.get().get('bids_filters', {}).get('dwi', {}).get('session')
        # )

        failed_reports = generate_reports(
            config.execution.participant_label,
            # session_list=session_list,
        )
        write_derivative_description(
            config.execution.bids_dir,
            config.execution.qsiprep_dir,
            # dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.qsiprep_dir)

        if failed_reports:
            print(failed_reports)
            # msg = (
            #     'Report generation was not successful for the following participants '
            #     f': {", ".join(failed_reports)}.'
            # )
            # config.loggers.cli.error(msg)
            # if sentry_sdk is not None:
            #     sentry_sdk.capture_message(msg, level='error')
        if not config.execution.run_preproc_and_recon:
            sys.exit(int(errno + failed_reports) > 0)

        # If preprocessing and recon are requested in the same call, start the recon workflow now.
        if errno > 0:
            if config.nipype.stop_on_first_crash:
                config.loggers.workflow.critical(
                    "Errors occurred during preprocessing - Recon will not run."
                )
                sys.exit(int(errno + failed_reports) > 0)

    # POST-PREP RECON
    del qsiprep_wf
    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    if "pdb" not in config.execution.debug:
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), "QSIRecon", retval))
            p.start()
            p.join()
            retval = dict(retval.items())  # Convert to base dictionary

            if p.exitcode:
                retval["return_code"] = p.exitcode

    else:
        retval = build_workflow(str(config_file), "QSIRecon", {})

    exitcode = retval.get("return_code", 0)
    qsirecon_wf = retval.get("workflow", None)
    exec_mode = "QSIRecon"

    # CRITICAL It would be bad to let the config file be changed between prep and recon.
    # config.load(config_file)

    if qsirecon_wf and config.execution.write_graph:
        qsirecon_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

    exitcode = exitcode or (qsirecon_wf is None) * EX_SOFTWARE
    if exitcode != 0:
        sys.exit(exitcode)

    # Generate boilerplate
    with Manager() as mgr:
        from .workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), qsirecon_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(exitcode > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if sentry_sdk is not None:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("run_uuid", config.execution.run_uuid)
            scope.set_tag("npart", len(config.execution.participant_label))
        sentry_sdk.add_breadcrumb(message="QSIPostRecon started", level="info")
        sentry_sdk.capture_message("QSIRecon started", level="info")

    config.loggers.workflow.log(
        15,
        "\n".join(["QSIRecon config:"] + ["\t\t%s" % s for s in config.dumps().splitlines()]),
    )
    config.loggers.workflow.log(25, "QSIRecon started!")
    errno = 1  # Default is error exit unless otherwise set
    try:
        qsirecon_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        if not config.execution.notrack:
            from ..utils.sentry import process_crashfile

            crashfolders = [
                config.execution.qsiprep_dir / f"sub-{s}" / "log" / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if sentry_sdk is not None and "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        config.loggers.workflow.critical("QSIPrep failed: %s", e)
        raise
    else:
        config.loggers.workflow.log(25, "QSIPrep finished successfully!")
        if sentry_sdk is not None:
            success_message = "QSIPostRecon finished without errors"
            sentry_sdk.add_breadcrumb(message=success_message, level="info")
            sentry_sdk.capture_message(success_message, level="info")

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = config.execution.qsirecon_dir / "logs" / "CITATION.md"
        if boiler_file.exists():
            if config.environment.exec_env in (
                "singularity",
                "docker",
                "qsiprep-docker",
            ):
                boiler_file = Path("<OUTPUT_PATH>") / boiler_file.relative_to(
                    config.execution.output_dir
                )
            config.loggers.workflow.log(
                25,
                "Works derived from this QSIRecon execution should include the "
                f"boilerplate text found in {boiler_file}.",
            )

        errno = 0
    finally:

        from ..viz.reports import generate_reports

        # Generate reports phase
        # session_list = (
        #     config.execution.get().get('bids_filters', {}).get('dwi', {}).get('session')
        # )

        failed_reports = generate_reports(
            config.execution.participant_label,
            # session_list=session_list,
        )
        write_derivative_description(
            config.execution.bids_dir,
            config.execution.qsiprep_dir,
            # dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.qsirecon)

        if failed_reports:
            print(failed_reports)
