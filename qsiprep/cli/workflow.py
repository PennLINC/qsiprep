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
``multiprocessing.Process`` that allows qsiprep to enforce
a hard-limited memory-scope.

"""
from pathlib import Path

from pkg_resources import resource_filename as pkgrf


def build_workflow(config_file, exec_mode, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""

    from niworkflows.utils.bids import collect_participants

    # from niworkflows.utils.misc import check_valid_fs_license
    # from ..utils.bids import check_pipeline_version
    from .. import config
    from ..utils.misc import check_deps
    from ..viz.reports import generate_reports
    from ..workflows.base import init_qsiprep_wf
    from ..workflows.recon import init_qsirecon_wf

    config.load(config_file)
    build_log = config.loggers.workflow

    # qsiprep_dir = config.execution.qsiprep_dir
    version = config.environment.version

    retval["return_code"] = 1
    retval["workflow"] = None

    # Figure out which workflow we want automatically if
    if exec_mode == "auto":
        if config.execution.recon_input:
            exec_mode = "QSIRecon"
            build_log.info("Running recon-only mode: --recon-input was used.")
            if config.execution.recon_only:
                config.loggers.workflow.warning(
                    "Argument --recon-only is not needed if --recon-input is specified."
                )
        else:
            exec_mode = "QSIPrep"

    if exec_mode == "QSIPrep":
        workflow_builder = init_qsiprep_wf
    elif exec_mode == "QSIRecon":
        workflow_builder = init_qsirecon_wf
    else:
        raise Exception(f"Unknown mode: {exec_mode}")

    banner = [f"Running {exec_mode} version {version}"]
    notice_path = Path(pkgrf("qsiprep", "data/NOTICE"))
    if notice_path.exists():
        banner[0] += "\n"
        banner += [f"License NOTICE {'#' * 50}"]
        banner += [f"QSIPrep {version}"]
        banner += notice_path.read_text().splitlines(keepends=False)[1:]
        banner += ["#" * len(banner[1])]
    build_log.log(25, f"\n{' ' * 9}".join(banner))

    # warn if older results exist: check for dataset_description.json in output folder
    # msg = check_pipeline_version("QSIPrep", version, qsiprep_dir / "dataset_description.json")
    # if msg is not None:
    #     build_log.warning(msg)

    # Please note this is the input folder's dataset_description.json
    dset_desc_path = config.execution.bids_dir / "dataset_description.json"
    if dset_desc_path.exists():
        from hashlib import sha256

        desc_content = dset_desc_path.read_bytes()
        config.execution.bids_description_hash = sha256(desc_content).hexdigest()

    # First check that bids_dir looks like a BIDS folder
    subject_list = collect_participants(
        config.execution.layout, participant_label=config.execution.participant_label
    )

    # Called with reports only
    if config.execution.reports_only and exec_mode == "QSIPrep":
        build_log.log(25, "Running --reports-only on participants %s", ", ".join(subject_list))
        session_list = (
            config.execution.bids_filters.get("dwi", {}).get("session")
            if config.execution.bids_filters
            else None
        )

        failed_reports = generate_reports(
            config.execution.participant_label,
            config.execution.qsiprep_dir,
            config.execution.run_uuid,
            session_list=session_list,
        )
        if failed_reports:
            config.loggers.cli.error(
                "Report generation was not successful for the following participants : %s.",
                ", ".join(failed_reports),
            )

        retval["return_code"] = len(failed_reports)
        return retval

    # Build main workflow
    init_msg = [
        f"Building {exec_mode}'s workflow:",
        f"BIDS dataset path: {config.execution.bids_dir}.",
        f"Participant list: {subject_list}.",
        f"Run identifier: {config.execution.run_uuid}.",
    ]

    if config.execution.fs_subjects_dir and exec_mode == "QSIRecon":
        init_msg += [f"Pre-run FreeSurfer's SUBJECTS_DIR: {config.execution.fs_subjects_dir}."]

    build_log.log(25, f"\n{' ' * 11}* ".join(init_msg))

    # Check for FS license after building the workflow
    if not Path(config.execution.fs_license_file).exists():
        build_log.critical(
            """\
ERROR: a valid license file is required for FreeSurfer to run. QSIPrep looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html"""
        )
        retval["return_code"] = 126  # 126 == Command invoked cannot execute.
        return retval

    # If qsiprep is being run on already preprocessed data:
    retval["exec_mode"] = exec_mode
    retval["workflow"] = workflow_builder()

    # Check workflow for missing commands
    missing = check_deps(retval["workflow"])
    if missing:
        build_log.critical(
            "Cannot run %s. Missing dependencies:%s",
            retval["exec_mode"],
            "\n\t* ".join([""] + [f"{cmd} (Interface: {iface})" for iface, cmd in missing]),
        )
        retval["return_code"] = 127  # 127 == command not found.
        return retval

    config.to_filename(config_file)
    build_log.info(
        "%s workflow graph with %d nodes built successfully.",
        retval["exec_mode"],
        len(retval["workflow"]._get_all_nodes()),
    )
    retval["return_code"] = 0
    return retval


def build_boilerplate(config_file, workflow):
    """Write boilerplate in an isolated process."""
    from .. import config

    config.load(config_file)
    logs_path = config.execution.qsiprep_dir / "logs"
    boilerplate = workflow.visit_desc()
    citation_files = {
        ext: logs_path / ("CITATION.%s" % ext) for ext in ("bib", "tex", "md", "html")
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

    citation_files["md"].write_text(boilerplate)

    if citation_files["md"].exists():
        from subprocess import CalledProcessError, TimeoutExpired, check_call

        bib_text = Path(pkgrf("qsiprep", "data/boilerplate.bib")).read_text()
        citation_files["bib"].write_text(
            bib_text.replace("QSIPrep <version>", f"QSIPrep {config.environment.version}")
        )

        # Generate HTML file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(citation_files["bib"]),
            "--filter",
            "pandoc-citeproc",
            "--metadata",
            'pagetitle="QSIPrep citation boilerplate"',
            str(citation_files["md"]),
            "-o",
            str(citation_files["html"]),
        ]

        config.loggers.cli.info("Generating an HTML version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.html file:\n%s", " ".join(cmd))

        # Generate LaTex file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(citation_files["bib"]),
            "--natbib",
            str(citation_files["md"]),
            "-o",
            str(citation_files["tex"]),
        ]
        config.loggers.cli.info("Generating a LaTeX version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.tex file:\n%s", " ".join(cmd))
