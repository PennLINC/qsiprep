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
from pathlib import Path

from nireports.assembler.report import Report

from .. import data


def run_reports(
    output_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename="report.html",
    reportlets_dir=None,
    errorname="report.err",
    **entities,
):
    """Run the reports."""
    robj = Report(
        output_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        out_filename=out_filename,
        reportlets_dir=reportlets_dir,
        plugins=None,
        plugin_meta=None,
        metadata=None,
        **entities,
    )

    # Count nbr of subject for which report generation failed
    try:
        robj.generate_report()
    except:  # noqa: E722
        import sys
        import traceback

        # Store the list of subjects for which report generation failed
        traceback.print_exception(*sys.exc_info(), file=str(Path(output_dir) / "logs" / errorname))
        return subject_label

    return None


def generate_reports(
    processing_list, output_level, output_dir, run_uuid, bootstrap_file=None, work_dir=None
):
    """Generate reports for a list of processing groups.

    Parameters
    ----------
    output_level {"sessionwise", "unbiased", "first-alphabetically"}
    """
    errors = []
    for subject_label, session_list in processing_list:
        subject_id = subject_label[4:] if subject_label.startswith("sub-") else subject_label

        if bootstrap_file is not None:
            # If a config file is precised, we do not override it
            html_report = "report.html"
        else:
            # If there are only a few session for this subject,
            # we aggregate them in a single visual report.
            bootstrap_file = data.load("reports-spec.yml")
            html_report = "report.html"

        # We only make this one if it's all the sessions or just the anat and not sessionwise
        if output_level != "sessionwise":
            report_error = run_reports(
                output_dir,
                subject_label,
                run_uuid,
                bootstrap_file=bootstrap_file,
                out_filename=html_report,
                reportlets_dir=output_dir,
                errorname=f"report-{run_uuid}-{subject_label}.err",
                subject=subject_label,
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

        else:
            # Beyond a certain number of sessions per subject,
            # we separate the dwi reports per session
            # If output_level is "sessionwise",
            # the session-wise anatomical reports are in here too
            session_list = [ses[4:] if ses.startswith("ses-") else ses for ses in session_list]

            for session_label in session_list:
                bootstrap_file = data.load("reports-spec.yml")
                session_dir = output_dir / f"sub-{subject_id}" / f"ses-{session_label}"
                html_report = f"sub-{subject_id}_ses-{session_label}.html"

                report_error = run_reports(
                    session_dir,
                    subject_label,
                    run_uuid,
                    bootstrap_file=bootstrap_file,
                    out_filename=html_report,
                    reportlets_dir=output_dir,
                    errorname=f"report-{run_uuid}-{subject_label}-{session_label}.err",
                    subject=subject_label,
                    session=session_label,
                )
                # If the report generation failed, append the subject label for which it failed
                if report_error is not None:
                    errors.append(report_error)

    return errors
