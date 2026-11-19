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

from nireports.assembler.report import Report as NiReportsReport

from qsiprep import config, data

# nireports titles each combination of ordering entities "Reports for: <entities>.",
# which is "Reports for: ." for figures that have none of them.
_EMPTY_COMBINATION_TITLE = 'Reports for: .'
_UNORDERED_TITLE = 'Reports for: merged scans.'


class Report(NiReportsReport):
    """A nireports Report that keeps figures lacking every ordering entity.

    A section with an ``ordering`` (e.g., ``session,acquisition,run``) is split into one
    block per combination of those entities' values.
    nireports drops the combination in which every ordering entity is absent,
    so once any figure carries one of the entities,
    figures without any of them disappear from the report.
    In QSIPrep, that happens when several runs are merged:
    the per-run denoising and unringing figures keep their ``run`` entity,
    while the figures of the merged series have none.
    """

    @staticmethod
    def _process_orderings(orderings, hits):
        entities, value_combos = NiReportsReport._process_orderings(orderings, hits)
        if not value_combos:
            # No figure has an ordering entity, so the section is not split at all.
            return entities, value_combos

        none_member = (None,) * len(orderings)
        has_unordered = any(
            tuple(hit.get_entities().get(k, None) for k in orderings) == none_member
            for hit in hits
        )
        if has_unordered:
            # Query with every kept entity absent, after the per-entity blocks.
            value_combos.append((None,) * len(entities))

        return entities, value_combos

    def index(self, config):
        super().index(config)
        for section in self.sections:
            for reportlet in section.reportlets:
                if reportlet.title == _EMPTY_COMBINATION_TITLE:
                    reportlet.title = _UNORDERED_TITLE


def run_reports(
    output_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename='report.html',
    reportlets_dir=None,
    errorname='report.err',
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
        traceback.print_exception(*sys.exc_info(), file=str(Path(output_dir) / 'logs' / errorname))
        return subject_label

    return None


def generate_reports(
    processing_list,
    subject_anatomical_reference,
    report_output_level,
    output_dir,
    run_uuid,
    bootstrap_file=None,
    work_dir=None,
):
    """Generate reports for a list of processing groups.

    Parameters
    ----------
    processing_list : list of tuple
        (subject label, list of session labels) for each processing group.
    subject_anatomical_reference : {'sessionwise', 'unbiased', 'first-lex'}
        Determines what each report covers.
        With "sessionwise" there is one report per session,
        otherwise there is one report per subject.
    report_output_level : {'root', 'subject', 'session'}
        Directory level at which the reports are written.
        Session-level reports are only possible for session-wise reports,
        so subject-wise reports fall back to the subject level with a warning.
    output_dir : str or pathlib.Path
        QSIPrep output directory, which also holds the reportlets.
    run_uuid : str
        Unique identifier of this run, used in the names of error files.
    bootstrap_file : str or pathlib.Path or None, optional
        Report specification file. If None, the packaged ``reports-spec.yml`` is used.
    work_dir : str or None, optional
        Working directory. Currently unused.
    """
    bootstrap_file = data.load('reports-spec.yml') if bootstrap_file is None else bootstrap_file

    errors = []
    for subject_label, session_list in processing_list:
        subject_id = subject_label.removeprefix('sub-')

        if subject_anatomical_reference == 'sessionwise' and session_list:
            # With session-wise anatomical processing,
            # the session-wise anatomical reports are in here too.
            session_ids = [session.removeprefix('ses-') for session in session_list]
        else:
            # The report covers the subject as a whole.
            session_ids = [None]

        for session_label in session_ids:
            output_level = report_output_level
            if output_level == 'session' and session_label is None:
                output_level = 'subject'
                config.loggers.workflow.warning(
                    'Session-level reports were requested, '
                    f'but the report for subject {subject_id} covers no single session. '
                    'Writing out reports to subject level.'
                )

            if session_label is None:
                html_report = f'sub-{subject_id}.html'
                errorname = f'report-{run_uuid}-{subject_label}.err'
                session_entity = {}
            else:
                html_report = f'sub-{subject_id}_ses-{session_label}.html'
                errorname = f'report-{run_uuid}-{subject_label}-{session_label}.err'
                session_entity = {'session': session_label}

            if output_level == 'root':
                report_dir = Path(output_dir)
            elif output_level == 'subject':
                report_dir = Path(output_dir) / f'sub-{subject_id}'
            else:
                report_dir = Path(output_dir) / f'sub-{subject_id}' / f'ses-{session_label}'

            report_error = run_reports(
                report_dir,
                subject_label,
                run_uuid,
                bootstrap_file=bootstrap_file,
                out_filename=html_report,
                reportlets_dir=output_dir,
                errorname=errorname,
                subject=subject_label,
                **session_entity,
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

    return errors
