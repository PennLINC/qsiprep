#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIprep reports builder
^^^^^^^^^^^^^^^^^^^^^^^^


"""
import json
import re
from pathlib import Path

import jinja2
from nipype.utils.filemanip import copyfile
from pkg_resources import resource_filename as pkgrf

from ..utils.bids import BIDS_NAME
from ..utils.misc import read_crashfile


class Element(object):
    """
    Just a basic component of a report
    """

    def __init__(self, name, title=None):
        self.name = name
        self.title = title


class Reportlet(Element):
    """
    A reportlet has title, description and a list of graphical components
    """

    def __init__(self, name, file_pattern=None, title=None, description=None, raw=False):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.source_files = []
        self.contents = []
        self.raw = raw


class SubReport(Element):
    """
    SubReports are sections within a Report
    """

    def __init__(self, name, reportlets=None, title=''):
        self.name = name
        self.title = title
        self.reportlets = []
        if reportlets:
            self.reportlets += reportlets
        self.isnested = False


class Report(object):
    """
    The full report object
    """

    def __init__(self, path, config, out_dir, run_uuid, out_filename='report.html',
                 sentry_sdk=None):
        self.root = path
        self.sections = []
        self.errors = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename
        self.run_uuid = run_uuid
        self.sentry_sdk = sentry_sdk
        self.template_path = None
        self.packagename = None

        self._load_config(config)

    def _load_config(self, config):
        config = Path(config)
        with config.open('r') as configfh:
            settings = json.load(configfh)

        self.packagename = settings.get('package', None)
        if self.packagename:
            self.out_dir = self.out_dir / self.packagename

        template_path = Path(settings.get('template_path', 'report.tpl'))
        if not str(template_path).startswith('/'):
            template_path = config.parent / template_path
        self.template_path = template_path.resolve()
        self.index(settings['sections'])

    def index(self, config):
        fig_dir = 'figures'
        subject_dir = self.root.split('/')[-1]
        subject = re.search('^(?P<subject_id>sub-[a-zA-Z0-9]+)$', subject_dir).group()
        svg_dir = self.out_dir / self.packagename / subject / fig_dir
        svg_dir.mkdir(parents=True, exist_ok=True)
        reportlet_list = list(sorted([str(f) for f in Path(self.root).glob('**/*.*')]))

        for subrep_cfg in config:
            reportlets = []
            for reportlet_cfg in subrep_cfg['reportlets']:
                rlet = Reportlet(**reportlet_cfg)
                for src in reportlet_list:
                    ext = src.split('.')[-1]
                    if rlet.file_pattern.search(src):
                        contents = None
                        if ext == 'html':
                            with open(src) as fp:
                                contents = fp.read().strip()
                        elif ext == 'svg':
                            fbase = Path(src).name
                            copyfile(src, str(svg_dir / fbase),
                                     copy=True, use_hardlink=True)
                            contents = str(Path(subject) / fig_dir / fbase)

                        if contents:
                            rlet.source_files.append(src)
                            rlet.contents.append(contents)

                if rlet.source_files:
                    reportlets.append(rlet)

            if reportlets:
                sub_report = SubReport(
                    subrep_cfg['name'], reportlets=reportlets,
                    title=subrep_cfg.get('title'))
                self.sections.append(order_by_run(sub_report))

        error_dir = self.out_dir / self.packagename / subject / 'log' / self.run_uuid
        if error_dir.is_dir():
            self.index_error_dir(error_dir)

    def index_error_dir(self, error_dir):
        """
        Crawl subjects crash directory for the corresponding run, report to sentry, and
        populate self.errors.
        """
        for crashfile in error_dir.glob('crash*.*'):
            crash_info = read_crashfile(str(crashfile))
            if self.sentry_sdk:
                with self.sentry_sdk.push_scope() as scope:
                    node_name = crash_info['node'].split('.')[-1]
                    # last line is probably most informative summary
                    gist = crash_info['traceback'].split('\n')[-1]
                    exception_text_start = 1
                    for line in crash_info['traceback'].split('\n')[1:]:
                        if not line[0].isspace():
                            break
                        exception_text_start += 1

                    exception_text = '\n'.join(crash_info['traceback'].split('\n')[
                                     exception_text_start:])

                    scope.set_tag("node_name", node_name)

                    chunk_size = 16384

                    for k, v in crash_info.items():
                        if k == 'inputs':
                            scope.set_extra(k, dict(v))
                        elif isinstance(v, str) and len(v) > chunk_size:
                            chunks = [v[i:i + chunk_size] for i in range(0, len(v), chunk_size)]
                            for i, chunk in enumerate(chunks):
                                scope.set_extra('%s_%02d' % (k, i), chunk)
                        else:
                            scope.set_extra(k, v)
                    scope.level = 'fatal'

                    # Group common events with pre specified fingerprints
                    fingerprint_dict = {
                        'permission-denied': [
                            "PermissionError: [Errno 13] Permission denied"],
                        'memory-error': ["MemoryError", "Cannot allocate memory"],
                        'reconall-already-running': [
                            "ERROR: it appears that recon-all is already running"],
                        'no-disk-space': [
                            "OSError: [Errno 28] No space left on device",
                            "[Errno 122] Disk quota exceeded"],
                        'sigkill': ["Return code: 137"],
                        'keyboard-interrupt': ["KeyboardInterrupt"],
                    }

                    fingerprint = ''
                    issue_title = node_name + ': ' + gist
                    for new_fingerprint, error_snippets in fingerprint_dict.items():
                        for error_snippet in error_snippets:
                            if error_snippet in crash_info['traceback']:
                                fingerprint = new_fingerprint
                                issue_title = new_fingerprint
                                break
                        if fingerprint:
                            break

                    message = issue_title + '\n\n'
                    message += exception_text[-(8192 - len(message)):]
                    if fingerprint:
                        self.sentry_sdk.add_breadcrumb([fingerprint], 'fatal')
                    else:
                        # remove file paths
                        fingerprint = re.sub(r"(/[^/ ]*)+/?", '', message)
                        # remove words containing numbers
                        fingerprint = re.sub(r"([a-zA-Z]*[0-9]+[a-zA-Z]*)+", '', fingerprint)
                        # adding the return code if it exists
                        for line in message.split('\n'):
                            if line.startswith("Return code"):
                                fingerprint += line
                                break

                    scope.fingerprint = [fingerprint]
                    self.sentry_sdk.capture_message(message, 'fatal')

            self.errors.append(crash_info)

    def generate_report(self):
        logs_path = self.out_dir / 'logs'

        boilerplate = []
        boiler_idx = 0

        if (logs_path / 'CITATION.html').exists():
            text = (logs_path / 'CITATION.html').read_text(encoding='UTF-8')
            text = '<div class="boiler-html">%s</div>' % re.compile(
                '<body>(.*?)</body>',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            boilerplate.append((boiler_idx, 'HTML', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.md').exists():
            text = '<pre>%s</pre>\n' % (logs_path / 'CITATION.md').read_text(encoding='UTF-8')
            boilerplate.append((boiler_idx, 'Markdown', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.tex').exists():
            text = (logs_path / 'CITATION.tex').read_text(encoding='UTF-8')
            text = re.compile(
                r'\\begin{document}(.*?)\\end{document}',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            text = '<pre>%s</pre>\n' % text
            text += '<h3>Bibliography</h3>\n'
            text += '<pre>%s</pre>\n' % Path(
                pkgrf(self.packagename, 'data/boilerplate.bib')).read_text(encoding='UTF-8')
            boilerplate.append((boiler_idx, 'LaTeX', text))
            boiler_idx += 1

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(sections=self.sections, errors=self.errors,
                                          boilerplate=boilerplate)

        # Write out report
        (self.out_dir / self.out_filename).write_text(report_render, encoding='UTF-8')
        return len(self.errors)


def order_by_run(subreport):
    ordered = []
    run_reps = {}

    for element in subreport.reportlets:
        if len(element.source_files) == 1 and element.source_files[0]:
            ordered.append(element)
            continue

        for filename, file_contents in zip(element.source_files, element.contents):
            name, title = generate_name_title(filename)
            if not filename or not name:
                continue

            new_element = Reportlet(
                name=element.name, title=element.title, file_pattern=element.file_pattern,
                description=element.description, raw=element.raw)
            new_element.contents.append(file_contents)
            new_element.source_files.append(filename)

            if name not in run_reps:
                run_reps[name] = SubReport(name, title=title)

            run_reps[name].reportlets.append(new_element)

    if run_reps:
        keys = list(sorted(run_reps.keys()))
        for key in keys:
            ordered.append(run_reps[key])
        subreport.isnested = True

    subreport.reportlets = ordered
    return subreport


def generate_name_title(filename):
    fname = Path(filename).name
    expr = re.compile(BIDS_NAME)
    outputs = expr.search(fname)
    if outputs:
        outputs = outputs.groupdict()
    else:
        return None, None

    name = '{session}{task}{acq}{rec}{run}'.format(
        session="_ses-" + outputs['session_id'] if outputs['session_id'] else '',
        task="_task-" + outputs['task_id'] if outputs['task_id'] else '',
        acq="_acq-" + outputs['acq_id'] if outputs['acq_id'] else '',
        rec="_rec-" + outputs['rec_id'] if outputs['rec_id'] else '',
        run="_run-" + outputs['run_id'] if outputs['run_id'] else ''
    )
    title = '{session}{task}{acq}{rec}{run}'.format(
        session=" Session: " + outputs['session_id'] if outputs['session_id'] else '',
        task=" Task: " + outputs['task_id'] if outputs['task_id'] else '',
        acq=" Acquisition: " + outputs['acq_id'] if outputs['acq_id'] else '',
        rec=" Reconstruction: " + outputs['rec_id'] if outputs['rec_id'] else '',
        run=" Run: " + outputs['run_id'] if outputs['run_id'] else ''
    )
    return name.strip('_'), title


def run_reports(reportlets_dir, out_dir, subject_label, run_uuid, config,
                sentry_sdk=None):
    """
    Runs the reports
    """
    reportlet_path = Path(reportlets_dir)
    reportlet_path = str(reportlet_path / ("sub-%s" % subject_label))
    out_filename = 'sub-{}.html'.format(subject_label)
    report = Report(reportlet_path, config, out_dir, run_uuid, out_filename,
                    sentry_sdk=sentry_sdk)
    return report.generate_report()


def generate_reports(subject_list, output_dir, work_dir, run_uuid, config,
                     sentry_sdk=None):
    """
    A wrapper to run_reports on a given ``subject_list``
    """
    reports_dir = str(Path(work_dir) / 'reportlets')
    report_errors = [
        run_reports(reports_dir, output_dir, subject_label, run_uuid, config,
                    sentry_sdk=sentry_sdk)
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging
        logger = logging.getLogger('cli')
        error_list = ', '.join('%s (%d)' % (subid, err)
                               for subid, err in zip(subject_list, report_errors) if err)
        logger.error(
            'Preprocessing did not finish successfully. Errors occurred while processing '
            'data from participants: %s. Check the HTML reports for details.', error_list)
    return errno
