# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from niworkflows,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
"""Core objects representing reports."""

import re
from collections import defaultdict
from itertools import compress
from pathlib import Path

import jinja2
import yaml
from bids.layout import BIDSLayout, BIDSLayoutIndexer, add_config_paths
from bids.layout.writing import build_path
from nireports.assembler import data
from nireports.assembler.reportlet import Reportlet

# Add a new figures spec
try:
    add_config_paths(figures=data.load('nipreps.json'))
except ValueError as e:
    if "Configuration 'figures' already exists" != str(e):
        raise

PLURAL_SUFFIX = defaultdict('s'.format, [('echo', 'es')])

OUTPUT_NAME_PATTERN = [
    'sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_ce-{ceagent}]'
    '[_rec-{reconstruction}][_dir-{direction}][_run-{run}][_echo-{echo}][_part-{part}]'
    '[_space-{space}][_cohort-{cohort}][_desc-{desc}][_{suffix<bold|sbref>}]'
    '{extension<.html|.svg>|.html}',
    'sub-{subject}[_ses-{session}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]'
    '[_dir-{direction}][_run-{run}][_space-{space}][_cohort-{cohort}][_desc-{desc}]'
    '[_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|'
    'angio|dseg|mask|dwi|epiref|T2starw|MTw|TSE>}]{extension<.html|.svg>|.html}',
    # "sub-{subject}[_ses-{session}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]"
    # "[_run-{run}][_space-{space}][_cohort-{cohort}][_fmapid-{fmapid}][_desc-{desc}]_"
    # "{suffix<fieldmap>}{extension<.html|.svg>|.html}",
]


class SubReport:
    """SubReports are sections within a Report."""

    __slots__ = {
        'name': 'a unique subreport name',
        'title': 'a header for the content included in the subreport',
        'reportlets': 'the collection of reportlets in this subreport',
        'isnested': '``True`` if this subreport is a reportlet to another subreport',
    }

    def __init__(self, name, isnested=False, reportlets=None, title=''):
        self.name = name
        self.title = title
        self.reportlets = reportlets or []
        self.isnested = isnested


class Report:
    """
    The full report object. This object maintains a BIDSLayout to index
    all reportlets.

    .. testsetup::

       >>> from shutil import copytree
       >>> from bids.layout import BIDSLayout
       >>> from nireports.assembler import data
       >>> test_data_path = data.load()
       >>> testdir = Path(tmpdir)
       >>> data_dir = copytree(
       ...     test_data_path / 'tests' / 'work',
       ...     str(testdir / 'work'),
       ...     dirs_exist_ok=True,
       ... )
       >>> REPORT_BASELINE_LENGTH = 41770
       >>> RATING_WIDGET_LENGTH = 83308


    Examples
    --------
    Output naming can be automated or customized:

    >>> summary_meta = {
    ...     "Summary": {
    ...         "Structural images": 1,
    ...         "FreeSurfer reconstruction": "Pre-existing directory",
    ...         "Output spaces":
    ...             "<code>MNI152NLin2009cAsym</code>, <code>fsaverage5</code>",
    ...     }
    ... }
    >>> robj = Report(
    ...     output_dir,
    ...     'madeoutuuid',
    ...     bootstrap_file=test_data_path / "tests" / "minimal.yml"
    ... )
    >>> str(robj.out_filename)  # doctest: +ELLIPSIS
    '.../report.html'

    >>> robj = Report(
    ...     output_dir,
    ...     'madeoutuuid',
    ...     bootstrap_file=test_data_path / "tests" / "minimal.yml",
    ...     out_filename="override.html"
    ... )
    >>> str(robj.out_filename) == str(output_dir / "override.html")
    True

    When ``bids_filters`` are available, the report will take up all the
    entities for naming.
    Therefore, the user must be careful to only include the necessary
    entities:

    >>> robj = Report(
    ...     output_dir,
    ...     'madeoutuuid',
    ...     bootstrap_file=test_data_path / "tests" / "minimal.yml",
    ...     metadata={"summary-meta": summary_meta},
    ...     subject="17",
    ...     acquisition="mprage",
    ...     suffix="T1w",
    ... )
    >>> str(robj.out_filename)
    '.../sub-17_acq-mprage_T1w.html'

    Report generation, first with a bootstrap file that contains special
    reportlets (namely, "errors" and "boilerplate").
    The first generated test does not have errors, and the CITATION files
    are missing (failing the boilerplate generation):

    >>> robj = Report(
    ...     output_dir / 'nireports',
    ...     'madeoutuuid',
    ...     reportlets_dir=testdir / 'work' / 'reportlets' / 'nireports',
    ...     plugins=[],
    ...     metadata={"summary-meta": summary_meta},
    ...     subject='01',
    ... )
    >>> robj.generate_report()
    0

    Test including a crashfile, but no CITATION files (therefore, failing
    boilerplate generation):

    >>> robj = Report(
    ...     output_dir / 'nireports',
    ...     'madeoutuuid02',
    ...     reportlets_dir=testdir / 'work' / 'reportlets' / 'nireports',
    ...     plugins=[],
    ...     metadata={"summary-meta": summary_meta},
    ...     subject='02',
    ... )
    >>> robj.generate_report()
    0

    Test including CITATION files (i.e., boilerplate generation is successful)
    and no crashfiles (no errors reported):

    >>> robj = Report(
    ...     output_dir / 'nireports',
    ...     'madeoutuuid03',
    ...     reportlets_dir=testdir / 'work' / 'reportlets' / 'nireports',
    ...     plugins=[],
    ...     metadata={"summary-meta": summary_meta},
    ...     subject='03',
    ... )
    >>> robj.generate_report()
    0

    >>> robj = Report(
    ...     output_dir / 'nireports',
    ...     'madeoutuuid03',
    ...     reportlets_dir=testdir / 'work' / 'reportlets' / 'nireports',
    ...     plugins=[{
    ...         "module": "nireports.assembler",
    ...         "path": "data/rating-widget/bootstrap.yml",
    ...     }],
    ...     metadata={"summary-meta": summary_meta},
    ...     subject='03',
    ...     task="faketaskwithruns",
    ...     suffix="bold",
    ... )
    >>> robj.generate_report()
    0

    Check contents (roughly, by length of the generated HTML file):

    >>> len((
    ...     output_dir / 'nireports' / 'sub-01.html'
    ... ).read_text()) - REPORT_BASELINE_LENGTH
    0
    >>> len((
    ...     output_dir / 'nireports' / 'sub-02.html'
    ... ).read_text()) - (REPORT_BASELINE_LENGTH + 3254)
    0
    >>> len((
    ...     output_dir / 'nireports' / 'sub-03.html'
    ... ).read_text()) - (REPORT_BASELINE_LENGTH + 51892)
    0

    >>> len((
    ...     output_dir / 'nireports' / 'sub-03_task-faketaskwithruns_bold.html'
    ... ).read_text()) - RATING_WIDGET_LENGTH
    0

    """

    __slots__ = {
        'title': 'the title that will be shown in the browser',
        'sections': 'a header for the content included in the subreport',
        'out_filename': 'output path where report will be stored',
        'template_path': 'location of a JINJA2 template for the output HTML',
        'header': 'plugins can modify the default HTML elements of the report',
        'navbar': 'plugins can modify the default HTML elements of the report',
        'footer': 'plugins can modify the default HTML elements of the report',
    }

    def __init__(
        self,
        out_dir,
        run_uuid,
        bootstrap_file=None,
        out_filename='report.html',
        reportlets_dir=None,
        plugins=None,
        plugin_meta=None,
        metadata=None,
        **bids_filters,
    ):
        out_dir = Path(out_dir)
        root = Path(reportlets_dir or out_dir)

        if bids_filters.get('subject'):
            subject_id = bids_filters['subject']
            bids_filters['subject'] = (
                subject_id[4:] if subject_id.startswith('sub-') else subject_id
            )

        if bids_filters.get('session'):
            session_id = bids_filters['session']
            bids_filters['session'] = (
                session_id[4:] if session_id.startswith('ses-') else session_id
            )

        if bids_filters and out_filename == 'report.html':
            out_filename = build_path(bids_filters, OUTPUT_NAME_PATTERN)

        metadata = metadata or {}
        if 'filename' not in metadata:
            metadata['filename'] = Path(out_filename).name.replace(
                ''.join(Path(out_filename).suffixes), ''
            )

        # Initialize structuring elements
        self.sections = []

        bootstrap_file = Path(bootstrap_file or data.load('default.yml'))

        bootstrap_text = []

        # Massage metadata for string interpolation in the template
        meta_repl = {
            'run_uuid': run_uuid if run_uuid is not None else 'null',
            'out_dir': str(out_dir),
            'reportlets_dir': str(root),
        }
        meta_repl.update({kk: vv for kk, vv in metadata.items() if isinstance(vv, str)})
        meta_repl.update(bids_filters)
        expr = re.compile(f'{{({"|".join(meta_repl.keys())})}}')

        for line in bootstrap_file.read_text().splitlines(keepends=False):
            if expr.search(line):
                line = line.format(**meta_repl)
            bootstrap_text.append(line)

        # Load report schema (settings YAML file)
        settings = yaml.safe_load('\n'.join(bootstrap_text))

        # Set the output path
        self.out_filename = Path(out_filename)
        if not self.out_filename.is_absolute():
            self.out_filename = Path(out_dir) / self.out_filename

        # Path to the Jinja2 template
        self.template_path = (
            Path(settings['template_path'])
            if 'template_path' in settings
            else data.load('report.tpl').absolute()
        )

        if not self.template_path.is_absolute():
            self.template_path = bootstrap_file / self.template_file

        assert self.template_path.exists()

        settings['root'] = root
        settings['out_dir'] = out_dir
        settings['run_uuid'] = run_uuid
        self.title = settings.get('title', 'Visual report generated by NiReports')

        settings['bids_filters'] = bids_filters or {}
        settings['metadata'] = metadata or {}
        self.index(settings)

        # Override plugins specified in the bootstrap with arg
        if plugins is not None or (plugins := settings.get('plugins', [])):
            settings['plugins'] = [
                yaml.safe_load(data.Loader(plugin['module']).readable(plugin['path']).read_text())
                for plugin in plugins
            ]

        self.process_plugins(settings, plugin_meta)

    def index(self, config):
        """
        Traverse the reports config definition and instantiate reportlets.

        This method also places figures in their final location.
        """
        # Initialize a BIDS layout
        _indexer = BIDSLayoutIndexer(
            config_filename=data.load('nipreps.json'),
            index_metadata=False,
            validate=False,
        )
        layout = BIDSLayout(
            config['root'],
            config='figures',
            indexer=_indexer,
            validate=False,
        )

        bids_filters = config.get('bids_filters', {})
        metadata = config.get('metadata', {})
        out_dir = Path(config['out_dir'])
        for subrep_cfg in config['sections']:
            # First determine whether we need to split by some ordering
            # (ie. sessions / tasks / runs), which are separated by commas.
            orderings = [s for s in subrep_cfg.get('ordering', '').strip().split(',') if s]
            entities, list_combos = self._process_orderings(orderings, layout.get(**bids_filters))

            if not list_combos:  # E.g. this is an anatomical reportlet
                reportlets = [
                    Reportlet(
                        layout,
                        config=cfg,
                        out_dir=out_dir,
                        bids_filters=bids_filters,
                        metadata=metadata,
                    )
                    for cfg in subrep_cfg['reportlets']
                ]
                list_combos = subrep_cfg.get('nested', False)
            else:
                raise Exception(entities, list_combos)
                # Do not use dictionary for queries, as we need to preserve ordering
                # of ordering columns.
                reportlets = []
                for c in list_combos:
                    # do not display entities with the value None.
                    c_filt = [
                        f'{key} <span class="bids-entity">{c_value}</span>'
                        for key, c_value in zip(entities, c)
                        if c_value is not None
                    ]
                    # Set a common title for this particular combination c
                    title = 'Reports for: %s.' % ', '.join(c_filt)

                    for cfg in subrep_cfg['reportlets']:
                        cfg['bids'].update({entities[i]: c[i] for i in range(len(c))})
                        rlet = Reportlet(
                            layout,
                            config=cfg,
                            out_dir=out_dir,
                            bids_filters=bids_filters,
                            metadata=metadata,
                        )
                        if not rlet.is_empty():
                            rlet.title = title
                            title = None
                            reportlets.append(rlet)

            # Filter out empty reportlets
            reportlets = [r for r in reportlets if not r.is_empty()]
            if reportlets:
                sub_report = SubReport(
                    subrep_cfg['name'],
                    isnested=bool(list_combos),
                    reportlets=reportlets,
                    title=subrep_cfg.get('title'),
                )
                self.sections.append(sub_report)

    def process_plugins(self, config, metadata=None):
        """Add components to header/navbar/footer to extend the default report."""
        self.header = []
        self.navbar = []
        self.footer = []

        plugins = config.get('plugins', None)
        for plugin in plugins or []:
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(
                    searchpath=str(Path(__file__).parent / 'data' / f'{plugin["type"]}')
                ),
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False,
            )

            plugin_meta = plugin.get('defaults', {})
            plugin_meta.update((metadata or {}).get(plugin['type'], {}))
            for member in ('header', 'navbar', 'footer'):
                old_value = getattr(self, member)
                setattr(
                    self,
                    member,
                    old_value
                    + [
                        env.get_template(f'{member}.tpl').render(
                            config=plugin,
                            metadata=plugin_meta,
                        )
                    ],
                )

    def generate_report(self):
        """Once the Report has been indexed, the final HTML can be generated"""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(
            title=self.title,
            sections=self.sections,
            header=self.header,
            navbar=self.navbar,
            footer=self.footer,
        )

        # Write out report
        self.out_filename.parent.mkdir(parents=True, exist_ok=True)
        self.out_filename.write_text(report_render, encoding='UTF-8')
        return 0

    @staticmethod
    def _process_orderings(orderings, hits):
        """
        Generate relevant combinations of orderings with observed values.

        Arguments
        ---------
        orderings : :obj:`list` of :obj:`list` of :obj:`str`
            Sections prescribing an ordering to select across sessions, acquisitions, runs, etc.
        hits : :obj:`list`
            The output of a BIDS query of the layout

        Returns
        -------
        entities: :obj:`list` of :obj:`str`
            The relevant orderings that had unique values
        value_combos: :obj:`list` of :obj:`tuple`
            Unique value combinations for the entities

        """
        # get a set of all unique entity combinations
        all_value_combos = {
            tuple(bids_file.get_entities().get(k, None) for k in orderings) for bids_file in hits
        }
        # remove the all None member if it exists
        none_member = tuple([None for k in orderings])
        if none_member in all_value_combos:
            all_value_combos.remove(tuple([None for k in orderings]))
        # see what values exist for each entity
        unique_values = [
            {value[idx] for value in all_value_combos} for idx in range(len(orderings))
        ]
        # if all values are None for an entity, we do not want to keep that entity
        keep_idx = [
            False if (len(val_set) == 1 and None in val_set) or not val_set else True
            for val_set in unique_values
        ]
        # the "kept" entities
        entities = list(compress(orderings, keep_idx))
        # the "kept" value combinations
        value_combos = [tuple(compress(value_combo, keep_idx)) for value_combo in all_value_combos]
        # sort the value combinations alphabetically from the first entity to the last entity
        value_combos.sort(
            key=lambda entry: tuple(str(value) if value is not None else '0' for value in entry)
        )

        return entities, value_combos
