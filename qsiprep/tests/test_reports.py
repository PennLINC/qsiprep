"""Tests for visual report assembly."""

from pathlib import Path

import pytest


def test_subject_summary_counts_inputs_uniquely():
    """A file in several outputs (virtual acquisition mode) is one input, not N."""
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        subject_id='01',
        template='MNI152NLin2009cAsym',
        dwi_groupings={
            'sub-01_acq-solo_dir-AP': {
                'pe_dir': 'j-',
                'dwi_files': ['sub-01_dir-AP_dwi.nii.gz'],
                'fieldmap': None,
            },
            'sub-01_acq-pair': {
                'pe_dir': 'j-',
                'dwi_files': ['sub-01_dir-AP_dwi.nii.gz', 'sub-01_dir-PA_dwi.nii.gz'],
                'fieldmap': 'pepolar',
            },
        },
    )
    segment = summary._generate_segment()
    assert 'inputs 2, outputs 2' in segment


@pytest.fixture
def collect_reports(monkeypatch):
    """Replace run_reports with a recorder of the report directories and filenames."""
    from qsiprep.reports import core

    calls = []

    def _fake_run_reports(
        output_dir, subject_label, run_uuid, out_filename='report.html', **kwargs
    ):
        calls.append((Path(output_dir), out_filename))
        return None

    monkeypatch.setattr(core, 'run_reports', _fake_run_reports)
    return calls


def test_generate_reports_root_level(tmp_path, collect_reports):
    """Subject-wise reports are written to the output directory root."""
    from qsiprep.reports.core import generate_reports

    errors = generate_reports(
        processing_list=[['01', ['01', '02']]],
        subject_anatomical_reference='unbiased',
        report_output_level='root',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert not errors
    assert collect_reports == [(tmp_path, 'sub-01.html')]


def test_generate_reports_subject_level(tmp_path, collect_reports):
    """Subject-level reports are written into the subject directory."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', ['01', '02']]],
        subject_anatomical_reference='unbiased',
        report_output_level='subject',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [(tmp_path / 'sub-01', 'sub-01.html')]


def test_generate_reports_session_level(tmp_path, collect_reports):
    """Session-wise reports are written into the session directory."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', ['01']], ['01', ['02']]],
        subject_anatomical_reference='sessionwise',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [
        (tmp_path / 'sub-01' / 'ses-01', 'sub-01_ses-01.html'),
        (tmp_path / 'sub-01' / 'ses-02', 'sub-01_ses-02.html'),
    ]


def test_generate_reports_session_level_root_output(tmp_path, collect_reports):
    """Session-wise reports keep their session-specific names at the root level."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', ['01']]],
        subject_anatomical_reference='sessionwise',
        report_output_level='root',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [(tmp_path, 'sub-01_ses-01.html')]


def test_generate_reports_session_level_without_sessions(tmp_path, collect_reports, caplog):
    """Cross-sectional data fall back to subject-level reports with a warning."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', []]],
        subject_anatomical_reference='sessionwise',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [(tmp_path / 'sub-01', 'sub-01.html')]
    assert 'Writing out reports to subject level' in caplog.text


def test_generate_reports_session_level_with_subject_wise_reports(
    tmp_path, collect_reports, caplog
):
    """Reports spanning multiple sessions fall back to subject level with a warning."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', ['01', '02']]],
        subject_anatomical_reference='unbiased',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [(tmp_path / 'sub-01', 'sub-01.html')]
    assert 'Writing out reports to subject level' in caplog.text


def test_generate_reports_session_fallback_is_not_sticky(tmp_path, collect_reports):
    """A subject without sessions does not downgrade later subjects' reports."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['01', []], ['02', ['01']]],
        subject_anatomical_reference='sessionwise',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [
        (tmp_path / 'sub-01', 'sub-01.html'),
        (tmp_path / 'sub-02' / 'ses-01', 'sub-02_ses-01.html'),
    ]


def test_generate_reports_strips_entity_prefixes(tmp_path, collect_reports):
    """Subject and session labels may include their BIDS prefixes."""
    from qsiprep.reports.core import generate_reports

    generate_reports(
        processing_list=[['sub-01', ['ses-01']]],
        subject_anatomical_reference='sessionwise',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert collect_reports == [(tmp_path / 'sub-01' / 'ses-01', 'sub-01_ses-01.html')]


def test_generate_reports_session_level_finds_reportlets(tmp_path):
    """A report nested in a session directory still picks up reportlets at the output root."""
    from qsiprep.reports.core import generate_reports

    figures_dir = tmp_path / 'sub-01' / 'ses-01' / 'figures'
    figures_dir.mkdir(parents=True)
    reportlet = figures_dir / 'sub-01_ses-01_dseg.svg'
    reportlet.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>',
        encoding='utf-8',
    )

    errors = generate_reports(
        processing_list=[['01', ['01']]],
        subject_anatomical_reference='sessionwise',
        report_output_level='session',
        output_dir=tmp_path,
        run_uuid='madeoutuuid',
    )

    assert not errors
    out_report = tmp_path / 'sub-01' / 'ses-01' / 'sub-01_ses-01.html'
    assert out_report.is_file()
    # The reportlet is referenced relative to the report, not the output root
    assert f'src="./{reportlet.relative_to(out_report.parent)}"' in out_report.read_text(
        encoding='utf-8'
    )


def test_template_to_report_entities():
    from qsiprep.workflows.anatomical.volume import _template_to_report_entities

    assert _template_to_report_entities('MNI152NLin2009cAsym') == {
        'space': 'MNI152NLin2009cAsym',
    }
    assert _template_to_report_entities('MNIInfant+3') == {
        'space': 'MNIInfant',
        'cohort': '3',
    }


def test_anat_spatial_normalization_reportlet_allows_template_cohort(tmp_path):
    """MNIInfant reportlets use fMRIPrep-style space/cohort entities."""
    from nireports.assembler.report import Report

    from qsiprep import data

    figures_dir = tmp_path / 'sub-01' / 'figures'
    figures_dir.mkdir(parents=True)
    svg_reportlets = [
        figures_dir / 'sub-01_space-MNIInfant_cohort-3_T1w.svg',
        figures_dir / 'sub-01_dseg.svg',
        figures_dir / 'sub-01_desc-vsm_fieldmap.svg',
        figures_dir / 'sub-01_desc-fmapCoreg_fieldmap.svg',
        figures_dir / 'sub-01_desc-sdc_dwi.svg',
        figures_dir / 'sub-01_desc-b0ref_dwi.svg',
        figures_dir / 'sub-01_desc-shoreline_dwi.gif',
    ]
    for reportlet in svg_reportlets:
        reportlet.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>',
            encoding='utf-8',
        )

    html_reportlets = [
        figures_dir / 'sub-01_desc-grouping_T1w.html',
        figures_dir / 'sub-01_desc-summary_T1w.html',
        figures_dir / 'sub-01_desc-conform_T1w.html',
        figures_dir / 'sub-01_desc-about_T1w.html',
    ]
    for reportlet in html_reportlets:
        reportlet.write_text('<div>reportlet</div>', encoding='utf-8')

    out_report = tmp_path / 'report.html'
    robj = Report(
        tmp_path,
        'madeoutuuid',
        bootstrap_file=data.load('reports-spec.yml'),
        out_filename=out_report,
        reportlets_dir=tmp_path,
        subject='01',
    )

    assert robj.generate_report() == 0

    report_html = out_report.read_text(encoding='utf-8')
    assert 'Spatial normalization of the anatomical reference' in report_html
    for reportlet in [*svg_reportlets, *html_reportlets]:
        assert (
            reportlet.name in report_html or reportlet.read_text(encoding='utf-8') in report_html
        )


def test_gradient_plot_emits_interactive_iframe(tmp_path, monkeypatch):
    """GradientPlot writes a self-contained ``<iframe srcdoc>`` sampling-scheme
    reportlet with before/after panels, colored by source file, in a form that
    is charset-independent (pure ASCII)."""
    import html as html_lib
    import json
    import re

    import numpy as np

    from qsiprep.interfaces.reports import GradientPlot

    # Two DIPY-style (N, 3) split series that sample the same directions.
    dirs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    bvals = [0, 1000, 1000, 2000]
    for name in ('ap', 'pa'):
        np.savetxt(tmp_path / f'{name}.bvec', dirs)
        (tmp_path / f'{name}.bval').write_text(' '.join(str(b) for b in bvals))
    np.savetxt(tmp_path / 'final.bvec', np.vstack([dirs, dirs]))

    monkeypatch.chdir(tmp_path)
    result = GradientPlot(
        orig_bvec_files=[str(tmp_path / 'ap.bvec'), str(tmp_path / 'pa.bvec')],
        orig_bval_files=[str(tmp_path / 'ap.bval'), str(tmp_path / 'pa.bval')],
        source_files=['sub-01_dir-AP_dwi.nii.gz'] * 4 + ['sub-01_dir-PA_dwi.nii.gz'] * 4,
        source_pe_dirs={'sub-01_dir-AP_dwi.nii.gz': 'j-', 'sub-01_dir-PA_dwi.nii.gz': 'j'},
        final_bvec_file=str(tmp_path / 'final.bvec'),
    ).run()

    out = Path(result.outputs.plot_file)
    assert out.suffix == '.html'
    markup = out.read_text()
    assert markup.startswith('<iframe')
    assert markup.isascii()  # decodes correctly under any host-page charset

    document = html_lib.unescape(re.search(r'srcdoc="(.*?)"\s', markup, re.S).group(1))
    payload = re.search(r'application/json">(\{.*?\})</script>', document, re.S).group(1)
    data = json.loads(payload.replace('<\\/', '</'))
    assert [panel['title'] for panel in data['panels']] == [
        'Acquired (original b-vectors)',
        'After preprocessing (rotated b-vectors)',
    ]
    assert data['files'] == ['sub-01_dir-AP_dwi.nii.gz', 'sub-01_dir-PA_dwi.nii.gz']
    assert data['pes'] == ['j-', 'j']  # colorable by phase encoding
    assert {point['pe'] for point in data['meta']} == {'j-', 'j'}
    assert len(data['meta']) == 8
