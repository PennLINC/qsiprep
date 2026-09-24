"""Tests for visual report assembly."""

from pathlib import Path
from types import SimpleNamespace

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


def test_subject_summary_warns_about_the_development_branch():
    """Put an unmissable warning in the report when unreleased MRtrix3 was used."""
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        subject_id='01',
        template='MNI152NLin2009cAsym',
        mrtrix_version='dev',
    )
    segment = summary._generate_segment()

    assert 'alert alert-warning' in segment
    assert 'role="alert"' in segment
    assert 'development branch' in segment.lower()
    assert '--mrtrix-version dev' in segment


def test_subject_summary_is_quiet_under_stable_mrtrix():
    """Show no warning banner for a released MRtrix3, which is the default."""
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        subject_id='01',
        template='MNI152NLin2009cAsym',
        mrtrix_version='stable',
    )
    segment = summary._generate_segment()

    assert 'alert' not in segment
    assert 'development branch' not in segment.lower()


def test_about_summary_records_the_mrtrix_installation():
    """State which MRtrix3 ran, warning or not: version and path both known."""
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='dev',
        mrtrix3_home='/opt/mrtrix3-dev',
        mrtrix3_version='3.0.8-2071-gb98b54e9',
    )._generate_segment()

    assert 'dev' in segment
    assert '/opt/mrtrix3-dev' in segment
    assert '3.0.8-2071-gb98b54e9' in segment
    assert 'None' not in segment
    assert '()' not in segment
    assert 'MRtrix3: dev 3.0.8-2071-gb98b54e9 (/opt/mrtrix3-dev)' in segment


def test_about_summary_omits_an_unknown_mrtrix_path():
    """Mark the version as unresolved, not a bare claim, when neither is declared.

    Nothing has actually been selected in this case: whatever MRtrix3 is on PATH ran,
    which may not be the requested version. The report must not assert a version it
    cannot back.
    """
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='stable',
    )._generate_segment()

    assert 'MRtrix3' in segment
    assert 'stable' in segment
    assert 'None' not in segment
    assert '()' not in segment
    assert 'resolved from PATH' in segment
    assert 'MRtrix3: stable (requested; no declared installation, resolved from PATH)' in segment


def test_about_summary_records_a_version_with_no_declared_path():
    """Mark the version as unresolved when it is known but no install path is declared."""
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='dev',
        mrtrix3_version='3.0.8-2071-gb98b54e9',
    )._generate_segment()

    assert 'None' not in segment
    assert '()' not in segment
    assert 'resolved from PATH' in segment
    assert (
        'MRtrix3: dev 3.0.8-2071-gb98b54e9 '
        '(requested; no declared installation, resolved from PATH)'
    ) in segment


def test_about_summary_records_a_path_with_no_known_version():
    """Report the install path without a version claim when the version is unknown."""
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='stable',
        mrtrix3_home='/opt/mrtrix3-stable',
    )._generate_segment()

    assert 'None' not in segment
    assert '()' not in segment
    assert 'resolved from PATH' not in segment
    assert 'MRtrix3: stable (/opt/mrtrix3-stable)' in segment


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


def test_gradient_plot_emits_inline_scheme(tmp_path, monkeypatch):
    """GradientPlot writes a self-contained inline sampling-scheme reportlet
    (no iframe, so it flows in the report instead of scrolling in a fixed frame)
    with before/after panels, colored by source file."""
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
    # Inlines directly into the report: no iframe, so no fixed-height scroll frame.
    assert '<iframe' not in markup
    assert 'class="qspace-viewer"' in markup

    payload = re.search(r'application/json">(\{.*?\})</script>', markup, re.S).group(1)
    data = json.loads(payload.replace('<\\/', '</'))
    assert [panel['title'] for panel in data['panels']] == [
        'Acquired (original b-vectors)',
        'After preprocessing (rotated b-vectors)',
    ]
    assert data['files'] == ['sub-01_dir-AP_dwi.nii.gz', 'sub-01_dir-PA_dwi.nii.gz']
    assert data['pes'] == ['j-', 'j']  # colorable by phase encoding
    assert {point['pe'] for point in data['meta']} == {'j-', 'j'}
    assert len(data['meta']) == 8


@pytest.mark.parametrize(
    ('warp_dim', 'basis', 'expected'),
    [
        ('3D', 'metadata', '3D (from ImageType)'),
        ('1D', 'metadata', 'through-plane only (ImageType: DIS2D)'),
        (None, 'metadata', 'b-matrix only (ImageType: DIS3D)'),
        ('3D', 'forced', 'forced 3D'),
        ('1D', 'forced', 'forced 1D'),
    ],
)
def test_describe_gradient_correction(warp_dim, basis, expected):
    from qsiprep.workflows.dwi.gradwarp import GradwarpPlan, describe_gradient_correction

    plan = GradwarpPlan(coeff_file='/opt/c.grad', warp_dim=warp_dim, is_ge=False, basis=basis)
    assert describe_gradient_correction(plan) == expected


def test_describe_gradient_correction_without_a_plan():
    from qsiprep.workflows.dwi.gradwarp import describe_gradient_correction

    assert describe_gradient_correction(None) == 'none'


def test_diffusion_summary_renders_gradient_correction():
    from qsiprep.interfaces.reports import DiffusionSummary

    summary = DiffusionSummary(
        distortion_correction='TOPUP',
        pe_direction='j',
        hmc_transform='Affine',
        hmc_model='eddy',
        dwi2anat_dof=6,
        denoise_method='dwidenoise',
        dwidenoise_window=5,
        gradient_correction='through-plane only (ImageType: DIS2D)',
    )
    assert 'through-plane only' in summary._generate_segment()


def _diffusion_summary(**overrides):
    from qsiprep.interfaces.reports import DiffusionSummary

    inputs = {
        'distortion_correction': 'TOPUP',
        'pe_direction': 'j',
        'hmc_model': 'eddy',
        'dwi2anat_dof': 6,
        'denoise_method': 'dwidenoise',
        'dwidenoise_window': 5,
    }
    inputs.update(overrides)
    return DiffusionSummary(**inputs)


def test_diffusion_summary_omits_hmc_transform_when_undefined():
    segment = _diffusion_summary()._generate_segment()
    assert 'HMC Transform' not in segment
    assert 'HMC Model: eddy' in segment


def test_diffusion_summary_shows_hmc_transform_when_given():
    segment = _diffusion_summary(hmc_model='3dSHORE', hmc_transform='Rigid')._generate_segment()
    assert '<li>HMC Transform: Rigid</li>' in segment
    assert 'HMC Model: 3dSHORE' in segment


# --- SeriesQC: n/a values and the QC warnings reportlet ----------------------


def _merged_qc_csv(path, warning='', neighbor_corr=0.99):
    """A merged_qc.csv as DSIStudioMergeQC writes it."""
    import pandas as pd

    from qsiprep.interfaces.dsi_studio import QC_WARNINGS_COLUMN

    pd.DataFrame(
        {
            'neighbor_corr': [neighbor_corr],
            'coherence_index': [0.4],
            QC_WARNINGS_COLUMN: [warning],
        }
    ).to_csv(path, index=False)
    return str(path)


def _run_series_qc(tmp_path, monkeypatch, pre_qc, t1_qc=None):
    import qsiprep.interfaces.reports as reports_mod
    from qsiprep.interfaces.reports import SeriesQC

    # Motion summary is not what these tests are about; it needs a full
    # confounds table otherwise.
    monkeypatch.setattr(reports_mod, 'calculate_motion_summary', lambda _: {'mean_fd': [0.1]})
    confounds = tmp_path / 'confounds.tsv'
    confounds.touch()

    interface = SeriesQC(
        pre_qc=pre_qc,
        confounds_file=str(confounds),
        output_file_name='sub-01_ses-1_dwi',
    )
    if t1_qc is not None:
        interface.inputs.t1_qc = t1_qc
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    return interface._results


def test_series_qc_writes_missing_values_as_bids_na(tmp_path, monkeypatch):
    """BIDS spells a missing TSV value n/a, not an empty cell."""
    results = _run_series_qc(
        tmp_path,
        monkeypatch,
        _merged_qc_csv(tmp_path / 'pre.csv'),
        t1_qc=_merged_qc_csv(tmp_path / 't1.csv', neighbor_corr=float('nan')),
    )

    lines = open(results['series_qc_file']).read().splitlines()
    row = dict(zip(lines[0].split('\t'), lines[1].split('\t'), strict=True))
    assert row['t1_neighbor_corr'] == 'n/a'
    assert row['raw_neighbor_corr'] == '0.99'


def test_series_qc_keeps_the_warning_column_out_of_the_table(tmp_path, monkeypatch):
    results = _run_series_qc(tmp_path, monkeypatch, _merged_qc_csv(tmp_path / 'pre.csv'))

    header = open(results['series_qc_file']).readline()
    assert 'qc_warnings' not in header


def test_series_qc_writes_no_report_when_qc_succeeded(tmp_path, monkeypatch):
    """DerivativesMaybeDataSink then writes nothing, so the report is quiet."""
    results = _run_series_qc(tmp_path, monkeypatch, _merged_qc_csv(tmp_path / 'pre.csv'))

    assert 'qc_warnings_report' not in results


def test_series_qc_reports_which_stage_failed_and_why(tmp_path, monkeypatch):
    results = _run_series_qc(
        tmp_path,
        monkeypatch,
        _merged_qc_csv(tmp_path / 'pre.csv'),
        t1_qc=_merged_qc_csv(
            tmp_path / 't1.csv',
            warning='SRC QC: DSI Studio was killed by SIGSEGV.',
            neighbor_corr=float('nan'),
        ),
    )

    report = open(results['qc_warnings_report']).read()
    assert 'class="alert alert-warning"' in report
    assert '<li>Resampled data: SRC QC: DSI Studio was killed by SIGSEGV.</li>' in report
    assert 'Raw data' not in report  # that stage succeeded


def test_series_qc_escapes_the_warning_text(tmp_path, monkeypatch):
    results = _run_series_qc(
        tmp_path,
        monkeypatch,
        _merged_qc_csv(tmp_path / 'pre.csv', warning='SRC QC: <b>odd</b> & worse'),
    )

    report = open(results['qc_warnings_report']).read()
    assert '&lt;b&gt;odd&lt;/b&gt; &amp; worse' in report


def test_diffusion_summary_warns_only_under_dwi_biascorrect_auto():
    """`auto` is a heuristic over metadata, so the report says so.

    How well the ImageType check generalises across vendors and sequences is not
    established, so a run that let it decide carries a warning box. An explicit
    n4/none run does not.
    """
    for mode in ('n4', 'none'):
        segment = _diffusion_summary(
            dwi_biascorrect=mode, dwi_biascorrect_applied=(mode == 'n4')
        )._generate_segment()
        assert 'alert-warning' not in segment

    segment = _diffusion_summary(
        dwi_biascorrect='auto', dwi_biascorrect_applied=False
    )._generate_segment()
    assert 'alert-warning' in segment


def test_diffusion_summary_reports_the_resolved_biascorrect_outcome():
    """Under `auto` the mode alone cannot say whether N4 ran, so state the outcome."""
    applied = _diffusion_summary(
        dwi_biascorrect='auto', dwi_biascorrect_applied=True
    )._generate_segment()
    skipped = _diffusion_summary(
        dwi_biascorrect='auto', dwi_biascorrect_applied=False
    )._generate_segment()

    assert 'applied' in applied
    assert 'skipped' in skipped
    assert applied != skipped
