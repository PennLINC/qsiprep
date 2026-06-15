"""Tests for visual report assembly."""


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
