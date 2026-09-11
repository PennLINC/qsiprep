"""The b=0-to-anatomical reportlet must carry a GM/WM boundary contour.

Without landmarks the flicker panel shows two similar-looking brains and gives
no way to judge the coregistration -- the same reason the distortion correction
reportlets contour the white matter (see ``init_fmap_unwarp_report_wf``).
"""


def _coreg_wf(write_report=True, name='coreg_report'):
    from qsiprep import config
    from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf

    config.nipype.omp_nthreads = 1
    return init_b0_to_anat_registration_wf(
        write_report=write_report,
        transform_type='Rigid',
        name=name,
    )


def _connect(wf, source, target):
    return wf._graph.get_edge_data(wf.get_node(source), wf.get_node(target))['connect']


def test_wm_contour_reaches_the_reportlet():
    wf = _coreg_wf()
    names = [n.name for n in wf._get_all_nodes()]
    assert 'sel_wm' in names
    assert 'coreg_rpt' in names

    assert _connect(wf, 'inputnode', 'sel_wm') == [('t1_seg', 'in_seg')]
    assert _connect(wf, 'sel_wm', 'coreg_rpt') == [('out', 'wm_seg')]


def test_the_reportlet_flickers_the_two_coregistered_images():
    """``warped_image`` is the b=0 ANTs already resampled onto the anatomy."""
    wf = _coreg_wf(name='coreg_flicker')
    assert _connect(wf, 'inputnode', 'coreg_rpt') == [('t1_brain', 'before')]
    assert _connect(wf, 'b0_to_anat', 'coreg_rpt') == [('warped_image', 'after')]
    assert _connect(wf, 'coreg_rpt', 'outputnode') == [('out_report', 'report')]
    assert wf.get_node('b0_to_anat').inputs.output_warped_image


def test_the_contour_is_not_resampled():
    """Both panels and the segmentation live in the anatomical space.

    The reportlet warps the b=0 into the anatomy rather than the other way
    round, so ``t1_seg`` needs no transform -- unlike the SDC reportlet, which
    plots in b=0 space and has to pull the segmentation through the inverse.
    """
    wf = _coreg_wf(name='coreg_no_resample')
    assert not any(n.name.startswith('map_seg') for n in wf._get_all_nodes())


def test_no_report_nodes_when_no_report_is_written():
    """Callers that skip the reportlet (synb0) never connect ``t1_seg``."""
    wf = _coreg_wf(write_report=False, name='coreg_no_report')
    names = [n.name for n in wf._get_all_nodes()]
    assert 'sel_wm' not in names
    assert 'coreg_rpt' not in names


def test_acpc_reg_does_not_render_a_discarded_reportlet():
    """``init_direct_b0_acpc_wf`` reports through ACPCReport, not the registration.

    The registration node used to be an ``ANTSRegistrationRPT`` with
    ``generate_report=True``, but its ``out_report`` was never connected -- an
    SVG rendered on every run and thrown away.
    """
    from nipype.interfaces.ants import Registration

    from qsiprep import config
    from qsiprep.workflows.dwi.registration import init_direct_b0_acpc_wf

    config.nipype.omp_nthreads = 1
    wf = init_direct_b0_acpc_wf(write_report=True, name='acpc_report')

    assert type(wf.get_node('acpc_reg').interface) is Registration
    assert _connect(wf, 'acpc_report', 'outputnode') == [('out_report', 'report')]


def test_acpc_reportlet_is_skipped_when_no_report_is_written():
    from qsiprep import config
    from qsiprep.workflows.dwi.registration import init_direct_b0_acpc_wf

    config.nipype.omp_nthreads = 1
    wf = init_direct_b0_acpc_wf(write_report=False, name='acpc_no_report')
    assert 'acpc_report' not in [n.name for n in wf._get_all_nodes()]


def test_fieldmap_coreg_reportlet_is_its_own_node(monkeypatch):
    """The desc-fmapCoreg figure comes from a reportlet, not the ANTs node."""
    from nipype.interfaces.ants import Registration

    from qsiprep import config
    from qsiprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf

    monkeypatch.setenv('FSLDIR', '/opt/fsl')
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    wf = init_sdc_unwarp_wf(name='sdc_unwarp_report')

    assert type(wf.get_node('fmap2ref_reg').interface) is Registration
    # The fieldmap reference resampled onto the EPI reference, flickered
    # against it -- the same pair the RPT interface used to plot internally.
    assert _connect(wf, 'fmap2ref_reg', 'fmap2ref_rpt') == [('warped_image', 'before')]
    assert _connect(wf, 'inputnode', 'fmap2ref_rpt') == [('in_reference_brain', 'after')]
    assert _connect(wf, 'fmap2ref_rpt', 'ds_report_reg') == [('out_report', 'in_file')]
    assert wf.get_node('fmap2ref_reg').inputs.output_warped_image
