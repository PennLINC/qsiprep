"""Construction tests for the unpile fieldmap-less correction on every head motion backend."""

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.workflows.fieldmap.base import init_sdc_wf, unpile_target
from qsiprep.workflows.fieldmap.unpile import (
    init_unpile_sdc_wf,
    unpile_fit_grid_mm,
    unpile_fit_settings,
)

SRC = '/data/sub-01/dwi/sub-01_dwi.nii.gz'
T2W = ['/data/sub-01/anat/sub-01_T2w.nii.gz']
T1W = ['/data/sub-01/anat/sub-01_T1w.nii.gz']


@pytest.fixture(autouse=True)
def _config():
    saved = {
        'hmc_method': config.workflow.hmc_method,
        'sdc_method': config.workflow.sdc_method,
        'two_pass_sdc': config.workflow.two_pass_sdc,
        'anat_modality': config.workflow.anat_modality,
        'sloppy': config.execution.sloppy,
        'omp_nthreads': config.nipype.omp_nthreads,
    }
    config.workflow.hmc_method = 'eddy'
    config.workflow.sdc_method = None
    config.workflow.two_pass_sdc = True
    config.workflow.anat_modality = 'T1w'
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.execution.sloppy = False
    config.nipype.omp_nthreads = 2
    yield
    for key, value in saved.items():
        target = config.execution if key == 'sloppy' else config.nipype
        if key in ('hmc_method', 'sdc_method', 'two_pass_sdc', 'anat_modality'):
            target = config.workflow
        setattr(target, key, value)


def _unit(reference='t2w', hmc='eddy', readout_time=0.05):
    config.workflow.hmc_method = hmc
    anat = T2W if reference == 't2w' else T1W
    return make_preproc_unit(
        [SRC],
        method=CorrectionMethod.UNPILE,
        estimation_sources=anat,
        anat_files=anat,
        reference=reference,
        readout_time=readout_time,
    )


def _edge(wf, src, dst):
    edge = wf._graph.get_edge_data(wf.get_node(src), wf.get_node(dst))
    return (
        []
        if edge is None
        else [(s if isinstance(s, str) else s[0], d) for s, d in edge['connect']]
    )


def test_fit_grid_follows_two_thirds_of_the_epi_voxel():
    assert unpile_fit_grid_mm(3.0) == 2.0
    assert unpile_fit_grid_mm(2.4) == 1.6
    assert unpile_fit_grid_mm(2.0) == 1.5  # floored
    assert unpile_fit_grid_mm(1.7) == 1.5
    assert unpile_fit_grid_mm(1.5) is None  # already fine: native grid
    assert unpile_fit_grid_mm(1.2) is None


def test_fit_settings_per_reference():
    assert unpile_fit_settings('synb0') == {'metric': 'ssd', 'lam_mm': 5.0}
    assert unpile_fit_settings('t2w') == {'metric': 'lncc', 'lam_mm': 60.0, 'pose': True}
    sloppy = unpile_fit_settings('t2w', sloppy=True)
    assert sloppy['levels'] == '24,12,6'
    assert sloppy['iters'] == 40


def test_unpile_target_reads_the_plan_stage():
    import dataclasses

    assert unpile_target(_unit('t2w')) == 't2w'
    assert unpile_target(_unit('synb0')) == 'synb0'
    assert unpile_target(make_preproc_unit([SRC])) is None
    # The GRE candidate gre_seed_unit puts in front of an unpile unit keeps the unit's
    # plan but is the GRE estimation's: init_gre_seed_wf must not build a second fit.
    gre = make_preproc_unit(
        [SRC],
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01/fmap/sub-01_phasediff.nii.gz'],
    )
    seeded = dataclasses.replace(_unit('t2w'), estimation=gre.estimation)
    assert seeded.run.stage_with('unpile') is not None
    assert unpile_target(seeded) is None


@pytest.mark.parametrize('hmc', ['eddy', 'shoreline', 'tortoise'])
def test_sdc_wf_runs_unpile_with_every_hmc_method(hmc):
    unit = _unit('t2w', hmc=hmc)
    wf = init_sdc_wf(unit, source_file=SRC)
    assert wf.name == 'sdc_wf'
    unpile_wf = wf.get_node('unpile_sdc_wf')
    assert unpile_wf is not None
    assert wf.get_node('outputnode').inputs.method == 'unpile (T2w)'
    assert ('outputnode.out_warp', 'out_warp') in _edge(wf, 'unpile_sdc_wf', 'outputnode')
    assert ('outputnode.out_reference', 'b0_ref') in _edge(wf, 'unpile_sdc_wf', 'outputnode')
    assert ('outputnode.out_hz', 'fieldmap_hz') in _edge(wf, 'unpile_sdc_wf', 'outputnode')
    assert ('t2w_unfatsat', 'inputnode.t2w_unfatsat') in _edge(wf, 'inputnode', 'unpile_sdc_wf')


def test_t2w_reference_is_aligned_then_fitted_on_a_fine_grid():
    unit = _unit('t2w')
    wf = init_unpile_sdc_wf(unit, source_file=SRC, target='t2w')

    assert wf.get_node('t2w_to_b0_wf') is not None
    assert wf.get_node('synb0_wf') is None
    assert ('outputnode.structural_aligned', 'reference') in _edge(
        wf, 't2w_to_b0_wf', 'prepare_unpile'
    )
    # The fit grid is decided at run time from the EPI header (unpile_fit_grid_mm).
    prepare = wf.get_node('prepare_unpile')
    assert prepare.inputs.fine_grid is True
    assert prepare.inputs.pe_dir == 'j'
    assert prepare.inputs.raw_dwi_file == SRC
    fit = wf.get_node('unpile_fit')
    assert fit.inputs.metric == 'lncc'
    assert fit.inputs.pose is True
    assert fit.inputs.lam_mm == 60.0
    assert fit.inputs.total_readout_time == 0.05
    assert fit.inputs.threads == 2
    assert ('acq_voxel_mm', 'acq_voxel_mm') in _edge(wf, 'prepare_unpile', 'unpile_fit')
    assert ('field_hz', 'field_hz') in _edge(wf, 'unpile_fit', 'unpile_field_to_warp')
    assert ('b0_ref', 'reference') in _edge(wf, 'inputnode', 'unpile_field_to_warp')
    assert ('pose_xfm', 'pose_xfm') in _edge(wf, 'unpile_fit', 'outputnode')
    assert wf.get_node('ds_unpile_score') is not None
    assert 'two thirds of the DWI voxel' in wf.__desc__
    assert 'rigid refinement of the T2w position' in wf.__desc__


def test_sloppy_keeps_the_fine_grid_with_the_short_schedule():
    config.execution.sloppy = True
    wf = init_unpile_sdc_wf(_unit('t2w'), source_file=SRC, target='t2w')
    assert wf.get_node('prepare_unpile').inputs.fine_grid is True
    fit = wf.get_node('unpile_fit')
    assert fit.inputs.levels == '24,12,6'
    assert fit.inputs.iters == 40


def test_synb0_reference_runs_two_passes_by_default():
    unit = _unit('synb0')
    wf = init_unpile_sdc_wf(unit, source_file=SRC, target='synb0')

    assert wf.get_node('t2w_to_b0_wf') is None
    for name in (
        'synb0_wf',
        'unpile_fit',
        'unwarp_b0_pass1',
        'modulate_unwarped_b0',
        'synb0_b0_ref_wf_pass2',
        'synb0_wf_pass2',
        'prepare_unpile_pass2',
        'unpile_fit_pass2',
        'unpile_field_to_warp_pass2',
    ):
        assert wf.get_node(name) is not None, name
    fit = wf.get_node('unpile_fit')
    assert fit.inputs.metric == 'ssd'
    assert fit.inputs.lam_mm == 5.0
    assert fit.inputs.pose is False
    assert wf.get_node('prepare_unpile').inputs.fine_grid is False
    # The first field unwarps and modulates the b=0 the second synthesis starts from...
    assert ('out_warp', 'warp') in _edge(wf, 'unpile_field_to_warp', 'modulate_unwarped_b0')
    assert ('out_file', 'inputnode.b0_template') in _edge(
        wf, 'modulate_unwarped_b0', 'synb0_b0_ref_wf_pass2'
    )
    # ...while the second fit still matches the original b=0 to the new synthetic one.
    assert ('b0_ref', 'epi') in _edge(wf, 'inputnode', 'prepare_unpile_pass2')
    assert ('outputnode.synthetic_b0', 'reference') in _edge(
        wf, 'synb0_wf_pass2', 'prepare_unpile_pass2'
    )
    # The last pass is what leaves the workflow.
    assert ('out_warp', 'out_warp') in _edge(wf, 'unpile_field_to_warp_pass2', 'outputnode')
    assert ('outputnode.synthetic_b0', 'synthetic_b0') in _edge(wf, 'synb0_wf_pass2', 'outputnode')
    assert 'fitted again' in wf.__desc__


def test_one_pass_sdc_skips_the_second_synthesis():
    unit = _unit('synb0')
    wf = init_unpile_sdc_wf(unit, source_file=SRC, target='synb0', two_pass=False)
    assert wf.get_node('synb0_wf_pass2') is None
    assert ('out_warp', 'out_warp') in _edge(wf, 'unpile_field_to_warp', 'outputnode')
    assert 'fitted again' not in wf.__desc__


def test_sdc_wf_synb0_target_writes_the_synb0_outputs():
    unit = _unit('synb0')
    wf = init_sdc_wf(unit, source_file=SRC)
    assert wf.get_node('outputnode').inputs.method == 'unpile (SynB0)'
    for name in ('ds_report_synb0_acquired', 'ds_report_synb0_unet', 'ds_synb0_dwiref'):
        assert wf.get_node(name) is not None, name


def test_gre_prior_is_applied_as_given_against_the_synb0():
    unit = _unit('synb0')
    wf = init_unpile_sdc_wf(unit, source_file=SRC, target='synb0', gre_prior=True)
    fit = wf.get_node('unpile_fit')
    assert fit.inputs.fieldmap_refine == 'none'
    assert ('fieldmap_hz', 'fieldmap') in _edge(wf, 'inputnode', 'prepare_unpile')
    assert ('fieldmap', 'fieldmap') in _edge(wf, 'prepare_unpile', 'unpile_fit')
    assert 'applied as given' in wf.__desc__
    # Both passes take the prior.
    assert ('fieldmap', 'fieldmap') in _edge(wf, 'prepare_unpile_pass2', 'unpile_fit_pass2')


def test_gre_prior_is_refined_in_pose_against_the_t2w():
    wf = init_unpile_sdc_wf(_unit('t2w'), source_file=SRC, target='t2w', gre_prior=True)
    assert wf.get_node('unpile_fit').inputs.fieldmap_refine == 'pose'
    assert 'its position refined' in wf.__desc__


def test_missing_readout_time_is_an_error():
    unit = _unit('t2w', readout_time=None)
    with pytest.raises(ValueError, match='TotalReadoutTime'):
        init_unpile_sdc_wf(unit, source_file=SRC, target='t2w')


def test_t2w_target_without_anatomical_processing_is_an_error():
    config.workflow.anat_modality = 'none'
    with pytest.raises(ValueError, match='anat-modality none'):
        init_sdc_wf(_unit('t2w'), source_file=SRC)


def test_unknown_target_is_an_error():
    with pytest.raises(ValueError, match='unpile target'):
        init_unpile_sdc_wf(_unit('t2w'), source_file=SRC, target='invt1w')
