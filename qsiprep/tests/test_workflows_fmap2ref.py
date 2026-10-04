"""The fieldmap-to-EPI registration: whole-head images, masked metric, no matching, rigid.

On the TRXScan phasediff fixture (truth = the simulated fieldmap offset; a realistic
magnitude with scalp, receive bias and Gibbs ringing) the brain-cropped images with histogram
matching landed ``fmap2ref_reg`` 4.6 deg off, matching off alone 2.6 deg, and the brain masks
as metric masks 0.5-0.8 deg. The production settings' Affine stage absorbed the EPI distortion
as a 6-12 % scale plus a 4-5 deg tilt, so it is Rigid now.

What a rigid cannot absorb it splits: registering the undistorted magnitude to the distorted
b=0 leaves a shift along the phase-encoding axis. The second pass registers to the reference
unwarped with the first pass's field, and the field is applied through it.
"""

import json

import pytest

from qsiprep.data import load as load_data


def _edges_into(wf, node_name):
    node = wf.get_node(node_name)
    return [(u.name, d['connect']) for u, v, d in wf._graph.in_edges(node, data=True)]


def _sources(wf, node_name):
    """Map each input of ``node_name`` to ``(source node, source field)``."""
    out = {}
    for name, conns in _edges_into(wf, node_name):
        for src, dst in conns:
            out[dst] = (name, src[0] if isinstance(src, tuple) else src)
    return out


@pytest.fixture
def unwarp_wf(monkeypatch):
    from qsiprep import config
    from qsiprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf

    monkeypatch.setenv('FSLDIR', '/opt/fsl')
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    return init_sdc_unwarp_wf(name='unwarp')


def test_fmap2ref_registers_to_the_whole_head_reference(unwarp_wf):
    conns = _sources(unwarp_wf, 'fmap2ref_reg')
    assert conns['fixed_image'] == ('inputnode', 'in_reference')
    assert conns['fixed_image_masks'] == ('inputnode', 'in_mask')
    assert conns['moving_image_masks'] == ('inputnode', 'fmap_mask')
    assert conns['moving_image'] == ('inputnode', 'fmap_ref')


def test_second_pass_registers_to_the_unwarped_reference(unwarp_wf):
    """The second registration sees the reference unwarped with the first field."""
    from nipype.interfaces.ants import Registration

    assert type(unwarp_wf.get_node('fmap2ref_reg2').interface) is Registration
    conns = _sources(unwarp_wf, 'fmap2ref_reg2')
    assert conns['fixed_image'] == ('fmap_apply_pass1_wf', 'outputnode.out_reference')
    assert conns['initial_moving_transform'] == ('fmap2ref_reg', 'composite_transform')
    assert conns['moving_image'] == ('inputnode', 'fmap_ref')
    assert conns['moving_image_masks'] == ('inputnode', 'fmap_mask')
    # the same metric masks; dilating the EPI mask for the unwarped reference measured worse
    assert conns['fixed_image_masks'] == ('inputnode', 'in_mask')
    # same settings as the first pass
    first = unwarp_wf.get_node('fmap2ref_reg').interface.inputs
    second = unwarp_wf.get_node('fmap2ref_reg2').interface.inputs
    assert second.transforms == first.transforms
    assert second.use_histogram_matching == first.use_histogram_matching
    assert second.metric == first.metric


def test_first_field_only_feeds_the_second_registration(unwarp_wf):
    """The first pass's field unwarps the reference for pass 2 and nothing else."""
    pass1 = _sources(unwarp_wf, 'fmap_apply_pass1_wf')
    assert pass1['inputnode.transforms'] == ('fmap2ref_reg', 'composite_transform')
    pass1_node = unwarp_wf.get_node('fmap_apply_pass1_wf')
    consumers = {v.name for u, v in unwarp_wf._graph.out_edges(pass1_node)}
    assert consumers == {'fmap2ref_reg2'}


def test_refinement_is_guarded_against_running_away(unwarp_wf):
    from qsiprep.interfaces.itk import GuardRefinedTransform

    assert type(unwarp_wf.get_node('guard_refinement').interface) is GuardRefinedTransform
    guard = _sources(unwarp_wf, 'guard_refinement')
    assert guard['initial_transform'] == ('fmap2ref_reg', 'composite_transform')
    assert guard['refined_transform'] == ('fmap2ref_reg2', 'composite_transform')
    assert guard['reference_image'] == ('inputnode', 'in_reference')


def test_outputs_come_from_the_guarded_second_transform(unwarp_wf):
    final = _sources(unwarp_wf, 'fmap_apply_wf')
    assert final['inputnode.transforms'] == ('guard_refinement', 'out_transform')
    assert final['inputnode.fmap'] == ('inputnode', 'fmap')
    out = _sources(unwarp_wf, 'outputnode')
    assert out['out_hz'] == ('fmap_apply_wf', 'outputnode.out_hz')
    assert out['out_warp'] == ('fmap_apply_wf', 'outputnode.out_warp')
    assert out['out_reference'] == ('apply_fov_mask', 'out_file')
    assert _sources(unwarp_wf, 'apply_fov_mask')['in_file'] == (
        'fmap_apply_wf',
        'outputnode.out_reference',
    )
    assert _sources(unwarp_wf, 'fmap_fov2ref_apply')['transforms'] == (
        'guard_refinement',
        'out_transform',
    )


def test_fmap_apply_wf_resamples_the_field_through_the_transform():
    from qsiprep.workflows.fieldmap.unwarp import init_fmap_apply_wf

    wf = init_fmap_apply_wf(name='apply')
    apply = _sources(wf, 'fmap2ref_apply')
    assert apply['transforms'] == ('inputnode', 'transforms')
    assert apply['input_image'] == ('inputnode', 'fmap')
    assert apply['reference_image'] == ('inputnode', 'in_reference')
    # Hz on the reference grid, no unit conversion in between
    assert _sources(wf, 'outputnode')['out_hz'] == ('fmap2ref_apply', 'output_image')
    assert _sources(wf, 'torads')['in_file'] == ('fmap2ref_apply', 'output_image')
    assert _sources(wf, 'gen_vsm')['in_file'] == ('torads', 'out_file')
    assert _sources(wf, 'vsm2dfm')['in_file'] == ('gen_vsm', 'shift_out_file')
    assert _sources(wf, 'unwarp_reference')['transforms'] == ('vsm2dfm', 'out_file')
    assert _sources(wf, 'outputnode')['out_reference'] == ('unwarp_reference', 'output_image')
    assert _sources(wf, 'outputnode')['out_warp'] == ('vsm2dfm', 'out_file')


def test_fmap2ref_settings_are_rigid():
    for name in ('fmap-any_registration.json', 'fmap-any_registration_testing.json'):
        settings = json.loads(load_data(name).read_text())
        assert 'Affine' not in settings['transforms'], name
        assert all(t in ('Translation', 'Rigid') for t in settings['transforms']), name
        assert not any(settings['use_histogram_matching']), name
