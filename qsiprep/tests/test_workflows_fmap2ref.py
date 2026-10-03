"""The fieldmap-to-EPI registration: whole-head images, brain masks as metric masks, no
histogram matching, rigid.

On the TRXScan phasediff fixture (truth = the simulated fieldmap offset; a realistic
magnitude with scalp, receive bias and Gibbs ringing) the brain-cropped images with histogram
matching landed ``fmap2ref_reg`` 4.6 deg off, matching off alone 2.6 deg, and the brain masks
as metric masks 0.5-0.8 deg. The production settings' Affine stage absorbed the EPI distortion
as a 6-12 % scale plus a 4-5 deg tilt, so it is Rigid now.
"""

import json

from qsiprep.data import load as load_data


def _edges_into(wf, node_name):
    node = wf.get_node(node_name)
    return [(u.name, d['connect']) for u, v, d in wf._graph.in_edges(node, data=True)]


def test_fmap2ref_registers_to_the_whole_head_reference():
    from qsiprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf

    wf = init_sdc_unwarp_wf(name='unwarp')
    conns = {dst: src for name, cs in _edges_into(wf, 'fmap2ref_reg') for src, dst in cs}
    assert conns['fixed_image'] == 'in_reference'
    assert conns['fixed_image_masks'] == 'in_mask'
    assert conns['moving_image_masks'] == 'fmap_mask'


def test_fmap2ref_settings_are_rigid():
    for name in ('fmap-any_registration.json', 'fmap-any_registration_testing.json'):
        settings = json.loads(load_data(name).read_text())
        assert 'Affine' not in settings['transforms'], name
        assert all(t in ('Translation', 'Rigid') for t in settings['transforms']), name
        assert not any(settings['use_histogram_matching']), name
