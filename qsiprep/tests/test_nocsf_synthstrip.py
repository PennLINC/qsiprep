"""``--force no-csf-synthstrip`` only changes the nonlinear registration mask.

qsirecon resamples qsiprep's anatomical ``desc-brain_mask`` onto the DWI grid
and feeds it to dwi2response/dwi2fod/mtnormalise, so the saved masks must
keep the default SynthStrip model. The tighter ``--no-csf`` mask is only ever
the moving-image mask of the SyN registration to the template.
"""

from pathlib import Path

import pytest


def _config(force_nocsf):
    from qsiprep import config

    config.execution.sloppy = True
    config.execution.output_dir = Path('/tmp/qsiprep_test_out')
    config.execution.skip_anat_based_spatial_normalization = False
    config.nipype.omp_nthreads = 1
    config.workflow.anat_modality = 'T1w'
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    config.workflow.hmc_method = 'eddy'
    config.workflow.sdc_method = 'topup'
    config.workflow.dwi2anat_dof = 6
    config.workflow.force_nocsf_synthstrip = force_nocsf
    return config


def _build(force_nocsf, name):
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    _config(force_nocsf)
    return init_anat_preproc_wf(
        num_anat_images=1,
        num_additional_t2ws=0,
        has_rois=False,
        anatomical_template='MNI152NLin2009cAsym',
        name=name,
    )


def _connections_into(wf, dest):
    """Map source node name -> (src_field, dest_field) pairs feeding ``dest``."""
    return {
        src.name: wf._graph.get_edge_data(src, dest)['connect']
        for src in wf._graph.predecessors(dest)
    }


def _synthstrip_nodes(wf):
    """Map enclosing synthstrip workflow name -> its (mock) SynthStrip node."""
    return {
        n.fullname.split('.')[-2]: n for n in wf._get_all_nodes() if n.name == 'mocksynthstrip'
    }


def test_parser_sets_force_nocsf_synthstrip(tmp_path):
    from qsiprep.cli.parser import _build_parser

    bids = tmp_path / 'bids'
    bids.mkdir()
    base = [str(bids), str(tmp_path / 'out'), 'participant', '--output-resolution', '2']
    parser = _build_parser()
    assert parser.parse_args(base).force_nocsf_synthstrip is False
    opts = parser.parse_args([*base, '--force', 'no-csf-synthstrip'])
    assert opts.force == ['no-csf-synthstrip']
    assert opts.force_nocsf_synthstrip is True


def test_default_runs_one_synthstrip_and_uses_it_everywhere():
    wf = _build(False, 'nocsf_off')
    strips = _synthstrip_nodes(wf)
    assert list(strips) == ['synthstrip_anat_wf']
    assert strips['synthstrip_anat_wf'].inputs.no_csf is False

    into_norm = _connections_into(wf, wf.get_node('anat_normalization_wf'))
    assert into_norm['synthstrip_anat_wf'] == [
        ('outputnode.brain_mask', 'inputnode.brain_mask'),
        ('outputnode.brain_mask', 'inputnode.nonlinear_brain_mask'),
    ]


def test_force_adds_a_nocsf_synthstrip_for_the_nonlinear_mask_only():
    wf = _build(True, 'nocsf_on')
    strips = _synthstrip_nodes(wf)
    assert strips['synthstrip_anat_wf'].inputs.no_csf is False
    assert strips['synthstrip_anat_nocsf_wf'].inputs.no_csf is True

    norm = wf.get_node('anat_normalization_wf')
    into_norm = _connections_into(wf, norm)
    # AC-PC affine keeps the default mask; SyN gets the no-CSF mask.
    assert into_norm['synthstrip_anat_wf'] == [('outputnode.brain_mask', 'inputnode.brain_mask')]
    assert into_norm['synthstrip_anat_nocsf_wf'] == [
        ('outputnode.brain_mask', 'inputnode.nonlinear_brain_mask')
    ]
    into_acpc = _connections_into(norm, norm.get_node('acpc_reg'))
    assert ('brain_mask', 'moving_mask') in into_acpc['inputnode']
    into_syn_mask = _connections_into(norm, norm.get_node('rigid_acpc_resample_mask'))
    assert ('nonlinear_brain_mask', 'input_image') in into_syn_mask['inputnode']

    # Everything that becomes a derivative or feeds the DWI workflows still
    # descends from the default SynthStrip.
    for node_name in ('rigid_acpc_resample_mask', 'rigid_acpc_resample_brain'):
        into = _connections_into(wf, wf.get_node(node_name))
        assert 'synthstrip_anat_wf' in into
        assert 'synthstrip_anat_nocsf_wf' not in into
    nocsf_wf = wf.get_node('synthstrip_anat_nocsf_wf')
    assert [n.name for n in wf._graph.successors(nocsf_wf)] == ['anat_normalization_wf']


@pytest.mark.parametrize('force_nocsf', [False, True])
def test_boilerplate_mentions_the_nocsf_mask_only_when_forced(force_nocsf):
    wf = _build(force_nocsf, f'boiler_{int(force_nocsf)}')
    assert ('--no-csf' in wf.__postdesc__) is force_nocsf
