"""antsAI rotation-search initialization for inter-modality coregistrations.

The b=0 -> anat, b=0 -> template AC-PC, and T2w -> T1w registrations previously
started from a center(-of-mass) alignment only. A Mattes registration recovers
rotations of a few tens of degrees at most, so a subject positioned very
differently for the dMRI than for the anatomical (common in infant studies,
where the head can be rotated toward sideways between scans) converges to a
rotated local optimum. Each of these registrations now starts from the best
candidate of an ``antsAI`` grid search over rotations (+-81 degrees per axis,
so that a 20-degree grid point lands next to identity) run on 4 mm resamples
of the inputs.

The intramodal b=0 template registrations and the SyN fieldmap registration are
deliberately untouched: their inputs share an orientation by construction, and
``affine.json`` (shared with the fieldmap workflow) must keep its
center-of-mass initialization.

The structural image handed to TORTOISE gets the same protection one layer
down: DRBUDDI and DIFFPREP's ``--epi T2Wreg`` (EPIREG) rigidly register their
structural input to the b=0 internally, but only from a center-of-mass
initialization (EPIREG has no multistart fallback at all), so the T2w is
pre-aligned into the b=0 frame with the antsAI search before TORTOISE sees it.
The synb0 structural target needs none of this: the synthetic b=0 is produced
on the native b=0 grid already.
"""

import pytest
from nipype.interfaces.base import isdefined


def _config(sloppy=False):
    from qsiprep import config

    config.execution.sloppy = sloppy
    config.nipype.omp_nthreads = 1
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    return config


def _node(wf, name):
    return next(n for n in wf._get_all_nodes() if n.name == name)


def _incoming_fields(wf, dest_name):
    """Destination fields of every top-level connection into a named node."""
    fields = set()
    for _src, dest, meta in wf._graph.edges(data=True):
        if dest.name == dest_name:
            for _src_field, dest_field in meta['connect']:
                fields.add(dest_field)
    return fields


def test_b0_to_anat_coreg_is_initialized_by_rotation_search():
    from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf

    _config()
    wf = init_b0_to_anat_registration_wf(write_report=False, name='b0_coreg_search')

    search = _node(wf, 'rotation_search')
    assert search.inputs.transform[0] == 'Rigid'
    assert search.inputs.search_factor == (20.0, 0.45)
    assert 'initial_moving_transform' in _incoming_fields(wf, 'b0_to_anat')

    coreg = _node(wf, 'b0_to_anat')
    assert not isdefined(coreg.inputs.initial_moving_transform_com)


def test_direct_acpc_is_initialized_by_rotation_search():
    """The AC-PC target is a template, so the search must also fit scale."""
    from qsiprep.workflows.dwi.registration import init_direct_b0_acpc_wf

    _config()
    wf = init_direct_b0_acpc_wf(write_report=False, name='b0_acpc_search')

    search = _node(wf, 'rotation_search')
    assert search.inputs.transform[0] == 'Similarity'
    assert 'initial_moving_transform' in _incoming_fields(wf, 'acpc_reg')

    acpc_reg = _node(wf, 'acpc_reg')
    assert not isdefined(acpc_reg.inputs.initial_moving_transform_com)


def test_t2w_coreg_is_initialized_by_rotation_search():
    from qsiprep.workflows.anatomical.volume import init_t2w_preproc_wf

    _config()
    wf = init_t2w_preproc_wf(num_t2ws=1, name='t2w_search')

    search = _node(wf, 'rotation_search')
    assert search.inputs.transform[0] == 'Rigid'
    assert 'initial_moving_transform' in _incoming_fields(wf, 't2_brain_to_t1_brain')

    coreg = _node(wf, 't2_brain_to_t1_brain')
    assert not isdefined(coreg.inputs.initial_moving_transform_com)


def test_sloppy_mode_narrows_the_search():
    """CI runs get 8 candidate orientations instead of 729."""
    from qsiprep.workflows.dwi.registration import init_rotation_search_wf

    _config(sloppy=True)
    wf = init_rotation_search_wf(name='sloppy_search')
    search = _node(wf, 'rotation_search')
    assert search.inputs.search_factor == (20.0, 0.1)

    _config(sloppy=False)
    wf = init_rotation_search_wf(name='precise_search')
    search = _node(wf, 'rotation_search')
    assert search.inputs.search_factor == (20.0, 0.45)


def test_grid_has_a_start_next_to_identity():
    """With arc 0.5 the 20-degree grid straddles zero (-10, +10); 0.45 lands at -1.

    Most studies roughly align head position, so the common case deserves a
    candidate next to the identity rather than 10 degrees off it.
    """
    import numpy as np

    from qsiprep.workflows.dwi.registration import init_rotation_search_wf

    _config()
    search = _node(init_rotation_search_wf(name='grid_search'), 'rotation_search')
    step, arc = search.inputs.search_factor
    angles = np.arange(-180 * arc, 180 * arc + 1e-6, step)
    assert np.abs(angles).min() < 2.0, angles


@pytest.mark.parametrize(
    ('ants_version', 'warns'),
    [('2.5.4', True), ('2.6.0', False), ('2.6.2', False), (None, False)],
)
def test_pre_2_6_0_ants_gets_a_warning(monkeypatch, ants_version, warns):
    """ANTs PR #1861 fixed multi-start state leaking between candidates."""
    from unittest import mock

    from nipype.interfaces.ants.base import Info

    from qsiprep.workflows.dwi.registration import init_rotation_search_wf

    config = _config()
    monkeypatch.setattr(Info, '_version', ants_version)
    logger = mock.Mock()
    monkeypatch.setattr(config.loggers, 'workflow', logger)
    init_rotation_search_wf(name=f'version_search_{warns}')
    assert logger.warning.called is warns


def test_search_runs_on_downsampled_images():
    """A several-hundred-start search at full resolution would take hours."""
    from qsiprep.workflows.dwi.registration import init_rotation_search_wf

    _config()
    wf = init_rotation_search_wf(name='downsample_search')
    for name in ('res_fixed', 'res_moving'):
        node = _node(wf, name)
        assert node.inputs.zooms == (4.0, 4.0, 4.0)
        assert node.inputs.smooth is True
    assert 'fixed_image' in _incoming_fields(wf, 'rotation_search')
    assert 'moving_image' in _incoming_fields(wf, 'rotation_search')


def test_shared_affine_settings_keep_com_for_the_fieldmap_workflow():
    """affine.json is also loaded by qsiprep.workflows.fieldmap.syn.

    The T2w workflow clears the center-of-mass initialization on its node
    instead of editing the file, so the file must keep the setting.
    """
    import json

    from qsiprep.data import load as load_data

    cfg = json.loads(load_data('affine.json').read_text())
    assert cfg.get('initial_moving_transform_com') == 1


def test_intermodal_acpc_settings_carry_no_initialization():
    """The com key would collide (xor) with the connected initial transform."""
    import json

    from qsiprep.data import load as load_data

    cfg = json.loads(load_data('intermodal_ACPC.json').read_text())
    assert 'initial_moving_transform_com' not in cfg


def _connect_fields(wf, src, dst):
    edge = wf._graph.get_edge_data(wf.get_node(src), wf.get_node(dst))
    return [] if edge is None else list(edge['connect'])


def test_drbuddi_t2w_is_prealigned_to_the_b0(tmp_path):
    """The raw ACPC T2w must not reach DRBUDDI; the aligned copy must."""
    from qsiprep.tests.test_workflows_native import _cfg, _rpe_unit
    from qsiprep.workflows.fieldmap import init_drbuddi_wf

    _cfg(hmc_model='tortoise', pepolar_method='DRBUDDI')
    wf = init_drbuddi_wf(_rpe_unit(tmp_path), t2w_sdc=True)

    assert wf.get_node('t2w_to_b0_wf') is not None
    assert ('outputnode.structural_aligned', 'structural_image') in _connect_fields(
        wf, 't2w_to_b0_wf', 'drbuddi'
    )
    assert ('t2w_unfatsat', 'structural_image') not in _connect_fields(wf, 'inputnode', 'drbuddi')
    # The alignment consumes the caller-provided b=0 reference
    assert ('b0_ref', 'inputnode.b0_ref') in _connect_fields(wf, 'inputnode', 't2w_to_b0_wf')


def test_drbuddi_without_t2w_builds_no_alignment(tmp_path):
    from qsiprep.tests.test_workflows_native import _cfg, _rpe_unit
    from qsiprep.workflows.fieldmap import init_drbuddi_wf

    _cfg(hmc_model='tortoise', pepolar_method='DRBUDDI')
    wf = init_drbuddi_wf(_rpe_unit(tmp_path), t2w_sdc=False)
    assert wf.get_node('t2w_to_b0_wf') is None


def test_t2wreg_structural_is_prealigned_to_the_b0():
    """EPIREG runs a single COM-initialized rigid with no multistart fallback."""
    from qsiprep.tests.test_interfaces_diffprep import _base_config, _build, _make_unit

    _base_config()
    wf = _build(_make_unit(None), t2w_sdc=True, name='dp_t2w_prealign')

    assert wf.get_node('t2w_to_b0_wf') is not None
    assert ('outputnode.structural_aligned', 'structural_image') in _connect_fields(
        wf, 't2w_to_b0_wf', 'diffprep'
    )
    assert ('t2w_unfatsat', 'structural_image') not in _connect_fields(wf, 'inputnode', 'diffprep')
    # The alignment target is the raw distorted b=0 average of this run
    assert ('b0_average', 'inputnode.b0_ref') in _connect_fields(wf, 't2wreg_b0s', 't2w_to_b0_wf')


def test_drbuddi_callers_supply_a_b0_reference_in_the_sdc_frame(tmp_path):
    """Every DRBUDDI caller feeds inputnode.b0_ref from a pre-SDC b=0.

    eddy: the pre-eddy b=0 reference; DIFFPREP: the corrected-series b=0
    average; SHORELine: the motion-corrected b=0 template.
    """
    from qsiprep.tests.test_interfaces_diffprep import _base_config, _build
    from qsiprep.tests.test_workflows_gradwarp import (
        _cfg_for_fsl,
        _cfg_for_shoreline,
        _fsl_wf,
        _rpe_unit,
        _shoreline_wf,
    )
    from qsiprep.tests.test_workflows_native import _rpe_unit as _native_rpe_unit

    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))
    assert ('outputnode.ref_image', 'inputnode.b0_ref') in _connect_fields(
        wf, 'pre_eddy_b0_ref_wf', 'drbuddi_sdc_wf'
    )

    _base_config()
    wf = _build(_native_rpe_unit(tmp_path), t2w_sdc=True, name='dp_drbuddi_b0ref')
    assert ('b0_average', 'inputnode.b0_ref') in _connect_fields(
        wf, 'extract_b0s', 'drbuddi_sdc_wf'
    )

    _cfg_for_shoreline(tmp_path)
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path))
    assert ('outputnode.final_template', 'inputnode.b0_ref') in _connect_fields(
        wf, 'dwi_hmc_wf', 'drbuddi_sdc_wf'
    )
