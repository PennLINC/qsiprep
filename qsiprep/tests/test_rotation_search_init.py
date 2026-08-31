"""antsAI rotation-search initialization for inter-modality coregistrations.

The b=0 -> anat, b=0 -> template AC-PC, and T2w -> T1w registrations previously
started from a center(-of-mass) alignment only. A Mattes registration recovers
rotations of a few tens of degrees at most, so a subject positioned very
differently for the dMRI than for the anatomical (common in infant studies,
where the head can be rotated toward sideways between scans) converges to a
rotated local optimum. Each of these registrations now starts from the best
candidate of an ``antsAI`` grid search over rotations (+-90 degrees per axis)
run on 4 mm resamples of the inputs.

The intramodal b=0 template registrations and the SyN fieldmap registration are
deliberately untouched: their inputs share an orientation by construction, and
``affine.json`` (shared with the fieldmap workflow) must keep its
center-of-mass initialization.
"""

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
    assert search.inputs.search_factor == (20.0, 0.5)
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
    """CI runs get 8 candidate orientations instead of 1000."""
    from qsiprep.workflows.dwi.registration import init_rotation_search_wf

    _config(sloppy=True)
    wf = init_rotation_search_wf(name='sloppy_search')
    search = _node(wf, 'rotation_search')
    assert search.inputs.search_factor == (20.0, 0.1)

    _config(sloppy=False)
    wf = init_rotation_search_wf(name='precise_search')
    search = _node(wf, 'rotation_search')
    assert search.inputs.search_factor == (20.0, 0.5)


def test_search_runs_on_downsampled_images():
    """A ~1000-start search at full resolution would take hours."""
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
