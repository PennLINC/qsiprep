"""Anatomical bias-correction control and mask-driven template merging.

Covers three defects found together:

1. ``--b1-biascorrect-stage`` governs only the DWIs, so there was no way to skip
   N4 on anatomicals -- a problem for console-normalized data, where N4 can add
   artifacts rather than remove them.
2. The anatomical merge registered whole heads. A rigid fit is then driven partly
   by face, jaw and neck, which move relative to the brain when head placement
   changes, so non-brain tissue could pull the brains out of alignment.
3. Add-on modalities (T2w) inherited both, because ``init_t2w_preproc_wf`` shares
   ``init_anat_template_wf`` with T1w.
"""

import pytest


def _config(**overrides):
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    for key, value in overrides.items():
        setattr(config.workflow, key, value)
    return config


def _has(wf, fragment):
    return any(fragment in name for name in wf.list_node_names())


@pytest.mark.parametrize('num_images', [1, 3])
@pytest.mark.parametrize('do_biascorr', [True, False])
def test_n4_is_gated_by_do_biascorr(num_images, do_biascorr):
    """N4 appears only when asked for, for both the single- and multi-image paths."""
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config()
    wf = init_anat_template_wf(num_images=num_images, do_biascorr=do_biascorr)
    assert _has(wf, 'n4_correct') is do_biascorr


def test_bias_corrected_port_is_populated_without_n4():
    """Downstream consumers read `bias_corrected`; it must be connected either way.

    The port name is not a promise -- with N4 off it carries the conformed image.
    Leaving it dangling would break ACPC normalization and the skull-on
    derivative, both of which read it.
    """
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config()
    wf = init_anat_template_wf(num_images=1, do_biascorr=False)
    targets = set()
    for _, v, data in wf._graph.edges(data=True):
        if v.name == 'outputnode':
            targets.update(dst for _, dst in data.get('connect', []))
    assert 'bias_corrected' in targets


def test_merge_registration_is_masked():
    """The multi-image merge must drive registration with a brain mask."""
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config()
    wf = init_anat_template_wf(num_images=3, do_biascorr=True)
    assert _has(wf, 'merge_mask_wf'), 'no mask is generated for the merge'

    hits = []

    def walk(w):
        for node in w._graph.nodes():
            if hasattr(node, '_graph'):
                for _, v, data in node._graph.edges(data=True):
                    hits.extend(
                        (v.name, dst)
                        for _, dst in data.get('connect', [])
                        if 'fixed_image_masks' in str(dst)
                    )
                walk(node)

    walk(wf)
    assert hits, 'brain mask never reaches an antsRegistration fixed_image_masks input'


def test_single_image_needs_no_merge_mask():
    """With one image there is no merge, so no mask should be built for it."""
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config()
    wf = init_anat_template_wf(num_images=1, do_biascorr=True)
    assert not _has(wf, 'merge_mask_wf')


@pytest.mark.parametrize(
    ('mode', 'image_type', 'expected'),
    [
        ('n4', ['ORIGINAL', 'PRIMARY', 'M', 'NORM'], True),  # explicit wins over metadata
        ('none', ['ORIGINAL', 'PRIMARY', 'M'], False),
        ('auto', ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM'], False),  # normalized -> skip
        ('auto', ['ORIGINAL', 'PRIMARY', 'M'], True),  # not normalized -> run
        ('auto', [], True),  # missing metadata -> conservative, run
    ],
)
def test_auto_reads_image_type(monkeypatch, mode, image_type, expected):
    from qsiprep.workflows.anatomical.volume import anat_biascorrect_enabled

    config = _config(anat_biascorrect=mode)

    class _Layout:
        def get_metadata(self, _):
            return {'ImageType': image_type} if image_type else {}

    monkeypatch.setattr(config.execution, 'layout', _Layout(), raising=False)
    assert anat_biascorrect_enabled(['/data/sub-01_T1w.nii.gz']) is expected


def test_auto_runs_n4_when_only_some_images_are_normalized(monkeypatch):
    """A mixed set still needs N4 -- it cannot be merged consistently otherwise."""
    from qsiprep.workflows.anatomical.volume import anat_biascorrect_enabled

    config = _config(anat_biascorrect='auto')

    class _Layout:
        def get_metadata(self, path):
            norm = path.endswith('a.nii.gz')
            return {'ImageType': ['ORIGINAL', 'NORM'] if norm else ['ORIGINAL']}

    monkeypatch.setattr(config.execution, 'layout', _Layout(), raising=False)
    assert anat_biascorrect_enabled(['/a.nii.gz', '/b.nii.gz']) is True


def test_auto_falls_back_to_n4_without_a_layout(monkeypatch):
    from qsiprep.workflows.anatomical.volume import anat_biascorrect_enabled

    config = _config(anat_biascorrect='auto')
    monkeypatch.setattr(config.execution, 'layout', None, raising=False)
    assert anat_biascorrect_enabled(['/data/sub-01_T1w.nii.gz']) is True


def test_t2w_shares_the_anatomical_template_path():
    """T2w must inherit both fixes, since DRBUDDI/T2Wreg register to this image."""
    from qsiprep.workflows.anatomical.volume import init_t2w_preproc_wf

    _config()
    wf_off = init_t2w_preproc_wf(num_t2ws=3, do_biascorr=False, name='t2w_off')
    assert not _has(wf_off, 'n4_correct')
    assert _has(wf_off, 'merge_mask_wf')

    wf_on = init_t2w_preproc_wf(num_t2ws=3, do_biascorr=True, name='t2w_on')
    assert _has(wf_on, 'n4_correct')
