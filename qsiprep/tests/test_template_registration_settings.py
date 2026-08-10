"""Unbiased template creation gets its own antsRegistration parameters.

The SHORELine settings are tuned for within-scan b=0 motion correction: noisy,
2 mm, contrast that varies between volumes. Template creation -- the anatomical
merge and the intramodal b=0 template -- runs on high-SNR images that share
contrast, and inherited those settings by accident.

The measured difference is the convergence threshold. At 1e-08 with a window of
20 the test never fires, so every registration burns its full iteration budget
after converging. Measured on 0.94 mm T1w against the merge template, three
images: 16x faster at 1e-06, reaching an identical Mattes value to five decimal
places (-0.98862, -1.02834, -0.96790 for both arms) and landing within 0.028
degrees / 0.008 mm.
"""

import json

import pytest

from qsiprep.data import load as load_data

FAMILIES = ['shoreline', 'unbiased_template']
PRECISIONS = ['precise', 'sloppy']
TRANSFORMS = ['Rigid', 'Affine']


def _cfg(family, precision, transform):
    return json.loads(load_data(f'{family}_{precision}_{transform}.json').read_text())


@pytest.mark.parametrize('precision', PRECISIONS)
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_every_variant_exists(precision, transform):
    """The filename is built from precision and transform, so all four must exist."""
    cfg = _cfg('unbiased_template', precision, transform)
    assert cfg['transforms'][0] == 'Rigid'
    if transform == 'Affine':
        assert cfg['transforms'] == ['Rigid', 'Affine']


@pytest.mark.parametrize('precision', PRECISIONS)
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_convergence_threshold_can_actually_fire(precision, transform):
    cfg = _cfg('unbiased_template', precision, transform)
    assert all(t == 1e-06 for t in cfg['convergence_threshold']), cfg['convergence_threshold']


@pytest.mark.parametrize('precision', PRECISIONS)
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_per_stage_lists_match_the_stage_count(precision, transform):
    """A list longer than `transforms` describes stages that never run.

    nipype silently truncates to the number of transforms, so the extra entries
    were dead -- the shipped shoreline_*_Rigid.json still carries some.
    """
    cfg = _cfg('unbiased_template', precision, transform)
    n = len(cfg['transforms'])
    per_stage = [
        'transform_parameters',
        'convergence_threshold',
        'convergence_window_size',
        'number_of_iterations',
        'metric',
        'metric_weight',
        'sampling_percentage',
        'sampling_strategy',
        'radius_or_number_of_bins',
        'smoothing_sigmas',
        'shrink_factors',
        'sigma_units',
        'use_histogram_matching',
    ]
    for key in per_stage:
        if key in cfg:
            assert len(cfg[key]) == n, f'{key} has {len(cfg[key])} entries for {n} stage(s)'


@pytest.mark.parametrize('precision', PRECISIONS)
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_only_the_threshold_differs_from_shoreline(precision, transform):
    """Guard the scope of the change.

    Only the convergence threshold was validated against the baseline. If a
    future edit changes sampling, bins or the resolution schedule, that needs its
    own measurement -- so make it fail here rather than pass silently.
    """
    ours = _cfg('unbiased_template', precision, transform)
    theirs = _cfg('shoreline', precision, transform)
    n = len(ours['transforms'])

    for key in set(ours) | set(theirs):
        if key == 'convergence_threshold':
            continue
        a, b = ours.get(key), theirs.get(key)
        if isinstance(b, list) and len(b) > n:
            b = b[:n]  # the trimmed dead entries
        assert a == b, f'{key} differs: {a!r} vs {b!r}'


def test_shoreline_is_untouched():
    """Within-scan b=0 HMC must keep its own settings."""
    cfg = _cfg('shoreline', 'precise', 'Rigid')
    assert cfg['convergence_threshold'][0] == 1e-08


def _config(**overrides):
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    config.workflow.b0_threshold = 100
    config.workflow.hmc_transform = 'Rigid'
    config.workflow.hmc_model = 'diffprep_quadratic'
    config.workflow.b0_motion_corr_to = 'iterative'
    for key, value in overrides.items():
        setattr(config.workflow, key, value)
    return config


def _thresholds(wf):
    """Convergence thresholds on every antsRegistration node in a built workflow.

    Asserting on the loaded value rather than the filename: `from_file` is
    consumed at construction and not retained on the inputs, and the threshold is
    the thing that actually changes behaviour.
    """
    out = {}
    for node in wf._get_all_nodes():
        thr = getattr(node.interface.inputs, 'convergence_threshold', None)
        if thr is not None and not isinstance(thr, type(NotImplemented)):
            try:
                out[node.name] = list(thr)
            except TypeError:
                pass
    return out


def test_anat_merge_uses_the_template_settings(tmp_path):
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_anat_template_wf(num_images=3, do_biascorr=True)
    thr = {k: v for k, v in _thresholds(wf).items() if k.startswith('reg_')}
    assert thr, 'no registration nodes found'
    assert all(v == [1e-06] for v in thr.values()), thr


def test_intramodal_b0_template_uses_the_template_settings(tmp_path):
    from qsiprep.workflows.dwi.intramodal_template import init_intramodal_template_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_intramodal_template_wf(
        inputs_list=['scan1', 'scan2'],
        t1w_source_file='/data/sub-01/anat/sub-01_T1w.nii.gz',
        transform='Rigid',
        num_iterations=2,
    )
    thr = {k: v for k, v in _thresholds(wf).items() if k.startswith('reg_')}
    assert thr, 'no registration nodes found'
    assert all(v == [1e-06] for v in thr.values()), thr


def test_nonlinear_intramodal_template_does_not_use_these_settings(tmp_path):
    """BSplineSyN is built by antsMultivariateTemplateConstruction2.

    That path never goes through init_b0_hmc_wf, so it carries its own
    registration parameters inside the ANTs script and this change does not reach
    it. Documented here so the boundary is not mistaken for a gap.
    """
    from qsiprep.workflows.dwi.intramodal_template import init_intramodal_template_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_intramodal_template_wf(
        inputs_list=['scan1', 'scan2'],
        t1w_source_file='/data/sub-01/anat/sub-01_T1w.nii.gz',
        transform='BSplineSyN',
        num_iterations=2,
    )
    names = {n.name for n in wf._get_all_nodes()}
    assert 'ants_mvtc2' in names
    assert not any(n.startswith('reg_') for n in names), names


def test_within_scan_hmc_still_uses_shoreline(tmp_path):
    """The default family must not have moved underneath SHORELine."""
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid')
    thr = {k: v for k, v in _thresholds(wf).items() if k.startswith('reg_')}
    assert thr, 'no registration nodes found'
    # the shipped shoreline file still carries a trailing entry for a stage it
    # does not have; nipype loads both but only passes the first
    assert all(all(x == 1e-08 for x in v) for v in thr.values()), thr


def test_init_b0_hmc_wf_has_no_spatial_bias_correct():
    """init_qsiprep_intramodal_template_wf passes spatial_bias_correct= to
    init_b0_hmc_wf, which does not accept it; it would raise TypeError if it
    ever ran and is left unwired deliberately."""
    import inspect

    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    assert 'spatial_bias_correct' not in inspect.signature(init_b0_hmc_wf).parameters
