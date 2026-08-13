"""The intramodal registration reportlet must actually show the registration.

Previously it flickered the session's b=0 against the group template: two
different images, in two different spaces, with the "after" frame byte-identical
for every session. It could not distinguish a good registration from a failed
one -- the only thing it exists to do -- and carried no landmarks.
"""


def _config():
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.intramodal_template_iters = 2
    config.workflow.b0_to_anat_transform = 'Rigid'
    return config


def _template_wf(transform='Rigid', name='imt_report'):
    from qsiprep.workflows.dwi.intramodal_template import init_intramodal_template_wf

    _config()
    return init_intramodal_template_wf(
        inputs_list=['a', 'b'],
        t1w_source_file='/data/sub-01_T1w.nii.gz',
        transform=transform,
        num_iterations=2,
        name=name,
    )


def test_white_matter_is_carried_into_template_space():
    """Landmarks come from the anatomy, via the template->anat affine inverted.

    The ordering works because b0_coreg_wf registers the template to the anatomy,
    so the affine exists once the template is built and before anything
    downstream needs it.
    """
    wf = _template_wf()
    names = [n.name for n in wf._get_all_nodes()]
    assert 'seg_to_template' in names
    assert 'template_wm' in names

    seg = next(n for n in wf._get_all_nodes() if n.name == 'seg_to_template')
    # anat -> template is the INVERSE of the affine b0_coreg_wf produces
    assert seg.inputs.invert_transform_flags == [True]
    # a segmentation must not be interpolated continuously
    assert seg.inputs.interpolation == 'MultiLabel'


def test_wm_seg_is_exposed_for_downstream_reports():
    wf = _template_wf(name='imt_expose')
    outs = wf.get_node('outputnode').outputs.copyable_trait_names()
    assert 'intramodal_template_wm_seg' in outs


def test_report_compares_one_image_before_and_after_its_own_transform():
    """Both frames must be the session b=0 on the template grid.

    Guards the specific defect: 'after' being the template itself, which made the
    frame identical across sessions.
    """
    import inspect

    from qsiprep.workflows.dwi import finalize

    src = inspect.getsource(finalize)
    assert "('intramodal_template', 'after')" not in src, 'after frame is the template again'
    assert "(b0_to_template_grid, b0_to_im_template, [('output_image', 'before')])" in src
    assert "(b0_aligned_to_template, b0_to_im_template, [('output_image', 'after')])" in src
    assert "('intramodal_template_wm_seg', 'wm_seg')" in src


def test_both_report_frames_land_on_the_template_grid():
    """If the frames differ in grid, the flicker shows resampling, not registration."""
    import inspect

    from qsiprep.workflows.dwi import finalize

    src = inspect.getsource(finalize)
    start = src.index('b0_to_template_grid = pe.Node')
    window = src[start : start + 2500]
    assert window.count("('intramodal_template', 'reference_image')") == 2
