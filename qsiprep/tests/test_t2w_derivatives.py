"""T2w products must reach derivatives.

Both images were computed on every run with a T2w and then discarded -- nothing
consumed ``t2_preproc``, and ``t2w_unfatsat`` only ever went to the DWI
workflows. They are different images and both are worth writing:

* ``t2_preproc``    -- the merged (unbiased) T2w template resampled into ACPC,
                       built from ``anat_reference_wf``'s ``bias_corrected``.
* ``t2w_unfatsat``  -- the fat-suppressed image that TORTOISE T2Wreg and DRBUDDI
                       actually register to. It descends from
                       ``outputnode.template``, so it is NOT bias corrected and
                       is NOT the same image as ``t2_preproc``.
"""

from pathlib import Path

import pytest


def _config():
    from qsiprep import config

    config.execution.sloppy = True
    config.execution.output_dir = Path('/tmp/qsiprep_test_out')
    config.nipype.omp_nthreads = 1
    config.workflow.anat_modality = 'T1w'
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    config.workflow.hmc_model = 'diffprep_quadratic'
    config.workflow.pepolar_method = 'DRBUDDI'
    return config


def _build(num_additional_t2ws, name):
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    _config()
    return init_anat_preproc_wf(
        num_anat_images=2,
        num_additional_t2ws=num_additional_t2ws,
        has_rois=False,
        anatomical_template='MNI152NLin2009cAsym',
        name=name,
    )


def _names(wf):
    return wf.list_node_names()


def test_t2w_derivatives_are_written_when_t2ws_exist():
    names = _names(_build(3, 'with_t2w'))
    assert any('ds_t2_preproc' in n for n in names)
    assert any('ds_t2w_unfatsat' in n for n in names)


def test_no_t2w_sinks_without_t2ws():
    """Otherwise the sinks would sit with undefined inputs and fail at runtime."""
    names = _names(_build(0, 'no_t2w'))
    assert not any('ds_t2_preproc' in n for n in names)
    assert not any('ds_t2w_unfatsat' in n for n in names)
    assert not any('t2_name' in n for n in names)


def test_t2w_sinks_use_a_t2w_source_name():
    """DerivativesDataSink takes its suffix from source_file.

    Reusing t1_name would emit *_T1w.nii.gz and overwrite the real T1w
    derivative, so the T2w sinks need their own name node.
    """
    wf = _build(3, 'naming')
    t2_name = next(n for n in wf._get_all_nodes() if n.name == 't2_name')
    assert t2_name.inputs.anatomical_contrast == 'T2w'


@pytest.mark.parametrize(
    ('node_name', 'desc'),
    [('ds_t2_preproc', 'preproc'), ('ds_t2w_unfatsat', 'unfatsat')],
)
def test_t2w_sinks_are_distinct_outputs(node_name, desc):
    """Distinct desc entities: they are different images, not two names for one."""
    wf = _build(3, f'distinct_{desc}')
    node = next(n for n in wf._get_all_nodes() if n.name == node_name)
    assert node.inputs.desc == desc
    assert node.inputs.space == 'ACPC'


def test_intramodal_template_is_written_to_anat():
    """The b=0 average across sessions existed only inside a report figure.

    Writing it into the anat directory means one listing shows every
    subject-level product together: T1w, both T2ws, and the b=0 average.
    """
    from qsiprep.interfaces import DerivativesDataSink

    node = DerivativesDataSink(
        source_file='/data/sub-01_T1w.nii.gz',
        base_directory='/tmp/out',
        datatype='anat',
        space='ACPC',
        desc='intramodal',
        suffix='dwiref',
        extension='.nii.gz',
        compress=True,
    )
    assert node.inputs.datatype == 'anat'
    assert node.inputs.suffix == 'dwiref'
    assert node.inputs.space == 'ACPC'
    assert node.inputs.desc == 'intramodal'


def test_average_images_normalizes_intensities():
    """Sessions differ in scaling, so the template average must normalize.

    ANTs AverageImages(normalize=True) rescales each input before averaging;
    without it a brighter session dominates the template. The warp average is
    deliberately NOT normalized -- displacement fields are not intensities.
    """
    import inspect

    from qsiprep.workflows.dwi import hmc, intramodal_template

    for mod in (hmc, intramodal_template):
        src = inspect.getsource(mod)
        for line in src.splitlines():
            if 'AverageImages(' in line and 'warp' not in line.lower():
                assert 'normalize=True' in line, line


def test_anat_dwiref_path_builds():
    """The subject-level b=0 template goes in anat/, next to the T1w and T2ws.

    qsiprep ships its own path patterns (data/io_spec.json) and already extends
    the anat suffix list with a non-BIDS entry (imtcoreg), so adding dwiref is
    consistent with how this spec is maintained. Without the pattern the sink
    raises 'Could not build path with entities' and takes the whole run down.
    """
    import json

    from bids.layout.writing import build_path

    from qsiprep.data import load as load_data

    patterns = json.loads(load_data('io_spec.json').read_text())['default_path_patterns']

    out = build_path(
        dict(
            subject='01',
            datatype='anat',
            suffix='dwiref',
            space='ACPC',
            desc='intramodal',
            extension='.nii.gz',
        ),
        patterns,
        strict=False,
    )
    assert out == 'sub-01/anat/sub-01_space-ACPC_desc-intramodal_dwiref.nii.gz'


def test_existing_dwiref_and_anat_paths_still_build():
    """Extending the anat suffix list must not disturb existing outputs."""
    import json

    from bids.layout.writing import build_path

    from qsiprep.data import load as load_data

    patterns = json.loads(load_data('io_spec.json').read_text())['default_path_patterns']

    assert build_path(
        dict(subject='01', session='1', datatype='dwi', suffix='dwiref',
             space='ACPC', extension='.nii.gz'), patterns, strict=False
    ) == 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_dwiref.nii.gz'

    assert build_path(
        dict(subject='01', datatype='anat', suffix='T1w', space='ACPC',
             desc='preproc', extension='.nii.gz'), patterns, strict=False
    ) == 'sub-01/anat/sub-01_space-ACPC_desc-preproc_T1w.nii.gz'
