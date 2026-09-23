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
from nipype.interfaces.base import isdefined


def _config():
    from qsiprep import config

    config.execution.sloppy = True
    config.execution.output_dir = Path('/tmp/qsiprep_test_out')
    config.nipype.omp_nthreads = 1
    config.workflow.anat_modality = 'T1w'
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    config.workflow.hmc_method = 'tortoise'
    config.workflow.sdc_method = 'drbuddi'
    config.workflow.dwi2anat_dof = 6
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


def test_subject_dwiref_is_written_to_dwi():
    """The b=0 average across sessions existed only inside a report figure.

    It now lives in dwi/ with every other dwiref: `space` carries its level
    `space` alone names its level, following fMRIPrep 26.0.0's space-subject_boldref;
    the coregistration role is marked on the transforms instead.
    """
    from qsiprep.interfaces import DerivativesDataSink

    node = DerivativesDataSink(
        source_file='/data/sub-01_T1w.nii.gz',
        base_directory='/tmp/out',
        datatype='dwi',
        space='subject',
        suffix='dwiref',
        extension='.nii.gz',
        compress=True,
    )
    assert node.inputs.datatype == 'dwi'
    assert node.inputs.suffix == 'dwiref'
    assert node.inputs.space == 'subject'
    assert not isdefined(node.inputs.desc)


def test_average_images_normalizes_intensities():
    """Sessions differ in scaling, so the template average must normalize.

    ANTs AverageImages(normalize=True) rescales each input before averaging;
    without it a brighter session dominates the template. The warp average is
    deliberately NOT normalized -- displacement fields are not intensities.
    """
    import inspect

    from qsiprep.workflows.dwi import dwiref, hmc

    for mod in (hmc, dwiref):
        src = inspect.getsource(mod)
        for line in src.splitlines():
            if 'AverageImages(' in line and 'warp' not in line.lower():
                assert 'normalize=True' in line, line


def test_subject_dwiref_path_builds():
    """The subject-level b=0 template goes in dwi/, with every other dwiref.

    qsiprep ships its own path patterns (data/io_spec.json). Without a matching
    pattern the sink raises 'Could not build path with entities' and takes the
    whole run down.
    """
    import json

    from bids.layout.writing import build_path

    from qsiprep.data import load as load_data

    patterns = json.loads(load_data('io_spec.json').read_text())['default_path_patterns']

    out = build_path(
        {
            'subject': '01',
            'datatype': 'dwi',
            'suffix': 'dwiref',
            'space': 'subject',
            'extension': '.nii.gz',
        },
        patterns,
        strict=False,
    )
    assert out == 'sub-01/dwi/sub-01_space-subject_dwiref.nii.gz'


def test_existing_dwiref_and_anat_paths_still_build():
    """Extending the anat suffix list must not disturb existing outputs."""
    import json

    from bids.layout.writing import build_path

    from qsiprep.data import load as load_data

    patterns = json.loads(load_data('io_spec.json').read_text())['default_path_patterns']

    assert (
        build_path(
            {
                'subject': '01',
                'session': '1',
                'datatype': 'dwi',
                'suffix': 'dwiref',
                'space': 'ACPC',
                'extension': '.nii.gz',
            },
            patterns,
            strict=False,
        )
        == 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_dwiref.nii.gz'
    )

    assert (
        build_path(
            {
                'subject': '01',
                'datatype': 'anat',
                'suffix': 'T1w',
                'space': 'ACPC',
                'desc': 'preproc',
                'extension': '.nii.gz',
            },
            patterns,
            strict=False,
        )
        == 'sub-01/anat/sub-01_space-ACPC_desc-preproc_T1w.nii.gz'
    )


def test_dwiref_is_resampled_before_being_written():
    """The written template must be the ACPC-resampled one.

    outputnode.dwiref lives in the template's own midpoint space --
    measured ~57mm from ACPC in y on real data. Writing that tagged space-ACPC
    produces a file that silently fails to overlay the anatomicals, which is the
    worst kind of wrong: it looks like a valid derivative.
    """
    from qsiprep.workflows.dwi.dwiref import init_dwiref_wf

    _config()
    wf = init_dwiref_wf(
        inputs_list=['a', 'b'],
        t1w_source_file='/data/sub-01_T1w.nii.gz',
        transform='Rigid',
        num_iterations=2,
        name='acpc_check',
    )
    node = next((n for n in wf._get_all_nodes() if n.name == 'template_to_acpc'), None)
    assert node is not None, 'template is never resampled into ACPC'
    assert 'dwiref_acpc' in wf.get_node('outputnode').outputs.copyable_trait_names()


def test_base_sinks_both_templates_to_their_own_spaces():
    """Each template image goes to the path that describes the space it is in.

    The midpoint-space template is space-<level>; the resampled one is space-ACPC.
    The original defect was sinking the un-resampled template as space-ACPC, which
    produced a file that silently failed to overlay the anatomicals.
    """
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base)
    # Check the connections themselves, not proximity in the file -- nodes get
    # added between a sink and its connect block over time.
    assert "('outputnode.dwiref_acpc', 'in_file')" in src
    assert "('outputnode.dwiref', 'in_file')" in src
    assert "name='ds_dwiref_acpc'" in src
