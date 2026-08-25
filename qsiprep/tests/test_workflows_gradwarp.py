"""Construction tests for the gradwarp workflow and its wiring."""

import inspect

import pytest

from qsiprep import config
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients, write_siemens_grad
from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.fixture(autouse=True)
def _reset_config():
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    config.nipype.omp_nthreads = 1
    yield
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []


def _unit(tmp_path, image_type=None):
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit([dwi], metadata=metadata)


def test_gradwarp_wf_is_none_without_a_coefficient_file(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    assert init_gradwarp_wf(_unit(tmp_path)) is None


def test_gradwarp_wf_builds_field_and_mask_nodes(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path))

    assert wf.get_node('make_field') is not None
    assert wf.get_node('mask_field') is not None
    assert wf.get_node('outputnode') is not None


def test_gradwarp_wf_masks_to_through_plane_for_dis2d(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, ['ORIGINAL', 'DIS2D']))

    assert wf.get_node('mask_field').inputs.warp_dim == '1D'
    assert wf.plan.warp_dim == '1D'


def test_gradwarp_wf_still_builds_a_field_for_dis3d(tmp_path):
    """grad_dev needs a field even when no spatial correction is applied."""
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, ['ORIGINAL', 'DIS3D']))

    assert wf.get_node('make_field') is not None
    assert wf.plan.warp_dim is None


def test_gradwarp_wf_passes_is_ge_through(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi], metadata={'Manufacturer': 'GE MEDICAL SYSTEMS'})

    assert init_gradwarp_wf(unit).get_node('make_field').inputs.is_ge is True


# --- Task 9: threading the field through resampling and base -----------------


def _trans_wf_gradwarp_sources(wf):
    """Names of nodes feeding compose_transforms.gradwarp, if any."""
    compose = wf.get_node('compose_transforms')
    return [
        edge[0].name
        for edge in wf._graph.in_edges(compose)
        if any(dest == 'gradwarp' for _, dest in wf._graph.get_edge_data(*edge)['connect'])
    ]


def test_dwi_trans_wf_exposes_a_gradwarp_field_input():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    config.workflow.output_resolution = 1.2
    wf = init_dwi_trans_wf(
        source_file='sub-1_dwi.nii.gz', mem_gb=1, name='trans_wf', use_compression=False
    )
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_dwi_trans_wf_connects_gradwarp_to_compose_transforms():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    config.workflow.output_resolution = 1.2
    wf = init_dwi_trans_wf(
        source_file='sub-1_dwi.nii.gz', mem_gb=1, name='trans_wf', use_compression=False
    )
    assert _trans_wf_gradwarp_sources(wf) == ['inputnode']


def test_listify_wraps_a_single_value():
    from qsiprep.workflows.dwi.resampling import _listify

    assert _listify('a.nii.gz') == ['a.nii.gz']


def test_listify_passes_undefined_through():
    from nipype.interfaces.base import Undefined

    from qsiprep.workflows.dwi.resampling import _listify

    assert _listify(Undefined) is Undefined


def test_listify_rejects_a_list_input():
    """A mis-wire that feeds ``_listify`` an already-listed value must fail loudly.

    ``ComposeTransforms.gradwarp`` silently drops a list whose length matches
    neither 1 nor the DWI count (unlike ``fieldwarps``, which warns), so a
    mis-wire here must not be allowed to vanish silently downstream.
    """
    from qsiprep.workflows.dwi.resampling import _listify

    with pytest.raises(AssertionError):
        _listify(['a.nii.gz', 'b.nii.gz'])


def _finalize_cfg(tmp_path):
    config.execution.output_dir = str(tmp_path)
    config.execution.sloppy = False
    config.workflow.pepolar_method = 'TOPUP'
    config.workflow.output_resolution = 1.2
    config.workflow.intramodal_template_iters = 0
    config.nipype.omp_nthreads = 1


def _finalize_wf(tmp_path):
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    _finalize_cfg(tmp_path)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi])
    return init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=dwi,
        output_prefix='sub-01',
        write_derivatives=False,
    )


def test_dwi_finalize_wf_exposes_a_gradwarp_field_input(tmp_path):
    wf = _finalize_wf(tmp_path)
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_dwi_finalize_wf_connects_gradwarp_field_to_trans_wf(tmp_path):
    wf = _finalize_wf(tmp_path)
    trans_wf = wf.get_node('transform_dwis_t1')
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), trans_wf)
    assert edge is not None
    assert ('gradwarp_field', 'inputnode.gradwarp_field') in edge['connect']


class _StubFile:
    def get_entities(self):
        return {}


class _StubLayout:
    """Minimal layout stand-in -- see test_workflows_native.py's version."""

    def get_metadata(self, path):
        return {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}

    def get_entities(self, metadata=False):
        return {}

    def get_file(self, path):
        return _StubFile()

    def get(self, **query):
        return []


def _dwi_preproc_cfg(tmp_path):
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.execution.layout = _StubLayout()
    config.execution.output_dir = str(tmp_path)
    config.workflow.hmc_model = 'eddy'
    config.workflow.pepolar_method = 'TOPUP'
    config.workflow.b0_threshold = 100
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwi_denoise_window = 5
    config.workflow.shoreline_iters = 2
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.anat_modality = 't1w'
    config.workflow.b0_to_anat_transform = 'Rigid'
    config.workflow.hmc_transform = 'Affine'


def _preproc_wf(tmp_path, image_type=None):
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    _dwi_preproc_cfg(tmp_path)
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    unit = make_preproc_unit([dwi], metadata=metadata)

    return init_dwi_preproc_wf(
        unit,
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=dwi,
        anatomical_template='MNI152NLin2009cAsym',
    )


def test_dwi_preproc_wf_builds_gradwarp_and_feeds_pre_hmc_reference(tmp_path):
    """A resolved plan builds gradwarp_wf and feeds it the pre-HMC reference."""
    wf = _preproc_wf(tmp_path)

    gradwarp_wf = wf.get_node('gradwarp_wf')
    assert gradwarp_wf is not None

    pre_hmc_wf = wf.get_node('pre_hmc_wf')
    edge = wf._graph.get_edge_data(pre_hmc_wf, gradwarp_wf)
    assert edge is not None
    assert ('outputnode.dwi_file', 'inputnode.ref_image') in edge['connect']

    # No ImageType tags -> plan defaults to '3D' -> connected into outputnode.
    outputnode = wf.get_node('outputnode')
    edge = wf._graph.get_edge_data(gradwarp_wf, outputnode)
    assert edge is not None
    assert ('outputnode.gradwarp_field', 'gradwarp_field') in edge['connect']


def test_dwi_preproc_wf_dis3d_builds_field_but_leaves_outputnode_unconnected(tmp_path):
    """DIS3D still builds a field (grad_dev needs it) but skips resampling wiring."""
    wf = _preproc_wf(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    gradwarp_wf = wf.get_node('gradwarp_wf')
    assert gradwarp_wf is not None
    assert gradwarp_wf.plan.warp_dim is None

    outputnode = wf.get_node('outputnode')
    edge = wf._graph.get_edge_data(gradwarp_wf, outputnode)
    assert edge is None


def test_dwi_preproc_wf_without_gradient_file_has_no_gradwarp_wf(tmp_path):
    """The default path (no --gradient-coils) is untouched."""
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    _dwi_preproc_cfg(tmp_path)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi], metadata={'Manufacturer': 'SIEMENS'})

    wf = init_dwi_preproc_wf(
        unit,
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=dwi,
        anatomical_template='MNI152NLin2009cAsym',
    )

    assert wf.get_node('gradwarp_wf') is None
    assert 'gradwarp_field' in wf.get_node('outputnode').inputs.trait_get()


def test_single_subject_wf_wires_gradwarp_field_to_finalize():
    """``dwi_preproc_wf`` and ``dwi_finalize_wf`` are siblings built side-by-side
    in ``init_single_subject_wf``; ``gradwarp_field`` must cross between them the
    same way ``fieldwarps`` does, in the connect block joining the two per-unit
    workflows (too heavy to build end-to-end in a unit test -- BIDS layout,
    anatomical workflow, etc. -- so this checks the wiring is textually present,
    matching the precedent in test_intramodal_transforms.py).
    """
    from qsiprep.workflows import base

    src = inspect.getsource(base.init_single_subject_wf)
    assert "('outputnode.gradwarp_field', 'inputnode.gradwarp_field')" in src
