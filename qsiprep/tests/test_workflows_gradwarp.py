"""Construction tests for the gradwarp workflow and its wiring."""

import inspect

import pytest

from qsiprep import config
from qsiprep.tests.gradient_fixtures import (
    write_dwi_with_gradients,
    write_itk_field,
    write_siemens_grad,
)
from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.fixture(autouse=True)
def _reset_config():
    from qsiprep.workflows.dwi.gradwarp import _reset_plan_logging

    # resolve_gradwarp_plan suppresses exact repeats of a rendered log line, so
    # that memory has to be cleared between tests or one test can silence
    # another's expected message.
    _reset_plan_logging()
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    # The boilerplate branches on the HMC backend, so leaking this between
    # tests would silently change the text another test asserts on.
    config.workflow.hmc_model = '3dSHORE'
    config.nipype.omp_nthreads = 1
    yield
    _reset_plan_logging()
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


def test_gradwarp_wf_builds_no_field_for_dis3d(tmp_path):
    """A DIS3D unit builds no field at all.

    Nothing consumes one: the scanner already corrected the geometry, so no
    resampling uses the field, and finalize's grad_dev node is fed the
    *coefficient* file rather than a field. Building one would invoke an
    external binary per unit and throw both of its outputs away.
    """
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, ['ORIGINAL', 'DIS3D']))

    assert wf is not None
    assert wf.plan.warp_dim is None
    assert wf.get_node('make_field') is None
    assert wf.get_node('mask_field') is None
    assert list(wf._graph.nodes()) == []
    # The plan and the methods text still have to survive.
    assert wf.needs_reference is False


def test_gradwarp_wf_skips_make_field_for_a_displacement_field_input(tmp_path):
    """A ``.nii`` --gradient-file is already a field.

    ``CreateNonlinearityDisplacementMap`` is the *coefficient expander* and
    does no extension dispatch of its own (TORTOISE branches on the extension
    before ever calling it), so handing it a binary NIfTI would feed a text
    parser -- either throwing, or silently yielding an all-zero field.
    """
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    field = tmp_path / 'gradwarp_field.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4, 1, 3), dtype='float32'), np.eye(4)).to_filename(str(field))
    config.workflow.gradient_file = str(field)
    wf = init_gradwarp_wf(_unit(tmp_path))

    assert wf.get_node('make_field') is None
    assert wf.needs_reference is False
    # The supplied field goes straight into the dimension mask.
    assert wf.get_node('mask_field').inputs.in_file == str(field)


def test_gradwarp_wf_builds_make_field_for_a_coefficient_input(tmp_path):
    """The other half of the dispatch: coefficients still need expanding."""
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path))

    assert wf.get_node('make_field') is not None
    assert wf.needs_reference is True


@pytest.mark.parametrize(
    ('gradient_file', 'expected'),
    [
        ('/opt/coeff.grad', False),
        ('/opt/coeff.dat', False),
        ('/opt/coeff.gc', False),
        ('/opt/field.nii', True),
        ('/opt/field.nii.gz', True),
    ],
)
def test_is_displacement_field_covers_every_accepted_extension(gradient_file, expected):
    """Every extension --gradient-file accepts must land on one branch."""
    from qsiprep.workflows.dwi.gradwarp import is_displacement_field

    assert is_displacement_field(gradient_file) is expected


# --- Backend-specific resampling boilerplate ---------------------------------
#
# eddy and DIFFPREP write out motion/eddy-corrected volumes before qsiprep's
# final resampling, so the composed chain carries no HMC transform at all.
# Claiming a single raw-to-final interpolation there would be a methods-section
# error.


@pytest.mark.parametrize('warp_dim', ['3D', '1D'])
@pytest.mark.parametrize(
    ('hmc_model', 'backend'), [('eddy', 'FSL eddy'), ('tortoise', 'DIFFPREP')]
)
def test_boilerplate_does_not_claim_single_resampling_for_preresampling_backends(
    hmc_model, backend, warp_dim
):
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.hmc_model = hmc_model
    text = gradwarp_boilerplate(warp_dim)

    assert 'resampled only once' not in text
    assert backend in text
    assert 'single resampling' in text


@pytest.mark.parametrize('warp_dim', ['3D', '1D'])
@pytest.mark.parametrize('hmc_model', ['3dSHORE', 'tensor', 'none'])
def test_boilerplate_claims_single_resampling_for_transform_preserving_backends(
    hmc_model, warp_dim
):
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.hmc_model = hmc_model
    text = gradwarp_boilerplate(warp_dim)

    assert 'resampled only once' in text
    assert 'FSL eddy' not in text
    assert 'DIFFPREP' not in text


@pytest.mark.parametrize('hmc_model', ['eddy', 'tortoise', '3dSHORE'])
def test_dis3d_boilerplate_makes_no_resampling_claim_on_any_backend(hmc_model):
    """A DIS3D unit gets no field, so there is nothing to have been combined
    with anything -- on any backend."""
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.hmc_model = hmc_model
    text = gradwarp_boilerplate(None)

    assert 'resampl' not in text
    assert 'displacement field' not in text


# --- The GE coefficient-expansion guard --------------------------------------
#
# TORTOISE shifts the field's z origin after expanding GE coefficients, in
# TORTOISEProcess rather than in the standalone binary qsiprep calls, so
# qsiprep cannot reproduce the placement. resolve_gradwarp_plan refuses the
# combination. Each test below removes exactly one leg of that three-part
# condition, so dropping any leg from _guard_ge_field fails a test.


def _ge_unit(tmp_path, image_type=None):
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'GE MEDICAL SYSTEMS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit([dwi], metadata=metadata)


def test_ge_coefficients_are_refused(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import resolve_gradwarp_plan

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))

    with pytest.raises(ValueError, match='not supported for GE data'):
        resolve_gradwarp_plan(_ge_unit(tmp_path))


def test_ge_coefficients_are_refused_when_forced(tmp_path):
    """--force gradients must not become a way around the guard: it forces 3D,
    which is exactly the field that cannot be placed."""
    from qsiprep.workflows.dwi.gradwarp import resolve_gradwarp_plan

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    config.workflow.force = ['gradients']

    with pytest.raises(ValueError, match='not supported for GE data'):
        resolve_gradwarp_plan(_ge_unit(tmp_path, ['ORIGINAL', 'DIS3D']))


def test_ge_dis3d_still_resolves_for_grad_dev(tmp_path):
    """No field is built for a DIS3D unit, and CreateGradientNonlinearityBMatrix
    does its own GE recentring, so grad_dev is unaffected and must survive."""
    from qsiprep.workflows.dwi.gradwarp import resolve_gradwarp_plan

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))

    plan = resolve_gradwarp_plan(_ge_unit(tmp_path, ['ORIGINAL', 'DIS3D']))

    assert plan is not None
    assert plan.warp_dim is None
    assert plan.is_ge is True


def test_ge_displacement_field_is_allowed(tmp_path):
    """A ready-made field is used as given -- nothing is expanded, so the
    origin-shift defect cannot apply."""
    from qsiprep.workflows.dwi.gradwarp import resolve_gradwarp_plan

    config.workflow.gradient_file = str(write_itk_field(tmp_path / 'field.nii.gz'))

    plan = resolve_gradwarp_plan(_ge_unit(tmp_path))

    assert plan.warp_dim == '3D'
    assert plan.is_ge is True


def test_non_ge_coefficients_are_untouched_by_the_guard(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))

    wf = init_gradwarp_wf(_unit(tmp_path))

    assert wf.plan.is_ge is False
    assert wf.get_node('make_field').inputs.is_ge is False


@pytest.mark.parametrize(
    ('image_type', 'warp_dim'),
    [
        (None, '3D'),
        (['ORIGINAL', 'DIS2D'], '1D'),
        (['ORIGINAL', 'DIS3D'], None),
    ],
)
def test_gradwarp_wf_desc_matches_the_resolved_warp_dim(tmp_path, image_type, warp_dim):
    """workflow.__desc__ must be the boilerplate for the resolved plan,
    not just any entry -- report text that doesn't track the plan would be a
    methods-section error."""
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate, init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, image_type))

    assert wf.plan.warp_dim == warp_dim
    assert wf.__desc__ == gradwarp_boilerplate(warp_dim)


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


def _finalize_wf(tmp_path, write_derivatives=False):
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    _finalize_cfg(tmp_path)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi])
    return init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=dwi,
        output_prefix='sub-01',
        write_derivatives=write_derivatives,
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
    """A resolved plan builds gradwarp_wf and feeds it a 3D reference.

    CreateNonlinearityDisplacementMap's underlying tool reads its reference
    image as a 3D NIfTI, not the 4D series pre_hmc_wf.outputnode.dwi_file is,
    so pre_hmc_wf must NOT feed gradwarp_wf.inputnode.ref_image directly --
    an extraction node has to sit between them.
    """
    wf = _preproc_wf(tmp_path)

    gradwarp_wf = wf.get_node('gradwarp_wf')
    assert gradwarp_wf is not None

    pre_hmc_wf = wf.get_node('pre_hmc_wf')

    # No direct edge from pre_hmc_wf to gradwarp_wf -- that would be the 4D
    # merged series reaching a tool that requires a 3D reference.
    assert wf._graph.get_edge_data(pre_hmc_wf, gradwarp_wf) is None

    gradwarp_ref = wf.get_node('gradwarp_ref')
    assert gradwarp_ref is not None

    edge = wf._graph.get_edge_data(pre_hmc_wf, gradwarp_ref)
    assert edge is not None
    assert ('outputnode.dwi_file', 'in_file') in edge['connect']

    edge = wf._graph.get_edge_data(gradwarp_ref, gradwarp_wf)
    assert edge is not None
    assert ('out_file', 'inputnode.ref_image') in edge['connect']

    # No ImageType tags -> plan defaults to '3D' -> connected into outputnode.
    outputnode = wf.get_node('outputnode')
    edge = wf._graph.get_edge_data(gradwarp_wf, outputnode)
    assert edge is not None
    assert ('outputnode.gradwarp_field', 'gradwarp_field') in edge['connect']


def test_dwi_preproc_wf_dis3d_runs_nothing_and_wires_nothing(tmp_path):
    """A DIS3D unit neither builds a field nor extracts a reference for one.

    Nothing downstream consumes either: no resampling uses the field, and
    finalize's grad_dev node takes the coefficient file. The previous wiring
    ran an external binary plus a nibabel node per unit and discarded both.
    """
    wf = _preproc_wf(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    gradwarp_wf = wf.get_node('gradwarp_wf')
    assert gradwarp_wf is not None
    assert gradwarp_wf.plan.warp_dim is None

    # No extraction node, so nothing feeds the (nonexistent) field builder.
    assert wf.get_node('gradwarp_ref') is None

    outputnode = wf.get_node('outputnode')
    assert wf._graph.get_edge_data(gradwarp_wf, outputnode) is None
    # ...and it contributes no runnable nodes to the flattened graph.
    flat = [node.name for node in wf._create_flat_graph().nodes()]
    assert not [name for name in flat if name.startswith('gradwarp')]


def test_dwi_preproc_wf_dis3d_still_emits_the_dis3d_boilerplate(tmp_path):
    """The DIS3D methods text is not optional -- it is why the state exists.

    ``LiterateWorkflow.visit_desc`` walks the parent graph, so a gradwarp_wf
    that contributes no nodes still has to be *in* that graph.
    """
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    wf = _preproc_wf(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    assert gradwarp_boilerplate(None) in wf.visit_desc()


def test_dwi_preproc_wf_dis3d_report_line_survives(tmp_path):
    """The report line is derived from ``gradwarp_wf.plan``, which must survive
    the workflow having no nodes."""
    wf = _preproc_wf(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    assert wf.get_node('summary').inputs.gradient_correction == 'b-matrix only (ImageType: DIS3D)'


def test_dwi_preproc_wf_skips_the_reference_node_for_a_displacement_field(tmp_path):
    """A supplied ITK field is already on its own grid; nothing to extract."""
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    _dwi_preproc_cfg(tmp_path)
    field = tmp_path / 'gradwarp_field.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4, 1, 3), dtype='float32'), np.eye(4)).to_filename(str(field))
    config.workflow.gradient_file = str(field)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi], metadata={'Manufacturer': 'SIEMENS'})
    wf = init_dwi_preproc_wf(
        unit,
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=dwi,
        anatomical_template='MNI152NLin2009cAsym',
    )

    assert wf.get_node('gradwarp_ref') is None
    # The field is still wired into resampling: only the field *builder* went.
    gradwarp_wf = wf.get_node('gradwarp_wf')
    edge = wf._graph.get_edge_data(gradwarp_wf, wf.get_node('outputnode'))
    assert edge is not None
    assert ('outputnode.gradwarp_field', 'gradwarp_field') in edge['connect']


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


def test_extract_first_volume_returns_a_3d_image(tmp_path):
    """The extraction node's function must actually produce a 3D file.

    CreateNonlinearityDisplacementMap's underlying tool reads its reference
    with a 3D-only reader (readImageD<ImageType3D>), so a 4D DWI series would
    throw at runtime if fed to it directly.
    """
    import nibabel as nb

    from qsiprep.workflows.dwi.base import _extract_first_volume

    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz', nvols=6)
    out = _extract_first_volume(str(dwi), newpath=str(tmp_path))

    out_img = nb.load(out)
    assert out_img.ndim == 3
    # Same grid as the input series (the field only depends on the grid).
    in_img = nb.load(str(dwi))
    assert out_img.shape == in_img.shape[:3]
    assert (out_img.affine == in_img.affine).all()


def test_extract_first_volume_passes_an_already_3d_image_through(tmp_path):
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.base import _extract_first_volume

    path = tmp_path / 'vol.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(path))

    assert _extract_first_volume(str(path)) == str(path)


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


# --- Task 10: gradwarp-correcting the SDC estimation inputs ------------------


def _edge_pairs(wf, src_name, dst_name):
    """``(source_field, dest_field)`` pairs on the edge between two named nodes."""
    edge = wf._graph.get_edge_data(wf.get_node(src_name), wf.get_node(dst_name))
    return [] if edge is None else list(edge['connect'])


def _connects(wf, src_name, dst_name, source_field, dest_field):
    """True when ``src.source_field`` feeds ``dst.dest_field``.

    Sources wrapped in a helper function (``(('gradwarp_field', _listify), ...)``)
    are matched on the field name alone.
    """
    for source, dest in _edge_pairs(wf, src_name, dst_name):
        name = source[0] if isinstance(source, tuple) else source
        if name == source_field and dest == dest_field:
            return True
    return False


def _rpe_unit(tmp_path, image_type=None):
    from qsiplan.models import CorrectionMethod

    main = write_dwi_with_gradients(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = write_dwi_with_gradients(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
        metadata=metadata,
    )


def _syn_unit(tmp_path):
    from qsiplan.models import CorrectionMethod

    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    return make_preproc_unit(
        [dwi],
        method=CorrectionMethod.NIPREPS_SYN,
        estimation_sources=[str(tmp_path / 'sub-01_T1w.nii.gz')],
        metadata={'Manufacturer': 'SIEMENS'},
    )


def _cfg_for_fsl(tmp_path, pepolar_method):
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    config.workflow.hmc_model = 'eddy'
    config.workflow.pepolar_method = pepolar_method
    config.workflow.b0_threshold = 100
    config.workflow.eddy_config = None
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1


def _fsl_wf(tmp_path, unit):
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    return init_fsl_hmc_wf(unit, source_file='/data/x_dwi.nii.gz', t2w_sdc=False)


def test_fsl_hmc_wf_exposes_a_gradwarp_field_input(tmp_path):
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_topup_branch_does_not_gradwarp_sdc_inputs(tmp_path):
    """eddy applies the TOPUP field to raw data, so the field must be estimated
    on raw data too -- it is baked in upstream of ``ComposeTransforms``."""
    _cfg_for_fsl(tmp_path, 'TOPUP')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))
    assert wf.get_node('gradwarp_sdc_inputs') is None
    # Positively: topup still estimates from the raw b=0 series, with nothing
    # interposed under any name.
    assert _connects(wf, 'gather_inputs', 'topup', 'topup_imain', 'in_file')


def test_drbuddi_branch_gradwarps_sdc_inputs(tmp_path):
    """DRBUDDI's warp is applied downstream of gradwarp, so its inputs must be
    corrected first -- matching ``DRBUDDI::Step0_CreateImages``."""
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))

    assert wf.get_node('gradwarp_sdc_inputs') is not None
    # The field actually reaches the resampling node...
    assert _connects(wf, 'inputnode', 'gradwarp_sdc_inputs', 'gradwarp_field', 'transforms')
    # ...the corrected volumes actually reach DRBUDDI...
    assert _connects(
        wf, 'gradwarp_sdc_inputs', 'drbuddi_sdc_wf', 'output_image', 'inputnode.dwi_files'
    )
    # ...and the raw volumes no longer do.
    assert not _connects(
        wf, 'split_eddy_lps', 'drbuddi_sdc_wf', 'dwi_files', 'inputnode.dwi_files'
    )


def test_drbuddi_plus_topup_still_gradwarps_the_drbuddi_inputs(tmp_path):
    """The rule is per SDC node, not per workflow.

    In the mixed method eddy bakes the TOPUP field into its output, but
    DRBUDDI then runs on that output and its warp still lands in
    ``to_dwi_ref_warps`` -- downstream of gradwarp.
    """
    _cfg_for_fsl(tmp_path, 'DRBUDDI+TOPUP')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))

    assert wf.get_node('topup') is not None
    assert _connects(
        wf, 'gradwarp_sdc_inputs', 'drbuddi_sdc_wf', 'output_image', 'inputnode.dwi_files'
    )


def test_fsl_syn_branch_gradwarps_the_sdc_reference(tmp_path):
    """SyN's warp stays in ``to_dwi_ref_warps``, so estimate it on corrected b0s."""
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _syn_unit(tmp_path))

    assert _connects(wf, 'gradwarp_sdc_inputs', 'sdc_wf', 'output_image', 'inputnode.b0_ref')
    assert _connects(
        wf, 'gradwarp_sdc_inputs_brain', 'sdc_wf', 'output_image', 'inputnode.b0_ref_brain'
    )
    assert _connects(wf, 'gradwarp_sdc_inputs_mask', 'sdc_wf', 'output_image', 'inputnode.b0_mask')
    assert not _connects(
        wf, 'b0_ref_for_coreg', 'sdc_wf', 'outputnode.ref_image', 'inputnode.b0_ref'
    )
    # A binary mask must not be sinc-interpolated.
    assert wf.get_node('gradwarp_sdc_inputs_mask').inputs.interpolation == 'NearestNeighbor'


def test_gradwarp_sdc_resampling_nodes_write_float(tmp_path):
    """``gradwarp_sdc_inputs`` is a MapNode over every volume in the series.

    Every adjacent resampling node in the codebase sets ``float=True``
    (``resampling.py``, ``diffprep.py``); without it these would be the only
    double-precision per-volume nodes in the pipeline.
    """
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _syn_unit(tmp_path))

    for name in ('gradwarp_sdc_inputs', 'gradwarp_sdc_inputs_brain', 'gradwarp_sdc_inputs_mask'):
        assert wf.get_node(name).inputs.float is True, name


def test_gradwarp_sdc_resampling_honours_sloppy(tmp_path):
    """Matches the adjacent per-volume ApplyTransforms in hmc_sdc.py."""
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    config.execution.sloppy = True
    try:
        wf = _fsl_wf(tmp_path, _syn_unit(tmp_path))
        assert wf.get_node('gradwarp_sdc_inputs').inputs.interpolation == 'NearestNeighbor'
    finally:
        config.execution.sloppy = False


def test_no_gradwarp_node_without_a_coefficient_file(tmp_path):
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    config.workflow.gradient_file = None
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))
    assert wf.get_node('gradwarp_sdc_inputs') is None


def test_dis3d_does_not_gradwarp_sdc_inputs(tmp_path):
    """No spatial correction means nothing to apply before SDC estimation."""
    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path, ['ORIGINAL', 'DIS3D']))
    assert wf.get_node('gradwarp_sdc_inputs') is None


def _cfg_for_diffprep(tmp_path):
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    config.workflow.hmc_model = 'tortoise'
    config.workflow.diffprep_config = None
    config.workflow.b0_threshold = 100
    config.workflow.pepolar_method = 'DRBUDDI'
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.gpu = None
    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1


def _diffprep_wf(tmp_path, unit):
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    return init_diffprep_hmc_wf(unit, source_file='/data/x_dwi.nii.gz', t2w_sdc=False)


def test_diffprep_hmc_wf_exposes_a_gradwarp_field_input(tmp_path):
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_wf(tmp_path, _rpe_unit(tmp_path))
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_diffprep_drbuddi_branch_gradwarps_sdc_inputs(tmp_path):
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_wf(tmp_path, _rpe_unit(tmp_path))

    assert _connects(wf, 'inputnode', 'gradwarp_sdc_inputs', 'gradwarp_field', 'transforms')
    assert _connects(
        wf, 'gradwarp_sdc_inputs', 'drbuddi_sdc_wf', 'output_image', 'inputnode.dwi_files'
    )
    assert not _connects(wf, 'split_outputs', 'drbuddi_sdc_wf', 'dwi_files', 'inputnode.dwi_files')


def test_diffprep_dis3d_does_not_gradwarp_sdc_inputs(tmp_path):
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_wf(tmp_path, _rpe_unit(tmp_path, ['ORIGINAL', 'DIS3D']))
    assert wf.get_node('gradwarp_sdc_inputs') is None


def test_diffprep_syn_branch_gradwarps_the_sdc_reference(tmp_path):
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_wf(tmp_path, _syn_unit(tmp_path))

    assert _connects(wf, 'gradwarp_sdc_inputs', 'sdc_wf', 'output_image', 'inputnode.b0_ref')
    assert not _connects(
        wf, 'b0_ref_for_coreg', 'sdc_wf', 'outputnode.ref_image', 'inputnode.b0_ref'
    )


# --- The coregistration reference ---------------------------------------------
#
# ComposeTransforms applies the b0->T1w affine after gradwarp, so the b=0 that
# affine is estimated from must be gradwarp-corrected too. The DRBUDDI and
# GRE/SyN branches get that for free from their SDC correction; these are the
# branches that need it wired explicitly.


def _plain_unit(tmp_path, image_type=None):
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit([dwi], metadata=metadata)


def test_topup_only_branch_gradwarps_the_coregistration_reference(tmp_path):
    """eddy has already applied TOPUP's field to this image, so gradwarp is the
    only transform still missing before coregistration. Correcting it does not
    touch TOPUP, whose own inputs stay raw (asserted separately above)."""
    _cfg_for_fsl(tmp_path, 'TOPUP')
    wf = _fsl_wf(tmp_path, _rpe_unit(tmp_path))

    assert _connects(wf, 'gradwarp_coreg_ref', 'outputnode', 'output_image', 'b0_template')
    assert not _connects(
        wf, 'b0_ref_for_coreg', 'outputnode', 'outputnode.ref_image', 'b0_template'
    )
    assert _connects(wf, 'inputnode', 'gradwarp_coreg_ref', 'gradwarp_field', 'transforms')


def test_fsl_no_fieldmap_branch_gradwarps_the_coregistration_reference(tmp_path):
    _cfg_for_fsl(tmp_path, 'TOPUP')
    wf = _fsl_wf(tmp_path, _plain_unit(tmp_path))

    assert _connects(wf, 'gradwarp_coreg_ref', 'outputnode', 'output_image', 'b0_template')
    assert not _connects(
        wf, 'b0_ref_for_coreg', 'outputnode', 'outputnode.ref_image', 'b0_template'
    )


@pytest.mark.parametrize('unit_factory', [_rpe_unit, _plain_unit])
def test_fsl_dis3d_leaves_the_coregistration_reference_raw(tmp_path, unit_factory):
    """No field means nothing to apply -- the node must not be built."""
    _cfg_for_fsl(tmp_path, 'TOPUP')
    wf = _fsl_wf(tmp_path, unit_factory(tmp_path, ['ORIGINAL', 'DIS3D']))

    assert wf.get_node('gradwarp_coreg_ref') is None
    assert _connects(wf, 'b0_ref_for_coreg', 'outputnode', 'outputnode.ref_image', 'b0_template')


def test_diffprep_no_fieldmap_branch_gradwarps_the_coregistration_reference(tmp_path):
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_wf(tmp_path, _plain_unit(tmp_path))

    assert _connects(wf, 'gradwarp_coreg_ref', 'outputnode', 'output_image', 'b0_template')
    assert not _connects(
        wf, 'b0_ref_for_coreg', 'outputnode', 'outputnode.ref_image', 'b0_template'
    )


def _diffprep_t2wreg_wf(tmp_path, unit):
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    return init_diffprep_hmc_wf(unit, source_file='/data/x_dwi.nii.gz', t2w_sdc=True)


def test_diffprep_t2wreg_reference_gets_gradwarp_then_sdc(tmp_path):
    """Both transforms go into the one resampling of the pre-SDC b=0 that this
    branch already performs, in chain order. ANTs applies a transform list
    last-first, so the SDC warp is in1 and the gradwarp field in2."""
    _cfg_for_diffprep(tmp_path)
    wf = _diffprep_t2wreg_wf(tmp_path, _plain_unit(tmp_path))

    assert _connects(wf, 'diffprep', 'sdc_then_gradwarp', 'sdc_warp', 'in1')
    assert _connects(wf, 'inputnode', 'sdc_then_gradwarp', 'gradwarp_field', 'in2')
    assert _connects(wf, 'sdc_then_gradwarp', 'apply_sdc_to_b0', 'out', 'transforms')
    # No second correction downstream: b0_ref_for_coreg derives the mask from
    # this same image, so both stay in one geometry.
    assert wf.get_node('gradwarp_coreg_ref') is None
    assert _connects(wf, 'b0_ref_for_coreg', 'outputnode', 'outputnode.ref_image', 'b0_template')


def test_diffprep_t2wreg_without_gradwarp_applies_sdc_alone(tmp_path):
    _cfg_for_diffprep(tmp_path)
    config.workflow.gradient_file = None
    wf = _diffprep_t2wreg_wf(tmp_path, _plain_unit(tmp_path))

    assert wf.get_node('sdc_then_gradwarp') is None
    assert wf.get_node('gradwarp_coreg_ref') is None
    assert _connects(wf, 'diffprep', 'apply_sdc_to_b0', 'sdc_warp', 'transforms')


def _cfg_for_shoreline(tmp_path):
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    config.workflow.hmc_model = '3dSHORE'
    config.workflow.hmc_transform = 'Affine'
    config.workflow.shoreline_iters = 2
    config.workflow.b0_threshold = 100
    config.workflow.b0_motion_corr_to = 'iterative'
    config.workflow.pepolar_method = 'DRBUDDI'
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1


def _shoreline_wf(tmp_path, unit):
    from qsiprep.workflows.dwi.hmc_sdc import init_qsiprep_hmcsdc_wf

    return init_qsiprep_hmcsdc_wf(
        unit,
        source_file='/data/x_dwi.nii.gz',
        t2w_sdc=False,
        anatomical_template='MNI152NLin2009cAsym',
    )


def test_hmcsdc_wf_exposes_a_gradwarp_field_input(tmp_path):
    _cfg_for_shoreline(tmp_path)
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path))
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_shoreline_drbuddi_branch_gradwarps_sdc_inputs(tmp_path):
    _cfg_for_shoreline(tmp_path)
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path))

    assert _connects(wf, 'inputnode', 'gradwarp_sdc_inputs', 'gradwarp_field', 'transforms')
    assert _connects(
        wf, 'gradwarp_sdc_inputs', 'drbuddi_sdc_wf', 'output_image', 'inputnode.dwi_files'
    )
    assert not _connects(
        wf, 'uncorrect_model_images', 'drbuddi_sdc_wf', 'output_image', 'inputnode.dwi_files'
    )


def test_shoreline_dis3d_does_not_gradwarp_sdc_inputs(tmp_path):
    _cfg_for_shoreline(tmp_path)
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path, ['ORIGINAL', 'DIS3D']))
    assert wf.get_node('gradwarp_sdc_inputs') is None


def test_shoreline_syn_branch_gradwarps_the_sdc_reference(tmp_path):
    _cfg_for_shoreline(tmp_path)
    wf = _shoreline_wf(tmp_path, _syn_unit(tmp_path))

    assert _connects(wf, 'gradwarp_sdc_inputs', 'sdc_wf', 'output_image', 'inputnode.b0_ref')
    assert _connects(wf, 'gradwarp_sdc_inputs_mask', 'sdc_wf', 'output_image', 'inputnode.b0_mask')
    assert not _connects(
        wf, 'dwi_hmc_wf', 'sdc_wf', 'outputnode.final_template', 'inputnode.b0_ref'
    )


def test_shoreline_without_a_fieldmap_gradwarps_the_bypass_reference(tmp_path):
    """No fieldmap means ``init_sdc_wf`` is a pure pass-through, so there is no
    susceptibility field to estimate -- but the bypass forwards ``b0_ref``
    straight to ``outputnode.b0_template``, the DWI/T1w coregistration
    reference, and the coregistration affine is applied after gradwarp. So the
    reference must be corrected even though no SDC is happening.
    """
    _cfg_for_shoreline(tmp_path)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi], metadata={'Manufacturer': 'SIEMENS'})
    wf = _shoreline_wf(tmp_path, unit)

    assert unit.method is None
    assert wf.get_node('sdc_bypass_wf') is not None
    assert _connects(
        wf, 'gradwarp_sdc_inputs', 'sdc_bypass_wf', 'output_image', 'inputnode.b0_ref'
    )
    assert not _connects(
        wf, 'dwi_hmc_wf', 'sdc_bypass_wf', 'outputnode.final_template', 'inputnode.b0_ref'
    )


def test_shoreline_dis3d_without_a_fieldmap_leaves_the_bypass_reference_raw(tmp_path):
    """A DIS3D unit has no field to apply, so the reference stays raw and the
    correction nodes are not built at all."""
    _cfg_for_shoreline(tmp_path)
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit(
        [dwi], metadata={'Manufacturer': 'SIEMENS', 'ImageType': ['ORIGINAL', 'DIS3D']}
    )
    wf = _shoreline_wf(tmp_path, unit)

    assert wf.get_node('gradwarp_sdc_inputs') is None
    assert _connects(
        wf, 'dwi_hmc_wf', 'sdc_bypass_wf', 'outputnode.final_template', 'inputnode.b0_ref'
    )


def test_dwi_preproc_wf_connects_gradwarp_field_to_the_hmc_workflow(tmp_path):
    """Without this edge every ``gradwarp_sdc_inputs`` node above is dead code."""
    wf = _preproc_wf(tmp_path)

    gradwarp_wf = wf.get_node('gradwarp_wf')
    hmc_wf = wf.get_node('hmc_sdc_wf')
    edge = wf._graph.get_edge_data(gradwarp_wf, hmc_wf)
    assert edge is not None
    assert ('outputnode.gradwarp_field', 'inputnode.gradwarp_field') in edge['connect']


def test_dwi_preproc_wf_dis3d_does_not_feed_gradwarp_field_to_the_hmc_workflow(tmp_path):
    """A DIS3D unit applies no spatial correction anywhere, SDC estimation included."""
    wf = _preproc_wf(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    edge = wf._graph.get_edge_data(wf.get_node('gradwarp_wf'), wf.get_node('hmc_sdc_wf'))
    assert edge is None


# --- Task 11: the grad_dev derivative ----------------------------------------


def test_io_spec_has_a_graddev_pattern():
    """grad_dev is neither a spatial transform nor a tissue map: it needs its
    own suffix rather than xfm or dwimap."""
    import json

    from qsiprep.data import load as load_data

    with open(load_data('io_spec.json')) as handle:
        spec = json.load(handle)

    assert any('graddev' in pattern for pattern in spec['default_path_patterns'])


def test_graddev_filename_renders_with_space_entity(tmp_path):
    import gzip

    from qsiprep.interfaces import DerivativesDataSink

    # niworkflows' DerivativesDataSink _copy_any opens a ".gz"-suffixed source
    # with gzip.open regardless of the extension= kwarg below (it reads the
    # actual extension off in_file), so the payload must be real gzip content,
    # not just a byte with a .nii.gz name -- otherwise the run() raises
    # BadGzipFile before ever reaching the filename this test checks.
    payload = tmp_path / 'graddev.nii.gz'
    with gzip.open(payload, 'wb') as handle:
        handle.write(b'\x00')
    sink = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file='/data/sub-01/dwi/sub-01_dwi.nii.gz',
        space='ACPC',
        suffix='graddev',
        extension='.nii.gz',
        in_file=str(payload),
    ).run()

    out = sink.outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert out.endswith('sub-01_space-ACPC_graddev.nii.gz')


def _finalize_wf_with_gradients(tmp_path, image_type=None, write_derivatives=True):
    """A finalize_wf with a resolved gradwarp plan, for the grad_dev tests."""
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    _finalize_cfg(tmp_path)
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    unit = make_preproc_unit([dwi], metadata=metadata)
    return init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=dwi,
        output_prefix='sub-01',
        write_derivatives=write_derivatives,
    )


def test_dwi_finalize_wf_has_no_grad_dev_without_a_coefficient_file(tmp_path):
    wf = _finalize_wf(tmp_path, write_derivatives=True)
    assert wf.get_node('grad_dev') is None
    assert wf.get_node('ds_grad_dev') is None


def test_dwi_finalize_wf_builds_grad_dev_when_a_plan_resolves(tmp_path):
    wf = _finalize_wf_with_gradients(tmp_path)

    assert wf.get_node('grad_dev') is not None
    assert wf.get_node('ds_grad_dev') is not None


def test_dwi_finalize_wf_builds_grad_dev_for_dis3d(tmp_path):
    """No spatial correction happens for a DIS3D unit, but grad_dev is still
    produced -- no scanner can correct the diffusion encoding itself."""
    wf = _finalize_wf_with_gradients(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    grad_dev = wf.get_node('grad_dev')
    assert grad_dev is not None
    assert wf.get_node('ds_grad_dev') is not None


def test_dwi_finalize_wf_grad_dev_initial_image_is_extracted_not_the_raw_4d_series(tmp_path):
    """CreateGradientNonlinearityBMatrix's ``-i`` is read as a 3D NIfTI
    (TORTOISE's ``main`` calls ``readImageD<ImageType3D>`` for both ``-f`` and
    ``-i``); ``raw_concatenated`` is the raw series in a single 4D file, so it
    must never reach ``initial_image`` directly -- it needs an extraction node
    in between, same as ``gradwarp_ref`` in base.py.
    """
    wf = _finalize_wf_with_gradients(tmp_path)

    inputnode = wf.get_node('inputnode')
    grad_dev = wf.get_node('grad_dev')

    # No direct edge -- that would be the 4D raw series reaching a 3D-only tool.
    edge = wf._graph.get_edge_data(inputnode, grad_dev)
    assert edge is None or ('raw_concatenated', 'initial_image') not in edge['connect']

    # An extraction node sits between them instead.
    extractor = wf.get_node('grad_dev_initial_ref')
    assert extractor is not None

    in_edge = wf._graph.get_edge_data(inputnode, extractor)
    assert in_edge is not None
    assert ('raw_concatenated', 'in_file') in in_edge['connect']

    out_edge = wf._graph.get_edge_data(extractor, grad_dev)
    assert out_edge is not None
    assert ('out_file', 'initial_image') in out_edge['connect']


def test_dwi_finalize_wf_grad_dev_final_image_is_the_final_b0_reference(tmp_path):
    """The final b0 ref (``init_dwi_reference_wf``'s ``ref_image``) is already a
    single volume, so ``-f`` needs no extraction -- unlike ``-i``."""
    wf = _finalize_wf_with_gradients(tmp_path)

    outputnode = wf.get_node('outputnode')
    grad_dev = wf.get_node('grad_dev')
    edge = wf._graph.get_edge_data(outputnode, grad_dev)
    assert edge is not None
    assert ('t1_b0_ref', 'final_image') in edge['connect']


def test_dwi_finalize_wf_grad_dev_sidecar_records_coefficient_basename_only(tmp_path):
    """The sidecar must never leak the host path of the coefficient file."""
    wf = _finalize_wf_with_gradients(tmp_path)

    ds_grad_dev = wf.get_node('ds_grad_dev')
    meta = ds_grad_dev.inputs.meta_dict
    assert meta['GradientCoefficientFile'] == 'coeff.grad'
    assert '/' not in meta['GradientCoefficientFile']
    assert str(tmp_path) not in meta['GradientCoefficientFile']


def test_dwi_finalize_wf_grad_dev_sidecar_records_the_orientation_approximation(tmp_path):
    """The L matrix is oriented by a transform TORTOISE re-derives internally,
    not by the coregistration affine qsiprep resampled the data with. A reader
    cannot tell that from the file, so the sidecar has to say it."""
    wf = _finalize_wf_with_gradients(tmp_path)

    meta = wf.get_node('ds_grad_dev').inputs.meta_dict
    note = meta['GradientDeviationOrientation']

    assert 'CreateGradientNonlinearityBMatrix' in note
    assert 'not by the coregistration transform' in note


def test_dwi_finalize_wf_adds_gradient_warp_dimensions_to_the_main_sidecar(tmp_path):
    wf = _finalize_wf_with_gradients(tmp_path, image_type=['ORIGINAL', 'DIS3D'])

    merged_sidecar = wf.get_node('merged_sidecar')
    assert merged_sidecar.inputs.sidecar_data['GradientWarpDimensions'] == 'none'


def test_dwi_finalize_wf_main_sidecar_has_no_gradient_warp_dimensions_without_a_plan(tmp_path):
    wf = _finalize_wf(tmp_path, write_derivatives=True)

    merged_sidecar = wf.get_node('merged_sidecar')
    assert 'GradientWarpDimensions' not in merged_sidecar.inputs.sidecar_data


def test_dwi_finalize_wf_grad_dev_sidecar_records_is_ge_as_a_boolean(tmp_path):
    """The flag resolved is whether TORTOISE's GE code path was taken.

    A key named ``...Manufacturer`` implies a real DICOM Manufacturer value,
    and ``'non-GE'`` is not one. This ships into derivative sidecars that
    downstream readers parse, so the name has to be honest.
    """
    wf = _finalize_wf_with_gradients(tmp_path)

    meta = wf.get_node('ds_grad_dev').inputs.meta_dict
    assert meta['GradientCoefficientIsGE'] is False
    assert 'GradientCoefficientManufacturer' not in meta


def test_dwi_finalize_wf_grad_dev_initial_image_is_the_first_b0(tmp_path):
    """``-i`` is the native-space counterpart of ``-f``, a b=0 in ACPC space.

    Volume 0 of the raw series is not guaranteed to be a b=0; if it is
    diffusion-weighted, whatever transform the tool derives between the two is
    cross-contrast, and a bad one gives a silently wrong L map. Picking the
    first b=0 costs nothing -- same grid, same affine.
    """
    wf = _finalize_wf_with_gradients(tmp_path)

    extractor = wf.get_node('grad_dev_initial_ref')
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), extractor)
    assert ('b0_indices', 'b0_indices') in edge['connect']


def test_extract_first_b0_picks_the_named_volume(tmp_path):
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.finalize import _extract_first_b0

    data = np.zeros((4, 4, 4, 5), dtype='float32')
    for volume in range(5):
        data[..., volume] = volume
    path = tmp_path / 'raw.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))

    out = _extract_first_b0(str(path), [3, 4], newpath=str(tmp_path))
    out_img = nb.load(out)
    assert out_img.ndim == 3
    assert np.allclose(np.asanyarray(out_img.dataobj), 3)


def test_extract_first_b0_falls_back_to_volume_zero(tmp_path):
    """An empty or unconnected ``b0_indices`` must not crash the node."""
    import nibabel as nb
    import numpy as np
    from nipype.interfaces.base import Undefined

    from qsiprep.workflows.dwi.finalize import _extract_first_b0

    data = np.zeros((4, 4, 4, 3), dtype='float32')
    data[..., 0] = 7
    path = tmp_path / 'raw.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))

    for indices in ([], Undefined):
        out = _extract_first_b0(str(path), indices, newpath=str(tmp_path))
        assert np.allclose(np.asanyarray(nb.load(out).dataobj), 7)


def test_extract_first_b0_passes_an_already_3d_image_through(tmp_path):
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.finalize import _extract_first_b0

    path = tmp_path / 'vol.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(str(path))

    assert _extract_first_b0(str(path), [2]) == str(path)


def test_dwi_finalize_wf_warns_when_graddev_will_not_be_written(tmp_path, caplog):
    """--distortion-group-merge writes its outputs from a workflow with no
    grad_dev node. Silence is the one unacceptable option."""
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        _finalize_wf_with_gradients(tmp_path, write_derivatives=False)

    assert 'graddev' in caplog.text
    assert 'distortion-group-merge' in caplog.text


def test_dwi_finalize_wf_does_not_warn_when_graddev_is_written(tmp_path, caplog):
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        _finalize_wf_with_gradients(tmp_path, write_derivatives=True)

    assert 'graddev' not in caplog.text
