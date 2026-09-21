"""Tests for the qsiprep.workflows.anatomical.volume module."""

import os

import nibabel as nb
import numpy as np
import pytest


@pytest.fixture(scope='module')
def t1w_pair(data_dir, tmp_path_factory):
    """Two copies of one real T1w, the second shifted by a known translation.

    Only the affine is shifted, so both images hold identical voxels and the
    rigid transform between them is known exactly.
    """
    if not data_dir:
        pytest.skip('--data_dir was not provided')

    source = os.path.join(
        data_dir,
        'forrest_gump',
        'sub-01',
        'ses-forrestgump',
        'anat',
        'sub-01_ses-forrestgump_rec-autobox_T1w.nii.gz',
    )
    if not os.path.isfile(source):
        pytest.skip(f'forrest_gump dataset is unavailable; missing {source}')

    tmpdir = tmp_path_factory.mktemp('t1w_pair')
    img = nb.load(source)
    first = str(tmpdir / 'sub-01_run-01_T1w.nii.gz')
    second = str(tmpdir / 'sub-01_run-02_T1w.nii.gz')

    img.to_filename(first)
    affine = img.affine.copy()
    affine[:3, 3] += [6.0, -4.0, 2.0]
    img.__class__(np.asanyarray(img.dataobj), affine, img.header).to_filename(second)
    return [first, second]


@pytest.mark.parametrize(
    ('num_images', 'reference', 'shifts'),
    [
        # "unbiased" recentres the average by the inverse of the mean transform,
        # so the template sits between the inputs and both move half way.
        (2, 'unbiased', (0.5, 0.5)),
        # "first-lex" registers onto the first image, which therefore does not
        # move, while the second takes up the whole offset.
        (2, 'first-lex', (0.0, 1.0)),
        # One image is already the reference, so there is nothing to merge.
        (1, 'unbiased', (0.0,)),
    ],
)
def test_subject_anatomical_reference_places_the_template(
    t1w_pair, tmp_path, num_images, reference, shifts
):
    """``--subject-anatomical-reference`` decides where the merged template lands.

    Runs the merge on real anatomicals a known distance apart and reads the
    resulting transforms to see how far each image had to move.
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from scipy.io import loadmat

    from qsiprep import config
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    images = t1w_pair[:num_images]

    config.execution.sloppy = False
    config.execution.output_dir = str(tmp_path)
    config.nipype.omp_nthreads = 2
    config.workflow.anat_modality = 'T1w'
    config.workflow.anat_biascorrect = 'none'
    config.workflow.hmc_transform = 'Rigid'
    config.workflow.subject_anatomical_reference = reference

    workflow = pe.Workflow(name='test_wf')
    workflow.base_dir = str(tmp_path)
    template_wf = init_anat_template_wf(num_images=len(images), do_biascorr=False)
    template_wf.inputs.inputnode.images = images
    # nipype drops IdentityInterface nodes when it flattens a workflow to run it,
    # so outputnode is not in the graph run() returns. Hand the outputs to a node
    # that survives.
    outputs = pe.Node(
        niu.Function(
            input_names=['template', 'transforms'],
            output_names=['template', 'transforms'],
            function='def collect(template, transforms):\n    return template, transforms\n',
        ),
        name='collect',
    )
    workflow.connect([
        (template_wf, outputs, [
            ('outputnode.template', 'template'),
            ('outputnode.template_transforms', 'transforms'),
        ]),
    ])  # fmt:skip

    graph = workflow.run(plugin='Linear')
    result = next(node for node in graph.nodes() if node.name == 'collect').result.outputs
    assert os.path.isfile(result.template)
    assert len(result.transforms) == num_images

    translations = []
    for transform in result.transforms:
        # antsRegistration returns a list of transforms per image.
        path = transform[0] if isinstance(transform, list | tuple) else transform
        if not path.endswith('.mat'):
            # A lone image gets the shipped identity transform, not a fitted one.
            assert 'itkIdentityTransform' in path
            translations.append(np.zeros(3))
            continue
        params = loadmat(path)
        key = next(k for k in params if 'AffineTransform' in k)
        translations.append(np.asarray(params[key]).ravel()[9:12])

    origins = [nb.load(image).affine[:3, 3] for image in images]
    separation = np.linalg.norm(origins[-1] - origins[0])

    # Registration recovers the offset to within ~0.02 mm, against a 3.7 mm gap
    # between the two placements being told apart.
    for translation, shift in zip(translations, shifts, strict=True):
        assert np.linalg.norm(translation) == pytest.approx(shift * separation, abs=0.25)

    # Wherever the template is placed, the images stay the same distance apart.
    assert np.linalg.norm(translations[-1] - translations[0]) == pytest.approx(
        separation, abs=0.25
    )
