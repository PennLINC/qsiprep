"""Tests for the qsiprep.workflows.anatomical.volume module."""

import os

import nibabel as nb
import numpy as np
import pytest


def _collect(template, transforms):
    return template, transforms


@pytest.fixture(scope='module')
def t1w_pair(data_dir, tmp_path_factory):
    """One real T1w and a copy whose affine is translated by a known offset."""
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
    ('reference', 'shifts'),
    [
        # The unbiased template sits midway, so each image moves half the offset.
        ('unbiased', (0.5, 0.5)),
        # The first image is the template, so the second takes up the whole offset.
        ('first-lex', (0.0, 1.0)),
    ],
)
def test_subject_anatomical_reference_places_the_template(
    t1w_pair, tmp_path, monkeypatch, reference, shifts
):
    """``--subject-anatomical-reference`` decides where the merged template lands."""
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from scipy.io import loadmat

    from qsiprep import config
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    monkeypatch.setattr(config.execution, 'sloppy', False)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 2)
    monkeypatch.setattr(config.workflow, 'subject_anatomical_reference', reference)

    workflow = pe.Workflow(name='test_wf', base_dir=str(tmp_path))
    template_wf = init_anat_template_wf(num_images=2, do_biascorr=False)
    template_wf.inputs.inputnode.images = t1w_pair
    # outputnode is pruned from the executed graph, so collect its outputs downstream.
    collect = pe.Node(
        niu.Function(
            input_names=['template', 'transforms'],
            output_names=['template', 'transforms'],
            function=_collect,
        ),
        name='collect',
    )
    workflow.connect([
        (template_wf, collect, [
            ('outputnode.template', 'template'),
            ('outputnode.template_transforms', 'transforms'),
        ]),
    ])  # fmt:skip

    graph = workflow.run(plugin='Linear')
    result = next(node for node in graph.nodes() if node.name == 'collect').result.outputs
    assert os.path.isfile(result.template)

    translations = []
    for transform in result.transforms:
        # antsRegistration returns a list of transforms per image.
        path = transform[0] if isinstance(transform, list | tuple) else transform
        params = loadmat(path)
        key = next(k for k in params if 'AffineTransform' in k)
        translations.append(np.asarray(params[key]).ravel()[9:12])

    origins = [nb.load(image).affine[:3, 3] for image in t1w_pair]
    separation = np.linalg.norm(origins[1] - origins[0])

    for translation, shift in zip(translations, shifts, strict=True):
        assert np.linalg.norm(translation) == pytest.approx(shift * separation, abs=0.25)

    # Wherever the template is placed, the images stay the same distance apart.
    assert np.linalg.norm(translations[1] - translations[0]) == pytest.approx(separation, abs=0.25)


def test_single_image_template_uses_identity_transform(monkeypatch):
    """A single anatomical image is its own template, so it gets no fitted transform."""
    from qsiprep import config
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    template_wf = init_anat_template_wf(num_images=1, do_biascorr=False)
    transforms = template_wf.get_node('outputnode').inputs.template_transforms
    assert [os.path.basename(t) for t in transforms] == ['itkIdentityTransform.txt']
