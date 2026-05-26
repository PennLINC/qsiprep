"""Tests for the qsiprep.interfaces.ants module."""

import os

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.ants import MultivariateTemplateConstruction2


def _make_input_image(path):
    """Write a tiny NIfTI so File(exists=True) traits are satisfied."""
    img = nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4))
    img.to_filename(path)


def _touch(path):
    """Create an empty file so File(exists=True) validates against it."""
    with open(path, 'w') as fobj:
        fobj.write('')


def test_mvtc2_list_outputs_affine_only(tmp_path, monkeypatch):
    """Affine-only MVTC2 must emit single-element transform lists.

    Regression test: with ``transform='Affine'`` the antsMultivariateTemplateConstruction2.sh
    script does not produce ``*1Warp.nii.gz`` / ``*1InverseWarp.nii.gz`` files. The previous
    implementation hunted for them unconditionally, so ``_list_outputs`` returned non-existent
    paths and the ``File(exists=True)`` trait on the OutputSpec rejected the result.
    """
    monkeypatch.chdir(tmp_path)
    inputs = [str(tmp_path / f'img{i}.nii.gz') for i in range(2)]
    for path in inputs:
        _make_input_image(path)

    # MVTC2 writes affine files into cwd. Mimic only the affines (no warps).
    prefix = 'antsBTP'
    expected_affines = []
    for i, in_path in enumerate(inputs):
        fname = os.path.basename(in_path).replace('.nii.gz', '')
        affine_path = tmp_path / f'{prefix}{fname}{i}0GenericAffine.mat'
        _touch(str(affine_path))
        expected_affines.append(str(affine_path))

    interface = MultivariateTemplateConstruction2(
        input_images=inputs,
        transform='Affine',
        dimension=3,
    )
    outputs = interface._list_outputs()

    assert outputs['forward_transforms'] == [[a] for a in expected_affines]
    assert outputs['reverse_transforms'] == [[a] for a in expected_affines]


def test_mvtc2_list_outputs_nonlinear_unchanged(tmp_path, monkeypatch):
    """Nonlinear (default) MVTC2 still emits ``[affine, warp]`` / ``[inv_warp, affine]`` pairs."""
    monkeypatch.chdir(tmp_path)
    inputs = [str(tmp_path / f'img{i}.nii.gz') for i in range(2)]
    for path in inputs:
        _make_input_image(path)

    prefix = 'antsBTP'
    expected_forward = []
    expected_reverse = []
    for i, in_path in enumerate(inputs):
        fname = os.path.basename(in_path).replace('.nii.gz', '')
        affine_path = tmp_path / f'{prefix}{fname}{i}0GenericAffine.mat'
        warp_path = tmp_path / f'{prefix}{fname}{i}1Warp.nii.gz'
        inv_warp_path = tmp_path / f'{prefix}{fname}{i}1InverseWarp.nii.gz'
        for path in (affine_path, warp_path, inv_warp_path):
            _touch(str(path))
        expected_forward.append([str(affine_path), str(warp_path)])
        expected_reverse.append([str(inv_warp_path), str(affine_path)])

    interface = MultivariateTemplateConstruction2(
        input_images=inputs,
        # transform defaults to BSplineSyN
        dimension=3,
    )
    outputs = interface._list_outputs()

    assert outputs['forward_transforms'] == expected_forward
    assert outputs['reverse_transforms'] == expected_reverse


@pytest.mark.parametrize('transform', ['Affine', 'BSplineSyN', 'SyN'])
def test_mvtc2_templates_output(tmp_path, monkeypatch, transform):
    """Templates output is independent of transform model (covers both branches)."""
    monkeypatch.chdir(tmp_path)
    inputs = [str(tmp_path / 'img.nii.gz')]
    _make_input_image(inputs[0])

    prefix = 'antsBTP'
    fname = 'img'
    _touch(str(tmp_path / f'{prefix}{fname}00GenericAffine.mat'))
    if transform != 'Affine':
        _touch(str(tmp_path / f'{prefix}{fname}01Warp.nii.gz'))
        _touch(str(tmp_path / f'{prefix}{fname}01InverseWarp.nii.gz'))
    _touch(str(tmp_path / f'{prefix}template0.nii.gz'))

    interface = MultivariateTemplateConstruction2(
        input_images=inputs, transform=transform, dimension=3
    )
    outputs = interface._list_outputs()

    assert outputs['templates'] == [str(tmp_path / f'{prefix}template0.nii.gz')]
