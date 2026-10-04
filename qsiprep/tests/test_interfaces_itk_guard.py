"""GuardRefinedTransform keeps a refinement only when it stays near its starting transform."""

import nibabel as nb
import numpy as np
import pytest
import SimpleITK as sitk

from qsiprep.interfaces.itk import GuardRefinedTransform


def _write(tmp_path, name, rotation=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    xfm = sitk.Euler3DTransform()
    xfm.SetCenter((20.0, -10.0, 5.0))
    xfm.SetRotation(*np.radians(rotation))
    xfm.SetTranslation(tuple(float(v) for v in translation))
    composite = sitk.CompositeTransform([xfm])
    path = tmp_path / name
    sitk.WriteTransform(composite, str(path))
    return str(path)


@pytest.fixture
def reference(tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (-40.0, -50.0, -30.0)
    path = tmp_path / 'ref.nii.gz'
    nb.Nifti1Image(np.ones((40, 50, 30), dtype=np.float32), affine).to_filename(path)
    return str(path)


def _run(tmp_path, reference, initial, refined, **kwargs):
    node = GuardRefinedTransform(
        initial_transform=initial, refined_transform=refined, reference_image=reference, **kwargs
    )
    node.inputs.trait_set()
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return node.run().outputs
    finally:
        os.chdir(cwd)


def test_small_refinement_is_kept(tmp_path, reference):
    initial = _write(
        tmp_path, 'initial.h5', rotation=(5.0, -3.0, 2.0), translation=(3.0, -4.0, 1.0)
    )
    refined = _write(
        tmp_path, 'refined.h5', rotation=(5.5, -3.0, 2.0), translation=(3.0, -2.0, 1.0)
    )
    out = _run(tmp_path, reference, initial, refined)
    assert out.accepted
    assert out.shift_mm == pytest.approx(2.0, abs=0.3)
    assert out.rotation_deg == pytest.approx(0.5, abs=0.05)
    assert np.allclose(
        sitk.ReadTransform(out.out_transform).TransformPoint((1.0, 2.0, 3.0)),
        sitk.ReadTransform(refined).TransformPoint((1.0, 2.0, 3.0)),
    )


def test_runaway_refinement_falls_back_to_the_initial_transform(tmp_path, reference):
    initial = _write(tmp_path, 'initial.h5', translation=(0.0, -7.7, 0.0))
    refined = _write(
        tmp_path, 'refined.h5', rotation=(25.0, 10.0, 5.0), translation=(60.0, -75.0, 50.0)
    )
    out = _run(tmp_path, reference, initial, refined)
    assert not out.accepted
    assert out.shift_mm > 10
    assert out.rotation_deg > 10
    assert np.allclose(
        sitk.ReadTransform(out.out_transform).TransformPoint((1.0, 2.0, 3.0)),
        sitk.ReadTransform(initial).TransformPoint((1.0, 2.0, 3.0)),
    )


def test_rotation_alone_can_reject(tmp_path, reference):
    initial = _write(tmp_path, 'initial.h5')
    refined = _write(tmp_path, 'refined.h5', rotation=(12.0, 0.0, 0.0))
    out = _run(tmp_path, reference, initial, refined, max_shift_mm=1000.0)
    assert not out.accepted
    assert out.rotation_deg == pytest.approx(12.0, abs=0.05)
