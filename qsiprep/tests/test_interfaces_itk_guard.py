"""GuardRefinedTransform keeps whichever pass matched its own target better."""

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


def _run(tmp_path, reference, initial, refined, initial_similarity=-0.5, refined_similarity=-0.6):
    node = GuardRefinedTransform(
        initial_transform=initial,
        refined_transform=refined,
        reference_image=reference,
        initial_similarity=initial_similarity,
        refined_similarity=refined_similarity,
    )
    node.inputs.trait_set()
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return node.run().outputs
    finally:
        os.chdir(cwd)


def test_refinement_that_matches_better_is_kept(tmp_path, reference):
    initial = _write(
        tmp_path, 'initial.h5', rotation=(5.0, -3.0, 2.0), translation=(3.0, -4.0, 1.0)
    )
    refined = _write(
        tmp_path, 'refined.h5', rotation=(5.5, -3.0, 2.0), translation=(3.0, -2.0, 1.0)
    )
    out = _run(
        tmp_path, reference, initial, refined, initial_similarity=-0.50, refined_similarity=-0.55
    )
    assert out.accepted
    assert out.shift_mm == pytest.approx(2.0, abs=0.3)
    assert out.rotation_deg == pytest.approx(0.5, abs=0.05)
    assert np.allclose(
        sitk.ReadTransform(out.out_transform).TransformPoint((1.0, 2.0, 3.0)),
        sitk.ReadTransform(refined).TransformPoint((1.0, 2.0, 3.0)),
    )


def test_refinement_that_matches_worse_falls_back_to_the_initial_transform(tmp_path, reference):
    """A runaway second pass scores worse on its own target, however far it moved."""
    initial = _write(tmp_path, 'initial.h5', translation=(0.0, -7.7, 0.0))
    refined = _write(
        tmp_path, 'refined.h5', rotation=(25.0, 10.0, 5.0), translation=(60.0, -75.0, 50.0)
    )
    out = _run(
        tmp_path, reference, initial, refined, initial_similarity=-0.50, refined_similarity=-0.20
    )
    assert not out.accepted
    assert out.shift_mm > 10  # reported for QC, not used for the decision
    assert np.allclose(
        sitk.ReadTransform(out.out_transform).TransformPoint((1.0, 2.0, 3.0)),
        sitk.ReadTransform(initial).TransformPoint((1.0, 2.0, 3.0)),
    )


def test_a_recovered_second_pass_is_kept_however_far_it_moved(tmp_path, reference):
    """A first pass that failed is no prior: the second pass wins on its metric."""
    initial = _write(
        tmp_path, 'initial.h5', rotation=(15.0, 0.0, 0.0), translation=(13.0, 0.0, 0.0)
    )
    refined = _write(tmp_path, 'refined.h5', translation=(0.0, -2.0, 0.0))
    out = _run(
        tmp_path, reference, initial, refined, initial_similarity=-0.20, refined_similarity=-0.55
    )
    assert out.accepted
    assert out.rotation_deg == pytest.approx(15.0, abs=0.1)


def test_a_tie_keeps_the_refinement(tmp_path, reference):
    initial = _write(tmp_path, 'initial.h5')
    refined = _write(tmp_path, 'refined.h5', translation=(0.0, -2.0, 0.0))
    assert _run(tmp_path, reference, initial, refined, -0.5, -0.5).accepted
