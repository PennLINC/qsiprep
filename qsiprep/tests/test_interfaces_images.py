"""Tests for the qsiprep.interfaces.images module."""

import nibabel as nb
import numpy as np
from nipype.interfaces.base import isdefined

from qsiprep.interfaces.images import ConformDwi
from qsiprep.tests.utils import build_test_dataset

# An image already in LPS, and one in RAS that must be reoriented to reach LPS.
LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])
RAS_AFFINE = np.eye(4)

BARE_DWI = {'01': [{'dwi': [{'suffix': 'dwi'}]}]}

COMPLEX_DWI = {
    '01': [
        {
            'dwi': [
                {'part': 'mag', 'suffix': 'dwi'},
                {'part': 'phase', 'suffix': 'dwi'},
            ],
        },
    ],
}

GRADIENTS = {
    'sub-01/dwi/sub-01_dwi.bval': '0 1000\n',
    'sub-01/dwi/sub-01_dwi.bvec': '1 0\n0 1\n0 0\n',
}


def _run(interface, work_dir):
    """Run an interface in a fresh working directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    return interface.run(cwd=str(work_dir))


def test_conform_dwi_uses_colocated_gradients(tmp_path):
    """Gradients sitting beside the DWI are found."""
    root = build_test_dataset(
        tmp_path / 'ds', BARE_DWI, extra_files=GRADIENTS, n_volumes=2, affine=LPS_AFFINE
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    result = _run(ConformDwi(dwi_file=str(dwi_dir / 'sub-01_dwi.nii.gz')), tmp_path / 'work')

    assert result.outputs.bval_file == str(dwi_dir / 'sub-01_dwi.bval')
    assert result.outputs.bvec_file == str(dwi_dir / 'sub-01_dwi.bvec')


def test_conform_dwi_inherits_gradients(tmp_path):
    """A part-mag DWI inherits the gradients shared with its phase counterpart (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds', COMPLEX_DWI, extra_files=GRADIENTS, n_volumes=2, affine=LPS_AFFINE
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    result = _run(
        ConformDwi(dwi_file=str(dwi_dir / 'sub-01_part-mag_dwi.nii.gz')), tmp_path / 'work'
    )

    assert result.outputs.bval_file == str(dwi_dir / 'sub-01_dwi.bval')
    assert result.outputs.bvec_file == str(dwi_dir / 'sub-01_dwi.bvec')


def test_conform_dwi_inherits_gradients_for_a_phase_image(tmp_path):
    """The phase image resolves to the same shared gradients (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds', COMPLEX_DWI, extra_files=GRADIENTS, n_volumes=2, affine=LPS_AFFINE
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    result = _run(
        ConformDwi(dwi_file=str(dwi_dir / 'sub-01_part-phase_dwi.nii.gz')), tmp_path / 'work'
    )

    assert result.outputs.bval_file == str(dwi_dir / 'sub-01_dwi.bval')
    assert result.outputs.bvec_file == str(dwi_dir / 'sub-01_dwi.bvec')


def test_conform_dwi_prefers_explicit_gradients(tmp_path):
    """Explicitly supplied gradients override the ones that would be resolved."""
    root = build_test_dataset(
        tmp_path / 'ds', BARE_DWI, extra_files=GRADIENTS, n_volumes=2, affine=LPS_AFFINE
    )
    chosen_dir = tmp_path / 'elsewhere'
    chosen_dir.mkdir()
    (chosen_dir / 'chosen.bval').write_text('0 3000\n')
    (chosen_dir / 'chosen.bvec').write_text('0 1\n1 0\n0 0\n')

    result = _run(
        ConformDwi(
            dwi_file=str(root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'),
            bval_file=str(chosen_dir / 'chosen.bval'),
            bvec_file=str(chosen_dir / 'chosen.bvec'),
        ),
        tmp_path / 'work',
    )

    assert result.outputs.bval_file == str(chosen_dir / 'chosen.bval')
    assert result.outputs.bvec_file == str(chosen_dir / 'chosen.bvec')


def test_conform_dwi_flips_inherited_bvecs_on_reorientation(tmp_path):
    """Reorienting RAS to LPS negates the first two bvec rows of an inherited bvec."""
    root = build_test_dataset(
        tmp_path / 'ds', COMPLEX_DWI, extra_files=GRADIENTS, n_volumes=2, affine=RAS_AFFINE
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_part-mag_dwi.nii.gz'

    result = _run(ConformDwi(dwi_file=str(dwi), orientation='LPS'), tmp_path / 'work')

    np.testing.assert_allclose(
        np.loadtxt(result.outputs.bvec_file),
        np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]),
    )
    assert nb.aff2axcodes(nb.load(result.outputs.dwi_file).affine) == ('L', 'P', 'S')


def test_conform_dwi_without_gradients_still_conforms_the_image(tmp_path):
    """A phase image with no gradient table anywhere is reoriented without error."""
    root = build_test_dataset(tmp_path / 'ds', COMPLEX_DWI, n_volumes=2, affine=RAS_AFFINE)
    phase = root / 'sub-01' / 'dwi' / 'sub-01_part-phase_dwi.nii.gz'

    result = _run(ConformDwi(dwi_file=str(phase), orientation='LPS'), tmp_path / 'work')

    assert nb.aff2axcodes(nb.load(result.outputs.dwi_file).affine) == ('L', 'P', 'S')
    assert not isdefined(result.outputs.bval_file)
    assert not isdefined(result.outputs.bvec_file)


def test_conform_dwi_reports_bvals_when_only_bvals_exist(tmp_path):
    """A bval with no matching bvec is still reported (it used to be dropped)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        BARE_DWI,
        extra_files={'sub-01/dwi/sub-01_dwi.bval': '0 1000\n'},
        n_volumes=2,
        affine=LPS_AFFINE,
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    result = _run(ConformDwi(dwi_file=str(dwi_dir / 'sub-01_dwi.nii.gz')), tmp_path / 'work')

    assert result.outputs.bval_file == str(dwi_dir / 'sub-01_dwi.bval')
    assert not isdefined(result.outputs.bvec_file)
