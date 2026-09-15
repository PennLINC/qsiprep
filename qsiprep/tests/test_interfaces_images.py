"""Tests for the qsiprep.interfaces.images module."""

import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces.base import isdefined

from qsiprep.interfaces import images
from qsiprep.interfaces.images import ConformDwi, bvec_to_rasb
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


class _FakeProc:
    """Stand-in for a finished subprocess.Popen."""

    def __init__(self, stdout, stderr, returncode):
        self._communicated = (stdout, stderr)
        self.returncode = returncode

    def communicate(self):
        return self._communicated


def _write_lps_dwi(tmp_path):
    """Write a small LPS-oriented 4D DWI with matching bval/bvec files."""
    rng = np.random.default_rng(0)
    img_file = tmp_path / 'dwi.nii.gz'
    data = rng.uniform(0, 100, size=(4, 4, 4, 2)).astype('f4')
    nb.Nifti1Image(data, LPS_AFFINE).to_filename(str(img_file))

    bval_file = tmp_path / 'dwi.bval'
    bval_file.write_text('0 1000\n')
    bvec_file = tmp_path / 'dwi.bvec'
    bvec_file.write_text('0 1\n0 0\n0 0\n')

    return str(img_file), str(bval_file), str(bvec_file)


@pytest.mark.skipif(shutil.which('mrinfo') is None, reason='MRtrix3 mrinfo not installed')
def test_bvec_to_rasb_tolerates_mrinfo_stderr(tmp_path):
    """LPS images make recent mrinfo write an advisory to stderr; that is not a failure."""
    img_file, bval_file, bvec_file = _write_lps_dwi(tmp_path)
    workdir = tmp_path / 'work'
    workdir.mkdir()

    rasb = bvec_to_rasb(bval_file, bvec_file, img_file, str(workdir))

    assert rasb.shape == (3,)
    assert np.all(np.isfinite(rasb))


@pytest.mark.skipif(shutil.which('mrinfo') is None, reason='MRtrix3 mrinfo not installed')
def test_bvec_to_rasb_raises_when_mrinfo_fails(tmp_path):
    """A genuinely failing mrinfo call still raises."""
    _, bval_file, bvec_file = _write_lps_dwi(tmp_path)
    workdir = tmp_path / 'work'
    workdir.mkdir()

    with pytest.raises(RuntimeError, match='return code'):
        bvec_to_rasb(bval_file, bvec_file, str(tmp_path / 'does_not_exist.nii.gz'), str(workdir))


def test_bvec_to_rasb_ignores_stderr_when_the_command_succeeds(tmp_path, monkeypatch):
    """A zero return code is success even when mrinfo writes an advisory to stderr.

    Development-branch mrinfo prints "axes realigned to approximate RAS" for any
    non-RAS image, and QSIPrep conforms everything to LPS+. This runs without the
    binary so the regression is caught wherever the suite runs.
    """
    _, bval_file, bvec_file = _write_lps_dwi(tmp_path)
    workdir = tmp_path / 'work'
    workdir.mkdir()

    advisory = b'mrinfo: Image "lps.nii.gz" axes realigned to approximate RAS\n'
    monkeypatch.setattr(
        images, 'Popen', lambda *args, **kwargs: _FakeProc(b'0 1 0 1000\n', advisory, 0)
    )

    rasb = bvec_to_rasb(bval_file, bvec_file, 'unused.nii.gz', str(workdir))

    assert np.allclose(rasb, [0, 1, 0])


def test_bvec_to_rasb_raises_on_nonzero_return_code(tmp_path, monkeypatch):
    """A non-zero return code raises, and the message keeps the command and stderr."""
    _, bval_file, bvec_file = _write_lps_dwi(tmp_path)
    workdir = tmp_path / 'work'
    workdir.mkdir()

    monkeypatch.setattr(
        images, 'Popen', lambda *args, **kwargs: _FakeProc(b'', b'mrinfo: no such file\n', 1)
    )

    with pytest.raises(RuntimeError, match='no such file'):
        bvec_to_rasb(bval_file, bvec_file, 'missing.nii.gz', str(workdir))


# fslsplit/fslmerge come from fsl-avwutils, which qsiprep no longer installs;
# these ops are reimplemented in nibabel so the FSL-eddy path needs no avwutils.


def test_split_merge_roundtrip_without_fsl(tmp_path):
    """_split_4d_to_3d then _merge_3d_to_4d reproduces the 4D image and its header.

    Matches fslsplit/fslmerge fidelity: distinct qform/sform (and codes), the
    oblique orientation, the TR in pixdim[4], and xyzt units all survive the
    round trip. (fslsplit keeps the TR even in the 3D volumes.)
    """
    from qsiprep.interfaces.images import _merge_3d_to_4d, _split_4d_to_3d

    data = np.arange(6 * 7 * 5 * 4, dtype='float32').reshape(6, 7, 5, 4)
    theta = np.deg2rad(12)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    sform = np.eye(4)
    sform[:3, :3] = rot @ np.diag([1.7, 1.7, 2.0])
    sform[:3, 3] = [10.5, -20.25, 3.125]
    qform = sform.copy()
    qform[:3, :3] = np.diag([1.7, 1.7, 2.0])  # qform axis-aligned, sform oblique
    tr = 3.7

    src = nb.Nifti1Image(data, sform)
    src.header.set_sform(sform, code=2)
    src.header.set_qform(qform, code=1)
    src.header['pixdim'][4] = tr
    src.header.set_xyzt_units('mm', 'sec')
    src.to_filename(str(tmp_path / 'dwi.nii.gz'))

    vols = _split_4d_to_3d(str(tmp_path / 'dwi.nii.gz'), str(tmp_path))
    assert len(vols) == 4
    for i, vol in enumerate(vols):
        img = nb.load(vol)
        assert img.shape == (6, 7, 5)  # a real 3D volume, not 4D-with-singleton
        assert np.allclose(img.get_fdata(), data[..., i])
        assert np.isclose(img.header['pixdim'][4], tr)  # fslsplit keeps the TR
        assert img.header.get_xyzt_units() == ('mm', 'sec')

    merged = nb.load(_merge_3d_to_4d(vols, str(tmp_path)))
    assert merged.shape == (6, 7, 5, 4)
    assert np.allclose(merged.get_fdata(), data)
    # full spatial + temporal header fidelity
    q, qc = merged.get_qform(coded=True)
    s, sc = merged.get_sform(coded=True)
    assert (qc, sc) == (1, 2)
    assert np.allclose(q, qform)
    assert np.allclose(s, sform)
    assert np.isclose(merged.header['pixdim'][4], tr)
    assert merged.header.get_xyzt_units() == ('mm', 'sec')


def test_split_dwis_fsl_uses_no_fsl_binary(tmp_path):
    """SplitDWIsFSL splits in nibabel (its node crashed CI when fslsplit vanished)."""
    from qsiprep.interfaces.images import SplitDWIsFSL

    n = 5
    nb.Nifti1Image(np.random.rand(4, 4, 4, n).astype('float32'), np.eye(4)).to_filename(
        str(tmp_path / 'dwi.nii.gz')
    )
    np.savetxt(str(tmp_path / 'dwi.bval'), [[0, 1000, 1000, 0, 1000]], fmt='%d')
    np.savetxt(str(tmp_path / 'dwi.bvec'), np.zeros((3, n)), fmt='%.1f')

    work = tmp_path / 'w'
    work.mkdir()
    result = SplitDWIsFSL(
        dwi_file=str(tmp_path / 'dwi.nii.gz'),
        bval_file=str(tmp_path / 'dwi.bval'),
        bvec_file=str(tmp_path / 'dwi.bvec'),
    ).run(cwd=str(work))

    assert len(result.outputs.dwi_files) == n
    assert result.outputs.b0_indices == [0, 3]
    assert len(result.outputs.b0_images) == 2
