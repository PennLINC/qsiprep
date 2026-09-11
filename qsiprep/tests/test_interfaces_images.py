"""Tests for the qsiprep.interfaces.images module."""

import shutil
import time
from pathlib import Path

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


def test_get_template_uses_resolution_and_cohort(tmp_path):
    from qsiprep.interfaces.anatomical import GetTemplate

    iface = GetTemplate(
        template_name='MNIInfant',
        cohort='2',
        resolution='2',
        anatomical_contrast='T1w',
    )
    result = iface.run(cwd=str(tmp_path))
    name = Path(result.outputs.template_file).name
    assert 'cohort-2' in name
    assert 'res-2' in name


def test_get_template_defaults_to_res_1(tmp_path):
    from qsiprep.interfaces.anatomical import GetTemplate

    iface = GetTemplate(template_name='MNI152NLin2009cAsym', anatomical_contrast='T1w')
    result = iface.run(cwd=str(tmp_path))
    assert 'res-01' in Path(result.outputs.template_file).name


def _write_image(path, zooms):
    import nibabel as nb
    import numpy as np

    affine = np.diag([*zooms, 1.0])
    nb.Nifti1Image(np.zeros((4, 4, 4)), affine).to_filename(path)
    return str(path)


def test_voxel_size_chooser_max_across_runs(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    # The largest zoom lives in the second image, so a regression that silently used
    # only input_images[0] would fail this test instead of passing by coincidence.
    a = _write_image(tmp_path / 'a.nii.gz', (2.0, 2.0, 2.0))
    b = _write_image(tmp_path / 'b.nii.gz', (3.0, 4.0, 5.0))
    result = VoxelSizeChooser(input_images=[a, b], anisotropic_strategy='max').run(
        cwd=str(tmp_path)
    )
    assert result.outputs.voxel_size == 5.0


def test_voxel_size_chooser_min_across_runs(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    a = _write_image(tmp_path / 'a.nii.gz', (3.0, 4.0, 5.0))
    b = _write_image(tmp_path / 'b.nii.gz', (2.5, 2.5, 2.5))
    result = VoxelSizeChooser(input_images=[a, b], anisotropic_strategy='min').run(
        cwd=str(tmp_path)
    )
    assert result.outputs.voxel_size == 2.5


def test_voxel_size_chooser_explicit_size_wins(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    a = _write_image(tmp_path / 'a.nii.gz', (3.0, 4.0, 5.0))
    result = VoxelSizeChooser(input_images=[a], voxel_size=1.7).run(cwd=str(tmp_path))
    assert result.outputs.voxel_size == 1.7


def test_choose_interpolator_from_grid(tmp_path):
    from qsiprep.interfaces.images import ChooseInterpolator

    dwi = _write_image(tmp_path / 'dwi.nii.gz', (2.0, 2.0, 2.0))
    coarse_grid = _write_image(tmp_path / 'coarse.nii.gz', (2.0, 2.0, 2.0))
    fine_grid = _write_image(tmp_path / 'fine.nii.gz', (1.0, 1.0, 1.0))

    same = ChooseInterpolator(dwi_files=[dwi], output_grid=coarse_grid).run(cwd=str(tmp_path))
    assert same.outputs.interpolation_method == 'LanczosWindowedSinc'

    upsampled = ChooseInterpolator(dwi_files=[dwi], output_grid=fine_grid).run(cwd=str(tmp_path))
    assert upsampled.outputs.interpolation_method == 'Linear'


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


# ---------------------------------------------------------------------------
# Concurrent TemplateFlow fetches.
#
# templateflow's client._s3_get streams a download straight into its final cache
# path (``filepath.open('wb')``), with no temp file and no rename, so the file is
# observable half-written for the whole download. --output-spaces builds one
# GetTemplate node per standard space in anat_preproc_wf and another in
# anat_derivatives_wf, with no dependency between them, so MultiProc runs them at
# the same time and one node can copy what the other is still downloading.
# ---------------------------------------------------------------------------


def _partial_download_get(cache_dir, delay):
    """Stand in for ``templateflow.api.get``: a non-atomic streaming download."""

    def _get(template_name, **kwargs):
        target = Path(cache_dir) / f'tpl-{template_name}_{kwargs.get("suffix")}.nii.gz'
        if not target.exists():
            with open(target, 'wb') as handle:
                handle.write(b'A' * 512)
                handle.flush()
                time.sleep(delay)
                handle.write(b'B' * 512)
        return target

    return _get


def _fetch_in_child(cwd, cache_dir, delay, start_after):
    """Run one GetTemplate against the fake cache, as a separate process would."""
    from unittest import mock

    time.sleep(start_after)
    from qsiprep.interfaces.anatomical import GetTemplate

    with mock.patch('templateflow.api.get', _partial_download_get(cache_dir, delay)):
        GetTemplate(template_name='FAKE', anatomical_contrast='T1w').run(cwd=str(cwd))


def test_get_template_never_copies_a_partial_download(tmp_path, monkeypatch):
    """Two nodes fetching one template must not yield a truncated copy.

    Reproduces the ``3dcalc`` failure on a multi-space run: "data bytes input =
    -1 ... Can't load dataset ... is it complete?" on a template brain mask.
    """
    import multiprocessing

    cache_dir = tmp_path / 'templateflow'
    cache_dir.mkdir()
    monkeypatch.setenv('TEMPLATEFLOW_HOME', str(cache_dir))

    first, second = tmp_path / 'node_a', tmp_path / 'node_b'
    first.mkdir()
    second.mkdir()

    ctx = multiprocessing.get_context('fork')
    # The second node starts while the first is mid-download, which is exactly
    # the interleaving the run log shows.
    procs = [
        ctx.Process(target=_fetch_in_child, args=(first, cache_dir, 1.0, 0.0)),
        ctx.Process(target=_fetch_in_child, args=(second, cache_dir, 1.0, 0.3)),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)

    copied = sorted(p for node in (first, second) for p in node.glob('tpl-FAKE_*.nii.gz'))
    assert len(copied) == 4, f'expected two copies per node, got {copied}'
    truncated = [str(p) for p in copied if p.stat().st_size != 1024]
    assert not truncated, f'copied a half-written download: {truncated}'
