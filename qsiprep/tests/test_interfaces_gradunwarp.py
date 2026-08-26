"""Tests for the TORTOISE gradient nonlinearity interfaces.

Pure-Python behaviour -- command-line construction and field masking -- is
tested unconditionally. Tests that exercise the real TORTOISE binaries are
guarded with ``shutil.which`` and skip when those binaries are absent. They are
*not* permanently offline: CircleCI's ``unit_tests`` job runs pytest inside the
``pennlinc/qsiprep:test`` image, which ships the TORTOISE tools.
"""

import shutil
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from qsiprep.tests.gradient_fixtures import (
    write_dwi_with_gradients,
    write_itk_field,
    write_siemens_grad,
)


def _require(*binaries):
    """Skip unless every named TORTOISE binary is on PATH."""
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        pytest.skip(f'{", ".join(missing)} required for this test')


def _components(path):
    """Return the (X, Y, Z, 3) displacement components of an ITK field."""
    data = np.asanyarray(nb.load(str(path)).dataobj)
    return data.reshape(data.shape[:3] + (3,))


@pytest.mark.parametrize(
    ('warp_dim', 'zeroed'),
    [('3D', ()), ('2D', (2,)), ('1D', (0, 1))],
)
def test_mask_warp_dimensions(tmp_path, warp_dim, zeroed):
    """2D zeroes the through-plane component; 1D keeps only that component."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim=warp_dim).run(cwd=str(tmp_path))

    before = _components(field)
    after = _components(result.outputs.out_file)
    for component in range(3):
        if component in zeroed:
            assert np.all(after[..., component] == 0)
        else:
            assert np.allclose(after[..., component], before[..., component])


def test_mask_warp_dimensions_preserves_geometry(tmp_path):
    """The masked field must stay on the same grid to compose correctly."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))

    original, masked = nb.load(str(field)), nb.load(result.outputs.out_file)
    assert masked.shape == original.shape
    assert np.allclose(masked.affine, original.affine)


def test_mask_warp_dimensions_does_not_modify_input(tmp_path):
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    before = _components(field).copy()
    MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))
    assert np.allclose(_components(field), before)


def test_displacement_map_puts_coefficients_first(tmp_path):
    """mk_displacement(argv[1], img, is_GE): coefficient file, then NIfTI.

    A stale unbuilt copy of that source has the arguments reversed; getting
    this backwards produces a plausible-looking wrong field, not an error.
    """
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(coeff_file=str(coeff), ref_image=ref)

    args = iface.cmdline.split()
    assert args[0] == 'CreateNonlinearityDisplacementMap'
    assert args[1] == str(coeff)
    assert args[2] == ref


def test_displacement_map_omits_is_ge_when_false(tmp_path):
    """The wrapper omits the argument rather than passing "0" for false.

    The built tool reads a supplied fourth argument via ``(bool)atoi(argv[4])``,
    so passing "0" would in fact correctly read as false -- unlike the stale,
    unbuilt duplicate at ``src/tools/CreateNonlinearityDisplacementMap/``, which
    casts the pointer itself and would read "0" as true. Either way, omitting
    the argument is unambiguous, so that is what the wrapper does.
    """
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(coeff_file=str(coeff), ref_image=ref, is_ge=False)
    assert len(iface.cmdline.split()) == 4


def test_displacement_map_appends_is_ge_when_true(tmp_path):
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(coeff_file=str(coeff), ref_image=ref, is_ge=True)
    args = iface.cmdline.split()
    assert len(args) == 5
    assert args[4] == '1'


def test_displacement_map_runs_on_synthetic_coefficients(tmp_path):
    """End-to-end against the real binary, in CI's container."""
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    _require('CreateNonlinearityDisplacementMap')
    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    result = CreateNonlinearityDisplacementMap(coeff_file=str(coeff), ref_image=ref).run(
        cwd=str(tmp_path)
    )

    field = nb.load(result.outputs.out_field)
    assert field.shape[:3] == (8, 8, 8)
    # A parsed coefficient file must produce a non-trivial field; an all-zero
    # result means the .grad fixture did not parse.
    assert np.abs(np.asanyarray(field.dataobj)).max() > 0


def test_bmatrix_cmdline_flags(tmp_path):
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')
    initial = write_dwi_with_gradients(tmp_path / 'initial_b0.nii.gz')
    cmd = CreateGradientNonlinearityBMatrix(
        final_image=final, initial_image=initial, nonlinearity=str(coeff)
    ).cmdline

    assert f'-f {final}' in cmd
    assert f'-i {initial}' in cmd
    assert f'-g {coeff}' in cmd


def test_bmatrix_final_image_is_staged_with_copyfile():
    """copyfile=True is what makes nipype's Node stage final_image into the
    node's working directory before running. Without it, the tool writes its
    outputs beside the *original* final_image, not beside the copy that
    _list_outputs assumes -- a bare Interface.run() never exercises staging,
    so this has to be pinned on the trait metadata directly."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    trait = CreateGradientNonlinearityBMatrix.input_spec().traits()['final_image']
    assert trait.copyfile is True


def test_bmatrix_list_outputs_resolves_against_cwd(tmp_path, monkeypatch):
    """_list_outputs must derive output paths from the *staged* final_image
    (i.e. from cwd), not from final_image's original directory -- under a
    real Node, copyfile=True stages the file into cwd, and this is the other
    half of that contract. Put final_image in one directory, chdir into a
    different one, and confirm the computed outputs land in cwd."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    original_dir = tmp_path / 'original'
    original_dir.mkdir()
    staged_dir = tmp_path / 'staged'
    staged_dir.mkdir()

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(original_dir / 'final_b0.nii.gz')

    monkeypatch.chdir(staged_dir)
    iface = CreateGradientNonlinearityBMatrix(final_image=final, nonlinearity=str(coeff))
    outputs = iface._list_outputs()

    assert Path(outputs['grad_dev']).parent == staged_dir
    assert Path(outputs['gradwarp_field']).parent == staged_dir


def test_bmatrix_is_ge_uses_a_value_not_omission(tmp_path):
    """Unlike CreateNonlinearityDisplacementMap, this tool's getIsGE() uses
    atoi(), so --isGE 0 correctly means false."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')
    iface = CreateGradientNonlinearityBMatrix(final_image=final, nonlinearity=str(coeff))

    assert '--isGE 0' in iface.cmdline
    iface.inputs.is_ge = True
    assert '--isGE 1' in iface.cmdline


def test_bmatrix_output_suffix_depends_on_nonlinearity_type(tmp_path):
    """Coefficients produce _graddev_c.nii; a field produces _graddev_f.nii."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')

    from_coeffs = CreateGradientNonlinearityBMatrix(
        final_image=final, nonlinearity=str(write_siemens_grad(tmp_path / 'coeff.grad'))
    )
    assert from_coeffs._graddev_suffix() == '_graddev_c.nii'

    from_field = CreateGradientNonlinearityBMatrix(
        final_image=final, nonlinearity=str(write_itk_field(tmp_path / 'field.nii'))
    )
    assert from_field._graddev_suffix() == '_graddev_f.nii'


def test_bmatrix_runs_on_synthetic_coefficients(tmp_path):
    """End-to-end against the real binary, in CI's container."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    _require('CreateGradientNonlinearityBMatrix')
    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz', nvols=1)
    result = CreateGradientNonlinearityBMatrix(final_image=final, nonlinearity=str(coeff)).run(
        cwd=str(tmp_path)
    )

    grad_dev = nb.load(result.outputs.grad_dev)
    # Nine components: the row-major 3x3 L matrix per voxel.
    assert grad_dev.shape[-1] == 9
