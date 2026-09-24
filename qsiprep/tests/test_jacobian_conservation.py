"""Total-signal conservation: the oracle for the modulation direction.

For a pull-back map phi, corrected intensity is
``I_corr(x) = I_dist(phi(x)) . |det grad phi(x)|``, so

    integral I_corr = integral I_dist(phi(x)) |det grad phi| dx
                    = integral I_dist(y) dy

Conservation is therefore exact up to interpolation error, and it fails if the
weight is divided rather than multiplied, or if the field direction is
inverted. This is what distinguishes a correct implementation from one that is
merely self-consistent.

Requires ANTs; skips locally, runs in CircleCI's ``unit_tests`` job.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces import ants

from qsiprep.interfaces.jacobian import jacobian_determinant


def _compressing_field(path, shape=(24, 24, 24), amplitude=3.0):
    """Write a field that compresses along x, so det is well away from 1."""
    coords = np.stack(
        np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in shape], indexing='ij'),
        axis=-1,
    )
    data = np.zeros(shape + (1, 3), dtype='float32')
    # Smooth, monotone displacement along the first axis.
    data[..., 0, 0] = amplitude * np.sin(np.pi * coords[..., 0])
    # ITK reads a displacement field in LPS, so the x component is negated
    # for storage -- see ``_write_linear_field`` in test_interfaces_jacobian.py.
    data[..., 0, 0] *= -1
    img = nb.Nifti1Image(data, np.eye(4))
    img.header.set_intent(1007)
    img.to_filename(str(path))
    return str(path)


def _blob(path, shape=(24, 24, 24)):
    coords = np.stack(
        np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing='ij'),
        axis=-1,
    )
    radius = np.linalg.norm(coords, axis=-1)
    data = (100.0 * np.exp(-3.0 * radius**2)).astype('float32')
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


def test_multiplying_by_the_jacobian_conserves_total_signal(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')

    source = _blob(tmp_path / 'source.nii.gz')
    field = _compressing_field(tmp_path / 'field.nii.gz')

    warped = str(tmp_path / 'warped.nii.gz')
    xfm = ants.ApplyTransforms(
        input_image=source,
        reference_image=source,
        transforms=[field],
        output_image=warped,
        interpolation='LanczosWindowedSinc',
        dimension=3,
        float=True,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    xfm.run()

    determinant = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'), pe_axis=0)

    raw_total = float(np.asanyarray(nb.load(source).dataobj).sum())
    warped_data = np.asanyarray(nb.load(warped).dataobj)
    weights = np.asanyarray(nb.load(determinant).dataobj)

    modulated_total = float((warped_data * weights).sum())
    unmodulated_total = float(warped_data.sum())
    divided_total = float((warped_data / weights).sum())

    multiply_error = abs(modulated_total - raw_total)
    divide_error = abs(divided_total - raw_total)
    skip_error = abs(unmodulated_total - raw_total)

    assert modulated_total == pytest.approx(raw_total, rel=0.02), (
        f'Modulated total {modulated_total:.1f} should match the raw total '
        f'{raw_total:.1f}. If it is off by roughly the square of the expected '
        'factor, the weight is being applied twice.'
    )

    # A tolerance alone does not discriminate: for a mild deformation both
    # multiply and divide can land inside it. Require multiplying to beat both
    # alternatives by a clear margin, which is what actually pins the
    # convention.
    assert multiply_error * 3 < divide_error, (
        f'Multiplying by the Jacobian left an error of {multiply_error:.1f}; '
        f'dividing left {divide_error:.1f}. Multiplying must be decisively '
        'better. If they are comparable, this deformation is too mild to '
        'discriminate -- raise the amplitude in _compressing_field.'
    )
    assert multiply_error * 3 < skip_error, (
        f'Multiplying left an error of {multiply_error:.1f}; not weighting at '
        f'all left {skip_error:.1f}. If those are comparable the test proves '
        'nothing -- raise the amplitude in _compressing_field.'
    )
