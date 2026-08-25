"""Synthetic gradient nonlinearity inputs for tests.

The Siemens ``.grad`` writer targets the reader at
``src/tools/gradnonlin/gradcal.cxx:99-183`` in TORTOISEV4, which is stricter
than it looks: the axis letter must be the last character on the line (the
coefficient substring drops the final character) and the only x/y/z on it (the
axis is found by searching the whole line), and R0 is read from columns 1-5.
``test_gradient_fixtures.py`` asserts each of these.
"""

from pathlib import Path

import nibabel as nb
import numpy as np

#: A small, physically plausible third-order set. Real coefficient files hold
#: dozens of terms; these are enough to produce a non-trivial field.
DEFAULT_TERMS = [
    ('x', 3, 1, -0.0234),
    ('y', 3, 1, 0.0198),
    ('z', 3, 0, 0.0456),
    ('z', 5, 0, 0.0011),
]


def write_siemens_grad(path, terms=None, r0_m=0.250):
    """Write a Siemens-format ``.grad`` coefficient file.

    ``terms`` is a list of ``(axis, l, m, coefficient)``; ``r0_m`` is the
    reference radius in metres (TORTOISE multiplies it by 1000).
    """
    terms = DEFAULT_TERMS if terms is None else terms
    path = Path(path)
    lines = [
        ' Synthetic gradient coefficients for tests',
        f' {r0_m:.3f} = R0',
        '',
    ]
    for index, (axis, l_val, m_val, coefficient) in enumerate(terms, start=1):
        # Two leading columns put "(" at index >= 3. The coefficient is written
        # with no trailing content except the axis letter, which must be last.
        lines.append(f'{index:>3d} A({l_val:>2d},{m_val:>2d}) {coefficient: .6f} {axis}')
    lines.append('')
    path.write_text('\n'.join(lines))
    return path


def write_itk_field(path, shape=(8, 8, 8), amplitude=0.5):
    """Write a smooth, non-zero ITK displacement field as a 5D vector NIfTI."""
    path = Path(path)
    grid = np.meshgrid(
        *[np.linspace(-1.0, 1.0, n) for n in shape],
        indexing='ij',
    )
    data = np.zeros(shape + (1, 3), dtype='float32')
    for component in range(3):
        data[..., 0, component] = amplitude * grid[component] ** 2
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return path


def write_dwi_with_gradients(path, nvols=6):
    """Write a tiny 4D DWI plus sibling .bval/.bvec, and return its path."""
    path = Path(path)
    nb.Nifti1Image(
        np.random.default_rng(0).random((8, 8, 8, nvols)).astype('float32'), np.eye(4)
    ).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    bvecs = np.zeros((3, nvols))
    bvecs[0, 1:] = 1.0
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', bvecs, fmt='%.6f')
    return str(path)
