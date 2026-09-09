#!/usr/bin/env python3
"""Apply gradient nonlinearity correction to one DWI series, on its own.

    python3 scripts/apply_gradunwarp.py IN_DWI OUT_DIR \
        --gradient-file coeff.grad --work-dir /tmp/gradunwarp

Runs the same pieces a full QSIPrep run uses -- ``CreateNonlinearityDisplacementMap``
to expand the coefficients onto the series' own grid, ``MaskWarpDimensions`` to
drop the components the scanner already corrected, and ``antsApplyTransforms``
to resample -- and writes the corrected series plus the before/after figure into
``OUT_DIR``.

Unlike a real run this does *not* read ``ImageType``: ``--warp-dim`` defaults to
``3D``, so the figure shows the full correction the coefficients imply. Pass
``--warp-dim 1D`` to reproduce what a ``DIS2D``-tagged acquisition actually gets
(through-plane only), which is a far smaller change.

Everything here needs the TORTOISE and ANTs binaries, so run it inside the
qsiprep container.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

LOGGER = logging.getLogger('apply_gradunwarp')


def _parse(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_file', type=Path, help='raw 4D (or 3D) DWI NIfTI')
    parser.add_argument('output_dir', type=Path, help='where to write the results')
    parser.add_argument(
        '--gradient-file',
        type=Path,
        required=True,
        help='scanner coefficients (.grad/.dat/.gc) or a ready-made ITK displacement field',
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        required=True,
        help='scratch directory for the intermediate field',
    )
    parser.add_argument(
        '--warp-dim',
        choices=['3D', '2D', '1D'],
        default='3D',
        help='which displacement components to keep (default: 3D, the full field)',
    )
    parser.add_argument('--is-ge', action='store_true', help='the coefficients are in GE format')
    parser.add_argument(
        '--interpolation',
        default='LanczosWindowedSinc',
        help='antsApplyTransforms interpolator (default: LanczosWindowedSinc)',
    )
    return parser.parse_args(argv[1:])


def _split_extension(input_file):
    """Split ``.nii.gz`` as one extension, which ``Path.suffix`` will not do."""
    name = input_file.name
    for extension in ('.nii.gz', '.nii'):
        if name.endswith(extension):
            return name[: -len(extension)], extension
    return input_file.stem, input_file.suffix


def _derivative_name(input_file):
    """``sub-x_dwi.nii.gz`` -> ``(sub-x_desc-gradunwarp_dwi, .nii.gz)``.

    ``desc-gradunwarp`` goes before the BIDS suffix, so the result is a valid
    derivative name and can never be mistaken for the raw input.
    """
    stem, extension = _split_extension(input_file)
    entities, _, suffix = stem.rpartition('_')
    if not entities:  # nothing that looks like a BIDS suffix to sit behind
        return f'{stem}_desc-gradunwarp', extension
    return f'{entities}_desc-gradunwarp_{suffix}', extension


def _field_summary(field_file, reference_file):
    """Peak displacement inside the head, which is what the figure has to show."""
    vectors = np.asarray(nb.load(str(field_file)).dataobj, dtype='float64')
    magnitude = np.linalg.norm(vectors.reshape(-1, vectors.shape[-1]), axis=1)
    reference = np.asarray(nb.load(str(reference_file)).dataobj, dtype='float64').reshape(-1)
    inside = magnitude[reference > np.percentile(reference, 60)]
    return magnitude.max(), inside.max(), np.median(inside)


def main(argv):
    options = _parse(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # Nipype interfaces write into the current directory.
    work_dir = options.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = options.input_file.resolve()
    gradient_file = options.gradient_file.resolve()

    import os

    os.chdir(work_dir)

    from nipype.interfaces import ants
    from niworkflows.interfaces.reportlets.registration import SimpleBeforeAfterRPT

    from qsiprep.interfaces.gradunwarp import (
        CreateNonlinearityDisplacementMap,
        MaskWarpDimensions,
    )
    from qsiprep.utils.gradcal import sanitize_siemens_coefficients
    from qsiprep.workflows.dwi.base import _extract_first_volume
    from qsiprep.workflows.dwi.gradwarp import is_displacement_field

    # The coefficient expander reads its reference as a 3D NIfTI, so a 4D series
    # has to be reduced first. The field depends only on the sampling grid.
    reference = _extract_first_volume(str(input_file), newpath=str(work_dir))
    print(f'reference volume: {reference}')

    # TORTOISE's Siemens reader parses comment lines as coefficients and aborts
    # on them. A full run sanitizes the file in parse_args; this script bypasses
    # the CLI, so it has to do the same or it dies with "what(): stof".
    gradient_file = sanitize_siemens_coefficients(gradient_file, work_dir, logger=LOGGER)

    if is_displacement_field(gradient_file):
        field = str(gradient_file)
        print(f'using the supplied displacement field: {field}')
    else:
        print('expanding coefficients onto the series grid ...')
        made = CreateNonlinearityDisplacementMap(
            coeff_file=str(gradient_file),
            ref_image=reference,
            is_ge=options.is_ge,
        ).run()
        field = made.outputs.out_field

    masked = MaskWarpDimensions(in_file=field, warp_dim=options.warp_dim).run()
    field = masked.outputs.out_file
    peak, head_peak, head_median = _field_summary(field, reference)
    print(
        f'displacement ({options.warp_dim}): peak {peak:.3f} mm over the FOV, '
        f'{head_peak:.3f} mm within the head, median {head_median:.3f} mm'
    )

    # The figure: the reference against itself resampled through the field, so
    # the two panels differ only by the gradwarp displacement.
    corrected_reference = ants.ApplyTransforms(
        input_image=reference,
        reference_image=reference,
        transforms=[field],
        dimension=3,
        interpolation=options.interpolation,
        float=True,
        output_image=str(work_dir / 'corrected_ref.nii.gz'),
    ).run()

    stem, extension = _derivative_name(input_file)
    report = SimpleBeforeAfterRPT(
        before=reference,
        after=corrected_reference.outputs.output_image,
        before_label='Distorted',
        after_label='Corrected',
        out_report=str(output_dir / f'{stem}.svg'),
    ).run()
    print(f'wrote {report.outputs.out_report}')

    # The full series, resampled once through the same field. -e 3 tells
    # antsApplyTransforms the input is a time series of 3D volumes.
    out_file = output_dir / f'{stem}{extension}'
    ants.ApplyTransforms(
        input_image=str(input_file),
        reference_image=reference,
        transforms=[field],
        dimension=3,
        input_image_type=3,
        interpolation=options.interpolation,
        float=True,
        output_image=str(out_file),
    ).run()
    print(f'wrote {out_file}')

    # Carry the gradients and metadata across so the result is usable as-is.
    # The b-vectors are NOT rotated: gradwarp changes the encoding per voxel,
    # which is what QSIPrep's graddev image records, not something a single
    # bvec rotation can express.
    input_stem, _ = _split_extension(input_file)
    for sidecar in ('.bval', '.bvec', '.json'):
        source = input_file.parent / (input_stem + sidecar)
        if source.exists():
            shutil.copy2(source, output_dir / f'{stem}{sidecar}')
            print(f'copied {source.name}')

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
