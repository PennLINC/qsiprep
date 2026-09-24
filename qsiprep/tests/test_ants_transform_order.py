"""Pin the ANTs transform-list ordering convention.

``ComposeJacobianWeights`` composes only the gradwarp and SDC stages out of
QSIPrep's full transform chain, which is sound only if the stages it drops sit
at the ends of that chain rather than in the middle. Working out which end
requires knowing what ANTs does with a transform list -- and ANTs and Nipype
document it as "the last specified transform will be applied first", which
describes *image warp* order and is the exact opposite of *point map* order.

Two non-commuting transforms discriminate the two candidate composites, so
this test answers the question instead of arguing about it. It also fails
loudly if a future ANTs or Nipype release changes the convention.

Guarded with ``shutil.which``: skips locally, runs in CircleCI's ``unit_tests``
job, which uses the ``pennlinc/qsiprep:test`` image.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces import ants

# Deliberately non-commuting, and discriminating in a way that survives the
# RAS/LPS sign flip between NIfTI and ITK: a uniform 2x scaling and a 10mm
# translation along one axis.
#
#   phi = B . A  (first-listed innermost)  -> phi(0) = 2*0 + 10 = 10
#   phi = A . B  (last-listed innermost)   -> phi(0) = 2*(0 + 10) = 20
#
# The predictions differ in *magnitude* (10 vs 20), so a sign flip cannot turn
# one into the other.
_SCALE_2X = 'Parameters: 2 0 0 0 2 0 0 0 2 0 0 0'
_TRANSLATE_10 = 'Parameters: 1 0 0 0 1 0 0 0 1 10 0 0'


def _write_itk_affine(path, parameters_line):
    """Write a 3D ITK affine transform in the text format ANTs reads."""
    path.write_text(
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: MatrixOffsetTransformBase_double_3_3\n'
        f'{parameters_line}\n'
        'FixedParameters: 0 0 0\n'
    )
    return str(path)


def test_first_listed_transform_is_applied_first_to_the_point(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')

    reference = tmp_path / 'ref.nii.gz'
    nb.Nifti1Image(np.zeros((8, 8, 8), dtype='float32'), np.eye(4)).to_filename(str(reference))
    scale = _write_itk_affine(tmp_path / 'scale.txt', _SCALE_2X)
    translate = _write_itk_affine(tmp_path / 'translate.txt', _TRANSLATE_10)

    composite = tmp_path / 'composite.nii.gz'
    xfm = ants.ApplyTransforms(
        input_image=str(reference),
        reference_image=str(reference),
        transforms=[scale, translate],
        output_image=str(composite),
        print_out_composite_warp_file=True,
        interpolation='LanczosWindowedSinc',
        dimension=3,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    xfm.run()

    # The composite warp stores phi(x) - x at each reference voxel. Voxel
    # (0, 0, 0) is world origin under the identity affine, so the stored
    # displacement is phi(0) itself.
    field = np.asanyarray(nb.load(str(composite)).dataobj)
    displacement = field.reshape(field.shape[:3] + (3,))[0, 0, 0]
    magnitude = np.abs(displacement).max()

    # Separate "the other convention" from "something else entirely" (an
    # affine collapse, a zero field, a units problem). Only 10 and 20 are
    # convention answers; anything else means the experiment itself is wrong
    # and neither party's prediction has been tested.
    assert magnitude == pytest.approx(10.0, abs=0.5) or magnitude == pytest.approx(
        20.0, abs=0.5
    ), (
        f'Composite displacement at the origin was {magnitude:.3f}mm, which is '
        'neither 10mm nor 20mm. This test has not discriminated anything -- the '
        'experiment is broken (collapsed affines, an empty field, or a units '
        'mismatch), not the convention. Investigate before reading anything '
        'into it.'
    )

    assert magnitude == pytest.approx(10.0, abs=0.5), (
        'transforms=[scale, translate] produced a displacement of '
        f'{magnitude:.3f}mm. 10mm means phi = translate . scale, i.e. the '
        'FIRST-listed transform is innermost (applied first to the point), '
        'which is what ComposeJacobianWeights assumes. 20mm means the opposite '
        'convention: STOP, and rework the composition to include the '
        'per-volume HMC affine so the gradwarp and SDC determinants are '
        'evaluated at the right coordinates. Do not work around it. See '
        '"Disambiguating \'applied first\'" in the design spec, and note that '
        "under the 20mm reading QSIPrep's existing resampling would be "
        'sampling native per-volume data at b=0-reference coordinates, so '
        'check that too.'
    )
