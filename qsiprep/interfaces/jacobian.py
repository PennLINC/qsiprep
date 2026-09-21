"""Jacobian intensity modulation for QSIPrep's spatial distortion corrections.

The weight applied to the resampled DWIs is ``|det grad phi|`` of the composed
*native-space* distortion warps -- gradient nonlinearity, susceptibility, and
TORTOISE eddy current -- and of nothing else. See
``docs/superpowers/specs/2026-09-17-jacobian-weighting-design.md`` for why head
motion, coregistration and the intramodal/template warps are excluded, and for
the derivation showing that excluding them does not move the coordinates at
which the remaining determinants are evaluated.

The eddy-current and gradwarp/SDC factors that *do* remain are evaluated in
two different coordinate domains -- TORTOISE's eddy-current Jacobian on
DIFFPREP's distorted-native grid (see ``diffprep.py:578``'s
``extract_b0s.b0_average``), the gradwarp/SDC determinant in undistorted
b0-reference space -- so combining them by a plain per-voxel product would be
evaluating a native-domain function at reference-space coordinates. C1 fix:
``ComposeJacobianWeights`` transports the eddy-current Jacobian through the
same composed gradwarp/SDC warp before multiplying (``transport_scalar_map``),
except in the one case where no such warp exists at all (no gradwarp, no SDC),
where the eddy-current Jacobian is already in the run's sole coordinate
domain.

TORTOISE eddy-current Jacobian: sourced formula
------------------------------------------------

The Okan quadratic coordinate-map is not documented anywhere in this
repository, and TORTOISE's public documentation
(https://tortoisedti.nichd.nih.gov/transformation-files.html) covers only the
older 14-column (v3) format. The 24-column (V4) layout implemented here was
sourced directly from TORTOISE's own C++ source
(https://github.com/rordenlab/TORTOISEV4, Apache-2.0 -- the source tree the
container's ``/src/TORTOISEV4`` binaries are built from):

* ``src/main/itkOkanQuadraticTransform.h`` -- ``NQUADPARAMS = 24``.
* ``src/main/itkOkanQuadraticTransform.hxx`` -- ``TransformPoint`` (~line 155),
  ``ComputeMatrix`` (~line 421), ``ComputeJacobianWithRespectToPosition``
  (~line 583), and the read-back constructor
  ``OkanQuadraticTransform(const ParametersType params)`` (~line 74) that
  infers the phase axis from columns 6-8 when no other metadata is available.
* ``src/main/DIFFPREP.cxx`` -- ``WriteOutputFiles`` (~line 1828: the identity
  row and per-volume ``GetParameters()`` serialization to
  ``_moteddy_transformations.txt``), ``ChangeImageHeaderToDP`` (~line 267:
  the coordinate frame), the ``correction_mode != "off"`` resampling loop
  (~line 2147: confirms the transform is used by
  ``itk::ResampleImageFilter`` as an output-space -> input-space map, matching
  qsiprep's non-slice-to-volume, non-outlier-replacement configuration), and
  ``PE_string`` (~line 58: phase axis from the BIDS ``PhaseEncodingDirection``
  at *write* time).
* ``settings/mecc_settings/*.mec`` -- the 1-indexed column commentary matches
  the 0-indexed source exactly (off by one), and confirms that qsiprep's
  correction modes (``quadratic.mec``, ``cubic.mec``, ``rigid.mec`` -- the
  non-``_isoc`` presets) never optimize the eddy centre: columns 21-23 stay at
  their initial value of 0 for every qsiprep run.

**Parameter layout** (0-indexed, 24 columns per DWI volume, one row per
volume in acquisition order)::

    0-2    rigid translation x, y, z (mm)
    3-5    Euler rotation angles theta_x, theta_y, theta_z (radians);
           R = Rz . Ry . Rx, composed in exactly that order (ComputeMatrix)
    6-8    linear coefficients of the eddy-current phase-axis polynomial,
           along x, y, z respectively. At identity, the coefficient for the
           volume's own phase-encode axis is 1 and the other two are 0.
    9-11   quadratic cross terms xy, xz, yz
    12-13  quadratic terms (x^2 - y^2) and (2z^2 - x^2 - y^2)
    14-20  cubic terms, active only when correction_mode == 'cubic'
    21-23  eddy-current/rotation centre x, y, z (mm); always 0 for qsiprep
           (which never requests a "_isoc" .mec preset)

**The map** (``TransformPoint``; forward, output-space -> input-space, i.e.
exactly what ``itk::ResampleImageFilter`` needs -- no inversion)::

    p -= center                # columns 21-23
    p  = R @ p + T             # columns 3-5, 0-2 ("rigid" part)
    new_phase = c6*p.x + c7*p.y + c8*p.z
              + c9*p.x*p.y + c10*p.x*p.z + c11*p.y*p.z
              + c12*(p.x**2 - p.y**2) + c13*(2*p.z**2 - p.x**2 - p.y**2)
    p[phase] = new_phase       # ASSIGNMENT, not addition: only the
                                # phase-encode axis is replaced -- eddy
                                # currents distort that axis alone
    p += center

``phase`` in ``{0, 1, 2}`` ("Read"/"Phase"/"Slice", i.e. the array axis, not a
scanner axis) comes from the DWI's ``PhaseEncodingDirection`` ("i"->0,
"j"->1, "k"->2) when TORTOISE *writes* the file. Reading it back (this module
has no other source for it) reproduces TORTOISE's own heuristic: compare
columns 6, 7, 8 and take the largest, since identity leaves that column at 1
and the other two at 0 (see ``_okan_phase_axis``).

**Coordinate frame.** ``TransformPoint`` does not operate in the DWI's own
oblique scanner-physical space. ``ChangeImageHeaderToDP`` rewrites the image
header to identity direction cosines before motion+eddy estimation and
resampling, re-centring the origin so a continuous (generally non-integer)
voxel index ``indo`` lands at physical ``(0, 0, 0)`` -- the scanner isocenter,
for qsiprep's default ``rot_eddy_center="isocenter"``. Physical coordinates in
this "DP frame" are therefore ``spacing * (index - indo)``, with **no
rotation applied**: TORTOISE operates directly on the image's own
voxel-aligned Read/Phase/Slice axes. ``indo`` still depends on the original
image's true (possibly oblique) affine -- see ``_okan_coordinate_frame``.
Because the DP frame is a fixed per-image *rotation* of physical space (by
the image's own direction cosines), a Jacobian determinant computed in the DP
frame equals the one computed in true LPS/RAS space (determinants are
invariant under a shared orthogonal change of domain and codomain
coordinates); the *absolute* coordinates fed into the polynomial still need
the DP frame's centring to come out right, which is why
``_okan_coordinate_frame`` reproduces it rather than assuming the array's
geometric centre.

**EC-only Jacobian.** ``okan_quadratic_jacobian`` uses only columns 6-23,
treating the rigid part (0-5) as identity (``R = I``, ``T = 0``): head motion
is excluded from Jacobian weighting by policy (see the design spec). Because
the map replaces only ``p[phase]`` and leaves the other two coordinates
untouched, its Jacobian matrix has two rows equal to elementary basis
vectors, so ``det = d(new_phase)/d(p[phase])`` -- the partial derivative of
the polynomial with respect to its own axis, holding the other two fixed
(this matches the ``do_cubic == False`` branch of
``ComputeJacobianWithRespectToPosition``, restricted to the ``R = I`` case).

**Cubic is out of scope.** ``correction_mode == 'cubic'`` activates columns
14-20 (``do_cubic``); this module does not implement that determinant.
``OkanQuadraticJacobian`` returns ``Undefined`` for it rather than either
raising (which would break otherwise-working runs) or silently applying the
quadratic-only formula to cubic parameters (which would under-count the
warp's volume change) -- see the mode-gating tests in
``test_interfaces_jacobian.py``.
"""

import itertools
import os

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype import logging
from nipype.interfaces import ants
from nipype.utils.filemanip import fname_presuffix

from .tortoise import _read_okan_transformations

LOGGER = logging.getLogger('nipype.interface')

#: NIfTI intent code ITK's NiftiImageIO uses to recognize a 5D image as a
#: displacement field. Without it, ``CreateJacobianDeterminantImage`` still
#: runs, but silently miscomputes cross-derivatives of the first vector
#: component -- see ``jacobian_determinant``.
VECTOR_INTENT_CODE = 1007

#: Tolerances for comparing a displacement field's affine against the
#: composition reference. Loose enough for float32 header round-trips, tight
#: enough that a different lattice or orientation fails.
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4

#: Floor for a resampled weight map. Lanczos ringing can push a positive
#: scalar map slightly negative; a weight can be small but never zero or
#: negative.
WEIGHT_FLOOR = 1e-3

#: In-mask median outside this range is a smell, not an error: a correct
#: distortion field redistributes signal without changing its total, so the
#: median determinant should sit near unity.
MEDIAN_WARN_RANGE = (0.9, 1.1)


def weight_key(gradwarp, fieldwarp):
    """Cache key for one unique (gradwarp, fieldwarp) combination.

    Falsy when neither field is present, which means a unity weight rather
    than a cache entry.
    """
    return tuple(path for path in (gradwarp, fieldwarp) if path)


#: How far apart (mm, along any one axis) two images' world-frame bounding
#: boxes may sit and still be considered "the same coordinate domain". A
#: small negative slack tolerates float32 header round-trips leaving the
#: boxes just touching rather than cleanly overlapping; it is not big enough
#: to paper over a genuinely different subject or space.
WORLD_OVERLAP_SLACK_MM = 1e-2


def _world_bounding_box(img):
    """The axis-aligned world-space bounding box of ``img``'s voxel grid."""
    shape = img.shape[:3]
    corners = np.array(list(itertools.product(*[(0, dim - 1) for dim in shape])))
    world = nb.affines.apply_affine(img.affine, corners)
    return world.min(axis=0), world.max(axis=0)


def _assert_world_frames_overlap(path_a, path_b, img_a=None, img_b=None):
    """Raise unless ``path_a`` and ``path_b`` occupy overlapping world space.

    This is the one thing ANTs' physical-space composition and resampling
    cannot rescue: an input in a completely unrelated coordinate frame (wrong
    subject, wrong units, a header bug) produces a plausible-looking but
    meaningless result rather than an error. A different sampling lattice --
    different shape, different voxel size, different origin from padding or
    cropping -- is not an error; ANTs resamples through each input's own
    affine, so lattice disagreement alone is not evidence of anything wrong.
    """
    img_a = img_a if img_a is not None else nb.load(path_a)
    img_b = img_b if img_b is not None else nb.load(path_b)
    min_a, max_a = _world_bounding_box(img_a)
    min_b, max_b = _world_bounding_box(img_b)
    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    if np.any(overlap_max - overlap_min < -WORLD_OVERLAP_SLACK_MM):
        raise ValueError(
            f'{path_a} and {path_b} do not occupy overlapping world space '
            f'({(min_a, max_a)} vs {(min_b, max_b)}). A different sampling '
            'lattice is fine -- ANTs composes and resamples in physical '
            'coordinates -- but these are not in a compatible coordinate '
            'domain at all.'
        )


def resample_like(source_path, like_path, out_path, interpolation='nearest', fill_value=0):
    """Resample ``source_path`` onto ``like_path``'s grid, if it is not already.

    A no-op that returns ``source_path`` unchanged when the two already share
    a lattice, so the common (already-matching) case does no I/O and stays
    byte-identical. Used to make a mask or scalar map meaningful against a
    weight map or reference that legitimately lives on a different grid (e.g.
    a native-space mask against a DRBUDDI-output-grid reference) -- a
    mislatticed input must not silently move which voxels a downstream check
    inspects, so it is resampled onto the grid actually being inspected rather
    than compared in place or ignored.

    ``fill_value`` is what appears outside ``source_path``'s own field of
    view once resampled onto ``like_path``'s (generally larger) grid. It
    defaults to 0, the correct value for a mask (a mask is absent, i.e.
    false, outside its own FOV) -- see ``multiply_maps`` for why a
    determinant factor needs a different default (C1 fix notes).

    ``force_resample=True`` is passed to ``nilearn.image.resample_to_img``
    unconditionally, on every code path that actually resamples (never on the
    no-op early return above, which is ours, not nilearn's, and never touches
    nilearn at all). Without it, nilearn silently ignores ``fill_value``
    whenever the two grids differ only by an axis-aligned, whole-voxel
    translation -- e.g. exactly DRBUDDI's own padding of a native grid -- by
    taking an internal "padding" fast path that fills the non-overlapping
    region with hard zeros regardless of ``fill_value`` (see
    ``nilearn.image.resampling.resample_img``, the ``not force_resample``
    branch that special-cases ``A == I`` and integer ``b``). Whether that fast
    path triggers is sensitive to floating-point noise in the affine
    inversion, so relying on it *not* triggering is not a fix -- only
    ``force_resample=True`` reliably disables it.
    """
    source = nb.load(source_path)
    like = nb.load(like_path)
    if source.shape[:3] == like.shape[:3] and np.allclose(
        source.affine, like.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL
    ):
        return source_path
    resampled = nim.resample_to_img(
        source,
        like,
        interpolation=interpolation,
        fill_value=fill_value,
        force_resample=True,
    )
    resampled.to_filename(out_path)
    return out_path


def validate_field_geometry(field_path, reference_path):
    """Raise unless ``field_path`` is a plausible displacement field for composition
    against ``reference_path``.

    Composition and resampling both happen in ANTs, in physical (world)
    coordinates, so ``field_path`` does not need to share ``reference_path``'s
    sampling lattice -- only its coordinate domain. This used to require exact
    shape-and-affine agreement; that was too strict; see the module's C1 fix
    notes. What is still checked is what ANTs cannot rescue: a field with the
    wrong number of vector components, or one in a world frame that does not
    even overlap the reference (see ``_assert_world_frames_overlap``).

    This is a cheap structural screen. It cannot prove that a field encodes the
    intended coordinate domain, and it cannot detect an inverted field -- an
    inverse has identical headers. Direction is established behaviourally, by
    the conservation oracle and the positivity guard.

    This does not check the NIfTI vector intent code; that normalization
    happens in ``jacobian_determinant`` immediately before the ANTs shellout
    that depends on it, not here.
    """
    field = nb.load(field_path)
    reference = nb.load(reference_path)

    # Only the two real displacement-field layouts are accepted: 4D
    # (X, Y, Z, 3) or 5D (X, Y, Z, 1, 3) -- ITK's own convention for a vector
    # image. Reading ``field.shape[-1]`` as a component count for any
    # non-5D image (the previous check) let a plain 3D scalar image whose
    # last spatial dimension happened to equal 3 pass as a displacement
    # field, so anything else is rejected outright, by shape, before a
    # component count is even considered.
    shape = field.shape
    if field.ndim == 4:
        components = shape[3]
    elif field.ndim == 5 and shape[3] == 1:
        components = shape[4]
    else:
        raise ValueError(
            f'Displacement field {field_path} has shape {shape}, expected a 4D '
            'image shaped (X, Y, Z, 3) or a 5D image shaped (X, Y, Z, 1, 3).'
        )

    if components != 3:
        raise ValueError(
            f'Displacement field {field_path} has {components} vector components, '
            'expected 3.'
        )

    _assert_world_frames_overlap(field_path, reference_path, img_a=field, img_b=reference)


def check_weight_map(map_path, mask_path):
    """Raise on a non-finite or non-positive in-mask weight; warn if off-unity.

    A folded warp produces a non-positive determinant, and silently multiplying
    DWI data by a negative number is worse than failing.
    """
    weights = np.asanyarray(nb.load(map_path).dataobj)
    mask = np.asanyarray(nb.load(mask_path).dataobj) > 0

    if not np.isfinite(weights).all():
        raise ValueError(
            f'Jacobian weight map {map_path} contains non-finite values.'
        )

    inside = weights[mask]
    if inside.size and inside.min() <= 0:
        raise ValueError(
            f'Jacobian weight map {map_path} has non-positive values inside the '
            f'brain mask (minimum {inside.min():.4f}). This means the composed '
            'warp folds. Applying it would multiply DWI signal by a negative '
            'number.'
        )

    if inside.size:
        median = float(np.median(inside))
        if not MEDIAN_WARN_RANGE[0] <= median <= MEDIAN_WARN_RANGE[1]:
            LOGGER.warning(
                'Jacobian weight map %s has an in-mask median of %.4f, outside '
                '%s. A correct distortion field redistributes signal without '
                'changing its total, so this may indicate a wrong field, an '
                'inverted field, or a wrong composition order.',
                map_path,
                median,
                MEDIAN_WARN_RANGE,
            )


def multiply_maps(paths, out_path, like_path=None):
    """Voxelwise product of one or more scalar maps, reconciled onto one lattice.

    ``nilearn.image.math_img`` requires exact voxel correspondence between its
    inputs and raises ``ValueError`` when their shapes or affines differ. That
    is reachable in production: with ``--hmc-method tortoise --sdc-method
    drbuddi`` and the default ``correction_mode='quadratic'``, the SDC
    determinant is evaluated on whatever grid its warp arrived on (DRBUDDI's
    own output grid) while a lone-field determinant (or, in the no-gradwarp-
    no-SDC case, the EC Jacobian used directly) can be on DIFFPREP's native
    input grid -- two genuinely different *lattices* for the same physical
    head, but still the same coordinate domain: resampling here is lattice
    realignment in a shared world frame, not a coordinate-domain change.

    This is deliberately *not* the tool that relates the EC Jacobian's own
    coordinate domain to the SDC/gradwarp determinant's -- those two are
    evaluated in genuinely different domains (DIFFPREP's distorted-native
    space vs. undistorted b0-reference space), related only by the composed
    gradwarp/SDC warp itself, not by an affine lattice realignment. That
    reconciliation happens once, in ``transport_scalar_map``, before an EC
    Jacobian and a composed-warp determinant ever reach this function
    together -- see the module docstring's C1 fix notes and
    ``ComposeJacobianWeights._run_interface``.

    ``like_path`` is the grid every factor is resampled onto (via
    ``resample_like``, with linear interpolation -- these are continuous
    determinant maps, not masks, so ``resample_like``'s nearest-neighbour
    default would introduce blocky discontinuities into a smoothly varying
    weight) before multiplying. It defaults to ``paths[0]``, so a lone factor
    already on that grid -- the common case -- costs no extra I/O, since
    ``resample_like`` itself no-ops when the grids already match.

    Every factor is also resampled with ``fill_value=1.0`` (C1 fix): a
    determinant factor undefined outside its own field of view means "no
    volume change known here", whose multiplicative identity is 1.0, not
    nilearn's default of 0. A gradwarp field is shared by every volume and
    typically already spans the whole reference grid, but an EC Jacobian is
    evaluated on DIFFPREP's native grid, which is routinely smaller than
    DRBUDDI's padded output grid -- zero-filling that gap would multiply real
    DWI signal in the padded band by (after ``_floor_nonpositive_weights``'s
    floor) roughly 1/1000.

    ``ComposeJacobianWeights`` passes its own ``b0_ref_image`` explicitly:
    that input's own docstring already commits to being "the lattice the
    weight maps live on", and every factor reaching this function has already
    been validated to at least share its world frame (``validate_field_
    geometry`` / ``validate_scalar_geometry``), which is what makes resampling
    onto it safe rather than an incidental fix.
    """
    target = like_path if like_path is not None else paths[0]
    onlattice = [
        resample_like(
            path,
            target,
            fname_presuffix(
                path, suffix='_onlattice', newpath=os.path.dirname(out_path) or None
            ),
            interpolation='linear',
            fill_value=1.0,
        )
        for path in paths
    ]
    if len(onlattice) == 1:
        return onlattice[0]
    product = nim.load_img(onlattice[0])
    for path in onlattice[1:]:
        product = nim.math_img('a*b', a=product, b=path)
    product.to_filename(out_path)
    return out_path


def compose_fields(field_paths, reference, out_path):
    """Compose displacement fields into one, on ``reference``'s grid.

    ``field_paths`` is given in QSIPrep's native-to-target chain order; it is
    reversed here for ANTs, matching ``ComposeTransforms``. The reference is a
    *native* lattice: the composite's domain is undistorted b=0-reference
    space, and materialising it on the output grid would evaluate a
    native-domain function at output coordinates.
    """
    xfm = ants.ApplyTransforms(
        # input_image is ignored because print_out_composite_warp_file is True
        input_image=reference,
        reference_image=reference,
        transforms=list(field_paths)[::-1],
        output_image=out_path,
        print_out_composite_warp_file=True,
        interpolation='LanczosWindowedSinc',
        dimension=3,
        float=True,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime
    LOGGER.info(runtime.cmdline)
    return out_path


def transport_scalar_map(image_path, transform_path, reference_path, out_path):
    """Resample a scalar map through ``transform_path`` onto ``reference_path``'s
    grid -- coordinate transport, not lattice realignment (C1 fix).

    ``image_path`` is a scalar field defined over one coordinate domain (e.g.
    the eddy-current Jacobian, evaluated on DIFFPREP's distorted-native grid);
    ``transform_path`` is the same warp/composite already used to relate that
    domain to ``reference_path``'s (e.g. the gradwarp/SDC composite that
    ``jacobian_determinant`` is evaluated from). Applying it here means: for
    each point ``v`` of ``reference_path``'s grid, sample ``image_path`` at
    ``transform_path``'s corresponding native-space point -- exactly what
    ``antsApplyTransforms`` does for any other image, and exactly what
    ``multiply_maps``'s ``resample_like``-based reconciliation does *not* do
    (that only realigns two images already agreed to share one coordinate
    domain onto one lattice; see its docstring).

    ``interpolation='Linear'`` because this is a smooth, continuous
    determinant map, not a mask (nearest-neighbour would introduce blocky
    discontinuities). ``default_value=1.0`` keeps the same "no volume change
    known here" convention as ``multiply_maps``'s ``fill_value=1.0``: outside
    the transported footprint (e.g. DIFFPREP's native grid is routinely
    smaller than a DRBUDDI-padded output grid), the multiplicative identity is
    1.0, not ANTs' own default of 0.
    """
    xfm = ants.ApplyTransforms(
        input_image=image_path,
        reference_image=reference_path,
        transforms=[transform_path],
        output_image=out_path,
        interpolation='Linear',
        default_value=1.0,
        dimension=3,
        float=True,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime
    LOGGER.info(runtime.cmdline)
    return out_path


def jacobian_determinant(field_path, out_path, mask_path=None):
    """``|det grad phi|`` of a displacement field, on the field's own grid.

    ``CreateJacobianDeterminantImage`` takes no reference image; it emits on
    the deformation field's grid. ``doLogJacobian=0`` because the weight is
    multiplicative, and ``useGeometric=0`` for the plain determinant.

    A folded warp produces a *negative* determinant, so the sign is inspected
    before any absolute value is taken -- otherwise a fold at -0.5 would become
    an innocuous-looking weight of 0.5 and no later check could recover it.
    Pass ``mask_path`` to get that check; without it the sign is only logged.

    A field lacking the ``NIFTI_INTENT_VECTOR`` (1007) intent code is silently
    miscomputed by ``CreateJacobianDeterminantImage``: without it, ANTs treats
    the file as a generic vector array rather than a displacement field, and
    cross-derivatives of the first vector component are folded into the
    diagonal, giving a determinant that is positive, plausible, and wrong
    (e.g. a pure shear with det = 1 measured back at 1.3 for one hand-authored
    field, 0.7 for another -- the exact wrong value depends on the field's own
    sign convention, which is the point: it is never flagged as wrong). The
    positivity guard cannot catch this because the number looks fine. A
    displacement field reaching this function is a displacement field by
    contract (the ``gradwarp_field``/``fieldwarps``
    slots it comes from admit nothing else), so the intent code is normalized
    here rather than validated-and-rejected: a producer such as
    ``MaskWarpDimensions`` forwards its input header verbatim and never sets
    it, and a user-supplied ``--gradient-file`` field is never checked either.
    The caller's file is never mutated in place; a corrected copy is written
    to scratch and ANTs runs against that copy instead.
    """
    intent_code = nb.load(field_path).header.get_intent('code')[0]
    if intent_code != VECTOR_INTENT_CODE:
        corrected_path = fname_presuffix(
            field_path, suffix='_vecintent', newpath=os.path.dirname(out_path) or None
        )
        img = nb.load(field_path)
        header = img.header.copy()
        header.set_intent(VECTOR_INTENT_CODE)
        nb.Nifti1Image(np.asanyarray(img.dataobj), img.affine, header).to_filename(
            corrected_path
        )
        LOGGER.warning(
            'Displacement field %s is missing the NIFTI_INTENT_VECTOR intent '
            'code; CreateJacobianDeterminantImage silently miscomputes '
            'cross-derivatives of the first component without it. Normalized '
            'a copy at %s and using that instead.',
            field_path,
            corrected_path,
        )
        field_path = corrected_path

    jac = ants.CreateJacobianDeterminantImage(
        imageDimension=3,
        deformationField=field_path,
        outputImage=out_path,
        doLogJacobian=0,
        useGeometric=0,
    )
    jac.terminal_output = 'allatonce'
    jac.resource_monitor = False
    runtime = jac.run().runtime
    LOGGER.info(runtime.cmdline)

    img = nb.load(out_path)
    signed = np.asanyarray(img.dataobj)

    # Inspect the SIGN before taking any absolute value. In-mask negatives mean
    # the composed warp folds, which is a hard error; outside the mask,
    # CreateJacobianDeterminantImage legitimately emits small negatives at the
    # field's edge from one-sided differences, and those are harmless.
    if mask_path is not None:
        mask = np.asanyarray(nb.load(mask_path).dataobj) > 0
        inside = signed[mask]
        if inside.size and inside.min() <= 0:
            raise ValueError(
                f'Jacobian determinant of {field_path} is non-positive inside '
                f'the brain mask (minimum {inside.min():.4f}), i.e. the composed '
                'warp folds there. Refusing to build a weight map from it.'
            )
    elif np.nanmin(signed) <= 0:
        LOGGER.warning(
            'Jacobian determinant of %s contains non-positive values and no '
            'mask was supplied to localise them. Edge negatives are expected; '
            'interior ones are a fold.',
            field_path,
        )

    nb.Nifti1Image(
        np.abs(signed).astype('float32'), img.affine, img.header
    ).to_filename(out_path)
    return out_path


def validate_scalar_geometry(image_path, reference_path):
    """Raise unless a scalar map is a plausible input for this composition.

    ``validate_field_geometry`` covers displacement fields; this covers scalar
    inputs -- the eddy-current Jacobians and the brain mask. It used to require
    an identical sampling lattice against ``reference_path``, which is wrong
    for the same reason ``validate_field_geometry``'s equivalent requirement
    was: a native-space mask against a DRBUDDI-output-grid reference is a
    legitimate, differently-sampled pairing, not an error (see the module's C1
    fix notes). What is still checked is a genuinely wrong input -- something
    that is not a 3D scalar map at all, or sits in a world frame that does not
    even overlap the reference.

    Callers that actually combine a scalar map against something on a
    different grid (the positivity guard's mask, in particular) are
    responsible for resampling it there first with ``resample_like``; this
    function only screens for gross mistakes, it does not make two grids
    compatible.
    """
    image = nb.load(image_path)
    reference = nb.load(reference_path)

    extra_dims = image.shape[3:]
    if extra_dims and any(dim != 1 for dim in extra_dims):
        raise ValueError(
            f'{image_path} is not a 3D scalar map (shape {image.shape}).'
        )

    _assert_world_frames_overlap(image_path, reference_path, img_a=image, img_b=reference)


def _abspath(path, cwd):
    return path if os.path.isabs(path) else os.path.join(cwd, path)


from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _ComposeJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='split DWI volumes, in their native grid; supplies the volume count',
    )
    b0_ref_image = File(
        exists=True,
        mandatory=True,
        desc='undistorted b=0 reference; the lattice the weight maps live on and '
        'the reference for composing two fields',
    )
    # NOTE for the implementer: only this image's *grid* is used -- as the
    # ``-r`` reference for composition and as the geometry the input fields are
    # validated against. Its voxel content is never read. That matters because
    # on the DRBUDDI-with-T2w path ``b0_ref_image`` is the *structural* image
    # rather than a b=0 (``DRBUDDIAggregateOutputs`` returns
    # ``structural_image`` as ``b0_ref`` when one exists,
    # ``qsiprep/interfaces/tortoise.py:503-507``). That is still correct here:
    # ``init_structural_to_b0_alignment_wf`` resamples the T2w with ``b0_ref``
    # as its ``reference_image`` (``qsiprep/workflows/dwi/registration.py:157``),
    # so the structural is on the b=0 lattice. Do not "fix" this by reaching for
    # a different input.
    mask = File(
        exists=True,
        mandatory=True,
        desc='native-space brain mask, for the positivity guard',
    )
    gradwarp_field = InputMultiObject(
        File(exists=True),
        desc='gradient nonlinearity displacement field (one, shared by every volume)',
    )
    fieldwarps = InputMultiObject(
        File(exists=True),
        desc='SDC displacement field(s): one shared, or one per DWI volume',
    )
    ec_jacobian_images = InputMultiObject(
        File(exists=True),
        desc='per-volume eddy-current Jacobian determinants (TORTOISE DIFFPREP)',
    )


class _ComposeJacobianWeightsOutputSpec(TraitedSpec):
    jacobian_weight_images = OutputMultiObject(
        File(exists=True),
        desc='one weight map per DWI volume, with repeats where volumes share one',
    )


class ComposeJacobianWeights(SimpleInterface):
    """Build per-volume Jacobian weight maps for the native distortion warps.

    The weight for a volume is ``|det grad(gradwarp . fieldwarp)|`` evaluated in
    undistorted b=0-reference space, times that volume's eddy-current Jacobian
    when one exists -- the eddy-current factor evaluated not at the same
    b0-reference point, but at the native-space point that same composed
    gradwarp/SDC warp maps it to (C1 fix; see ``transport_scalar_map``). When
    there is no gradwarp and no SDC warp, the eddy-current Jacobian is already
    in the run's one and only coordinate domain, so no transport is applied.
    Head motion, coregistration and the intramodal/template warps are excluded
    by policy; see the design spec.

    Unique ``(gradwarp, fieldwarp)`` combinations are computed once and shared,
    so a run with one gradwarp field and one SDC warp costs a single ANTs call
    even with hundreds of volumes.
    """

    input_spec = _ComposeJacobianWeightsInputSpec
    output_spec = _ComposeJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        num_dwis = len(self.inputs.dwi_files)
        reference = self.inputs.b0_ref_image

        gradwarp = None
        if isdefined(self.inputs.gradwarp_field) and self.inputs.gradwarp_field:
            if len(self.inputs.gradwarp_field) != 1:
                raise ValueError(
                    'Expected a single gradwarp field, got '
                    f'{len(self.inputs.gradwarp_field)}.'
                )
            gradwarp = self.inputs.gradwarp_field[0]
            validate_field_geometry(gradwarp, reference)

        fieldwarps = [None] * num_dwis
        if isdefined(self.inputs.fieldwarps) and self.inputs.fieldwarps:
            supplied = list(self.inputs.fieldwarps)
            if len(supplied) == 1:
                LOGGER.info('Using a single SDC warp for all DWI volumes')
                fieldwarps = supplied * num_dwis
            elif len(supplied) == num_dwis:
                LOGGER.info('Using per-volume SDC warps')
                fieldwarps = supplied
            else:
                raise ValueError(
                    f'Got {len(supplied)} SDC warps for {num_dwis} DWI volumes; '
                    'expected 1 or one per volume.'
                )
            for warp in set(fieldwarps):
                validate_field_geometry(warp, reference)

        ec_images = [None] * num_dwis
        if isdefined(self.inputs.ec_jacobian_images) and self.inputs.ec_jacobian_images:
            supplied = list(self.inputs.ec_jacobian_images)
            if len(supplied) != num_dwis:
                raise ValueError(
                    f'Got {len(supplied)} eddy-current Jacobians for {num_dwis} '
                    'DWI volumes; expected one per volume.'
                )
            for ec_image in set(supplied):
                validate_scalar_geometry(ec_image, reference)
            ec_images = supplied

        if gradwarp is None and not any(fieldwarps) and not any(ec_images):
            LOGGER.info('No distortion transforms to modulate; no weights produced')
            return runtime

        # The mask is only needed from here on (the positivity guard, on
        # whatever grid each determinant/weight map ends up on). Validated and
        # used only past the no-op return above (C2): a run with nothing to
        # modulate must not be killed by a mask/reference mismatch it never
        # needed to resolve.
        validate_scalar_geometry(self.inputs.mask, reference)

        # One determinant per unique (gradwarp, fieldwarp) pair. `composed`
        # (the gradwarp/SDC warp itself, not just its determinant) is kept
        # too: it is also the transform that relates the EC Jacobian's own
        # coordinate domain (DIFFPREP's distorted-native grid) to
        # `reference`'s (C1 fix -- see `transport_scalar_map`).
        determinants = {}
        composed_transforms = {}
        for fieldwarp in dict.fromkeys(fieldwarps):
            key = weight_key(gradwarp, fieldwarp)
            if not key or key in determinants:
                continue
            fields = [path for path in (gradwarp, fieldwarp) if path]
            if len(fields) == 2:
                composed = compose_fields(
                    fields,
                    reference,
                    os.path.join(runtime.cwd, f'composite{len(determinants)}.nii.gz'),
                )
            else:
                composed = fields[0]
            composed_transforms[key] = composed
            # The determinant is emitted on `composed`'s own grid (see
            # jacobian_determinant), which is not always `reference`'s grid
            # (e.g. a lone SDC warp already on its own output grid). Resample
            # the mask there rather than assume it lines up, so a mislatticed
            # mask cannot silently move which voxels the fold check inspects.
            mask_for_determinant = resample_like(
                self.inputs.mask,
                composed,
                os.path.join(runtime.cwd, f'mask_for_jacobian{len(determinants)}.nii.gz'),
            )
            determinants[key] = jacobian_determinant(
                composed,
                os.path.join(runtime.cwd, f'jacobian{len(determinants)}.nii.gz'),
                mask_path=mask_for_determinant,
            )

        # Per-volume weight = shared determinant x that volume's EC Jacobian,
        # the latter transported through the same composed warp (C1 fix): the
        # EC Jacobian lives on DIFFPREP's distorted-native grid, while the
        # determinant above is evaluated in `reference`'s undistorted space,
        # so `factors` must never combine a raw `ec_image` with a `key`
        # (a composed gradwarp/SDC warp) -- see `transport_scalar_map` and the
        # module docstring's C1 fix notes. Only when there is truly no
        # gradwarp and no SDC warp (`key` falsy) is `ec_image` already in the
        # right space, and transporting it through a nonexistent transform
        # would be both wrong (no reference-relating transform to use) and
        # pointless (nothing to reconcile).
        weights = []
        cache = {}
        for index, (fieldwarp, ec_image) in enumerate(zip(fieldwarps, ec_images, strict=True)):
            factors = []
            key = weight_key(gradwarp, fieldwarp)
            if key:
                factors.append(determinants[key])

            # Tag the roles rather than collapsing missing factors into an
            # untagged tuple: (None, 'f.nii.gz') and ('f.nii.gz', None) are
            # different weights that would otherwise share a key.
            cache_key = (gradwarp, fieldwarp, ec_image)
            if cache_key not in cache:
                ec_factor = ec_image
                if ec_image and key:
                    ec_factor = transport_scalar_map(
                        ec_image,
                        composed_transforms[key],
                        reference,
                        fname_presuffix(
                            ec_image,
                            suffix=f'_ectransport-{len(cache):05d}',
                            newpath=runtime.cwd,
                            use_ext=True,
                        ),
                    )
                if ec_factor:
                    factors.append(ec_factor)
                weight_map = multiply_maps(
                    factors,
                    fname_presuffix(
                        self.inputs.dwi_files[index],
                        suffix=f'_jacobian-{index:05d}',
                        newpath=runtime.cwd,
                        use_ext=True,
                    ),
                    # Reconcile onto `reference`'s grid -- see multiply_maps'
                    # docstring for why that is the deliberate, documented
                    # choice rather than an incidental one.
                    like_path=reference,
                )
                cache[cache_key] = weight_map
                # Same reasoning as mask_for_determinant above: the weight map
                # is on whatever grid its factors are on, not necessarily
                # `reference`'s. This branch runs once per unique weight map,
                # so no further dedup is needed here.
                mask_for_weight = resample_like(
                    self.inputs.mask,
                    weight_map,
                    fname_presuffix(
                        self.inputs.mask,
                        suffix=f'_mask-{len(cache):05d}',
                        newpath=runtime.cwd,
                        use_ext=True,
                    ),
                )
                check_weight_map(weight_map, mask_for_weight)
            weights.append(cache[cache_key])

        self._results['jacobian_weight_images'] = weights
        return runtime


#: Columns per DIFFPREP ``_moteddy_transformations.txt`` row: 6 rigid + 8
#: quadratic + 7 cubic + 3 eddy centre. See the module docstring for the full
#: layout and its source.
OKAN_NPARAMS = 24


def _okan_phase_axis(parameters):
    """Infer the eddy-current phase-encode axis (0/1/2) from one 24-parameter
    row.

    Reproduces the tie-broken heuristic in TORTOISE's own
    ``OkanQuadraticTransform(const ParametersType params)`` read-back
    constructor (``itkOkanQuadraticTransform.hxx``): compare columns 6, 7, 8
    in that order, with each later comparison overwriting the previous one on
    a tie. Real transformation files always resolve this cleanly, because the
    phase-encode column of an identity row is 1 and the other two are 0, and
    a fitted eddy-current correction perturbs that column only slightly.
    """
    c6, c7, c8 = parameters[6], parameters[7], parameters[8]
    phase = 1
    if c8 >= c7 and c8 >= c6:
        phase = 2
    if c6 >= c7 and c6 >= c8:
        phase = 0
    if c7 >= c6 and c7 >= c8:
        phase = 1
    return phase


def _okan_coordinate_frame(affine):
    """TORTOISE's "DP frame" axis spacing and isocenter index for ``affine``.

    Returns ``(spacing, indo)``: the per-axis voxel spacing (mm) and the
    continuous voxel index of the physical origin (scanner isocenter),
    computed the way ``DIFFPREP::ChangeImageHeaderToDP`` does for
    ``rot_eddy_center="isocenter"`` (qsiprep's default; see the module
    docstring). ``affine`` is a nibabel-style RAS+ affine; it is converted to
    ITK's LPS convention first (negate x, y), matching TORTOISE's own
    ITK-based image I/O.
    """
    affine = np.asarray(affine, dtype=float)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    linear = lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing
    origin = lps[:3, 3]
    indo = (direction.T @ (-origin)) / spacing
    return spacing, indo


def okan_quadratic_jacobian(parameters, shape, affine):
    """Analytic ``det grad phi`` of the eddy-current-only component.

    Uses only columns 6-23 of one DIFFPREP 24-parameter row, treating the
    rigid part (columns 0-5) as identity -- see the module docstring for the
    sourced formula and the derivation of why this reduces to a single
    partial derivative.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.size != OKAN_NPARAMS:
        raise ValueError(
            f'expected {OKAN_NPARAMS} Okan transform parameters, got {parameters.size}'
        )

    phase = _okan_phase_axis(parameters)
    spacing, indo = _okan_coordinate_frame(affine)

    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    x = spacing[0] * (ii - indo[0]) - parameters[21]
    y = spacing[1] * (jj - indo[1]) - parameters[22]
    z = spacing[2] * (kk - indo[2]) - parameters[23]

    c6, c7, c8, c9, c10, c11, c12, c13 = parameters[6:14]
    if phase == 0:
        det = c6 + c9 * y + c10 * z + 2 * c12 * x - 2 * c13 * x
    elif phase == 1:
        det = c7 + c9 * x + c11 * z - 2 * c12 * y - 2 * c13 * y
    else:
        det = c8 + c10 * x + c11 * y + 4 * c13 * z
    return det


def _okan_transform_point(px, py, pz, parameters):
    """Full 24-parameter forward map (rigid + quadratic + cubic), vectorized.

    Implements ``OkanQuadraticTransform::TransformPoint`` exactly (see the
    module docstring). ``px, py, pz`` are DP-frame physical coordinates
    (broadcastable arrays). Used only by ``resample_with_okan_transform``, the
    ship-gate helper -- production Jacobian weighting never needs the full
    point map, only its EC-only determinant (``okan_quadratic_jacobian``).
    """
    phase = _okan_phase_axis(parameters)
    center_x, center_y, center_z = parameters[21], parameters[22], parameters[23]
    x = px - center_x
    y = py - center_y
    z = pz - center_z

    ax, ay, az = parameters[3], parameters[4], parameters[5]
    cos_x, sin_x = np.cos(ax), np.sin(ax)
    cos_y, sin_y = np.cos(ay), np.sin(ay)
    cos_z, sin_z = np.cos(az), np.sin(az)
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    matrix = rot_z @ rot_y @ rot_x

    rx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] * z + parameters[0]
    ry = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] * z + parameters[1]
    rz = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * z + parameters[2]

    c6, c7, c8, c9, c10, c11, c12, c13 = parameters[6:14]
    new_phase = (
        c6 * rx
        + c7 * ry
        + c8 * rz
        + c9 * rx * ry
        + c10 * rx * rz
        + c11 * ry * rz
        + c12 * (rx**2 - ry**2)
        + c13 * (2 * rz**2 - rx**2 - ry**2)
    )

    c14, c15, c16, c17, c18, c19, c20 = parameters[14:21]
    total_change = (
        c14 * rx * ry * rz
        + c15 * rz * (rx**2 - ry**2)
        + c16 * rx * (4 * rz**2 - rx**2 - ry**2)
        + c17 * ry * (4 * rz**2 - rx**2 - ry**2)
        + c18 * rx * (rx**2 - 3 * ry**2)
        + c19 * ry * (3 * rx**2 - ry**2)
        + c20 * rz * (2 * rz**2 - 3 * rx**2 - 3 * ry**2)
    )

    out = [rx, ry, rz]
    out[phase] = new_phase + total_change
    out[0] = out[0] + center_x
    out[1] = out[1] + center_y
    out[2] = out[2] + center_z
    return out[0], out[1], out[2]


def resample_with_okan_transform(image, transformations_file, out_path):
    """Reconstruct DIFFPREP's motion+eddy resampling from its own parameters.

    Ship-gate helper only (see ``test_reconstructed_transform_reproduces_
    moteddy`` in ``test_interfaces_diffprep.py``): applies the *full*
    24-parameter ``OkanQuadraticTransform`` (rigid + quadratic + cubic, all
    unconditionally -- columns a given ``correction_mode`` never populates
    are simply zero, so including them is a no-op) to every volume of
    ``image`` and resamples it against itself, replicating TORTOISE's own
    ``ResampleImageFilter``-based resampling in
    ``DIFFPREP::WriteOutputFiles`` (see the module docstring): for each
    output voxel, the transform maps the output DP-frame physical point to
    the point to sample in the input image, and ITK's default (linear)
    interpolator reads it there.
    """
    from scipy.ndimage import map_coordinates

    img = nb.load(image)
    data = np.asanyarray(img.dataobj).astype(np.float64)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    shape3 = data.shape[:3]
    nvols = data.shape[3]

    rows = _read_okan_transformations(transformations_file)
    if len(rows) != nvols:
        raise ValueError(
            f'{transformations_file} has {len(rows)} transform rows but {image} '
            f'has {nvols} volumes.'
        )

    spacing, indo = _okan_coordinate_frame(img.affine)
    ii, jj, kk = np.meshgrid(
        np.arange(shape3[0]), np.arange(shape3[1]), np.arange(shape3[2]), indexing='ij'
    )
    x_out = spacing[0] * (ii - indo[0])
    y_out = spacing[1] * (jj - indo[1])
    z_out = spacing[2] * (kk - indo[2])

    out = np.zeros(data.shape, dtype=np.float32)
    for vol in range(nvols):
        params = np.asarray(rows[vol], dtype=float)
        if params.size < OKAN_NPARAMS:
            raise ValueError(f'expected {OKAN_NPARAMS} columns per row, got {params.size}')
        x_in, y_in, z_in = _okan_transform_point(x_out, y_out, z_out, params)
        idx_in = np.stack(
            [
                x_in / spacing[0] + indo[0],
                y_in / spacing[1] + indo[1],
                z_in / spacing[2] + indo[2],
            ]
        )
        out[..., vol] = map_coordinates(data[..., vol], idx_in, order=1, mode='constant', cval=0.0)

    result_data = out[..., 0] if out.shape[3] == 1 else out
    nb.Nifti1Image(result_data, img.affine, img.header).to_filename(out_path)
    return out_path


class _OkanQuadraticJacobianInputSpec(BaseInterfaceInputSpec):
    transformations_file = File(
        exists=True,
        mandatory=True,
        desc='DIFFPREP _moteddy_transformations.txt file with 24 columns per volume',
    )
    reference_image = File(
        exists=True,
        mandatory=True,
        desc='DIFFPREP input grid (the extract_b0s.b0_average grid): the '
        'eddy-current Jacobian is evaluated here, not on the output grid',
    )
    correction_mode = traits.Enum(
        'motion',
        'quadratic',
        'cubic',
        mandatory=True,
        desc="qsiprep's effective_correction_mode (after the --sloppy "
        'downgrade). "motion" has no eddy-current component; "cubic" is a '
        'valid DIFFPREP mode this module does not implement a determinant '
        'for, so it degrades (Undefined output, logged warning) rather than '
        'raising or misapplying the quadratic formula.',
    )


class _OkanQuadraticJacobianOutputSpec(TraitedSpec):
    ec_jacobian_images = OutputMultiObject(
        File(exists=True),
        desc="per-volume eddy-current Jacobian determinants; Undefined for "
        "correction_mode in ('motion', 'cubic')",
    )


class OkanQuadraticJacobian(SimpleInterface):
    """Per-volume TORTOISE eddy-current Jacobian determinant maps.

    See the module docstring for the sourced Okan quadratic-transform formula
    this implements. Only ``correction_mode == 'quadratic'`` is supported:
    'motion' has no eddy-current component to weight, and 'cubic' is a valid,
    existing DIFFPREP mode this module does not implement a determinant for.
    Both return ``Undefined`` rather than raising, so a weighting-enabled run
    never aborts on an otherwise-working DIFFPREP configuration -- the
    'cubic' gap reaches the sidecar via
    ``qsiprep.workflows.dwi.jacobian_provenance.jacobian_provenance_for``,
    computed independently of this interface's execution.
    """

    input_spec = _OkanQuadraticJacobianInputSpec
    output_spec = _OkanQuadraticJacobianOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.correction_mode == 'motion':
            LOGGER.warning(
                "correction_mode='motion' has no eddy-current component to "
                'weight; ec_jacobian_images will be Undefined.'
            )
            return runtime
        if self.inputs.correction_mode == 'cubic':
            LOGGER.warning(
                "correction_mode='cubic' is not supported for eddy-current "
                'Jacobian weighting (only the quadratic terms are '
                'implemented here); ec_jacobian_images will be Undefined and '
                'the eddy-current component of this run will be unmodulated.'
            )
            return runtime

        rows = _read_okan_transformations(self.inputs.transformations_file)
        ref = nb.load(self.inputs.reference_image)
        shape = ref.shape[:3]
        affine = ref.affine

        images = []
        for index, row in enumerate(rows):
            det = okan_quadratic_jacobian(row, shape, affine)
            out_path = fname_presuffix(
                self.inputs.reference_image,
                suffix=f'_ecjac-{index:05d}',
                newpath=runtime.cwd,
                use_ext=True,
            )
            nb.Nifti1Image(det.astype('float32'), affine, ref.header).to_filename(out_path)
            images.append(out_path)

        self._results['ec_jacobian_images'] = images
        return runtime


def _jacobian_sidecar(weight_index, applied, unmodulated, reason):
    """Sidecar for the Jacobian weight derivative.

    ``weight_index`` is zero-based, one entry per volume of the preprocessed
    DWI series, indexing volumes of the 4D weight file. Repeated maps appear as
    repeated indices, which makes the dedup visible rather than implicit; the
    collapsed single-map case is all zeros, written in full so consumers need
    no special case.

    ``weight_index`` is guarded with ``isdefined`` rather than assumed present:
    ``weight_images`` defined with ``weight_index`` left Undefined is
    unreachable in the current wiring (``StackJacobianWeights`` always
    receives both together), but ``list(Undefined)`` raises ``TypeError``
    rather than something informative, so it is cheap to guard against a
    future caller that decouples them.
    """
    sidecar = {
        'JacobianWeightIndex': list(weight_index) if isdefined(weight_index) else [],
        'AppliedCorrections': list(applied),
        'UnmodulatedCorrections': list(unmodulated),
        'Description': (
            'Multiplicative Jacobian intensity modulation applied to the '
            'preprocessed DWI series immediately after spatial resampling. '
            'Dividing by the indexed volume reverses that multiplication at '
            'that point in the pipeline; it does not recover an unmodulated '
            'series, because denoising and bias-field correction run after '
            'resampling and do not commute with it.'
        ),
    }
    if unmodulated and reason:
        sidecar['UnmodulatedReason'] = reason
    return sidecar


class _StackJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    # Not mandatory: QSIPrep's own inputnode.jacobian_weight_images is
    # Undefined whenever ComposeJacobianWeights applied no weights (weighting
    # disabled, or every modulation was internal to the HMC backend). That is
    # a normal, expected run outcome, not an error -- see the module and
    # derivatives-workflow docstrings for the "no weights" contract. A
    # mandatory trait here would make nipype's own mandatory-input check raise
    # ``ValueError`` on exactly that run instead of letting this interface
    # no-op and leave its outputs Undefined for ``DerivativesMaybeDataSink``.
    weight_images = InputMultiObject(
        File(exists=True),
        desc='unique output-grid weight maps, first-appearance order',
    )
    weight_index = traits.List(
        traits.Int(), desc='per-volume index into weight_images'
    )
    # Build-time facts about this run, computed by
    # ``qsiprep.workflows.dwi.jacobian_provenance.jacobian_provenance_for``
    # and set as node inputs during workflow construction -- not read from
    # ``config.workflow`` here, because they are per-run facts (which
    # corrections a *specific* unit's compiled plan actually Jacobian-
    # modulated), not invocation-global ones. See that module's docstring.
    applied_corrections = traits.List(
        traits.Str(), usedefault=True, desc='corrections QSIPrep itself Jacobian-modulated'
    )
    unmodulated_corrections = traits.List(
        traits.Str(), usedefault=True, desc='corrections that ran without Jacobian modulation'
    )
    unmodulated_reason = traits.Either(
        None,
        traits.Str(),
        default=None,
        usedefault=True,
        desc='why the unmodulated corrections went unmodulated',
    )


class _StackJacobianWeightsOutputSpec(TraitedSpec):
    out_file = File(desc='3D if one unique map, else 4D; Undefined if no weights were applied')
    meta_dict = traits.Dict(desc='sidecar for the derivative')


class StackJacobianWeights(SimpleInterface):
    """Stack the unique weight maps and build the sidecar.

    3D when every volume shares one map, 4D otherwise. The index is written
    out in full either way, so consumers need no special case. Undefined
    ``weight_images`` (no weights were applied this run) is propagated as
    Undefined outputs rather than raising or synthesizing a map of ones.
    """

    input_spec = _StackJacobianWeightsInputSpec
    output_spec = _StackJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.weight_images):
            return runtime

        images = [nb.load(path) for path in self.inputs.weight_images]
        out_file = os.path.join(runtime.cwd, 'jacobian_weights.nii.gz')
        if len(images) == 1:
            data = np.asanyarray(images[0].dataobj)
        else:
            data = np.stack([np.asanyarray(img.dataobj) for img in images], axis=-1)
        nb.Nifti1Image(
            data.astype('float32'), images[0].affine, images[0].header
        ).to_filename(out_file)

        self._results['out_file'] = out_file
        self._results['meta_dict'] = _jacobian_sidecar(
            weight_index=self.inputs.weight_index,
            applied=self.inputs.applied_corrections,
            unmodulated=self.inputs.unmodulated_corrections,
            reason=self.inputs.unmodulated_reason,
        )
        return runtime
