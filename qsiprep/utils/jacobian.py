"""Intensity modulation for QSIPrep's spatial distortion corrections.

Correcting a spatial distortion moves signal between voxels, so the corrected
image has to be rescaled by the local volume change or regions the acquisition
compressed stay too bright. QSIPrep follows TORTOISE's definitions throughout
(``src/main/FINALDATA.cxx`` in the TORTOISE source the container is built
from):

* **Jacobian** (``--output_signal_redist_method Jac``): each volume is
  multiplied by ``1 + d(u_PE)/d(PE)``, the derivative of the composed
  native-space displacement along the phase-encoding axis
  (``FINALDATA::ComputeDetImgFromAllTransExceptStr``). The composition covers
  gradient nonlinearity, susceptibility, and TORTOISE's own eddy-current
  transform. Head motion, coregistration and the template warps are excluded:
  a rigid realignment is not a volume change, and modulating by a
  normalization warp would be VBM-style modulation of the DWI signal. TORTOISE
  uses this method whenever only one phase-encoding polarity was acquired.
* **LSR** (``--output_signal_redist_method LSR``, TORTOISE's default when a
  reverse-polarity acquisition exists): each blip's volumes are multiplied by
  the ratio ``b0_corrected_final / blip_b0_corrected`` that DRBUDDI writes,
  where ``b0_corrected_final`` is the harmonic mean of the two
  geometry-corrected b=0 images. That ratio replaces the Jacobian entirely,
  so no gradient-nonlinearity or eddy-current factor is applied on top of it
  (``FINALDATA.cxx``, the ``method == "LSR"`` branch).

TORTOISE eddy-current Jacobian
------------------------------

DIFFPREP writes one 24-parameter row per volume to
``_moteddy_transformations.txt`` (``itkOkanQuadraticTransform.h``,
``NQUADPARAMS = 24``; ``DIFFPREP::WriteOutputFiles``). Columns, 0-indexed::

    0-2    rigid translation x, y, z (mm)
    3-5    Euler angles theta_x, theta_y, theta_z (radians); R = Rz . Ry . Rx
    6-8    linear coefficients of the phase-axis polynomial along x, y, z
    9-11   quadratic cross terms xy, xz, yz
    12-13  quadratic terms (x^2 - y^2) and (2z^2 - x^2 - y^2)
    14-20  cubic terms, active only when correction_mode == 'cubic'
    21-23  eddy-current/rotation centre (mm); 0 for every qsiprep preset

``TransformPoint`` maps an output-space point to the input-space point
``ResampleImageFilter`` samples::

    p -= center
    p  = R @ p + T
    p[phase] = c6*p.x + c7*p.y + c8*p.z + c9*p.x*p.y + c10*p.x*p.z + c11*p.y*p.z
               + c12*(p.x**2 - p.y**2) + c13*(2*p.z**2 - p.x**2 - p.y**2)
    p += center

Only the phase-encoding coordinate is replaced, so the Jacobian matrix has two
rows equal to basis vectors and its determinant is the partial derivative of
the polynomial with respect to the phase coordinate, evaluated at the rigidly
moved point ``R @ p + T`` (``ComputeJacobianWithRespectToPosition``). The
phase axis is inferred the way TORTOISE's own read-back constructor infers it:
the largest of columns 6, 7, 8 (identity leaves the phase column at 1).

Coordinates are TORTOISE's "DP frame" (``DIFFPREP::ChangeImageHeaderToDP``):
the image's own voxel axes with identity direction cosines, origin placed so
that the eddy centre sits at (0, 0, 0). ``rot_eddy_center`` selects that
centre: the scanner isocenter (default), the centre voxel, or the centre slice.
"""

import itertools
import os

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


#: Tolerances for comparing two images' affines: loose enough for float32
#: header round-trips, tight enough that a different lattice fails.
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4


#: What a non-positive resampled weight becomes. 1.0 leaves the voxel's
#: intensity alone. A determinant at or below zero says the volume change is
#: unrecoverable there (signal pile-up) or is resampling undershoot; neither is
#: a measurement to scale by.
UNMODULATED_WEIGHT = 1.0


#: Fraction of in-mask voxels with a non-positive determinant above which the
#: warning says the composed warp probably folds across the brain. A real
#: susceptibility field drives the determinant to zero where EPI signal piles
#: up (orbitofrontal, temporal poles), so a localised region is expected.
FOLD_FRACTION_LIMIT = 0.05


#: In-mask median outside this range is worth a warning: a correct distortion
#: field redistributes signal without changing its total.
MEDIAN_WARN_RANGE = (0.9, 1.1)


#: Sidecar vocabulary for how the weights were obtained.
METHOD_JACOBIAN = 'Jacobian'
METHOD_LSR = 'LSR'


#: How far apart (mm, along any axis) two images' world bounding boxes may sit
#: and still count as the same coordinate domain.
WORLD_OVERLAP_SLACK_MM = 1e-2


def weight_key(gradwarp, fieldwarp):
    """Return the cache key for one combination of distortion fields.

    Parameters
    ----------
    gradwarp : str or None
        Gradient nonlinearity displacement field.
    fieldwarp : str or None
        Susceptibility displacement field.

    Returns
    -------
    tuple of str
        The fields that are present, in order. Empty (falsy) when neither is,
        which means a unity weight.
    """
    return tuple(path for path in (gradwarp, fieldwarp) if path)


def _world_bounding_box(img):
    """Return the axis-aligned world-space bounding box of an image's voxel grid.

    Parameters
    ----------
    img : nibabel.spatialimages.SpatialImage
        Image whose first three axes define the grid.

    Returns
    -------
    tuple of numpy.ndarray
        The minimum and maximum world coordinates (mm) of the grid's corners.
    """
    shape = img.shape[:3]
    corners = np.array(list(itertools.product(*[(0, dim - 1) for dim in shape])))
    world = nb.affines.apply_affine(img.affine, corners)
    return world.min(axis=0), world.max(axis=0)


def _assert_world_frames_overlap(path_a, path_b, img_a=None, img_b=None):
    """Raise unless two images occupy overlapping world space.

    ANTs composes and resamples in physical coordinates, so a different
    sampling lattice is fine. An input in an unrelated coordinate frame (wrong
    subject, wrong units, a header bug) is not.

    Parameters
    ----------
    path_a, path_b : str
        The two images, named in the error message.
    img_a, img_b : nibabel.spatialimages.SpatialImage, optional
        The images already loaded, to avoid reading them again.

    Raises
    ------
    ValueError
        If the bounding boxes are farther apart than ``WORLD_OVERLAP_SLACK_MM``.
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
            'lattice is fine, but these are not in a compatible coordinate domain.'
        )


def resample_like(source_path, like_path, out_path, interpolation='nearest', fill_value=0):
    """Resample an image onto another image's grid, if it is not already on it.

    ``force_resample=True`` is passed to nilearn, which otherwise takes a
    padding fast path for whole-voxel translations that ignores ``fill_value``.

    Parameters
    ----------
    source_path : str
        Image to resample.
    like_path : str
        Image whose grid to resample onto.
    out_path : str
        Where to write the resampled image.
    interpolation : str, optional
        nilearn interpolation method.
    fill_value : float, optional
        Value outside the source's field of view: 0 suits a mask, 1.0 a
        multiplicative weight.

    Returns
    -------
    str
        ``source_path`` if the two images already share a lattice, else
        ``out_path``.
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


def _field_components(img):
    """Return the displacement vectors of an ITK displacement-field image.

    Parameters
    ----------
    img : nibabel.spatialimages.SpatialImage
        A 4D (X, Y, Z, 3) or 5D (X, Y, Z, 1, 3) displacement field.

    Returns
    -------
    numpy.ndarray
        The (X, Y, Z, 3) displacement array, as float.

    Raises
    ------
    ValueError
        If the image has neither layout.
    """
    shape = img.shape
    if img.ndim == 4 and shape[3] == 3:
        return np.asanyarray(img.dataobj).astype(float)
    if img.ndim == 5 and shape[3] == 1 and shape[4] == 3:
        return np.asanyarray(img.dataobj).astype(float)[:, :, :, 0, :]
    raise ValueError(
        f'Displacement field has shape {shape}, expected a 4D image shaped '
        '(X, Y, Z, 3) or a 5D image shaped (X, Y, Z, 1, 3).'
    )


def validate_field_geometry(field_path, reference_path):
    """Raise unless a displacement field can be composed against a reference image.

    Only the layout (three vector components) and the world frame are checked.
    The direction of a field cannot be checked from headers; it is established
    by the conservation test and the positivity guard.

    Parameters
    ----------
    field_path : str
        Displacement field.
    reference_path : str
        Image on whose grid the field will be composed.

    Raises
    ------
    ValueError
        If the field's layout or world frame is wrong.
    """
    field = nb.load(field_path)
    reference = nb.load(reference_path)
    try:
        _field_components(field)
    except ValueError as err:
        raise ValueError(f'{field_path}: {err}') from err
    _assert_world_frames_overlap(field_path, reference_path, img_a=field, img_b=reference)


def _report_nonpositive(inside, subject, consequence):
    """Warn about non-positive determinants inside the brain mask.

    A localized region is expected where susceptibility distortion piles EPI
    signal up. Above ``FOLD_FRACTION_LIMIT`` of the mask, the warning says the
    field probably folds across the brain.

    Parameters
    ----------
    inside : numpy.ndarray
        In-mask values.
    subject : str
        What the values are, for the message.
    consequence : str
        What happens to the non-positive voxels, for the message.
    """
    nonpositive = int((inside <= 0).sum())
    if not nonpositive:
        return

    fraction = nonpositive / inside.size
    if fraction > FOLD_FRACTION_LIMIT:
        LOGGER.warning(
            '%s is non-positive over %.1f%% of the brain mask (%d of %d voxels, '
            'minimum %.4f). That is more than signal pile-up usually explains, so '
            'check the field: a wrong, inverted or wrongly ordered field folds '
            'across the brain. %s',
            subject,
            100 * fraction,
            nonpositive,
            inside.size,
            float(inside.min()),
            consequence,
        )
        return

    LOGGER.warning(
        '%s is non-positive at %d of %d in-mask voxels (%.2f%%, minimum %.4f). '
        'This is expected where susceptibility distortion piles EPI signal up; '
        'those voxels are left unmodulated (weight %g) downstream.',
        subject,
        nonpositive,
        inside.size,
        100 * fraction,
        float(inside.min()),
        UNMODULATED_WEIGHT,
    )


def check_weight_map(map_path, mask_path):
    """Check a weight map inside a brain mask.

    Non-positive values and an in-mask median outside ``MEDIAN_WARN_RANGE``
    produce warnings; a correct distortion field redistributes signal without
    changing its total.

    Parameters
    ----------
    map_path : str
        Weight map.
    mask_path : str
        Brain mask on the same grid.

    Raises
    ------
    ValueError
        If the map contains non-finite values or has no voxels in the mask.
    """
    weights = np.asanyarray(nb.load(map_path).dataobj)
    mask = np.asanyarray(nb.load(mask_path).dataobj) > 0

    if not np.isfinite(weights).all():
        raise ValueError(f'Jacobian weight map {map_path} contains non-finite values.')

    inside = weights[mask]
    if inside.size == 0:
        raise ValueError(
            f'Jacobian weight map {map_path} has no voxels in mask {mask_path}. '
            'This usually means the mask and the weight map do not share a world '
            'frame, not that the mask is genuinely empty.'
        )

    _report_nonpositive(
        inside,
        f'Jacobian weight map {map_path}',
        'Those voxels are left unmodulated.',
    )

    median = float(np.median(inside))
    if not MEDIAN_WARN_RANGE[0] <= median <= MEDIAN_WARN_RANGE[1]:
        LOGGER.warning(
            'Jacobian weight map %s has an in-mask median of %.4f, outside %s. '
            'A correct distortion field redistributes signal without changing '
            'its total, so this may indicate a wrong field, an inverted field, '
            'or a wrong composition order.',
            map_path,
            median,
            MEDIAN_WARN_RANGE,
        )


def multiply_maps(paths, out_path, like_path=None):
    """Multiply scalar maps voxelwise on one lattice.

    Every factor is first resampled linearly onto ``like_path``'s grid, with
    1.0 outside its field of view. This realigns lattices within one world
    frame; :func:`transport_scalar_map` relates different coordinate domains.

    Parameters
    ----------
    paths : list of str
        The factors.
    out_path : str
        Where to write the product.
    like_path : str, optional
        Image whose grid to use; the first factor's by default.

    Returns
    -------
    str
        A lone factor already on the target grid, else ``out_path``.
    """
    target = like_path if like_path is not None else paths[0]
    onlattice = [
        resample_like(
            path,
            target,
            fname_presuffix(path, suffix='_onlattice', newpath=os.path.dirname(out_path) or None),
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


def compose_fields(field_paths, reference, out_path, num_threads=1):
    """Compose displacement fields into one, on a reference image's grid.

    Parameters
    ----------
    field_paths : list of str
        Displacement fields in QSIPrep's native-to-target chain order; they are
        reversed for ANTs, as in ``ComposeTransforms``.
    reference : str
        Native-lattice reference image: the composite's domain is undistorted
        b=0-reference space.
    out_path : str
        Where to write the composite field.
    num_threads : int, optional
        Threads for ``antsApplyTransforms``.

    Returns
    -------
    str
        ``out_path``.
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
        num_threads=num_threads,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    LOGGER.info('Composing %d displacement fields: %s', len(field_paths), xfm.cmdline)
    xfm.run()
    return out_path


def transport_scalar_map(image_path, transform_path, reference_path, out_path, num_threads=1):
    """Resample a scalar map through a transform onto a reference image's grid.

    This brings TORTOISE's eddy-current Jacobian, evaluated on DIFFPREP's
    distorted-native grid, to the undistorted b=0-reference coordinates at
    which the gradient nonlinearity and susceptibility determinant is
    evaluated.

    Parameters
    ----------
    image_path : str
        Scalar map to resample.
    transform_path : str
        Transform to resample through.
    reference_path : str
        Image whose grid to resample onto.
    out_path : str
        Where to write the result.
    num_threads : int, optional
        Threads for ``antsApplyTransforms``.

    Returns
    -------
    str
        ``out_path``. Voxels outside the transported footprint are 1.0, the
        multiplicative identity.
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
        num_threads=num_threads,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    LOGGER.info('Transporting %s through %s: %s', image_path, transform_path, xfm.cmdline)
    xfm.run()
    return out_path


def pe_axis_from_direction(pe_dir):
    """Return the voxel axis of a BIDS phase-encoding direction.

    Parameters
    ----------
    pe_dir : str
        ``PhaseEncodingDirection`` value, e.g. ``"j-"``.

    Returns
    -------
    int
        0, 1 or 2.

    Raises
    ------
    ValueError
        If the value is not recognized.
    """
    try:
        return {'i': 0, 'j': 1, 'k': 2, 'x': 0, 'y': 1, 'z': 2}[str(pe_dir)[0]]
    except (KeyError, IndexError) as err:
        raise ValueError(f'Unrecognized PhaseEncodingDirection {pe_dir!r}') from err


def jacobian_determinant(field_path, out_path, pe_axis, mask_path=None):
    """Compute TORTOISE's Jacobian of a displacement field, ``1 + d(u_PE)/d(PE)``.

    ``FINALDATA::ComputeDetImgFromAllTransExceptStr`` differentiates the
    composed displacement along the phase-encoding voxel axis only, with
    central differences and unity at the two edge slices. For a field that
    displaces along that axis alone (DRBUDDI, GRE, SyN) this equals the full
    determinant; for a field with other components (gradient nonlinearity) it
    is the phase-encoding volume change TORTOISE applies.

    The field is an ITK displacement field (LPS world-space millimeters,
    output space to input space). The displacement is projected onto the
    phase-encoding voxel axis through the header's direction cosines, so
    oblique acquisitions are handled like axis-aligned ones. The sign is
    checked before the absolute value is taken, so a fold (negative value) is
    reported rather than turned into a plausible weight.

    Parameters
    ----------
    field_path : str
        Displacement field.
    out_path : str
        Where to write the determinant.
    pe_axis : int
        Phase-encoding voxel axis (0, 1 or 2).
    mask_path : str, optional
        Brain mask that localizes the check for non-positive values.

    Returns
    -------
    str
        ``out_path``, holding the absolute value of the determinant as float32.
    """
    img = nb.load(field_path)
    displacement = _field_components(img)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ img.affine
    linear = lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing

    along_pe = displacement @ direction[:, pe_axis]
    signed = 1.0 + np.gradient(along_pe, spacing[pe_axis], axis=pe_axis)
    # TORTOISE leaves the two edge slices along the phase axis at unity.
    edge = [slice(None)] * 3
    for index in (0, -1):
        edge[pe_axis] = index
        signed[tuple(edge)] = 1.0

    if mask_path is not None:
        mask = np.asanyarray(nb.load(mask_path).dataobj) > 0
        inside = signed[mask]
        if inside.size:
            _report_nonpositive(
                inside,
                f'Jacobian determinant of {field_path}',
                'Those voxels are left unmodulated.',
            )
    elif np.nanmin(signed) <= 0:
        LOGGER.warning(
            'Jacobian determinant of %s contains non-positive values and no '
            'mask was supplied to localise them. Edge negatives are expected; '
            'interior ones are a fold.',
            field_path,
        )

    nb.Nifti1Image(np.abs(signed).astype('float32'), img.affine).to_filename(out_path)
    return out_path


def validate_scalar_geometry(image_path, reference_path):
    """Raise unless an image is a 3D scalar map in a reference image's world frame.

    Parameters
    ----------
    image_path : str
        Scalar map.
    reference_path : str
        Reference image.

    Raises
    ------
    ValueError
        If the image is not 3D or does not overlap the reference in world space.
    """
    image = nb.load(image_path)
    reference = nb.load(reference_path)

    if image.ndim != 3:
        raise ValueError(f'{image_path} is not a 3D scalar map (shape {image.shape}).')

    _assert_world_frames_overlap(image_path, reference_path, img_a=image, img_b=reference)


def float32_header(img):
    """Return a copy of an image's header that stores float32 data.

    nibabel writes an image in its header's data type, not the array's, so an
    array cast to float32 but paired with a float64 source header is written
    as float64.

    Parameters
    ----------
    img : nibabel.spatialimages.SpatialImage
        Image whose header to copy.

    Returns
    -------
    nibabel.Nifti1Header
        The copied header, with a float32 data type.
    """
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    return header


#: Columns per DIFFPREP ``_moteddy_transformations.txt`` row: 6 rigid + 8
#: quadratic + 7 cubic + 3 eddy centre.
OKAN_NPARAMS = 24

ROT_EDDY_CENTERS = ('isocenter', 'center_voxel', 'center_slice')


def _okan_phase_axis(parameters):
    """Infer the eddy-current phase-encoding axis from one DIFFPREP transform row.

    Reproduces the tie-broken comparison in TORTOISE's
    ``OkanQuadraticTransform(const ParametersType params)`` constructor: the
    phase-encoding column of an identity row is 1 and the other two are 0, and
    a fitted correction perturbs it only slightly.

    Parameters
    ----------
    parameters : array-like
        One 24-parameter row.

    Returns
    -------
    int
        0, 1 or 2.
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


def _okan_coordinate_frame(affine, shape=None, rot_eddy_center='isocenter'):
    """Return the voxel spacing and center index of TORTOISE's "DP frame".

    Parameters
    ----------
    affine : array-like
        nibabel RAS+ affine; converted to ITK's LPS convention.
    shape : tuple of int, optional
        Image shape, needed for the two center-based frames.
    rot_eddy_center : {'isocenter', 'center_voxel', 'center_slice'}, optional
        Which point ``DIFFPREP::ChangeImageHeaderToDP`` places at physical
        (0, 0, 0).

    Returns
    -------
    spacing : numpy.ndarray
        Voxel spacing per axis (mm).
    indo : numpy.ndarray
        Continuous voxel index of the frame's center.

    Raises
    ------
    ValueError
        If ``rot_eddy_center`` is unknown, or ``shape`` is missing for a
        center-based frame.
    """
    if rot_eddy_center not in ROT_EDDY_CENTERS:
        raise ValueError(
            f'rot_eddy_center must be one of {ROT_EDDY_CENTERS}, got {rot_eddy_center!r}'
        )
    affine = np.asarray(affine, dtype=float)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    linear = lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing
    origin = lps[:3, 3]

    if rot_eddy_center == 'isocenter':
        return spacing, (direction.T @ (-origin)) / spacing

    if shape is None:
        raise ValueError(f'rot_eddy_center={rot_eddy_center!r} needs the image shape')
    center_voxel_index = (np.asarray(shape[:3], dtype=float) - 1) / 2.0
    if rot_eddy_center == 'center_voxel':
        return spacing, center_voxel_index

    # center_slice: in-plane isocenter, through-plane the centre voxel's slice.
    center_voxel_point = direction @ (spacing * center_voxel_index) + origin
    center_point = np.array([0.0, 0.0, center_voxel_point[2]])
    return spacing, (direction.T @ (center_point - origin)) / spacing


def _okan_rigid(parameters):
    """Return the rigid part of one DIFFPREP transform row.

    Parameters
    ----------
    parameters : array-like
        One 24-parameter row.

    Returns
    -------
    rotation : numpy.ndarray
        3x3 rotation, ``Rz . Ry . Rx``.
    translation : numpy.ndarray
        Translation (mm).
    """
    ax, ay, az = parameters[3], parameters[4], parameters[5]
    cos_x, sin_x = np.cos(ax), np.sin(ax)
    cos_y, sin_y = np.cos(ay), np.sin(ay)
    cos_z, sin_z = np.cos(az), np.sin(az)
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x, np.asarray(parameters[0:3], dtype=float)


def okan_quadratic_jacobian(parameters, shape, affine, rot_eddy_center='isocenter'):
    """Compute the Jacobian determinant of DIFFPREP's eddy-current correction.

    The determinant is the partial derivative of the phase-axis polynomial
    with respect to its own coordinate, evaluated at the rigidly moved point
    ``R @ p + T``, as ``OkanQuadraticTransform::
    ComputeJacobianWithRespectToPosition`` does (the rigid rows contribute
    ``det R = 1``). Head motion therefore does not scale the weight, but it
    decides where the polynomial is sampled.

    Parameters
    ----------
    parameters : array-like
        One 24-parameter row of ``_moteddy_transformations.txt``.
    shape : tuple of int
        Grid shape.
    affine : array-like
        Grid affine (nibabel RAS+).
    rot_eddy_center : {'isocenter', 'center_voxel', 'center_slice'}, optional
        DIFFPREP's eddy-current center.

    Returns
    -------
    numpy.ndarray
        The determinant on the grid.

    Raises
    ------
    ValueError
        If ``parameters`` does not have 24 entries.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.size != OKAN_NPARAMS:
        raise ValueError(
            f'expected {OKAN_NPARAMS} Okan transform parameters, got {parameters.size}'
        )

    phase = _okan_phase_axis(parameters)
    spacing, indo = _okan_coordinate_frame(affine, shape, rot_eddy_center)

    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    x0 = spacing[0] * (ii - indo[0]) - parameters[21]
    y0 = spacing[1] * (jj - indo[1]) - parameters[22]
    z0 = spacing[2] * (kk - indo[2]) - parameters[23]
    matrix, translation = _okan_rigid(parameters)
    x = matrix[0, 0] * x0 + matrix[0, 1] * y0 + matrix[0, 2] * z0 + translation[0]
    y = matrix[1, 0] * x0 + matrix[1, 1] * y0 + matrix[1, 2] * z0 + translation[1]
    z = matrix[2, 0] * x0 + matrix[2, 1] * y0 + matrix[2, 2] * z0 + translation[2]

    c6, c7, c8, c9, c10, c11, c12, c13 = parameters[6:14]
    if phase == 0:
        det = c6 + c9 * y + c10 * z + 2 * c12 * x - 2 * c13 * x
    elif phase == 1:
        det = c7 + c9 * x + c11 * z - 2 * c12 * y - 2 * c13 * y
    else:
        det = c8 + c10 * x + c11 * y + 4 * c13 * z
    return det


def _okan_transform_point(px, py, pz, parameters):
    """Apply DIFFPREP's full 24-parameter forward map to points.

    Implements ``OkanQuadraticTransform::TransformPoint`` (rigid, quadratic and
    cubic terms), vectorized.

    Parameters
    ----------
    px, py, pz : numpy.ndarray
        DP-frame physical coordinates.
    parameters : array-like
        One 24-parameter row.

    Returns
    -------
    tuple of numpy.ndarray
        The transformed x, y and z coordinates.
    """
    phase = _okan_phase_axis(parameters)
    center_x, center_y, center_z = parameters[21], parameters[22], parameters[23]
    x = px - center_x
    y = py - center_y
    z = pz - center_z

    matrix, translation = _okan_rigid(parameters)
    rx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] * z + translation[0]
    ry = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] * z + translation[1]
    rz = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * z + translation[2]

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


def resample_with_okan_transform(
    image, transformations_file, out_path, rot_eddy_center='isocenter'
):
    """Reconstruct DIFFPREP's motion and eddy-current resampling from its parameters.

    Applies the full 24-parameter transform to every volume and resamples it
    against itself with linear interpolation, replicating
    ``DIFFPREP::WriteOutputFiles``. Used to check the parameter convention
    against TORTOISE's ``_moteddy.nii``.

    Parameters
    ----------
    image : str
        DWI series DIFFPREP corrected.
    transformations_file : str
        DIFFPREP's ``_moteddy_transformations.txt``, one row per volume.
    out_path : str
        Where to write the resampled series.
    rot_eddy_center : {'isocenter', 'center_voxel', 'center_slice'}, optional
        DIFFPREP's eddy-current center.

    Returns
    -------
    str
        ``out_path``.

    Raises
    ------
    ValueError
        If the file's row count differs from the volume count, or a row has
        fewer than 24 columns.
    """
    from scipy.ndimage import map_coordinates

    from ..interfaces.tortoise import _read_okan_transformations

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

    spacing, indo = _okan_coordinate_frame(img.affine, shape3, rot_eddy_center)
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


def jacobian_sidecar(weight_index, applied, unmodulated, reason, method, volume_deviation=None):
    """Build the sidecar of the intensity weight derivative.

    Parameters
    ----------
    weight_index : list of int or Undefined
        Zero-based index into the weight file, one entry per DWI volume.
        Repeated maps appear as repeated indices; a single map is all zeros.
    applied : list of str
        Corrections whose modulation QSIPrep applied.
    unmodulated : list of str
        Corrections that ran without modulation.
    reason : str or None
        Why the unmodulated corrections went unmodulated.
    method : {'Jacobian', 'LSR'}
        How the weights were obtained.
    volume_deviation : float, optional
        Given when the file holds the mean over volumes of distinct per-volume
        weights: the 99th percentile, over voxels, of the largest relative
        difference between a volume's own weight and the mean.

    Returns
    -------
    dict
        The sidecar.
    """
    sidecar = {
        'JacobianWeightIndex': list(weight_index) if isdefined(weight_index) else [],
        'SignalRedistributionMethod': method,
        'AppliedCorrections': list(applied),
        'UnmodulatedCorrections': list(unmodulated),
        'Description': (
            'Multiplicative intensity modulation applied to the preprocessed '
            'DWI series immediately after spatial resampling, following '
            "TORTOISE's signal redistribution: 'Jacobian' is 1 + d(u_PE)/d(PE) "
            "of the composed distortion field, 'LSR' is DRBUDDI's "
            'b0_corrected_final / blip_b0_corrected ratio. Dividing by the '
            'indexed volume reverses that multiplication at that point in the '
            'pipeline; it does not recover an unmodulated series, because '
            'denoising and bias-field correction run after resampling and do '
            'not commute with it.'
        ),
    }
    if volume_deviation is not None:
        sidecar['JacobianWeightCombination'] = 'mean'
        sidecar['JacobianWeightVolumeDeviation'] = volume_deviation
        sidecar['Description'] += (
            ' The weights differed between volumes; this file is their mean over '
            "volumes, and each volume's own weight differs from it by at most "
            'JacobianWeightVolumeDeviation (as a fraction, 99th percentile over '
            'voxels). Run with --write-jacobian-4d to write one map per volume.'
        )
    if unmodulated and reason:
        sidecar['UnmodulatedReason'] = reason
    return sidecar


def volume_mean(images, weight_index):
    """Return the mean over volumes of distinct weight maps.

    Map ``k`` counts once for every volume whose ``weight_index`` entry is
    ``k``, or every map once if no index is given.

    Parameters
    ----------
    images : list of nibabel.spatialimages.SpatialImage
        The distinct weight maps, on one grid.
    weight_index : list of int
        Index into ``images``, one entry per volume.

    Returns
    -------
    mean : numpy.ndarray
        The mean map, as float32.
    deviation : float
        The 99th percentile, over voxels where the mean is non-zero, of the
        largest relative difference between a used map and the mean.
    """
    counts = np.bincount(weight_index, minlength=len(images)) if weight_index else None
    if counts is None or not counts.any():
        counts = np.ones(len(images), dtype=int)
    total = np.zeros(images[0].shape, dtype=np.float64)
    for img, count in zip(images, counts, strict=True):
        if count:
            total += count * np.asanyarray(img.dataobj, dtype=np.float64)
    mean = total / counts.sum()
    inside = mean != 0
    largest = np.zeros(int(inside.sum()))
    for img, count in zip(images, counts, strict=True):
        if count:
            values = np.asanyarray(img.dataobj, dtype=np.float64)[inside]
            np.maximum(largest, np.abs(values / mean[inside] - 1), out=largest)
    deviation = float(np.percentile(largest, 99)) if largest.size else 0.0
    return mean.astype(np.float32), round(deviation, 6)
