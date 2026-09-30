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
* The fieldmap-less T2Wreg (EPIREG) field is applied geometrically but never
  modulated: EPIREG's final stage is not restricted to the phase-encoding
  direction, and ``FINALDATA`` composes an EPI field into the Jacobian only
  when it came from DRBUDDI. ``--force jacobian`` overrides that, using the
  phase-encoding component of the field.

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
from nipype.utils.filemanip import fname_presuffix

from .tortoise import _read_okan_transformations

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
    """Return the cache key for one unique (gradwarp, fieldwarp) combination.

    Falsy when neither field is present, which means a unity weight.
    """
    return tuple(path for path in (gradwarp, fieldwarp) if path)


def _world_bounding_box(img):
    """Return the axis-aligned world-space bounding box of ``img``'s voxel grid."""
    shape = img.shape[:3]
    corners = np.array(list(itertools.product(*[(0, dim - 1) for dim in shape])))
    world = nb.affines.apply_affine(img.affine, corners)
    return world.min(axis=0), world.max(axis=0)


def _assert_world_frames_overlap(path_a, path_b, img_a=None, img_b=None):
    """Raise unless ``path_a`` and ``path_b`` occupy overlapping world space.

    ANTs composes and resamples in physical coordinates, so a different
    sampling lattice is fine. An input in an unrelated coordinate frame (wrong
    subject, wrong units, a header bug) is not.
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
    """Resample ``source_path`` onto ``like_path``'s grid, if it is not already.

    Returns ``source_path`` unchanged when the two already share a lattice.
    ``fill_value`` is what appears outside the source's own field of view; it
    defaults to 0, which is right for a mask, while a multiplicative weight
    needs 1.0. ``force_resample=True`` is required because nilearn otherwise
    takes a padding fast path for whole-voxel translations that ignores
    ``fill_value``.
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
    """Return the (X, Y, Z, 3) displacement array of an ITK displacement-field image."""
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
    """Raise unless ``field_path`` is a plausible displacement field for ``reference_path``.

    The field is checked as a candidate for composition against ``reference_path``.
    Only the layout (three vector components) and the world frame are checked.
    Direction cannot be checked from headers; it is established behaviourally
    by the conservation test and the positivity guard.
    """
    field = nb.load(field_path)
    reference = nb.load(reference_path)
    try:
        _field_components(field)
    except ValueError as err:
        raise ValueError(f'{field_path}: {err}') from err
    _assert_world_frames_overlap(field_path, reference_path, img_a=field, img_b=reference)


def _report_nonpositive(inside, subject, consequence):
    """Warn about non-positive in-mask determinants, loudly above the fold limit."""
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
    """Raise on a non-finite in-mask weight; warn on non-positive or off-unity values."""
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
    """Multiply one or more scalar maps voxelwise, reconciled onto one lattice.

    Every factor is resampled (linearly, ``fill_value=1.0``) onto
    ``like_path``'s grid, which defaults to the first factor's, before
    multiplying. A lone factor already on that grid is returned unchanged.
    This is lattice realignment in a shared world frame; relating two
    genuinely different coordinate domains is ``transport_scalar_map``'s job.
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
    """Compose displacement fields into one, on ``reference``'s grid.

    ``field_paths`` is given in QSIPrep's native-to-target chain order; it is
    reversed here for ANTs, matching ``ComposeTransforms``. The reference is a
    native lattice: the composite's domain is undistorted b=0-reference space.
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
    """Resample a scalar map through ``transform_path`` onto ``reference_path``'s grid.

    Used to bring TORTOISE's eddy-current Jacobian, evaluated on DIFFPREP's
    distorted-native grid, to the undistorted b=0-reference coordinates the
    gradwarp/susceptibility determinant is evaluated at. ``default_value=1.0``
    keeps the multiplicative identity outside the transported footprint.
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
    """Return the voxel axis (0, 1, 2) of a BIDS ``PhaseEncodingDirection`` value."""
    try:
        return {'i': 0, 'j': 1, 'k': 2, 'x': 0, 'y': 1, 'z': 2}[str(pe_dir)[0]]
    except (KeyError, IndexError) as err:
        raise ValueError(f'Unrecognized PhaseEncodingDirection {pe_dir!r}') from err


def jacobian_determinant(field_path, out_path, pe_axis, mask_path=None):
    """Compute TORTOISE's Jacobian of a displacement field: ``1 + d(u_PE)/d(PE)``.

    ``FINALDATA::ComputeDetImgFromAllTransExceptStr`` differentiates the
    composed displacement along the phase-encoding voxel axis only, with
    central differences and unity at the two edge slices. For a field that
    displaces along that axis alone (DRBUDDI, GRE, SyN) this equals the full
    determinant; for a field with other components (gradient nonlinearity, a
    forced T2Wreg field) it is the phase-encoding volume change TORTOISE
    applies, and the other components are ignored.

    The field is an ITK displacement field: LPS world-space millimetres,
    output-space to input-space. The displacement is projected onto the
    phase-encoding voxel axis through the header's direction cosines, so an
    oblique acquisition is handled the same way as an axis-aligned one.

    The sign is inspected before any absolute value is taken: a folded warp
    gives a negative value, and a fold at -0.5 must not turn into an innocuous
    weight of 0.5 unreported. Pass ``mask_path`` to localise the check.
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
    """Raise unless ``image_path`` is a 3D scalar map in ``reference_path``'s world frame."""
    image = nb.load(image_path)
    reference = nb.load(reference_path)

    if image.ndim != 3:
        raise ValueError(f'{image_path} is not a 3D scalar map (shape {image.shape}).')

    _assert_world_frames_overlap(image_path, reference_path, img_a=image, img_b=reference)


class _ComposeJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='split DWI volumes, in their native grid; supplies the volume count',
    )
    # Only this image's grid is used: as the composition reference and as the
    # lattice the weight maps are written on. On the DRBUDDI-with-T2w path it
    # is the structural image, which init_structural_to_b0_alignment_wf has
    # already resampled onto the b=0 lattice.
    b0_ref_image = File(
        exists=True,
        mandatory=True,
        desc='undistorted b=0 reference; the lattice the weight maps live on',
    )
    # Not mandatory: nipype checks mandatory inputs before _run_interface, and
    # a failure there leaves MultiProc hanging on a missing result file
    # instead of reporting a node error. Required whenever there is something
    # to modulate; checked inside _run_interface.
    mask = File(exists=True, desc='native-space brain mask, for the positivity guard')
    pe_axis = traits.Either(
        None,
        traits.Range(0, 2),
        default=None,
        usedefault=True,
        desc='voxel axis of the phase-encoding direction; required for the '
        'Jacobian method whenever a field is supplied',
    )
    gradwarp_field = InputMultiObject(
        File(exists=True),
        desc='gradient nonlinearity displacement field (one, shared by every volume)',
    )
    fieldwarps = InputMultiObject(
        File(exists=True),
        desc='susceptibility displacement field(s): one shared, or one per DWI volume',
    )
    weight_fieldwarps = traits.Bool(
        True,
        usedefault=True,
        desc='False leaves the susceptibility field unmodulated: TORTOISE '
        'applies its T2Wreg (EPIREG) field without Jacobian modulation',
    )
    ec_jacobian_images = InputMultiObject(
        File(exists=True),
        desc='per-volume eddy-current Jacobian determinants (TORTOISE DIFFPREP)',
    )
    sdc_scaling_images = InputMultiObject(
        File(exists=True),
        desc="per-volume LSR ratio images from DRBUDDI (TORTOISE's default "
        'signal redistribution for reverse phase-encoded data). When given, '
        'they are the whole weight: no Jacobian factor is applied on top.',
    )
    num_threads = traits.Int(
        1, usedefault=True, nohash=True, desc='ITK threads for each ANTs call'
    )


class _ComposeJacobianWeightsOutputSpec(TraitedSpec):
    jacobian_weight_images = OutputMultiObject(
        File(exists=True),
        desc='one weight map per DWI volume, with repeats where volumes share one',
    )
    method = traits.Str(desc="'Jacobian' or 'LSR'; Undefined when no weights were built")


class ComposeJacobianWeights(SimpleInterface):
    """Build per-volume intensity weight maps for the native distortion corrections.

    With ``sdc_scaling_images`` (DRBUDDI's LSR ratios) each volume's weight is
    its blip's ratio, and nothing else, as in TORTOISE's default. Otherwise
    the weight is TORTOISE's Jacobian ``1 + d(u_PE)/d(PE)`` of the composed
    gradwarp/susceptibility field, evaluated in undistorted b=0-reference
    space, times that volume's eddy-current Jacobian when one exists. The
    eddy-current factor lives on DIFFPREP's distorted-native grid, so it is
    transported through the same composed field before multiplying; with no
    gradwarp and no susceptibility field there is nothing to transport through
    and the run's only coordinate domain is that grid.

    Unique ``(gradwarp, fieldwarp)`` combinations are computed once and shared,
    so a run with one gradwarp field and one susceptibility field costs a
    single composition even with hundreds of volumes.
    """

    input_spec = _ComposeJacobianWeightsInputSpec
    output_spec = _ComposeJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        num_dwis = len(self.inputs.dwi_files)
        reference = self.inputs.b0_ref_image
        num_threads = self.inputs.num_threads
        LOGGER.info('Building intensity weights for %d volumes', num_dwis)

        if isdefined(self.inputs.sdc_scaling_images) and self.inputs.sdc_scaling_images:
            return self._run_lsr(runtime, num_dwis, reference)

        gradwarp = None
        if isdefined(self.inputs.gradwarp_field) and self.inputs.gradwarp_field:
            if len(self.inputs.gradwarp_field) != 1:
                raise ValueError(
                    f'Expected a single gradwarp field, got {len(self.inputs.gradwarp_field)}.'
                )
            gradwarp = self.inputs.gradwarp_field[0]
            validate_field_geometry(gradwarp, reference)

        fieldwarps = [None] * num_dwis
        if (
            self.inputs.weight_fieldwarps
            and isdefined(self.inputs.fieldwarps)
            and self.inputs.fieldwarps
        ):
            supplied = list(self.inputs.fieldwarps)
            if len(supplied) == 1:
                LOGGER.info('Using a single susceptibility field for all DWI volumes')
                fieldwarps = supplied * num_dwis
            elif len(supplied) == num_dwis:
                LOGGER.info('Using per-volume susceptibility fields')
                fieldwarps = supplied
            else:
                raise ValueError(
                    f'Got {len(supplied)} susceptibility fields for {num_dwis} DWI '
                    'volumes; expected 1 or one per volume.'
                )
            for warp in set(fieldwarps):
                validate_field_geometry(warp, reference)
        elif isdefined(self.inputs.fieldwarps) and self.inputs.fieldwarps:
            LOGGER.info(
                'Susceptibility field supplied but left unmodulated (weight_fieldwarps=False)'
            )

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

        # The mask is only needed from here on. A run with nothing to modulate
        # must not be killed by a mask/reference mismatch it never needed.
        if not isdefined(self.inputs.mask):
            raise ValueError(
                'ComposeJacobianWeights has distortion transforms to modulate but no '
                'brain mask, so the folded-warp guard cannot run. Connect the DWI '
                'reference mask to this node.'
            )
        validate_scalar_geometry(self.inputs.mask, reference)

        pe_axis = self.inputs.pe_axis
        if (gradwarp is not None or any(fieldwarps)) and pe_axis is None:
            raise ValueError(
                'ComposeJacobianWeights needs pe_axis to differentiate a displacement '
                'field along the phase-encoding direction.'
            )

        # One determinant per unique (gradwarp, fieldwarp) pair. The composed
        # field itself is kept too: it is the transform that relates the
        # eddy-current Jacobian's grid to the reference grid.
        determinants = {}
        composed_transforms = {}
        unique_fieldwarps = list(dict.fromkeys(fieldwarps))
        for fieldwarp in unique_fieldwarps:
            key = weight_key(gradwarp, fieldwarp)
            if not key or key in determinants:
                continue
            fields = [path for path in (gradwarp, fieldwarp) if path]
            if len(fields) == 2:
                composed = compose_fields(
                    fields,
                    reference,
                    os.path.join(runtime.cwd, f'composite{len(determinants)}.nii.gz'),
                    num_threads=num_threads,
                )
            else:
                composed = fields[0]
            composed_transforms[key] = composed
            # The determinant is emitted on the field's own grid, which is not
            # always the reference's (a lone DRBUDDI field is on DRBUDDI's
            # output grid), so the mask is resampled there for the fold check.
            mask_for_determinant = resample_like(
                self.inputs.mask,
                composed,
                os.path.join(runtime.cwd, f'mask_for_jacobian{len(determinants)}.nii.gz'),
            )
            determinants[key] = jacobian_determinant(
                composed,
                os.path.join(runtime.cwd, f'jacobian{len(determinants)}.nii.gz'),
                pe_axis,
                mask_path=mask_for_determinant,
            )
        LOGGER.info('Computed %d unique Jacobian determinant(s)', len(determinants))

        weights = []
        cache = {}
        for index, (fieldwarp, ec_image) in enumerate(zip(fieldwarps, ec_images, strict=True)):
            factors = []
            key = weight_key(gradwarp, fieldwarp)
            if key:
                factors.append(determinants[key])

            # Tag the roles: (None, 'f.nii.gz') and ('f.nii.gz', None) are
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
                        num_threads=num_threads,
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
                    like_path=reference,
                )
                cache[cache_key] = weight_map
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

        LOGGER.info('Finished: %d unique weight map(s) for %d volumes', len(cache), num_dwis)
        self._results['jacobian_weight_images'] = weights
        self._results['method'] = METHOD_JACOBIAN
        return runtime

    def _run_lsr(self, runtime, num_dwis, reference):
        """Weight each volume by its blip's DRBUDDI ratio image (LSR), and nothing else."""
        supplied = list(self.inputs.sdc_scaling_images)
        if len(supplied) != num_dwis:
            raise ValueError(
                f'Got {len(supplied)} LSR scaling images for {num_dwis} DWI volumes; '
                'expected one per volume.'
            )
        LOGGER.info(
            "Using DRBUDDI's LSR signal redistribution (%d unique ratio image(s)); "
            'no Jacobian factor is applied on top of it, as in TORTOISE',
            len(set(supplied)),
        )
        if not isdefined(self.inputs.mask):
            raise ValueError(
                'ComposeJacobianWeights has LSR scaling images but no brain mask, so '
                'the weight guard cannot run. Connect the DWI reference mask to this node.'
            )
        validate_scalar_geometry(self.inputs.mask, reference)

        cache = {}
        weights = []
        for index, ratio in enumerate(supplied):
            if ratio not in cache:
                validate_scalar_geometry(ratio, reference)
                weight_map = multiply_maps(
                    [ratio],
                    fname_presuffix(
                        self.inputs.dwi_files[index],
                        suffix=f'_lsr-{index:05d}',
                        newpath=runtime.cwd,
                        use_ext=True,
                    ),
                    like_path=reference,
                )
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
                cache[ratio] = weight_map
            weights.append(cache[ratio])

        self._results['jacobian_weight_images'] = weights
        self._results['method'] = METHOD_LSR
        return runtime


#: Columns per DIFFPREP ``_moteddy_transformations.txt`` row: 6 rigid + 8
#: quadratic + 7 cubic + 3 eddy centre.
OKAN_NPARAMS = 24

ROT_EDDY_CENTERS = ('isocenter', 'center_voxel', 'center_slice')


def _okan_phase_axis(parameters):
    """Infer the eddy-current phase-encode axis (0/1/2) from one 24-parameter row.

    Reproduces the tie-broken comparison in TORTOISE's
    ``OkanQuadraticTransform(const ParametersType params)`` constructor: the
    phase-encode column of an identity row is 1 and the other two are 0, and a
    fitted correction perturbs it only slightly.
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
    """Return TORTOISE's "DP frame" spacing and centre index for ``affine``.

    Returns ``(spacing, indo)``: the per-axis voxel spacing (mm) and the
    continuous voxel index that ``DIFFPREP::ChangeImageHeaderToDP`` places at
    physical (0, 0, 0) for the requested ``rot_eddy_center``. ``affine`` is a
    nibabel RAS+ affine; it is converted to ITK's LPS convention first.
    ``shape`` is needed for the two centre-based frames.
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
    """Return the rigid part (R, T) of one 24-parameter row, ``R = Rz . Ry . Rx``."""
    ax, ay, az = parameters[3], parameters[4], parameters[5]
    cos_x, sin_x = np.cos(ax), np.sin(ax)
    cos_y, sin_y = np.cos(ay), np.sin(ay)
    cos_z, sin_z = np.cos(az), np.sin(az)
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x, np.asarray(parameters[0:3], dtype=float)


def okan_quadratic_jacobian(parameters, shape, affine, rot_eddy_center='isocenter'):
    """Compute the analytic ``det grad phi`` of DIFFPREP's eddy-current correction.

    The determinant is the phase-axis polynomial's partial derivative with
    respect to its own coordinate, evaluated at the rigidly moved point
    ``R @ p + T``, exactly as ``OkanQuadraticTransform::
    ComputeJacobianWithRespectToPosition`` does (the rigid rows contribute
    ``det R = 1``). Head motion therefore does not scale the weight, but it
    does decide where the eddy-current polynomial is sampled.
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
    """Apply the full 24-parameter forward map (rigid + quadratic + cubic), vectorized.

    Implements ``OkanQuadraticTransform::TransformPoint``. ``px, py, pz`` are
    DP-frame physical coordinates. Used by ``resample_with_okan_transform``
    to check the parameter convention against TORTOISE's own resampling.
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
    """Reconstruct DIFFPREP's motion+eddy resampling from its own parameters.

    Applies the full 24-parameter transform to every volume of ``image`` and
    resamples it against itself with linear interpolation, replicating
    ``DIFFPREP::WriteOutputFiles``. Used by the integration test that pins the
    parameter convention against TORTOISE's ``_moteddy.nii``.
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
        desc="qsiprep's effective_correction_mode (after the --sloppy downgrade). "
        '"motion" has no eddy-current component; "cubic" has no implemented '
        'determinant and degrades to an Undefined output with a warning.',
    )
    rot_eddy_center = traits.Enum(
        *ROT_EDDY_CENTERS,
        usedefault=True,
        desc="DIFFPREP's rot_eddy_center setting, which fixes the origin of the "
        'coordinate frame the eddy-current polynomial is expressed in',
    )


class _OkanQuadraticJacobianOutputSpec(TraitedSpec):
    ec_jacobian_images = OutputMultiObject(
        File(exists=True),
        desc='per-volume eddy-current Jacobian determinants; Undefined for '
        "correction_mode in ('motion', 'cubic')",
    )


class OkanQuadraticJacobian(SimpleInterface):
    """Per-volume TORTOISE eddy-current Jacobian determinant maps.

    Only ``correction_mode == 'quadratic'`` produces maps: 'motion' has no
    eddy-current component and 'cubic' has no implemented determinant. Both
    leave the output Undefined rather than raising, so a run never aborts on
    an otherwise-working DIFFPREP configuration; the 'cubic' gap reaches the
    sidecar through ``qsiprep.utils.jacobian_provenance``.
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
                'Jacobian weighting (only the quadratic terms are implemented); '
                'ec_jacobian_images will be Undefined and the eddy-current '
                'component of this run will be unmodulated.'
            )
            return runtime

        rows = _read_okan_transformations(self.inputs.transformations_file)
        ref = nb.load(self.inputs.reference_image)
        shape = ref.shape[:3]
        affine = ref.affine

        images = []
        for index, row in enumerate(rows):
            det = okan_quadratic_jacobian(row, shape, affine, self.inputs.rot_eddy_center)
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


def _jacobian_sidecar(weight_index, applied, unmodulated, reason, method):
    """Build the sidecar for the intensity weight derivative.

    ``weight_index`` is zero-based, one entry per volume of the preprocessed
    DWI series, indexing volumes of the 4D weight file. Repeated maps appear
    as repeated indices; the single-map case is written out as all zeros so
    consumers need no special case.
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
    if unmodulated and reason:
        sidecar['UnmodulatedReason'] = reason
    return sidecar


class _StackJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    # Not mandatory: weight_images is Undefined whenever ComposeJacobianWeights
    # applied no weights, which is a normal run outcome, not an error.
    weight_images = InputMultiObject(
        File(exists=True),
        desc='unique output-grid weight maps, first-appearance order',
    )
    weight_index = traits.List(traits.Int(), desc='per-volume index into weight_images')
    method = traits.Str(desc="'Jacobian' or 'LSR', from ComposeJacobianWeights")
    # Build-time facts about this run, computed by
    # qsiprep.utils.jacobian_provenance.jacobian_provenance_for and set as
    # node inputs during workflow construction.
    applied_corrections = traits.List(
        traits.Str(), usedefault=True, desc='corrections QSIPrep itself modulated'
    )
    unmodulated_corrections = traits.List(
        traits.Str(), usedefault=True, desc='corrections that ran without modulation'
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

    3D when every volume shares one map, 4D otherwise. Undefined
    ``weight_images`` (no weights were applied this run) is propagated as
    Undefined outputs rather than synthesizing a map of ones.
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
        nb.Nifti1Image(data.astype('float32'), images[0].affine, images[0].header).to_filename(
            out_file
        )

        method = self.inputs.method if isdefined(self.inputs.method) else METHOD_JACOBIAN
        self._results['out_file'] = out_file
        self._results['meta_dict'] = _jacobian_sidecar(
            weight_index=self.inputs.weight_index,
            applied=self.inputs.applied_corrections,
            unmodulated=self.inputs.unmodulated_corrections,
            reason=self.inputs.unmodulated_reason,
            method=method,
        )
        return runtime
