"""Jacobian intensity modulation for QSIPrep's spatial distortion corrections.

The weight applied to the resampled DWIs is ``|det grad phi|`` of the composed
*native-space* distortion warps -- gradient nonlinearity, susceptibility, and
TORTOISE eddy current -- and of nothing else. See
``docs/superpowers/specs/2026-09-17-jacobian-weighting-design.md`` for why head
motion, coregistration and the intramodal/template warps are excluded, and for
the derivation showing that excluding them does not move the coordinates at
which the remaining determinants are evaluated.
"""

import os

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype import logging
from nipype.interfaces import ants
from nipype.utils.filemanip import fname_presuffix

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


def validate_field_geometry(field_path, reference_path):
    """Raise unless ``field_path`` is sampled on ``reference_path``'s lattice.

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

    if field.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f'Displacement field {field_path} has spatial shape {field.shape[:3]}, '
            f'but the composition reference {reference_path} has '
            f'{reference.shape[:3]}. A field on a different lattice cannot be '
            'composed here; see the coordinate-domain table in the design spec.'
        )
    if not np.allclose(field.affine, reference.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
        raise ValueError(
            f'Displacement field {field_path} has an affine that does not match '
            f'the composition reference {reference_path}. QSIPrep deliberately '
            'requires exact agreement here rather than resampling: telling a '
            'world-compatible-but-differently-sampled field apart from one in '
            'the wrong coordinate domain needs case work no current input '
            'exercises.'
        )

    components = field.shape[4] if field.ndim == 5 else field.shape[-1]
    if components != 3:
        raise ValueError(
            f'Displacement field {field_path} has {components} vector components, '
            'expected 3.'
        )


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


def multiply_maps(paths, out_path):
    """Voxelwise product of one or more scalar maps."""
    if len(paths) == 1:
        return paths[0]
    product = nim.load_img(paths[0])
    for path in paths[1:]:
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
    """Raise unless a scalar map shares ``reference_path``'s sampling grid.

    ``validate_field_geometry`` covers displacement fields. Scalar inputs --
    the eddy-current Jacobians, the brain mask, and every map that goes into a
    product -- need the same check, or a mislatticed input is multiplied
    elementwise against the wrong voxels or silently moves which voxels the
    positivity guard inspects.
    """
    image = nb.load(image_path)
    reference = nb.load(reference_path)
    if image.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f'{image_path} has spatial shape {image.shape[:3]}, but '
            f'{reference_path} has {reference.shape[:3]}.'
        )
    if not np.allclose(image.affine, reference.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
        raise ValueError(
            f'{image_path} has an affine that does not match {reference_path}.'
        )


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
    when one exists. Head motion, coregistration and the intramodal/template
    warps are excluded by policy; see the design spec.

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

        validate_scalar_geometry(self.inputs.mask, reference)

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

        # One determinant per unique (gradwarp, fieldwarp) pair.
        determinants = {}
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
            determinants[key] = jacobian_determinant(
                composed,
                os.path.join(runtime.cwd, f'jacobian{len(determinants)}.nii.gz'),
                mask_path=self.inputs.mask,
            )

        # Per-volume weight = shared determinant x that volume's EC Jacobian.
        weights = []
        cache = {}
        for index, (fieldwarp, ec_image) in enumerate(zip(fieldwarps, ec_images, strict=True)):
            factors = []
            key = weight_key(gradwarp, fieldwarp)
            if key:
                factors.append(determinants[key])
            if ec_image:
                factors.append(ec_image)

            # Tag the roles rather than collapsing missing factors into an
            # untagged tuple: (None, 'f.nii.gz') and ('f.nii.gz', None) are
            # different weights that would otherwise share a key.
            cache_key = (gradwarp, fieldwarp, ec_image)
            if cache_key not in cache:
                cache[cache_key] = multiply_maps(
                    factors,
                    fname_presuffix(
                        self.inputs.dwi_files[index],
                        suffix=f'_jacobian-{index:05d}',
                        newpath=runtime.cwd,
                        use_ext=True,
                    ),
                )
                check_weight_map(cache[cache_key], self.inputs.mask)
            weights.append(cache[cache_key])

        self._results['jacobian_weight_images'] = weights
        return runtime
