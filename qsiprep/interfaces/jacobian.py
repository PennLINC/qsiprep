"""Interfaces for QSIPrep's intensity modulation of spatial distortion corrections.

The computations, and the TORTOISE conventions they follow, are in
:mod:`qsiprep.utils.jacobian`.
"""

import os

import nibabel as nb
import numpy as np
from nipype import logging
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

from ..utils.jacobian import (
    METHOD_JACOBIAN,
    METHOD_LSR,
    ROT_EDDY_CENTERS,
    check_weight_map,
    compose_fields,
    float32_header,
    jacobian_determinant,
    jacobian_sidecar,
    multiply_maps,
    okan_quadratic_jacobian,
    resample_like,
    transport_scalar_map,
    validate_field_geometry,
    validate_scalar_geometry,
    volume_mean,
    weight_key,
)
from .tortoise import _read_okan_transformations

LOGGER = logging.getLogger('nipype.interface')


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
        if isdefined(self.inputs.fieldwarps) and self.inputs.fieldwarps:
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
            nb.Nifti1Image(det.astype('float32'), affine, float32_header(ref)).to_filename(
                out_path
            )
            images.append(out_path)

        self._results['ec_jacobian_images'] = images
        return runtime


class _StackJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    # Not mandatory: weight_images is Undefined whenever ComposeJacobianWeights
    # applied no weights, which is a normal run outcome, not an error.
    weight_images = InputMultiObject(
        File(exists=True),
        desc='unique output-grid weight maps, first-appearance order',
    )
    weight_index = traits.List(traits.Int(), desc='per-volume index into weight_images')
    volumes = traits.Bool(
        False,
        usedefault=True,
        desc='write every distinct map (4D) instead of their mean over volumes (3D)',
    )
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
    out_file = File(
        desc='3D unless the maps differ and volumes is set; Undefined if no weights were applied'
    )
    meta_dict = traits.Dict(desc='sidecar for the derivative')


class StackJacobianWeights(SimpleInterface):
    """Write the weight maps as one derivative and build its sidecar.

    When every volume shares one map, that map is written (3D). When the maps
    differ between volumes, as with eddy-current correction, the default is
    their mean over volumes (3D), each map counted once per volume that used
    it: the per-volume differences are small next to the spatial variation, and
    a 4D map is as large as the DWI series. ``volumes`` writes every distinct
    map instead (4D). Undefined ``weight_images`` (no weights were applied this
    run) is propagated as Undefined outputs rather than synthesizing a map of
    ones.
    """

    input_spec = _StackJacobianWeightsInputSpec
    output_spec = _StackJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.weight_images):
            return runtime

        images = [nb.load(path) for path in self.inputs.weight_images]
        out_file = os.path.join(runtime.cwd, 'jacobian_weights.nii.gz')
        weight_index = (
            list(self.inputs.weight_index) if isdefined(self.inputs.weight_index) else []
        )
        deviation = None
        if len(images) == 1:
            data = np.asanyarray(images[0].dataobj, dtype=np.float32)
        elif self.inputs.volumes:
            data = np.stack([np.asanyarray(img.dataobj, dtype=np.float32) for img in images], -1)
        else:
            data, deviation = volume_mean(images, weight_index)
            weight_index = [0] * len(weight_index)
        nb.Nifti1Image(data, images[0].affine, float32_header(images[0])).to_filename(out_file)

        method = self.inputs.method if isdefined(self.inputs.method) else METHOD_JACOBIAN
        self._results['out_file'] = out_file
        self._results['meta_dict'] = jacobian_sidecar(
            weight_index=weight_index,
            applied=self.inputs.applied_corrections,
            unmodulated=self.inputs.unmodulated_corrections,
            reason=self.inputs.unmodulated_reason,
            method=method,
            volume_deviation=deviation,
        )
        return runtime
