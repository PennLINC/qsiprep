"""Gradient nonlinearity correction.

Resolves, per correction unit, whether the images need spatial gradwarp
correction and in which dimensions, then builds the field with TORTOISE's
standalone tools.

The spatial correction and the voxelwise b-matrix (``grad_dev``) are separable:
a scanner that tags a run ``DIS3D`` has already corrected the geometry, but no
scanner can correct the diffusion encoding, because the bval/bvec table holds
one value per volume and cannot express a spatially varying encoding.
"""

import dataclasses

from ... import config

#: Ordering used to reconcile disagreeing runs in one unit. Lower is less
#: correction, and less correction is the recoverable error.
_WARP_RANK = {None: 0, '1D': 1, '3D': 2}
_RANK_TO_WARP = {rank: warp for warp, rank in _WARP_RANK.items()}


@dataclasses.dataclass(frozen=True)
class GradwarpPlan:
    """What gradient correction to apply to one correction unit.

    ``warp_dim`` is ``'3D'`` (full spatial correction), ``'1D'`` (through-plane
    residual only, for scanner-corrected DIS2D data), or ``None`` (no spatial
    correction; the grad_dev map is still produced).
    """

    coeff_file: str
    warp_dim: str | None
    is_ge: bool
    basis: str  # 'metadata' | 'forced'


def _image_type_tags(metadata):
    """Normalise ImageType, which may be a list or a backslash-joined string."""
    image_type = metadata.get('ImageType') or ()
    if isinstance(image_type, str):
        image_type = image_type.split('\\')
    return {str(tag).strip().upper() for tag in image_type}


def _warp_dim_for(metadata):
    """Residual spatial distortion implied by one run's ImageType.

    DIS3D wins over DIS2D when both are present: it is the more-corrected
    claim, so it leaves the smaller residual.
    """
    tags = _image_type_tags(metadata)
    if 'DIS3D' in tags:
        return None
    if 'DIS2D' in tags:
        return '1D'
    return '3D'


def _is_ge(metadata):
    """True when the DICOM Manufacturer field names GE.

    Free text from DICOM: 'GE MEDICAL SYSTEMS', 'SIEMENS', 'Philips Medical
    Systems'. Drives the z-origin shift TORTOISE applies for GE coefficients.
    """
    return str(metadata.get('Manufacturer', '')).strip().upper().startswith('GE')


def resolve_gradwarp_plan(unit):
    """Decide the gradient correction for one PreprocUnit, or None."""
    coeff_file = config.workflow.gradient_file
    if not coeff_file or 'gradients' in (config.workflow.ignore or []):
        return None

    records = unit.dwi_records
    is_ge = any(_is_ge(record.metadata) for record in records)

    if 'gradients' in (config.workflow.force or []):
        config.loggers.workflow.info(
            'Gradient correction: forced 3D spatial warp for %s (--force gradients).',
            unit.output_name,
        )
        return GradwarpPlan(str(coeff_file), '3D', is_ge, 'forced')

    per_file = {record.path: _warp_dim_for(record.metadata) for record in records}
    ranks = {path: _WARP_RANK[warp] for path, warp in per_file.items()}
    warp_dim = _RANK_TO_WARP[min(ranks.values())]

    if len(set(ranks.values())) > 1:
        config.loggers.workflow.warning(
            'Runs in %s disagree about scanner gradwarp correction (%s). These '
            'series are concatenated before head motion correction and share one '
            'field, so the least-correcting value (%s) is used to avoid '
            'double-correcting an already-corrected run.',
            unit.output_name,
            ', '.join(f'{path}: {warp or "none"}' for path, warp in sorted(per_file.items())),
            warp_dim or 'none',
        )
    else:
        config.loggers.workflow.info(
            'Gradient correction for %s: spatial warp %s (from ImageType).',
            unit.output_name,
            warp_dim or 'disabled',
        )

    return GradwarpPlan(str(coeff_file), warp_dim, is_ge, 'metadata')
