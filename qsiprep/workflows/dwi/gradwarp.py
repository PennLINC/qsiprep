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

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradunwarp import CreateNonlinearityDisplacementMap, MaskWarpDimensions
from .resampling import _listify

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


#: Boilerplate fragments, keyed by the resolved warp dimensionality. The text
#: must track the plan: claiming 3D correction on DIS2D data would be a
#: methods-section error.
_BOILERPLATE = {
    '3D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4. The full three-dimensional gradwarp '
        'displacement field was combined with the head motion, eddy current, '
        'and susceptibility distortion transforms, so the data were resampled '
        'only once.'
    ),
    '1D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4. Because the scanner had already applied '
        'in-plane gradwarp correction (DIS2D), only the residual through-plane '
        'component was applied here, combined with the head motion, eddy '
        'current, and susceptibility distortion transforms so the data were '
        'resampled only once.'
    ),
    None: (
        'The scanner had already applied full three-dimensional gradwarp '
        'correction (DIS3D), so no further spatial correction was performed. '
        'A voxelwise gradient deviation map was computed with TORTOISE V4 to '
        'account for the spatially varying diffusion encoding.'
    ),
}


#: Report phrasing for each resolved state.
_REPORT_TEXT = {
    '3D': '3D (from ImageType)',
    '1D': 'through-plane only (ImageType: DIS2D)',
    None: 'b-matrix only (ImageType: DIS3D)',
}


def describe_gradient_correction(plan):
    """One-line description of the resolved plan, for the HTML report."""
    if plan is None:
        return 'none'
    if plan.basis == 'forced':
        return 'forced 3D'
    return _REPORT_TEXT[plan.warp_dim]


def init_gradwarp_wf(unit, name='gradwarp_wf'):
    """Build the gradwarp displacement field for one correction unit.

    Returns ``None`` when no gradient correction was requested. The field node
    runs even when ``warp_dim`` is ``None``: the grad_dev map needs a field, and
    only the wiring into the composed transform chain is suppressed.
    """
    plan = resolve_gradwarp_plan(unit)
    if plan is None:
        return None

    workflow = Workflow(name=name)
    workflow.__desc__ = _BOILERPLATE[plan.warp_dim]
    workflow.plan = plan

    inputnode = pe.Node(niu.IdentityInterface(fields=['ref_image']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['gradwarp_field']), name='outputnode')

    make_field = pe.Node(
        CreateNonlinearityDisplacementMap(coeff_file=plan.coeff_file, is_ge=plan.is_ge),
        name='make_field',
    )
    # '3D' is a passthrough, but keeping the node unconditional means the graph
    # shape does not depend on the plan.
    mask_field = pe.Node(MaskWarpDimensions(warp_dim=plan.warp_dim or '3D'), name='mask_field')

    workflow.connect([
        (inputnode, make_field, [('ref_image', 'ref_image')]),
        (make_field, mask_field, [('out_field', 'in_file')]),
        (mask_field, outputnode, [('out_file', 'gradwarp_field')]),
    ])  # fmt:skip

    return workflow


# --- Gradwarp-correcting the inputs to susceptibility estimation -------------
#
# TORTOISE resamples the b=0/FA images through the gradwarp field *before* it
# estimates the susceptibility field (``DRBUDDI::Step0_CreateImages``,
# ``EPIREG.cxx``). QSIPrep matches that exactly where the resulting SDC warp is
# applied *downstream* of gradwarp in the composed transform chain:
#
# * DRBUDDI, GRE/phase fieldmaps and SyN keep their warp in
#   ``to_dwi_ref_warps``, which ``ComposeTransforms`` applies after gradwarp --
#   so their estimation inputs must be gradwarp-corrected.
# * FSL ``eddy`` consumes TOPUP's field via ``--field`` and resamples the raw
#   data once itself, baking that field in *upstream* of ``ComposeTransforms``.
#   Estimating it on gradwarp-corrected b=0 images would place it in a space it
#   was never measured in, so TOPUP's inputs are deliberately left raw.
#
# The decision is per SDC node rather than per backend: ``--pepolar-method
# DRBUDDI+TOPUP`` runs both, and only DRBUDDI's inputs are corrected.


def connect_gradwarp_sdc_volumes(workflow, inputnode, source, source_field, drbuddi_wf):
    """Gradwarp the per-volume DWI series that DRBUDDI estimates its field from.

    ``source.source_field`` is the list of split volumes that would otherwise
    feed ``drbuddi_wf.inputnode.dwi_files``. Every volume is corrected, not just
    the b=0s, because DRBUDDI builds its FA registration target from the whole
    series.
    """
    resample = pe.MapNode(
        ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc'),
        iterfield=['input_image', 'reference_image'],
        name='gradwarp_sdc_inputs',
    )
    workflow.connect([
        (inputnode, resample, [(('gradwarp_field', _listify), 'transforms')]),
        (source, resample, [
            (source_field, 'input_image'),
            (source_field, 'reference_image'),
        ]),
        (resample, drbuddi_wf, [('output_image', 'inputnode.dwi_files')]),
    ])  # fmt:skip


def connect_gradwarp_sdc_reference(workflow, inputnode, source, source_fields, b0_sdc_wf):
    """Gradwarp the b=0 reference trio that :func:`init_sdc_wf` estimates from.

    ``source_fields`` names the reference image, its skull-stripped version and
    its mask on ``source``, in that order. The mask is resampled with nearest
    neighbours: sinc-interpolating a binary image would leave it non-binary.
    """
    ref_field, brain_field, mask_field = source_fields
    for name, source_field, dest, interpolation in (
        ('gradwarp_sdc_inputs', ref_field, 'inputnode.b0_ref', 'LanczosWindowedSinc'),
        (
            'gradwarp_sdc_inputs_brain',
            brain_field,
            'inputnode.b0_ref_brain',
            'LanczosWindowedSinc',
        ),
        ('gradwarp_sdc_inputs_mask', mask_field, 'inputnode.b0_mask', 'NearestNeighbor'),
    ):
        resample = pe.Node(
            ants.ApplyTransforms(dimension=3, interpolation=interpolation),
            name=name,
        )
        workflow.connect([
            (inputnode, resample, [(('gradwarp_field', _listify), 'transforms')]),
            (source, resample, [
                (source_field, 'input_image'),
                (source_field, 'reference_image'),
            ]),
            (resample, b0_sdc_wf, [('output_image', dest)]),
        ])  # fmt:skip
