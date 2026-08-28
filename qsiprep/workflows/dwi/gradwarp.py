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


#: Plan log lines already emitted, as rendered strings.
_LOGGED_PLAN_MESSAGES = set()


def _reset_plan_logging():
    """Forget which plan log lines have been emitted (tests only)."""
    _LOGGED_PLAN_MESSAGES.clear()


def _log_plan_once(level, message, *args):
    """Emit one plan log line, suppressing exact repeats.

    ``resolve_gradwarp_plan`` is deliberately cheap and stateless, and it is
    called three times for a single correction unit: once in ``base`` (through
    :func:`init_gradwarp_wf`), once in the selected HMC/SDC backend, and once
    in ``finalize``. Every message names ``unit.output_name``, so an exact
    repeat is always the same unit being resolved again -- and one plan
    printed three times reads like three units, while the mixed-``ImageType``
    warning printed three times reads like three separate problems.
    """
    rendered = (level, message % args)
    if rendered in _LOGGED_PLAN_MESSAGES:
        return
    _LOGGED_PLAN_MESSAGES.add(rendered)
    getattr(config.loggers.workflow, level)(message, *args)


#: Message for the GE coefficient-expansion guard. See :func:`_guard_ge_field`.
_GE_GUARD = (
    'Gradient nonlinearity correction from a coefficient file is not supported '
    'for GE data (%s).\n\n'
    "TORTOISE's own pipeline applies a z-origin shift to the displacement field "
    'after expanding the coefficients, for GE data only (TORTOISE.cxx, the '
    '``if(is_GE)`` block following ``mk_displacement``). That shift lives in '
    'TORTOISEProcess, not in the standalone CreateNonlinearityDisplacementMap '
    'binary qsiprep calls, whose main() writes the field with the reference '
    "image's own origin. qsiprep would therefore place the field at a different "
    'z than TORTOISE does, and the correct sign has not been verified against a '
    'TORTOISE run on GE data.\n\n'
    'Options:\n'
    '  * pass --gradient-file a ready-made ITK displacement field (.nii/.nii.gz) '
    'instead of coefficients; qsiprep uses it as given and expands nothing, so '
    'this does not apply.\n'
    '  * pass --ignore gradients to skip gradient correction entirely.\n\n'
    'Siemens and Philips data are unaffected: the shift is applied only when '
    'the Manufacturer field names GE.'
)


def _guard_ge_field(plan, unit):
    """Refuse to expand GE coefficients into a field we cannot place correctly.

    Scoped as narrowly as the defect is. It fires only when all three hold:

    * the scanner is GE -- the shift is applied nowhere else;
    * a spatial field is actually built -- a ``DIS3D`` unit builds none, and
      ``CreateGradientNonlinearityBMatrix`` does its own, self-contained GE
      recentring, so ``grad_dev`` is unaffected either way;
    * the field is expanded from coefficients rather than supplied whole.
    """
    if not plan.is_ge or plan.warp_dim is None:
        return plan
    if is_displacement_field(plan.coeff_file):
        return plan
    raise ValueError(_GE_GUARD % unit.output_name)


def resolve_gradwarp_plan(unit):
    """Decide the gradient correction for one PreprocUnit, or None."""
    coeff_file = config.workflow.gradient_file
    if not coeff_file or 'gradients' in (config.workflow.ignore or []):
        return None

    records = unit.dwi_records
    is_ge = any(_is_ge(record.metadata) for record in records)

    if 'gradients' in (config.workflow.force or []):
        plan = _guard_ge_field(GradwarpPlan(str(coeff_file), '3D', is_ge, 'forced'), unit)
        _log_plan_once(
            'info',
            'Gradient correction: forced 3D spatial warp for %s (--force gradients).',
            unit.output_name,
        )
        return plan

    per_file = {record.path: _warp_dim_for(record.metadata) for record in records}
    ranks = {path: _WARP_RANK[warp] for path, warp in per_file.items()}
    warp_dim = _RANK_TO_WARP[min(ranks.values())]
    plan = _guard_ge_field(GradwarpPlan(str(coeff_file), warp_dim, is_ge, 'metadata'), unit)

    if len(set(ranks.values())) > 1:
        _log_plan_once(
            'warning',
            'Runs in %s disagree about scanner gradwarp correction (%s). These '
            'series are concatenated before head motion correction and share one '
            'field, so the least-correcting value (%s) is used to avoid '
            'double-correcting an already-corrected run.',
            unit.output_name,
            ', '.join(f'{path}: {warp or "none"}' for path, warp in sorted(per_file.items())),
            warp_dim or 'none',
        )
    else:
        _log_plan_once(
            'info',
            'Gradient correction for %s: spatial warp %s (from ImageType).',
            unit.output_name,
            warp_dim or 'disabled',
        )

    return plan


#: HMC backends that write out motion- and eddy-corrected volumes *before*
#: qsiprep's final resampling. On these the composed chain carries no head
#: motion or eddy current transform at all -- ``GatherEddyInputs`` emits an
#: empty ``forward_transforms`` (``interfaces/eddy.py:190``) and
#: ``DIFFPREPSplitOutputs`` emits per-volume identities
#: (``interfaces/tortoise.py:1275``) -- so the series has already been resampled
#: once by the time gradwarp is applied, and boilerplate claiming a single
#: raw-to-final interpolation would be a methods-section error.
_PRERESAMPLED_BY = {'eddy': 'FSL eddy', 'tortoise': "TORTOISE's DIFFPREP"}


#: What was corrected, keyed by the resolved warp dimensionality. The text must
#: track the plan: claiming 3D correction on DIS2D data would be a
#: methods-section error.
_CORRECTION_TEXT = {
    '3D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4, applying the full three-dimensional '
        'gradwarp displacement field.'
    ),
    '1D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4. Because the scanner had already applied '
        'in-plane gradwarp correction (DIS2D), only the residual through-plane '
        'component was applied here.'
    ),
    None: (
        'The scanner had already applied full three-dimensional gradwarp '
        'correction (DIS3D), so no further spatial correction was performed. '
        'A voxelwise gradient deviation map was computed with TORTOISE V4 to '
        'account for the spatially varying diffusion encoding.'
    ),
}


def _resampling_sentence():
    """How many times the data were interpolated, which depends on the backend.

    Notes
    -----
    This reads the legacy ``hmc_model`` key off ``config.workflow`` rather than
    the compiled plan, and is allowlisted as such in
    ``test_workflows_native.test_legacy_method_keys_read_only_at_allowlisted_sites``.
    It is display vocabulary, not routing -- the same category as the other
    allowlisted sites -- but it works only because ``eddy`` and ``tortoise`` are
    spelled identically in the legacy key and in :class:`~qsiplan.methods.HmcMethod`.
    The plan-native question is ``unit.run.hmc_stage.tool``; asking it would mean
    threading the run through :func:`gradwarp_boilerplate` and its callers, which
    is worth doing if this ever needs to distinguish a backend the two
    vocabularies spell differently (SHORELine is ``3dSHORE``/``tensor`` here and
    ``shoreline`` there).
    """
    backend = _PRERESAMPLED_BY.get(config.workflow.hmc_model)
    if backend is None:
        return (
            ' The displacement field was combined with the head motion, eddy '
            'current, and susceptibility distortion transforms, so the data '
            'were resampled only once.'
        )
    return (
        ' The displacement field was combined with the remaining susceptibility '
        'distortion and coregistration transforms and applied in a single '
        f'resampling, following the motion and eddy current correction {backend} '
        'had already applied.'
    )


def gradwarp_boilerplate(warp_dim):
    """Methods text for the resolved plan and the selected HMC backend.

    A ``DIS3D`` unit gets no displacement field, so it gets no resampling
    sentence either -- there is nothing to have been combined with anything.
    """
    if warp_dim is None:
        return _CORRECTION_TEXT[None]
    return _CORRECTION_TEXT[warp_dim] + _resampling_sentence()


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


def is_displacement_field(gradient_file):
    """True when ``--gradient-file`` is a ready-made ITK field, not coefficients.

    TORTOISE dispatches on the extension itself (``TORTOISE.cxx:1943-2023``),
    but the standalone ``CreateNonlinearityDisplacementMap`` does not: it *is*
    the coefficient expander (``mk_displacement(argv[1], img, is_GE)``), so
    handing it a binary NIfTI feeds a text parser, which either throws or
    yields zero coefficients and an all-zero field.
    """
    return str(gradient_file).endswith(('.nii', '.nii.gz'))


def init_gradwarp_wf(unit, name='gradwarp_wf'):
    """Build the gradwarp displacement field for one correction unit.

    Returns ``None`` when no gradient correction was requested. Otherwise the
    returned workflow always carries the resolved ``.plan`` and the methods
    boilerplate that goes with it, but it only builds the nodes whose outputs
    are actually consumed:

    * ``plan.warp_dim is None`` (the scanner already applied full 3D gradwarp,
      ``DIS3D``): no nodes at all. Nothing resamples through a field for such a
      unit, and ``finalize``'s ``grad_dev`` node is fed the *coefficient* file
      rather than a field, so building one here would invoke an external
      binary per unit and discard both of its outputs. The workflow still
      exists so callers keep the plan, and so the ``DIS3D`` boilerplate still
      reaches the methods section.
    * a ``.nii``/``.nii.gz`` ``--gradient-file``: no ``make_field``. The user
      supplied the displacement field, so only the dimension masking applies.

    ``.needs_reference`` says whether ``inputnode.ref_image`` is consumed, so a
    caller knows whether to build the 3D extraction node that feeds it.
    """
    plan = resolve_gradwarp_plan(unit)
    if plan is None:
        return None

    workflow = Workflow(name=name)
    workflow.__desc__ = gradwarp_boilerplate(plan.warp_dim)
    workflow.plan = plan
    workflow.needs_reference = False

    if plan.warp_dim is None:
        return workflow

    outputnode = pe.Node(niu.IdentityInterface(fields=['gradwarp_field']), name='outputnode')
    mask_field = pe.Node(MaskWarpDimensions(warp_dim=plan.warp_dim), name='mask_field')

    if is_displacement_field(plan.coeff_file):
        mask_field.inputs.in_file = plan.coeff_file
    else:
        workflow.needs_reference = True
        inputnode = pe.Node(niu.IdentityInterface(fields=['ref_image']), name='inputnode')
        make_field = pe.Node(
            CreateNonlinearityDisplacementMap(coeff_file=plan.coeff_file, is_ge=plan.is_ge),
            name='make_field',
        )
        workflow.connect([
            (inputnode, make_field, [('ref_image', 'ref_image')]),
            (make_field, mask_field, [('out_field', 'in_file')]),
        ])  # fmt:skip

    workflow.connect([
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


def _sdc_interpolation():
    """Interpolator for the gradwarp resampling nodes.

    Matches the adjacent per-volume ``ApplyTransforms`` in ``hmc_sdc.py``, so
    ``--sloppy`` speeds these up the same way it speeds up everything else.
    """
    return 'NearestNeighbor' if config.execution.sloppy else 'LanczosWindowedSinc'


def connect_gradwarp_sdc_volumes(workflow, inputnode, source, source_field, drbuddi_wf):
    """Gradwarp the per-volume DWI series that DRBUDDI estimates its field from.

    ``source.source_field`` is the list of split volumes that would otherwise
    feed ``drbuddi_wf.inputnode.dwi_files``. Every volume is corrected, not just
    the b=0s, because DRBUDDI builds its FA registration target from the whole
    series.

    ``float=True`` matches every adjacent resampling node in the codebase
    (``resampling.py``, ``diffprep.py``): this is a MapNode over the whole
    series, so double-precision output would double the working memory of the
    heaviest per-volume step in the pipeline.
    """
    resample = pe.MapNode(
        ants.ApplyTransforms(
            dimension=3,
            interpolation=_sdc_interpolation(),
            float=True,
        ),
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
    smooth = _sdc_interpolation()
    for name, source_field, dest, interpolation in (
        ('gradwarp_sdc_inputs', ref_field, 'inputnode.b0_ref', smooth),
        ('gradwarp_sdc_inputs_brain', brain_field, 'inputnode.b0_ref_brain', smooth),
        ('gradwarp_sdc_inputs_mask', mask_field, 'inputnode.b0_mask', 'NearestNeighbor'),
    ):
        resample = pe.Node(
            ants.ApplyTransforms(dimension=3, interpolation=interpolation, float=True),
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


# --- Gradwarp-correcting the DWI/T1w coregistration reference ----------------
#
# ``ComposeTransforms`` applies the b0->T1w affine *after* gradwarp, so the
# image that affine was estimated from should itself be gradwarp-corrected.
# ``b0_template`` -- the image ``init_b0_to_anat_registration_wf`` registers --
# gets that for free on the branches whose SDC estimation inputs were corrected
# above (DRBUDDI, GRE/SyN), because those branches forward the SDC workflow's
# own reference. The branches that forward an uncorrected reference instead
# need this, so the geometry is the same whichever backend ran.
#
# Unlike the SDC rule, this one has no TOPUP carve-out. That carve-out is about
# where TOPUP's *field* is estimated, and eddy applies that field to raw data.
# The coregistration reference is derived from eddy's *output*, so correcting
# it does not touch TOPUP at all.


def connect_gradwarp_coreg_reference(
    workflow,
    inputnode,
    source,
    source_field,
    outputnode,
    name='gradwarp_coreg_ref',
):
    """Gradwarp the b=0 that DWI/T1w coregistration is estimated from.

    For branches whose reference carries no SDC warp either. DIFFPREP's T2Wreg
    branch needs both and folds them into its existing ``apply_sdc_to_b0``
    node instead, so that the mask ``b0_ref_for_coreg`` derives stays in the
    same geometry as the reference.
    """
    resample = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation=_sdc_interpolation(), float=True),
        name=name,
    )
    workflow.connect([
        (inputnode, resample, [(('gradwarp_field', _listify), 'transforms')]),
        (source, resample, [
            (source_field, 'input_image'),
            (source_field, 'reference_image'),
        ]),
        (resample, outputnode, [('output_image', 'b0_template')]),
    ])  # fmt:skip
    return resample
