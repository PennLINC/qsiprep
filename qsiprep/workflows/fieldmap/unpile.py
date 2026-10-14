# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Fieldmap-less susceptibility distortion correction with unpile.

Fieldmap-less susceptibility distortion correction with unpile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`unpile <https://unpile.readthedocs.io>`__ estimates the susceptibility field from the
motion-corrected b=0 and an undistorted reference with a model of the EPI readout: the
reference is moved and stretched along the phase-encoding axis the way the readout does,
pile-up included, and the field that makes it match the b=0 is fitted. The reference is a
SynB0 synthetic b=0 (same contrast; :ref:`sdc_synb0`) or the subject's T2w, whose position
is fitted together with the field. A GRE fieldmap that also lists the series is handed to
unpile as a prior and refined rather than replaced. The field is applied in the final
resampling, after any head motion method.
"""

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.fmap import ModulateByWarpJacobian
from ...interfaces.unpile import (
    FINEST_GRID_MM,
    PrepareUnpileInputs,
    UnpileFieldToWarp,
    UnpileFit,
    unpile_fit_grid_mm,
)
from ...utils.sdc import pe_readout_time
from ..dwi.registration import init_structural_to_b0_alignment_wf
from ..dwi.util import init_dwi_reference_wf
from .synb0 import init_synb0_wf

#: unpile's bending-energy penalty in displacement units, per reference contrast.
LAM_MM = {'synb0': 5.0, 't2w': 60.0}
#: The multi-resolution schedule (control-grid spacings in mm) and iterations under --sloppy:
#: the 3 mm level dropped, which on the 3 mm fixtures costs nothing.
SLOPPY_LEVELS = '24,12,6'
SLOPPY_ITERS = 40

__all__ = ['init_unpile_sdc_wf', 'unpile_fit_grid_mm', 'unpile_fit_settings', 'FINEST_GRID_MM']


def unpile_fit_settings(target, sloppy=False):
    """Return the :class:`~qsiprep.interfaces.unpile.UnpileFit` inputs for ``target``.

    A SynB0 reference has the b=0's contrast, so the sum of squared differences with a
    fitted gain is the data term; a T2w is compared by local normalized cross-correlation
    and its rigid position is fitted with the field. ``--sloppy`` runs a shorter schedule.
    """
    settings = {'metric': 'lncc' if target == 't2w' else 'ssd', 'lam_mm': LAM_MM[target]}
    if target == 't2w':
        settings['pose'] = True
    if sloppy:
        settings['levels'] = SLOPPY_LEVELS
        settings['iters'] = SLOPPY_ITERS
    return settings


def _lead_dwi(unit):
    return unit.plus_files[0] if unit.has_bidirectional_dwi else unit.dwi_files[0]


def init_unpile_sdc_wf(
    unit,
    source_file=None,
    target='synb0',
    two_pass=True,
    gre_prior=False,
    name='unpile_sdc_wf',
):
    """Estimate susceptibility distortion with unpile from a motion-corrected b=0.

    With ``target='synb0'`` a synthetic distortion-free b=0 is generated from the b=0 and
    the T1w and serves as the reference. The U-Net carries part of its input's distortion
    into the synthetic b=0, so with ``two_pass`` the b=0 is unwarped with the first field
    (Jacobian-modulated), a new synthetic b=0 is generated from it, and unpile fits the
    original b=0 against the new synthetic one. With ``target='t2w'`` the subject's T2w is
    placed in the b=0 frame by a rigid registration (an ``antsAI`` rotation search, then a
    masked rigid ``antsRegistration``) and unpile fits its position together with the
    field, on a grid finer than the EPI's (:func:`unpile_fit_grid_mm`) so that the T2w
    keeps its detail; ``--sloppy`` shortens the schedule but keeps that grid, since the
    native grid loses a third of the field. With ``gre_prior`` the ``fieldmap_hz`` input,
    a GRE field on the b=0 grid, is the prior of the fit: against the T2w its position is
    refined; against the synthetic b=0 it is applied as given, since the synthetic
    image's own residual distortion pulls any refinement off the measured field.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep_docs import example_unit
        from qsiprep.workflows.fieldmap.unpile import init_unpile_sdc_wf
        wf = init_unpile_sdc_wf(example_unit('single'), target='synb0')

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The DWI series to correct; its lead series' sidecar gives the phase-encoding
        direction and readout time
    source_file : str, optional
        Naming source for the derivatives; none are written without it
    target : {'synb0', 't2w'}
        The reference unpile fits the b=0 against
    two_pass : bool
        Regenerate the synthetic b=0 from the unwarped b=0 and fit again (``synb0`` only)
    gre_prior : bool
        Whether ``inputnode.fieldmap_hz`` carries a GRE field to refine

    Inputs
    ------
    b0_ref
        Motion-corrected, distorted b=0 average (head), native grid
    b0_ref_brain
        Skull-stripped ``b0_ref``
    b0_mask
        Brain mask of ``b0_ref``
    t2w_unfatsat
        The subject's T2w (``target='t2w'``)
    fieldmap_hz
        GRE fieldmap in Hz on the ``b0_ref`` grid (``gre_prior``)
    t1_preproc, t1_brain, t1_seg, to_template_affine_transform, acpc_inv_transform,
    dwi_sampling_grid
        As for :func:`~qsiprep.workflows.fieldmap.synb0.init_synb0_wf` (``target='synb0'``)

    Outputs
    -------
    out_warp
        ITK displacement field on the ``b0_ref`` grid
    out_hz
        The fitted field (Hz) on the ``b0_ref`` grid
    out_reference
        ``b0_ref`` unwarped with ``out_warp``
    pose_xfm
        The fitted position of the T2w as an ITK affine (``target='t2w'``)
    score_json
        unpile's record of the fit
    synthetic_b0, synthetic_b0_acpc, qc_file, acquired_synthetic_report, unet_input_report
        Passed through from the last pass's :func:`init_synb0_wf` (``target='synb0'``)
    """
    if target not in LAM_MM:
        raise ValueError(f"unpile target must be 'synb0' or 't2w', not {target!r}")
    raw_dwi_file = _lead_dwi(unit)
    readout_time = pe_readout_time(unit)
    if readout_time is None or not unit.pe_dir:
        raise ValueError(
            'unpile distortion correction needs the PhaseEncodingDirection and '
            f'TotalReadoutTime of {raw_dwi_file}.'
        )
    sloppy = config.execution.sloppy
    two_pass = two_pass and target == 'synb0'

    workflow = Workflow(name=name)
    workflow.__desc__ = _boilerplate(target, two_pass, gre_prior)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_ref',
                'b0_ref_brain',
                'b0_mask',
                't2w_unfatsat',
                'fieldmap_hz',
                't1_preproc',
                't1_brain',
                't1_seg',
                'to_template_affine_transform',
                'acpc_inv_transform',
                'dwi_sampling_grid',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_warp',
                'out_hz',
                'out_reference',
                'pose_xfm',
                'score_json',
                'synthetic_b0',
                'synthetic_b0_acpc',
                'qc_file',
                'acquired_synthetic_report',
                'unet_input_report',
            ]
        ),
        name='outputnode',
    )

    fit_settings = unpile_fit_settings(target, sloppy)
    if gre_prior:
        # Against the T2w, a rigid pose refines the fieldmap's position. The shim
        # polynomial and the smooth residual are left off: on the phasediff fixture
        # (fieldmap registered to 0.4 deg) each lowered the field's amplitude below
        # the fieldmap as given (shim 0.83 of the true slope against the T2w, 0.84
        # against the SynB0; residual 0.65-0.75; as given 0.98). The synthetic b=0
        # carries part of the distortion it is meant to undo, so against it the
        # fieldmap is applied as given.
        fit_settings['fieldmap_refine'] = 'pose' if target == 't2w' else 'none'

    def _add_pass(suffix, reference, epi=(inputnode, 'b0_ref')):
        """Add prepare -> fit -> field-to-warp for one reference; return the three nodes."""
        prepare = pe.Node(
            PrepareUnpileInputs(
                raw_dwi_file=raw_dwi_file, pe_dir=unit.pe_dir, fine_grid=target == 't2w'
            ),
            name=f'prepare_unpile{suffix}',
        )
        fit = pe.Node(
            UnpileFit(
                total_readout_time=readout_time,
                threads=config.nipype.omp_nthreads,
                **fit_settings,
            ),
            name=f'unpile_fit{suffix}',
            n_procs=config.nipype.omp_nthreads,
            mem_gb=4,
        )
        field_to_warp = pe.Node(
            UnpileFieldToWarp(
                raw_dwi_file=raw_dwi_file, pe_dir=unit.pe_dir, readout_time=readout_time
            ),
            name=f'unpile_field_to_warp{suffix}',
        )
        workflow.connect([
            (epi[0], prepare, [(epi[1], 'epi')]),
            (inputnode, prepare, [('b0_mask', 'mask')]),
            (reference[0], prepare, [(reference[1], 'reference')]),
            (prepare, fit, [
                ('epi', 'epi'),
                ('mask', 'mask'),
                ('reference', 'reference'),
                ('pe_dir', 'pe_dir'),
                ('acq_voxel_mm', 'acq_voxel_mm'),
            ]),
            (fit, field_to_warp, [('field_hz', 'field_hz')]),
            (inputnode, field_to_warp, [('b0_ref', 'reference')]),
        ])  # fmt:skip
        if gre_prior:
            workflow.connect([
                (inputnode, prepare, [('fieldmap_hz', 'fieldmap')]),
                (prepare, fit, [('fieldmap', 'fieldmap')]),
            ])  # fmt:skip
        return prepare, fit, field_to_warp

    if target == 'synb0':
        synb0_wf = init_synb0_wf(name='synb0_wf')
        workflow.connect([
            (inputnode, synb0_wf, [
                ('t1_preproc', 'inputnode.t1_preproc'),
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_seg', 'inputnode.t1_seg'),
                ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
                ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
                ('dwi_sampling_grid', 'inputnode.output_grid'),
                ('b0_ref', 'inputnode.b0_ref'),
                ('b0_ref_brain', 'inputnode.b0_ref_brain'),
            ]),
        ])  # fmt:skip
        _prepare, fit, field_to_warp = _add_pass('', (synb0_wf, 'outputnode.synthetic_b0'))
        if two_pass:
            # Unwarp the b=0 with the first field, modulate it, and synthesize again from
            # it; the second fit still matches the ORIGINAL b=0 to the new synthetic one.
            unwarp_pass1 = pe.Node(
                ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc', float=True),
                name='unwarp_b0_pass1',
            )
            modulate = pe.Node(ModulateByWarpJacobian(), name='modulate_unwarped_b0')
            b0_ref_wf_pass2 = init_dwi_reference_wf(
                gen_report=False,
                desc='b0_for_synb0',
                name='synb0_b0_ref_wf_pass2',
                source_file=source_file,
            )
            synb0_wf_pass2 = init_synb0_wf(name='synb0_wf_pass2')
            workflow.connect([
                (inputnode, unwarp_pass1, [
                    ('b0_ref', 'input_image'),
                    ('b0_ref', 'reference_image'),
                ]),
                (field_to_warp, unwarp_pass1, [(('out_warp', _as_list), 'transforms')]),
                (unwarp_pass1, modulate, [('output_image', 'in_file')]),
                (field_to_warp, modulate, [('out_warp', 'warp'), ('pe_axis', 'pe_axis')]),
                (modulate, b0_ref_wf_pass2, [('out_file', 'inputnode.b0_template')]),
                (inputnode, synb0_wf_pass2, [
                    ('t1_preproc', 'inputnode.t1_preproc'),
                    ('t1_brain', 'inputnode.t1_brain'),
                    ('t1_seg', 'inputnode.t1_seg'),
                    ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
                    ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
                    ('dwi_sampling_grid', 'inputnode.output_grid'),
                ]),
                (b0_ref_wf_pass2, synb0_wf_pass2, [
                    ('outputnode.ref_image', 'inputnode.b0_ref'),
                    ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                ]),
            ])  # fmt:skip
            synb0_wf = synb0_wf_pass2
            _prepare, fit, field_to_warp = _add_pass(
                '_pass2', (synb0_wf, 'outputnode.synthetic_b0')
            )
        workflow.connect([
            (synb0_wf, outputnode, [
                ('outputnode.synthetic_b0', 'synthetic_b0'),
                ('outputnode.synthetic_b0_acpc', 'synthetic_b0_acpc'),
                ('outputnode.qc_file', 'qc_file'),
                ('outputnode.acquired_synthetic_report', 'acquired_synthetic_report'),
                ('outputnode.unet_input_report', 'unet_input_report'),
            ]),
        ])  # fmt:skip
    else:
        # The T2w is placed in the b=0 frame first: unpile's joint pose refines a start
        # within its capture range, it does not search for one.
        t2w_to_b0_wf = init_structural_to_b0_alignment_wf(name='t2w_to_b0_wf', masked=True)
        workflow.connect([
            (inputnode, t2w_to_b0_wf, [
                ('t2w_unfatsat', 'inputnode.structural_image'),
                ('b0_ref', 'inputnode.b0_ref'),
                ('b0_mask', 'inputnode.b0_mask'),
            ]),
        ])  # fmt:skip
        _prepare, fit, field_to_warp = _add_pass(
            '', (t2w_to_b0_wf, 'outputnode.structural_aligned')
        )
        workflow.connect([(fit, outputnode, [('pose_xfm', 'pose_xfm')])])  # fmt:skip

    unwarp_ref = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc', float=True),
        name='unwarp_ref',
    )
    workflow.connect([
        (inputnode, unwarp_ref, [
            ('b0_ref', 'input_image'),
            ('b0_ref', 'reference_image'),
        ]),
        (field_to_warp, unwarp_ref, [(('out_warp', _as_list), 'transforms')]),
        (field_to_warp, outputnode, [
            ('out_warp', 'out_warp'),
            ('out_hz', 'out_hz'),
        ]),
        (unwarp_ref, outputnode, [('output_image', 'out_reference')]),
        (fit, outputnode, [('score_json', 'score_json')]),
    ])  # fmt:skip

    if source_file is not None:
        from ...interfaces import DerivativesDataSink

        ds_score = pe.Node(
            DerivativesDataSink(
                source_file=source_file, desc='unpile', suffix='qc', extension='.json'
            ),
            name='ds_unpile_score',
            run_without_submitting=True,
            mem_gb=0.01,
        )
        workflow.connect([(fit, ds_score, [('score_json', 'in_file')])])  # fmt:skip

    return workflow


def _as_list(value):
    return [value]


def _boilerplate(target, two_pass, gre_prior):
    reference = (
        'a synthetic distortion-free b=0 generated from it and the T1w with SynB0-DISCO '
        '[@synb0disco]'
        if target == 'synb0'
        else "the subject's T2w image, placed in the b=0 frame by a rigid registration"
    )
    metric = (
        'the sum of squared differences'
        if target == 'synb0'
        else 'local normalized cross-correlation, fitting a rigid refinement of the T2w '
        'position together with the field'
    )
    desc = (
        'Susceptibility distortion was estimated with unpile [@unpile] after head motion '
        f'correction: the motion-corrected b=0 average was matched to {reference} by '
        f'{metric}, through a model of the EPI readout that displaces the reference along '
        'the phase-encoding axis and accounts for signal pile-up.'
    )
    if target == 't2w':
        desc += (
            ' The fit was carried out on a grid at about two thirds of the DWI voxel (no finer '
            f'than {FINEST_GRID_MM:g} mm) to keep the detail of the T2w.'
        )
    if gre_prior:
        desc += (
            ' A GRE fieldmap acquired for the series was handed to unpile as a prior, with '
            'its position refined against the b=0.'
            if target == 't2w'
            else ' A GRE fieldmap acquired for the series was handed to unpile and applied '
            'as given, since refining it against a synthetic b=0 pulls it toward the '
            "synthetic image's own residual distortion."
        )
    if two_pass:
        desc += (
            ' The b=0 was then unwarped with that field (Jacobian-modulated), a new synthetic '
            'b=0 was generated from it, and the field was fitted again against the new '
            'synthetic b=0.'
        )
    desc += (
        ' The resulting field was converted to a displacement field along the '
        'phase-encoding axis and applied in the final resampling.\n'
    )
    return desc
