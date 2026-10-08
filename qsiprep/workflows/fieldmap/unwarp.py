# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for unwarping susceptibility distortions with a fieldmap.

.. _sdc_unwarp :

Unwarping
~~~~~~~~~

.. topic :: Abbreviations

    fmap
        fieldmap
    VSM
        voxel-shift map -- a 3D nifti where displacements are in pixels (not mm)
    DFM
        displacements field map -- a nifti warp file compatible with ANTs (mm)

"""

import os

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nibabel import ApplyMask, FilledImageLike
from niworkflows.interfaces.reportlets.registration import ANTSApplyTransformsRPT

from ... import config
from ...data import load as load_data
from ...interfaces import DerivativesDataSink
from ...interfaces.fmap import FieldmapToVSM, FieldToRadS
from ...interfaces.fmap import get_ees as _get_ees
from ...interfaces.itk import GuardRefinedTransform
from ...interfaces.niworkflows import FUGUEvsm2ANTSwarp


def init_sdc_unwarp_wf(name='sdc_unwarp_wf'):
    """Build a workflow that converts a fieldmap into an ANTs-compatible warp.

    This workflow takes in a displacements fieldmap and calculates the corresponding
    displacements field (in other words, an ANTs-compatible warp file).

    The fieldmap's magnitude image is registered to the EPI reference twice. The first
    pass registers it to the distorted reference; the field it brings in unwarps the
    reference, and the second pass registers the magnitude to that unwarped reference,
    starting from the first transform. Each pass is scored by the registration metric on
    its own unwarped reference and the better one is used. See
    :func:`init_fmap_apply_wf` for the field-to-warp conversion.

    It also calculates a new mask for the input dataset that takes into account the distortions.
    The mask is restricted to the field of view of the fieldmap since outside of it corrections
    could not be performed.

    Inputs
    ------
    in_reference
        the reference image
    in_reference_brain
        the reference image (skull-stripped)
    in_mask
        a brain mask corresponding to ``in_reference``
    metadata
        metadata associated to the ``in_reference`` EPI input
    fmap
        the fieldmap in Hz
    fmap_ref
        the reference (anatomical) image corresponding to ``fmap``
    fmap_mask
        a brain mask corresponding to ``fmap``


    Outputs
    -------
    out_reference
        the ``in_reference`` after unwarping
    out_reference_brain
        the ``in_reference`` after unwarping and skullstripping
    out_warp
        the corresponding :abbr:`DFM (displacements field map)` compatible with
        ANTs
    out_mask
        mask of the unwarped input file
    out_hz
        the fieldmap in Hz on the ``in_reference`` grid (eddy's ``--field``)

    """
    omp_nthreads = config.nipype.omp_nthreads
    fsl_check = os.environ.get('FSLDIR', False)
    if not fsl_check:
        raise Exception(
            """Container in use does not have FSL. To use this workflow,
            please download the qsiprep container with FSL installed."""
        )
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_reference',
                'in_reference_brain',
                'in_mask',
                'metadata',
                'fmap_ref',
                'fmap_mask',
                'fmap',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_reference',
                'out_reference_brain',
                'out_warp',
                'out_mask',
                'out_hz',
            ]
        ),
        name='outputnode',
    )

    # Register the reference of the fieldmap to the reference of the target image (the one
    # that shall be corrected). Whole-head images with the brain masks as *metric* masks and no
    # histogram matching: on the TRXScan phasediff fixture (truth = the fieldmap's offset, a
    # realistic magnitude with scalp, receive bias and ringing) the cropped images with
    # histogram matching landed 4.6 deg off; matching off alone 2.6 deg; the masks 0.5-0.8 deg.
    # The biased, flat-brain magnitude and the EPI b0 have unrelated histograms, and matching
    # them hands the mutual information a wrong intensity correspondence.
    ants_settings = str(load_data('fmap-any_registration.json'))
    if config.execution.sloppy:
        ants_settings = str(load_data('fmap-any_registration_testing.json'))
    fmap2ref_reg = pe.Node(
        ants.Registration(from_file=ants_settings, output_warped_image=True),
        name='fmap2ref_reg',
        n_procs=omp_nthreads,
    )
    fmap_apply_pass1_wf = init_fmap_apply_wf(name='fmap_apply_pass1_wf', generate_report=True)

    # A rigid fit of an undistorted magnitude to a distorted b=0 splits the difference between
    # the stretched and the compressed side: a shift along the phase-encoding axis. The second
    # pass registers to the reference unwarped with the first field, starting from the first
    # transform. The EPI mask is reused as is; dilating it to cover tissue the unwarping moved
    # made the fit worse.
    fmap2ref_reg2 = pe.Node(
        ants.Registration(from_file=ants_settings, output_warped_image=True),
        name='fmap2ref_reg2',
        n_procs=omp_nthreads,
    )
    fmap_apply_pass2_wf = init_fmap_apply_wf(name='fmap_apply_pass2_wf', generate_report=True)

    # Either pass can fail (random metric sampling): the first can land far off and the second
    # recover, or the second can run away from a good first. Each is scored by the registration
    # metric on its own target, the reference unwarped with the field in that pose, and the
    # better one is used downstream.
    similarity = {
        'dimension': 3,
        'metric': 'MI',
        'metric_weight': 1.0,
        'radius_or_number_of_bins': 32,
        'sampling_strategy': 'Regular',
        'sampling_percentage': 1.0,
    }
    sim_pass1 = pe.Node(ants.MeasureImageSimilarity(**similarity), name='sim_pass1')
    sim_pass2 = pe.Node(ants.MeasureImageSimilarity(**similarity), name='sim_pass2')
    guard_refinement = pe.Node(GuardRefinedTransform(), name='guard_refinement')
    choose_pass = pe.Node(
        niu.Function(
            function=_choose_pass,
            input_names=[
                'accepted',
                'hz1',
                'hz2',
                'warp1',
                'warp2',
                'reference1',
                'reference2',
                'report1',
                'report2',
            ],
            output_names=['out_hz', 'out_warp', 'out_reference', 'out_report'],
        ),
        name='choose_pass',
        run_without_submitting=True,
    )

    # Flicker the unwarped EPI reference (brain-masked, so the cuts sit on the brain) against
    # the fieldmap reference resampled onto it through the final transform.
    fmap_ref2ref = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc', float=True),
        name='fmap_ref2ref',
    )
    mask_unwarped_ref = pe.Node(ApplyMask(), name='mask_unwarped_ref')
    fmap2ref_rpt = pe.Node(
        SimpleBeforeAfter(before_label='Fieldmap reference', after_label='EPI reference'),
        name='fmap2ref_rpt',
        mem_gb=0.1,
    )

    ds_report_reg = pe.Node(
        DerivativesDataSink(datatype='figures', desc='fmapCoreg', suffix='fieldmap'),
        name='ds_report_reg',
        mem_gb=0.01,
        run_without_submitting=True,
    )

    ds_report_reg_vsm = pe.Node(
        DerivativesDataSink(datatype='figures', desc='vsm', suffix='fieldmap'),
        name='ds_report_vsm',
        mem_gb=0.01,
        run_without_submitting=True,
    )

    fieldmap_fov_mask = pe.Node(FilledImageLike(dtype='uint8'), name='fieldmap_fov_mask')

    fmap_fov2ref_apply = pe.Node(
        ANTSApplyTransformsRPT(
            generate_report=False, dimension=3, interpolation='NearestNeighbor', float=True
        ),
        name='fmap_fov2ref_apply',
    )

    apply_fov_mask = pe.Node(ApplyMask(), name='apply_fov_mask')

    workflow.connect([
        # pass 1: against the distorted reference
        (inputnode, fmap2ref_reg, [
            ('fmap_ref', 'moving_image'),
            ('in_reference', 'fixed_image'),
            ('in_mask', 'fixed_image_masks'),
            ('fmap_mask', 'moving_image_masks'),
        ]),
        (inputnode, fmap_apply_pass1_wf, [
            ('in_reference', 'inputnode.in_reference'),
            ('metadata', 'inputnode.metadata'),
            ('fmap', 'inputnode.fmap'),
        ]),
        (fmap2ref_reg, fmap_apply_pass1_wf, [('composite_transform', 'inputnode.transforms')]),
        # pass 2: against the reference unwarped with the first field
        (inputnode, fmap2ref_reg2, [
            ('fmap_ref', 'moving_image'),
            ('in_mask', 'fixed_image_masks'),
            ('fmap_mask', 'moving_image_masks'),
        ]),
        (fmap_apply_pass1_wf, fmap2ref_reg2, [('outputnode.out_reference', 'fixed_image')]),
        (fmap2ref_reg, fmap2ref_reg2, [('composite_transform', 'initial_moving_transform')]),
        (inputnode, fmap_apply_pass2_wf, [
            ('in_reference', 'inputnode.in_reference'),
            ('metadata', 'inputnode.metadata'),
            ('fmap', 'inputnode.fmap'),
        ]),
        (fmap2ref_reg2, fmap_apply_pass2_wf, [('composite_transform', 'inputnode.transforms')]),
        # which pass to use
        (fmap_apply_pass1_wf, sim_pass1, [('outputnode.out_reference', 'fixed_image')]),
        (fmap2ref_reg, sim_pass1, [('warped_image', 'moving_image')]),
        (inputnode, sim_pass1, [('in_mask', 'fixed_image_mask')]),
        (fmap_apply_pass2_wf, sim_pass2, [('outputnode.out_reference', 'fixed_image')]),
        (fmap2ref_reg2, sim_pass2, [('warped_image', 'moving_image')]),
        (inputnode, sim_pass2, [('in_mask', 'fixed_image_mask')]),
        (fmap2ref_reg, guard_refinement, [('composite_transform', 'initial_transform')]),
        (fmap2ref_reg2, guard_refinement, [('composite_transform', 'refined_transform')]),
        (sim_pass1, guard_refinement, [('similarity', 'initial_similarity')]),
        (sim_pass2, guard_refinement, [('similarity', 'refined_similarity')]),
        (inputnode, guard_refinement, [('in_reference', 'reference_image')]),
        (guard_refinement, choose_pass, [('accepted', 'accepted')]),
        (fmap_apply_pass1_wf, choose_pass, [
            ('outputnode.out_hz', 'hz1'),
            ('outputnode.out_warp', 'warp1'),
            ('outputnode.out_reference', 'reference1'),
            ('outputnode.out_report', 'report1'),
        ]),
        (fmap_apply_pass2_wf, choose_pass, [
            ('outputnode.out_hz', 'hz2'),
            ('outputnode.out_warp', 'warp2'),
            ('outputnode.out_reference', 'reference2'),
            ('outputnode.out_report', 'report2'),
        ]),
        (choose_pass, outputnode, [
            ('out_hz', 'out_hz'),
            ('out_warp', 'out_warp'),
        ]),
        # reports
        (inputnode, fmap_ref2ref, [('fmap_ref', 'input_image')]),
        (choose_pass, fmap_ref2ref, [('out_reference', 'reference_image')]),
        (guard_refinement, fmap_ref2ref, [('out_transform', 'transforms')]),
        (choose_pass, mask_unwarped_ref, [('out_reference', 'in_file')]),
        (inputnode, mask_unwarped_ref, [('in_mask', 'in_mask')]),
        (fmap_ref2ref, fmap2ref_rpt, [('output_image', 'before')]),
        (mask_unwarped_ref, fmap2ref_rpt, [('out_file', 'after')]),
        (fmap2ref_rpt, ds_report_reg, [('out_report', 'in_file')]),
        (choose_pass, ds_report_reg_vsm, [('out_report', 'in_file')]),
        # crop the unwarped reference to the fieldmap's field of view
        (inputnode, fieldmap_fov_mask, [('fmap_ref', 'in_file')]),
        (fieldmap_fov_mask, fmap_fov2ref_apply, [('out_file', 'input_image')]),
        (inputnode, fmap_fov2ref_apply, [('in_reference', 'reference_image')]),
        (guard_refinement, fmap_fov2ref_apply, [('out_transform', 'transforms')]),
        (fmap_fov2ref_apply, apply_fov_mask, [('output_image', 'in_mask')]),
        (choose_pass, apply_fov_mask, [('out_reference', 'in_file')]),
        (apply_fov_mask, outputnode, [
            ('out_file', 'out_reference'),
            ('out_file', 'out_reference_brain'),
        ]),
    ])  # fmt:skip

    return workflow


def init_fmap_apply_wf(name='fmap_apply_wf', generate_report=False):
    """Resample a Hz fieldmap onto the EPI reference and unwarp the reference with it.

    The field is resampled through the fieldmap-to-reference transform, converted to a
    voxel shift map along the phase-encoding axis with the reference's effective echo
    spacing, and turned into an ANTs displacement field that unwarps the reference.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.unwarp import init_fmap_apply_wf
        wf = init_fmap_apply_wf()

    Parameters
    ----------
    name : str
        Workflow name
    generate_report : bool
        Draw the resampled field over the reference (the ``desc-vsm`` reportlet)

    Inputs
    ------
    in_reference
        the EPI reference image
    metadata
        metadata associated to ``in_reference``
    fmap
        the fieldmap in Hz
    transforms
        the fieldmap-to-reference transform(s), for ``antsApplyTransforms``

    Outputs
    -------
    out_hz
        the fieldmap in Hz on the ``in_reference`` grid
    out_warp
        the corresponding :abbr:`DFM (displacements field map)` compatible with ANTs
    out_reference
        ``in_reference`` unwarped with ``out_warp``
    out_report
        the reportlet, when ``generate_report`` is set

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_reference', 'metadata', 'fmap', 'transforms']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_hz', 'out_warp', 'out_reference', 'out_report']),
        name='outputnode',
    )

    # Map the field into the EPI space
    fmap2ref_apply = pe.Node(
        ANTSApplyTransformsRPT(
            generate_report=generate_report, dimension=3, interpolation='BSpline', float=True
        ),
        name='fmap2ref_apply',
    )

    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(FieldToRadS(fmap_range=0.5), name='torads')

    get_ees = pe.Node(niu.Function(function=_get_ees, output_names=['ees']), name='get_ees')

    gen_vsm = pe.Node(FieldmapToVSM(), name='gen_vsm')
    # Convert the VSM into a DFM (displacements field map)
    # or: FUGUE shift to ANTS warping.
    vsm2dfm = pe.Node(FUGUEvsm2ANTSwarp(), name='vsm2dfm')

    unwarp_reference = pe.Node(
        ANTSApplyTransformsRPT(
            dimension=3, generate_report=False, float=True, interpolation='LanczosWindowedSinc'
        ),
        name='unwarp_reference',
    )

    workflow.connect([
        (inputnode, fmap2ref_apply, [
            ('fmap', 'input_image'),
            ('in_reference', 'reference_image'),
            ('transforms', 'transforms'),
        ]),
        (fmap2ref_apply, torads, [('output_image', 'in_file')]),
        (fmap2ref_apply, outputnode, [
            ('output_image', 'out_hz'),
            ('out_report', 'out_report'),
        ]),
        (inputnode, get_ees, [
            ('in_reference', 'in_file'),
            ('metadata', 'in_meta'),
        ]),
        (get_ees, gen_vsm, [('ees', 'dwell_time')]),
        (inputnode, gen_vsm, [(('metadata', _get_pedir_bids), 'pe_dir')]),
        (inputnode, vsm2dfm, [(('metadata', _get_pedir_bids), 'pe_dir')]),
        (torads, gen_vsm, [('out_file', 'in_file')]),
        (gen_vsm, vsm2dfm, [('shift_out_file', 'in_file')]),
        (vsm2dfm, unwarp_reference, [('out_file', 'transforms')]),
        (inputnode, unwarp_reference, [
            ('in_reference', 'reference_image'),
            ('in_reference', 'input_image'),
        ]),
        (vsm2dfm, outputnode, [('out_file', 'out_warp')]),
        (unwarp_reference, outputnode, [('output_image', 'out_reference')]),
    ])  # fmt:skip

    return workflow


def init_fmap_unwarp_report_wf(name='fmap_unwarp_report_wf'):
    """Build a workflow that generates a reportlet showing the effect of fieldmap unwarping.

    This workflow generates and saves a reportlet showing the effect of fieldmap
    unwarping a DWI image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.unwarp import init_fmap_unwarp_report_wf
        wf = init_fmap_unwarp_report_wf()

    Parameters
    ----------
    name : str, optional
        Workflow name (default: fmap_unwarp_report_wf)

    Inputs
    ------
    in_pre
        Reference image, before unwarping
    in_post
        Reference image, after unwarping
    in_seg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    in_xfm
        Affine transform from T1 space to b0 space (ITK format)

    """
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    from ...interfaces.images import ExtractWM

    DEFAULT_MEMORY_MIN_GB = 0.01

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_pre', 'in_post', 'in_seg', 'in_xfm']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['report']), name='outputnode')
    map_seg = pe.Node(
        ApplyTransforms(
            dimension=3, float=True, interpolation='MultiLabel', invert_transform_flags=[True]
        ),
        name='map_seg',
        mem_gb=0.3,
    )

    sel_wm = pe.Node(ExtractWM(), name='sel_wm', mem_gb=DEFAULT_MEMORY_MIN_GB)

    dwi_rpt = pe.Node(SimpleBeforeAfter(), name='dwi_rpt', mem_gb=0.1)

    workflow.connect([
        (inputnode, dwi_rpt, [
            ('in_pre', 'before'),
            ('in_post', 'after'),
        ]),
        (inputnode, map_seg, [
            ('in_post', 'reference_image'),
            ('in_seg', 'input_image'),
            ('in_xfm', 'transforms'),
        ]),
        (map_seg, sel_wm, [('output_image', 'in_seg')]),
        (sel_wm, dwi_rpt, [('out', 'wm_seg')]),
        (dwi_rpt, outputnode, [('out_report', 'report')])
    ])  # fmt:skip

    return workflow


# Helper functions
# ------------------------------------------------------------


def _get_pedir_bids(in_dict):
    return in_dict['PhaseEncodingDirection']


def _choose_pass(accepted, hz1, hz2, warp1, warp2, reference1, reference2, report1, report2):
    """Return the second pass's outputs when it was accepted, the first pass's otherwise."""
    if accepted:
        return hz2, warp2, reference2, report2
    return hz1, warp1, reference1, report1
