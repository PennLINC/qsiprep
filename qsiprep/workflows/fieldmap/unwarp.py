#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
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

import pkg_resources as pkgr
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import ants, fsl, utility as niu
from ...niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ...niworkflows.interfaces import itk
from ...niworkflows.interfaces.images import DemeanImage, FilledImageLike
from ...niworkflows.interfaces.registration import ANTSApplyTransformsRPT, ANTSRegistrationRPT

from ...interfaces import DerivativesDataSink
from ...interfaces.fmap import get_ees as _get_ees, FieldToRadS, FieldToHz


def init_sdc_unwarp_wf(omp_nthreads, fmap_demean, debug, name='sdc_unwarp_wf'):
    """
    This workflow takes in a displacements fieldmap and calculates the corresponding
    displacements field (in other words, an ANTs-compatible warp file).

    It also calculates a new mask for the input dataset that takes into account the distortions.
    The mask is restricted to the field of view of the fieldmap since outside of it corrections
    could not be performed.



    Inputs

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

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_jacobian
            the jacobian of the field (for drop-out alleviation)
        out_mask
            mask of the unwarped input file
        out_hz
            fieldmap in Hz that can be sent to eddy

    """
    fsl_check = os.environ.get('FSLDIR', False)
    if not fsl_check:
        raise Exception(
            """Container in use does not have FSL. To use this workflow,
            please download the qsiprep container with FSL installed.""")
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask', 'metadata',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask',
                'out_jacobian', 'out_hz']), name='outputnode')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('qsiprep', 'data/fmap-any_registration.json')
    if debug:
        ants_settings = pkgr.resource_filename(
            'qsiprep', 'data/fmap-any_registration_testing.json')
    fmap2ref_reg = pe.Node(
        ANTSRegistrationRPT(generate_report=True, from_file=ants_settings,
                            output_inverse_warped_image=True, output_warped_image=True),
        name='fmap2ref_reg', n_procs=omp_nthreads)

    ds_reg = pe.Node(DerivativesDataSink(suffix='fmap_reg'), name='ds_report_reg',
                     mem_gb=0.01, run_without_submitting=True)

    # Map the VSM into the EPI space
    fmap2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=True, dimension=3, interpolation='BSpline', float=True),
        name='fmap2ref_apply')

    fmap_mask2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='MultiLabel',
        float=True),
        name='fmap_mask2ref_apply')

    ds_reg_vsm = pe.Node(DerivativesDataSink(suffix='fmap_reg_vsm'), name='ds_report_vsm',
                         mem_gb=0.01, run_without_submitting=True)

    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(FieldToRadS(fmap_range=0.5), name='torads')

    # Make one in Hz for eddy
    tohz = pe.Node(FieldToHz(range_hz=1), name='tohz')

    get_ees = pe.Node(niu.Function(function=_get_ees, output_names=['ees']), name='get_ees')

    gen_vsm = pe.Node(fsl.FUGUE(save_unmasked_shift=True, save_fmap=True), name='gen_vsm')
    # Convert the VSM into a DFM (displacements field map)
    # or: FUGUE shift to ANTS warping.
    vsm2dfm = pe.Node(itk.FUGUEvsm2ANTSwarp(), name='vsm2dfm')
    jac_dfm = pe.Node(ants.CreateJacobianDeterminantImage(
        imageDimension=3, outputImage='jacobian.nii.gz'), name='jac_dfm')

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    fieldmap_fov_mask = pe.Node(FilledImageLike(dtype='uint8'), name='fieldmap_fov_mask')

    fmap_fov2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor',
        float=True),
        name='fmap_fov2ref_apply')

    apply_fov_mask = pe.Node(fsl.ApplyMask(), name="apply_fov_mask")

    workflow.connect([
        (inputnode, fmap2ref_reg, [('fmap_ref', 'moving_image')]),
        (inputnode, fmap2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap2ref_apply, [
            ('composite_transform', 'transforms')]),
        (inputnode, fmap_mask2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap_mask2ref_apply, [
            ('composite_transform', 'transforms')]),
        (fmap2ref_apply, ds_reg_vsm, [('out_report', 'in_file')]),
        (inputnode, fmap2ref_reg, [('in_reference_brain', 'fixed_image')]),
        (fmap2ref_reg, ds_reg, [('out_report', 'in_file')]),
        (inputnode, fmap2ref_apply, [('fmap', 'input_image')]),
        (inputnode, fmap_mask2ref_apply, [('fmap_mask', 'input_image')]),
        (fmap2ref_apply, torads, [('output_image', 'in_file')]),
        (fmap2ref_apply, tohz, [('output_image', 'in_file')]),
        (tohz, outputnode, [('out_file', 'out_hz')]),
        (inputnode, get_ees, [('in_reference', 'in_file'),
                              ('metadata', 'in_meta')]),
        (fmap_mask2ref_apply, gen_vsm, [('output_image', 'mask_file')]),
        (get_ees, gen_vsm, [('ees', 'dwell_time')]),
        (inputnode, gen_vsm, [(('metadata', _get_pedir_fugue), 'unwarp_direction')]),                      
        (inputnode, vsm2dfm, [(('metadata', _get_pedir_bids), 'pe_dir')]),
        (torads, gen_vsm, [('out_file', 'fmap_in_file')]),
        (gen_vsm, outputnode, [('fmap_out_file', 'fieldcoef')]),
        (vsm2dfm, unwarp_reference, [('out_file', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image')]),
        (inputnode, unwarp_reference, [('in_reference', 'input_image')]),
        (vsm2dfm, outputnode, [('out_file', 'out_warp')]),
        (vsm2dfm, jac_dfm, [('out_file', 'deformationField')]),
        (inputnode, fieldmap_fov_mask, [('fmap_ref', 'in_file')]),
        (fieldmap_fov_mask, fmap_fov2ref_apply, [('out_file', 'input_image')]),
        (inputnode, fmap_fov2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap_fov2ref_apply, [('composite_transform', 'transforms')]),
        (fmap_fov2ref_apply, apply_fov_mask, [('output_image', 'mask_file')]),
        (unwarp_reference, apply_fov_mask, [('output_image', 'in_file')]),
        (apply_fov_mask, outputnode, [
            ('out_file', 'out_reference'),
            ('out_file', 'out_reference_brain')]),
        (jac_dfm, outputnode, [('jacobian_image', 'out_jacobian')]),
    ])

    if fmap_demean:
        # Demean within mask
        demean = pe.Node(DemeanImage(), name='demean')

        workflow.connect([
            (gen_vsm, demean, [('shift_out_file', 'in_file')]),
            (fmap_mask2ref_apply, demean, [('output_image', 'in_mask')]),
            (demean, vsm2dfm, [('out_file', 'in_file')]),
        ])

    else:
        workflow.connect([
            (gen_vsm, vsm2dfm, [('shift_out_file', 'in_file')]),
        ])

    return workflow


def init_fmap_unwarp_report_wf(name='fmap_unwarp_report_wf', suffix='hmcsdc'):
    """
    This workflow generates and saves a reportlet showing the effect of fieldmap
    unwarping a DWI image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.unwarp import init_fmap_unwarp_report_wf
        wf = init_fmap_unwarp_report_wf()

    **Parameters**

        name : str, optional
            Workflow name (default: fmap_unwarp_report_wf)
        suffix : str, optional
            Suffix to be appended to this reportlet

    **Inputs**

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
    from ...niworkflows.interfaces import SimpleBeforeAfter
    from ...interfaces.images import ExtractWM
    from ...niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    DEFAULT_MEMORY_MIN_GB = 0.01

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_pre', 'in_post', 'in_seg', 'in_xfm']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['report']), name='outputnode')
    map_seg = pe.Node(
        ApplyTransforms(dimension=3, float=True, interpolation='MultiLabel',
                        invert_transform_flags=[True]),
        name='map_seg',
        mem_gb=0.3)

    sel_wm = pe.Node(ExtractWM(), name='sel_wm',
                     mem_gb=DEFAULT_MEMORY_MIN_GB)

    dwi_rpt = pe.Node(SimpleBeforeAfter(), name='dwi_rpt',
                      mem_gb=0.1)

    workflow.connect([
        (inputnode, dwi_rpt, [('in_pre', 'before'), ('in_post', 'after')]),
        (inputnode, map_seg, [('in_post', 'reference_image'),
                              ('in_seg', 'input_image'),
                              ('in_xfm', 'transforms')]),
        (map_seg, sel_wm, [('output_image', 'in_seg')]),
        (sel_wm, dwi_rpt, [('out', 'wm_seg')]),
        (dwi_rpt, outputnode, [('out_report', 'report')])
    ])

    return workflow


# Helper functions
# ------------------------------------------------------------


def _get_pedir_bids(in_dict):
    return in_dict['PhaseEncodingDirection']

def _get_pedir_fugue(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('i', 'x').replace('j', 'y').replace('k', 'z')