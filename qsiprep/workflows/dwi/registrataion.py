#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_bold_reg_wf
.. autofunction:: init_bold_t1_trans_wf
.. autofunction:: init_bbreg_wf
.. autofunction:: init_fsl_bbr_wf

"""

import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, c3, ants
from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from fmriprep.engine import Workflow
from ...interfaces import MultiApplyTransforms, DerivativesDataSink

from ...interfaces.nilearn import Merge
from ...interfaces.images import extract_wm
from ...interfaces.freesurfer import (
        PatchedConcatenateLTA as ConcatenateLTA,
        PatchedBBRegisterRPT as BBRegisterRPT,
        PatchedMRICoregRPT as MRICoregRPT,
        PatchedLTAConvert as LTAConvert)


DEFAULT_MEMORY_MIN_GB = 0.01


def init_b0_to_anat_registration_wf(biascorrect_anat=False,
                                    biascorrect_b0=False):
    """
    registers a b0 to an anatomical scan. Bias corrects each
    (if requested) and coregisters the b0 to the anat. Returns
    the transform and Mattes score
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["b0_image", "anat_image"]),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_to_anat_transform", "coreg_metric"]),
        name='outputnode')
    b0_anat_coreg_wf = Workflow(name="b0_anat_coreg")

    # Defines a coregistration operation
    coreg = ants.Registration()
    coreg.inputs.metric = ["Mattes"]
    coreg.inputs.transforms = ["Rigid"]
    coreg.inputs.shrink_factors = [[8, 4, 2, 1]]
    coreg.inputs.smoothing_sigmas = [[7., 3., 1., 0.]]
    coreg.inputs.sigma_units = ["vox"]
    coreg.inputs.sampling_strategy = ['Random']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.radius_or_number_of_bins = [32]
    coreg.inputs.initial_moving_transform_com = 0
    coreg.inputs.interpolation = 'HammingWindowedSinc'
    coreg.inputs.dimension = 3
    coreg.inputs.winsorize_lower_quantile = 0.025
    coreg.inputs.winsorize_upper_quantile = 0.975
    coreg.inputs.number_of_iterations = [[10000, 1000, 10000, 10000]]
    coreg.inputs.transform_parameters = [[0.2]]
    coreg.inputs.convergence_threshold = [1e-06]
    coreg.inputs.collapse_output_transforms = True
    coreg.inputs.write_composite_transform = False
    coreg.inputs.output_warped_image = True
    b0_to_anat = pe.Node(coreg, name="b0_to_anat")

    b0_anat_coreg_wf.connect(inputnode, "anat_image", b0_to_anat, "fixed_image")
    b0_anat_coreg_wf.connect(inputnode, "b0_image", b0_to_anat, "moving_image")

    b0_anat_coreg_wf.connect(b0_to_anat, "forward_transforms", outputnode,
                             "b0_to_anat_transform")
    b0_anat_coreg_wf.connect(b0_to_anat, "metric_value", outputnode,
                             "coreg_metric")

    return b0_anat_coreg_wf


def init_bold_reg_wf(freesurfer, use_bbr, bold2t1w_dof, mem_gb, omp_nthreads,
                     use_compression=True, write_report=True, name='bold_reg_wf'):
    """
    Calculates the registration between a reference b0 image and T1-space
    using a boundary-based registration (BBR) cost function.

    If FreeSurfer-based preprocessing is enabled, the ``bbregister`` utility
    is used to align the BOLD images to the reconstructed subject, and the
    resulting transform is adjusted to target the T1 space.
    If FreeSurfer-based preprocessing is disabled, FSL FLIRT is used with the
    BBR cost function to directly target the T1 space.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.bold.registration import init_bold_reg_wf
        wf = init_bold_reg_wf(freesurfer=True,
                              mem_gb=3,
                              omp_nthreads=1,
                              use_bbr=True,
                              bold2t1w_dof=9)

    **Parameters**

        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister)
        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_reg_wf``)
        use_compression : bool
            Save registered BOLD series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to T1
        write_report : bool
            Whether a reportlet should be stored

    **Inputs**

        ref_bold_brain
            Reference image to which BOLD series is aligned
            If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    **Outputs**

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        itk_t1_to_bold
            Affine transform from T1 space to BOLD space (ITK format)
        fallback
            Boolean indicating whether BBR was rejected (mri_coreg registration returned)


    **Subworkflows**

        * :py:func:`~qsiprep.workflows.bold.registration.init_bbreg_wf`
        * :py:func:`~qsiprep.workflows.bold.registration.init_fsl_bbr_wf`

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['ref_bold_brain', 't1_brain', 't1_seg',
                    'subjects_dir', 'subject_id', 't1_2_fsnative_reverse_transform']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_bold_to_t1', 'itk_t1_to_bold', 'fallback']),
        name='outputnode'
    )

    if freesurfer:
        bbr_wf = init_bbreg_wf(use_bbr=use_bbr, bold2t1w_dof=bold2t1w_dof,
                               omp_nthreads=omp_nthreads)
    else:
        bbr_wf = init_fsl_bbr_wf(use_bbr=use_bbr, bold2t1w_dof=bold2t1w_dof)

    workflow.connect([
        (inputnode, bbr_wf, [
            ('ref_bold_brain', 'inputnode.in_file'),
            ('t1_2_fsnative_reverse_transform', 'inputnode.t1_2_fsnative_reverse_transform'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_brain', 'inputnode.t1_brain')]),
        (bbr_wf, outputnode, [('outputnode.itk_bold_to_t1', 'itk_bold_to_t1'),
                              ('outputnode.itk_t1_to_bold', 'itk_t1_to_bold'),
                              ('outputnode.fallback', 'fallback')]),
    ])

    if write_report:
        ds_report_reg = pe.Node(
            DerivativesDataSink(),
            name='ds_report_reg', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        def _bold_reg_suffix(fallback, freesurfer):
            if fallback:
                return 'coreg' if freesurfer else 'flirtnobbr'
            return 'bbregister' if freesurfer else 'flirtbbr'

        workflow.connect([
            (bbr_wf, ds_report_reg, [
                ('outputnode.out_report', 'in_file'),
                (('outputnode.fallback', _bold_reg_suffix, freesurfer), 'suffix')]),
        ])

    return workflow


def init_bold_t1_trans_wf(freesurfer, mem_gb, omp_nthreads, use_fieldwarp=False,
                          use_compression=True, name='bold_t1_trans_wf'):
    """
    This workflow registers the reference BOLD image to T1-space, using a
    boundary-based registration (BBR) cost function.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.bold.registration import init_bold_t1_trans_wf
        wf = init_bold_t1_trans_wf(freesurfer=True,
                                   mem_gb=3,
                                   omp_nthreads=1)

    **Parameters**

        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister)
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to T1
        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        use_compression : bool
            Save registered BOLD series as ``.nii.gz``
        name : str
            Name of workflow (default: ``bold_reg_wf``)

    **Inputs**

        name_source
            BOLD series NIfTI file
            Used to recover original information lost during processing
        ref_bold_brain
            Reference image to which BOLD series is aligned
            If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
        ref_bold_mask
            Skull-stripping mask of reference image
        t1_brain
            Skull-stripped bias-corrected structural template image
        t1_mask
            Mask of the skull-stripped template image
        t1_aseg
            FreeSurfer's ``aseg.mgz`` atlas projected into the T1w reference
            (only if ``recon-all`` was run).
        t1_aparc
            FreeSurfer's ``aparc+aseg.mgz`` atlas projected into the T1w reference
            (only if ``recon-all`` was run).
        bold_split
            Individual 3D BOLD volumes, not motion corrected
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        fieldwarp
            a :abbr:`DFM (displacements field map)` in ITK format

    **Outputs**

        bold_t1
            Motion-corrected BOLD series in T1 space
        bold_t1_ref
            Reference, contrast-enhanced summary of the motion-corrected BOLD series in T1w space
        bold_mask_t1
            BOLD mask in T1 space
        bold_aseg_t1
            FreeSurfer's ``aseg.mgz`` atlas, in T1w-space at the BOLD resolution
            (only if ``recon-all`` was run).
        bold_aparc_t1
            FreeSurfer's ``aparc+aseg.mgz`` atlas, in T1w-space at the BOLD resolution
            (only if ``recon-all`` was run).


    **Subworkflows**

        * :py:func:`~qsiprep.workflows.bold.registration.init_bbreg_wf`
        * :py:func:`~qsiprep.workflows.bold.registration.init_fsl_bbr_wf`

    """
    from .util import init_bold_reference_wf
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['name_source', 'ref_bold_brain', 'ref_bold_mask',
                    't1_brain', 't1_mask', 't1_aseg', 't1_aparc',
                    'bold_split', 'fieldwarp', 'hmc_xforms',
                    'itk_bold_to_t1']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'bold_t1', 'bold_t1_ref', 'bold_mask_t1',
            'bold_aseg_t1', 'bold_aparc_t1']),
        name='outputnode'
    )

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref',
                      mem_gb=0.3)  # 256x256x256 * 64 / 8 ~ 150MB

    mask_t1w_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True),
        name='mask_t1w_tfm', mem_gb=0.1
    )

    workflow.connect([
        (inputnode, gen_ref, [('ref_bold_brain', 'moving_image'),
                              ('t1_brain', 'fixed_image'),
                              ('t1_mask', 'fov_mask')]),
        (inputnode, mask_t1w_tfm, [('ref_bold_mask', 'input_image')]),
        (gen_ref, mask_t1w_tfm, [('out_file', 'reference_image')]),
        (inputnode, mask_t1w_tfm, [('itk_bold_to_t1', 'transforms')]),
        (mask_t1w_tfm, outputnode, [('output_image', 'bold_mask_t1')]),
    ])

    if freesurfer:
        # Resample aseg and aparc in T1w space (no transforms needed)
        aseg_t1w_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', transforms='identity', float=True),
            name='aseg_t1w_tfm', mem_gb=0.1)
        aparc_t1w_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', transforms='identity', float=True),
            name='aparc_t1w_tfm', mem_gb=0.1)

        workflow.connect([
            (inputnode, aseg_t1w_tfm, [('t1_aseg', 'input_image')]),
            (inputnode, aparc_t1w_tfm, [('t1_aparc', 'input_image')]),
            (gen_ref, aseg_t1w_tfm, [('out_file', 'reference_image')]),
            (gen_ref, aparc_t1w_tfm, [('out_file', 'reference_image')]),
            (aseg_t1w_tfm, outputnode, [('output_image', 'bold_aseg_t1')]),
            (aparc_t1w_tfm, outputnode, [('output_image', 'bold_aparc_t1')]),
        ])

    # Merge transforms placing the head motion correction last
    nforms = 2 + int(use_fieldwarp)
    merge_xforms = pe.Node(niu.Merge(nforms), name='merge_xforms',
                           run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, merge_xforms, [('hmc_xforms', 'in%d' % nforms)])
    ])

    if use_fieldwarp:
        workflow.connect([
            (inputnode, merge_xforms, [('fieldwarp', 'in2')])
        ])

    bold_to_t1w_transform = pe.Node(
        MultiApplyTransforms(interpolation="LanczosWindowedSinc", float=True, copy_dtype=True),
        name='bold_to_t1w_transform', mem_gb=mem_gb * 3 * omp_nthreads, n_procs=omp_nthreads)

    merge = pe.Node(Merge(compress=use_compression), name='merge', mem_gb=mem_gb)

    # Generate a reference on the target T1w space
    gen_final_ref = init_bold_reference_wf(omp_nthreads, pre_mask=True)

    workflow.connect([
        (inputnode, merge_xforms, [('itk_bold_to_t1', 'in1')]),
        (merge_xforms, bold_to_t1w_transform, [('out', 'transforms')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, bold_to_t1w_transform, [('bold_split', 'input_image')]),
        (gen_ref, bold_to_t1w_transform, [('out_file', 'reference_image')]),
        (bold_to_t1w_transform, merge, [('out_files', 'in_files')]),
        (merge, gen_final_ref, [('out_file', 'inputnode.bold_file')]),
        (mask_t1w_tfm, gen_final_ref, [('output_image', 'inputnode.bold_mask')]),
        (merge, outputnode, [('out_file', 'bold_t1')]),
        (gen_final_ref, outputnode, [('outputnode.ref_image', 'bold_t1_ref')]),
    ])

    return workflow
