# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import os

import nibabel as nb
from nipype import logging
from collections import defaultdict

from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import FunctionalSummary
from fmriprep.engine import Workflow

# dwi workflows
from ..fieldmap.opposite_phase_series import init_opposite_phase_series_wf
from ..fieldmap.fieldmapless import init_no_fieldmap_wf


DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_preproc_wf(dwi_files,
                        ignore,
                        b0_motion_corr_to,
                        use_bbr,
                        b0_to_t1w_dof,
                        reportlets_dir,
                        freesurfer,
                        output_spaces,
                        dwi_denoise_window,
                        denoise_before_combining,
                        combine_all_dwis,
                        discard_repeated_samples,
                        template,
                        output_dir,
                        omp_nthreads,
                        fmap_bspline,
                        fmap_demean,
                        use_syn,
                        force_syn,
                        low_mem,
                        layout=None,
                        num_dwi=1):
    """
    Thi workflow controls the dwi preprocessing stages of qsiprep.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_dwi_preproc_wf
        wf = init_dwi_preproc_wf(['/completely/made/up/path/sub-01_dwi.nii.gz'],
                                  omp_nthreads=1,
                                  ignore=[],
                                  reportlets_dir='.',
                                  output_dir='.',
                                  template='MNI152NLin2009cAsym',
                                  output_spaces=['T1w', 'template'],
                                  freesurfer=False,
                                  use_bbr=True,
                                  dwi_denoise_window=7,
                                  denoise_before_combining=True,
                                  combine_all_dwis=True,
                                  discard_repeated_samples=True,
                                  b0_motion_corr_to='iterative',
                                  b0_to_t1w_dof=9,
                                  fmap_bspline=True,
                                  fmap_demean=True,
                                  use_syn=True,
                                  force_syn=True,
                                  low_mem=False,
                                  num_dwi=1)

    **Parameters**

        dwi_files : str or list
            dwi series NIfTI file or list of files to be combined
        ignore : list
            Preprocessing steps to skip (eg "slicetiming", "fieldmaps")
        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister) and
            resampling dwi series to FreeSurfer surface meshes.
        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        b0_motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        b0_to_t1w_dof : 6, 9 or 12
            Degrees-of-freedom for b0-T1w registration
        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0, '
            '``dwidwenoise`` will not be run'
        denoise_before_combining : bool
            'run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``'
        combine_all_dwis : bool
            Combine all dwi sequences within a session into a single data set
        discard_repeated_samples : Bool
            Ignore images if their q space coordinate has already been sampled
        reportlets_dir : str
            Directory in which to save reportlets
        output_spaces : list
            List of output spaces functional images are to be resampled to.
            Some parts of pipeline will only be instantiated for some output s
            paces.

            Valid spaces:

                - T1w
                - template

        template : str
            Name of template targeted by ``template`` output space
        output_dir : str
            Directory in which to save derivatives
        omp_nthreads : int
            Maximum number of threads an individual process may use
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        use_syn : bool
            **Experimental**: Enable ANTs SyN-based susceptibility distortion
            correction (SDC). If fieldmaps are present and enabled, this is not
            run, by default.
        force_syn : bool
            **Temporary**: Always run SyN-based SDC
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        layout : BIDSLayout
            BIDSLayout structure to enable metadata retrieval
        num_dwi : int
            Total number of dwi files that have been set for preprocessing
            (default is 1)

    **Inputs**

        dwi_files
            dwi file or list of dwi files
        t1_preproc
            Bias-corrected structural template image
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to
            FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed
            subject space to T1w


    **Outputs**

        dwi_t1
            dwi series, resampled to T1w space
        dwi_mask_t1
            dwi series mask in T1w space
        dwi_mni
            dwi series, resampled to template space
        dwi_mask_mni
            dwi series mask in template space



    **Subworkflows**

        * :py:func:`~qsiprep.workflows.dwi.util.init_dwi_reference_wf`
        * :py:func:`~qsiprep.workflows.dwi.hmc.init_dwi_hmc_wf`
        * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_t1_trans_wf`
        * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_reg_wf`
        * :py:func:`~qsiprep.workflows.dwi.confounds.init_dwi_confounds_wf`
        * :py:func:`~qsiprep.workflows.dwi.confounds.init_ica_aroma_wf`
        * :py:func:`~qsiprep.workflows.dwi.resampling.init_dwi_mni_trans_wf`
        * :py:func:`~qsiprep.workflows.dwi.resampling.init_dwi_surf_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.pepolar.init_pepolar_unwarp_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.init_fmap_estimator_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.init_sdc_unwarp_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.init_nonlinear_sdc_wf`

    """

    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    dwi_nvols = 10
    multiple_scans = isinstance(dwi_files, list)

    if multiple_scans:
        for scan in dwi_files:
            _dwi_nvols, _mem_gb = _create_mem_gb(scan)
            dwi_nvols += _dwi_nvols
            mem_gb['filesize'] += _mem_gb['filesize']
            mem_gb['resampled'] += _mem_gb['resampled']
            mem_gb['largemem'] += _mem_gb['largemem']
    else:
        dwi_nvols, mem_gb = _create_mem_gb(dwi_files)
        dwi_files = [dwi_files]

    wf_name = _get_wf_name(dwi_files[0])
    workflow = Workflow(name=wf_name)
    LOGGER.log(25, ('Creating dwi processing workflow "%s" '
                    '(%.2f GB / %d DWIs). '
                    'Memory resampled/largemem=%.2f/%.2f GB.'), wf_name,
               mem_gb['filesize'], dwi_nvols, mem_gb['resampled'],
               mem_gb['largemem'])

    # Create a list of (dwi image, PE dir, fieldmaps)
    parsed_dwis = []
    for ref_file in dwi_files:
        # Find associated sbref, if possible
        entities = layout.parse_file_entities(ref_file)
        entities['type'] = 'sbref'
        files = layout.get(**entities, extensions=['nii', 'nii.gz'])
        if len(files):
            LOGGER.warning("Ignoring sbref for %s", ref_file)
        metadata = layout.get_metadata(ref_file)

        # Find fieldmaps. Options: (epi only)
        fmaps = []
        fmap_file = ''
        if 'fieldmaps' not in ignore:
            fmaps = layout.get_fieldmap(ref_file, return_list=True)
            fmap_files = []
            for fmap in fmaps:
                if fmap['type'] == 'epi':
                    fmap_files.append(fmap['epi'])
            num_fmaps = len(fmap_files)
            if num_fmaps > 1:
                LOGGER.warning("Multiple field maps found for %s", ref_file)
            if num_fmaps >= 1:
                fmap_file = fmap_files[0]
        parsed_dwis.append(
            (ref_file, metadata['PhaseEncodingDirection'], fmap_file))

    # Depending on how the scans are organized, either use the fieldmap or the RPE series
    if not combine_all_dwis:
        raise Exception('Currently multiple separate reconstructions within a session is not '
                        'supported. Consider re-organizing your BIDS structure.')
    groups = defaultdict(list)
    for ref_file, pe_dir, fmap in parsed_dwis:
        groups[(pe_dir, fmap)].append(ref_file)
    if len(groups) > 2:
        LOGGER.critical("Unable to match fieldmaps to DWIs.")

    # Process the groupings
    if len(groups) == 2:
        # assign the plus and minus series:
        for (pe_dir, fmap), dwi_files in groups.items():
            if pe_dir.endswith("-"):
                series_minus = dwi_files
            else:
                series_plus = dwi_files

        if not force_syn:
            LOGGER.warning("Using b0 templates from opposite PEs to unwarp instead of fieldmaps")
            opposite_phase_series_wf = init_opposite_phase_series_wf(
                template_plus_pe=pe_dir[0],
                dwi_denoise_window=dwi_denoise_window,
                output_spaces=output_spaces,
                denoise_before_combining=denoise_before_combining,
                combine_all_dwis=combine_all_dwis, name='opposite_phase_series_wf')
            opposite_phase_series_wf.inputs.inputnode.pe_plus_dwis = series_plus
            opposite_phase_series_wf.inputs.inputnode.pe_minus_dwis = series_minus
            preproc_wf = opposite_phase_series_wf
        else:
            raise Exception("Applying two opposite polarity SDC not yet implemented")
    else:
        (pe_dir, fmap), dwi_files = list(groups.items())[0]
        if fmap:
            raise Exception("PEPOLAR on a single series not yet supported")
        else:
            dwi_no_fieldmap_wf = init_no_fieldmap_wf(
                use_syn=use_syn,
                pe_dir=pe_dir[0],
                dwi_denoise_window=dwi_denoise_window,
                output_spaces=output_spaces,
                denoise_before_combining=denoise_before_combining
            )
            dwi_no_fieldmap_wf.inputs.inputnode.dwi_files = dwi_files
            preproc_wf = dwi_no_fieldmap_wf

    return preproc_wf

#         workflow.__desc__ = """
#
# DWI data preprocessing
#
# : For each of the {num_dwi} dwi runs found per subject (across all
# tasks and sessions), the following preprocessing was performed.
#     """.format(num_dwi=num_dwi)
#         for ref_file, pe_dir, fmap in parsed_dwis:
#             scans_and_maps.append((ref_file, fmap))

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_file', 'sbref_file', 'subjects_dir', 'subject_id',
            't1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
            't1_aseg', 't1_aparc', 't1_2_mni_forward_transform',
            't1_2_mni_reverse_transform', 't1_2_fsnative_forward_transform',
            't1_2_fsnative_reverse_transform'
        ]),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_t1', 'dwi_t1_ref', 'dwi_mask_t1', 'dwi_aseg_t1',
            'dwi_aparc_t1', 'dwi_mni', 'dwi_mni_ref'
            'dwi_mask_mni', 'nonaggr_denoised_file'
        ]),
        name='outputnode')

"""
    # dwi buffer: an identity used as a pointer to either the original dwi
    # or the STC'ed one for further use.
    dwibuffer = pe.Node(
        niu.IdentityInterface(fields=['dwi_file']), name='dwibuffer')

    summary = pe.Node(
        FunctionalSummary(
            output_spaces=output_spaces,
            slice_timing=run_stc,
            registration='FreeSurfer' if freesurfer else 'FSL',
            registration_dof=b0_to_t1w_dof,
            pe_direction=metadata.get("PhaseEncodingDirection")),
        name='summary',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True)

    func_derivatives_wf = init_func_derivatives_wf(
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template,
        freesurfer=freesurfer)

    workflow.connect([
        (inputnode, func_derivatives_wf, [('dwi_file',
                                           'inputnode.source_file')]),
        (outputnode, func_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_t1_ref', 'inputnode.dwi_t1_ref'),
          ('dwi_aseg_t1', 'inputnode.dwi_aseg_t1'),
          ('dwi_aparc_t1', 'inputnode.dwi_aparc_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mni_ref', 'inputnode.dwi_mni_ref'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('confounds', 'inputnode.confounds'),
          ('surfaces', 'inputnode.surfaces'),
          ('aroma_noise_ics', 'inputnode.aroma_noise_ics'),
          ('melodic_mix', 'inputnode.melodic_mix'),
          ('nonaggr_denoised_file', 'inputnode.nonaggr_denoised_file'),
          ('dwi_cifti', 'inputnode.dwi_cifti'),
          ('cifti_variant', 'inputnode.cifti_variant'),
          ('cifti_variant_key', 'inputnode.cifti_variant_key')]),
    ])

    # Generate a tentative dwiref
    dwi_reference_wf = init_dwi_reference_wf(omp_nthreads=omp_nthreads)

    # HMC on the dwi
    dwi_hmc_wf = init_dwi_hmc_wf(
        name='dwi_hmc_wf',
        mem_gb=mem_gb['filesize'],
        omp_nthreads=omp_nthreads)

    # calculate dwi registration to T1w
    dwi_reg_wf = init_dwi_reg_wf(
        name='dwi_reg_wf',
        freesurfer=freesurfer,
        use_bbr=use_bbr,
        b0_to_t1w_dof=b0_to_t1w_dof,
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=False)

    # apply dwi registration to T1w
    dwi_t1_trans_wf = init_dwi_t1_trans_wf(
        name='dwi_t1_trans_wf',
        freesurfer=freesurfer,
        use_fieldwarp=(fmaps is not None or use_syn),
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=False)


    # Apply transforms in 1 shot
    # Only use uncompressed output if AROMA is to be run
    dwi_dwi_trans_wf = init_dwi_preproc_trans_wf(
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=not low_mem,
        use_fieldwarp=(fmaps is not None or use_syn),
        name='dwi_dwi_trans_wf')

    # MAIN WORKFLOW STRUCTURE ################################################
    workflow.connect([
        # Generate early reference
        (inputnode, dwi_reference_wf, [('dwi_file', 'inputnode.dwi_file'),
                                        ('sbref_file',
                                         'inputnode.sbref_file')]),
        # corrected if it was run, original otherwise
        (dwibuffer, dwi_split, [('dwi_file', 'in_file')]),
        # HMC
        (dwi_reference_wf, dwi_hmc_wf,
         [('outputnode.raw_ref_image', 'inputnode.raw_ref_image'),
          ('outputnode.dwi_file', 'inputnode.dwi_file')]),
        # EPI-T1 registration workflow
        (
            inputnode,
            dwi_reg_wf,
            [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_seg', 'inputnode.t1_seg'),
                # Undefined if --no-freesurfer, but this is safe
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('t1_2_fsnative_reverse_transform',
                 'inputnode.t1_2_fsnative_reverse_transform')
            ]),
        (inputnode, dwi_t1_trans_wf, [('dwi_file', 'inputnode.name_source'),
                                       ('t1_brain', 'inputnode.t1_brain'),
                                       ('t1_mask', 'inputnode.t1_mask'),
                                       ('t1_aseg', 'inputnode.t1_aseg'),
                                       ('t1_aparc', 'inputnode.t1_aparc')]),
        (dwi_split, dwi_t1_trans_wf, [('out_files',
                                         'inputnode.dwi_split')]),
        (dwi_hmc_wf, dwi_t1_trans_wf, [('outputnode.xforms',
                                          'inputnode.hmc_xforms')]),
        (dwi_reg_wf, dwi_t1_trans_wf, [('outputnode.itk_dwi_to_t1',
                                          'inputnode.itk_dwi_to_t1')]),
        (dwi_t1_trans_wf, outputnode,
         [('outputnode.dwi_t1', 'dwi_t1'),
          ('outputnode.dwi_t1_ref', 'dwi_t1_ref'),
          ('outputnode.dwi_aseg_t1', 'dwi_aseg_t1'),
          ('outputnode.dwi_aparc_t1', 'dwi_aparc_t1')]),
        (dwi_reg_wf, summary, [('outputnode.fallback', 'fallback')]),
        # SDC (or pass-through workflow)
        (inputnode, dwi_sdc_wf, [('t1_brain', 'inputnode.t1_brain'),
                                  ('t1_2_mni_reverse_transform',
                                   'inputnode.t1_2_mni_reverse_transform')]),
        (dwi_reference_wf, dwi_sdc_wf,
         [('outputnode.ref_image', 'inputnode.dwi_ref'),
          ('outputnode.ref_image_brain', 'inputnode.dwi_ref_brain'),
          ('outputnode.dwi_mask', 'inputnode.dwi_mask')]),
        (dwi_sdc_wf, dwi_reg_wf, [('outputnode.dwi_ref_brain',
                                     'inputnode.ref_dwi_brain')]),
        (dwi_sdc_wf, dwi_t1_trans_wf,
         [('outputnode.dwi_ref_brain', 'inputnode.ref_dwi_brain'),
          ('outputnode.dwi_mask', 'inputnode.ref_dwi_mask'),
          ('outputnode.out_warp', 'inputnode.fieldwarp')]),
        (dwi_sdc_wf, dwi_dwi_trans_wf,
         [('outputnode.out_warp', 'inputnode.fieldwarp'),
          ('outputnode.dwi_mask', 'inputnode.dwi_mask')]),
        (dwi_sdc_wf, summary, [('outputnode.method',
                                 'distortion_correction')]),
        # Connect dwi_confounds_wf
        (inputnode, dwi_confounds_wf, [('t1_tpms', 'inputnode.t1_tpms'),
                                        ('t1_mask', 'inputnode.t1_mask')]),
        (dwi_hmc_wf, dwi_confounds_wf, [('outputnode.movpar_file',
                                           'inputnode.movpar_file')]),
        (dwi_reg_wf, dwi_confounds_wf, [('outputnode.itk_t1_to_dwi',
                                           'inputnode.t1_dwi_xform')]),
        (dwi_reference_wf, dwi_confounds_wf, [('outputnode.skip_vols',
                                                 'inputnode.skip_vols')]),
        (dwi_confounds_wf, outputnode, [
            ('outputnode.confounds_file', 'confounds'),
        ]),
        # Connect dwi_dwi_trans_wf
        (inputnode, dwi_dwi_trans_wf, [('dwi_file',
                                          'inputnode.name_source')]),
        (dwi_split, dwi_dwi_trans_wf, [('out_files',
                                           'inputnode.dwi_file')]),
        (dwi_hmc_wf, dwi_dwi_trans_wf, [('outputnode.xforms',
                                            'inputnode.hmc_xforms')]),
        (dwi_dwi_trans_wf, dwi_confounds_wf,
         [('outputnode.dwi', 'inputnode.dwi'),
          ('outputnode.dwi_mask', 'inputnode.dwi_mask')]),
        # Summary
        (outputnode, summary, [('confounds', 'confounds_file')]),
    ])

    if fmaps:
        from ..fieldmap.unwarp import init_fmap_unwarp_report_wf
        sdc_type = fmaps[0]['type']

        # Report on dwi correction
        fmap_unwarp_report_wf = init_fmap_unwarp_report_wf(
            suffix='sdc_%s' % sdc_type)
        workflow.connect([
            (inputnode, fmap_unwarp_report_wf, [('t1_seg',
                                                 'inputnode.in_seg')]),
            (dwi_reference_wf, fmap_unwarp_report_wf,
             [('outputnode.ref_image', 'inputnode.in_pre')]),
            (dwi_reg_wf, fmap_unwarp_report_wf, [('outputnode.itk_t1_to_dwi',
                                                   'inputnode.in_xfm')]),
            (dwi_sdc_wf, fmap_unwarp_report_wf, [('outputnode.dwi_ref',
                                                   'inputnode.in_post')]),
        ])

        if force_syn and sdc_type != 'syn':
            syn_unwarp_report_wf = init_fmap_unwarp_report_wf(
                suffix='forcedsyn', name='syn_unwarp_report_wf')
            workflow.connect([
                (inputnode, syn_unwarp_report_wf, [('t1_seg',
                                                    'inputnode.in_seg')]),
                (dwi_reference_wf, syn_unwarp_report_wf,
                 [('outputnode.ref_image', 'inputnode.in_pre')]),
                (dwi_reg_wf, syn_unwarp_report_wf,
                 [('outputnode.itk_t1_to_dwi', 'inputnode.in_xfm')]),
                (dwi_sdc_wf, syn_unwarp_report_wf,
                 [('outputnode.syn_dwi_ref', 'inputnode.in_post')]),
            ])

    # Map final dwi mask into T1w space (if required)
    if 'T1w' in output_spaces:
        from niworkflows.interfaces.fixes import (FixHeaderApplyTransforms as
                                                  ApplyTransforms)

        dwimask_to_t1w = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', float=True),
            name='dwimask_to_t1w',
            mem_gb=0.1)
        workflow.connect([
            (dwi_dwi_trans_wf, dwimask_to_t1w, [('outputnode.dwi_mask',
                                                    'input_image')]),
            (dwi_reg_wf, dwimask_to_t1w, [('outputnode.itk_dwi_to_t1',
                                             'transforms')]),
            (dwi_t1_trans_wf, dwimask_to_t1w, [('outputnode.dwi_mask_t1',
                                                  'reference_image')]),
            (dwimask_to_t1w, outputnode, [('output_image', 'dwi_mask_t1')]),
        ])

    if 'template' in output_spaces:
        # Apply transforms in 1 shot
        dwi_mni_trans_wf = init_dwi_mni_trans_wf(
            template=template,
            mem_gb=mem_gb['resampled'],
            omp_nthreads=omp_nthreads,
            template_out_grid=None,
            use_compression=not low_mem,
            use_fieldwarp=fmaps is not None,
            name='dwi_mni_trans_wf')
        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb['resampled'],
            metadata=metadata,
            name='carpetplot_wf')

        workflow.connect([
            (inputnode, dwi_mni_trans_wf,
             [('dwi_file', 'inputnode.name_source'),
              ('t1_2_mni_forward_transform',
               'inputnode.t1_2_mni_forward_transform')]),
            (dwi_split, dwi_mni_trans_wf, [('out_files',
                                              'inputnode.dwi_split')]),
            (dwi_hmc_wf, dwi_mni_trans_wf, [('outputnode.xforms',
                                               'inputnode.hmc_xforms')]),
            (dwi_reg_wf, dwi_mni_trans_wf, [('outputnode.itk_dwi_to_t1',
                                               'inputnode.itk_dwi_to_t1')]),
            (dwi_dwi_trans_wf, dwi_mni_trans_wf, [('outputnode.dwi_mask',
                                                      'inputnode.dwi_mask')]),
            (dwi_sdc_wf, dwi_mni_trans_wf, [('outputnode.out_warp',
                                               'inputnode.fieldwarp')]),
            (dwi_mni_trans_wf, outputnode,
             [('outputnode.dwi_mni', 'dwi_mni'),
              ('outputnode.dwi_mni_ref', 'dwi_mni_ref'),
              ('outputnode.dwi_mask_mni', 'dwi_mask_mni')]),
            (dwi_dwi_trans_wf, carpetplot_wf,
             [('outputnode.dwi', 'inputnode.dwi'),
              ('outputnode.dwi_mask', 'inputnode.dwi_mask')]),
            (inputnode, carpetplot_wf,
             [('t1_2_mni_reverse_transform',
               'inputnode.t1_2_mni_reverse_transform')]),
            (dwi_reg_wf, carpetplot_wf, [('outputnode.itk_t1_to_dwi',
                                           'inputnode.t1_dwi_xform')]),
            (dwi_confounds_wf, carpetplot_wf, [('outputnode.confounds_file',
                                                 'inputnode.confounds_file')]),
        ])

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(suffix='summary'),
        name='ds_report_summary',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    ds_report_validation = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir, suffix='validation'),
        name='ds_report_validation',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (dwi_reference_wf, ds_report_validation,
         [('outputnode.validation_report', 'in_file')]),
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = ref_file
"""


def init_func_derivatives_wf(output_dir,
                             output_spaces,
                             template,
                             freesurfer,
                             use_aroma,
                             cifti_output,
                             name='func_derivatives_wf'):

    """Set up a battery of datasinks to store derivatives in the right location.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'source_file', 'dwi_t1', 'dwi_t1_ref', 'dwi_mask_t1',
            'dwi_mni', 'dwi_mni_ref', 'dwi_mask_mni', 'dwi_aseg_t1',
            'dwi_aparc_t1', 'cifti_variant_key', 'confounds', 'surfaces',
            'aroma_noise_ics', 'melodic_mix', 'nonaggr_denoised_file',
            'dwi_cifti', 'cifti_variant'
        ]),
        name='inputnode')

    ds_confounds = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='confounds', suffix='regressors'),
        name="ds_confounds",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, ds_confounds, [('source_file', 'source_file'),
                                   ('confounds', 'in_file')]),
    ])

    # Resample to T1w space
    if 'T1w' in output_spaces:
        ds_dwi_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='preproc',
                keep_dtype=True,
                compress=True),
            name='ds_dwi_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_dwi_t1_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space='T1w', suffix='dwiref'),
            name='ds_dwi_t1_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_dwi_mask_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='brain',
                suffix='mask'),
            name='ds_dwi_mask_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_dwi_t1, [('source_file', 'source_file'),
                                     ('dwi_t1', 'in_file')]),
            (inputnode, ds_dwi_t1_ref, [('source_file', 'source_file'),
                                         ('dwi_t1_ref', 'in_file')]),
            (inputnode, ds_dwi_mask_t1, [('source_file', 'source_file'),
                                          ('dwi_mask_t1', 'in_file')]),
        ])

    # Resample to template (default: MNI)
    if 'template' in output_spaces:
        ds_dwi_mni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=template,
                desc='preproc',
                keep_dtype=True,
                compress=True),
            name='ds_dwi_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_dwi_mni_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space=template, suffix='dwiref'),
            name='ds_dwi_mni_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_dwi_mask_mni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=template,
                desc='brain',
                suffix='mask'),
            name='ds_dwi_mask_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_dwi_mni, [('source_file', 'source_file'),
                                      ('dwi_mni', 'in_file')]),
            (inputnode, ds_dwi_mni_ref, [('source_file', 'source_file'),
                                          ('dwi_mni_ref', 'in_file')]),
            (inputnode, ds_dwi_mask_mni, [('source_file', 'source_file'),
                                           ('dwi_mask_mni', 'in_file')]),
        ])

    if freesurfer:
        ds_dwi_aseg_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='aseg',
                suffix='dseg'),
            name='ds_dwi_aseg_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_dwi_aparc_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='aparcaseg',
                suffix='dseg'),
            name='ds_dwi_aparc_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_dwi_aseg_t1, [('source_file', 'source_file'),
                                          ('dwi_aseg_t1', 'in_file')]),
            (inputnode, ds_dwi_aparc_t1, [('source_file', 'source_file'),
                                           ('dwi_aparc_t1', 'in_file')]),
        ])

    return workflow


def _get_series_len(dwi_fname):
    from niworkflows.interfaces.registration import _get_vols_to_discard
    img = nb.load(dwi_fname)
    if len(img.shape) < 4:
        return 1

    skip_vols = _get_vols_to_discard(img)

    return img.shape[3] - skip_vols


def _create_mem_gb(dwi_fname):
    dwi_size_gb = os.path.getsize(dwi_fname) / (1024**3)
    try:
        dwi_nvols = nb.load(dwi_fname).shape[3]
    except IndexError:
        dwi_nvols = 1
    mem_gb = {
        'filesize': dwi_size_gb,
        'resampled': dwi_size_gb * 4,
        'largemem': dwi_size_gb * (max(dwi_nvols / 100, 1.0) + 4),
    }

    return dwi_nvols, mem_gb


def _get_wf_name(dwi_fname):
    """Derive the workflow name for supplied DWI files.

    >>> _get_wf_name(['/made/up/sub-01_ses-01_run-1_dwi.nii.gz',
                      '/made/up/sub-01_ses-01_run-2_dwi.nii.gz'])
    'dwi_preproc_ses_01_wf'
    >>> _get_wf_name('/made/up/sub-01_ses-01_run-1_dwi.nii.gz')
    'dwi_preproc_ses_01_run_1_wf'
    """
    from nipype.utils.filemanip import split_filename
    if type(dwi_fname) is list:
        fname = split_filename(dwi_fname[0])[1]
        parts = fname.split("_")
        name = "_"
        if parts[1].startswith("ses"):
            name += parts[1].replace(
                ".", "_").replace(" ", "").replace("-", "_")
    else:
        fname_nosub = '_'.join(os.path.split(dwi_fname)[1].split("_")[1:-1])
        name = '_' + fname_nosub.replace(
            ".", "_").replace(" ", "").replace("-", "_")

    return "dwi_preproc" + name + "_wf"
