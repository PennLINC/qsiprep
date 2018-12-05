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

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import DiffusionSummary
from ...interfaces.images import SplitDWIs, ConcatRPESplits
from ...interfaces.gradients import WarpAndRecombineDWIs

from fmriprep.engine import Workflow

# dwi workflows
from ..fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
from ..fieldmap.base import init_sdc_wf
from .merge import init_merge_and_denoise_wf
from .hmc import init_b0_hmc_wf
from .util import init_dwi_reference_wf
from .registration import init_b0_to_anat_registration_wf

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
    This workflow controls the dwi preprocessing stages of qsiprep.

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
            List of dwi series NIfTI files to be combined or a dict of PE-dir -> files
        ignore : list
            Preprocessing steps to skip (eg "fieldmaps")
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
        t1_output_grid
            Image to write out DWIs aligned to t1
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
        dwi_sampling_grid
            A NIfTI1 file with the grid spacing and FoV to resample the DWIs

    **Outputs**

        dwi_t1
            dwi series, resampled to T1w space
        dwi_mask_t1
            dwi series mask in T1w space
        bvals_t1
            bvalues of the dwi series
        bvecs_t1
            bvecs after aligning to the T1w and resampling
        local_bvecs_t1
            voxelwise bvecs accounting for local displacements

        dwi_mni
            dwi series, resampled to template space
        dwi_mask_mni
            dwi series mask in template space
        bvals_mni
            bvalues of the dwi series
        bvecs_mni
            bvecs after aligning to the T1w and resampling
        local_bvecs_mni
            voxelwise bvecs accounting for local displacements

        confounds_file
            estimated motion parameters and zipper scores

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

    if type(dwi_files) is dict:
        doing_bidirectional_pepolar = True
        _dwi_lists = list(dwi_files.values())
        all_dwis = _dwi_lists[0] + _dwi_lists[1]
    else:
        doing_bidirectional_pepolar = False
        all_dwis = dwi_files

    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    dwi_nvols = 10

    for scan in all_dwis:
        _dwi_nvols, _mem_gb = _create_mem_gb(scan)
        dwi_nvols += _dwi_nvols
        mem_gb['filesize'] += _mem_gb['filesize']
        mem_gb['resampled'] += _mem_gb['resampled']
        mem_gb['largemem'] += _mem_gb['largemem']

    wf_name = _get_wf_name(all_dwis[0])
    workflow = Workflow(name=wf_name)
    LOGGER.log(25, ('Creating dwi processing workflow "%s" '
                    '(%.2f GB / %d DWIs). '
                    'Memory resampled/largemem=%.2f/%.2f GB.'), wf_name,
               mem_gb['filesize'], dwi_nvols, mem_gb['resampled'],
               mem_gb['largemem'])
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_files', 'sbref_file', 'subjects_dir', 'subject_id',
            't1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
            't1_aseg', 't1_aparc', 't1_2_mni_forward_transform',
            't1_2_mni_reverse_transform', 't1_2_fsnative_forward_transform',
            't1_2_fsnative_reverse_transform', 't1_output_grid'
        ]),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_t1', 'dwi_mask_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1', 't1_b0_ref',
            't1_b0_series', 'dwi_mni', 'dwi_mask_mni', 'bvals_mni', 'bvecs_mni',
            'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'confounds'
        ]),
        name='outputnode')

    # Special case: Two reverse PE DWI series
    if doing_bidirectional_pepolar:
        # Merge, denoise, split, hmc on the plus series
        plus_key, = [k for k in dwi_files.keys() if len(k) == 1]
        merge_plus = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                               denoise_before_combining=denoise_before_combining,
                                               name="merge_plus")
        split_plus = pe.Node(SplitDWIs(), name="split_plus")
        b0_hmc_plus = init_b0_hmc_wf(name="b0_hmc_plus")
        merge_plus.inputs.inputnode.dwi_files = dwi_files[plus_key]

        # Merge, denoise, split, hmc on the minus series
        merge_minus = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                                denoise_before_combining=denoise_before_combining,
                                                name="merge_minus")
        split_minus = pe.Node(SplitDWIs(), name="split_minus")
        b0_hmc_minus = init_b0_hmc_wf(name="b0_hmc_minus")
        merge_minus.inputs.inputnode.dwi_files = dwi_files[plus_key + "-"]

        # Get affines and warps to the b0 ref
        bidir_pepolar_wf = init_bidirectional_b0_unwarping_wf(
            template_plus_pe=list(dwi_files.keys())[0], name='bidir_pepolar_wf')

        # Combine the original images from the splits into one 'Split'
        concat_rpe_splits = pe.Node(ConcatRPESplits(), name="concat_rpe_splits")
        buffernode = concat_rpe_splits
        # Remove this!
        workflow.add_nodes([inputnode, outputnode])
        workflow.connect([
                # Merge, denoise, split, hmc on the plus series
                (merge_plus, split_plus, [('outputnode.merged_image', 'dwi_file'),
                                          ('outputnode.merged_bval', 'bval_file'),
                                          ('outputnode.merged_bvec', 'bvec_file')]),
                (split_plus, concat_rpe_splits,
                    [('bval_files', 'bval_plus'),
                     ('bvec_files', 'bvec_plus'),
                     ('dwi_files', 'dwi_plus'),
                     ('b0_images', 'b0_images_plus'),
                     ('b0_indices', 'b0_indices_plus')]),
                (split_plus, b0_hmc_plus, [('b0_images', 'inputnode.b0_images')]),
                (b0_hmc_plus, concat_rpe_splits, [
                    (('outputnode.forward_transforms', _list_squeeze), 'hmc_affines_plus')]),

                # Merge, denoise, split, hmc on the minus series
                (merge_minus, split_minus, [('outputnode.merged_image', 'dwi_file'),
                                            ('outputnode.merged_bval', 'bval_file'),
                                            ('outputnode.merged_bvec', 'bvec_file')]),
                (split_minus, concat_rpe_splits,
                    [('bval_files', 'bval_minus'),
                     ('bvec_files', 'bvec_minus'),
                     ('dwi_files', 'dwi_minus'),
                     ('b0_images', 'b0_images_minus'),
                     ('b0_indices', 'b0_indices_minus')]),
                (split_minus, b0_hmc_minus, [('b0_images', 'inputnode.b0_images')]),
                (b0_hmc_plus, concat_rpe_splits, [
                    (('outputnode.forward_transforms', _list_squeeze), 'hmc_affines_minus')]),

                # Use the hmc templates as the input for pepolar unwarping
                (b0_hmc_minus, bidir_pepolar_wf,
                    [('outputnode.final_template', 'inputnode.template_minus')]),
                (b0_hmc_plus, bidir_pepolar_wf,
                    [('outputnode.final_template', 'inputnode.template_plus')]),

                # send unwarping to the rpe recombiner
                (bidir_pepolar_wf, concat_rpe_splits, [
                    (('outputnode.out_affine_plus', _get_first), 'template_plus_to_ref_affine'),
                    (('outputnode.out_affine_minus', _get_first), 'template_minus_to_ref_affine'),
                    ('outputnode.out_warp_plus', 'template_plus_to_ref_warp'),
                    ('outputnode.out_warp_minus', 'template_minus_to_ref_warp'),
                    ('outputnode.out_reference', 'b0_ref_image'),
                    ('outputnode.out_mask', 'b0_ref_mask')
                    ]),
        ])

    # Normal cases. No RPE to worry about
    else:
        buffernode = pe.Node(niu.IdentityInterface(fields=[
            'b0_ref_image', 'b0_ref_mask', 'dwi_files', 'bvec_files', 'bval_files', 'b0_images',
            'b0_indices', 'to_dwi_ref_affines', 'to_dwi_ref_warps', 'original_grouping']),
            name="buffernode")
        # Merge, denoise, split, hmc
        merge_dwis = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                               denoise_before_combining=denoise_before_combining,
                                               name="merge_dwis")
        split_dwis = pe.Node(SplitDWIs(), name="split_dwis")
        b0_hmc = init_b0_hmc_wf(name="b0_hmc")
        merge_dwis.inputs.inputnode.dwi_files = dwi_files

        # Fieldmap time
        sbref_file = None
        # For doc building purposes
        if layout is None or dwi_files == 'dwi_preprocesing':
            LOGGER.log(25, 'No valid layout: building empty workflow.')
            metadata = {
                'PhaseEncodingDirection': 'j',
            }
            fmaps = [{
                'type': 'epi',
                'epi': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_epi.nii.gz'
            }]
        else:
            # These were grouped at the beginning to have the same fmap file
            ref_file = dwi_files[0]
            # Find associated sbref, if possible
            entities = layout.parse_file_entities(ref_file)
            entities['type'] = 'sbref'
            files = layout.get(**entities, extensions=['nii', 'nii.gz'])
            refbase = os.path.basename(ref_file)
            if 'sbref' in ignore:
                LOGGER.info("Single-band reference files ignored.")
            elif files:
                sbref_file = files[0].filename
                sbbase = os.path.basename(sbref_file)
                if len(files) > 1:
                    LOGGER.warning(
                        "Multiple single-band reference files found for {}; using "
                        "{}".format(refbase, sbbase))
                else:
                    LOGGER.log(25, "Using single-band reference file {}".format(sbbase))
            else:
                LOGGER.log(25, "No single-band-reference found for {}".format(refbase))

            metadata = layout.get_metadata(ref_file)

            # Find fieldmaps. Options: (epi|syn)
            fmaps = []
            if 'fieldmaps' not in ignore:
                fmaps = layout.get_fieldmap(ref_file, return_list=True)
                for fmap in fmaps:
                    fmap['metadata'] = layout.get_metadata(fmap[fmap['type']])

            # Run SyN if forced or in the absence of fieldmap correction
            if force_syn or (use_syn and not fmaps):
                fmaps.append({'type': 'syn'})

        dwi_ref_wf = init_dwi_reference_wf(name="dwi_ref_wf")

        b0_sdc_wf = init_sdc_wf(
            fmaps, metadata, omp_nthreads=omp_nthreads,
            fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)
        b0_sdc_wf.inputs.inputnode.template = template

        if not fmaps:
            LOGGER.warning('SDC: no fieldmaps found or they were ignored (%s).',
                           ref_file)
        elif fmaps[0]['type'] == 'syn':
            LOGGER.warning(
                'SDC: no fieldmaps found or they were ignored. '
                'Using EXPERIMENTAL "fieldmap-less SyN" correction '
                'for dataset %s.', ref_file)
        else:
            LOGGER.log(25, 'SDC: fieldmap estimation of type "%s" intended for %s found.',
                       fmaps[0]['type'], ref_file)

        workflow.connect([
            # Merge, denoise, split, hmc on the dwis
            (merge_dwis, split_dwis, [('outputnode.merged_image', 'dwi_file'),
                                      ('outputnode.merged_bval', 'bval_file'),
                                      ('outputnode.merged_bvec', 'bvec_file')]),
            (split_dwis, buffernode,
                [('outputnode.bval_files', 'bval_files'),
                 ('outputnode.bvec_files', 'bvec_files'),
                 ('outputnode.dwi_files', 'dwi_files'),
                 ('outputnode.b0_images', 'b0_images'),
                 ('outputnode.b0_indices', 'b0_indices')]),
            (split_dwis, b0_hmc, [('outputnode.b0_images', 'inputnode.b0_images')]),
            (b0_hmc, buffernode, [(('outputnode.forward_transforms', _list_squeeze),
                                  'to_dwi_ref_affines')]),
            (b0_hmc, dwi_ref_wf, [('outputnode.final_template', 'inputnode.b0_template')]),
            (dwi_ref_wf, b0_sdc_wf, [('outputnode.ref_image', 'inputnode.b0_ref'),
                                     ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                                     ('outputnode.dwi_mask', 'inputnode.b0_mask')]),
            (inputnode, b0_sdc_wf, [('t1_brain', 'inputnode.t1_brain'),
                                    ('t1_2t1_2_mni_reverse_transform',
                                     'inputnode.t1_2t1_2_mni_reverse_transform')]),
            (b0_sdc_wf, buffernode, [('outputnode.b0_ref', 'b0_ref_image'),
                                     ('outputnode.b0_mask', 'b0_ref_mask'),
                                     ('outputnode.out_warp', 'to_dwi_ref_warps')]),


        ])

    summary = pe.Node(
        DiffusionSummary(
            output_spaces=output_spaces),
        name='summary',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True)

    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template)
    '''
    workflow.connect([
        (inputnode, dwi_derivatives_wf, [('dwi_files', 'inputnode.source_file')]),
        (outputnode, dwi_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('bvals_t1', 'inputnode.bvals_t1'),
          ('bvecs_t1', 'inputnode.bvecs_t1'),
          ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
          ('t1_b0_ref', 'inputnode.t1_b0_ref'),
          ('t1_b0_series', 'inputnode.t1_b0_series'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('bvals_mni', 'inputnode.bvals_mni'),
          ('bvecs_mni', 'inputnode.bvecs_mni'),
          ('local_bvecs_mni', 'inputnode.local_bvecs_mni'),
          ('mni_b0_ref', 'inputnode.mni_b0_ref'),
          ('mni_b0_series', 'inputnode.mni_b0_series'),
          ('confounds', 'inputnode.confounds')])
    ])
    '''
    # calculate dwi registration to T1w
    b0_coreg_wf = init_b0_to_anat_registration_wf(omp_nthreads=omp_nthreads,
                                                  mem_gb=mem_gb['resampled'])

    workflow.connect([
        (inputnode, b0_coreg_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_seg', 'inputnode.t1_seg'),
                # Undefined if --no-freesurfer, but this is safe
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('t1_2_fsnative_reverse_transform',
                 'inputnode.t1_2_fsnative_reverse_transform')]),
        (buffernode, b0_coreg_wf, [('b0_ref_image',
                                   'inputnode.ref_b0_brain')]),
    ])

    if "T1w" in output_spaces:
        transform_dwis_t1 = pe.Node(WarpAndRecombineDWIs(), name='transform_dwis_t1')

        workflow.connect([
            (inputnode, transform_dwis_t1, [('t1_output_grid', 'output_grid')]),
                                            # ('t1_2_mni_forward_transform',
                                            # 't1_2_mni_forward_transform')]),
            (buffernode, transform_dwis_t1, [
                ('dwi_files', 'dwi_files'),
                ('bvec_files', 'bvec_files'),
                ('bval_files', 'bval_files'),
                ('b0_ref_image', 'b0_ref_image'),
                ('b0_ref_mask', 'dwi_mask'),
                ('b0_indices', 'original_b0_indices'),
                ('to_dwi_ref_affines', 'b0_hmc_affines'),
                ('to_dwi_ref_warps', 'dwi_ref_to_unwarped_warp')]),
            (b0_coreg_wf, transform_dwis_t1, [('outputnode.itk_b0_to_t1',
                                               'unwarped_dwi_ref_to_t1w_affine')]),

            #(transform_dwis_t1, outputnode, [()])
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
        (summary, ds_report_summary, [('out_report', 'in_file')])])
    #    (dwi_reference_wf, ds_report_validation,
    #         [('outputnode.validation_report', 'in_file')]),
    # ])

    # Fill-in datasinks of reportlets seen so far
    # for node in workflow.list_node_names():
    #     if node.split('.')[-1].startswith('ds_report'):
    #         workflow.get_node(node).inputs.base_directory = reportlets_dir
    #         workflow.get_node(node).inputs.source_file = dwi_files

    return workflow


def init_dwi_derivatives_wf(output_dir,
                            output_spaces,
                            template,
                            name='dwi_derivatives_wf'):

    """Set up a battery of datasinks to store derivatives in the right location.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'source_file', 'dwi_t1', 'dwi_mask_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1',
            't1_b0_ref', 't1_b0_series', 'dwi_mni', 'dwi_mask_mni', 'bvals_mni', 'bvecs_mni',
            'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'confounds'
        ]),
        name='inputnode')

    ds_confounds = pe.Node(DerivativesDataSink(
        base_directory=output_dir, desc='confounds', suffix='confounds'),
        name="ds_confounds", run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Resample to T1w space
    if 'T1w' in output_spaces:
        # 4D DWI in t1 space
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
        ds_bvals_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='preproc'),
            name='ds_bvals_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvecs_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='preproc'),
            name='ds_bvecs_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_local_bvecs_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='preproc'),
            name='ds_local_bvecs_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_t1_b0_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space='T1w', suffix='dwiref'),
            name='ds_t1_b0_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_t1_b0_series = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space='T1w', suffix='b0series'),
            name='ds_t1_b0_series',
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
            (inputnode, ds_bvals_t1, [('source_file', 'source_file'),
                                      ('bvals_t1', 'in_file')]),
            (inputnode, ds_bvecs_t1, [('source_file', 'source_file'),
                                      ('bvecs_t1', 'in_file')]),
            (inputnode, ds_local_bvecs_t1, [('source_file', 'source_file'),
                                            ('local_bvecs_t1', 'in_file')]),
            (inputnode, ds_t1_b0_ref, [('source_file', 'source_file'),
                                       ('t1_b0_ref', 'in_file')]),
            (inputnode, ds_t1_b0_series, [('source_file', 'source_file'),
                                          ('t1_b0_series', 'in_file')]),
            (inputnode, ds_dwi_mask_t1, [('source_file', 'source_file'),
                                         ('t1_b0_series', 'in_file')]),
            ])

    # Resample to template (default: MNI)
    if 'template' in output_spaces:
        # 4D DWI in t1 space
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
        ds_bvals_mni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=template,
                desc='preproc'),
            name='ds_bvals_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvecs_mni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=template,
                desc='preproc'),
            name='ds_bvecs_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_local_bvecs_mni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=template,
                desc='preproc'),
            name='ds_local_bvecs_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_mni_b0_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space=template, suffix='dwiref'),
            name='ds_mni_b0_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_mni_b0_series = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space=template, suffix='b0series'),
            name='ds_mni_b0_series',
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
            (inputnode, ds_bvals_mni, [('source_file', 'source_file'),
                                       ('bvals_mni', 'in_file')]),
            (inputnode, ds_bvecs_mni, [('source_file', 'source_file'),
                                       ('bvecs_mni', 'in_file')]),
            (inputnode, ds_local_bvecs_mni, [('source_file', 'source_file'),
                                             ('local_bvecs_mni', 'in_file')]),
            (inputnode, ds_mni_b0_ref, [('source_file', 'source_file'),
                                        ('mni_b0_ref', 'in_file')]),
            (inputnode, ds_mni_b0_series, [('source_file', 'source_file'),
                                           ('mni_b0_series', 'in_file')]),
            (inputnode, ds_dwi_mask_mni, [('source_file', 'source_file'),
                                          ('mni_b0_series', 'in_file')]),
            ])

    return workflow


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

def _list_squeeze(in_list):
    return [item[0] for item in in_list]
def _get_first(in_list):
    return in_list[0]
