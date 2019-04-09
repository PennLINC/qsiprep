"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsiprep_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import os

import nibabel as nb
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import DiffusionSummary, GradientPlot
from ...interfaces.images import SplitDWIs, ConcatRPESplits
from ...interfaces.gradients import ExtractB0s
from ...interfaces.confounds import DMRISummary
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from ..fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.unwarp import init_fmap_unwarp_report_wf
from .merge import init_merge_and_denoise_wf
from .hmc import init_dwi_hmc_wf
from .util import init_dwi_reference_wf
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_dwi_preproc_wf(dwi_files,
                            output_prefix,
                            ignore,
                            motion_corr_to,
                            b0_to_t1w_transform,
                            hmc_model,
                            hmc_transform,
                            impute_slice_threshold,
                            reportlets_dir,
                            freesurfer,
                            output_spaces,
                            dwi_denoise_window,
                            denoise_before_combining,
                            template,
                            output_dir,
                            omp_nthreads,
                            write_local_bvecs,
                            fmap_bspline,
                            fmap_demean,
                            use_syn,
                            force_syn,
                            low_mem,
                            layout=None,
                            num_dwi=1):
    """
    This workflow controls the dwi preprocessing stages using FSL tools.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.base import init_qsiprep_dwi_preproc_wf
        wf = init_fsl_dwi_preproc_wf(['/completely/made/up/path/sub-01_dwi.nii.gz'],
                                     omp_nthreads=1,
                                     ignore=[],
                                     reportlets_dir='.',
                                     output_dir='.',
                                     template='MNI152NLin2009cAsym',
                                     output_spaces=['T1w', 'template'],
                                     freesurfer=False,
                                     dwi_denoise_window=7,
                                     denoise_before_combining=True,
                                     motion_corr_to='iterative',
                                     b0_to_t1w_transform='Rigid',
                                     hmc_model='3dSHORE',
                                     hmc_transform='Affine',
                                     impute_slice_threshold=0,
                                     fmap_bspline=True,
                                     fmap_demean=True,
                                     use_syn=True,
                                     force_syn=True,
                                     low_mem=False,
                                     output_prefix='',
                                     write_local_bvecs=False,
                                     num_dwi=1)

    **Parameters**

        dwi_files : str or list
            List of dwi series NIfTI files to be combined or a dict of PE-dir -> files
        output_prefix : str
            beginning of the output file name (eg 'sub-1_buds-j')
        ignore : list
            Preprocessing steps to skip (eg "fieldmaps")
        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister) and
            resampling dwi series to FreeSurfer surface meshes.
        motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        b0_to_t1w_transform : "Rigid" or "Affine"
            Use a rigid or full affine transform for b0-T1w registration
        hmc_model : 'none', '3dSHORE' or 'MAPMRI'
            Model used to generate target images for head motion correction. If 'none'
            the transform from the nearest b0 will be used.
        hmc_transform : "Rigid" or "Affine"
            Type of transform used for head motion correction
        impute_slice_threshold : float
            Impute data in slices that are this many SDs from expected. If 0, no slices
            will be imputed.
        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0, '
            '``dwidwenoise`` will not be run'
        denoise_before_combining : bool
            'run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``'
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
            dwi file list or dict of dwi file lists indexed by PE dir
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
        gradient_table_t1
            MRTrix-style gradient table

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
        gradient_table_mni
            MRTrix-style gradient table

        confounds_file
            estimated motion parameters and zipper scores

    **Subworkflows**

        * :py:func:`~qsiprep.workflows.dwi.util.init_dwi_reference_wf`
        * :py:func:`~qsiprep.workflows.dwi.hmc.init_dwi_hmc_wf`
        * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_t1_trans_wf`
        * :py:func:`~qsiprep.workflows.dwi.registration.init_dwi_reg_wf`
        * :py:func:`~qsiprep.workflows.dwi.confounds.init_dwi_confounds_wf`
        * :py:func:`~qsiprep.workflows.dwi.resampling.init_dwi_trans_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.pepolar.init_pepolar_unwarp_wf`
        * :py:func:`~qsiprep.workflows.fieldmap.init_nonlinear_sdc_wf`

    """
    all_dwis = dwi_files

    # For naming outputs
    source_file = all_dwis[0]

    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    dwi_nvols = 10

    for scan in all_dwis:
        if not os.path.exists(scan):
            # For docs building
            continue
        _dwi_nvols, _mem_gb = _create_mem_gb(scan)
        dwi_nvols += _dwi_nvols
        mem_gb['filesize'] += _mem_gb['filesize']
        mem_gb['resampled'] += _mem_gb['resampled']
        mem_gb['largemem'] += _mem_gb['largemem']

    wf_name = _get_wf_name(output_prefix)
    workflow = Workflow(name=wf_name)
    LOGGER.log(25, ('Creating fsl processing workflow "%s" '
                    'to produce output %s '
                    '(%.2f GB / %d DWIs). '
                    'Memory resampled/largemem=%.2f/%.2f GB.'), wf_name,
               output_prefix, mem_gb['filesize'], dwi_nvols, mem_gb['resampled'],
               mem_gb['largemem'])
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_files', 'sbref_file', 'subjects_dir', 'subject_id',
            't1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
            't1_aseg', 't1_aparc', 't1_2_mni_forward_transform',
            't1_2_mni_reverse_transform', 't1_2_fsnative_forward_transform',
            't1_2_fsnative_reverse_transform', 'dwi_sampling_grid'
        ]),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_t1', 'dwi_mask_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1', 't1_b0_ref',
            't1_b0_series', 'dwi_mni', 'dwi_mask_mni', 'bvals_mni', 'bvecs_mni',
            'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'confounds', 'gradient_table_mni',
            'gradient_table_t1'
        ]),
        name='outputnode')

    # calculate dwi registration to T1w
    b0_coreg_wf = init_b0_to_anat_registration_wf(omp_nthreads=omp_nthreads,
                                                  mem_gb=mem_gb['resampled'],
                                                  write_report=True)
    buffernode = pe.Node(
        niu.IdentityInterface(
            fields=['b0_ref_image', 'b0_ref_mask', 'dwi_files', 'bvec_files', 'bval_files',
                    'b0_images', 'b0_indices', 'to_dwi_ref_affines', 'to_dwi_ref_warps',
                    'original_grouping', 'sdc_method', 'ideal_images']),
        name="buffernode")

    if hmc_model == 'eddy_ingress':
        buffernode.inputs.sdc_method = "eddy"
        # Merge, split
        merge_dwis = init_merge_and_denoise_wf(dwi_denoise_window=0,
                                               denoise_before_combining=False,
                                               name="merge_dwis")
        merge_dwis.inputs.inputnode.dwi_files = dwi_files
        split_dwis = pe.Node(SplitDWIs(), name="split_dwis")
        collect_b0s = pe.Node(ExtractB0s(), name="collect_b0s")
        dwi_ref_wf = init_dwi_reference_wf(name="dwi_ref_wf")
        workflow.connect([
            # Merge, denoise, split, hmc on the dwis
            (merge_dwis, split_dwis, [('outputnode.merged_image', 'dwi_file'),
                                      ('outputnode.merged_bval', 'bval_file'),
                                      ('outputnode.merged_bvec', 'bvec_file')]),
            (merge_dwis, collect_b0s, [('outputnode.merged_image', 'dwi_file'),
                                       ('outputnode.merged_bval', 'bval_file'),
                                       ('outputnode.merged_bvec', 'bvec_file')]),
            (split_dwis, buffernode, [('bval_files', 'bval_files'),
                                      ('bvec_files', 'bvec_files'),
                                      ('dwi_files', 'dwi_files'),
                                      ('b0_images', 'b0_images'),
                                      ('b0_indices', 'b0_indices')]),
            (merge_dwis, buffernode, [('outputnode.original_files', 'original_grouping')]),
            (collect_b0s, dwi_ref_wf, [('b0_average', 'inputnode.b0_template')]),
            (dwi_ref_wf, buffernode, [('outputnode.ref_image', 'b0_ref_image'),
                                      ('outputnode.dwi_mask', 'b0_ref_mask')]),
        ])

    else:
        dwi_hmc_wf = init_dwi_hmc_wf(hmc_transform, hmc_model, motion_corr_to,
                                     omp_nthreads=omp_nthreads, name="dwi_hmc_wf")

        # Fieldmap time
        sbref_file = None
        # These were grouped at the beginning to have the same fmap file
        ref_file = dwi_files[0]
        # For doc building purposes
        if layout is None or dwi_files == 'dwi_preprocesing':
            LOGGER.log(25, 'No valid layout: building empty workflow.')
            metadata = {
                'PhaseEncodingDirection': 'j',
            }
            fmaps = []
        else:
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


            if not fmaps:
                LOGGER.warning('SDC: no fieldmaps found or they were ignored (%s).',
                               ref_file)

            else:
                sdc_type = fmaps[0]['type']

                # Report on SDC
                fmap_unwarp_report_wf = init_fmap_unwarp_report_wf(
                    suffix='sdc_%s' % sdc_type)
                workflow.connect([
                    (inputnode, fmap_unwarp_report_wf, [
                        ('t1_seg', 'inputnode.in_seg')]),
                    (dwi_hmc_wf, fmap_unwarp_report_wf, [
                        ('outputnode.final_template', 'inputnode.in_pre')]),
                    (b0_coreg_wf, fmap_unwarp_report_wf, [
                        ('outputnode.itk_b0_to_t1', 'inputnode.in_xfm')]),
                    (b0_sdc_wf, fmap_unwarp_report_wf, [
                        ('outputnode.b0_ref', 'inputnode.in_post')]),
                    ])
                LOGGER.log(25, 'SDC: fieldmap estimation of type "%s" intended for %s found.',
                           fmaps[0]['type'], ref_file)

            workflow.connect([
                # Merge, denoise, split, hmc on the dwis
                (merge_dwis, split_dwis, [('outputnode.merged_image', 'dwi_file'),
                                          ('outputnode.merged_bval', 'bval_file'),
                                          ('outputnode.merged_bvec', 'bvec_file')]),
                (split_dwis, buffernode,
                    [('bval_files', 'bval_files'),
                     ('bvec_files', 'bvec_files'),
                     ('dwi_files', 'dwi_files'),
                     ('b0_images', 'b0_images'),
                     ('b0_indices', 'b0_indices')]),
                (merge_dwis, buffernode, [('outputnode.original_files', 'original_grouping')]),
                (split_dwis, dwi_hmc_wf, [('b0_images', 'inputnode.b0_images'),
                                          ('bval_files', 'inputnode.bvals'),
                                          ('bvec_files', 'inputnode.bvecs'),
                                          ('dwi_files', 'inputnode.dwi_files'),
                                          ('b0_indices', 'inputnode.b0_indices')]),
                (dwi_hmc_wf, buffernode, [(('outputnode.forward_transforms', _list_squeeze),
                                          'to_dwi_ref_affines'),
                                          ('outputnode.noise_free_dwis', 'ideal_images')]),
                (dwi_hmc_wf, dwi_ref_wf, [('outputnode.final_template', 'inputnode.b0_template')]),
                (dwi_ref_wf, b0_sdc_wf, [('outputnode.ref_image', 'inputnode.b0_ref'),
                                         ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
                                         ('outputnode.dwi_mask', 'inputnode.b0_mask')]),
                (inputnode, b0_sdc_wf, [('t1_brain', 'inputnode.t1_brain'),
                                        ('t1_2_mni_reverse_transform',
                                         'inputnode.t1_2_mni_reverse_transform')]),
                (b0_sdc_wf, buffernode, [('outputnode.b0_ref', 'b0_ref_image'),
                                         ('outputnode.b0_mask', 'b0_ref_mask'),
                                         ('outputnode.out_warp', 'to_dwi_ref_warps'),
                                         ('outputnode.method', 'sdc_method')]),


            ])

    # At this point, buffernode is either a ConcatRPESplit or a single PE output
    # summary = pe.Node(
    #     DiffusionSummary(
    #         pe_direction='j',
    #         hmc_model=hmc_model,
    #         b0_to_t1w_transform=b0_to_t1w_transform,
    #         hmc_transform=hmc_transform,
    #         impute_slice_threshold=impute_slice_threshold,
    #         dwi_denoise_window=dwi_denoise_window,
    #         output_spaces=output_spaces),
    #     name='summary',
    #     mem_gb=DEFAULT_MEMORY_MIN_GB,
    #     run_without_submitting=True)

    gradient_plot = pe.Node(GradientPlot(), name='gradient_plot', run_without_submitting=True)
    # ds_report_gradients = pe.Node(
    #     DerivativesDataSink(suffix='sampling_scheme'),
    #     name='ds_report_gradients', run_without_submitting=True,
    #     mem_gb=DEFAULT_MEMORY_MIN_GB)

    # CONNECT TO DERIVATIVES #####################
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_prefix=output_prefix,
        source_file=source_file,
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template,
        write_local_bvecs=write_local_bvecs)

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
          ('gradient_table_t1', 'inputnode.gradient_table_t1'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('bvals_mni', 'inputnode.bvals_mni'),
          ('bvecs_mni', 'inputnode.bvecs_mni'),
          ('local_bvecs_mni', 'inputnode.local_bvecs_mni'),
          ('mni_b0_ref', 'inputnode.mni_b0_ref'),
          ('mni_b0_series', 'inputnode.mni_b0_series'),
          ('gradient_table_mni', 'inputnode.gradient_table_mni'),
          ('confounds', 'inputnode.confounds')])
    ])

    workflow.connect([
        (inputnode, b0_coreg_wf,
            [('t1_brain', 'inputnode.t1_brain'),
             ('t1_seg', 'inputnode.t1_seg'),
             ('subjects_dir', 'inputnode.subjects_dir'),
             ('subject_id', 'inputnode.subject_id'),
             ('t1_2_fsnative_reverse_transform',
              'inputnode.t1_2_fsnative_reverse_transform')]),
        (buffernode, b0_coreg_wf, [('b0_ref_image',
                                    'inputnode.ref_b0_brain')]),
    ])

    # confounds_wf = init_dwi_confs_wf(mem_gb=mem_gb['resampled'], metadata=[],
    #                                 impute_slice_threshold=impute_slice_threshold,
    #                                 name='confounds_wf')

    # Carpetplot and confounds plot
    # conf_plot = pe.Node(DMRISummary(), name='conf_plot', mem_gb=mem_gb['resampled'])
    # ds_report_dwi_conf = pe.Node(
    #    DerivativesDataSink(suffix='carpetplot'),
    #    name='ds_report_dwi_conf', run_without_submitting=True,
    #    mem_gb=DEFAULT_MEMORY_MIN_GB)

    #  workflow.connect([
    #     (buffernode, slice_check, [('ideal_images', 'ideal_image_files'),
    #                                ('dwi_files', 'uncorrected_dwi_files'),
    #                                ('b0_ref_mask', 'mask_image')]),
    #     (slice_check, confounds_wf, [('slice_stats', 'inputnode.sliceqc_file')]),
    #     (buffernode, confounds_wf, [('to_dwi_ref_affines', 'inputnode.hmc_affines'),
    #                                 ('bval_files', 'inputnode.bval_files'),
    #                                 ('bvec_files', 'inputnode.bvec_files'),
    #                                 ('original_grouping', 'inputnode.original_files')]),
    #     (confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),
    #
    #     (confounds_wf, conf_plot, [('outputnode.confounds_file', 'confounds_file')]),
    #     (slice_check, conf_plot, [('slice_stats', 'sliceqc_file')]),
    #     (buffernode, conf_plot, []),
    #     (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
    #     (buffernode, gradient_plot, [('bvec_files', 'orig_bvec_files'),
    #                                  ('bval_files', 'orig_bval_files'),
    #                                  ('original_grouping', 'source_files')]),
    #     (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])
    #
    # ])

    if "T1w" in output_spaces:
        transform_dwis_t1 = init_dwi_trans_wf(name='transform_dwis_t1',
                                              template="ACPC",
                                              mem_gb=mem_gb['resampled'],
                                              use_fieldwarp=(fmaps is not None or use_syn),
                                              omp_nthreads=omp_nthreads,
                                              use_compression=False,
                                              to_mni=False,
                                              write_local_bvecs=write_local_bvecs
                                              )
        gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
        workflow.connect([
            (buffernode, transform_dwis_t1, [
                ('dwi', 'inputnode.dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('b0_ref_mask', 'inputnode.dwi_mask'),
                ('b0_indices', 'inputnode.b0_indices')]),
            (inputnode, transform_dwis_t1, [
                ('dwi_sampling_grid', 'inputnode.output_grid')]),
            (b0_coreg_wf, transform_dwis_t1, [
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_t1, outputnode, [('outputnode.bvals', 'bvals_t1'),
                                             ('outputnode.rotated_bvecs', 'bvecs_t1'),
                                             ('outputnode.dwi_resampled', 'dwi_t1'),
                                             ('outputnode.local_bvecs', 'local_bvecs_t1'),
                                             ('outputnode.dwi_mask_resampled', 'dwi_mask_t1'),
                                             ('outputnode.b0_series', 't1_b0_series'),
                                             ('outputnode.dwi_ref_resampled', 't1_b0_ref')]),
            (outputnode, gradient_plot, [('bvecs_t1', 'final_bvec_file')]),
            (transform_dwis_t1, gtab_t1, [('outputnode.bvals', 'bval_file'),
                                          ('outputnode.rotated_bvecs', 'bvec_file')]),
            (gtab_t1, outputnode, [('gradient_file', 'gradient_table_t1')])
        ])

    if "template" in output_spaces:
        transform_dwis_mni = init_dwi_trans_wf(name='transform_dwis_mni',
                                               template=template,
                                               mem_gb=mem_gb['resampled'],
                                               use_fieldwarp=(fmaps is not None or use_syn),
                                               omp_nthreads=omp_nthreads,
                                               use_compression=False,
                                               to_mni=True,
                                               write_local_bvecs=write_local_bvecs
                                               )
        gtab_mni = pe.Node(MRTrixGradientTable(), name='gtab_mni')
        workflow.connect([
            (buffernode, transform_dwis_mni, [
                ('dwi_files', 'dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
                ('b0_ref_image', 'inputnode.b0_ref_image'),
                ('b0_ref_mask', 'inputnode.dwi_mask'),
                ('b0_indices', 'inputnode.b0_indices'),
                ('to_dwi_ref_affines', 'inputnode.hmc_xforms'),
                ('to_dwi_ref_warps', 'inputnode.fieldwarps')]),
            (inputnode, transform_dwis_mni, [
                ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                ('dwi_sampling_grid', 'inputnode.output_grid')]),
            (b0_coreg_wf, transform_dwis_mni, [
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_mni, outputnode, [('outputnode.bvals', 'bvals_mni'),
                                              ('outputnode.rotated_bvecs', 'bvecs_mni'),
                                              ('outputnode.dwi_resampled', 'dwi_mni'),
                                              ('outputnode.dwi_mask_resampled', 'dwi_mask_mni'),
                                              ('outputnode.b0_series', 'mni_b0_series'),
                                              ('outputnode.local_bvecs', 'local_bvecs_mni'),
                                              ('outputnode.dwi_ref_resampled', 'mni_b0_ref')]),
            (transform_dwis_mni, gtab_mni, [('outputnode.bvals', 'bval_file'),
                                            ('outputnode.rotated_bvecs', 'bvec_file')]),
            (gtab_mni, outputnode, [('gradient_file', 'gradient_table_mni')])
        ])
        if "T1w" not in output_spaces:
            workflow.connect([(outputnode, gradient_plot, [('bvecs_mni', 'final_bvec_file')])])

    # REPORTING ############################################################
    # ds_report_summary = pe.Node(
    #     DerivativesDataSink(suffix='summary'),
    #     name='ds_report_summary',
    #     run_without_submitting=True,
    #     mem_gb=DEFAULT_MEMORY_MIN_GB)

    # workflow.connect([
    #   (summary, ds_report_summary, [('out_report', 'in_file')])
    # ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = source_file

    return workflow


def init_dwi_derivatives_wf(output_prefix,
                            source_file,
                            output_dir,
                            output_spaces,
                            template,
                            write_local_bvecs,
                            name='dwi_derivatives_wf'):

    """Set up a battery of datasinks to store derivatives in the right location.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'source_file', 'dwi_t1', 'dwi_mask_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1',
            't1_b0_ref', 't1_b0_series', 'gradient_table_t1', 'dwi_mni', 'dwi_mask_mni', 'bvals_mni', 'bvecs_mni',
            'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'gradient_table_mni', 'confounds'
        ]),
        name='inputnode')

    ds_confounds = pe.Node(DerivativesDataSink(
        prefix=output_prefix,
        source_file=source_file,
        base_directory=output_dir, suffix='confounds'),
        name="ds_confounds", run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, ds_confounds, [('confounds', 'in_file')])
    ])

    # Resample to T1w space
    if 'T1w' in output_spaces:
        # 4D DWI in t1 space
        ds_dwi_t1 = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                desc='preproc',
                suffix='dwi',
                extension='.nii.gz',
                compress=True),
            name='ds_dwi_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvals_t1 = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                suffix='dwi',
                extension='.bval',
                desc='preproc'),
            name='ds_bvals_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvecs_t1 = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                suffix='dwi',
                extension='.bvec',
                desc='preproc'),
            name='ds_bvecs_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_t1_b0_ref = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                suffix='dwiref',
                extension='.nii.gz',
                compress=True),
            name='ds_t1_b0_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_t1_b0_series = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                suffix='b0series',
                extension='.nii.gz',
                compress=True),
            name='ds_t1_b0_series',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_dwi_mask_t1 = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                desc='brain',
                suffix='mask',
                extension='.nii.gz',
                compress=True),
            name='ds_dwi_mask_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_gradient_table_t1 = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space='T1w',
                suffix='dwi',
                extension='.b',
                desc='preproc'),
            name='ds_gradient_table_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_dwi_t1, [('dwi_t1', 'in_file')]),
            (inputnode, ds_bvals_t1, [('bvals_t1', 'in_file')]),
            (inputnode, ds_bvecs_t1, [('bvecs_t1', 'in_file')]),
            (inputnode, ds_t1_b0_ref, [('t1_b0_ref', 'in_file')]),
            (inputnode, ds_t1_b0_series, [('t1_b0_series', 'in_file')]),
            (inputnode, ds_dwi_mask_t1, [('dwi_mask_t1', 'in_file')]),
            (inputnode, ds_gradient_table_t1, [('gradient_table_t1', 'in_file')])
            ])
        # If requested, write local bvecs
        if write_local_bvecs:
            ds_local_bvecs_t1 = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=source_file,
                    space='T1w',
                    suffix='bvec',
                    compress=True),
                name='ds_local_bvecs_t1',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([(inputnode, ds_local_bvecs_t1, [('local_bvecs_t1', 'in_file')])])
    # Resample to template (default: MNI)
    if 'template' in output_spaces:
        # 4D DWI in t1 space
        ds_dwi_mni = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                desc='preproc',
                suffix='dwi',
                extension='.nii.gz',
                keep_dtype=True,
                compress=True),
            name='ds_dwi_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvals_mni = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                suffix='dwi',
                extension='.bval',
                desc='preproc'),
            name='ds_bvals_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bvecs_mni = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                suffix='dwi',
                extension='.bvec',
                desc='preproc'),
            name='ds_bvecs_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_mni_b0_ref = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                suffix='dwiref',
                extension='.nii.gz',
                compress=True),
            name='ds_mni_b0_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_mni_b0_series = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                suffix='b0series',
                extension='.nii.gz',
                compress=True),
            name='ds_mni_b0_series',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_dwi_mask_mni = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                desc='brain',
                suffix='mask',
                extension='.nii.gz',
                compress=True),
            name='ds_dwi_mask_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_gradient_table_mni = pe.Node(
            DerivativesDataSink(
                prefix=output_prefix,
                source_file=source_file,
                base_directory=output_dir,
                space=template,
                suffix='dwi',
                extension='.b',
                desc='preproc'),
            name='ds_gradient_table_mni',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_dwi_mni, [('dwi_mni', 'in_file')]),
            (inputnode, ds_bvals_mni, [('bvals_mni', 'in_file')]),
            (inputnode, ds_bvecs_mni, [('bvecs_mni', 'in_file')]),
            (inputnode, ds_mni_b0_ref, [('mni_b0_ref', 'in_file')]),
            (inputnode, ds_mni_b0_series, [('mni_b0_series', 'in_file')]),
            (inputnode, ds_dwi_mask_mni, [('dwi_mask_mni', 'in_file')]),
            (inputnode, ds_gradient_table_mni, [('gradient_table_mni', 'in_file')])
            ])
        # Local bvecs?
        if write_local_bvecs:
            ds_local_bvecs_mni = pe.Node(
                DerivativesDataSink(
                    prefix=output_prefix,
                    source_file=source_file,
                    base_directory=output_dir,
                    space=template,
                    suffix='bvec',
                    compress=True),
                name='ds_local_bvecs_mni',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([(inputnode, ds_local_bvecs_mni, [('local_bvecs_mni', 'in_file')])])
    return workflow


def _create_mem_gb(dwi_fname):
    dwi_size_gb = os.path.getsize(dwi_fname) / (1024**3)
    try:
        dwi_nvols = nb.load(dwi_fname).shape[3]
    except IndexError:
        dwi_nvols = 1
    except nb.filebasedimages.ImageFileError:
        LOGGER.warning("Zero-sized image")
        dwi_nvols = 1
    mem_gb = {
        'filesize': dwi_size_gb,
        'resampled': dwi_size_gb * 4,
        'largemem': dwi_size_gb * (max(dwi_nvols / 100, 1.0) + 4),
    }

    return dwi_nvols, mem_gb


def _get_wf_name(dwi_fname):
    """Derive the workflow name based on the output file prefix."""
    spl = dwi_fname.split("_")
    nosub = "_".join(spl[1:])
    return ("dwi_fsl_preproc_" + nosub + "_wf").replace("__", "_")


def _list_squeeze(in_list):
    squeezed = []
    for item in in_list:
        if type(item) is not str:
            squeezed.append(item[0])
        else:
            squeezed.append(item)
    return squeezed


def _get_first(in_list):
    return in_list[0]
