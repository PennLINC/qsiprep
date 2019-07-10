"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf

"""

import os

from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import DiffusionSummary, GradientPlot
from ...interfaces.confounds import DMRISummary
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from ..fieldmap.unwarp import init_fmap_unwarp_report_wf
from .hmc_sdc import init_qsiprep_hmcsdc_wf
from .fsl import init_fsl_hmc_wf
from .pre_hmc import init_dwi_pre_hmc_wf
from .util import _create_mem_gb, _get_wf_name
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf
from .derivatives import init_dwi_derivatives_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_preproc_wf(scan_groups,
                        output_prefix,
                        ignore,
                        b0_threshold,
                        motion_corr_to,
                        b0_to_t1w_transform,
                        hmc_model,
                        hmc_transform,
                        shoreline_iters,
                        impute_slice_threshold,
                        eddy_config,
                        reportlets_dir,
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
                        layout=None):
    """
    This workflow controls the dwi preprocessing stages of qsiprep.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.base import init_dwi_preproc_wf
        wf = init_dwi_preproc_wf({'dwi_series': ['fake.nii'],
                                  'fieldmap_info': {'type': None},
                                  'dwi_series_pedir': 'j'},
                                 omp_nthreads=1,
                                 ignore=[],
                                 b0_threshold=100,
                                 reportlets_dir='.',
                                 output_dir='.',
                                 template='MNI152NLin2009cAsym',
                                 output_spaces=['T1w', 'template'],
                                 dwi_denoise_window=7,
                                 denoise_before_combining=True,
                                 motion_corr_to='iterative',
                                 b0_to_t1w_transform='Rigid',
                                 hmc_model='3dSHORE',
                                 hmc_transform='Affine',
                                 shoreline_iters=2,
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

        dwi_groups : list of dicts
            List of dicts grouping files by PE-dir
        output_prefix : str
            beginning of the output file name (eg 'sub-1_buds-j')
        ignore : list
            Preprocessing steps to skip (eg "fieldmaps")
        b0_threshold : int
            Images with b-values less than this value will be treated as a b=0 image.
        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister) and
            resampling dwi series to FreeSurfer surface meshes.
        motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        b0_to_t1w_transform : "Rigid" or "Affine"
            Use a rigid or full affine transform for b0-T1w registration
        hmc_model : 'none', '3dSHORE', 'eddy' or 'eddy_ingress'
            Model used to generate target images for head motion correction. If 'none'
            the transform from the nearest b0 will be used.
        hmc_transform : "Rigid" or "Affine"
            Type of transform used for head motion correction
        impute_slice_threshold : float
            Impute data in slices that are this many SDs from expected. If 0, no slices
            will be imputed.
        eddy_config: str
            Path to a JSON file containing config options for eddy
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
    # Check the inputs
    if layout is not None:
        all_dwis = scan_groups['dwi_series']
        source_file = all_dwis[0]
        fieldmap_info = scan_groups['fieldmap_info']
        dwi_metadata = layout.get_metadata(source_file)
    else:
        all_dwis = ['/fake/testing/path.nii.gz']
        source_file = all_dwis[0]
        fieldmap_info = {'type': None}
        dwi_metadata = {}

    fieldmap_type = fieldmap_info['type']
    doing_bidirectional_pepolar = fieldmap_type == 'rpe_series'
    preprocess_rpe_series = doing_bidirectional_pepolar and hmc_model == 'eddy'
    if fieldmap_type is not None:
        fmap_key = "phase1" if fieldmap_type == "phase" else fieldmap_type
        fieldmap_file = fieldmap_info[fmap_key]
        fieldmap_info['metadata'] = layout.get_metadata(fieldmap_file)

    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    dwi_nvols = 10

    # Determine resource usage
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
    LOGGER.log(25, ('Creating dwi processing workflow "%s" '
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
            'dwi_t1', 'dwi_mask_t1', 'cnr_map_t1', 'bvals_t1', 'bvecs_t1', 'local_bvecs_t1',
            't1_b0_ref', 't1_b0_series', 'dwi_mni', 'dwi_mask_mni', 'cnr_map_mni', 'bvals_mni',
            'bvecs_mni', 'local_bvecs_mni', 'mni_b0_ref', 'mni_b0_series', 'confounds',
            'gradient_table_mni', 'gradient_table_t1', 'hmc_optimization_data'
        ]),
        name='outputnode')

    pre_hmc_wf = init_dwi_pre_hmc_wf(scan_groups=scan_groups,
                                     b0_threshold=b0_threshold,
                                     preprocess_rpe_series=preprocess_rpe_series,
                                     dwi_denoise_window=dwi_denoise_window,
                                     low_mem=low_mem,
                                     denoise_before_combining=denoise_before_combining,
                                     omp_nthreads=omp_nthreads)

    if hmc_model in ('none', '3dSHORE', 'SH'):
        if not hmc_model == 'none' and shoreline_iters < 1:
            raise Exception("--shoreline-iters must be > 0 when --hmc-model is " + hmc_model)
        hmc_wf = init_qsiprep_hmcsdc_wf(
            scan_groups=scan_groups,
            hmc_transform=hmc_transform,
            hmc_model=hmc_model,
            hmc_align_to=motion_corr_to,
            template=template,
            shoreline_iters=shoreline_iters,
            impute_slice_threshold=impute_slice_threshold,
            omp_nthreads=omp_nthreads,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            use_syn=use_syn,
            force_syn=force_syn,
            dwi_metadata=dwi_metadata,
            name="hmc_wf")

    elif hmc_model == 'eddy':
        hmc_wf = init_fsl_hmc_wf(
            scan_groups=scan_groups,
            b0_threshold=b0_threshold,
            impute_slice_threshold=impute_slice_threshold,
            eddy_config=eddy_config,
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            dwi_metadata=dwi_metadata,
            name="hmc_wf")

    workflow.connect([
        (pre_hmc_wf, hmc_wf, [
            ('outputnode.dwi_files', 'inputnode.dwi_files'),
            ('outputnode.bval_files', 'inputnode.bval_files'),
            ('outputnode.bvec_files', 'inputnode.bvec_files'),
            ('outputnode.original_files', 'inputnode.original_files'),
            ('outputnode.b0_images', 'inputnode.b0_images'),
            ('outputnode.b0_indices', 'inputnode.b0_indices')]),
        (inputnode, hmc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform')])
    ])

    # calculate dwi registration to T1w
    b0_coreg_wf = init_b0_to_anat_registration_wf(omp_nthreads=omp_nthreads,
                                                  mem_gb=mem_gb['resampled'],
                                                  write_report=True)

    # Make a fieldmap report
    if fieldmap_type is not None:
        fmap_unwarp_report_wf = init_fmap_unwarp_report_wf()
        workflow.connect([
            (inputnode, fmap_unwarp_report_wf, [
                ('t1_seg', 'inputnode.in_seg')]),
            (hmc_wf, fmap_unwarp_report_wf, [
                ('outputnode.pre_sdc_template', 'inputnode.in_pre'),
                ('outputnode.b0_template', 'inputnode.in_post')]),
            (b0_coreg_wf, fmap_unwarp_report_wf, [
                ('outputnode.itk_b0_to_t1', 'inputnode.in_xfm')])])

    summary = pe.Node(
        DiffusionSummary(
            pe_direction=scan_groups['dwi_series_pedir'],
            hmc_model=hmc_model,
            b0_to_t1w_transform=b0_to_t1w_transform,
            hmc_transform=hmc_transform,
            impute_slice_threshold=impute_slice_threshold,
            dwi_denoise_window=dwi_denoise_window,
            output_spaces=output_spaces),
        name='summary',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True)

    workflow.connect([
        (inputnode, b0_coreg_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1_2_fsnative_reverse_transform',
             'inputnode.t1_2_fsnative_reverse_transform')]),
        (hmc_wf, b0_coreg_wf, [('outputnode.b0_template',
                                'inputnode.ref_b0_brain')]),
        (hmc_wf, summary, [('outputnode.sdc_method', 'distortion_correction')])
    ])

    gradient_plot = pe.Node(GradientPlot(), name='gradient_plot', run_without_submitting=True)
    ds_report_gradients = pe.Node(
        DerivativesDataSink(suffix='sampling_scheme'),
        name='ds_report_gradients', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # CONNECT TO DERIVATIVES #####################
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        output_prefix=output_prefix,
        source_file=source_file,
        output_dir=output_dir,
        output_spaces=output_spaces,
        template=template,
        write_local_bvecs=write_local_bvecs,
        hmc_model=hmc_model,
        shoreline_iters=shoreline_iters)

    workflow.connect([
        (inputnode, dwi_derivatives_wf, [('dwi_files', 'inputnode.source_file')]),
        (outputnode, dwi_derivatives_wf,
         [('dwi_t1', 'inputnode.dwi_t1'),
          ('dwi_mask_t1', 'inputnode.dwi_mask_t1'),
          ('cnr_map_t1', 'inputnode.cnr_map_t1'),
          ('bvals_t1', 'inputnode.bvals_t1'),
          ('bvecs_t1', 'inputnode.bvecs_t1'),
          ('local_bvecs_t1', 'inputnode.local_bvecs_t1'),
          ('t1_b0_ref', 'inputnode.t1_b0_ref'),
          ('t1_b0_series', 'inputnode.t1_b0_series'),
          ('gradient_table_t1', 'inputnode.gradient_table_t1'),
          ('dwi_mni', 'inputnode.dwi_mni'),
          ('dwi_mask_mni', 'inputnode.dwi_mask_mni'),
          ('cnr_map_mni', 'inputnode.cnr_map_mni'),
          ('bvals_mni', 'inputnode.bvals_mni'),
          ('bvecs_mni', 'inputnode.bvecs_mni'),
          ('local_bvecs_mni', 'inputnode.local_bvecs_mni'),
          ('mni_b0_ref', 'inputnode.mni_b0_ref'),
          ('mni_b0_series', 'inputnode.mni_b0_series'),
          ('gradient_table_mni', 'inputnode.gradient_table_mni'),
          ('confounds', 'inputnode.confounds'),
          ('hmc_optimization_data', 'inputnode.hmc_optimization_data')])
    ])

    # Compute and gather confounds
    confounds_wf = init_dwi_confs_wf(mem_gb=mem_gb['resampled'], metadata=[],
                                     impute_slice_threshold=impute_slice_threshold,
                                     name='confounds_wf')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(DMRISummary(), name='conf_plot', mem_gb=mem_gb['resampled'])
    ds_report_dwi_conf = pe.Node(
        DerivativesDataSink(suffix='carpetplot'),
        name='ds_report_dwi_conf', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (hmc_wf, confounds_wf, [
            ('outputnode.slice_quality', 'inputnode.sliceqc_file'),
            ('outputnode.motion_params', 'inputnode.motion_params')]),
        (pre_hmc_wf, confounds_wf, [
            ('outputnode.bval_files', 'inputnode.bval_files'),
            ('outputnode.bvec_files', 'inputnode.bvec_files'),
            ('outputnode.original_files', 'inputnode.original_files')]),
        (confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),
        (hmc_wf, outputnode, [('outputnode.hmc_optimization_data', 'hmc_optimization_data')]),
        (hmc_wf, conf_plot, [
            ('outputnode.slice_quality', 'sliceqc_file'),
            ('outputnode.b0_template_mask', 'sliceqc_mask')]),
        (confounds_wf, conf_plot, [
            ('outputnode.confounds_file', 'confounds_file')]),
        (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
        (pre_hmc_wf, gradient_plot, [
            ('outputnode.bvec_files', 'orig_bvec_files'),
            ('outputnode.bval_files', 'orig_bval_files'),
            ('outputnode.original_files', 'source_files')]),
        (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])
    ])

    if "T1w" in output_spaces:
        transform_dwis_t1 = init_dwi_trans_wf(name='transform_dwis_t1',
                                              template="ACPC",
                                              mem_gb=mem_gb['resampled'],
                                              use_fieldwarp=(fieldmap_type is not None
                                                             or use_syn),
                                              omp_nthreads=omp_nthreads,
                                              use_compression=False,
                                              to_mni=False,
                                              write_local_bvecs=write_local_bvecs
                                              )
        gtab_t1 = pe.Node(MRTrixGradientTable(), name='gtab_t1')
        workflow.connect([
            (pre_hmc_wf, transform_dwis_t1, [
                ('outputnode.b0_indices', 'inputnode.b0_indices'),
                ('outputnode.bval_files', 'inputnode.bval_files')]),
            (hmc_wf, transform_dwis_t1, [
                ('outputnode.bvec_files_to_transform', 'inputnode.bvec_files'),
                ('outputnode.b0_template', 'inputnode.b0_ref_image'),
                ('outputnode.cnr_map', 'inputnode.cnr_map'),
                ('outputnode.b0_template_mask', 'inputnode.dwi_mask'),
                ('outputnode.to_dwi_ref_affines', 'inputnode.hmc_xforms'),
                ('outputnode.to_dwi_ref_warps', 'inputnode.fieldwarps'),
                ('outputnode.dwi_files_to_transform', 'inputnode.dwi_files')]),
            (inputnode, transform_dwis_t1, [
                ('dwi_sampling_grid', 'inputnode.output_grid')]),
            (b0_coreg_wf, transform_dwis_t1, [
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1')]),
            (transform_dwis_t1, outputnode, [('outputnode.bvals', 'bvals_t1'),
                                             ('outputnode.rotated_bvecs', 'bvecs_t1'),
                                             ('outputnode.dwi_resampled', 'dwi_t1'),
                                             ('outputnode.cnr_map_resampled', 'cnr_map_t1'),
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
                                               use_fieldwarp=(fieldmap_type is not None
                                                              or use_syn),
                                               omp_nthreads=omp_nthreads,
                                               use_compression=False,
                                               to_mni=True,
                                               write_local_bvecs=write_local_bvecs)
        gtab_mni = pe.Node(MRTrixGradientTable(), name='gtab_mni')
        workflow.connect([
            (pre_hmc_wf, transform_dwis_mni, [
                ('outputnode.b0_indices', 'inputnode.b0_indices'),
                ('outputnode.bval_files', 'inputnode.bval_files')]),
            (hmc_wf, transform_dwis_mni, [
                ('outputnode.bvec_files_to_transform', 'inputnode.bvec_files'),
                ('outputnode.b0_template', 'inputnode.b0_ref_image'),
                ('outputnode.cnr_map', 'inputnode.cnr_map'),
                ('outputnode.b0_template_mask', 'inputnode.dwi_mask'),
                ('outputnode.to_dwi_ref_affines', 'inputnode.hmc_xforms'),
                ('outputnode.to_dwi_ref_warps', 'inputnode.fieldwarps'),
                ('outputnode.dwi_files_to_transform', 'inputnode.dwi_files')]),
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
    ds_report_summary = pe.Node(
        DerivativesDataSink(suffix='summary'),
        name='ds_report_summary',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (summary, ds_report_summary, [('out_report', 'in_file')])
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = source_file
    return workflow
