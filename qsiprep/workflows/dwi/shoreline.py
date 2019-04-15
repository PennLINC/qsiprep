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
from ...interfaces.gradients import SliceQC
from ...interfaces.confounds import DMRISummary
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from .hmc import init_dwi_hmc_wf
from .util import init_dwi_reference_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_shoreline_wf(dwi_file,
                      bval_file,
                      bvec_file,
                      output_prefix,
                      motion_corr_to,
                      hmc_model,
                      hmc_transform,
                      impute_slice_threshold,
                      reportlets_dir,
                      output_dir,
                      omp_nthreads,
                      low_mem=False,
                      layout=None):
    """
    This workflow implements the SHORELine motion correction method.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.shoreline import init_shoreline_wf
        wf = init_shoreline_wf('/completely/made/up/path/sub-01_dwi.nii.gz',
                               '/completely/made/up/path/sub-01_dwi.bval',
                               '/completely/made/up/path/sub-01_dwi.bvec',
                               omp_nthreads=1,
                               reportlets_dir='.',
                               output_dir='.',
                               motion_corr_to='iterative',
                               hmc_model='3dSHORE',
                               hmc_transform='Affine',
                               impute_slice_threshold=0,
                               low_mem=False,
                               output_prefix='')

    **Parameters**

        dwi_files : str or list
            List of dwi series NIfTI files to be combined or a dict of PE-dir -> files
        output_prefix : str
            beginning of the output file name (eg 'sub-1_buds-j')
        motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        hmc_model : 'none', '3dSHORE' or 'MAPMRI'
            Model used to generate target images for head motion correction. If 'none'
            the transform from the nearest b0 will be used.
        hmc_transform : "Rigid" or "Affine"
            Type of transform used for head motion correction
        impute_slice_threshold : float
            Impute data in slices that are this many SDs from expected. If 0, no slices
            will be imputed.
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        omp_nthreads : int
            Maximum number of threads an individual process may use
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage

    **Outputs**

        dwi_t1
            dwi series, resampled to T1w space
        dwi_mask_t1
            dwi series mask in T1w space
        bvals_t1
            bvalues of the dwi series
        bvecs_t1
            bvecs after aligning to the T1w and resampling
        gradient_table_t1
            MRTrix-style gradient table
        confounds_file
            estimated motion parameters and zipper scores

    **Subworkflows**

        * :py:func:`~qsiprep.workflows.dwi.util.init_dwi_reference_wf`
        * :py:func:`~qsiprep.workflows.dwi.hmc.init_dwi_hmc_wf`

    """
    # For naming outputs
    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    if not os.path.exists(dwi_file):
        raise FileExistsError(dwi_file)
    dwi_nvols, _mem_gb = _create_mem_gb(dwi_file)
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

    split_dwis = pe.Node(SplitDWIs(), name="split_dwis")
    dwi_hmc_wf = init_dwi_hmc_wf(hmc_transform, hmc_model, motion_corr_to,
                                 omp_nthreads=omp_nthreads, name="dwi_hmc_wf")

    dwi_ref_wf = init_dwi_reference_wf(name="dwi_ref_wf", gen_report=True)

    workflow.connect([
        # Merge, denoise, split, hmc on the dwis
        (split_dwis, buffernode,
         [('bval_files', 'bval_files'),
          ('bvec_files', 'bvec_files'),
          ('dwi_files', 'dwi_files'),
          ('b0_images', 'b0_images'),
          ('b0_indices', 'b0_indices')]),
        (split_dwis, dwi_hmc_wf, [('b0_images', 'inputnode.b0_images'),
                                  ('bval_files', 'inputnode.bvals'),
                                  ('bvec_files', 'inputnode.bvecs'),
                                  ('dwi_files', 'inputnode.dwi_files'),
                                  ('b0_indices', 'inputnode.b0_indices')]),
        (dwi_hmc_wf, buffernode, [(('outputnode.forward_transforms', _list_squeeze),
                                   'to_dwi_ref_affines'),
                                  ('outputnode.noise_free_dwis', 'ideal_images')]),
        (dwi_hmc_wf, dwi_ref_wf, [('outputnode.final_template', 'inputnode.b0_template')]),

    ])

    # At this point, buffernode is either a ConcatRPESplit or a single PE output
    summary = pe.Node(
        DiffusionSummary(
            pe_direction="",
            hmc_model=hmc_model,
            b0_to_t1w_transform="",
            hmc_transform=hmc_transform,
            impute_slice_threshold=impute_slice_threshold,
            dwi_denoise_window="N/A",
            output_spaces="Native"),
        name='summary',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True)

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

    # Compute and gather confounds
    slice_check = pe.Node(SliceQC(impute_slice_threshold=impute_slice_threshold),
                          name="slice_check")

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
        (buffernode, slice_check, [('ideal_images', 'ideal_image_files'),
                                   ('dwi_files', 'uncorrected_dwi_files'),
                                   ('b0_ref_mask', 'mask_image')]),
        (slice_check, confounds_wf, [('slice_stats', 'inputnode.sliceqc_file')]),
        (buffernode, confounds_wf, [('to_dwi_ref_affines', 'inputnode.hmc_affines'),
                                    ('bval_files', 'inputnode.bval_files'),
                                    ('bvec_files', 'inputnode.bvec_files'),
                                    ('original_grouping', 'inputnode.original_files')]),
        (confounds_wf, outputnode, [('outputnode.confounds_file', 'confounds')]),

        (confounds_wf, conf_plot, [('outputnode.confounds_file', 'confounds_file')]),
        (slice_check, conf_plot, [('slice_stats', 'sliceqc_file')]),
        (buffernode, conf_plot, []),
        (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
        (buffernode, gradient_plot, [('bvec_files', 'orig_bvec_files'),
                                     ('bval_files', 'orig_bval_files'),
                                     ('original_grouping', 'source_files')]),
        (gradient_plot, ds_report_gradients, [('plot_file', 'in_file')])

    ])


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
    return ("dwi_preproc_" + nosub + "_wf").replace("__", "_").replace("-", "_")


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
