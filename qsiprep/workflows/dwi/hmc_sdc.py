"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import os
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces.gradients import SliceQC, CombineMotions
from ..fieldmap.base import init_sdc_wf
from ...engine import Workflow

# dwi workflows
from .hmc import init_dwi_hmc_wf
from .util import init_dwi_reference_wf, _list_squeeze

LOGGER = logging.getLogger('nipype.workflow')


def init_qsiprep_hmcsdc_wf(scan_groups,
                           hmc_transform,
                           hmc_model,
                           hmc_align_to,
                           template,
                           shoreline_iters,
                           impute_slice_threshold,
                           omp_nthreads,
                           fmap_bspline,
                           fmap_demean,
                           use_syn,
                           force_syn,
                           dwi_metadata=None,
                           sloppy=False,
                           name='qsiprep_hmcsdc_wf'):
    """
    This workflow controls the head motion correction and susceptibility distortion
    correction parts of the qsiprep workflow. These parts have been combined because they're
    also combined in the eddy pipeline.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.shoreline import init_qsiprep_hmcsdc_wf
        wf = init_qsiprep_hmcsdc_wf({'dwi_series':[dwi1.nii, dwi2.nii],
                                          'fieldmap_info': {'suffix': None},
                                          'dwi_series_pedir': j},
                                         hmc_transform='Affine',
                                         hmc_model='3dSHORE',
                                         hmc_align_to='iterative',
                                         template='MNI152NLin2009cAsym',
                                         shoreline_iters=1,
                                         impute_slice_threshold=0,
                                         omp_nthreads=1,
                                         fmap_bspline=False,
                                         fmap_demean=False,
                                         use_syn=True,
                                         force_syn=False,
                                         name='qsiprep_hmcsdc_wf',
                                         sloppy=False,
                                         dwi_metadata={})
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'rpe_b0', 'b0_indices', 'bvec_files', 'bval_files', 'b0_images',
                    'original_files', 'rpe_b0_info', 'hmc_optimization_data', 't1_brain',
                    't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "b0_template_mask", "pre_sdc_template",
                    "hmc_optimization_data", "sdc_method", 'slice_quality', 'motion_params',
                    "cnr_map", "bvec_files_to_transform", "dwi_files_to_transform", "b0_indices",
                    "to_dwi_ref_affines", "to_dwi_ref_warps"]),
        name='outputnode')

    workflow = Workflow(name=name)

    dwi_series = scan_groups['dwi_series']
    source_file = dwi_series[0]
    fieldmap_info = scan_groups['fieldmap_info']
    # Run SyN if forced or in the absence of fieldmap correction
    fieldmap_type = fieldmap_info['suffix']

    if fieldmap_type is None:
        LOGGER.warning('SDC: no fieldmaps found or they were ignored (%s).',
                       source_file)
    elif fieldmap_type == 'syn':
        LOGGER.warning(
            'SDC: no fieldmaps found or they were ignored. '
            'Using EXPERIMENTAL "fieldmap-less SyN" correction '
            'for dataset %s.', source_file)
    else:
        LOGGER.log(25, 'SDC: fieldmap estimation of type "%s" intended for %s found.',
                   fieldmap_type, source_file)

    # Motion correct the data
    dwi_hmc_wf = init_dwi_hmc_wf(hmc_transform, hmc_model, hmc_align_to,
                                 source_file=source_file,
                                 num_model_iterations=shoreline_iters,
                                 sloppy=sloppy,
                                 omp_nthreads=omp_nthreads, name="dwi_hmc_wf")

    # Perform SDC if possible. This will pass-through if no sdc is to be done
    b0_sdc_wf = init_sdc_wf(
        scan_groups['fieldmap_info'], dwi_metadata, omp_nthreads=omp_nthreads,
        fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)
    b0_sdc_wf.inputs.inputnode.template = template

    # Create a b=0 reference for coregistration
    dwi_ref_wf = init_dwi_reference_wf(name="dwi_ref_wf", gen_report=True)

    # Impute slice data if requested
    slice_qc = pe.Node(SliceQC(impute_slice_threshold=impute_slice_threshold), name="slice_qc")

    # Compute distance travelled to the template
    summarize_motion = pe.Node(CombineMotions(), name="summarize_motion")

    workflow.connect([
        (inputnode, dwi_hmc_wf, [
            ('b0_images', 'inputnode.b0_images'),
            ('bval_files', 'inputnode.bvals'),
            ('bvec_files', 'inputnode.bvecs'),
            ('dwi_files', 'inputnode.dwi_files'),
            ('b0_indices', 'inputnode.b0_indices')]),
        (inputnode, slice_qc, [('dwi_files', 'uncorrected_dwi_files')]),
        (dwi_hmc_wf, outputnode, [
            ('outputnode.final_template', 'pre_sdc_template'),
            (('outputnode.forward_transforms', _list_squeeze),
             'to_dwi_ref_affines'),
            ('outputnode.optimization_data', "hmc_optimization_data"),
            ('outputnode.cnr_image', 'cnr_map')]),
        (dwi_hmc_wf, dwi_ref_wf, [
            ('outputnode.final_template', 'inputnode.b0_template')]),
        (dwi_hmc_wf, summarize_motion, [
            ('outputnode.final_template', 'ref_file'),
            (('outputnode.forward_transforms', _list_squeeze), 'transform_files')]),
        (inputnode, summarize_motion, [('dwi_files', 'source_files')]),
        (dwi_hmc_wf, slice_qc, [
            ('outputnode.noise_free_dwis', 'ideal_image_files')]),
        (dwi_ref_wf, b0_sdc_wf, [
            ('outputnode.ref_image', 'inputnode.b0_ref'),
            ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
            ('outputnode.dwi_mask', 'inputnode.b0_mask')]),
        (inputnode, b0_sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform',
             'inputnode.t1_2_mni_reverse_transform')]),
        (b0_sdc_wf, outputnode, [
            ('outputnode.method', 'sdc_method'),
            ('outputnode.b0_ref', 'b0_template'),
            ('outputnode.out_warp', 'to_dwi_ref_warps'),
            ('outputnode.b0_mask', 'b0_template_mask')]),
        (b0_sdc_wf, slice_qc, [('outputnode.b0_mask', 'mask_image')]),
        (summarize_motion, outputnode, [('spm_motion_file', 'motion_params')]),
        (inputnode, outputnode, [
            ('bvec_files', 'bvec_files_to_transform')]),
        (slice_qc, outputnode, [
            ('slice_stats', 'slice_quality'),
            ('imputed_images', 'dwi_files_to_transform')]),
    ])

    return workflow


def _get_first(in_list):
    return in_list[0]
