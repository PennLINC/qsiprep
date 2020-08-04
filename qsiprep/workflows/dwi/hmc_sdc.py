"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces.gradients import SliceQC, CombineMotions
from ...interfaces.images import SplitDWIs
from ..fieldmap.base import init_sdc_wf
from ...engine import Workflow

# dwi workflows
from .hmc import init_dwi_hmc_wf
from .util import _list_squeeze

LOGGER = logging.getLogger('nipype.workflow')


def init_qsiprep_hmcsdc_wf(scan_groups,
                           b0_threshold,
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
                           source_file,
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
                                         source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
                                         b0_threshold=100,
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
            fields=['dwi_file', 'bvec_file', 'bval_file', 'rpe_b0',
                    'original_files', 'rpe_b0_info', 'hmc_optimization_data', 't1_brain',
                    't1_2_mni_reverse_transform', 't1_mask', 't1_seg']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "b0_template_mask", "pre_sdc_template",
                    "hmc_optimization_data", "sdc_method", 'slice_quality', 'motion_params',
                    "cnr_map", "bvec_files_to_transform", "dwi_files_to_transform", "b0_indices",
                    "bval_files", "to_dwi_ref_affines", "to_dwi_ref_warps"]),
        name='outputnode')

    workflow = Workflow(name=name)

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

    # Split the input data into single volumes, put bvecs in LPS+ world reference frame
    split_dwis = pe.Node(SplitDWIs(b0_threshold=b0_threshold, deoblique_bvecs=True),
                         name='split_dwis')

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

    # Impute slice data if requested
    slice_qc = pe.Node(SliceQC(impute_slice_threshold=impute_slice_threshold), name="slice_qc")

    # Compute distance travelled to the template
    summarize_motion = pe.Node(CombineMotions(), name="summarize_motion")

    workflow.connect([
        (inputnode, split_dwis, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (split_dwis, dwi_hmc_wf, [
            ('dwi_files', 'inputnode.dwi_files'),
            ('bval_files', 'inputnode.bvals'),
            ('bvec_files', 'inputnode.bvecs'),
            ('b0_images', 'inputnode.b0_images'),
            ('b0_indices', 'inputnode.b0_indices')]),
        (inputnode, dwi_hmc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg')]),
        (split_dwis, slice_qc, [('dwi_files', 'uncorrected_dwi_files')]),
        (dwi_hmc_wf, outputnode, [
            ('outputnode.final_template', 'pre_sdc_template'),
            (('outputnode.forward_transforms', _list_squeeze),
             'to_dwi_ref_affines'),
            ('outputnode.optimization_data', "hmc_optimization_data"),
            ('outputnode.cnr_image', 'cnr_map'),
            ('outputnode.final_template_mask', 'b0_template_mask')]),
        (dwi_hmc_wf, summarize_motion, [
            ('outputnode.final_template', 'ref_file'),
            (('outputnode.forward_transforms', _list_squeeze), 'transform_files')]),
        (inputnode, summarize_motion, [('original_files', 'source_files')]),
        (dwi_hmc_wf, slice_qc, [
            ('outputnode.noise_free_dwis', 'ideal_image_files'),
            ('outputnode.final_template_mask', 'mask_image')]),
        (dwi_hmc_wf, b0_sdc_wf, [
            ('outputnode.final_template', 'inputnode.b0_ref'),
            ('outputnode.final_template_brain', 'inputnode.b0_ref_brain'),
            ('outputnode.final_template_mask', 'inputnode.b0_mask')]),
        (inputnode, b0_sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform',
             'inputnode.t1_2_mni_reverse_transform')]),
        (b0_sdc_wf, outputnode, [
            ('outputnode.method', 'sdc_method'),
            ('outputnode.b0_ref', 'b0_template'),
            ('outputnode.out_warp', 'to_dwi_ref_warps')]),
        (summarize_motion, outputnode, [('spm_motion_file', 'motion_params')]),
        (split_dwis, outputnode, [
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices')]),
        (slice_qc, outputnode, [
            ('slice_stats', 'slice_quality'),
            ('imputed_images', 'dwi_files_to_transform')]),
    ])

    return workflow


def _get_first(in_list):
    return in_list[0]
