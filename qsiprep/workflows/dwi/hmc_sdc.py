"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import ants, utility as niu

from .registration import init_b0_to_anat_registration_wf, init_direct_b0_acpc_wf
from ...interfaces.gradients import SliceQC, CombineMotions, GradientRotation
from ...interfaces.images import SplitDWIsBvals, TSplit
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf
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
                           raw_image_sdc,
                           fmap_demean,
                           pepolar_method,
                           source_file,
                           t2w_sdc,
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

        from qsiprep.workflows.dwi.shoreline import init_qsiprep_hmcsdc_coreg_wf
        wf = init_qsiprep_hmcsdc_wf(
            {'dwi_series':[dwi1.nii, dwi2.nii],
             'fieldmap_info': {'suffix': None},
             'dwi_series_pedir': j},
            source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
            b0_threshold=100,
            hmc_transform='Affine',
            hmc_model='3dSHORE',
            hmc_align_to='iterative',
            template='MNI152NLin2009cAsym',
            shoreline_iters=1,
            raw_image_sdc=True,
            impute_slice_threshold=0,
            pepolar_method='DRBUDDI',
            omp_nthreads=1,
            fmap_bspline=False,
            fmap_demean=False,
            name='qsiprep_hmcsdc_wf',
            sloppy=False,
            t2w_sdc=False,
            dwi_metadata={})
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_file', 'bvec_file', 'bval_file', 'rpe_b0', 't2w_unfatsat',
                    'original_files', 'rpe_b0_info', 'hmc_optimization_data', 't1_brain',
                    't1_2_mni_reverse_transform', 't1_mask','t1_seg','t2_brain']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "b0_template_mask", "pre_sdc_template",
                    "hmc_optimization_data", "sdc_method", 'slice_quality', 'motion_params',
                    "cnr_map", "bvec_files_to_transform", "dwi_files_to_transform", "b0_indices",
                    "bval_files", "to_dwi_ref_affines", "to_dwi_ref_warps", "sdc_scaling_images",
                    # From SDC
                    "fieldmap_type", "b0_up_image", "b0_up_corrected_image", "b0_down_image",
                    "b0_down_corrected_image", "up_fa_image", "up_fa_corrected_image", "down_fa_image",
                    "down_fa_corrected_image", "t2w_image"]),
        name='outputnode')

    workflow = Workflow(name=name)

    # Split the input data into single volumes, put bvecs in LPS+ world reference frame
    split_dwis = pe.Node(TSplit(digits=4, out_name='vol'),name='split_dwis')
    split_bvals = pe.Node(SplitDWIsBvals(b0_threshold=b0_threshold, deoblique_bvecs=True),name='split_bvals')

    # Motion correct the data
    dwi_hmc_wf = init_dwi_hmc_wf(hmc_transform, hmc_model, hmc_align_to,
                                 source_file=source_file,
                                 num_model_iterations=shoreline_iters,
                                 sloppy=sloppy,
                                 omp_nthreads=omp_nthreads, name="dwi_hmc_wf")

    # Impute slice data if requested
    slice_qc = pe.Node(SliceQC(impute_slice_threshold=impute_slice_threshold), name="slice_qc")

    # Compute distance travelled to the template
    summarize_motion = pe.Node(CombineMotions(), name="summarize_motion")

    workflow.connect([
        (inputnode, split_dwis, [
            ('dwi_file', 'in_file')]),
        (inputnode, split_bvals, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (split_dwis, split_bvals, [
            ('out_files','split_files')]),
        (split_bvals, dwi_hmc_wf, [
            ('bval_files', 'inputnode.bvals'),
            ('bvec_files', 'inputnode.bvecs'),
            ('b0_images', 'inputnode.b0_images'),
            ('b0_indices', 'inputnode.b0_indices')]),
        (split_dwis, dwi_hmc_wf, [
            ('out_files', 'inputnode.dwi_files')]),
        (inputnode, dwi_hmc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('original_files', 'inputnode.original_files')]),
        (split_dwis, slice_qc, [('out_files', 'uncorrected_dwi_files')]),
        (dwi_hmc_wf, outputnode, [
            ('outputnode.final_template', 'pre_sdc_template'),
            (('outputnode.forward_transforms', _list_squeeze),
             'to_dwi_ref_affines'),
            ('outputnode.optimization_data', "hmc_optimization_data"),
            ('outputnode.cnr_image', 'cnr_map'),
            ('outputnode.final_template_mask', 'b0_template_mask')]),
        (dwi_hmc_wf, summarize_motion, [
            ('outputnode.final_template', 'ref_file'),
            ('outputnode.final_template', 'source_files'),
            (('outputnode.forward_transforms', _list_squeeze), 'transform_files')]),
        (dwi_hmc_wf, slice_qc, [
            ('outputnode.noise_free_dwis', 'ideal_image_files'),
            ('outputnode.final_template_mask', 'mask_image')]),
        (summarize_motion, outputnode, [('spm_motion_file', 'motion_params')]),
        (split_bvals, outputnode, [
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices')]),
        (slice_qc, outputnode, [
            ('slice_stats', 'slice_quality'),
            ('imputed_images', 'dwi_files_to_transform')]),
    ])

    fieldmap_info = scan_groups['fieldmap_info']
    # Run SyN if forced or in the absence of fieldmap correction
    fieldmap_type = fieldmap_info['suffix']

    if fieldmap_type in ('epi', 'rpe_series'):
        if pepolar_method.lower() == "topup":
            raise Exception("TOPUP is not supported with SHORELine ")

        drbuddi_wf = init_drbuddi_wf(scan_groups=scan_groups, omp_nthreads=omp_nthreads,
                                     b0_threshold=b0_threshold, raw_image_sdc=raw_image_sdc,
                                     sloppy=sloppy, t2w_sdc=t2w_sdc)

        # apply the head motion correction transforms
        apply_hmc_transforms = pe.MapNode(
            ants.ApplyTransforms(dimension=3,
                                 interpolation='LanczosWindowedSinc'),
            iterfield=['input_image', 'reference_image', 'transforms'],
            name='uncorrect_model_images')
        rotate_gradients = pe.Node(GradientRotation(), name='rotate_gradients')
        workflow.connect([
            (dwi_hmc_wf, apply_hmc_transforms, [
                ('outputnode.forward_transforms', 'transforms')]),
            (dwi_hmc_wf, rotate_gradients, [
                (('outputnode.forward_transforms', _list_squeeze),
                 'affine_transforms')]),
            (split_dwis, apply_hmc_transforms, [
                ('out_files', 'input_image'),
                ('out_files', 'reference_image')]),
            (split_bvals, rotate_gradients, [
                ('bvec_files', 'bvec_files'),
                ('bval_files', 'bval_files')]),
            (apply_hmc_transforms, drbuddi_wf, [
                ('output_image', 'inputnode.dwi_files')]),
            (rotate_gradients, drbuddi_wf, [
                ('bvals', 'inputnode.bval_files'),
                ('bvecs', 'inputnode.bvec_files')]),
            (inputnode, drbuddi_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t2w_unfatsat', 'inputnode.t2w_unfatsat'),
                ('original_files', 'inputnode.original_files')]),
            (drbuddi_wf, outputnode, [
                ('outputnode.sdc_warps', 'to_dwi_ref_warps'),
                ('outputnode.sdc_scaling_images', 'sdc_scaling_images'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.fieldmap_type', 'fieldmap_type'),
                ('outputnode.b0_up_image', 'b0_up_image'),
                ('outputnode.b0_up_corrected_image', 'b0_up_corrected_image'),
                ('outputnode.b0_down_image', 'b0_down_image'),
                ('outputnode.b0_down_corrected_image', 'b0_down_corrected_image'),
                ('outputnode.up_fa_image', 'up_fa_image'),
                ('outputnode.up_fa_corrected_image', 'up_fa_corrected_image'),
                ('outputnode.down_fa_image', 'down_fa_image'),
                ('outputnode.down_fa_corrected_image', 'down_fa_corrected_image'),
                ('outputnode.t2w_image', 't2w_image'),
                ('outputnode.b0_ref', 'b0_template')])
        ])
        return workflow


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

    # Perform SDC if possible. This will pass-through if no sdc is to be done
    b0_sdc_wf = init_sdc_wf(
        scan_groups['fieldmap_info'], dwi_metadata, omp_nthreads=omp_nthreads,
        fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)
    b0_sdc_wf.inputs.inputnode.template = template

    workflow.connect([
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
            ('outputnode.out_warp', 'to_dwi_ref_warps')])
        ])

    return workflow


def _get_first(in_list):
    return in_list[0]


def _list_squeeze(in_list):
    from typing import Iterable
    from pathlib import Path
    def flatten(items):
        """Yield items from any nested iterable; see
        Beazley, D. and B. Jones. Recipe 4.14, Python Cookbook 3rd Ed.,
        O'Reilly Media Inc. Sebastopol, CA: 2013..
        https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        """
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes, Path)):
                for sub_x in flatten(x):
                    yield sub_x
            else:
                yield x
    return list(flatten(in_list))
