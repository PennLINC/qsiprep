# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_bold_surf_wf
.. autofunction:: init_bold_mni_trans_wf
.. autofunction:: init_bold_preproc_trans_wf

"""
import os.path as op


from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import Split as FSLSplit

from niworkflows import data as nid
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from fmriprep.engine import Workflow
from ...interfaces import MultiApplyTransforms
from ...interfaces.nilearn import Merge
from ...interfaces.gradients import WarpAndRecombineDWIs
from ..anatomical import TEMPLATE_MAP

from .util import init_bold_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_t1_trans_wf(mem_gb, omp_nthreads,
                         name='dwi_t1_trans_wf',
                         use_compression=True,
                         use_fieldwarp=True,
                         interpolation='LanczosWindowedSinc'):
    """
    This workflow resamples the input DWI after it's been aligned to its
    t1 reference image in a "single shot" from the original DWI series.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.bold import init_bold_preproc_trans_wf
        wf = init_bold_preproc_trans_wf(mem_gb=3, omp_nthreads=1)

    **Parameters**

        mem_gb : float
            Size of DWI file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_mni_trans_wf``)
        use_compression : bool
            Save registered DWI series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from DWI to MNI
        interpolation : str
            Interpolation type to be used by ANTs' ``applyTransforms``
            (default ``'LanczosWindowedSinc'``)

    **Inputs**

        dwi_files
            Individual 3D volumes, not motion corrected
        dwi_mask
            Skull-stripping mask of reference image
        b0_reference
            b0 reference image
        target_file
            Image defining the resampling grid
        name_source
            DWI series NIfTI file
            Used to recover original information lost during processing
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        fieldwarp
            a :abbr:`DFM (displacements field map)` in ITK format


    **Outputs**

        dwi
            DWI series, resampled in t1 space, including all preprocessing
        dwi_mask
            DWI series mask calculated with the new time-series
        bvals
            bvals file for the DWI series
        bvecs
            bvecs file for the DWI series
        local_bvecs
            bvecs defined in every voxel
        b0_ref
            DWI reference image: an average-like 3D image of the time-series
        b0_ref_brain
            Same as ``b0_ref``, but once the brain mask has been applied
        b0_series
            4d image of concatenated preprocessed b0 image

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The DWI time-series (including slice-timing correction when applied)
were resampled into an isotropic grid aligned with the t1 by applying
{transforms}.
These resampled DWI time-series will be referred to as *preprocessed
DWI in original T1 space*, or just *preprocessed DWI*.
""".format(transforms="""\
a single, composite transform to correct for head-motion and
susceptibility distortions""" if use_fieldwarp else """\
the transforms to correct for head-motion""")

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'b0_ref_image', 'b0_ref_mask', 'dwi_files', 'bvec_files', 'b0_images', 'b0_indices',
        'to_dwi_ref_affines', 'to_dwi_ref_warps', 'original_grouping', 'target_file',
        'b0_ref_2_t1_affines']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi', 'dwi_mask', 'bvals', 'bvecs', 'local_bvecs',
                              'b0_ref', 'b0_ref_brain', 'b0_series']), name='outputnode')

    transform_dwis = pe.Node(WarpAndRecombineDWIs(), name='transform_dwis')

    workflow.connect([
        (inputnode, transform_dwis, [('b0_ref_image', 'b0_ref_image'),
        ('b0_ref_mask', 'b0_ref_mask'), ('dwi_files', 'dwi_files'), ('bvec_files', 'bvec_files'),
        ('b0_images', 'b0_images'), ('b0_indices', 'b0_indices'),
        ('to_dwi_ref_affines', 'to_dwi_ref_affines'), ('to_dwi_ref_warps', 'to_dwi_ref_warps'),
        ('original_grouping', 'original_grouping')]),
        (bold_transform, merge, [('out_files', 'in_files')]),
        (merge, bold_reference_wf, [('out_file', 'inputnode.bold_file')]),
        (merge, outputnode, [('out_file', 'bold')]),
        (bold_reference_wf, outputnode, [
            ('outputnode.ref_image', 'bold_ref'),
            ('outputnode.ref_image_brain', 'bold_ref_brain'),
            ('outputnode.bold_mask', 'bold_mask')]),
    ])

    workflow.connect([
        (inputnode, bold_transform, [('bold_file', 'input_image'),
                                     (('bold_file', _first), 'reference_image')])
    ])

    if use_fieldwarp:
        merge_xforms = pe.Node(niu.Merge(2), name='merge_xforms',
                               run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, merge_xforms, [('fieldwarp', 'in1'),
                                       ('hmc_xforms', 'in2')]),
            (merge_xforms, bold_transform, [('out', 'transforms')]),
        ])
    else:
        def _aslist(val):
            return [val]
        workflow.connect([
            (inputnode, bold_transform, [(('hmc_xforms', _aslist), 'transforms')]),
        ])

    # Code ready to generate a pre/post processing report
    # bold_bold_report_wf = init_bold_preproc_report_wf(
    #     mem_gb=mem_gb['resampled'],
    #     reportlets_dir=reportlets_dir
    # )
    # workflow.connect([
    #     (inputnode, bold_bold_report_wf, [
    #         ('bold_file', 'inputnode.name_source'),
    #         ('bold_file', 'inputnode.in_pre')]),  # This should be after STC
    #     (bold_bold_trans_wf, bold_bold_report_wf, [
    #         ('outputnode.bold', 'inputnode.in_post')]),
    # ])

    return workflow


def init_dwi_mni_trans_wf(template, mem_gb, omp_nthreads,
                          name='bold_mni_trans_wf',
                          template_out_grid='2mm',
                          use_compression=True,
                          use_fieldwarp=False):
    """
    This workflow samples functional images to the MNI template in a "single shot"
    from the original DWI series.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.bold import init_bold_mni_trans_wf
        wf = init_bold_mni_trans_wf(template='MNI152NLin2009cAsym',
                                    mem_gb=3,
                                    omp_nthreads=1,
                                    template_out_grid='native')

    **Parameters**

        template : str
            Name of template targeted by ``template`` output space
        mem_gb : float
            Size of DWI file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_mni_trans_wf``)
        template_out_grid : str
            Keyword ('native', '1mm' or '2mm') or path of custom reference
            image for normalization.
        use_compression : bool
            Save registered DWI series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from DWI to MNI

    **Inputs**

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        bold_split
            Individual 3D volumes, not motion corrected
        bold_mask
            Skull-stripping mask of reference image
        name_source
            DWI series NIfTI file
            Used to recover original information lost during processing
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        fieldwarp
            a :abbr:`DFM (displacements field map)` in ITK format

    **Outputs**

        bold_mni
            DWI series, resampled to template space
        bold_mni_ref
            Reference, contrast-enhanced summary of the DWI series, resampled to template space
        bold_mask_mni
            DWI series mask in template space

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The DWI time-series were resampled to {tpl} standard space,
generating a *preprocessed DWI run in {tpl} space*.
""".format(tpl=template)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_bold_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'bold_split',
            'bold_mask',
            'hmc_xforms',
            'fieldwarp'
        ]),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_mni', 'bold_mni_ref', 'bold_mask_mni']),
        name='outputnode')

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref',
                      mem_gb=0.3)  # 256x256x256 * 64 / 8 ~ 150MB)
    template_str = TEMPLATE_MAP[template]
    gen_ref.inputs.fixed_image = op.join(nid.get_dataset(template_str), '1mm_T1.nii.gz')

    mask_mni_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True),
        name='mask_mni_tfm',
        mem_gb=1
    )

    # Write corrected file in the designated output dir
    mask_merge_tfms = pe.Node(niu.Merge(2), name='mask_merge_tfms', run_without_submitting=True,
                              mem_gb=DEFAULT_MEMORY_MIN_GB)

    nxforms = 4 if use_fieldwarp else 3
    merge_xforms = pe.Node(niu.Merge(nxforms), name='merge_xforms',
                           run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([(inputnode, merge_xforms, [('hmc_xforms', 'in%d' % nxforms)])])

    if use_fieldwarp:
        workflow.connect([(inputnode, merge_xforms, [('fieldwarp', 'in3')])])

    workflow.connect([
        (inputnode, gen_ref, [(('bold_split', _first), 'moving_image')]),
        (inputnode, mask_mni_tfm, [('bold_mask', 'input_image')]),
        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('itk_bold_to_t1', _aslist), 'in2')]),
        (mask_merge_tfms, mask_mni_tfm, [('out', 'transforms')]),
        (mask_mni_tfm, outputnode, [('output_image', 'bold_mask_mni')]),
    ])

    bold_to_mni_transform = pe.Node(
        MultiApplyTransforms(interpolation="LanczosWindowedSinc", float=True, copy_dtype=True),
        name='bold_to_mni_transform', mem_gb=mem_gb * 3 * omp_nthreads, n_procs=omp_nthreads)

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=mem_gb * 3)

    # Generate a reference on the target T1w space
    gen_final_ref = init_bold_reference_wf(
        omp_nthreads=omp_nthreads, pre_mask=True)

    workflow.connect([
        (inputnode, merge_xforms, [('t1_2_mni_forward_transform', 'in1'),
                                   (('itk_bold_to_t1', _aslist), 'in2')]),
        (merge_xforms, bold_to_mni_transform, [('out', 'transforms')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, bold_to_mni_transform, [('bold_split', 'input_image')]),
        (bold_to_mni_transform, merge, [('out_files', 'in_files')]),
        (merge, gen_final_ref, [('out_file', 'inputnode.bold_file')]),
        (mask_mni_tfm, gen_final_ref, [('output_image', 'inputnode.bold_mask')]),
        (merge, outputnode, [('out_file', 'bold_mni')]),
        (gen_final_ref, outputnode, [('outputnode.ref_image', 'bold_mni_ref')]),
    ])

    if template_out_grid == 'native':
        workflow.connect([
            (gen_ref, mask_mni_tfm, [('out_file', 'reference_image')]),
            (gen_ref, bold_to_mni_transform, [('out_file', 'reference_image')]),
        ])
    elif template_out_grid == '1mm' or template_out_grid == '2mm':
        mask_mni_tfm.inputs.reference_image = op.join(
            nid.get_dataset(template_str), '%s_brainmask.nii.gz' % template_out_grid)
        bold_to_mni_transform.inputs.reference_image = op.join(
            nid.get_dataset(template_str), '%s_T1.nii.gz' % template_out_grid)
    else:
        mask_mni_tfm.inputs.reference_image = template_out_grid
        bold_to_mni_transform.inputs.reference_image = template_out_grid
    return workflow


def _first(inlist):
    return inlist[0]
