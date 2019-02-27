# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_dwi_trans_wf

"""
import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, utility as niu, ants

from ...engine import Workflow
from ...interfaces.nilearn import Merge
from ...interfaces.gradients import (ComposeTransforms, ExtractB0s, GradientRotation,
                                     LocalGradientRotation)
from ...interfaces.itk import DisassembleTransform
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_trans_wf(template, mem_gb, omp_nthreads,
                      name='dwi_trans_wf',
                      use_compression=True,
                      use_fieldwarp=False,
                      to_mni=False,
                      write_local_bvecs=False):
    """
    This workflow samples dwi images to the ``output_grid`` in a "single shot"
    from the original DWI series.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf
        wf = init_dwi_trans_wf(template='MNI152NLin2009cAsym',
                               mem_gb=3,
                               omp_nthreads=1)

    **Parameters**

        template : str
            Name of template targeted by ``template`` output space
        mem_gb : float
            Size of DWI file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``dwi_trans_wf``)
        use_compression : bool
            Save registered DWI series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from DWI to MNI
        to_mni : bool
            Include warps to MNI
        write_local_bvecs : bool
            if true, local bvec niftis are written

    **Inputs**

        itk_b0_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        dwi_files
            Individual 3D volumes, not motion corrected
        bval_files
            individual bval files
        bvec_files
            one-lined bvec files
        b0_ref_image
            b0 template for the dwi series
        b0_indices
            List of indices that contain a b0 image
        dwi_mask
            Skull-stripping mask of reference image
        name_source
            DWI series NIfTI file
            Used to recover original information lost during processing
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        fieldwarps
            a :abbr:`DFM (displacements field map)` in ITK format
        output_grid
            File defining the output space

    **Outputs**

        dwi_resampled
            DWI series, resampled to template space
        dwi_ref_resampled
            Reference, contrast-enhanced summary of the DWI series, resampled to template space
        dwi_mask_resampled
            DWI series mask in template space
        bvals
            bvals file for the DWI series
        rotated_bvecs
            bvecs rotated for transforms to ``output_grid``
        local_bvecs
            NIfTI file containing the bvec rotation matrix (due to transforms) in each voxel.
            Includes rotations introduced by warping
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The DWI time-series were resampled to {tpl},
generating a *preprocessed DWI run in {tpl} space*.
""".format(tpl=template)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_b0_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'dwi_files',
            'bval_files',
            'bvec_files',
            'b0_ref_image',
            'b0_indices',
            'dwi_mask',
            'hmc_xforms',
            'fieldwarps',
            'output_grid'
        ]),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_resampled',
            'dwi_ref_resampled',
            'dwi_mask_resampled',
            'bvals',
            'rotated_bvecs',
            'local_bvecs',
            'b0_series']),
        name='outputnode')

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    # get composite warps and composed affines for warping and rotating
    compose_transforms = pe.Node(ComposeTransforms(), name='compose_transforms')

    workflow.connect([
        (inputnode, compose_transforms, [
            ('b0_indices', 'original_b0_indices'),
            ('output_grid', 'reference_image'),
            ('dwi_files', 'dwi_files'),
            ('hmc_xforms', 'hmc_to_ref_affines'),
            (('itk_b0_to_t1', _first), 'unwarped_dwi_ref_to_t1w_affine'),
            ])
    ])

    if use_fieldwarp:
        workflow.connect([(inputnode, compose_transforms, [
            ('fieldwarps', 'fieldwarps')])])

    # Rotate the bvecs
    rotate_gradients = pe.Node(GradientRotation(), name='rotate_gradients')

    # Transform the mask to the output space
    mask_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='MultiLabel', float=True),
        name='mask_tfm',
        mem_gb=1)

    if to_mni:
        # Disassemble the to-mni transform if it's a h5 (it should be!)
        disassemble_mni_xform = pe.Node(DisassembleTransform(), name='disassemble_mni_xform')

        # Write corrected file in the designated output dir
        mask_merge_tfms = pe.Node(niu.Merge(2), name='mask_merge_tfms',
                                  run_without_submitting=True,
                                  mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, disassemble_mni_xform, [('t1_2_mni_forward_transform',
                                                 'in_file')]),
            (disassemble_mni_xform, compose_transforms, [('out_transforms',
                                                          't1_2_mni_forward_transform')]),
            (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                          (('itk_b0_to_t1', _aslist), 'in2')]),
            (mask_merge_tfms, mask_tfm, [('out', 'transforms')]),
        ])
    else:
        workflow.connect([
            (inputnode, mask_tfm, [(('itk_b0_to_t1', _aslist), 'transforms')]),
        ])

    dwi_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc", float=True),
        name='dwi_transform', iterfield=['input_image', 'transforms'])

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=mem_gb * 3)

    extract_b0_series = pe.Node(ExtractB0s(), name="extract_b0_series")

    workflow.connect([
        (inputnode, rotate_gradients, [('bvec_files', 'bvec_files'),
                                       ('bval_files', 'bval_files')]),
        (compose_transforms, rotate_gradients, [('out_affines', 'affine_transforms')]),
        (rotate_gradients, outputnode, [('bvals', 'bvals'),
                                        ('bvecs', 'rotated_bvecs')]),
        (inputnode, mask_tfm, [('dwi_mask', 'input_image'),
                               ('output_grid', 'reference_image')]),
        (mask_tfm, outputnode, [('output_image', 'dwi_mask_resampled')]),
        (compose_transforms, dwi_transform, [('out_warps', 'transforms')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode,  dwi_transform, [('dwi_files', 'input_image'),
                                     ('output_grid', 'reference_image')]),
        (dwi_transform, merge, [('output_image', 'in_files')]),
        (merge, outputnode, [('out_file', 'dwi_resampled')]),
        (merge, extract_b0_series, [('out_file', 'dwi_series')]),
        (inputnode, extract_b0_series, [('b0_indices', 'b0_indices')]),
        (extract_b0_series, outputnode, [('b0_series', 'b0_series'),
                                         ('b0_average', 'dwi_ref_resampled')]),
    ])

    if write_local_bvecs:
        local_grad_rotation = pe.Node(LocalGradientRotation(), name='local_grad_rotation')
        workflow.connect([
            (compose_transforms, local_grad_rotation, [('out_warps', 'warp_transforms')]),
            (mask_tfm, local_grad_rotation, [('output_image', 'mask_image')]),
            (inputnode, local_grad_rotation, [('bvec_files', 'bvec_files')]),
            (local_grad_rotation, outputnode, [('local_bvecs', 'local_bvecs')])
        ])

    return workflow


def _first(inlist):
    return inlist[0]
