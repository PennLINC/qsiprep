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

from .util import init_dwi_reference_wf
from ...engine import Workflow
from ...interfaces.nilearn import Merge
from ...interfaces.gradients import (ComposeTransforms, ExtractB0s, GradientRotation,
                                     LocalGradientRotation, SplitIntramodalTransform)
from ...interfaces.itk import DisassembleTransform

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_trans_wf(template,
                      mem_gb,
                      omp_nthreads,
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
        cnr_map
            Contrast to noise map from model-based hmc
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
        t1_mask
            Brain mask from the t1w

    **Outputs**

        dwi_resampled
            DWI series, resampled to template space
        dwi_ref_resampled
            Reference, contrast-enhanced summary of the DWI series, resampled to template space
        dwi_mask_resampled
            DWI series mask in template space
        cnr_map_resampled
            Contrast to noise map resampled
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
            't1_mask',
            'b0_to_intramodal_template_transforms',
            'intramodal_template_to_t1_warp',
            'intramodal_template_to_t1_affine',
            't1_2_mni_forward_transform',
            'name_source',
            'dwi_files',
            'cnr_map',
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
            'cnr_map_resampled',
            'bvals',
            'resampled_dwi_mask',
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

    def _get_first(lll):
        from nipype.interfaces.base import isdefined
        if isdefined(lll):
            return lll[0]
        return lll

    workflow.connect([
        (inputnode, compose_transforms, [
            ('output_grid', 'reference_image'),
            ('dwi_files', 'dwi_files'),
            ('hmc_xforms', 'hmc_affines'),
            ('itk_b0_to_t1', 'hmcsdc_dwi_ref_to_t1w_affine'),
            ('fieldwarps', 'fieldwarps'),
            ('b0_to_intramodal_template_transforms', 'b0_to_intramodal_template_transforms'),
            (('intramodal_template_to_t1_affine', _get_first), 'intramodal_template_to_t1_affine'),
            ('intramodal_template_to_t1_warp', 'intramodal_template_to_t1_warp'),
            ])
    ])

    # Rotate the bvecs
    rotate_gradients = pe.Node(GradientRotation(), name='rotate_gradients')

    cnr_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
        name='cnr_tfm',
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
            (mask_merge_tfms, cnr_tfm, [('out', 'transforms')])
        ])
    else:
        workflow.connect([
            (compose_transforms, cnr_tfm, [(('out_warps', _get_first), 'transforms')])
        ])

    def _get_first(items):
        return items[0]

    dwi_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc", float=True),
        name='dwi_transform', iterfield=['input_image', 'transforms'])

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=mem_gb * 3)

    extract_b0_series = pe.Node(ExtractB0s(), name="extract_b0_series")

    # Use the T1w to make a final mask
    resample_t1_mask = pe.Node(
        ants.ApplyTransforms(dimension=3,
                             transforms='identity',
                             interpolation="MultiLabel"),
        name='resample_t1_mask')
    final_b0_ref = init_dwi_reference_wf(use_t1_prior=True)

    workflow.connect([
        (inputnode, rotate_gradients, [('bvec_files', 'bvec_files'),
                                       ('bval_files', 'bval_files')]),
        (compose_transforms, rotate_gradients, [('out_affines', 'affine_transforms')]),
        (rotate_gradients, outputnode, [('bvals', 'bvals'),
                                        ('bvecs', 'rotated_bvecs')]),
        (inputnode, cnr_tfm, [('cnr_map', 'input_image'),
                              ('output_grid', 'reference_image')]),
        (cnr_tfm, outputnode, [('output_image', 'cnr_map_resampled')]),
        (compose_transforms, dwi_transform, [('out_warps', 'transforms')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, dwi_transform, [('dwi_files', 'input_image'),
                                    ('output_grid', 'reference_image')]),
        (dwi_transform, merge, [('output_image', 'in_files')]),
        (merge, outputnode, [('out_file', 'dwi_resampled')]),
        (merge, extract_b0_series, [('out_file', 'dwi_series')]),
        (inputnode, extract_b0_series, [('b0_indices', 'b0_indices')]),
        (extract_b0_series, final_b0_ref, [('b0_average', 'inputnode.b0_template')]),
        (extract_b0_series, resample_t1_mask, [('b0_average', 'reference_image')]),
        (inputnode, resample_t1_mask, [('t1_mask', 'input_image')]),
        (resample_t1_mask, final_b0_ref, [('output_image', 'inputnode.t1_prior_mask')]),
        (final_b0_ref, outputnode, [
            ('outputnode.ref_image', 'dwi_ref_resampled'),
            ('outputnode.dwi_mask', 'resampled_dwi_mask')]),
        (extract_b0_series, outputnode, [('b0_series', 'b0_series')])])

    if write_local_bvecs:
        local_grad_rotation = pe.Node(LocalGradientRotation(), name='local_grad_rotation')
        workflow.connect([
            (compose_transforms, local_grad_rotation, [('out_warps', 'warp_transforms')]),
            (inputnode, local_grad_rotation, [('bvec_files', 'bvec_files')]),
            (local_grad_rotation, outputnode, [('local_bvecs', 'local_bvecs')])
        ])

    return workflow


def _first(inlist):
    return inlist[0]
