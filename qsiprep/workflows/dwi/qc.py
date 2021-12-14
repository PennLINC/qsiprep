# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_reference_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, afni
from ...engine import Workflow
from ...interfaces.dsi_studio import (
    DSIStudioSrcQC, DSIStudioCreateSrc, DSIStudioGQIReconstruction,
    DSIStudioFibQC, DSIStudioMergeQC)
from ...interfaces.reports import InteractiveReport
from ...interfaces.dipy import TensorReconstruction
from ...interfaces.anatomical import DiceOverlap

DEFAULT_MEMORY_MIN_GB = 0.01


def init_modelfree_qc_wf(dwi_files=None, name='dwi_qc_wf'):
    """
    This workflow runs DSI Studio's QC metrics

    **Parameters**

        dwi_files : list of DWI Files
            A b=0 image
        name : str
            Name of workflow (default: ``dwi_qc_wf``)

    **Inputs**
        dwi_file
            a single 4D dwi series
        bval_file
            bval file corresponding to the concatenated dwi_files inputs or dwi_file
        bvec_file
            bvec file corresponding to the concatenated dwi_files inputs or dwi_file

    **Outputs**

        qc file
            DSI Studio's src QC metrics for the input data


    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['qc_summary']),
        name='outputnode')

    raw_src = pe.Node(DSIStudioCreateSrc(), name='raw_src')
    raw_src_qc = pe.Node(DSIStudioSrcQC(), name='raw_src_qc')
    raw_gqi = pe.Node(DSIStudioGQIReconstruction(), name='raw_gqi')
    raw_fib_qc = pe.Node(DSIStudioFibQC(), name='raw_fib_qc')
    merged_qc = pe.Node(DSIStudioMergeQC(), name='merged_qc')
    workflow.connect([
        (inputnode, raw_src, [
            ('dwi_file', 'input_nifti_file'),
            ('bval_file', 'input_bvals_file'),
            ('bvec_file', 'input_bvecs_file')]),
        (raw_src, raw_src_qc, [('output_src', 'src_file')]),
        (raw_src, raw_gqi, [('output_src', 'input_src_file')]),
        (raw_gqi, raw_fib_qc, [('output_fib', 'src_file')]),
        (raw_fib_qc, merged_qc, [('qc_txt', 'fib_qc')]),
        (raw_src_qc, merged_qc, [('qc_txt', 'src_qc')]),
        (merged_qc, outputnode, [('qc_file', 'qc_summary')]),
    ])

    return workflow


def init_interactive_report_wf(name="interactive_report_wf"):
    """Create an interactive visual report for a web browser.

    **Inputs**
        dwi_file
            a single 4D dwi series
        bval_file
            bval file corresponding to the concatenated dwi_files inputs or dwi_file
        bvec_file
            bvec file corresponding to the concatenated dwi_files inputs or dwi_file

    **Outputs**

        qc file
            A JSON file that can be opened by dmriprep-viewer
    """

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "raw_dwi_file", "processed_dwi_file", "confounds_file", "bval_file",
            "bvec_file", "mask_file", "carpetplot_data", "series_qc_file"]),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["out_report"]), name="outputnode")
    interactive_report = pe.Node(InteractiveReport(), name='interactive_report')
    workflow = Workflow(name=name)
    tensor_fit = pe.Node(TensorReconstruction(), name='tensor_fit')
    workflow.connect([
        (inputnode, tensor_fit, [
            ('processed_dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('mask_file', 'mask_file')]),
        (tensor_fit, interactive_report, [('color_fa_image', 'color_fa')]),
        (inputnode, interactive_report, [
            ('series_qc_file', 'series_qc_file'),
            ('carpetplot_data', 'carpetplot_data'),
            ('raw_dwi_file', 'raw_dwi_file'),
            ('processed_dwi_file', 'processed_dwi_file'),
            ('confounds_file', 'confounds_file'),
            ('mask_file', 'mask_file')]),
        (interactive_report, outputnode, [('out_report', 'out_report')])
    ])
    return workflow


def init_mask_overlap_wf(name="mask_overlap_wf"):
    """Check the Dice overlap of a b=0 mask and a T1-based mask for QC.

    **Inputs**
        anatomical_mask
            Path to a high-resolution brain mask from a T1w image
        dwi_mask
            Path to a mask based on diffusion-weighted images

    **Outputs**
        dice_score
            float value of the dice overlap of the masks
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['anatomical_mask', 'dwi_mask']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['dice_score']), name='outputnode')

    downsample_t1_mask = pe.Node(afni.Resample(resample_mode="NN", outputtype="NIFTI_GZ"),
                                 name='downsample_t1_mask')
    calculate_dice = pe.Node(DiceOverlap(), name='calculate_dice')

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, downsample_t1_mask, [
            ('anatomical_mask', 'in_file'),
            ('dwi_mask', 'master')]),
        (inputnode, calculate_dice, [('dwi_mask', 'dwi_mask')]),
        (downsample_t1_mask, calculate_dice, [('out_file', 'anatomical_mask')]),
        (calculate_dice, outputnode, [('dice_score', 'dice_score')])])
    return workflow
