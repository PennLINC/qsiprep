# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

"""

from nipype.interfaces import afni
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.anatomical import DiceOverlap
from ...interfaces.dsi_studio import (
    DSIStudioCreateSrc,
    DSIStudioFibQC,
    DSIStudioGQIReconstruction,
    DSIStudioMergeQC,
    DSIStudioSrcQC,
)

DEFAULT_MEMORY_MIN_GB = 0.01


def init_modelfree_qc_wf(bvec_convention='DIPY', name='dwi_qc_wf'):
    """
    This workflow runs DSI Studio's QC metrics

    Parameters
    ----------
    bvec_convention : "DIPY", "FSL" or "auto"
        What kind of bvecs
    name : str
        Name of workflow (default: ``dwi_qc_wf``)

    Inputs
    ------
    dwi_file
        a single 4D dwi series
    bval_file
        bval file corresponding to the concatenated dwi_files inputs or dwi_file
    bvec_file
        bvec file corresponding to the concatenated dwi_files inputs or dwi_file

    Outputs
    -------
    qc file
        DSI Studio's src QC metrics for the input data


    """
    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['qc_summary']), name='outputnode')

    raw_src = pe.Node(
        DSIStudioCreateSrc(
            bvec_convention='FSL' if bvec_convention == 'auto' else bvec_convention
        ),
        name='raw_src',
        n_procs=omp_nthreads,
    )
    raw_src_qc = pe.Node(DSIStudioSrcQC(), name='raw_src_qc', n_procs=omp_nthreads)
    raw_gqi = pe.Node(
        DSIStudioGQIReconstruction(
            thread_count=omp_nthreads, check_btable=1 if bvec_convention == 'auto' else 0
        ),
        name='raw_gqi',
        n_procs=omp_nthreads,
    )
    raw_fib_qc = pe.Node(DSIStudioFibQC(), name='raw_fib_qc', n_procs=omp_nthreads)
    merged_qc = pe.Node(DSIStudioMergeQC(), name='merged_qc', n_procs=omp_nthreads)
    workflow.connect([
        (inputnode, raw_src, [
            ('dwi_file', 'input_nifti_file'),
            ('bval_file', 'input_bvals_file'),
            ('bvec_file', 'input_bvecs_file'),
        ]),
        (raw_src, raw_src_qc, [('output_src', 'src_file')]),
        (raw_src, raw_gqi, [('output_src', 'input_src_file')]),
        (raw_gqi, raw_fib_qc, [('output_fib', 'src_file')]),
        (raw_fib_qc, merged_qc, [('qc_txt', 'fib_qc')]),
        (raw_src_qc, merged_qc, [('qc_txt', 'src_qc')]),
        (merged_qc, outputnode, [('qc_file', 'qc_summary')]),
    ])  # fmt:skip

    return workflow


def init_mask_overlap_wf(name='mask_overlap_wf'):
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
        niu.IdentityInterface(fields=['anatomical_mask', 'dwi_mask']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['dice_score']), name='outputnode')

    downsample_t1_mask = pe.Node(
        afni.Resample(resample_mode='NN', outputtype='NIFTI_GZ'), name='downsample_t1_mask'
    )
    calculate_dice = pe.Node(DiceOverlap(), name='calculate_dice')

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, downsample_t1_mask, [
            ('anatomical_mask', 'in_file'),
            ('dwi_mask', 'master'),
        ]),
        (inputnode, calculate_dice, [('dwi_mask', 'dwi_mask')]),
        (downsample_t1_mask, calculate_dice, [('out_file', 'anatomical_mask')]),
        (calculate_dice, outputnode, [('dice_score', 'dice_score')]),
    ])  # fmt:skip
    return workflow
