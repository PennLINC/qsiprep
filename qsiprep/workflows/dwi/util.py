# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_reference_wf
.. autofunction:: init_enhance_and_skullstrip_dwi_wf

"""
import os
import nibabel as nb

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni, ants
from ...niworkflows.interfaces import SimpleBeforeAfter
from ...niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from ...niworkflows.interfaces.masks import SimpleShowMaskRPT
from ...niworkflows.interfaces.registration import EstimateReferenceImage
from ...niworkflows.interfaces.utils import CopyXForm

from ...engine import Workflow
from ...interfaces import ValidateImage, MatchHeader
from ...interfaces.dipy import HistEQ, MedianOtsu
from ...interfaces.nilearn import MaskEPI
from ...interfaces.ants import ImageMath
from ...interfaces import DerivativesDataSink


DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_reference_wf(omp_nthreads=1, dwi_file=None, name='dwi_reference_wf',
                          gen_report=False, use_t1_prior=False):
    """
    This workflow generates reference dwi image for a series

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.util import init_dwi_reference_wf
        wf = init_dwi_reference_wf(omp_nthreads=1)

    **Parameters**

        dwi_file : str
            A b=0 image
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``dwi_reference_wf``)
        gen_report : bool
            Whether a mask report node should be appended in the end

    **Inputs**

        b0_template
            the b0 template used as the motion correction reference
        dwi_mask : bool
            A tentative brain mask to initialize the workflow (requires ``pre_mask``
            parameter set ``True``).

    **Outputs**

        raw_ref_image
            Reference image to which dwi series is motion corrected
        ref_image
            Contrast-enhanced reference image
        ref_image_brain
            Skull-stripped reference image
        dwi_mask
            Skull-stripping (rough) mask of reference image
        validation_report
            HTML reportlet indicating whether ``dwi_file`` had a valid affine


    **Subworkflows**

        * :py:func:`~qsiprep.workflows.dwi.util.init_enhance_and_skullstrip_wf`

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
First, a reference volume and its skull-stripped version were generated
using a modified version of the custom methodology of *fMRIPrep*.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['b0_template', 'dwi_mask', 't1_prior_mask']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'raw_ref_image', 'ref_image', 'bias_image',
                                      'ref_image_brain', 'dwi_mask', 'validation_report']),
        name='outputnode')

    # Simplify manually setting input image
    if dwi_file is not None:
        inputnode.inputs.b0_template = dwi_file

    enhance_and_skullstrip_dwi_wf = init_enhance_and_skullstrip_dwi_wf(
        omp_nthreads=omp_nthreads)

    skullstrip_b0_wf = init_skullstrip_b0_wf(use_initial_mask=True, use_t1_prior=use_t1_prior,
                                             name='skullstrip_b0_wf')

    workflow.connect([
        (inputnode, outputnode, [('b0_template', 'raw_ref_image')]),
        (inputnode, enhance_and_skullstrip_dwi_wf, [('b0_template', 'inputnode.in_file')]),
        (enhance_and_skullstrip_dwi_wf, skullstrip_b0_wf, [
            ('outputnode.bias_corrected_file', 'inputnode.in_file'),
            ('outputnode.mask_file', 'inputnode.initial_dwi_mask')]),
        (enhance_and_skullstrip_dwi_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image')]),
        (inputnode, skullstrip_b0_wf, [('t1_prior_mask', 'inputnode.t1_prior_mask')]),
        (skullstrip_b0_wf, outputnode, [
            ('outputnode.mask_file', 'dwi_mask'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')])
    ])

    if gen_report:
        b0ref_reportlet = pe.Node(SimpleBeforeAfter(), name='b0ref_reportlet', mem_gb=0.1)
        ds_report_b0_mask = pe.Node(
            DerivativesDataSink(desc="sdccoreg", suffix='b0ref'), name='ds_report_b0_mask',
            mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True
        )

        workflow.connect([
            (inputnode, b0ref_reportlet, [('b0_template', 'before')]),
            (enhance_and_skullstrip_dwi_wf, b0ref_reportlet, [
                ('outputnode.bias_corrected_file', 'after')]),
            (skullstrip_b0_wf, b0ref_reportlet, [
                ('outputnode.mask_file', 'wm_seg')]),
            (b0ref_reportlet, outputnode, [('out_report', 'validation_report')]),
            (b0ref_reportlet, ds_report_b0_mask, [('out_report', 'in_file')])
        ])

    return workflow


def init_enhance_and_skullstrip_dwi_wf(name='enhance_and_skullstrip_dwi_wf', omp_nthreads=1):
    """
    https://community.mrtrix.org/t/dwibiascorrect-with-ants-high-intensity-in-cerebellum-brainstem/1338/3

    Truncates image intensities, runs N4, does a pre-mask

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.util import init_enhance_and_skullstrip_dwi_wf
        wf = init_enhance_and_skullstrip_dwi_wf(omp_nthreads=1)

    **Parameters**
        name : str
            Name of workflow (default: ``enhance_and_skullstrip_dwi_wf``)
        pre_mask : bool
            Indicates whether the ``pre_mask`` input will be set (and thus, step 1
            should be skipped).
        omp_nthreads : int
            number of threads available to parallel nodes

    **Inputs**

        in_file
            dwi image (single volume)
        pre_mask : bool
            A tentative brain mask to initialize the workflow (requires ``pre_mask``
            parameter set ``True``).


    **Outputs**

        bias_corrected_file
            the ``in_file`` after N4BiasFieldCorrection and sharpening
        skull_stripped_file
            the ``bias_corrected_file`` after soft skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'pre_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'mask_file', 'skull_stripped_file', 'bias_corrected_file']), name='outputnode')

    # Truncate intensity values so they're OK for N4
    truncate_values = pe.Node(
        ImageMath(dimension=3,
                  operation="TruncateImageIntensity",
                  secondary_arg="0.001 0.999 512"),
        name="truncate_values")

    # N4 will break if any negative values are present.
    rescale_image = pe.Node(
        ImageMath(dimension=3,
                  operation="RescaleImage",
                  secondary_arg="0 1000"),
        name="rescale_image"
    )

    # Run N4 normally, force num_threads=1 for stability (images are small, no need for >1)
    n4_correct = pe.Node(
        ants.N4BiasFieldCorrection(
            dimension=3,
            n_iterations=[200, 200],
            convergence_threshold=1e-6,
            bspline_order=3,
            bspline_fitting_distance=150,
            copy_header=True),
        name='n4_correct', n_procs=1)

    # Sharpen the b0 ref
    sharpen_image = pe.Node(
        ImageMath(dimension=3,
                  operation="Sharpen"),
        name="sharpen_image")

    # Basic mask
    initial_mask = pe.Node(afni.Automask(outputtype="NIFTI_GZ"),
                           name="initial_mask")

    # Fill holes left by Automask
    fill_holes = pe.Node(
        ImageMath(dimension=3,
                  operation='FillHoles',
                  secondary_arg='2'),
        name='fill_holes')

    # Dilate before smoothing
    dilate_mask = pe.Node(
        ImageMath(dimension=3,
                  operation='MD',
                  secondary_arg='1'),
        name='dilate_mask')

    # Smooth the mask and use it as a weight for N4
    smooth_mask = pe.Node(
        ImageMath(dimension=3,
                  operation='G',
                  secondary_arg='4'),
        name='smooth_mask')

    # Make a "soft" skull-stripped image
    apply_mask = pe.Node(
        ants.MultiplyImages(dimension=3, output_product_image="SkullStrippedRef.nii.gz"),
        name="apply_mask")

    workflow.connect([
        (inputnode, truncate_values, [('in_file', 'in_file')]),
        (truncate_values, rescale_image, [('out_file', 'in_file')]),
        (rescale_image, initial_mask, [('out_file', 'in_file')]),
        (initial_mask, fill_holes, [('out_file', 'in_file')]),
        (fill_holes, dilate_mask, [('out_file', 'in_file')]),
        (dilate_mask, smooth_mask, [('out_file', 'in_file')]),
        (rescale_image, n4_correct, [('out_file', 'input_image')]),
        (smooth_mask, n4_correct, [('out_file', 'weight_image')]),
        (n4_correct, sharpen_image, [('output_image', 'in_file')]),
        (sharpen_image, outputnode, [('out_file', 'bias_corrected_file')]),
        (sharpen_image, apply_mask, [('out_file', 'first_input')]),
        (smooth_mask, apply_mask, [('out_file', 'second_input')]),
        (apply_mask, outputnode, [('output_product_image', 'skull_stripped_file')]),
        (fill_holes, outputnode, [('out_file', 'mask_file')])])

    return workflow


def init_skullstrip_b0_wf(name='skullstrip_b0_wf', use_t1_prior=False, use_initial_mask=False):
    """
    This workflow applies skull-stripping to a DWI image.

    It is intended to be used on an image that has previously been
    bias-corrected and enhanced with
    :py:func:`~qsiprep.workflows.dwi.util.init_enhance_and_skullstrip_dwi_wf`

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.bold.util import init_skullstrip_b0_wf
        wf = init_skullstrip_b0_wf()


    Inputs

        in_file
            b0 image (single volume)
        initial_dwi_mask
            A rough mask from a prior pipeline
        t1_prior_mask
            A brain mask from a co-registered T1 in the same voxel grid as in_file


    Outputs

        skull_stripped_file
            the ``in_file`` after skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 't1_prior_mask', 'initial_dwi_mask']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mask_file', 'skull_stripped_file', 'out_report']),
        name='outputnode')

    pad_image = pe.Node(
        ImageMath(dimension=3,
                  operation="PadImage",
                  secondary_arg="10"),
        name="pad_image")

    unpad_image = pe.Node(
        ImageMath(dimension=3,
                  operation="PadImage",
                  secondary_arg="-10"),
        name="unpad_image")

    if use_initial_mask:
        workflow.connect([(inputnode, pad_image, [('initial_dwi_mask', 'in_file')])])
    else:
        initial_mask = pe.Node(afni.Automask(outputtype="NIFTI_GZ"),
                               name="initial_mask")
        workflow.connect([
            (inputnode, initial_mask, [('in_file', 'in_file')]),
            (initial_mask, pad_image, [('out_file', 'in_file')])
        ])

    erode1 = pe.Node(
        ImageMath(dimension=3,
                  operation="ME",
                  secondary_arg="2"),
        name="erode1")

    get_largest = pe.Node(
        ImageMath(dimension=3,
                  operation="GetLargestComponent"),
        name='get_largest')

    dilate1 = pe.Node(
        ImageMath(dimension=3,
                  operation="MD",
                  secondary_arg="4"),
        name='dilate1')

    fill_holes = pe.Node(
        ImageMath(dimension=3,
                  operation="FillHoles",
                  secondary_arg="2"),
        name='fill_holes')

    dilate2 = pe.Node(
        ImageMath(dimension=3,
                  operation="MD",
                  secondary_arg="5"),
        name='dilate2')

    erode2 = pe.Node(
        ImageMath(dimension=3,
                  operation="ME",
                  secondary_arg="7"),
        name="erode2")

    apply_mask = pe.Node(fsl.ApplyMask(), name='apply_mask')

    # Do the prior-less parts
    workflow.connect([
        (pad_image, erode1, [('out_file', 'in_file')]),
        (erode1, get_largest, [('out_file', 'in_file')]),
        (get_largest, dilate1, [('out_file', 'in_file')]),
        (dilate1, fill_holes, [('out_file', 'in_file')])
    ])

    # Add in a t1 prior if requested
    if use_t1_prior:
        pad_t1 = pe.Node(
            ImageMath(dimension=3,
                      operation="PadImage",
                      secondary_arg="10"),
            name="pad_t1")
        add_t1_prior = pe.Node(
            ImageMath(dimension=3,
                      operation="addtozero"),
            name="add_t1_prior")
        workflow.connect([
            (inputnode, pad_t1, [('t1_prior_mask', 'in_file')]),
            (pad_t1, add_t1_prior, [('out_file', 'secondary_file')]),
            (fill_holes, add_t1_prior, [('out_file', 'in_file')]),
            (add_t1_prior, dilate2, [('out_file', 'in_file')])
        ])
    else:
        workflow.connect(fill_holes, 'out_file', dilate2, 'in_file')

    workflow.connect([
        (dilate2, erode2, [('out_file', 'in_file')]),
        (erode2, unpad_image, [('out_file', 'in_file')]),
        (unpad_image, outputnode, [('out_file', 'mask_file')]),
        (inputnode, apply_mask, [('in_file', 'in_file')]),
        (unpad_image, apply_mask, [('out_file', 'mask_file')]),
        (apply_mask, outputnode, [('out_file', 'skull_stripped_file')]),
    ])

    return workflow


def _create_mem_gb(dwi_fname):
    dwi_size_gb = os.path.getsize(dwi_fname) / (1024**3)
    try:
        dwi_nvols = nb.load(dwi_fname).shape[3]
    except IndexError:
        dwi_nvols = 1
    except nb.filebasedimages.ImageFileError:
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
