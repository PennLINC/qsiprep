# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_reference_wf
.. autofunction:: init_enhance_and_skullstrip_dwi_wf

"""
from packaging.version import parse as parseversion, Version
from pkg_resources import resource_filename as pkgr_fn

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni, ants
from ...niworkflows.data import get_template
from ...niworkflows.interfaces.ants import AI
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

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_reference_wf(omp_nthreads=1, dwi_file=None, pre_mask=False,
                          name='dwi_reference_wf', gen_report=False):
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
            dwi series NIfTI file
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
            Skull-stripping mask of reference image
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
    inputnode = pe.Node(niu.IdentityInterface(fields=['b0_template', 'dwi_mask']),
                        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'raw_ref_image', 'ref_image',
                                      'ref_image_brain', 'dwi_mask', 'validation_report',
                                      'mask_report']),
        name='outputnode')

    # Simplify manually setting input image
    if dwi_file is not None:
        inputnode.inputs.b0_template = dwi_file

    enhance_and_skullstrip_dwi_wf = init_enhance_and_skullstrip_dwi_wf(
        omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, enhance_and_skullstrip_dwi_wf, [('b0_template', 'inputnode.in_file')]),
        (enhance_and_skullstrip_dwi_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image'),
            ('outputnode.mask_file', 'dwi_mask'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')]),
        (inputnode, outputnode, [('b0_template', 'raw_ref_image')])
    ])

    if gen_report:
        mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')
        workflow.connect([
            (enhance_and_skullstrip_dwi_wf, mask_reportlet, [
                ('outputnode.bias_corrected_file', 'background_file'),
                ('outputnode.mask_file', 'mask_file'),
            ]),
            (mask_reportlet, outputnode, [('out_report', 'validation_report')])
        ])

    return workflow


def init_enhance_and_skullstrip_dwi_wf(
        name='enhance_and_skullstrip_dwi_wf',
        omp_nthreads=1):
    """
    This workflow takes in a b0 template from head motion correction and sharpens the
    histogram with the application of the N4 algorithm for removing the
    :abbr:`INU (intensity non-uniformity)` bias field and calculates a signal
    mask.

    Steps of this workflow are:

      1. Us Dipy's ``median_otsu`` brain masking for a preliminary mask with a
         huge amount of dilation (8 voxels)
      2. Run ANTs' ``N4BiasFieldCorrection`` on the input
         :abbr:`dwi (blood-oxygen level-dependant)` average, using the
         mask generated in 1) instead of the internal Otsu thresholding.
      3. Apply Dipy's ``histeq`` to enhance the contrast of the data

    Step 1 can be skipped if the ``pre_mask`` argument is set to ``True`` and
    a tentative mask is passed in to the workflow throught the ``pre_mask``
    Nipype input.


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
            the ``in_file`` after `N4BiasFieldCorrection`_
        skull_stripped_file
            the ``bias_corrected_file`` after skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping

    .. _N4BiasFieldCorrection: https://hdl.handle.net/10380/3053
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'pre_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'mask_file', 'skull_stripped_file', 'bias_corrected_file']), name='outputnode')

    # Basic mask
    initial_mask = pe.Node(afni.Automask(dilate=3, outputtype="NIFTI_GZ"),
                           name="initial_mask")

    # Run N4 normally, force num_threads=1 for stability (images are small, no need for >1)
    n4_correct = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                         name='n4_correct', n_procs=1)

    hist_eq = pe.Node(HistEQ(), name='hist_eq')

    workflow.connect([
        (inputnode, initial_mask, [('in_file', 'in_file')]),
        (initial_mask, n4_correct, [('out_file', 'mask_image')]),
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (n4_correct, hist_eq, [('output_image', 'in_file')]),
        (initial_mask, hist_eq, [('out_file', 'mask_file')]),
        (hist_eq, outputnode, [('out_file', 'bias_corrected_file'),
                               ('out_file', 'skull_stripped_file')]),
        (initial_mask, outputnode, [('out_file', 'mask_file')]),
    ])

    return workflow


def init_skullstrip_b0_wf(name='skullstrip_b0_wf'):
    """
    This workflow applies skull-stripping to a DWI image.

    It is intended to be used on an image that has previously been
    bias-corrected with
    :py:func:`~qsiprep.workflows.bold.util.init_enhance_and_skullstrip_bold_wf`

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.bold.util import init_skullstrip_b0_wf
        wf = init_skullstrip_b0_wf()


    Inputs

        in_file
            b0 image (single volume)


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
        niu.IdentityInterface(fields=['in_file']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mask_file', 'skull_stripped_file', 'out_report']),
        name='outputnode')
    automask_dilate = pe.Node(
        afni.Automask(dilate=3, outputtype='NIFTI_GZ'),
        name='automask_dilate')
    apply_mask = pe.Node(fsl.ApplyMask(), name='apply_mask')
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, automask_dilate, [('in_file', 'in_file')]),
        (automask_dilate, outputnode, [('out_file', 'mask_file')]),
        # Masked file
        (inputnode, apply_mask, [('in_file', 'in_file')]),
        (automask_dilate, apply_mask, [('out_file', 'mask_file')]),
        (apply_mask, outputnode, [('out_file', 'skull_stripped_file')]),
        # Reportlet
        (inputnode, mask_reportlet, [('in_file', 'background_file')]),
        (automask_dilate, mask_reportlet, [('out_file', 'mask_file')]),
        (mask_reportlet, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow
