# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_reference_wf
.. autofunction:: init_enhance_and_skullstrip_dwi_wf
.. autofunction:: init_skullstrip_dwi_wf

"""
from packaging.version import parse as parseversion, Version
from pkg_resources import resource_filename as pkgr_fn

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni, ants
from niworkflows.data import get_template
from niworkflows.interfaces.ants import AI
from niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from niworkflows.interfaces.masks import SimpleShowMaskRPT
from niworkflows.interfaces.registration import EstimateReferenceImage
from niworkflows.interfaces.utils import CopyXForm

from ...engine import Workflow
from ...interfaces import ValidateImage, MatchHeader
from ...interfaces.dipy import HistEQ
from fmriprep.interfaces.nilearn import MaskEPI

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_reference_wf(omp_nthreads, dwi_file=None, pre_mask=False,
                          name='dwi_reference_wf', gen_report=False):
    """
    This workflow generates reference dwi image for a series

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.dwi import init_dwi_reference_wf
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

        dwi_file
            dwi series NIfTI file
        dwi_mask : bool
            A tentative brain mask to initialize the workflow (requires ``pre_mask``
            parameter set ``True``).

    **Outputs**

        dwi_file
            Validated dwi series NIfTI file
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

        * :py:func:`~fmriprep.workflows.dwi.util.init_enhance_and_skullstrip_wf`

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
        inputnode.inputs.dwi_file = dwi_file

    enhance_and_skullstrip_dwi_wf = init_enhance_and_skullstrip_dwi_wf(
        omp_nthreads=omp_nthreads, pre_mask=pre_mask)

    workflow.connect([
        (inputnode, enhance_and_skullstrip_dwi_wf, [('b0_template', 'inputnode.in_file')]),
        (enhance_and_skullstrip_dwi_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image'),
            ('outputnode.mask_file', 'dwi_mask'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')]),
    ])

    if gen_report:
        mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')
        workflow.connect([
            (enhance_and_skullstrip_dwi_wf, mask_reportlet, [
                ('outputnode.bias_corrected_file', 'background_file'),
                ('outputnode.mask_file', 'mask_file'),
            ]),
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

      1. Us Nilearn's brain masking for a preliminary mask
      3. Run ANTs' ``N4BiasFieldCorrection`` on the input
         :abbr:`dwi (blood-oxygen level-dependant)` average, using the
         mask generated in 1) instead of the internal Otsu thresholding.
      4. Calculate a loose mask using FSL's ``bet``, with one mathematical morphology
         dilation of one iteration and a sphere of 6mm as structuring element.
      5. Mask the :abbr:`INU (intensity non-uniformity)`-corrected image
         with the latest mask calculated in 3), then use AFNI's ``3dUnifize``
         to *standardize* the T2* contrast distribution.
      6. Calculate a mask using AFNI's ``3dAutomask`` after the contrast
         enhancement of 4).
      7. Calculate a final mask as the intersection of 4) and 6).
      8. Apply final mask on the enhanced reference.

    Step 1 can be skipped if the ``pre_mask`` argument is set to ``True`` and
    a tentative mask is passed in to the workflow throught the ``pre_mask``
    Nipype input.


    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.dwi.util import init_enhance_and_skullstrip_dwi_wf
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

    # Basic mask from Nilearn
    initial_mask = pe.Node(MaskEPI(enhance_t2=False, lower_cutoff=0.1, upper_cutoff=0.9),
                           name="initial_mask")
    # Dilate pre_mask
    pre_dilate = pe.Node(fsl.DilateImage(
        operation='max', kernel_shape='sphere', kernel_size=3.0,
        internal_datatype='char'), name='pre_mask_dilate')

    # Run N4 normally, force num_threads=1 for stability (images are small, no need for >1)
    n4_correct = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                         name='n4_correct', n_procs=1)

    hist_eq = pe.Node(HistEQ(), name='hist_eq')

    workflow.connect([
        (inputnode, initial_mask, [('in_file', 'in_files')]),
        (initial_mask, pre_dilate, [('out_mask', 'in_file')]),
        (pre_dilate, n4_correct, [('out_file', 'mask_image')]),
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (n4_correct, hist_eq, [('output_image', 'in_file')]),
        (hist_eq, outputnode, [('output_file', 'bias_corrected_file'),
                               ('output_file', 'skull_stripped_file')]),
        (pre_dilate, outputnode, [('out_file', 'mask_file')])
    ])

    return workflow
