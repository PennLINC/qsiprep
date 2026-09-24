.. include:: ../links.rst

.. _anatomical_methods:

#####################
Anatomical processing
#####################

:func:`qsiprep.workflows.anatomical.volume.init_anat_preproc_wf`

*QSIPrep* uses the anatomical reference for three things: a brain mask, a
tissue segmentation that the reports draw over the distortion correction
results, and the definition of the subject's ``ACPC`` space that every
output is written in. It also normalizes the reference to a template,
so that atlases can be brought into the subject's space later.

The reference can be the T1w or the T2w (``--anat-modality``). Whichever is
chosen, the steps are:

1. **Conform.** Every image of the reference contrast is reoriented to LPS+
   and resampled to a common voxel size.
2. **Merge.** If there are several images, they are N4-corrected and aligned
   to one another. With ``--subject-anatomical-reference first-lex`` (the
   default) they are registered to the first image in lexical order; with
   ``unbiased`` an unbiased mid-point template is built instead. The extra
   cost of the unbiased template is small for two images and about an order
   of magnitude for three or more.
3. **Brain extraction** with SynthStrip :footcite:p:`synthstrip`.

   .. figure:: ../_static/brainextraction_t1.svg
       :scale: 100%

4. **Tissue segmentation** with SynthSeg :footcite:p:`synthseg1`, giving the
   tissue-class ``dseg`` and the regional ``desc-aseg_dseg`` derivatives.

   .. figure:: ../_static/segmentation.svg
       :scale: 100%

   If the other contrast is present too (a T2w alongside a T1w reference),
   it is registered to the reference with an affine ``antsRegistration``,
   initialized by an ``antsAI`` rotation search, and written as an
   additional derivative.

5. **AC-PC alignment.** The reference is rigidly registered to the template,
   and the rigid transform defines ``ACPC`` space: the origin at the anterior
   commissure and the axes aligned with the template's, but with the
   subject's own brain shape and size. All outputs are written in this space.
6. **Spatial normalization.** ``antsRegistration`` registers the reference to
   the template in a multi-scale, mutual-information based, nonlinear scheme.
   The forward and inverse transforms are written with the anatomical
   derivatives. ``--skip-anat-based-spatial-normalization`` skips this step.

   .. figure:: ../_static/T1MNINormalization.svg
       :scale: 100%

       T1w to MNI normalization.

The preprocessed reference defines ``ACPC`` space. With several input
images, that space is not exactly aligned with any one of them; the
``from-orig_to-anat`` transforms map each input into it.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep_docs import ANATOMICAL_TEMPLATE
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    wf = init_anat_preproc_wf(
        num_anat_images=1,
        num_additional_t2ws=0,
        has_rois=False,
        anatomical_template=ANATOMICAL_TEMPLATE,
    )


*******
Lesions
*******

For patients with focal lesions (stroke, tumor resection), a binary lesion
mask (1 inside the lesion) in the same space and resolution as the T1w,
named ``sub-<label>_label-lesion_roi.nii.gz`` in ``anat/``, is used as a
mask during spatial normalization :footcite:p:`brett2001`, so that healthy
tissue is not warped into the lesion or the other way around.


********
Sessions
********

By default the anatomical images of all sessions are merged into one
reference and one ``ACPC`` space per subject. ``--subject-anatomical-reference
sessionwise`` processes each session with its own reference, its own space
and its own report, which is required with ``--infant`` and appropriate
whenever the anatomy changes between sessions.


***********
Infant data
***********

``--infant`` replaces the adult template with the MNIInfant cohort matching
the participant's age (see :ref:`running` for where the age is read from).
The cohort selection follows Nibabies. Brain extraction and segmentation are
the same as for adults.


*******************
No anatomical image
*******************

With ``--anat-modality none`` there is no anatomical workflow. The b=0
reference is registered directly to the template and only the rigid part of
that registration is kept, which gives an AC-PC aligned b=0 that keeps the
shape and size of the original. There is no tissue segmentation in the
reports and no template normalization.


**********
References
**********

.. footbibliography::
