.. include:: ../links.rst

.. _resampling_methods:

#############################
Coregistration and resampling
#############################

After head motion and distortion correction, the DWI is registered to the
anatomical reference and resampled once into ``ACPC`` space.


.. _dwi_ref:

*******************
DWI reference image
*******************

:func:`qsiprep.workflows.dwi.util.init_dwi_reference_wf`

A reference image is built from the b=0 volumes of the corrected series,
with a generous brain mask. It is the coregistration target, the image the
fieldmap-less methods register, and the ``desc-preproc_dwiref`` derivative.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep_docs import AP
    from qsiprep.workflows.dwi.util import init_dwi_reference_wf

    wf = init_dwi_reference_wf(source_file=AP, gen_report=True)


.. _b0_reg:

******************************
Registration to the anatomical
******************************

:func:`qsiprep.workflows.dwi.registration.init_b0_to_anat_registration_wf`

The reference is registered to the skull-stripped anatomical reference with
``antsRegistration``, rigidly by default or with an affine transform under
``--dwi2anat-dof 12``. Its result is the ``desc-coreg`` transform.

The registration is initialized by a global rotation search with
``antsAI``. A mutual-information registration reliably recovers rotations
of a few tens of degrees; a larger difference in head orientation between
the dMRI and the anatomical scan, which is common in infants, converges to a
wrong local optimum. ``antsAI`` therefore tries a grid of candidate
orientations (three Euler angles at 20 degree spacing, up to 81 degrees per
axis) on 4 mm resamples of the two images, refines each briefly, and hands
the best one to ``antsRegistration`` as the starting transform. The same
search initializes the T2w-to-T1w registration in the anatomical workflow
and the alignment of a T2w into the b=0 frame before DRBUDDI and T2Wreg.

With ``--anat-modality none`` the reference is registered to the template
instead, with a similarity transform in the rotation search since the scale
is unknown, and only the rigid part of the result is kept.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf

    wf = init_b0_to_anat_registration_wf()

Subject-level DWI reference
===========================

:func:`qsiprep.workflows.dwi.dwiref.init_dwiref_wf`

With ``--dwiref-definition subject``, the references of all the subject's
distortion groups are first combined into one mid-point template with
``antsMultivariateTemplateConstruction2``, using the transform and number of
iterations from ``--dwiref-construction-transform`` and
``--dwiref-construction-iters``. These registrations are initialized from
the images' centers of mass, which absorbs table-position shifts between
sessions; no rotation search is run at this step. That template is registered to the
anatomical once, and every group inherits the result, so preprocessed data
from different groups (for example different sessions) are directly
comparable. The agreement between each group's reference and the template
is written as a QC table and shown in the report. Which transforms are
written in each case is listed in :ref:`transforms`.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep_docs import T1W
    from qsiprep.workflows.dwi.dwiref import init_dwiref_wf

    wf = init_dwiref_wf(inputs_list=['sub-01_dir-AP', 'sub-01_dir-PA'], t1w_source_file=T1W)


.. _resampling:

**********
Resampling
**********

:func:`qsiprep.workflows.dwi.resampling.init_dwi_trans_wf`

Every transform that applies to a volume is composed into one: head motion
(when carried rather than applied by the backend), gradient nonlinearity,
susceptibility distortion, and coregistration. Each volume is resampled once
with that transform onto the ``ACPC`` grid at ``--output-resolution``, with
Lanczos windowed sinc interpolation :footcite:p:`lanczos`, or linear
interpolation when upsampling by more than 10%. The Jacobian weights
(:ref:`jacobian_methods`) are applied at the same step.

The gradient table is rotated by the linear part of each volume's
transform, and written in FSL (``.bval``/``.bvec``), MRtrix3 (``.b``) and
DSI Studio (``.b_table.txt``) formats. B1 bias field correction runs on the
resampled series (``--dwi-biascorrect``), followed by the quality measures
that fill the ``t1_`` columns of the image QC table.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep_docs import AP
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(source_file=AP, mem_gb=3)


.. _distortion_group_merge:

**********************
Merging corrected data
**********************

:func:`qsiprep.workflows.dwi.distortion_group_merge.init_distortion_group_merge_wf`

When an output's correction units were corrected separately, their
resampled results are combined according to ``--distortion-group-merge``:
concatenated along the fourth dimension (``concat``), averaged over images
that sampled the same q-space coordinate (``average``), or left as separate
outputs (``none``). The merged output gets its own report figures, QC table
and sidecar. The Jacobian and gradient deviation derivatives are not written
for merged outputs.


**********
References
**********

.. footbibliography::
