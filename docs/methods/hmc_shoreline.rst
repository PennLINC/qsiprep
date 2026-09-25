.. include:: ../links.rst

.. _hmc_shoreline:

#########
SHORELine
#########

:func:`qsiprep.workflows.dwi.hmc_sdc.init_qsiprep_hmcsdc_wf`

.. warning::
   SHORELine is scheduled for removal in a future major release. Use
   ``--hmc-method tortoise`` for non-shelled data; it corrects eddy currents
   as well as head motion, which SHORELine does not.

``--hmc-method shoreline`` runs *QSIPrep*'s own model-based head motion
correction :footcite:p:`cieslak2021qsiprep`, written for DSI and other
schemes that ``eddy`` cannot handle. It corrects head motion only.

What runs
=========

1. All b=0 images are aligned to a mid-point b=0, and each b>0 image gets
   the transform of its nearest b=0.
2. For each b>0 image, a 3dSHORE :footcite:p:`merlet3dshore` (or tensor)
   model is fit to all the other images, and used to predict the signal at
   the left-out image's q-space coordinate. The image is registered to that
   prediction and its gradient vector rotated accordingly.
3. A new model is fit to the updated images and vectors, and the
   leave-one-out registration is repeated (``"iters"`` times, default 2).
4. Susceptibility distortion correction runs on the result: DRBUDDI for
   reverse phase-encoded data, or a GRE fieldmap or SyN. ``synb0`` and
   ``t2w`` anatomical references are not supported on this backend.

The motion transforms are carried, not applied, so the data are interpolated
once, at the final resampling. Six (or twelve, with an affine transform)
parameters per volume go to the confounds file, and the per-slice model fit
is shown in the carpet plot.

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep_docs import example_unit, configure, AP, ANATOMICAL_TEMPLATE
    from qsiprep.workflows.dwi.hmc_sdc import init_qsiprep_hmcsdc_wf

    configure(workflow__hmc_method='shoreline', workflow__sdc_method='drbuddi')
    wf = init_qsiprep_hmcsdc_wf(
        example_unit('pepolar'),
        source_file=AP,
        t2w_sdc=False,
        anatomical_template=ANATOMICAL_TEMPLATE,
    )

.. _configure_shoreline:

Configuring SHORELine
=====================

``--shoreline-config`` is a JSON file; the default is `shoreline_params.json
<https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/shoreline_params.json>`__.
Every key is optional; unknown keys are an error.

.. list-table::
   :header-rows: 1
   :widths: 20 30 15 35

   * - Key
     - Allowed values
     - Default
     - Meaning
   * - ``model``
     - ``"3dshore"``, ``"tensor"``, ``"none"``
     - ``"3dshore"``
     - Signal model used to predict each left-out image. ``"none"`` skips
       the model and gives each b>0 image the transform of its nearest b=0
       image, which is not recommended.
   * - ``iters``
     - An integer of at least 1
     - ``2``
     - Number of SHORELine iterations.
   * - ``transform``
     - ``"Affine"``, ``"Rigid"``
     - ``"Affine"``
     - Transformation optimized during head motion correction.


**********
References
**********

.. footbibliography::
