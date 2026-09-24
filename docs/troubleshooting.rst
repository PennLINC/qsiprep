.. include:: links.rst

.. _troubleshooting:

###############
Troubleshooting
###############

************
Getting help
************

If you have a question about how to use *QSIPrep* or about its behavior,
post it on `NeuroStars <https://neurostars.org/tag/qsiprep>`_ with the
``qsiprep`` tag in the "Software Support" category. Earlier questions are at
https://neurostars.org/tag/qsiprep/.

To report a bug or request a feature, open an issue at
https://github.com/pennlinc/qsiprep/issues. Include the command line, the
version, and the relevant part of the log.


********************
Logs and crash files
********************

Logs and crash files are written to
``<output_dir>/sub-<label>/log/<run uuid>/``. A crash file is written for
each node that failed and contains the command that was run, its output and
the traceback. The Nipype tutorial's `Errors and Crashes
<https://miykael.github.io/nipype_tutorial/notebooks/basic_error_and_crashes.html>`_
page explains how to read them. Rerunning with the same ``--work-dir`` picks
up after the last successful node.


.. _grouping_errors:

****************************
Grouping and metadata errors
****************************

Before building the workflow, *QSIPrep* groups the subject's scans and checks
the metadata (see :ref:`grouping`). Warnings are printed and shown in the
report; errors stop the run before any processing, with the message::

    GroupingError: ... [<code>]

Run ``qsiplan`` on the dataset to see the same warnings and errors without
starting *QSIPrep* (:ref:`preview_grouping`). Every code is listed with its
meaning in the :external+qsiplan:doc:`qsiplan issue-code reference
<issue_codes>`. The ones seen most often:

``missing-pedir``
  A DWI series has no ``PhaseEncodingDirection``, so no susceptibility
  distortion correction can be estimated for it. Add the field to the
  sidecar.
``eddy-requires-shelled``
  The data are not shelled (Cartesian or random q-space sampling) and
  ``--hmc-method eddy`` cannot run. Use ``--hmc-method tortoise``.
``unlinked-fmap``
  A fieldmap in ``fmap/`` lists no DWI series in its ``IntendedFor`` or
  ``B0FieldIdentifier``, so it is not used. See :ref:`linking_fieldmaps`.
``fov-grid-mismatch`` / ``fov-oblique``
  Series that would be concatenated were acquired on different grids or with
  differently rotated fields of view. Use ``--separate-all-dwis``, or
  ``MultipartID`` to keep them apart; ``--ignore fov`` forces concatenation
  of rotated fields of view.
``no-sdc``
  No fieldmap reaches a series and no fallback is configured. The series is
  processed without distortion correction; set ``--sdc-anat-reference`` for a
  fieldmap-less correction.


***
FAQ
***

**Why did several runs become one output file?**
  By default all the DWI series of a session are concatenated so that head
  motion correction has as much data as possible. Use ``--separate-all-dwis``
  or ``MultipartID`` to change that (:ref:`grouping_flags`).

**Why is there no preprocessed DWI in MNI space?**
  The T1w cannot align white matter to a template accurately enough. Outputs
  are in the subject's ``ACPC`` space, and the ``ACPC`` to template transform
  is written so normalization can be done after model fitting, in
  `QSIRecon`_.

**My GRE fieldmap shows up as "also eligible" and is not applied.**
  The series is also linked to a reverse phase-encoded correction, which
  *QSIPrep* prefers. The GRE fieldmap is not wasted: it initializes DRBUDDI
  or T2Wreg (:ref:`sdc_gre_init`). To apply the GRE fieldmap itself, list
  only it in the series' ``B0FieldSource`` or pass ``--ignore pepolar-dwis``.

**SHORELine printed a removal notice.**
  ``--hmc-method shoreline`` is scheduled for removal. ``--hmc-method
  tortoise`` handles the same non-shelled data and also corrects eddy
  currents.

**How do I use the GPU with Apptainer?**
  Add ``--nv`` to the ``apptainer run`` call and pass ``--gpu`` with the tasks
  to accelerate (:ref:`hmc_flags`).

**QSIPrep refuses my GE gradient coefficient file.**
  Convert it to an ITK displacement field and pass that, or pass ``--ignore
  gradwarp`` (:ref:`gradient_files`).

**The run was killed with no error.**
  The process ran out of memory. On Docker Desktop, raise the memory limit to
  at least 6 GB. Otherwise lower ``--nprocs``, set ``--mem``, or add
  ``--low-mem``.

**How do I run only some sessions or files?**
  ``--session-label`` selects sessions, and ``--bids-filter-file`` any
  combination of BIDS entities (:ref:`running`).
