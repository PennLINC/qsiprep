.. include:: links.rst

.. _upgrading:

###################
Upgrading from 26.0
###################

26.1 reorganized the command line. Options are now grouped by decision in
``--help``, several were renamed, and options whose settings moved into a
configuration file were removed. Run ``qsiprep --help`` or see :doc:`cli`
for the current list.

**************************
Removed or renamed options
**************************

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - In 26.0
     - In 26.1
     - PR
   * - ``--session-id``
     - ``--session-label``
     - #1136
   * - ``--dwi-only``
     - ``--anat-modality none``
     - #1145
   * - ``--dwi-no-biascorr``, ``--b1-biascorrect-stage``
     - ``--dwi-biascorrect none`` (or ``n4``, ``auto``)
     - #1145, #1144
   * - ``--fs-license-file``
     - Removed. No FreeSurfer license is needed.
     - #1145
   * - ``--dwi-denoise-window``
     - ``--dwidenoise-window``
     - #1149
   * - ``--denoise-after-combining``
     - Removed. Series are always denoised before concatenation.
     - #1146
   * - ``--anat-only``
     - Removed.
     - #1151
   * - ``--hmc-model``, ``--hmc-transform``, ``--shoreline-iters``,
       ``--shoreline-model``
     - ``--shoreline-config`` (a JSON file; see :ref:`configure_shoreline`)
     - #1140
   * - ``--pepolar-method``
     - ``--sdc-method``
     - #1140
   * - ``--hmc-method diffprep``
     - ``--hmc-method tortoise``
     - #1085
   * - ``--force-syn``, ``--use-syn-sdc``
     - ``--sdc-anat-reference invt1w``, with ``--force sdc-anat-reference``
       to override fieldmaps
     - #1111
   * - ``--b0-to-t1w-transform``, ``--b0-to-anat-transform``
     - ``--dwi2anat-dof 6`` (rigid) or ``12`` (affine)
     - #1144
   * - ``--intramodal-template-iters``, ``--intramodal-template-transform``
     - ``--dwiref-definition subject`` with
       ``--dwiref-construction-iters`` and
       ``--dwiref-construction-transform``
     - #1144
   * - ``--longitudinal``
     - ``--subject-anatomical-reference unbiased``
     - #1126
   * - ``--subject-anatomical-reference first-alphabetically``
     - ``--subject-anatomical-reference first-lex``
     - #1157
   * - ``--b0-motion-corr-to``, ``--prefer-dedicated-fmaps``,
       ``--fmap-no-demean``, ``--fmap-bspline``
     - Removed.
     - #1124, #1125, #1130, #1131

***********
New options
***********

* ``--hmc-method tortoise`` and ``--diffprep-config``: TORTOISE DIFFPREP
  head motion and eddy-current correction (:doc:`methods/hmc_tortoise`).
* ``--sdc-method``: ``topup``, ``drbuddi`` or ``topup+drbuddi`` for reverse
  phase-encoded data (:doc:`methods/sdc`).
* ``--sdc-anat-reference``: ``synb0``, ``t2w``, ``invt1w`` or ``auto``
  fieldmap-less correction as a fallback, and ``--force sdc-anat-reference``.
* ``--gradient-file``, ``--force gradwarp1D``/``gradwarp3D``, ``--ignore
  gradwarp``: gradient nonlinearity correction (:ref:`gradwarp_flags`).
* ``--ignore jacobian``, ``--force jacobian``: intensity modulation after
  distortion correction (:ref:`jacobian_flags`).
* ``--gpu`` and ``--tortoise-gpu-cpu-ratio``: per-task GPU selection,
  replacing ``use_cuda`` in the configuration files.
* ``--denoise-method dwidenoise2`` and ``--dwidenoise2-config``.
* ``--mrtrix-version``: use the MRtrix3 development branch.
* ``--anat-biascorrect``, ``--report-output-level``, ``--ignore t2w``,
  ``--ignore phase``, ``--ignore shims``, ``--ignore fov``.

****************
Behavior changes
****************

* **Grouping is done by qsiplan.** The rules for which scans estimate a
  fieldmap, which fieldmap corrects which scan and which scans are
  concatenated are now shared with the standalone `qsiplan`_ tool, and every
  decision is reported with its provenance. ``B0FieldIdentifier`` and
  ``B0FieldSource`` are supported, and BIDS URIs work in ``IntendedFor``.
  Inconsistent metadata that used to be silently worked around now stops
  the run with a named error (:ref:`grouping_errors`).
* **GRE fieldmaps go into eddy.** With ``--hmc-method eddy``, a GRE fieldmap
  is passed to ``eddy`` and applied inside its model rather than to its
  outputs. ``--force gre-sdc-after-eddy`` restores the old behavior for
  comparison and is deprecated.
* **A GRE fieldmap initializes DRBUDDI and T2Wreg** when a series is also
  corrected by reverse phase encoding (:ref:`sdc_gre_init`).
* **Intensity modulation** after distortion correction is now applied by
  *QSIPrep* for the corrections it resamples itself, and written as a
  derivative.
* **Distortion correction displacement maps** are written for inspection.
* **GRE fieldmaps no longer use FSL.** Phase unwrapping and fieldmap
  application use niimath's ROMEO implementation instead of PRELUDE and
  FUGUE.
* **SHORELine is deprecated** and prints a removal notice.
* **Output naming** follows fMRIPrep 26.0 for the coregistration reference:
  ``space-distortiongroup`` or ``space-subject`` on the reference image and
  ``desc-coreg`` on the transforms (:ref:`transforms`).
