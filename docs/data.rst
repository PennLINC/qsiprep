.. include:: links.rst

.. _data:

###################
Preparing your data
###################

*QSIPrep* is a BIDS App: it reads a `BIDS`_ dataset and decides what to do from
the files and sidecar metadata it finds there. Most of the decisions that
matter, which scans are corrected by which fieldmap and which scans end up in
the same output file, are made from metadata. This page describes what
*QSIPrep* reads and how to encode your intent by curation.


*****************
BIDS requirements
*****************

The input dataset must be valid BIDS and contain at least one diffusion MRI
series. The T1w image and the DWI may be in different ``ses-<label>``
directories of the same subject. We recommend checking the dataset with the
`BIDS Validator <https://bids-standard.github.io/bids-validator/>`_ before
running.

*QSIPrep* reads:

``dwi/``
  ``*_dwi.nii.gz`` with its ``.bval``, ``.bvec`` and ``.json`` sidecar. The
  sidecar needs ``PhaseEncodingDirection`` and ``TotalReadoutTime`` for any
  kind of susceptibility distortion correction.
``fmap/``
  Reverse phase-encoded ``*_epi`` images, and GRE fieldmaps
  (``*_phasediff`` with one or two ``*_magnitude`` images, ``*_phase1`` and
  ``*_phase2`` with magnitudes, or a Hz ``*_fieldmap`` with a magnitude).
``anat/``
  ``*_T1w`` and ``*_T2w`` images. One contrast is the *anatomical reference*
  (``--anat-modality``); a T2w can also be a target for fieldmap-less
  distortion correction. A lesion mask, ``*_label-lesion_roi.nii.gz`` in the
  same space as the anatomical reference, is used to mask the template normalization.
``participants.tsv`` / ``*_sessions.tsv``
  An ``age`` (or ``age_months``) column, read only with ``--infant`` to pick
  the MNIInfant cohort. Add a JSON sidecar that states the unit is months.


*********
Fieldmaps
*********

Reverse phase-encoded b=0 images
================================

Reverse phase-encoded images are the most common acquisition for distortion
correction. For a dMRI scan, this means one or more b=0 volumes acquired with
the opposite phase encoding direction from the main scan.

It can be hard to acquire a scan that contains only b=0 volumes on Siemens
scanners, so a short dMRI run with a mix of b=0 and b>0 volumes is often
collected instead. *QSIPrep* expects these short scans in the ``fmap``
directory, not the ``dwi`` directory::

    sub-<label>/
        ses-<label>/
            dwi/
                sub-<label>_ses-<label>_dir-AP_dwi.nii.gz
                sub-<label>_ses-<label>_dir-AP_dwi.bval
                sub-<label>_ses-<label>_dir-AP_dwi.bvec
                sub-<label>_ses-<label>_dir-AP_dwi.json
            fmap/
                sub-<label>_ses-<label>_dir-PA_epi.nii.gz
                sub-<label>_ses-<label>_dir-PA_epi.bval
                sub-<label>_ses-<label>_dir-PA_epi.bvec
                sub-<label>_ses-<label>_dir-PA_epi.json  # Add IntendedFor here

As of BIDS v1.10.0, ``epi`` fieldmaps may have ``bval`` and ``bvec`` files, so
this layout is BIDS-compliant. If the short scan is organized as a ``dwi``
run instead, *QSIPrep* denoises it and concatenates it with the longer run,
which is likely not what you want. If it is a mix of b=0 and b>0 volumes and has no
``bval``/``bvec`` files, every volume is treated as b=0, which is worse.

A full reverse phase-encoded DWI series (for example ``dir-AP`` and ``dir-PA``
runs of the whole scheme) belongs in ``dwi/``. *QSIPrep* pairs the two and
uses both for distortion correction; see :ref:`grouping` below.

GRE fieldmaps
=============

Phase-difference, two-phase and Hz fieldmaps are read from ``fmap/`` with
their magnitude images. With ``--hmc-method eddy`` the field is handed to
``eddy`` and applied inside its model. With the other methods it is applied
to the motion-corrected series. When a series is corrected by reverse phase
encoding *and* a GRE fieldmap lists it, the reverse phase-encoded correction
wins and the GRE fieldmap is used to initialize it. See :ref:`sdc_gre` for
details.

.. _linking_fieldmaps:

Linking fieldmaps to DWI series
===============================

A fieldmap is only used for the series it is linked to. BIDS provides two ways
to express the link.

``B0FieldIdentifier`` and ``B0FieldSource``
  The BIDS-recommended method. Give every file of a fieldmap the same
  ``B0FieldIdentifier`` (for a reverse phase-encoded pair of DWI series, that
  includes both series), and list that identifier in the ``B0FieldSource`` of
  each DWI series it should correct. A series with a ``B0FieldSource`` ignores
  ``IntendedFor`` links to it. ``B0FieldSource`` may list several fieldmaps:
  *QSIPrep* applies one of them, preferring reverse phase encoding, and
  reports the rest as also eligible.

``IntendedFor``
  A list in the fieldmap's sidecar of the files it corrects, as paths relative
  to the subject directory (``ses-1/dwi/sub-1_ses-1_dwi.nii.gz``) or as BIDS
  URIs (``bids::sub-1/ses-1/dwi/sub-1_ses-1_dwi.nii.gz``). Absolute paths are
  accepted with a warning.

Without either, *QSIPrep* infers reverse phase-encoded pairs among the DWI
series of a session. Every inferred decision is reported as such in the log
and in the visual report, so you can see what was guessed.


.. _grouping:

************************
How scans become outputs
************************

Before any processing, *QSIPrep* sorts a subject's DWI scans into four kinds
of group: which series end up in one output file, which series share a
distortion, which files a fieldmap is estimated from, and, combining the
three, which series one head motion run works on. The report lists them in
that order, outputs first. The names below appear in
the log, the visual report and the `qsiplan`_ output.

Output
  The series that are concatenated into one preprocessed file, named by
  ``MultipartID``. By default all the DWI series of a session form one
  output. Series are never combined across sessions.

Distortion group
  DWI series that share the same susceptibility distortion: the same phase
  encoding direction, total readout time, shim setting and field of view.
  One distortion group needs one correction. Two ``dir-AP`` runs are one
  distortion group; a ``dir-AP`` and a ``dir-PA`` run are two.

Fieldmap estimation
  The set of files a fieldmap is estimated from, named by its
  ``B0FieldIdentifier``. For a reverse phase-encoded pair, that is the b=0
  images of both distortion groups. For a GRE fieldmap, it is the phase and
  magnitude images.

Correction unit
  The distortion groups that are corrected by one estimation and motion
  corrected together. A reverse phase-encoded pair is one unit. A curated
  fieldmap boundary or a re-shim is always a unit boundary.

Set ``B0FieldIdentifier``/``B0FieldSource`` and ``MultipartID`` in the
sidecars to control these groups; otherwise *QSIPrep* infers them and tags
each decision with its provenance. The walkthrough
:external+qsiplan:doc:`Understanding scan grouping <tutorials/grouping>`
shows how each field changes the result.

MultipartID
===========

``MultipartID`` is a per-subject string that marks a set of DWI series as parts
of one acquisition. Series that share a ``MultipartID`` are concatenated into
one output, including across phase encoding directions. Use it to combine
some runs but not others. If some series in a session carry a ``MultipartID``
and others do not, the others are still grouped by the default rule, so check
the grouping report before a large run.

ShimSetting and field of view
=============================

Series whose ``ShimSetting`` differs are not pooled into one reverse
phase-encoded estimation, because the field changed between them. Pass
``--ignore shims`` to treat all shim settings as compatible. Series whose
field of view is shifted are concatenated with a warning. Series whose field
of view is rotated are an error, unless ``--ignore fov`` is given, in which
case they are concatenated and the distortion correction is applied to data
it was not estimated on. Series on a different voxel grid are always an
error.


*******************
Complex-valued data
*******************

If you acquire complex-valued data, split it into BIDS ``part-mag`` and
``part-phase`` files. *QSIPrep* pairs each ``part-phase`` image with its
magnitude and denoises them as complex data (with ``--denoise-method
dwidenoise`` or ``dwidenoise2``), before series are concatenated. Every
later step runs on the magnitude. ``part-real`` and ``part-imag`` files are
ignored with a warning, so convert them to magnitude and phase first.


.. _gradient_files:

**************************
Gradient coefficient files
**************************

Gradient coils deviate from their nominal linear field. This displaces voxels,
increasingly so away from isocenter, and it means the diffusion gradient
applied at a voxel is not quite the one recorded in the ``bvec`` table.
Pass the scanner's coefficient file with ``--gradient-file`` to correct
both::

    --gradient-file /path/to/coeff.grad

Accepted formats are ``.grad`` (Siemens), ``.dat`` (GE), ``.gc`` (TORTOISE
binary) and ``.nii``/``.nii.gz`` (a ready-made ITK displacement field). One
file applies to every DWI run in the dataset, so process multi-site data one
site at a time. How much of the spatial correction is applied to each run is
read from its ``ImageType`` field; see :ref:`gradwarp_flags`.

.. warning::
   GE coefficient files are not accepted for the spatial correction. TORTOISE
   applies a z-origin shift to GE displacement fields that the standalone tool
   *QSIPrep* calls does not, so *QSIPrep* raises an error rather than place the
   field wrongly. Pass an ITK displacement field instead, or ``--ignore
   gradwarp``. The gradient deviation map is unaffected.


.. _preview_grouping:

********************************
Previewing the plan with qsiplan
********************************

Before running anything, you can ask how *QSIPrep* will interpret a dataset:
which scans estimate each fieldmap, which fieldmap corrects which scan, and
which scans are combined in each output. The `qsiplan`_ package does this
from metadata alone, without a *QSIPrep* installation
(``pipx install qsiplan``, no need for Docker)::

    qsiplan /path/to/bids --html grouping.html

The text report lists the groups above for each subject, tags each decision
with its provenance (curated, translated from ``IntendedFor``, set by a flag,
or inferred), and previews what each head motion and distortion correction
method would do with the data. Pass the same method flags you will give
*QSIPrep* (``--hmc-method``, ``--sdc-method``, ``--sdc-anat-reference``,
``--separate-all-dwis`` and the others listed in
:external+qsiplan:doc:`its documentation <cli>`) to preview that exact run.
The one flag that differs is the SHORELine model: ``qsiplan`` takes it as
``--shoreline-model``, where *QSIPrep* reads it from the ``"model"`` key of
``--shoreline-config``.
The ``--html`` page is a self-contained, interactive version of the same
report. Nothing is processed and nothing is written to your dataset.

If a decision surprises you, the report names the sidecar field to set. If
the report ends with an ``ERROR`` line, *QSIPrep* will stop at the same
point; see :ref:`grouping_errors`.
