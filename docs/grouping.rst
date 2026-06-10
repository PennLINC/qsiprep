.. include:: links.rst

.. _grouping:

##################################
Scan grouping and fieldmap mapping
##################################

Before *QSIPrep* preprocesses any diffusion data, it must answer four questions
for each participant (and, when present, each session):

#. Which DWI files share the same susceptibility distortions and should be
   processed together before head-motion correction?
#. Which files should be used to estimate each fieldmap?
#. Which DWI series should each fieldmap be *applied* to?
#. Which preprocessed series should be concatenated (or averaged) in the
   outputs?

*QSIPrep* answers these questions automatically, but you can take explicit
control with BIDS metadata (``B0FieldIdentifier``, ``B0FieldSource``,
``IntendedFor``, ``MultipartID``) and a handful of command-line parameters.
This page documents exactly how those metadata fields and parameters interact.

.. contents:: Contents
   :local:
   :depth: 2

.. note::

   Files are **never** grouped across subjects or sessions.
   Every rule below applies *within* a single subject/session.


******************
The four groupings
******************

Internally, *QSIPrep* produces four dictionaries per subject/session. Each one
controls a different part of the pipeline.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Grouping
     - What it controls
     - Primary inputs
   * - **Distortion groups**
     - DWI files that share distortions and are processed as one pre-HMC
       series. Denoising happens before or after this within-group merge
       depending on ``--denoise-after-combining``.
     - ``PhaseEncodingDirection``, ``ShimSetting``, ``TotalReadoutTime``,
       ``B0FieldIdentifier``
   * - **Fieldmap estimation groups**
     - Which files are combined to *estimate* each fieldmap.
     - ``B0FieldIdentifier``, ``IntendedFor``, phase-encoding heuristic
   * - **Fieldmap application groups**
     - Which distortion groups each fieldmap is *applied* to for SDC.
     - ``B0FieldSource``, ``IntendedFor``, estimation groups
   * - **Concatenation groups**
     - Which preprocessed distortion groups are concatenated/averaged in the
       output derivatives.
     - ``MultipartID``, session

The estimation and application groups are computed *after* the distortion
groups, so their members are referred to by distortion-group ID rather than by
raw filename. When fieldmap usage is disabled (``--ignore fieldmaps``), neither
fieldmap grouping is built.


***************
Metadata fields
***************

The following metadata fields (read from the JSON sidecars) drive grouping.
Each affects a specific subset of the four groupings.

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Field
     - Affects
     - Meaning in *QSIPrep*
   * - ``PhaseEncodingDirection``
     - Distortion groups; automatic fieldmap heuristic
     - Files in one distortion group must share a phase-encoding direction.
       Opposite directions on the same axis (e.g. ``j``/``j-``) are what the
       automatic heuristic pairs for PEPOLAR fieldmaps.
   * - ``ShimSetting``
     - Distortion groups; curated fieldmap groups
     - Files in one distortion group must share shim settings. Curated
       fieldmap groups that combine distortion groups with different shim
       settings raise an error (see :ref:`grouping_errors`).
   * - ``TotalReadoutTime``
     - Distortion groups; curated fieldmap groups
     - Same as ``ShimSetting``: files in one distortion group must agree, and
       curated fieldmap groups with conflicts raise an error.
   * - ``B0FieldIdentifier``
     - Distortion groups; fieldmap **estimation** groups
     - A label shared by all files that should be combined to estimate one
       fieldmap. Also participates in distortion grouping.
   * - ``B0FieldSource``
     - Fieldmap **application** groups
     - On a DWI file, names the ``B0FieldIdentifier`` of the fieldmap that
       should be applied to it.
   * - ``IntendedFor``
     - Fieldmap estimation groups; application fallback
     - The older mechanism. On an ``fmap/`` file, lists the DWI files the
       fieldmap should estimate from and be applied to. Used for estimation only
       when ``B0FieldIdentifier`` is absent; application is then derived from
       that estimation group unless ``B0FieldSource`` is present on DWIs.
   * - ``MultipartID``
     - Concatenation groups
     - A label shared by DWI runs that should be concatenated together in the
       final outputs (but kept separate from runs with a different label).

.. tip::

   ``B0FieldIdentifier`` and ``B0FieldSource`` are the preferred, most explicit
   and reproducible way to define fieldmaps. Prefer them over ``IntendedFor``.
   See the `BIDS specification on fieldmaps
   <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#expressing-intent-with-b0-field-mapping>`_.


**************************
How each grouping is built
**************************

.. _grouping_distortion:

Distortion groups
=================

A distortion group is a set of DWI files that share the same distortions and
are sent through one pre-HMC workflow. The files in a distortion group are
eventually concatenated before head-motion correction, but by default each
input series is denoised before that concatenation. With
``--denoise-after-combining``, denoising is instead run on the concatenated
series. Two files land in the same distortion group only when **all** of the
following match:

* session
* ``PhaseEncodingDirection``
* ``ShimSetting``
* ``TotalReadoutTime``
* ``B0FieldIdentifier``

Including ``B0FieldIdentifier`` in this key means that if a curator splits runs
into different fieldmap estimation groups, the distortion groups are split to
match (so a distortion group never spans two fieldmaps).

If ``--separate-all-dwis`` is set, this step is skipped and **every DWI file
becomes its own distortion group**.

The key of each distortion group is a generalized BIDS name built from the
entities the grouped files share (for example ``sub-01_dir-AP``). When two
groups would collapse to the same name, an indexed ``acq-`` entity is added to
keep them unique.


.. _grouping_estimation:

Fieldmap estimation groups
==========================

These determine which files are combined to *estimate* each fieldmap.
*QSIPrep* uses the first of the following strategies that applies (highest
priority first):

#. **B0FieldIdentifier** — if any DWI or ``fmap/`` file carries a
   ``B0FieldIdentifier``, files are grouped by that label. Group keys are the
   ``B0FieldIdentifier`` values themselves.
#. **IntendedFor** — if there is no ``B0FieldIdentifier`` but ``fmap/`` files
   carry ``IntendedFor``, fieldmaps are grouped by which DWIs they target.
   Group keys are auto-generated (``auto_00000``, ``auto_00001``, ...).
#. **Automatic heuristic** — with no curation metadata, *QSIPrep* pairs DWI
   distortion groups by phase-encoding direction (see below). Group keys are
   auto-generated.

.. admonition:: The automatic heuristic only uses ``dwi/`` files

   When falling back to the heuristic, *QSIPrep* pairs reverse phase-encoded
   **DWI** series with each other. It will *not* automatically pull an EPI
   fieldmap from ``fmap/`` into an estimation group — linking a ``fmap/`` file
   to a DWI series requires ``B0FieldIdentifier`` or ``IntendedFor``. A DWI
   series with no reverse-PE partner and no curation metadata will have no
   fieldmap (SyN-SDC may still be applied, depending on your parameters).

The heuristic's pairing behavior depends on ``--pepolar-method``:

* **Default (TOPUP)** — within a session (and within a ``MultipartID``
  partition), all distortion groups that have a ``PhaseEncodingDirection`` are
  combined into one estimation group as long as at least two distinct
  directions are present. TOPUP can use any mixture of compatible directions
  (e.g. AP, PA, and LR together).
* **DRBUDDI / TOPUP+DRBUDDI** (``estimate_per_axis=True`` internally) — only
  *reverse* directions on the **same axis** are paired (AP with PA, LR with RL),
  producing one estimation group per axis. DRBUDDI requires exactly opposing
  directions, so cross-axis mixtures are not grouped.


.. _grouping_application:

Fieldmap application groups
===========================

These map each fieldmap to the distortion groups it will correct.
*QSIPrep* uses the first strategy that applies:

#. **B0FieldSource** — if DWI files carry ``B0FieldSource``, each fieldmap is
   applied to exactly the distortion groups that name it. The presence of any
   ``B0FieldSource`` switches application into this explicit mode; DWI groups
   without ``B0FieldSource`` are not assigned fieldmaps by the estimation-group
   fallback.
#. **Derived from estimation groups** — otherwise, every DWI distortion group
   that contributed to a fieldmap's estimation is also corrected by that
   fieldmap. (Pure ``fmap/`` files are sources, not targets.)


.. _grouping_concatenation:

Concatenation groups
====================

These determine which preprocessed distortion groups are concatenated or
averaged in the output derivatives. *QSIPrep* chooses as follows:

#. If ``--separate-all-dwis`` is set, each distortion group is its own
   concatenation group (a warning is emitted if ``MultipartID`` is also
   present, since it is being overridden).
#. Otherwise, if ``MultipartID`` is present, distortion groups are concatenated
   by shared ``MultipartID`` (within a session). Distortion groups without a
   ``MultipartID`` remain separate from the labeled groups.
#. Otherwise, all distortion groups within a session are concatenated together.

How the corrected groups are actually merged is then controlled by
``--distortion-group-merge`` (see below).


*******************************
Parameters that affect grouping
*******************************

``--separate-all-dwis``
=======================

Disables concatenation entirely: every DWI file becomes its own distortion
group **and** its own concatenation group. Reverse-PE combinations are still
used to build fieldmaps. If ``MultipartID`` is present, ``--separate-all-dwis``
takes precedence and a warning is raised.

``--ignore fieldmaps``
======================

Disables fieldmap grouping and application. No fieldmaps from ``fmap/`` are
built, and the automatic reverse-PE DWI heuristic is also disabled. This
overrides ``IntendedFor``/``B0FieldIdentifier`` on ``fmap/`` files and
``B0FieldIdentifier``/``B0FieldSource`` on DWI files.

``--pepolar-method``
====================

Selects the SDC method used for PEPOLAR (phase-encoding-based) fieldmaps:
``TOPUP`` (default), ``DRBUDDI``, or ``TOPUP+DRBUDDI``. Beyond choosing the
algorithm, this changes how estimation groups are formed:

* ``TOPUP`` allows arbitrary combinations of phase-encoding directions in one
  fieldmap.
* ``DRBUDDI`` / ``TOPUP+DRBUDDI`` require reverse-PE pairs **per axis**. A
  curator-defined ``B0FieldIdentifier`` that spans multiple axes will raise an
  error in this mode (see :ref:`grouping_errors`).

``--distortion-group-merge``
============================

Controls how corrected images from *different* distortion groups in the same
concatenation group are combined:

* ``concat`` — append the images along the 4th dimension.
* ``average`` — if a whole sequence was duplicated in opposing PE directions,
  average the corrected images of the same q-space coordinate.
* ``none`` (default) — keep distortion groups separate in the outputs.

``--use-syn-sdc`` and ``--force-syn``
=====================================

These enable fieldmap-less (SyN-based) SDC. ``--force-syn`` applies SyN-SDC even
when a fieldmap is available; it sets every group's fieldmap type to ``syn``.
``--use-syn-sdc`` enables SyN as a fallback when no other fieldmap is found.


************************************
From groups to distortion correction
************************************

After grouping, each distortion group is assigned a fieldmap "type" that
determines which SDC workflow runs. The type is derived from the files in the
group's fieldmap estimation group:

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Fieldmap type
     - Source files
     - SDC method
   * - ``rpe_series``
     - A reverse phase-encoded DWI series (optionally plus an EPI fieldmap)
     - PEPOLAR (TOPUP or DRBUDDI)
   * - ``epi``
     - One or more EPI fieldmaps from ``fmap/``
     - PEPOLAR (TOPUP or DRBUDDI)
   * - ``fieldmap``
     - A direct fieldmap (``fieldmap`` + ``magnitude``)
     - Fieldmap-based (FMB)
   * - ``phasediff``
     - ``phasediff`` + magnitude image(s)
     - Fieldmap-based (FMB)
   * - ``phase1``/``phase2``
     - Two phase images + magnitude image(s)
     - Fieldmap-based (FMB)
   * - ``syn``
     - None (anatomical image only)
     - SyN-SDC (fieldmap-less)
   * - ``None``
     - No fieldmap
     - No SDC (unless SyN is requested)

.. note::

   PEPOLAR methods (``epi``, ``rpe_series``) are **not** compatible with the
   3dSHORE/SHORELine head-motion model when TOPUP is requested. GRE fieldmaps
   (``fieldmap``, ``phasediff``, ``phase1``/``phase2``) are routed to the
   fieldmap-based path regardless of ``--pepolar-method``.


.. _grouping_errors:

****************************
Consistency rules and errors
****************************

*QSIPrep* validates the groupings and raises an error (rather than silently
producing a questionable result) in these cases:

* **Incompatible acquisition metadata.** If fieldmap curation metadata
  (``B0FieldIdentifier``, ``B0FieldSource``, or ``IntendedFor``) groups
  distortion groups together, but those groups disagree on ``ShimSetting`` or
  ``TotalReadoutTime``, they cannot be combined for fieldmap processing --
  *QSIPrep* raises an error.
* **A fieldmap split across concatenation groups.** A fieldmap estimation
  group must be a subset of exactly one concatenation group. If ``MultipartID``
  would split a fieldmap's targets into different concatenation groups,
  *QSIPrep* raises an error. This check is skipped when
  ``--separate-all-dwis`` intentionally makes every DWI file its own
  concatenation group.
* **A cross-axis fieldmap under DRBUDDI.** With ``--pepolar-method DRBUDDI`` or
  ``TOPUP+DRBUDDI``, a single ``B0FieldIdentifier`` that spans more than one
  phase-encoding axis raises an error, because DRBUDDI estimates per axis.

When a distortion group spans multiple fieldmap estimation groups (for example,
a curator assigns run-1 and run-2 of the same direction to different
fieldmaps), *QSIPrep* automatically **splits** the distortion group so each
piece falls within a single estimation group. This is not an error.


********************
Choosing what to use
********************

* **You want explicit, reproducible control of fieldmaps** → use
  ``B0FieldIdentifier`` (on the files that estimate a fieldmap) together with
  ``B0FieldSource`` (on the DWIs that should be corrected by it).
* **You want only some runs concatenated in the outputs** → use
  ``MultipartID`` to label the runs that belong together.
* **You are happy with automatic behavior** → omit all grouping metadata, but
  make sure ``PhaseEncodingDirection`` is present and that files you expect to
  be grouped share ``ShimSetting`` and ``TotalReadoutTime``.

.. seealso::

   :ref:`merging` in the preprocessing guide for a higher-level overview of how
   merging affects the pipeline, and the :doc:`usage` page for the related
   command-line options.


***************
Worked examples
***************

These examples assume a single subject with multi-run, multi-PED DWI data
(directions AP, PA, LR, RL, each with run-1 and run-2) and no fieldmaps in
``fmap/``.

Automatic grouping (no curation metadata)
=========================================

With ``--pepolar-method TOPUP`` (default) and no grouping metadata, *QSIPrep*
forms one distortion group per direction, a single TOPUP estimation group from
all of them, and a single concatenation group:

* **Distortion groups:** ``sub-01_dir-AP``, ``sub-01_dir-PA``,
  ``sub-01_dir-LR``, ``sub-01_dir-RL`` (each containing two runs; by default,
  each run is denoised before the within-group concatenation)
* **Estimation group:** one group containing all four distortion groups
* **Concatenation group:** ``sub-01`` containing all four distortion groups

Under ``--pepolar-method DRBUDDI``, the single estimation group is instead
split into two per-axis groups: ``{AP, PA}`` and ``{LR, RL}``.

Per-run fieldmaps with ``B0FieldIdentifier``
============================================

If the curator labels the run-1 files with ``B0FieldIdentifier: topuprun01``
and run-2 files with ``B0FieldIdentifier: topuprun02`` (and matching
``B0FieldSource`` on the DWIs), *QSIPrep* splits the distortion groups by run to
honor the fieldmaps:

* **Distortion groups:** eight groups (``sub-01_dir-AP_run-1``,
  ``sub-01_dir-AP_run-2``, ...)
* **Estimation groups:** ``topuprun01`` (the four run-1 groups) and
  ``topuprun02`` (the four run-2 groups)
* **Concatenation group:** still a single ``sub-01`` (all eight groups), unless
  you also use ``MultipartID`` to keep runs separate

Conflicting ``MultipartID`` (raises an error)
=============================================

If the curator uses ``B0FieldIdentifier`` to say *all eight runs* estimate one
fieldmap, but ``MultipartID`` says run-1 and run-2 are separate concatenation
groups, the single fieldmap's targets would be split across two concatenation
groups. *QSIPrep* raises an error and asks you to resolve the inconsistency.
