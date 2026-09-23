.. include:: links.rst

#####
Usage
#####

The *QSIPrep* preprocessing workflow takes as principal input the path of
the dataset that is to be processed. The input dataset is required to be in
valid :abbr:`BIDS (Brain Imaging Data Structure)` format at least one
diffusion MRI series. The T1w image and the DWI may be in separate BIDS
<session> folders for a given subject. We highly recommend that you validate
your dataset with the free, online `BIDS Validator
<http://bids-standard.github.io/bids-validator/>`_.

The exact command to run *QSIPrep* depends on the Installation_ method.
The common parts of the command are similar to the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.

Example: ::

    qsiprep data/bids_root/ out/ participant -w work/ --output-resolution 2


**********************
Command-Line Arguments
**********************

.. argparse::
   :ref: qsiprep.cli.parser._build_parser
   :prog: qsiprep


***********
Infant mode
***********

If ``--infant`` is used, the pipeline will select an MNIInfant template with the
appropriate cohort based on the participant's age.

``--infant`` is only compatible with ``--subject-anatomical-reference sessionwise``.

.. note::

    *QSIPrep*'s cohort selection is derived from Nibabies.

Participant Ages
================

*QSIPrep* will attempt to automatically extract participant ages (in months) from the BIDS layout. Specifically, these two files will be checked:

Sessions file: <bids-root>/<subject>/<subject>_sessions.tsv

Participants file: <bids-root>/participants.tsv

Either file should include age (or if you wish to be more explicit: age_months) columns, and it is recommended to have an accompanying JSON file to further describe these fields, and explicitly state the values are in months.


**************************
Preparing data for QSIPrep
**************************

QSIPrep is a BIDS App, meaning that it expects the data to be in BIDS format.
However, QSIPrep does contain some idiosyncrasies that mean that the data may need to be prepared in a specific way.


Siemens Reverse Phase-Encoded "Field Maps"
==========================================

Reverse phase-encoded images are a common acquisition for distortion correction.
For a dMRI scan, this would mean acquiring one or more volumes of b=0 images with the
opposite phase encoding direction of the main dMRI scan.

It can be hard to acquire a scan that only contains b=0 volumes with Siemens scanners,
so researchers often acquire a short dMRI run with a mix of b=0 and b>0 volumes.
QSIPrep expects these short scans to be in the fmap directory, instead of the dwi directory.
If you acquire data like this, you should organize your data as below::

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
                sub-<label>_ses-<label>_dir-PA_epi.json  # Add IntendedFor field here

In this scenario, the short scan is organized as a field map, with the epi suffix.
As of BIDS v1.10.0, EPI field maps can have bval and bvec files, so this organization is completely BIDS-compliant.

If you organize your short scan as a dMRI run, QSIPrep will denoise the short scan and concatenate it with the longer run,
which is not optimal.

Moreover, if you have a short scan with a mix of b=0 and b>0 volumes, and you do not include the bval and bvec files,
QSIPrep will assume that all of the volumes are b=0, which will almost certainly produce suboptimal results.


Complex-Valued Data
===================

If you acquire complex-valued data, split it into BIDS ``part-mag`` (magnitude) and
``part-phase`` (phase) files. QSIPrep pairs each ``part-phase`` image with its magnitude
and combines them for complex denoising (with a ``dwidenoise`` or ``dwidenoise2`` denoising
method, applied before series are concatenated); every downstream step runs on the magnitude.

``part-real``/``part-imag`` (real/imaginary) parts are not supported: QSIPrep ignores them
with a warning rather than erroring, so convert them to magnitude and phase first.


BIDS-URIs
=========

BIDS-URIs are the recommended way to defined certain metadata fields, such as IntendedFor, in BIDS.
However, QSIPrep does not currently support BIDS-URIs for the IntendedFor field.
Therefore, you should use relative paths to the files, which is the older way to do things.


B0FieldIdentifier and B0FieldSource
===================================

``B0FieldIdentifier`` and ``B0FieldSource`` link field maps to the images they
correct, and are the BIDS-recommended alternative to ``IntendedFor``.
Give every file of a field map the same ``B0FieldIdentifier`` (for a reverse
phase-encoded pair of DWI series, that includes both series), and list it in the
``B0FieldSource`` of each DWI series it should correct.
A series with a ``B0FieldSource`` ignores ``IntendedFor`` links to it.
``B0FieldSource`` may list several field maps: QSIPrep applies one of them,
preferring reverse phase encoding, and reports the rest as also eligible.
See :ref:`gre_init_usage` for how a GRE field map among the rest is used.


MultipartID
===========

MultipartID is a metadata field that is used to identify a set of DWIs that should be considered as part of the same acquisition.
If you want to group certain runs of dMRI data together, but not all runs (the default behavior), you should use the MultipartID field.

However, please note that MultipartID may interact in unexpected ways with the IntendedFor field and the QSIPrep parameters that impact grouping (e.g., ``--distortion-group-merge``).
Therefore, we recommend that, if you use MultipartID, you check your outputs to make sure the runs are being grouped in the manner you expect.


.. _gre_init_usage:

***************************************************
Starting registration-based SDC from a GRE fieldmap
***************************************************

TORTOISE's registration-based distortion correction (DRBUDDI and T2Wreg) infers
the susceptibility field by matching images.
Where the field piles signal from several voxels into one, or drops it out,
many deformations match equally well.
A GRE fieldmap measures the field instead.
When a series is corrected by one of these registrations and a GRE fieldmap also
lists it, QSIPrep starts the registration from the GRE-derived warp and holds
that warp fixed through the registration's multi-resolution pyramid, so the
registration refines the GRE estimate rather than replacing it.

There is no option to turn this on. It happens whenever a GRE fieldmap
(``phasediff``, ``phase1``/``phase2`` or ``fieldmap``, with its magnitude
images) lists a series that a different correction wins:

* **reverse phase encoding**, corrected by DRBUDDI: with ``--hmc-method
  tortoise`` or ``shoreline``, or ``--hmc-method eddy --sdc-method drbuddi``.
  Not with ``--sdc-method topup+drbuddi``, where DRBUDDI only refines TOPUP's
  correction.
* **an anatomical reference** forced with ``--force sdc-anat-reference``,
  corrected by T2Wreg (``--hmc-method tortoise``).

``--ignore fieldmaps`` (which skips ``fmap/``) and ``--ignore sdc`` turn it off.
A GRE fieldmap that is itself the applied correction is handed to ``eddy``
under ``--hmc-method eddy`` (see :ref:`fsl_wf`).


DRBUDDI from a GRE fieldmap
===========================

When both a reverse phase-encoded correction and a GRE fieldmap list a series,
QSIPrep corrects the series with DRBUDDI and starts DRBUDDI from the GRE
fieldmap.

Reverse phase-encoded DWI series
--------------------------------

For a pair of DWI series acquired with opposite phase encoding (for example
``dir-AP`` and ``dir-PA``), the GRE fieldmap's ``IntendedFor`` is enough, as
long as nothing in the session carries ``B0FieldIdentifier`` or
``B0FieldSource``: QSIPrep pairs the two series itself and keeps the GRE
fieldmap as a candidate::

    sub-01/
      fmap/
        sub-01_phasediff.json    {"IntendedFor": ["dwi/sub-01_dir-AP_dwi.nii.gz",
                                                  "dwi/sub-01_dir-PA_dwi.nii.gz"], ...}
        sub-01_magnitude1.json
        sub-01_magnitude2.json
      dwi/
        sub-01_dir-AP_dwi.json
        sub-01_dir-PA_dwi.json

If the session uses ``B0FieldIdentifier``/``B0FieldSource``, link the pair
with them too.
Both series are sources of the reverse phase-encoded estimation, and both list
it and the GRE fieldmap::

    sub-01/
      fmap/
        sub-01_magnitude1.json   {"B0FieldIdentifier": "gre", ...}
        sub-01_magnitude2.json   {"B0FieldIdentifier": "gre", ...}
        sub-01_phasediff.json    {"B0FieldIdentifier": "gre", ...}
      dwi/
        sub-01_dir-AP_dwi.json   {"B0FieldIdentifier": "pepolar",
                                  "B0FieldSource": ["pepolar", "gre"], ...}
        sub-01_dir-PA_dwi.json   {"B0FieldIdentifier": "pepolar",
                                  "B0FieldSource": ["pepolar", "gre"], ...}

Any identifiers work, and the order within ``B0FieldSource`` does not matter.

A reverse phase-encoded EPI fieldmap
------------------------------------

With an ``epi`` fieldmap, ``IntendedFor`` is enough: have both the ``epi``
fieldmap and the GRE fieldmap name the DWI series::

    sub-01/
      fmap/
        sub-01_dir-PA_epi.json   {"IntendedFor": ["dwi/sub-01_dir-AP_dwi.nii.gz"], ...}
        sub-01_phasediff.json    {"IntendedFor": ["dwi/sub-01_dir-AP_dwi.nii.gz"], ...}
        sub-01_magnitude1.json
        sub-01_magnitude2.json
      dwi/
        sub-01_dir-AP_dwi.json

The ``B0FieldIdentifier``/``B0FieldSource`` form above works too, with the
``epi`` fieldmap carrying ``"B0FieldIdentifier": "pepolar"``.

Then run with an HMC method that corrects the pair with DRBUDDI, for example
TORTOISE, whose default ``--sdc-method`` is DRBUDDI::

    qsiprep /path/to/bids /path/to/output participant \
        --hmc-method tortoise --output-resolution 1.5

.. warning::
   Two setups look right but leave DRBUDDI skipped or unseeded:

   * **Other fieldmap links in the session.** Once any file in the session
     carries ``B0FieldIdentifier`` or ``B0FieldSource``, or an ``epi``
     fieldmap's ``IntendedFor`` names a series, QSIPrep stops pairing series by
     itself. A GRE fieldmap whose ``IntendedFor`` names an AP/PA pair that is
     not otherwise linked is then applied on its own and DRBUDDI does not run;
     link the pair with ``B0FieldIdentifier``/``B0FieldSource`` as above.
   * **The pair uses ``B0FieldSource``, the GRE fieldmap only ``IntendedFor``.**
     A series with a ``B0FieldSource`` ignores ``IntendedFor`` links to it, so
     the GRE fieldmap is not a candidate and DRBUDDI starts without it.
     Put the GRE fieldmap's ``B0FieldIdentifier`` in the series'
     ``B0FieldSource``.

To correct both series with the GRE fieldmap itself instead, name only the GRE
fieldmap in their ``B0FieldSource`` (or pass ``--ignore pepolar-dwis``).
QSIPrep then processes each phase encoding in its own pipeline, with a GRE warp
built for that encoding, and concatenates the corrected results.


T2Wreg from a GRE fieldmap
==========================

Link the GRE fieldmap to the DWI series as usual (``IntendedFor``, or
``B0FieldIdentifier``/``B0FieldSource``), and force an anatomical reference over
it::

    qsiprep /path/to/bids /path/to/output participant \
        --hmc-method tortoise \
        --sdc-anat-reference t2w --force sdc-anat-reference \
        --output-resolution 1.5

* ``t2w`` registers to the subject's T2w image, which must be in ``anat/``.
  ``synb0`` registers to a distortion-free b=0 image synthesized from the T1w
  instead, and needs ``--anat-modality T1w`` (the default).
* Without ``--force sdc-anat-reference``, the anatomical reference is only a
  fallback for series no fieldmap reaches, so the GRE fieldmap is applied on its
  own.
* ``--force sdc-anat-reference`` applies to every DWI series: series with no GRE
  fieldmap run T2Wreg without an initial warp.
  ``invt1w`` forces SyN, which takes no initial warp.


Checking the setup
==================

Before running, `qsiplan <https://github.com/PennLINC/QSIPlan>`__ (see
:doc:`quickstart`) shows how each series will be corrected.
Pass it the same method flags you will give QSIPrep::

    qsiplan /path/to/bids --hmc-method tortoise

For the reverse phase-encoded example above, each series should be corrected by
the reverse phase-encoded estimation, with the GRE fieldmap listed as also
eligible::

    Distortion group sub-01_dir-AP (PE j-, TRT 0.05s):
      - sub-01_dir-AP_dwi.nii.gz
      corrected by: pepolar [curated]
      (also eligible: gre)

For T2Wreg, add ``--sdc-anat-reference t2w --force sdc-anat-reference``; each
series should then be corrected by ``auto+t2wreg`` with the GRE fieldmap also
eligible.
In both cases the GRE fieldmap is labelled "(initializes DRBUDDI/T2Wreg)", and
its ``estimation-unused`` warning notes that it initializes those
registrations.

During the run, QSIPrep logs ``Initializing DRBUDDI for <output> with GRE
fieldmap <id>`` (or ``T2Wreg``).
In the HTML report, the distortion correction entry ends in
``(GRE-initialized)``, and the methods boilerplate describes the initialization.


********************************
Gradient nonlinearity correction
********************************

Gradient coils deviate from their nominal linear field.
This displaces voxels, increasingly so away from isocentre,
and it means the diffusion gradient actually applied at a voxel is not quite
the one recorded in the bval/bvec table.
Pass a scanner coefficient file with ``--gradient-file`` to correct both: ::

    --gradient-file /path/to/coeff.grad

Accepted formats are ``.grad`` (Siemens), ``.dat`` (GE), ``.gc`` (TORTOISE
binary), and ``.nii``/``.nii.gz`` (an ITK displacement field).
Only one file is accepted, and it applies to every DWI run in the dataset;
process multi-site data one site at a time.

Whether the *spatial* correction is applied to a given run, and how much of
it, is decided from that run's ``ImageType`` field:

===================  =======================================================
``ImageType`` tag    Behavior
===================  =======================================================
(no ``DIS`` tag)     Full 3D gradwarp correction
``DIS2D``            Through-plane correction only; the scanner already
                     corrected in-plane distortion
``DIS3D``            No spatial correction; the scanner already corrected it
===================  =======================================================

Use ``--force gradwarp3D`` to apply the full 3D correction, or
``--force gradwarp1D`` to apply the through-plane component of it only,
regardless of ``ImageType``, for data whose tags are absent or untrustworthy.
The two are mutually exclusive, and either requires ``--gradient-file``.

Use ``--ignore gradwarp`` to disable gradient nonlinearity correction
entirely, including the deviation map described below.

.. warning::
   **GE data: coefficient files are not accepted for spatial correction.**
   When expanding coefficients for a GE scanner, TORTOISE applies a z-origin
   shift to the resulting displacement field. That shift is applied by the
   ``TORTOISEProcess`` driver, not by the standalone
   ``CreateNonlinearityDisplacementMap`` binary QSIPrep calls, so QSIPrep
   cannot reproduce TORTOISE's placement of the field and raises an error
   rather than applying one it cannot place. Two ways forward:

   * pass a ready-made ITK displacement field (``.nii``/``.nii.gz``) to
     ``--gradient-file``. QSIPrep uses it as given and expands nothing, so the
     shift does not arise;
   * pass ``--ignore gradwarp`` to skip gradient correction.

   Runs tagged ``DIS3D`` are unaffected, since no spatial field is built for
   them, and the gradient deviation map below is unaffected on any GE run: it
   is produced by a different TORTOISE tool that handles GE internally.

Diffusion-encoding (gradient deviation) correction
==================================================

Independently of the spatial correction, a voxelwise gradient deviation map
is written as ``*_space-ACPC_graddev.nii.gz`` whenever ``--gradient-file`` is
given and ``--ignore gradwarp`` is absent -- **including for runs tagged**
``DIS3D``. No scanner can correct the diffusion encoding itself: the
bval/bvec table holds a single value per volume and has nowhere to record
information that varies across the image. At each voxel, the gradient
actually applied is ``L @ g``, where ``g`` is the nominal gradient vector and
``L`` is the voxel's local 3x3 gradient nonlinearity matrix. Because ``L``
captures scaling and shear rather than a pure rotation, both the b-vector
*and* the b-value deviate per voxel, not just the direction. The deviation
map holds this 3x3 matrix, in row-major order, as 9 volumes; downstream tools
that consume a gradient deviation file (e.g. DSI Studio) can use it directly.

The map is oriented into the output space by a rigid registration that
TORTOISE's ``CreateGradientNonlinearityBMatrix`` estimates internally between
the raw native b=0 and the final b=0, rather than by the coregistration
transform QSIPrep used on the data itself. The two are close but not
identical; the ``GradientDeviationOrientation`` key in the sidecar records
this.

.. warning::
   The deviation map is **not** written for outputs produced by
   ``--distortion-group-merge``. Those outputs are assembled by a separate
   merge workflow that has no gradient-deviation step, so neither
   ``*_graddev.nii.gz`` nor the ``GradientWarpDimensions`` sidecar key is
   written for them. The spatial gradwarp correction is still applied to the
   data. QSIPrep logs a warning naming each affected output.

For details on where gradient nonlinearity correction sits in the DWI
pipeline, how it differs by head-motion/distortion-correction backend, and
its known limitations, see :ref:`gradwarp`.


******************
Note on using CUDA
******************

The CUDA runtime version 11.1.1 is included in the *QSIPrep* docker image.
The CUDA version of eddy is dramatically faster than the openmp version.
Information on running Docker with CUDA enabled can be found on
`dockerhub <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA>`_. If running with Apptainer,
the call to Apptainer should include ``--nv``. To enable CUDA, see :ref:`configure_eddy`.


*********
Debugging
*********

Logs and crashfiles are outputted into the
``<output dir>/qsiprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`Errors and Crashes <https://miykael.github.io/nipype_tutorial/notebooks/basic_error_and_crashes.html>`_
page of the Nipype Tutorial.
