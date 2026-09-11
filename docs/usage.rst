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

    qsiprep data/bids_root/ out/ participant -w work/ --output-spaces acpc:res-2mm MNI152NLin2009cAsym


**********************
Command-Line Arguments
**********************

.. argparse::
   :ref: qsiprep.cli.parser._build_parser
   :prog: qsiprep


.. _output_spaces_ref:

*************
Output Spaces
*************

``--output-spaces`` tells *QSIPrep* where to write its outputs, and replaces the older
``--output-resolution``, ``--anatomical-template`` and ``--skip-anat-based-spatial-normalization``
flags (see :ref:`migrating_output_spaces` below). It takes one or more space-delimited
tokens, each naming a space and, optionally, a resolution or cohort:

.. code-block:: text

    --output-spaces acpc:res-2mm MNI152NLin2009cAsym

``acpc`` is required
====================

At least one ``acpc`` entry is required, because *QSIPrep* only ever writes preprocessed
DWI in ACPC (subject-native) space -- there is no way to ask for DWI resampled straight
into a standard space. ``acpc`` always needs an explicit resolution, for example
``acpc:res-2mm``.

The ``res-`` families
=====================

A ``res-`` suffix sets the resolution for a space. Which forms are accepted depends
on whether the space is ``acpc`` or a standard space:

On ``acpc``:

- **Isotropic physical size**, given in millimeters, e.g. ``res-2mm`` or the decimal
  form ``res-1p5mm`` (1.5 mm). ``acpc`` accepts only the isotropic form; an
  anisotropic size such as ``res-6x6x3mm`` is rejected, because reconstruction
  requires isotropic DWI.
- **Native-resolution strategies**, ``res-nativemin`` and ``res-nativemax``. These take
  the smallest or largest voxel dimension across the input DWI runs and use it
  isotropically -- a 3x4x5 mm input yields 3x3x3 mm for ``nativemin`` and 5x5x5 mm for
  ``nativemax``. Because the value is only known once the DWI headers are read, the
  resolved voxel size is recorded in the ``Resolution`` key of each output's JSON
  sidecar.

On standard spaces:

- **TemplateFlow resolution labels only**, e.g. ``res-1`` or ``res-2``. The label
  selects which TemplateFlow grid the template is fetched on, and appears verbatim as
  the ``res-`` entity in the output filename. Which labels exist varies by template;
  an unavailable label is rejected with the list of valid ones.

.. warning::

   Physical (``mm``) sizes are **not** implemented for standard spaces, and are
   rejected. QSIPrep does not resample standard-space output to an arbitrary voxel
   size, so a token such as ``MNI152NLin2009cAsym:res-1p5mm`` or
   ``MNI152NLin2009cAsym:res-6x6x3mm`` would write no ``res-`` entity and land on
   exactly the filenames the bare template writes, silently overwriting them. Use a
   TemplateFlow label instead. ``mm`` and ``native*`` sizes work only on ``acpc``.

Multiple ``acpc`` entries
=========================

.. note::

   Multiple ``acpc`` resolutions cannot be combined with ``--distortion-group-merge``
   (``concat`` or ``average``). The merge workflow writes one set of derivatives with
   no ``res-`` entity, so the extra resolutions would be resampled and then silently
   dropped; the combination is rejected instead. A *single* resolution merges
   normally, including ``res-nativemin``/``res-nativemax``, whose resolved voxel size
   is written to the merged output's JSON sidecar.

Listing ``acpc`` more than once (e.g. ``acpc:res-2mm acpc:res-1p5mm``) resamples and
writes the preprocessed DWI once per requested resolution. Each additional ``acpc``
entry costs roughly as much as another full resampling pass over the DWI data, so
requesting *N* ``acpc`` resolutions costs roughly *N* times the resampling.

Standard spaces
================

Any other space name (e.g. ``MNI152NLin2009cAsym``, ``MNIInfant``) is a standard,
TemplateFlow-hosted space. Standard spaces produce the anatomical-to-template
transforms and resampled anatomical derivatives (T1w/T2w, masks, segmentations), but
**DWI is never resampled into a standard space** -- only ``acpc`` DWI is written.
Each standard space requires its own nonlinear registration of the anatomical
reference to that template, so requesting *N* standard spaces costs *N* nonlinear
registrations. Requesting **no** standard space at all (e.g.
``--output-spaces acpc:res-2mm``) skips the anatomical nonlinear registration
entirely, unless fieldmap-less SyN distortion correction is in use -- that needs the
same transform.

A single template may be listed at more than one TemplateFlow resolution, either as
repeated tokens or with repeated ``res-`` keys:

.. code-block:: text

    --output-spaces acpc:res-2mm MNI152NLin2009cAsym:res-1:res-2

Each resolution gets its own resampled anatomical derivatives, tagged with its own
``res-`` entity. The ACPC-to-template transform does not depend on output resolution,
so it is written once.

``cohort-auto``
===============

Templates with cohorts (e.g. ``MNIInfant``) require a ``cohort-`` key. Passing
``cohort-auto`` (e.g. ``MNIInfant:cohort-auto``) defers cohort selection until
run time, when *QSIPrep* picks the appropriate cohort from each participant's age
(see `Infant mode`_ below). An explicit cohort, e.g. ``MNIInfant:cohort-3``, skips
that age-based lookup.

.. _migrating_output_spaces:

Migrating from ``--output-resolution``
========================================

``--output-resolution``, ``--anatomical-template`` and
``--skip-anat-based-spatial-normalization`` are deprecated in favor of
``--output-spaces`` and will be removed in 27.0.0. The table below shows the
equivalent ``--output-spaces`` invocation for each old flag combination:

.. list-table::
   :header-rows: 1

   * - Old
     - New
   * - ``--output-resolution 2``
     - ``--output-spaces acpc:res-2mm MNI152NLin2009cAsym``
   * - ``--output-resolution 1.5``
     - ``--output-spaces acpc:res-1p5mm MNI152NLin2009cAsym``
   * - ``--output-resolution 2 --infant``
     - ``--infant --output-spaces acpc:res-2mm MNIInfant:cohort-auto``
   * - ``--output-resolution 2 --skip-anat-based-spatial-normalization``
     - ``--output-spaces acpc:res-2mm``

.. important::

   ``--infant`` is **not** deprecated and must be kept in the new invocation. Beyond
   appending ``MNIInfant:cohort-auto`` to ``--output-spaces``, it narrows the autobox
   padding (4 mm instead of 8 mm), forces a T2w anatomical reference, and requires
   ``--subject-anatomical-reference sessionwise``. Dropping it in favor of the
   ``MNIInfant:cohort-auto`` token alone changes results.

.. note::

   ``--infant --skip-anat-based-spatial-normalization`` has no exact
   ``--output-spaces`` equivalent yet. ``--infant`` always appends
   ``MNIInfant:cohort-auto``, so the only way to keep the infant AC-PC anchor while
   writing no standard space is to keep using the deprecated flag, which continues to
   work (and keeps the infant anchor) until 27.0.0.


***********
Infant mode
***********

If ``--infant`` is used, ``MNIInfant:cohort-auto`` is appended to ``--output-spaces``
unless an infant template is already requested -- ``MNIInfant`` or ``UNCInfant``,
either of which anchors AC-PC. The pipeline selects the appropriate cohort based on
the participant's age.

On the deprecated path, ``--infant`` replaces the anatomical template rather than
adding to it, so ``--output-resolution 2 --infant --anatomical-template
MNI152NLin2009cAsym`` requests the infant template alone, as it did before
``--output-spaces`` existed.

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

If you acquire complex-valued data, you should split the data into magnitude and phase files (NOT real and imaginary!).

QSIPrep is not compatible with real and imaginary data.


BIDS-URIs
=========

BIDS-URIs are the recommended way to defined certain metadata fields, such as IntendedFor, in BIDS.
However, QSIPrep does not currently support BIDS-URIs for the IntendedFor field.
Therefore, you should use relative paths to the files, which is the older way to do things.


B0FieldIdentifier and B0FieldSource
===================================

B0FieldIdentifier and B0FieldSource are two metadata fields that are used to related images to field maps for distortion correction.
They are the preferred alternative to the IntendedFor field in BIDS, but QSIPrep does not currently support them.
Therefore, you should use the IntendedFor field with relative paths to the files, which is the older way to do things.


MultipartID
===========

MultipartID is a metadata field that is used to identify a set of DWIs that should be considered as part of the same acquisition.
If you want to group certain runs of dMRI data together, but not all runs (the default behavior), you should use the MultipartID field.

However, please note that MultipartID may interact in unexpected ways with the IntendedFor field and the QSIPrep parameters that impact grouping (e.g., ``--distortion-group-merge``).
Therefore, we recommend that, if you use MultipartID, you check your outputs to make sure the runs are being grouped in the manner you expect.


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

Use ``--force gradients`` to apply the full 3D correction regardless of
``ImageType``, for data whose tags are absent or untrustworthy.
``--force gradients`` requires ``--gradient-file``.

Use ``--ignore gradients`` to disable gradient nonlinearity correction
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
   * pass ``--ignore gradients`` to skip gradient correction.

   Runs tagged ``DIS3D`` are unaffected, since no spatial field is built for
   them, and the gradient deviation map below is unaffected on any GE run: it
   is produced by a different TORTOISE tool that handles GE internally.

Diffusion-encoding (gradient deviation) correction
==================================================

Independently of the spatial correction, a voxelwise gradient deviation map
is written as ``*_space-ACPC_graddev.nii.gz`` whenever ``--gradient-file`` is
given and ``--ignore gradients`` is absent -- **including for runs tagged**
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
