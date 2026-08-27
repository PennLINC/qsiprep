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
