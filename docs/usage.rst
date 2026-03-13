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

    qsiprep data/bids_root/ out/ participant -w work/


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

Sessions file: <bids-root>/<subject>/subject_sessions.tsv

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

If you acquire complex-valued data, you should (1) split the data into magnitude and phase files (NOT real and imaginary!)
and (2) retain a copy of the bval and bvec files with the part-mag entity included (i.e., do not use the inheritance principle on these files).

QSIPrep is not compatible with real and imaginary data.

Also, QSIPrep does not currently support using the inheritance principle for bval and bvec files.
While this is not normally a problem, since these files should not be inherited,
it would make sense from a BIDS readability perspective to inherit the bvals and bvecs across both the magnitude and phase files.
For example, the following organization would be perfectly BIDS-compliant::

    sub-<label>/
        ses-<label>/
            dwi/
                sub-<label>_ses-<label>_dwi.bval
                sub-<label>_ses-<label>_dwi.bvec
                sub-<label>_ses-<label>_part-mag_dwi.nii.gz
                sub-<label>_ses-<label>_part-mag_dwi.json
                sub-<label>_ses-<label>_part-phase_dwi.nii.gz
                sub-<label>_ses-<label>_part-phase_dwi.json

The inheritance principle ensures that the bvals and bvecs without the ``part`` entity
apply to both the magnitude and phase files.

Unfortunately, if you do this, QSIPrep will not be able to find the bvec and bval files.
Therefore, you should organize your data as below, to make sure QSIPrep will work::

    sub-<label>/
        ses-<label>/
            dwi/
                sub-<label>_ses-<label>_part-mag_dwi.nii.gz
                sub-<label>_ses-<label>_part-mag_dwi.bval
                sub-<label>_ses-<label>_part-mag_dwi.bvec
                sub-<label>_ses-<label>_part-mag_dwi.json
                sub-<label>_ses-<label>_part-phase_dwi.nii.gz
                sub-<label>_ses-<label>_part-phase_dwi.bval
                sub-<label>_ses-<label>_part-phase_dwi.bvec
                sub-<label>_ses-<label>_part-phase_dwi.json


BIDS-URIs
=========

BIDS-URIs are the recommended way to define metadata links (for example, in ``IntendedFor``).
QSIPrep accepts both BIDS-URI style entries (``bids::sub-...``) and relative paths.
Use one style consistently within your dataset.

However, please use B0FieldIdentifier and B0FieldSource instead of IntendedFor for fieldmap metadata.


Curating DWI Grouping and Fieldmaps
===================================

QSIPrep builds four internal groupings per subject/session:

1. Distortion groups (which DWI files are merged before denoising)
2. Fieldmap estimation groups (which files are used to estimate each fieldmap)
3. Fieldmap application groups (which DWI distortion groups each fieldmap is applied to)
4. Concatenation groups (which distortion groups are merged in outputs)

If your curation metadata conflicts with physical acquisition metadata
(``ShimSetting`` and ``TotalReadoutTime``), QSIPrep raises an error instead of silently grouping incompatible scans.


How QSIPrep decides groups
--------------------------

For fieldmap estimation groups (highest priority first):

1. ``B0FieldIdentifier`` (on DWI and/or fmap files)
2. ``IntendedFor`` (on fmap files)
3. Automatic heuristic based on phase-encoding directions

For fieldmap application groups:

1. ``B0FieldSource`` (on DWI files)
2. Otherwise derived from the estimation groups

For concatenation groups:

1. If not separating all DWIs: ``MultipartID`` if present
2. Otherwise automatic per-session grouping
3. If separating all DWIs: each distortion group remains separate

Files are never grouped across sessions.


Which metadata to use
---------------------

Use ``B0FieldIdentifier`` + ``B0FieldSource`` when you want explicit, reproducible control
over both fieldmap estimation and application.

Use ``MultipartID`` when you want only some runs concatenated together in final outputs.

Use no grouping metadata only when you are happy with automatic behavior.


Which parameters to use
-----------------------

The most important controls are:

- ``--separate-all-dwis``: keep each DWI separate in distortion and concatenation groups.
- ``--ignore fieldmaps``: ignore files in ``fmap/`` for fieldmap construction. Field maps using files in ``dwi/`` will still be used.
- ``--pepolar-method``: If set to "DRBUDDI" or "TOPUP+DRBUDDI", require reverse-PE pairing per axis (for example AP/PA and LR/RL handled separately).


Practical curation patterns
---------------------------

If you want one fieldmap from all PE directions:

- Provide one shared ``B0FieldIdentifier`` across those DWIs.
- Provide matching ``B0FieldSource`` on target DWIs.
- Use ``--pepolar-method TOPUP``.

If you want separate fieldmaps per run (for example run-1 vs run-2):

- Use run-specific ``B0FieldIdentifier`` values.
- Use matching run-specific ``B0FieldSource`` values.
- Optionally use ``MultipartID`` to keep output concatenations run-specific.

If you want automatic grouping:

- Omit ``B0FieldIdentifier``, ``B0FieldSource``, and ``IntendedFor``.
- Ensure ``PhaseEncodingDirection`` is present and correct.
- Provide consistent ``ShimSetting`` and ``TotalReadoutTime`` for files that should be grouped.

If you use ``--pepolar-method DRBUDDI`` or ``--pepolar-method TOPUP+DRBUDDI``:

- Ensure each estimation group has reverse directions on the same axis (for example ``j`` and ``j-``).
- A metadata-defined group spanning multiple axes with ``--pepolar-method DRBUDDI`` or ``--pepolar-method TOPUP+DRBUDDI`` will raise an error.


MultipartID
===========

MultipartID is a metadata field that is used to identify a set of DWIs that should be considered as part of the same acquisition.
If you want to group certain runs of dMRI data together, but not all runs (the default behavior), you should use the MultipartID field.

Important: ``MultipartID`` constraints must be compatible with fieldmap group definitions.
If one fieldmap application group would cross multiple ``MultipartID`` concatenation groups,
QSIPrep raises an error and asks you to resolve the inconsistency.


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
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.
