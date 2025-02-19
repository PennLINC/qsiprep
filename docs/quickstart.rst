.. include:: links.rst

###########
Quick Start
###########

There are many options for running *QSIPrep* but most have sensible defaults and
don't need to be changed. This page describes the options most likely to be
needed to be adjusted for your specific data. Suppose the following data is available
in the BIDS input::

  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-03_dwi.nii.gz
  sub-1/ses-1/fmap/sub-1_ses-1_dir-PA_epi.nii.gz

One way to process these data would be to call *QSIPrep* like this::

  qsiprep \
    /path/to/inputs /path/to/outputs participant \
    --output-resolution X \
    --fs-license-file /path/to/license.txt

.. warning::
   The above example sets the ``--output-resolution`` to ``X``, where in
   real usage this number should be set to a value in millimeters.
   See :ref:`output_resolution` for specifics.


**************
Grouping scans
**************

.. note::
   This section explains ``--separate-all-dwis``, ``--denoise-after-combining`` and
   ``--dwi-denoise-window``

Assuming that ``sub-1/ses-1/fmap/sub-1_dir-PA_epi.nii.gz`` has a JSON sidecar containing the ``IntendedFor`` field for fieldmap correction
(`see here <https://bids-specification.readthedocs.io/en/v1.10.0/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#expressing-the-mr-protocol-intent-for-fieldmaps>`_)::

  "IntendedFor": [
    "ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz",
    "ses-1/dwi/sub-1_ses-1_acq-multishell_run-02_dwi.nii.gz",
    "ses-1/dwi/sub-1_ses-1_acq-multishell_run-03_dwi.nii.gz"
  ]

*QSIPrep* will infer that the dwi scans are in the same **warped space** - that their
susceptibility distortions are shared and they can be combined before head motion correction.
Since we didn't specify ``--separate-all-dwis``,
the separate scans will be merged together before head motion
correction and the fully preprocessed outputs will be written to
``derivitaves/qsiprep/sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_desc-preproc_dwi.nii.gz``.
Otherwise, there will be one output in the derivatives directory for each input image in the bids input directory.

It is beneficial to have as much data as possible available for head motion correction.
However, the denoising preprocessing step has important caveats that should be considered.
For a discussion see :ref:`merge_denoise`.


******************
Specifying outputs
******************

.. note::
   This section covers ``--output-resolution X``, and
   ``--skip-t1-based-spatial-normalization``.

Unlike with fMRI, which can be coregistered to a T1w image and warped to a
template using the T1w image's spatial normalization, the T1w images do not
contain enough contrast to accurately align white matter structures to a
template. For this reason, spatial normalization is typically done *after*
models are fit.

All outputs will be registered to the T1w image (or the
AC-PC aligned b=0 template if ``--dwi-only`` was specified) but will have
an isotropic voxel size. Furthermore, all outputs are aligned according to the
AC-PC convention: the coordinates are changed from the native scanner
coordinates to a new system where $0, 0, 0$ is where the midline intersects
the anterior commissure (AC).

Cortex can be accurately spatially-normalized using the T1w image, so the T1w
image is still spatially normalized by default during preprocessing. The
transform from the T1w image to the ``MNI152NLin2009cAsym`` template is
included in the derivatives. This can be used during reconstruction to map
cortical parcellations from the template into the DWI in order to estimate
brain graphs. If you want to save ~20 minutes of computation time, this
normalization can be disabled with the
``--skip-t1-based-spatial-normalization`` option.

.. _output_resolution:
Output Resolution and Resampling
================================

The ``--output-resolution`` argument determines the spatial resolution of the
preprocessed dwi series. You can specify the resolution of the original data
or choose to upsample the dwi to a higher spatial resolution. Some
post-processing pipelines such as fixel-based analysis recommend resampling
your output to at least 1.3mm resolution. By choosing this resolution here,
it means your data will only be interpolated once: head motion correction,
susceptibility distortion correction, coregistration and upsampling will be
done in a single step. If you are upsampling your data by more than 10%,
*QSIPrep* will use Linear interpolation instead of Lanczos windowed Sinc
interpolation.


****************************
Head motion correction model
****************************

Although FSL's ``eddy`` is technically model-free, it is an option for
``--hmc-model`` along with ``3dSHORE`` and ``none``. Choosing ``eddy`` (the
default) runs FSL's ``eddy`` for head motion correction and eddy current
correction. This will work for single-shell and multi-shell sampling schemes.
The ``3dSHORE`` (aka "SHORELine") option works for multi-shell, Cartesian
grid sampling (DSI) and random q-space sampling (CS-DSI).

The option ``none`` will register all the b=0 images to one another and the
b>0 images will have the transform from the nearest b=0 image applied. This
is not recommended. Between ``eddy`` and ``3dSHORE``, all sampling schemes
can be motion corrected.


******************************************
Enabling and disabling preprocessing steps
******************************************

The image processing operations performed by *QSIPrep* are configured by default
to apply to most generic sequences. Depending on your sequence
and sampling scheme, you can elect to enable, disable or alter the behavior
of these steps to better match your data.

+-----------------+-----------------------------+---------------------------+------------------------------+
|                 |         Denoising           |      Gibbs Unringing      |    B1 Bias Field Correction  |
+=================+=============================+===========================+==============================+
| Description     |   Reduce random noise       |   Remove spatial ringing  |    Remove spatial non-       |
|                 |   in images.                |   artifact from images.   |    uniformity of images.     |
+-----------------+-----------------------------+---------------------------+------------------------------+
| Algorithms      |  ``dwidenoise`` (MRtrix3)   | ``mrdegibbs`` (MRtrix3)   | ``dwibiascorrect``           |
|                 |  patch2self (DIPY)          |                           | (ANTs/MRtrix3)               |
+-----------------+-----------------------------+---------------------------+------------------------------+
| Default         |  ``dwidenoise`` (MRtrix3)   | None applied              | ``dwibiascorrect``           |
|                 |                             |                           | (ANTs/MRtrix3)               |
+-----------------+-----------------------------+---------------------------+------------------------------+
| Disable with    |  ``--denoise-method none``  | Disabled by default       | ``--dwi-no-biascorr``        |
+-----------------+-----------------------------+---------------------------+------------------------------+
| Change behavior |  ``--dwi-denoise-window N`` | ``--unringing-method``    | No parameters                |
| with            |  changes denoising window   | enables Gibbs unringing   |                              |
|                 |  to N voxels                |                           |                              |
+-----------------+-----------------------------+---------------------------+------------------------------+
| Notes           |  Set the window to ``auto`` | Technically only supposed | Uses                         |
|                 |  or a specific voxel number | to be run on full Fourier | N4BiasFieldCorrection on     |
|                 |                             | acquisitions.             | b=0 images, applies          |
|                 |                             |                           | correction to the whole      |
|                 |                             |                           | series                       |
+-----------------+-----------------------------+---------------------------+------------------------------+

Not included in this table is the b=0 intensity harmonization step, which
applies simple scaling if there is more than one NIfTI file being processed.
It can be disabled with ``--no-b0-harmonization``.

Each of these steps can be applied at the same time, which by default is
before any images are concatenated. The user can instead run these steps
together *after* images are concatenated by specifying
``--denoise-after-combining``. See :ref:`merge_denoise` for more info.

*******************
What is happening??
*******************

While *QSIPrep* runs with `-v -v`, you will see lots of unintuitive output
in the terminal like::

  [Node] Setting-up "qsiprep_wf.single_subject_PNC_wf.dwi_finalize_acq_realistic_wf.transform_dwis_t1.final_b0_ref.b0ref_reportlet" in "/scratch/qsiprep_wf/single_subject_PNC_wf/dwi_finalize_acq_realistic_wf/transform_dwis_t1/final_b0_ref/b0ref_reportlet".
    201229-21:33:46,213 nipype.workflow INFO:
      [Node] Running "b0ref_reportlet" ("niworkflows.interfaces.registration.SimpleBeforeAfterRPT")
    201229-21:33:48,51 nipype.workflow INFO:
      [MultiProc] Running 2 tasks, and 3 jobs ready. Free memory (GB): 3.70/4.00, Free processors: 0/2.
                        Currently running:
                          * qsiprep_wf.single_subject_PNC_wf.dwi_finalize_acq_realistic_wf.transform_dwis_t1.final_b0_ref.b0ref_reportlet
                          * qsiprep_wf.single_subject_PNC_wf.anat_preproc_wf.mni_mask

These print-outs describe what is currently running. In this case, we can see that
``b0ref_reportlet`` and ``mni_mask`` are being run simultaneously. What exactly
are these steps and how do they fit into the overall workflow?

We can find the name of the node (aka "job") being run in the quotation marks.
This task can be found in the workflow diagrams in :ref:`workflow_details`.
In the case of ``mni_mask`` it is part of :ref:`t1preproc_steps`, while
``b0ref_reportlet`` is part of :ref:`dwi_ref`. The relative place of these
jobs' parent workflows in the overall workflow can be seen in the graph of
:ref:`dwi_overview`.

Also in this example you can see that *QSIPrep* was run with ``--nthreads 2``
(``Free processors: 0/2``) and that both open slots are running a job.
