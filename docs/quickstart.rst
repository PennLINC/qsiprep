.. include:: links.rst

Quick Start
------------------------

There are many options for running ``qsiprep`` but most have sensible defaults and
don't need to be changed. This page describes the options most most likely to be
needed to be adjusted for your specific data. Suppose the following data is available
in the BIDS input::

  sub-1/dwi/sub-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/dwi/sub-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/dwi/sub-1_acq-multishell_run-03_dwi.nii.gz
  sub-1/fmap/sub-1_dir-PA_epi.nii.gz

One way to process these data would be to call ``qsiprep`` like this::

  qsiprep \
    /path/to/inputs /path/to/outputs participant \
    --dwi-denoise-window 7 \
    --output-resolution 1.2 \
    --fs-license-file /path/to/license.txt


Grouping scans
======================

.. note::
   This section explains ``--separate-all-dwis``, ``--denoise-after-combining`` and
   ``--dwi-denoise-window``

Assuming that ``sub-1/fmap/sub-1_dir-PA_epi.nii.gz`` has a JSON sidecar that contains::

  "IntendedFor": [
    "dwi/sub-1_acq-multishell_run-01_dwi.nii.gz",
    "dwi/sub-1_acq-multishell_run-02_dwi.nii.gz",
    "dwi/sub-1_acq-multishell_run-03_dwi.nii.gz"
  ]

``qsiprep`` will infer that the dwi scans are in the same **warped space** - that their
susceptibility distortions are shared and they can be combined before head motion correction. Since
we didn't specify ``--separate-all-dwis`` the separate scans will be merged together before head motion
correction and the fully preprocessed outputs will be written to
``derivitaves/qsiprep/sub-1/dwi/sub-1_acq-multishell_desc-preproc_dwi.nii.gz``. otherwise
there will be one output in the derivatives directory for each input image in the bids input
directory.

It is beneficial to have as much data as possible available for head motion correction. However,
the denoising preprocessing step has important caveats that should be considered. For a
discussion see :ref:`merge_denoise`.


Specifying outputs
=====================

.. note::
   This section covers ``--output-resolution 1.2``, and
   ``--skip-t1-based-spatial-normalization``.

Unlike with fMRI, which can be coregistered to a T1w image and warped to a template using the
T1w image's spatial normalization, the T1w images do not contain enough contrast to accurately
align white matter structures to a template. For this reason, spatial normalization is typically
done *after* models are fit. Therefore we omit the ``--output-spaces`` argument from preprocessing.
All outputs will be registered to the T1w image but will have an isotropic voxel size.

Cortex can be accurately spatially-normalized using the T1w image, so the T1w image is still
spatially normalized by default during preprocessing. The transform from the T1w image to the
``MNI152NLin2009cAsym`` template is included in the derivatives. This can be used during
reconstruction to map cortical parcellations from the template into the DWI in order to estimate
brain graphs. If you want to save ~20 minutes of computation time, this normalization can be
disabled with the ``--skip-tq-based-spatial-normalization`` option.

The ``--output-resolution`` argument determines the spatial resolution of the preprocessed dwi
series. You can specify the resolution of the original data or choose to upsample the dwi to a
higher spatial resolution. Some post-processing pipelines such as fixel-based analysis recommend
resampling your output to at least 1.3mm resolution. By choosing this resolution here, it means
your data will only be interpolated once: head motion correction, susceptibility distortion
correction, coregistration and upsampling will be done in a single step. If your are upsampling
your data by more than 10%, QSIPrep will use BSpline interpolation instead of Lanczos windowed
sinc interpolation.


Head motion correction model
===============================

Although FSL's ``eddy`` is technically model-free, it is an option for ``--hmc-model`` along with
``3dSHORE`` and ``none``. Choosing ``eddy`` (the default) runs FSL's ``eddy`` for head motion
correction and eddy current correction. This will work for single-shell and multi-shell sampling
schemes. The ``3dSHORE`` (aka "SHORELine") option works for multi-shell, Cartesian grid sampling
(DSI) and random q-space sampling (CS-DSI).

The option ``none`` will register all the b=0 images to one another and the b>0 images will
have the transform from the nearest b=0 image applied. This is not recommended. Between ``eddy``
and ``3dSHORE``, all sampling schemes can be motion corrected.
