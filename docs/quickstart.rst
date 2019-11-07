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
    --combine-all-dwis \
    --denoise-before-combining \
    --dwi-denoise-window 7 \
    --output-resolution 1.2 \
    --output-space T1w \
    --force-spatial-normalization \
    --hmc-model eddy \
    --fs-license-file /path/to/license.txt


Grouping scans
======================

.. note::
   This section explains ``--combine-all-dwis``, ``--denoise-before-combining`` and
   ``--dwi-denoise-window``

Assuming that ``sub-1/fmap/sub-1_dir-PA_epi.nii.gz`` has a JSON sidecar that contains::

  "IntendedFor": [
    "dwi/sub-1_acq-multishell_run-01_dwi.nii.gz",
    "dwi/sub-1_acq-multishell_run-02_dwi.nii.gz",
    "dwi/sub-1_acq-multishell_run-03_dwi.nii.gz"
  ]

``qsiprep`` will infer that the dwi scans are in the same **warped space** - that their
susceptibility distortions are shared and they can be combined before head motion correction. Since
we specified ``--combine-all-dwis`` the separate scans will be merged together before head motion
correction and the fully preprocessed outputs will be written to
``derivitaves/qsiprep/sub-1/dwi/sub-1_acq-multishell_desc-preproc_dwi.nii.gz``.

It is beneficial to have as much data as possible available for head motion correction. However,
the denoising preprocessing step has important caveats that should be considered. It is up to the
user whether ``dwidenoise`` is run on ``run-01``, ``run-02``, ``run-03`` individually before they
are combined, or whether to combine these scans and run ``dwidenoise`` on the concatenated dwi
series. This is an unexplored trade-off space. The more volumes available, the more data MP-PCA has
to work with and a larger window size (specified with ``--dwi-denoise-window``) can be safely used.
However, if there if the head is in a vastly different location in different scans, performance may
be impacted.

By not specifying ``--combine-all-dwis`` there will be one output in derivatives for each input
image in the bids input directory.

Specifying outputs
=====================

.. note::
   This section covers ``--output-resolution 1.2``, ``--output-space T1w``, and
   ``--force-spatial-normalization``.

Just like ``FMRIPREP``, the ``--output-space`` argument can include ``T1w`` and the name of a
template (eg ``MNI152NLin2009cAsym``). Since the spatial normalization provided by ``qsiprep``
only uses the T1w image, we recommend writing outputs to ``T1w`` space only. If only ``T1w`` is
specified in ``--output-space``, then spatial normalization won't be run. It can still be useful
to have a transform to a template space to warp atlases to T1w space, or to warp model results
to a template space. The combination of ``--output-space T1w --force-spatial-normalization`` will
only write preprocessed dwi data to T1w space, but will also include the template-to-T1w transform
in the derivatives for later use.

The ``--output-resolution`` argument determines the spatial resolution of the preprocessed dwi
series. You can specify the resolution of the original data or choose to upsample the dwi to a
higher spatial resolution. Some post-processing pipelines such as fixel-based analysis recommend
resampling your output to at least 1.3mm resolution. By choosing this resolution here, it means
your data will only be interpolated once: head motion correction, susceptibility distortion
correction, coregistration and upsampling will be done in a single step. If your are upsampling
significantly be sure to check the outputs for ringing artifacts.


Head motion correction model
===============================

Although FSL's ``eddy`` is technically model-free, it is an option for ``--hmc-model`` along with
``3dSHORE`` and ``none``. Choosing ``eddy`` runs FSL's ``eddy`` for head motion correction and
eddy current correction. This will work for DTI (single-shell) and HARDI (multi-shell) sampling
schemes. The ``3dSHORE`` (aka "SHORELine") option works for multi-shell HARDI, Cartesian grid
sampling (DSI) and random q-space sampling (CS-DSI).

The option ``none`` will register all the b=0 images to one another and the b>0 images will
have the transform from the nearest b=0 image applied. This is not recommended. Between ``eddy``
and ``3dSHORE``, all sampling schemes can be motion corrected.
