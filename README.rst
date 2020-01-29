.. include:: links.rst

QSIprep: Preprocessing and analysis of q-space images
=======================================================

.. image:: https://readthedocs.org/projects/qsiprep/badge/?version=latest
  :target: http://qsiprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://circleci.com/gh/PennBBL/qsiprep/tree/master.svg?style=svg
  :target: https://circleci.com/gh/PennBBL/qsiprep/tree/master


Full documentation at https://qsiprep.readthedocs.io

About
-----

``qsiprep`` configures pipelines for processing diffusion-weighted MRI (dMRI) data.
The main features of this software are

  1. A BIDS-app approach to preprocessing nearly all kinds of modern diffusion MRI data.
  2. Automatically generated preprocessing pipelines that correctly group, distortion correct,
     motion correct, denoise, coregister and resample your scans, producing visual reports and
     QC metrics.
  3. A system for running state-of-the-art reconstruction pipelines that include algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.
  4. A novel motion correction algorithm that works on DSI and random q-space sampling schemes

.. image:: https://github.com/PennBBL/qsiprep/raw/master/docs/_static/workflow_full.png


.. _preprocessing_def:

Preprocessing
~~~~~~~~~~~~~~~

The preprocessing pipelines are built based on the available BIDS inputs, ensuring that fieldmaps
are handled correctly. The preprocessing workflow performs head motion correction, susceptibility
distortion correction, MP-PCA denoising, coregistration to T1w images, spatial normalization
using ANTs_ and tissue segmentation.


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~~~

The outputs from the :ref:`preprocessing_def` pipelines can be reconstructed in many other
software packages. We provide a curated set of :ref:`recon_workflows` in ``qsiprep``
that can run ODF/FOD reconstruction, tractography, Fixel estimation and regional
connectivity.



Note
------

The ``qsiprep`` pipeline uses much of the code from ``FMRIPREP``. It is critical
to note that the similarities in the code **do not imply that the authors of
FMRIPREP in any way endorse or support this code or its pipelines**.
