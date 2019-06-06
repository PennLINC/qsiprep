.. include:: links.rst

QSIprep: Preprocessing and analysis of q-space images
=======================================================

.. image:: https://readthedocs.org/projects/qsiprep/badge/?version=latest
  :target: http://qsiprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

About
-----

``qsiprep`` configures pipelines for processing diffusion-weighted MRI (dMRI) data.
The two main features of this software are

  1. A novel :ref:`preprocessing_def` pipeline for non-DTI q-space imaging sequences
  2. A system for building a :ref:`reconstruction_def` pipeline that includes algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.

.. _preprocessing_def:

Preprocessing
~~~~~~~~~~~~~~~

The preprocessing pipelines are designed to perform preprocessing and reconstruction of
q-space images. As of version 0.4 single shell sequences can be processed by setting the

either:

 - A multi-shell HARDI sampling scheme
 - A Cartesian grid (aka DSI) sampling scheme
 - A random q-space sampling scheme (eg for compressed sensing)

QSIprep's head motion correction algorithm relies on voxelwise estimates of
ensemble average propagators (EAPs), which require angular and radial
variability in the sampling scheme. The preprocessing workflow performs
head motion correction, susceptibility distortion correction, MP-PCA denoising,
coregistration to T1w images, spatial normalization using ANTs_ and tissue segmentation.


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~~~

For DTI and multi-shell data, the preprocessing tools available in FSL, MRTrix_
and others are very good. The outputs from these preprocessing pipelines can
be named to mimic the :ref:`outputs` from the ``qsiprep`` preprocessing pipeline
and then reconstructed using workflows from our curated set of :ref:`recon_workflows`.



Note
------

The ``qsiprep`` pipeline uses much of the code from ``FMRIPREP``. It is critical
to note that the similarities in the code **do not imply that the authors of
FMRIPREP in any way endorse or support this code or its pipelines**.  There are
also some noteworthy differences between ``FRMIPREP`` and ``qsiprep``.

  1. All images are conformed to LPS+ instead of RAS+. This greatly simplified
     spatial operations on vectors / vector images within ANTs and working with
     DSI Studio's internal data model.
  2. Co-registration is performed with ANTs instead of FSL or FreeSurfer. We have
     removed FSL and FreeSurfer as much as we could from the pipeline in an effort
     to keep the licensing of bundled software as permissive as possible. Our testing
     also showed that ANTs was more robust for co-registering b0 images to T1w images.
  3. It is common to scan multiple separate DWI sequences that are ultimately supposed
     to be combined for analysis. Whether scans are combined or not will alter when
     denoising is applied.
