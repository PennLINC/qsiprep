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
non-DTI q-space images. Non-DTI means that the imaging sequence used
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


Example use cases
-------------------

Consider the following use-cases:

Post-``eddy``

  You have already preprocessed your diffusion data using ``eddy``, and want
  to reconstruct it using ``MAPMRI`` and save the ODFs in a DSI Studio
  ``fib.gz`` file to perform fiber tracking and connectivity analysis.

CS-DSI group analysis

  You've acquired a compressed sensing DSI scan and would like to perform
  motion correction, coregistration to your T1w image and spatial normalization
  to MNI space. Then reconstruct ensemble average propagators using
  regularized ``3dSHORE`` and estimate the return-to-origin probability in
  MNI Space.

Fixel-based analysis on DSI data

  You have a classic Cartesian grid DSI dataset that you want to motion-correct
  and estimate ODFs with in T1 space. Then you want to convert these ODFs to the
  MRTrix sh representation and perform fixel-based analysis.


These are examples of pipelines that can be created and run through qsiprep. These
pipelines use Nipype and are therefore able to run on multiple cores. All the
dependencies of this software are included in the Docker_ image.

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
