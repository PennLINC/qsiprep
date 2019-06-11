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

  1. A novel :ref:`preprocessing_def` pipeline for non-DTI q-space imaging sequences
  2. A BIDS-app approach to preprocessing with ``TOPUP`` and ``eddy`` in a larger workflow
  3. A system for building a :ref:`reconstruction_def` pipeline that includes algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.

.. figure:: _static/qsiprep_workflow.svg
   :scale: 75%



.. _preprocessing_def:

Preprocessing
~~~~~~~~~~~~~~~

The preprocessing pipelines are designed to perform preprocessing and reconstruction of
q-space images. As of version 0.4 single shell sequences can be processed with FSL tools
if you use the ``--hmc-model eddy`` flag and the ``--eddy-config /path/to/config.json``
options. The preprocessing workflow performs
head motion correction, susceptibility distortion correction, MP-PCA denoising,
coregistration to T1w images, spatial normalization using ANTs_ and tissue segmentation.


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~~~

The outputs from these preprocessing pipelines can be reconstructed in many other
software packages. We provide a curated set of :ref:`recon_workflows` in ``qsiprep``
that can run ODF/FOD reconstruction, tractography, Fixel estimation and regional
connectivity.



Note
------

The ``qsiprep`` pipeline uses much of the code from ``FMRIPREP``. It is critical
to note that the similarities in the code **do not imply that the authors of
FMRIPREP in any way endorse or support this code or its pipelines**.
