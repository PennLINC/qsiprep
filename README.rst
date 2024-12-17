.. include:: links.rst

QSIPrep: Preprocessing and analysis of q-space images
=====================================================

.. image:: https://img.shields.io/badge/Source%20Code-pennlinc%2Fqsiprep-purple
  :target: https://github.com/PennLINC/qsiprep
  :alt: GitHub Repository

.. image:: https://readthedocs.org/projects/qsiprep/badge/?version=latest
  :target: http://qsiprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/docker-pennlinc/qsiprep-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/pennlinc/qsiprep/tags/
  :alt: Docker

.. image:: https://circleci.com/gh/PennLINC/qsiprep/tree/master.svg?style=svg
  :target: https://circleci.com/gh/PennLINC/qsiprep/tree/master
  :alt: Test Status

.. image:: https://img.shields.io/badge/Nature%20Methods-10.1038%2Fs41592--021--01185--5-purple
  :target: https://doi.org/10.1038/s41592-021-01185-5
  :alt: Publication DOI

.. image:: https://zenodo.org/badge/156589095.svg
  :target: https://doi.org/10.5281/zenodo.14187327
  :alt: Zenodo DOI

.. image:: https://img.shields.io/badge/License-BSD--3--Clause-green
  :target: https://opensource.org/licenses/BSD-3-Clause
  :alt: License


Full documentation at https://qsiprep.readthedocs.io

About
-----

*QSIPrep* configures pipelines for processing diffusion-weighted MRI (dMRI) data.
The main features of this software are

  1. A BIDS-app approach to preprocessing nearly all kinds of modern diffusion MRI data.
  2. Automatically generated preprocessing pipelines that correctly group, distortion correct,
     motion correct, denoise, coregister and resample your scans, producing visual reports and
     QC metrics.
  3. A system for running state-of-the-art reconstruction pipelines that include algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.
  4. A novel motion correction algorithm that works on DSI and random q-space sampling schemes

.. image:: https://github.com/PennLINC/qsiprep/raw/master/docs/_static/workflow_full.png


.. _preprocessing_def:

Preprocessing
~~~~~~~~~~~~~

The preprocessing pipelines are built based on the available BIDS inputs, ensuring that fieldmaps
are handled correctly. The preprocessing workflow performs head motion correction, susceptibility
distortion correction, MP-PCA denoising, coregistration to T1w images, spatial normalization
using ANTs_ and tissue segmentation.


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~

The outputs from the :ref:`preprocessing_def` pipelines can be reconstructed in many other
software packages.
We recommend passing *QSIPrep* derivatives along to
`QSIRecon <https://qsirecon.readthedocs.io/en/latest/>`_,
which provides a curated set of reconstruction workflows
that can run ODF/FOD reconstruction, tractography, Fixel estimation and regional
connectivity.


Note
----

The *QSIPrep* pipeline uses much of the code from *fMRIPrep*.
It is critical to note that the similarities in the code
**do not imply that the authors of QSIPrep in any way endorse or support this code or its
pipelines**.
