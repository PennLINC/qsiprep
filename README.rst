.. include:: links.rst

QSIprep: Preprocessing and analysis of q-space images
=======================================================

This pipeline was developed at UCSB and UPenn for processing q-space
images.

.. image:: https://readthedocs.org/projects/qsiprep/badge/?version=latest
  :target: http://qsiprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status


About
-----

``qsiprep`` is designed to perform preprocessing and reconstruction of
non-DTI q-space images. Non-DTI means that the imaging sequence used
either:

  - A multi-shell HARDI sampling scheme
  - A Cartesian grid (aka DSI) sampling scheme
  - A random q-space sampling scheme (eg for compressed sensing)

DTI is not supported here because there are already excellent preprocessing
tools in FSL, MRTrix, Tortoise and others, and because the head motion correction algorithm
relies on voxelwise estimates of ensemble average propagators (EAPs), which
require angular and radial variability in the sampling scheme.

The ``qsiprep`` pipeline uses much of the code from ``FMRIPREP``, but has
deviated in a few noteworthy ways.

  1. All images are conformed to LPS+ instead of RAS+. This greatly simplified
     spatial operations on vectors / vector images within ANTs and working with
     DSI Studio's internal data model.
  2. Co-registration is performed with ANTs instead of FSL or FreeSurfer. We have
     removed FSL and FreeSurfer as much as possible from the pipeline in an effort
     to keep the licensing of bundled software as permissive as possible. Our testing
     also showed that ANTs was more robust for co-registering b0 images to T1w images.
  3. It is common to scan multiple separate DWI sequences that are ultimately supposed
     to be combined for analysis. Whether scans are combined or not will alter when
     denoising is applied.

The ``qsiprep`` pipeline heavily uses ANTs_, Dipy_ and MRTrix_ for preprocessing and
supports `DSI Studio`_, MRTrix_ and Dipy_ for reconstruction/analysis. It can also convert
across the file formats of these software packages so one can, for example, reconstruct
using GQI in DSI Studio and then do fixel-based analysis in MRTrix.
