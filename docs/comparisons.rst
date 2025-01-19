##########################
Comparisons to other tools
##########################

***********************************
Comparisons to other dMRI pipelines
***********************************

Other pipelines for preprocessing DWI data are currently being developed.
Below are tables comparing their current feature sets.

* `Tractoflow <https://doi.org/10.1016/j.neuroimage.2020.116889>`_
* `PreQual <https://doi.org/10.1101/2020.09.14.260240>`_
* `MRtrix3_connectome <https://github.com/BIDS-Apps/MRtrix3_connectome>`_
* `dMRIPrep <https://github.com/nipreps/dmriprep>`_


Supported Sampling Schemes
==========================

.. list-table::
   :header-rows: 1

   * -
     - QSIPrep
     - Tractoflow
     - PreQual
     - MRtrix3_connectome
     - dMRIPrep
   * - Single Shell
     - ✔
     - ✔
     - ✔
     - ✔
     - ✔
   * - Multi Shell
     - ✔
     - ✔
     - ✔
     - ✔
     - ✔
   * - Cartesian
     - ✔
     - ✘
     - ✘
     - ✘
     - ✘
   * - Random (Compressed Sensing)
     - ✔
     - ✘
     - ✘
     - ✘
     - ✘


Preprocessing
=============

.. list-table::
   :header-rows: 1

   * -
     - QSIPrep
     - Tractoflow
     - PreQual
     - MRtrix3_connectome
     - dMRIPrep
   * - BIDS App
     - ✔
     - ✔
     - ✘
     - ✔
     - ✔
   * - Gradient direction sanity check
     - Q-form matching
     - ✘
     - ``dwigradcheck``
     - ✘
     - ✘
   * - Workflow management
     - NiPyPe
     - NextFlow
     - Custom
     - Custom
     - NiPyPe
   * - MP-PCA denoising
     - ✔
     - ✔
     - ✔
     - ✔
     - ✘
   * - Patch2self denoising
     - ✔
     - ✘
     - ✘
     - ✘
     - ✘
   * - Gibbs unringing
     - ``mrdegibbs``
     - ``mrdegibbs`` (disabled)
     - ``mrdegibbs`` (disabled)
     - ``mrdegibbs``
     - ✘
   * - B1 bias field correction
     - N4
     - N4
     - N4
     - N4
     - ✘
   * - Automatic distortion group concatenation
     - ✔
     - ✘
     - ✘
     - ✔
     - ✘
   * - T1w brain extraction
     - ANTs
     - ANTs
     - ✘
     - ✘
     - ANTs
   * - Intensity normalization
     - scaled by *b*=0 means
     - *b*=0 mean set to 1000
     - ✘
     - ``mtnormalize``
     - ✘
   * - b=0 to T1w coregistration
     - ANTs linear registration
     - ANTs Non-Linear SyN
     - ✘
     - ``mrregister``
     - FSL BBR
   * - Head Motion Correction (shelled schemes)
     - ``eddy``
     - ``eddy``
     - ``eddy``
     - ``eddy``
     - ✘
   * - Head Motion Correction (Cartesian / Random Schemes)
     - SHORELine
     - ✘
     - ✘
     - ✘
     - ✘
   * - PEPOLAR EPI distortion correction
     - ``TOPUP``
     - ``TOPUP``
     - ``TOPUP``
     - ``TOPUP``
     - ``TOPUP``
   * - GRE Fieldmap EPI distortion correction
     - ✔
     - ✘
     - ✘
     - ✘
     - ✔
   * - Fieldmapless Distortion Correction
     - PE-Direction SyN
     - ✘
     - Non-Linear registration
     - SyN b0-DISCO
     - PE-Direction SyN
   * - T1w-based Normalization
     - ANTs (SyN)
     - ✘
     - ✘
     - ✘
     - ANTs (SyN)
   * - HTML Report
     - ✔
     - ✘
     - ✔
     - ✘
     - ✔
   * - Containerized
     - ✔
     - ✔
     - ✔
     - ✔
     - ✔


Quality Control
===============

.. list-table::
   :header-rows: 1

   * -
     - QSIPrep
     - Tractoflow
     - PreQual
     - MRtrix3_connectome
   * - Automated methods boilerplate
     - ✔
     - ✘
     - ✘
     - ✘
   * - HTML Preprocessing Report
     - `NiWorkflows-based <preprocessing.html#visual-reports>`_
     - ✘
     - Custom
     - EddyQuad
   * - HTML Reconstruction Report
     - NiWorkflows-based
     - ✘
     - Custom
     - ✘


***********************************
QSIPrep versus other modality preps
***********************************

Diffusion processing has idiosyncrasies that may confuse users who are used to
working with other modalities.
This section is designed to orient users who are familiar with pipelines like fMRIPrep and ASLPrep.


Output spaces
=============

With fMRIPrep and other, similar pipelines, the user defines any output spaces they want,
and the workflow will write out preprocessed data in those spaces.
With QSIPrep, the pipeline will write out the preprocessed DWI data in a native anatomical space-
typically the T1w space, aligned to the ACPC.
The "output spaces" in this case will be provided as transforms from the T1w space to the
desired output space.
It is then up to the reconstruction pipeline (typically QSIRecon) to apply these transforms to the
preprocessed DWI data to get reconstructed outputs in the requested spaces.


Output resolution
=================


Concatenation across runs
=========================
