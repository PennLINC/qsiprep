.. include:: links.rst

===========================
Processing pipeline details
===========================

``qsiprep`` adapts its pipeline depending on what data and metadata are
available and are used as the input.


A (very) high-level view of the simplest pipeline (for a dataset with only
one DWI series and no reverse PE b0 acquisitions)
is presented below:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.base import init_single_subject_wf
    wf = init_single_subject_wf(subject_id="1",
                                name="test_wf",
                                reportlets_dir=".",
                                output_dir=".",
                                bids_dir=".",
                                ignore=[],
                                debug=False,
                                write_local_bvecs=False,
                                low_mem=False,
                                anat_only=False,
                                longitudinal=False,
                                denoise_before_combining=True,
                                dwi_denoise_window=7,
                                combine_all_dwis=True,
                                omp_nthreads=1,
                                skull_strip_template='OASIS',
                                skull_strip_fixed_seed=False,
                                freesurfer=False,
                                hires=False,
                                output_spaces=['T1w', 'template'],
                                template='MNI152NLin2009cAsym',
                                output_resolution=2.0,
                                prefer_dedicated_fmaps=False,
                                motion_corr_to='iterative',
                                b0_to_t1w_transform='Rigid',
                                hmc_model='3dSHORE',
                                hmc_transform='Affine',
                                impute_slice_threshold=0.,
                                fmap_bspline=True,
                                fmap_demean=True,
                                use_syn=False,
                                force_syn=False)


T1w/T2w preprocessing
---------------------
:mod:`qsiprep.workflows.anatomical.init_anat_preproc_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.anatomical import init_anat_preproc_wf
    wf = init_anat_preproc_wf(omp_nthreads=1,
                              reportlets_dir='.',
                              output_dir='.',
                              template='MNI152NLin2009cAsym',
                              output_spaces=['T1w', 'fsnative',
                                             'template', 'fsaverage5'],
                              skull_strip_template='OASIS',
                              skull_strip_fixed_seed=False,
                              freesurfer=True,
                              longitudinal=False,
                              debug=False,
                              hires=True,
                              num_t1w=1)

The anatomical sub-workflow begins by constructing an average image by
:ref:`conforming <conformation>` all found T1w images to LPS orientation and
a common voxel size, and, in the case of multiple images, averages them into a
single reference template (see `Longitudinal processing`_).

.. _t1preproc_steps:

Brain extraction, brain tissue segmentation and spatial normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, the T1w image/average is skull-stripped using ANTs' ``antsBrainExtraction.sh``,
which is an atlas-based brain extraction workflow.

.. figure:: _static/brainextraction_t1.svg
    :scale: 100%

    Brain extraction


Once the brain mask is computed, FSL ``fast`` is utilized for brain tissue segmentation.

.. figure:: _static/segmentation.svg
    :scale: 100%

    Brain tissue segmentation.


Finally, spatial normalization to MNI-space is performed using ANTs' ``antsRegistration``
in a multiscale, mutual-information based, nonlinear registration scheme.
In particular, spatial normalization is done using the `ICBM 2009c Nonlinear
Asymmetric template (1×1×1mm) <http://nist.mni.mcgill.ca/?p=904>`_ [Fonov2011]_.

When processing images from patients with focal brain lesions (e.g. stroke, tumor
resection), it is possible to provide a lesion mask to be used during spatial
normalization to MNI-space [Brett2001]_.
ANTs will use this mask to minimize warping of healthy tissue into damaged
areas (or vice-versa).
Lesion masks should be binary NIfTI images (damaged areas = 1, everywhere else = 0)
in the same space and resolution as the T1 image, and follow the naming convention specified in
`BIDS Extension Proposal 3: Common Derivatives <https://docs.google.com/document/d/1Wwc4A6Mow4ZPPszDIWfCUCRNstn7d_zzaWPcfcHmgI4/edit#heading=h.9146wuepclkt>`_
(e.g. ``sub-001_T1w_label-lesion_roi.nii.gz``).
This file should be placed in the ``sub-*/anat`` directory of the BIDS dataset
to be run through ``qsiprep``.

.. figure:: _static/T1MNINormalization.svg
    :scale: 100%

    Animation showing T1w to MNI normalization


Longitudinal processing
~~~~~~~~~~~~~~~~~~~~~~~
In the case of multiple T1w images (across sessions and/or runs), T1w images are
merged into a single template image using FreeSurfer's `mri_robust_template`_.
This template may be *unbiased*, or equidistant from all source images, or
aligned to the first image (determined lexicographically by session label).
For two images, the additional cost of estimating an unbiased template is
trivial and is the default behavior, but, for greater than two images, the cost
can be a slowdown of an order of magnitude.
Therefore, in the case of three or more images, ``qsiprep`` constructs
templates aligned to the first image, unless passed the ``--longitudinal``
flag, which forces the estimation of an unbiased template.

.. note::

    The preprocessed T1w image defines the ``T1w`` space.
    In the case of multiple T1w images, this space may not be precisely aligned
    with any of the original images.
    Reconstructed surfaces and functional datasets will be registered to the
    ``T1w`` space, and not to the input images.


DWI preprocessing
------------------
:mod:`qsiprep.workflows.dwi.base.init_dwi_preproc_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf
    wf = init_dwi_preproc_wf(['/completely/made/up/path/sub-01_dwi.nii.gz'],
                              omp_nthreads=1,
                              ignore=[],
                              reportlets_dir='.',
                              output_dir='.',
                              template='MNI152NLin2009cAsym',
                              output_spaces=['T1w', 'template'],
                              freesurfer=False,
                              use_bbr=False,
                              dwi_denoise_window=7,
                              denoise_before_combining=True,
                              motion_corr_to='iterative',
                              b0_to_t1w_transform='Rigid',
                              hmc_model='3dSHORE',
                              hmc_transform='Affine',
                              impute_slice_threshold=0,
                              fmap_bspline=True,
                              fmap_demean=True,
                              use_syn=True,
                              force_syn=True,
                              low_mem=False,
                              num_dwi=1)

Preprocessing of :abbr:`DWI (Diffusion Weighted Image)` files is
split into multiple sub-workflows described below.

.. _dwi_hmc:

Head-motion estimation
~~~~~~~~~~~~~~~~~~~~~~
:mod:`qsiprep.workflows.dwi.hmc.init_dwi_hmc_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep.workflows.dwi.hmc import init_dwi_hmc_wf
    wf = init_dwi_hmc_wf(hmc_transform="Affine",
                         hmc_model="3dSHORE",
                         hmc_align_to="iterative",
                         mem_gb=3,
                         omp_nthreads=1,
                         write_report=False)

A long-standing issue for q-space imaging techniques, particularly DSI, has
been the lack of motion correction methods. DTI and multi-shell HARDI have
had ``eddy_correct`` and ``eddy`` in FSL, but DSI has relied on aligning the
interleaved b0 images and applying the transforms to nearby non-b0 images.

``qsiprep`` introduces a method for head motion correction that iteratively
creates target images based on ``3dSHORE`` or ``MAPMRI`` fits.
First, all b0 images are aligned to a midpoint b0 image (or the first b0 image
if ``hmc_align_to="first"``) and each non-b0 image is transformed along with
its nearest b0 image.

Then, for each non-b0 image, a ``3dSHORE`` or ``MAPMRI``
model is fit to all the other images with that image left out. The model is then
used to generate a target signal image for the gradient direction and magnitude
(i.e. q-space coordinate) of the left-out image. The left-out image is registered
to the generated target
signal image and its vector is rotated accordingly. A new model is fit on the
transformed images and their rotated vectors. The leave-one-out procedure is
then repeated on this updated DWI and gradient set.

If ``"none"`` is specified as the hmc_model, then only the b0 images are used
and the non-b0 images are transformed based on their nearest b0 image. This
is not a great idea.

Ultimately a list of 6 (or 12)-parameters per time-step is written and
fed to the :ref:`confounds workflow <dwi_confounds>`. These are used to
estimate framewise displacement.  Additionally, measures of model fits
are saved for each slice for display in a carpet plot-like thing.

.. _dwi_ref:

DWI reference image estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`qsiprep.workflows.dwi.util.init_dwi_reference_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.dwi.util import init_dwi_reference_wf
    wf = init_dwi_reference_wf(omp_nthreads=1)

This workflow estimates a reference image for a DWI series. This
procedure is different from the BOLD reference image workflow in the
sense that true brain masking isn't usually done until later in the
pipeline for DWIs. It performs a generous automasking and uses
Dipy's histogram equalization on the b0 template generated during
motion correction.

Susceptibility Distortion Correction (SDC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`qsiprep.workflows.fieldmap.base.init_sdc_wf`

.. figure:: _static/unwarping.svg
    :scale: 100%

    Applying susceptibility-derived distortion correction, based on
    fieldmap estimation.

The PEPOLAR and SyN-SDC workflows from FMRIPREP are copied here.
They operate on the output of reference estimation, after head
motion correction.

.. _resampling:

Pre-processed DWIs in a different space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`qsiprep.workflows.dwi.resampling.init_dwi_trans_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf
    wf = init_dwi_trans_wf(template="ACPC",
                           use_fieldwarp=True,
                           use_compression=True,
                           to_mni=False,
                           write_local_bvecs=True,
                           mem_gb=3,
                           omp_nthreads=1)

A DWI series is resampled to an output space. The ``output_resolution`` is
specified on the commandline call. All transformations, including head motion
correction, susceptibility distortion correction, coregistration and optionally
normalization to the template is performed in a single shot using a Lanczos kernel.

There are two ways that the gradient vectors can be saved. This workflow always
produces a FSL-style bval/bvec pair for the image and a MRTrix .b gradient table
with the rotations from the linear transforms applied. You can also write out
a ``local_bvecs`` file that contains a 3d vector that has been rotated to account
for nonlinear transforms in each voxel. I'm not aware of any software that can
use these yet, but it's an interesting idea.


.. _b0_reg:

b0 to T1w registration
~~~~~~~~~~~~~~~~~~~~~~~
:mod:`qsiprep.workflows.dwi.registration.init_b0_to_anat_registration_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf
    wf = init_b0_to_anat_registration_wf(
                                         mem_gb=3,
                                         omp_nthreads=1,
                                         transform_type="Rigid",
                                         write_report=False)

This just uses `antsRegistration`.
