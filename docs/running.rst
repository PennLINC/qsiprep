.. include:: links.rst

.. _running:

###############
Running QSIPrep
###############

Most options have defaults that do not need to be changed. This page goes
through the options in the order they appear in ``qsiprep --help`` and says
what each one decides and when to change it. The full help text is on the
:doc:`cli` page.


*****************
A minimal command
*****************

Suppose the BIDS input contains::

  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-02_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-03_dwi.nii.gz
  sub-1/ses-1/fmap/sub-1_ses-1_dir-PA_epi.nii.gz

with the ``epi`` fieldmap's ``IntendedFor`` naming the three DWI runs. One way
to process these data is::

  qsiprep /path/to/bids /path/to/output participant \
      --output-resolution 1.7 \
      -w /path/to/work

The three positional arguments follow the BIDS App convention.
``--output-resolution`` has no default: it is the isotropic voxel size, in
mm, of the preprocessed DWI (see :ref:`output_resolution`). With these
options the three runs are denoised separately, concatenated, motion and
distortion corrected as one series with ``eddy`` and TOPUP, and written to
``/path/to/output/sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_space-ACPC_desc-preproc_dwi.nii.gz``.

How to run the container versions of this command is on the
:doc:`installation` page.


**************************************
Selecting subjects, sessions and files
**************************************

``--participant-label`` and ``--session-label`` take space-separated lists of
labels, with or without the ``sub-``/``ses-`` prefix. ``--bids-database-dir``
points PyBIDS at a reusable index, which saves minutes on large datasets, and
``--skip-bids-validation`` skips the validator.

``--bids-filter-file`` restricts which files are read. *QSIPrep* finds its
inputs with one query per data type::

  {
      "fmap": {"datatype": "fmap"},
      "t2w": {"datatype": "anat", "suffix": "T2w"},
      "t1w": {"datatype": "anat", "suffix": "T1w"},
      "roi": {"datatype": "anat", "suffix": "roi"},
      "dwi": {"datatype": "dwi", "suffix": "dwi"}
  }

The filter file adds BIDS entities to these queries. This file selects
session ``MR1`` for the T1w and DWI queries, and runs 2 and 3 of the DWI
query only::

  {
      "t1w": {"session": "MR1"},
      "dwi": {"session": "MR1", "run": [2, 3]}
  }

Values may be lists, and ``"regex_search": "true"`` inside a query turns its
values into regular expressions (``"acquisition": "(?i)mprage"``). The list
of entities is in the `PyBIDS configuration
<https://github.com/bids-standard/pybids/blob/main/src/bids/layout/config/bids.json>`__.
If you want some scans combined and others processed separately, run
*QSIPrep* more than once with different filter files.


.. _workflow_scope:

**************
Workflow scope
**************

``--boilerplate-only`` writes the methods text (see :ref:`boilerplate`) and
exits. ``--reports-only`` rebuilds the HTML reports from an existing output
directory without running anything.

``--ignore`` switches off processing the data would otherwise trigger, and
``--force`` switches on processing the metadata would otherwise skip. Both
take space-separated lists.

.. list-table:: ``--ignore`` values
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - ``fieldmaps``
     - Skip the ``fmap/`` directory. Reverse phase-encoded DWI runs still
       drive distortion correction.
   * - ``pepolar-dwis``
     - Stop pairing DWI series with each other for PEPOLAR estimation. The
       series are still processed, corrected by a fieldmap they are linked to,
       by a fieldmap-less method, or not at all.
   * - ``sdc``
     - No susceptibility distortion correction of any kind.
   * - ``t2w``
     - Drop T2w images, including T2w-based distortion correction.
   * - ``phase``
     - Ignore ``part-phase`` images; denoise as magnitude data.
   * - ``shims``
     - Treat all ``ShimSetting`` values as compatible when grouping series.
   * - ``fov``
     - Concatenate series with differently oriented fields of view anyway.
       The distortion correction is then misapplied to some of them.
   * - ``gradwarp``
     - No gradient nonlinearity correction, including the gradient deviation
       map.
   * - ``jacobian``
     - No Jacobian intensity modulation by *QSIPrep* itself. ``eddy``'s
       internal modulation is unaffected (see :ref:`jacobian_flags`).

.. list-table:: ``--force`` values
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - ``sdc-anat-reference``
     - Use the ``--sdc-anat-reference`` image for every DWI series, instead of
       only for series no fieldmap reaches.
   * - ``gradwarp3D`` / ``gradwarp1D``
     - Apply the full 3D, or only the through-plane, gradient nonlinearity
       correction to every run regardless of ``ImageType``. Mutually
       exclusive; both need ``--gradient-file``.
   * - ``jacobian``
     - Also Jacobian-modulate the fieldmap-less T2Wreg correction, which
       TORTOISE leaves unmodulated.
   * - ``gre-sdc-after-eddy``
     - Apply a GRE fieldmap to ``eddy``'s outputs, as *QSIPrep* did before
       26.1, instead of handing it to ``eddy``. For comparing the two on real
       data. Deprecated, and will be removed in a future release.


*********************
Anatomical processing
*********************

``--anat-modality`` picks the *anatomical reference*, ``T1w`` (the default)
or ``T2w``. The reference is skull-stripped and segmented, defines the
subject's ``ACPC`` space, and is normalized to the template. Choose ``none``
to use no anatomical image at all; the b=0 reference is then rigidly aligned
to the template instead. This is rarely a good idea, because it is rare to
have dMRI data without a T1w or T2w of the same person.

``--subject-anatomical-reference`` decides how many anatomical spaces a
subject gets. ``first-lex`` (the default) aligns all sessions' anatomical
images to the first one, ``unbiased`` builds a mid-point template from them,
and ``sessionwise`` gives every session its own space and its own report.
See :ref:`anatomical_methods` for what each involves.

``--anat-biascorrect`` runs N4 on the anatomical images (``n4``, the default),
never (``none``), or only when ``ImageType`` does not contain ``NORM``
(``auto``). ``--anatomical-template`` has one choice,
``MNI152NLin2009cAsym``; ``--infant`` replaces it with the infant template.
``--skip-anat-based-spatial-normalization`` skips the nonlinear
registration to the template, which saves about twenty minutes; the
template-space anatomical derivatives and the ``ACPC`` to template transform
are then not written.

``--infant`` swaps the template for the MNIInfant cohort matching the
participant's age in months, read from ``participants.tsv`` or the
``*_sessions.tsv`` file. It requires ``--subject-anatomical-reference
sessionwise``. The cohort selection follows `Nibabies
<https://nibabies.readthedocs.io/>`_.


.. _per_series:

************************
Per-series preprocessing
************************

Each DWI series is denoised on its own before series are concatenated. The
order is fixed: MP-PCA or patch2self denoising on the raw data, then Gibbs
unringing, then, after resampling, B1 bias field correction and b=0
intensity harmonization.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - Denoising
     - Gibbs unringing
     - B1 bias field correction
   * - What it removes
     - Random noise
     - Ringing at sharp edges
     - Smooth intensity non-uniformity
   * - Tools
     - ``dwidenoise`` (MRtrix3), ``dwidenoise2``, ``patch2self`` (DIPY)
     - ``mrdegibbs`` (MRtrix3), ``rpg`` (TORTOISE)
     - ``N4BiasFieldCorrection`` on the b=0 images, applied to the series
   * - Default
     - ``dwidenoise``
     - Off
     - On
   * - Option
     - ``--denoise-method``
     - ``--unringing-method``
     - ``--dwi-biascorrect``

``--dwidenoise-window`` sets the ``dwidenoise`` patch size in voxels, an odd
integer or ``auto`` (the default, derived from the number of volumes).
``dwidenoise2`` sizes its patches from its own schedule, set with
``--dwidenoise2-config`` (see below). ``rpg`` unringing is the method for
partial Fourier acquisitions; ``mrdegibbs`` assumes full Fourier sampling.
``--dwi-biascorrect auto`` skips bias correction when every DWI's
``ImageType`` contains ``NORM``, and ``none`` skips it always. We recommend
``none`` for prescan-normalized data. ``--no-b0-harmonization`` skips the
rescaling that matches b=0 intensities across series. ``--b0-threshold``
(default 100) is the b-value below which a volume counts as b=0.

With ``part-phase`` data, ``dwidenoise`` and ``dwidenoise2`` denoise the
complex signal. Whether the complex data are carried into unringing depends
on ``--mrtrix-version``: only the ``dev`` branch of MRtrix3 has a
complex-valued ``mrdegibbs``; with ``stable`` the data are reduced to
magnitude after denoising. ``--ignore phase`` drops the phase images
altogether.

dwidenoise2 settings
====================

``--dwidenoise2-config`` takes a JSON file with settings for
``--denoise-method dwidenoise2``. Every key is optional and unknown keys are
an error. Each key sets the ``dwidenoise2`` option of the same name, except
``filter_method``, which sets ``-filter``, and ``schedule``.

================================  ==========================================================
Key                               JSON value
================================  ==========================================================
``aggregator``                    ``"exclusive"``, ``"gaussian"``, ``"invl0"``, ``"rank"``
                                  or ``"uniform"``
``datatype``                      ``"float32"`` or ``"float64"``
``debias_anchor``                 ``"sample"`` or ``"group_mean"``
``decomposition``                 ``"bdcsvd"`` or ``"selfadjoint"``
``demean``                        ``"none"``, ``"volume_groups"``, ``"shells"`` or ``"all"``
``demod_axes``                    a list of non-negative integers, such as ``[0, 1]``
``demodulate``                    ``"none"``, ``"linear"``, ``"hann"`` or ``"apc"``
``estimator``                     ``"exp1"``, ``"exp2"``, ``"med"``, ``"mrm2023"`` or
                                  ``"tbme2022"``
``filter_method``                 ``"optshrink"``, ``"optthresh"`` or ``"truncate"``
``fixed_rank``                    an integer of at least 1
``noise_dof``                     an integer of at least 1
``noise_in``                      a number of at least 0; noise-map files are not supported
``preserve_noise_bias``           ``true`` or ``false``
``vst_method``                    ``"none"``, ``"linear"``, ``"foi"``, ``"koay"`` or ``"mom"``
``schedule``                      a list of iterations, or the name of a bundled schedule
================================  ==========================================================

``-aggregator_fwhm``, ``-rankpermm_in`` and the diagnostic export options
cannot be set. ``demodulate`` other than ``"none"`` needs ``part-phase``
data. ``noise_in`` only seeds the variance-stabilizing transform.

``schedule`` lists the iterations of the multi-resolution noise estimation.
Each iteration is an object whose keys are the columns of a ``dwidenoise2``
schedule file: ``spatial_subsample`` (an integer or a list of three),
``kernel`` (``"aspect=2.0"``, ``"rmse=0.02"``, ``"rank"``, ``"radius=4"``,
``"voxels=100"``, ``"cuboid=1x"`` or ``"rank_fixed"``), ``smooth_noise`` and
``update_noise`` (booleans), ``temporal_subsample`` (a number in (0, 1]),
``partitions`` and ``max_partition_size``. The last iteration is the
reconstruction pass.

.. code-block:: json

  {
    "estimator": "exp2",
    "decomposition": "bdcsvd",
    "schedule": [
      {"spatial_subsample": 8, "kernel": "aspect=2.0", "update_noise": true},
      {"spatial_subsample": [4, 4, 2], "kernel": "rmse=0.02", "update_noise": true},
      {"spatial_subsample": 2, "kernel": "rank", "update_noise": false}
    ]
  }

``schedule`` may instead name a bundled schedule: ``"default"``,
``"legacy"`` or ``"vlarge"`` (recommended by ``dwidenoise2`` for more than
255 volumes). Without a ``schedule`` key, ``dwidenoise2`` uses its default,
except that ``fixed_rank`` selects the single-iteration ``fixedrank``
schedule and ``"vst_method": "none"`` a single iteration. The file is checked
when the command line is parsed, and a copy is written to
``sub-<label>/log/<run uuid>/dwidenoise2.json`` in the output directory.


.. _grouping_flags:

**************************
Grouping and concatenation
**************************

By default the DWI series of a session are concatenated into one output, so
head motion correction sees as much data as possible. The rules are on the
:ref:`data preparation <grouping>` page; two options change them.

``--separate-all-dwis`` processes every DWI series on its own, giving one
output per input file. ``--distortion-group-merge`` decides what happens to
the corrected halves of an output when they were corrected separately, as
they are for a reverse phase-encoded pair with TORTOISE: ``concat`` (the
default) appends them along the fourth dimension, ``average`` averages the
images at the same q-space coordinate, and ``none`` writes each correction
unit as its own output. Whether an output's series are corrected together
or separately depends on the head motion method: ``eddy`` with TOPUP corrects
a reverse phase-encoded pair as one series, so there is nothing to merge.

Use ``average`` for HCP-style data, where the whole sampling scheme was
acquired in both phase encoding directions::

  --hmc-method tortoise --distortion-group-merge average

This gives one image per q-space sample, like the HCP pipelines. ``concat``
keeps both copies.


.. _hmc_flags:

***************************************
Head motion and eddy-current correction
***************************************

``--hmc-method`` picks the software:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Method
     - Works on
     - Corrects
   * - ``eddy`` (default)
     - Single-shell and multi-shell schemes only
     - Head motion, eddy currents, and (with TOPUP or a GRE fieldmap)
       susceptibility distortion in one model. FSL ``eddy``.
   * - ``tortoise``
     - Any scheme, including Cartesian DSI and random q-space sampling
     - Rigid head motion and 24-parameter quadratic eddy currents. TORTOISE
       DIFFPREP.
   * - ``shoreline``
     - Any scheme
     - Head motion only, by registering each volume to a signal-model
       prediction. Scheduled for removal; use ``tortoise`` for non-shelled
       data.

``eddy`` refuses non-shelled data: the run stops with an
``eddy-requires-shelled`` error. What each method does, step by step, is in
:doc:`methods/hmc_eddy`, :doc:`methods/hmc_tortoise` and
:doc:`methods/hmc_shoreline`.

Each method reads its settings from a JSON file, and a default is used when
none is given:

``--eddy-config``
  The arguments to ``eddy``. The default is
  `eddy_params.json <https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/eddy_params.json>`__.
  Two keys are worth knowing: ``"method"`` (``jac`` or ``lsr``) is ``eddy``'s
  resampling method, which also decides whether ``eddy`` applies Jacobian
  modulation itself, and ``"estimate_move_by_susceptibility"`` lets ``eddy``
  model how the field changes with head position when it has a field to work
  with.
``--diffprep-config``
  The arguments to DIFFPREP. The default is
  `diffprep_params.json <https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/diffprep_params.json>`__.
  ``"correction_mode"`` selects ``"motion"`` (rigid head motion only),
  ``"quadratic"`` (the default) or ``"cubic"`` eddy-current models.
  ``"drbuddi_synth_shell_bval"`` is described with DRBUDDI in
  :doc:`methods/hmc_tortoise`.
``--shoreline-config``
  ``"model"`` (``"3dshore"``, ``"tensor"`` or ``"none"``), ``"iters"``
  (default 2) and ``"transform"`` (``"Affine"`` or ``"Rigid"``). Only valid
  with ``--hmc-method shoreline``.

GPU acceleration
================

``--gpu`` runs selected tasks on the GPU: ``eddy``, ``diffprep``,
``drbuddi``, ``synthstrip``, ``synthseg``, or ``all``. GPU memory is usually
the limit, so tasks are picked individually; an 8 GB card typically runs
``eddy``, ``diffprep`` and ``drbuddi`` but not the FreeSurfer deep-learning
tools. ``--gpu`` overrides ``use_cuda`` in the eddy and DIFFPREP config
files. GPU builds are not numerically identical to the CPU builds, so this
changes results, not only run time. The GPU must be exposed to the container
(``docker run --gpus all`` or ``apptainer run --nv``). The image ships CUDA
12.2. ``--tortoise-gpu-cpu-ratio`` tunes how DIFFPREP shares volumes between
the GPU and the CPU threads; the help text gives the formula.


.. _sdc_flags:

************************************
Susceptibility distortion correction
************************************

Which correction a series gets is decided from the data: a fieldmap linked to
the series is used if there is one, and reverse phase encoding is preferred
over a GRE fieldmap when both are linked. The options below change the tool
and provide a fallback for series that no fieldmap reaches.

``--sdc-method``
  The tool for reverse phase-encoded (PEPOLAR) data. ``auto`` (the default)
  is TOPUP for ``eddy`` and DRBUDDI otherwise. ``topup`` and
  ``topup+drbuddi`` (TOPUP inside ``eddy``, then DRBUDDI refining its result)
  require ``--hmc-method eddy``; ``drbuddi`` works with every method.
``--sdc-anat-reference``
  A fieldmap-less correction for series that no fieldmap reaches. ``synb0``
  synthesizes a distortion-free b=0 from the T1w with SynB0-DISCO; ``t2w``
  registers to the T2w (TORTOISE T2Wreg); ``invt1w`` registers to the
  inverted-contrast T1w with SyN, as fMRIPrep does; ``auto`` picks ``synb0``
  with a T1w, else ``t2w`` with a T2w, else nothing. The default ``none``
  leaves such series uncorrected. Add ``--force sdc-anat-reference`` to use
  the anatomical reference for every series, fieldmap or not.

What runs for each kind of data:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Data available
     - ``--hmc-method eddy``
     - ``--hmc-method tortoise``
     - ``--hmc-method shoreline``
   * - Reverse phase-encoded DWI series or ``epi`` fieldmap
     - TOPUP inside ``eddy``; or DRBUDDI after ``eddy``; or both
     - DRBUDDI after DIFFPREP
     - DRBUDDI after SHORELine
   * - GRE fieldmap
     - Inside ``eddy``
     - Applied after DIFFPREP
     - Applied after SHORELine
   * - T1w and ``--sdc-anat-reference synb0``
     - Synthetic b=0 joins the TOPUP inputs (with ``topup`` in
       ``--sdc-method``)
     - T2Wreg to the synthetic b=0
     - Not corrected
   * - T2w and ``--sdc-anat-reference t2w``
     - Not corrected
     - T2Wreg to the T2w
     - Not corrected
   * - T1w and ``--sdc-anat-reference invt1w``
     - SyN after ``eddy``
     - SyN after DIFFPREP
     - SyN after SHORELine
   * - Nothing
     - Not corrected
     - Not corrected
     - Not corrected

A selection that cannot correct a series is reported as a warning in the log
and the report, or as an error if the fieldmap-less method was curated or
forced. The tools themselves are described in :doc:`methods/sdc`.

.. _gre_init_flags:

Starting DRBUDDI or T2Wreg from a GRE fieldmap
==============================================

When a series is corrected by DRBUDDI or T2Wreg and a GRE fieldmap also
lists it, the registration starts from the field the GRE fieldmap measured
and refines it, instead of the GRE fieldmap going unused. There is no option
to turn this on; it happens whenever the linkage is there, and ``--ignore
fieldmaps`` or ``--ignore sdc`` turns it off. It applies to:

* a reverse phase-encoded pair corrected by DRBUDDI: ``--hmc-method
  tortoise`` or ``shoreline``, or ``--hmc-method eddy --sdc-method drbuddi``
  (not ``topup+drbuddi``, where DRBUDDI only refines TOPUP);
* an anatomical reference forced with ``--force sdc-anat-reference``,
  corrected by T2Wreg (``--hmc-method tortoise``).

The GRE fieldmap must list the series. For a reverse phase-encoded pair of
DWI series, its ``IntendedFor`` is enough as long as nothing in the session
uses ``B0FieldIdentifier``/``B0FieldSource``; once anything does, link the
pair and the GRE fieldmap that way::

    sub-01/
      fmap/
        sub-01_magnitude1.json   {"B0FieldIdentifier": "gre", ...}
        sub-01_magnitude2.json   {"B0FieldIdentifier": "gre", ...}
        sub-01_phasediff.json    {"B0FieldIdentifier": "gre", ...}
      dwi/
        sub-01_dir-AP_dwi.json   {"B0FieldIdentifier": "pepolar",
                                  "B0FieldSource": ["pepolar", "gre"], ...}
        sub-01_dir-PA_dwi.json   {"B0FieldIdentifier": "pepolar",
                                  "B0FieldSource": ["pepolar", "gre"], ...}

For an ``epi`` fieldmap, have both it and the GRE fieldmap name the DWI
series in their ``IntendedFor``. In the ``qsiplan`` report the series is
then ``corrected by: pepolar`` with the GRE fieldmap listed as ``also
eligible`` and labeled ``(initializes DRBUDDI/T2Wreg)``. During the run,
*QSIPrep* logs ``Initializing DRBUDDI for <output> with GRE fieldmap <id>``,
and the report's distortion correction entry ends in ``(GRE-initialized)``.


.. _output_resolution:

***************************
Output space and resolution
***************************

All outputs are written in the subject's ``ACPC`` space: aligned to the
anatomical reference and rotated so that the origin is at the anterior
commissure, with the same orientation convention as the MNI templates. The
T1w cannot align white matter to a template accurately, so the preprocessed
DWI is not written in template space; spatial normalization is done after
models are fit, in `QSIRecon`_. The transform from ``ACPC`` to
``MNI152NLin2009cAsym`` is written with the anatomical derivatives.

``--output-resolution`` sets the isotropic voxel size of the preprocessed
DWI. Pass the acquired resolution to keep it, or a smaller value to upsample;
some downstream methods, such as fixel-based analysis, recommend at least
1.3 mm. Head motion correction, distortion correction, coregistration and
resampling are combined so the data are interpolated as few times as
possible (once with SHORELine, twice with ``eddy`` and DIFFPREP, which write
their own corrected volumes). Resampling uses Lanczos windowed sinc
interpolation, or linear interpolation when upsampling by more than 10%.

``--dwi2anat-dof`` is the number of degrees of freedom of the DWI to
anatomical registration, 6 (rigid, the default) or 12 (affine).

.. _dwiref_flags:

DWI reference construction
==========================

``--dwiref-definition`` decides which image is registered to the anatomical
reference. ``distortion-group`` (the default) registers each distortion
group's own b=0 reference. ``subject`` first builds one mid-point template
from every group's reference, registers that once, and has every group
inherit the result, so that data from different groups end up on exactly the
same alignment. ``--dwiref-construction-iters`` (at least 2) and
``--dwiref-construction-transform`` (``Rigid``, ``Affine``, ``BSplineSyN`` or
``SyN``) configure the template build. Which transforms are written in each
case is described in :ref:`transforms`.


.. _gradwarp_flags:

**********************************************
Gradient nonlinearity and intensity modulation
**********************************************

``--gradient-file`` enables gradient nonlinearity correction; the accepted
file formats are on the :ref:`data preparation <gradient_files>` page. Two
things are corrected: the spatial displacement of voxels, and the diffusion
encoding, which is written as a voxelwise gradient deviation map
(``*_graddev.nii.gz``). How much spatial correction a run gets is read from
its ``ImageType`` field:

===================  =======================================================
``ImageType`` tag    Behavior
===================  =======================================================
(no ``DIS`` tag)     Full 3D correction
``DIS2D``            Through-plane correction only; the scanner already
                     corrected in-plane distortion
``DIS3D``            No spatial correction; the scanner already applied it
===================  =======================================================

``--force gradwarp3D`` or ``--force gradwarp1D`` overrides the tag for every
run, and ``--ignore gradwarp`` disables both corrections. The deviation map
is written for every run that is not ignored, ``DIS3D`` included, because
no scanner corrects the encoding. It is not written for outputs assembled by
``--distortion-group-merge``.

.. _jacobian_flags:

Intensity modulation
====================

Correcting a spatial distortion moves signal between voxels, so *QSIPrep*
also rescales the corrected image by the local volume change, following
TORTOISE. Which component applies it depends on the backend; the table is in
:ref:`jacobian_methods`. ``--ignore jacobian`` disables only the modulation
*QSIPrep* itself applies. ``eddy`` modulates its own eddy-current and
susceptibility corrections internally whenever its resampling method is
``jac`` (the default), and *QSIPrep* cannot undo that; with ``"method":
"lsr"`` in ``--eddy-config`` those corrections are not modulated at all.
``--force jacobian`` additionally modulates the T2Wreg correction.


***********************
Resources and execution
***********************

``--nprocs`` (also ``--nthreads`` or ``--n-cpus``) caps the number of
threads across all processes and ``--omp-nthreads`` the number per process;
``--mem`` (or ``--mem-mb``) caps memory in MB, and accepts a ``G`` suffix.
``--low-mem`` trades disk in the working directory for memory.
``--use-plugin`` (or ``--nipype-plugin-file``) points at a Nipype plugin
file for cluster schedulers.

``-w``/``--work-dir`` is where intermediate files go; a run that stops can be
restarted with the same working directory and picks up where it left off.
``--report-output-level`` chooses where the HTML reports are written
(``root``, ``subject``, ``session``, or ``auto``, which is ``session`` for
``--subject-anatomical-reference sessionwise`` and ``root`` otherwise).
``--config-file`` loads the settings of a previous run (its ``qsiprep.toml``
is in the log directory); command-line options override it.
``--stop-on-first-crash`` stops at the first failed node instead of running
everything that does not depend on it. ``--resource-monitor`` records
memory and CPU use per node. ``--notrack`` opts out of usage reporting.
``-v``/``--verbose`` raises the log level; ``-vvv`` is debug. ``--version``
prints the version and exits.

``--sloppy``, ``--debug`` and ``--write-graph`` are for developing *QSIPrep*
itself. ``--sloppy`` output is not fit for analysis.


.. _reading_the_log:

***************
Reading the log
***************

With ``-v -v``, the terminal shows lines like::

  [Node] Setting-up "qsiprep_wf.single_subject_PNC_wf.dwi_finalize_acq_realistic_wf.transform_dwis_t1.final_b0_ref.b0ref_reportlet" in "/scratch/qsiprep_wf/single_subject_PNC_wf/dwi_finalize_acq_realistic_wf/transform_dwis_t1/final_b0_ref/b0ref_reportlet".
    201229-21:33:46,213 nipype.workflow INFO:
      [Node] Running "b0ref_reportlet" ("niworkflows.interfaces.registration.SimpleBeforeAfterRPT")
    201229-21:33:48,51 nipype.workflow INFO:
      [MultiProc] Running 2 tasks, and 3 jobs ready. Free memory (GB): 3.70/4.00, Free processors: 0/2.
                        Currently running:
                          * qsiprep_wf.single_subject_PNC_wf.dwi_finalize_acq_realistic_wf.transform_dwis_t1.final_b0_ref.b0ref_reportlet
                          * qsiprep_wf.single_subject_PNC_wf.anat_preproc_wf.mni_mask

These describe what is running. Here ``b0ref_reportlet`` and ``mni_mask``
run at the same time, and both of the two processors allowed by
``--nprocs 2`` are busy. The quoted name is the node, and the dotted path
before it is the chain of workflows it belongs to. Those workflow names
appear in the graphs on the :doc:`methods/index` pages: ``mni_mask`` is part
of the anatomical workflow, ``b0ref_reportlet`` of the DWI reference
workflow, and both sit under the subject workflow.
