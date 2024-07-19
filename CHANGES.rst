0.22.0 (July 19, 2024)
======================

Continuing the update to current NiPreps best practices, an update to PyAFQ and a few bugfixes.

 * [ENH] Compress recon derivatives (#787)
 * [FIX] Make hsvs workflow work (#784)
 * [FIX] Use new bundle names (#783)
 * [RF] Drop functions to read sidecar JSONs (#760)
 * [FIX] FUGUE bug (#769)
 * [ENH] Save SIFT2 mu parameter (#774)
 * [FIX] Fix the mif2fib workflow (#778)
 * [FIX] CI test file checking
 * [CI] Update expected AFQ outputs again
 * [CI] add optional bundles (#771)
 * [ENH] Update dependencies in docker image (#768)
 * [ENH] Upgrade pyAFQ to 1.3.2 (#764)
 * [RF] Run integration tests with pytest (#763)
 * [FIX] API documentation (#748)
 * [ENH] Update to NiPreps-style config file (#745)


0.21.4 (May 4, 2024)
====================


## What's Changed
The effort to update to current NiPreps standards has begun, plus another bugfix in the TOPUP workflow.

### Other Changes
* Remove copy of Niworkflows in favor of dependency by @tsalo in https://github.com/PennLINC/qsiprep/pull/709
* [FIX] stop bizarre argsort behavior in best b=0 by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/744



0.21.3 (May 2, 2024)
====================

## What's Changed

This update addresses an important coregistration bug (#740), makes CPU Eddy much more usable and fixes a bug in the PyAFQ workflow.

### Other Changes
* [FIX] PyAFQ was not being written to derivatives by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/738
* [FIX] use b0 ref from eddy-processed data by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/739
* [MAINT] remove pypi by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/742
* [ENH] add parallel argument for topup by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/741
* [ENH] Allow multithreading in eddy by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/743



0.21.1 (Apr 25, 2024)
=====================

There are important software updates in this release, along with a lot of infrastructure improvements.

## Important Changes
 * FSL version 6.0.7.8 is now in qsiprep. This contains 2 [serious bugfixes](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/development/history/changelog-6.0.7.8.md). One has to do with susceptibility-by-volume correction and the other resulted in incorrect CNR values being calculated.
 * DSI Studio has been updated to fix a bug in [Neighboring DWI Correlation](https://github.com/frankyeh/DSI-Studio/issues/84).
 * You can use `"mporder"` in your eddy config json file and the slice timings will automatically be created and passed to eddy (even if you're concatenating runs)
 * Recon workflows can now include a "bundle mapping" and "scalar mapping", where scalars created in individual workflows can be mapped to a template or summarized inside autotrack bundles. This does not do tract profiles or surface mapping - yet. See `qsiprep/data/pipelines/hbcd_scalar_maps.json` for an example.
 * The recon derivatives are written approximately according to BEP-016

## New Features

* [ENH] Add workflow to resample recon scalars to template space by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/688
* Support complex-valued dwidenoising by @tsalo in https://github.com/PennLINC/qsiprep/pull/679
* [ENH] Add ReconScalars node to recon workflows by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/683
* Pin current version of pyAFQ (1.3.1) by @arokem in https://github.com/PennLINC/qsiprep/pull/729
* [ENH] Add TORTOISE Estimator recon workflows by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/674
* [ENH] Make Eddy's slice2vol much easier to use by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/710

## Other Changes
* Apply stylistic changes to `workflows/base.py` by @tsalo in https://github.com/PennLINC/qsiprep/pull/678
* Pin Nilearn version by @tsalo in https://github.com/PennLINC/qsiprep/pull/687
* Add a series of infrastructure files from ASLPrep/XCP-D by @tsalo in https://github.com/PennLINC/qsiprep/pull/684
* Run isort and remove unused imports by @tsalo in https://github.com/PennLINC/qsiprep/pull/690
* Apply stylistic changes to `workflows/utils.py` by @tsalo in https://github.com/PennLINC/qsiprep/pull/680
* Remove unused classes flagged by vulture by @tsalo in https://github.com/PennLINC/qsiprep/pull/693
* [RF] Use Black to reformat package by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/701

## Bugfixes
* [FIX] Prevent undecodable terminal output in calc_connectivity by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/711
* [FIX] save recon anat files by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/702
* [FIX] revert citeproc for old pandoc by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/736
* [FIX] recon workflows when --anat-modality T2w was used  by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/734
* [FIX] remove old nodes from csdsi_3dshore.json by @kjamison in https://github.com/PennLINC/qsiprep/pull/733
* [ENH] Update TORTOISE for lower cuda memory use by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/730



0.20.0 (Jan 12, 2024)
======================
Please note there is no corresponding release on PyPI for this version

## New features
* [ENH] allow topup+drbuddi for hbcd by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/667
   * Adds a 2-stage distortion correction option `--pepolar-method TOPUP+DRBUDDI`, which will run TOPUP -> Eddy -> DRBUDDI
* [ENH] Use UKB processed data as input for recon workflows by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/651
   * This adds the --recon-input-pipeline, which lets you run recon workflows on UKB data
* [ENH] Update to python 3.10 by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/670
   * Hopefully this will address the hang-after-crashing problem in the recent releases

## Bugfixes/Docs
* DOC: Add SMeisler and JHLegarreta to contributors list by @jhlegarreta in https://github.com/PennLINC/qsiprep/pull/642
* Fixes typos on FreeSurfer requirements for ss3t hsvs recon by @pcamach2 in https://github.com/PennLINC/qsiprep/pull/414
* Fix RTD build by @tsalo in https://github.com/PennLINC/qsiprep/pull/652
* ENH: conform bvals to shells separated by b0_threshold by @cookpa in https://github.com/PennLINC/qsiprep/pull/660
* [FIX] remove unneeded "method" from tracking by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/641
* FIX: allow finding of lesion rois by @psadil in https://github.com/PennLINC/qsiprep/pull/659
* MISC: Remove outdated dsi_studio tracking parameters by @cookpa in https://github.com/PennLINC/qsiprep/pull/668
* [DOC] Add documentation for dsi_studio_autotrack reconstruction workflow by @valeriejill in https://github.com/PennLINC/qsiprep/pull/669
* [ENH] Update BIDS validator to 1.8.4 by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/671


0.19.0 (August 10, 2023)
========================

Addresses stability issues in the 0.18 releases. Huge improvements to AutoTrack recon workflow
CPU use and improvements in memory use for synthseg and synthstrip

 * [ENH] limit the synths to 1 thread (#608)
 * [DOC] fix typo in docs (#606)
 * [ENH] Stabilize autotrack performance (#604)
 * [CI] Add test for tensor-based head motion correction (#605)
 * [FIX] fixes steinhardt computation (#603)


0.18.1 (June 26, 2023)
======================

Bugfixes since 0.18.0

Bugfix:
 * [FIX] add btable to merge when averaging outputs (#594)

0.18.0 (June 9, 2023)
=====================

No technical changes to the pipeline here, but citations and methods boilerplate have been updated to
reflect the changes in 0.18.0alpha0.



0.18.0alpha0 (May 26, 2023)
===========================

First release moving towards 1.0! Please open bug reports if anything suspicious comes up. This release
changes the anatomical workflow significantly, synthstrip and synthseg are used. The recon workflow
"dsi_studio_autotrack" has also been added.

## What's Changed
* Bump sentry-sdk from 0.13.1 to 1.14.0 by @dependabot in https://github.com/PennLINC/qsiprep/pull/539
* [ENH] Update FreeSurfer to 7.3.1, dmri-amico to 1.5.4 by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/537
* WIP: ENH: Make pyAFQ tests faster, add export all by @36000 in https://github.com/PennLINC/qsiprep/pull/534
* [ENH] move biascorrect so it runs on resampled data by default by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/527
* [Fix] Fix threading on DRBUDDI interface by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/540
* [ENH] add CNR to the imageqc.csv by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/541
* [FIX] pin pandas version to < 2.0.0 by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/543
* ENH: Replace avscale with non-fsl tools by @jbh1091 in https://github.com/PennLINC/qsiprep/pull/542
* ENH: Replace fsl applymask by @jbh1091 in https://github.com/PennLINC/qsiprep/pull/544
* Replace fsl split by @jbh1091 in https://github.com/PennLINC/qsiprep/pull/548
* [FIX] Update distortion_group_merge.py by @smeisler in https://github.com/PennLINC/qsiprep/pull/555
* [ENH] Redo anatomical workflow by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/553
* [FIX] remove pre bids-filter acq type argument by @octomike in https://github.com/PennLINC/qsiprep/pull/557
* FIX: Replace deprecated `np.int` instances by @smeisler in https://github.com/PennLINC/qsiprep/pull/558
* [WIP] ENH: 482 remove fsl dependency by @jbh1091 in https://github.com/PennLINC/qsiprep/pull/498
* [ENH] Update TORTOISE for improved T2w registration by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/564
* [FIX] T2w anat-modality issues by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/565
* [FIX] update boost in tortoise by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/569
* [FIX] connections on multi-anat workflow by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/572
* [ENH] Update DSI Studio to the latest commit by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/573
* [ENH] Add DSI Studio AutoTrack recon workflow by @mattcieslak in https://github.com/PennLINC/qsiprep/pull/576

## New Contributors
* @dependabot made their first contribution in https://github.com/PennLINC/qsiprep/pull/539
* @jbh1091 made their first contribution in https://github.com/PennLINC/qsiprep/pull/542
* @smeisler made their first contribution in https://github.com/PennLINC/qsiprep/pull/555

**Full Changelog**: https://github.com/PennLINC/qsiprep/compare/0.17.0...0.18.0alpha0


0.16.1 (October 10, 2022)
=========================

Adds a critical fix for ABCD-style acquisitions (described in #449). This change forces
TOPUP to use the raw, unprocessed b=0 images from the DWI series and the epi fieldmaps to
estimate distortion. Previously, the most-denoised version of each image was used in
TOPUP. To disable this change and return to the previous behavior, use the
`--denoised-image-sdc` flag.

Note, **this is a change in the default behavior of QSIPrep!!**

*Upgrades*

 * Update ITK to 5.3, update ANTs #449
 * Add `--denoised-image-sdc` #465


*Bug fixes*

 * Use safe_load instead of load for yaml #443
 * Add fugue and prelude back to the qsiprep image #463


0.16.0RC2 (June 1, 2022)
========================

 * Adds multithreading to connectome2tck #429

0.16.0RC2 (June 1, 2022)
========================

Fixes a naming error in the schaefer 400 atlas #428

0.16.0RC1 (May 30, 2022)
========================

Major additions to the reconstruction workflows! Most notably PyAFQ is available
as a reconstruction workflow. The default atlases included in QSIPrep have been
updated to include subcortical regions if they weren't already present in the
original atlas.

 * Add PyAFQ reconstruction workflows #398 Credit: @36000
 * Make sure all recon workflows respect omp_nthreads #368
 * Add DKI derivatives #371
 * Properly transform 4D CNR images from Eddy #393
 * Update amico to version 22.4.1 #394
 * Fix concatenation bug #403 credit: @cookpa
 * Prevent divide by zero error #405 credit: @cookpa
 * Critical Fix, use correct transform to get atlases into T1w space #417
 * Add resampled atlases back into derivatives #418
 * Add connectome2tck exemplar streamlines for mrtrix connectivity workflows #420
 * Update the atlases to include subcortical regions #426 [details here](https://github.com/PennLINC/qsiprep-atlas/blob/main/QSIRecon%20atlases.ipynb)

0.15.2 (March 3, 2022) DEPRECATED
==================================

**WARNING** There is an bug in the connectome pipelines that makes the connectivity
matrices unreliable. Do not use this version for connectome estimation.

Due to persistent difficulties with crashing ODF plots in the reconstruction workflows,
there is now a `--skip-odf-reports` option that will disable the ODF and peak plots
in the html reports. This should only be used once you've run some test workflows
with the reports still enabled, so you know that your ODFs are correctly oriented.

 * Make ODF Plots optional (#364)
 * Bugfix: ABCD gradient data for extrapolation (#363)
 * Adds `dipy_dki` reconstruction workflow (#366)


0.15.1 (February 28, 2022) DEPRECATED
======================================

**WARNING** There is an bug in the connectome pipelines that makes the connectivity
matrices unreliable. Do not use this version for connectome estimation.

A lot of changes in QSIPrep. The big-picture changes are

 1. The build system was redone so a multistage build is used in a
    different repository (https://github.com/PennLINC/qsiprep_build).
    The container should be about half as big as the last release.
 2. The way anatomical masks are handled in reconstruction workflows
    has been changed so that FreeSurfer data can be incorporated.
 3. FAST-based anatomically-constrained tractography is now deprecated in
    QSIPrep. If you're going to use anatomical constraints, they should be
    very accurate. The hybrid surface-volume segmentation (HSVS) is
    *amazing* and should be considered the default way to use the
    MRtrix3/3Tissue workflows. The
    [documentation](https://qsiprep.readthedocs.io/en/latest/reconstruction.html)
    describes the new built-in workflow names.
 4. The reconstruction workflows have been totally refactored. This won't
    affect the outputs of the reconstruction workflows, but will affect
    anyone who is using intermediate files from the working directory.
    The working directories no longer have those unfortunate `..`'s in
    their names.
 5. FSL is updated to 6.0.5.1!

Since these are a lot of changes, please be vigilant and check your results!
The QSIPrep preprocessing workflows have not changed with this release, but
the dependencies have been upgraded for almost everything.

 * Update FSL to 6.0.5.1 (#334)
 * Move ODF plotting to a cli tool so xvfb is handled more robustly (#357)
 * Better FreeSurfer license documentation (#355)
 * Edited libQt5Core.so.5 so it's loadable in singularity on CentOS (#336)
 * Fixed typo in patch2self (#338)
 * Inaccurate bids-validator errors were removed (#340)
 * Bug in `--recon-input` fixed #286
 * Correct streamline count is reported in the mrtrix connectivity matrices (#330)
 * Add option to ingress freesurfer data (#287)
 * Add Nature Methods citation to dataset_description.json
 * Refactor build system (#341)
 * SHORELine bugfixes (#301)
 * Bugfix: handle cases where there is only one b=0 (#279)

0.14.3 (September 16, 2021)
===========================
Change in behavior in Patch2Self:

 * Updates Patch2Self with optimal parameters (use OLS instead of ridge)

0.14.2 (July 11, 2021)
======================
Bugfixes and documentation

 * Updates documentation for containers (#270)
 * Fixes a bug when reading fieldmap metadata from datalad inputs (#271)
 * Change incorrect option in the documentation (#272)

0.14.0 (July 2, 2021)
=====================
Adds a new reconstruction workflow for the NODDI model.

 * Adds NODDI reconstruction workflow (#257). Thanks @cookpa!
 * Fixes issue with unequal aspect ratios in q-space plots (#266)

0.13.1 (June 14, 2021)
======================

 * Adds a flag for a BIDS filter file #256
 * Fixes a bug where --dwi-only is selected along with --intramodal-template

0.13.0 (May 5, 2021)
====================
Many bugfixes

 * Fix bug that produced flipped scalar images (#251)
 * Added a default working directory to prevent uninterpretable error message (#250)
 * Fix a bug in the `dipy_3dshore` reconstruction workflow (#249)
 * Remove hardlinking from DSI Studio interfaces (#214)
 * Add an option to use a BIDS database directory (#247)
 * Fix bug in interactive reports for HCP-style acquisitions (#238)
 * Update defaults for `Patch2Self` (#230, #239)
 * Remove cmake installer from docker image after compiling ANTS (#229)

0.13.0RC1 (January 19, 2021)
============================
This version introduces major changes to the TOPUP/eddy workflow. Feedback would be greatly
appreciated!

 * Added new algorithm for selecting b=0 images for distortion corretion (#202)
 * Added the Patch2Self denoising method (#203, credit to @ShreyasFadnavis)
 * Documentation has been expanded significantly (#212)
 * Boilerplate for DWI preprocessing is greatly expanded (#200)


0.12.2 (November 7, 2020)
=========================
Adds options for processing infant dMRI data. Also enables running without a T1w
image.

 * Adds ``--dwi-only`` and ``--infant`` options to QSIPrep. (#177)


0.11.0 (August 12, 2020)
========================
NEW: Workflow defaults have changed. T1w-based spatial normalization is done by
default (disabled by ``--skip-t1-based-spatial-normalization``) and dwi scans
are merged before motion correction by default (disabled by ``--separate-all-dwis``).

 * Deprecate some commandline arguments, change defaults (#168)
 * Fix typo in workflow names (#162)
 * Fix bug from 0.10.0 where ODFs were not appearing in plots (#160)


0.10.0 (August 4, 2020)
=======================

 * Adds support for oblique acquisitions (#156)


0.9.0beta1 (June 17, 2020)
==========================

 * Adds support for HCP lifespan sequences
 * Introduces --distortion-group-merge option for combining paired scans

0.8.0 (February 12, 2020)
=========================

 * Removes oblique angles from T1w headers to fix N4 bug (#103)

0.7.2 (February 4, 2020)
========================

 * Fixed a bug in b=0 masking when images have high signal intensity in ventricles (#99)

0.7.1 (January 29, 2020)
========================

 * Image QC summary data is produced for each output (#95)
 * Update DSI Studio (#88)
 * Update ANTs (#80)
 * Include workflows for ss3t (#82)
 * Add some boilerplate to the FSL workflow (#38)
 * Reduce the number of calls to N4 (#74, #89)
 * Add CUDA capability in the containers (#75)
 * Add mrdegibbs and accompanying reports (#58)
 * Fix reports graphics (#64)
 * Rework the DWI grouping algorithm (#92)

0.6.7 (January 9 2020)
======================
This release adds some rather big updates to QSIPrep.
 * FSL is updated to version 6.0.3
 * CUDA v9.1 support is added to the image (works with GPUS in Docker and Singularity)
 * A new robust b=0 masking algorith is introduced.

0.6.5 (Nov 21, 2019)
====================
 * Improved handling of Freesurfer path (#50)
 * Better logic in commandline argument checking (#50, #62)
 * More robust brain masking for b=0 reference images (#73)
 * Bugfix for reverse phase encoding directon dwi series (#68)
 * Bugfix for warping eddy's CNR output (#72)

0.6.4, 0.6.4-1 (Nov 11, 2019)
==============================
 * IMPORTANT: commandline call changed to use official BIDS App
 * eddy will use multiple cores if available
 * Fixed bug in sentry interaction


0.6.2, 0.6.3RC1, 0.6.3RC2 (October 27, 2019)
============================================

 * Bugfix: masking was not working on eddy.
 * Bugfix: static versioning was not workign in the container.
 * New graphics in the documentation.
 * Use BSpline Interpolation if --output-resolution is higher than the input resolution.


0.6.0RC1, 0.6.2 (October 13, 2019)
==================================

An issue was discovered in how voxel orientation interacts with TOPUP/eddy and outside
fieldmaps. Unless everything is in LAS+ prior to going into TOPUP/eddy, the warps are
incorrectly applied at the end of eddy. This resulted in fieldmap unwarping reports that
looked good but a final output that is bizarrely warped. Additionally, GRE fieldmaps would
result in outputs being under-unwarped. To fix all of these, TOPUP (if PEPOLAR fieldmaps are
being used) and eddy occur in LAS+, then their outputs are converted to LPS+ for GRE fieldmaps,
SyN. The rest of the pipeline happens in LPS+, like the SHORELine version.

 * Update installation method to match fMRIPrep
 * Add CI tests for reconstruction workflows
 * Make the ``--sloppy`` option affect the reconstruction workflows
 * Fixes bug in 3dSHORE reconstruction (incorrect scaling)
 * CRITICAL bug fix: convert everything to LAS+ if eddy is going to be used
 * Added built-in reconstruction workflows
 * Added Brainnetome, AICHA and the remaining Schaefer atlases


0.5.1, 0.5.1a, 0.5.2 (September 10, 2019)
==========================================

 * Address issues in Nipype causing random crashes


0.5.0 (August 11, 2019)
=======================

 * Use antsMultiVariateTemplateConstruction2.sh to make a b=0 template across scan groups
 * Control the number of template iterations and deformation model with
   ``--intramodal_template_iters`` and ``--intramodal_template_transform``.

0.4.6 (July 23, 2019)
=====================

 * More documentation updates
 * MSD calculated for MAPMRI

0.4.5 (July 22, 2019)
=====================

 * Scalar outputs from MAPMRI

0.4.4 (July 19, 2019)
======================

 * Default eddy configuation changed to not use CUDA by default.
 * Valerie added content to documentation

0.4.3 (July 18, 2019)
=====================

FSL tools are used to match SHORELine motion parameters to those from eddy.

 * Fieldcoefs are calculated from PEPOLAR and GRE fieldmaps and sent to TOPUP
 * Motion estimates from SHORELine match eddy

0.4.0 (June 7, 2019)
====================

Add workflows for eddy and TOPUP.

  * Adds eddy tests on CircleCI.
