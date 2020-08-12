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
