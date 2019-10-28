0.6.2, 0.6.3RC1, 0.6.3RC2 (October 27, 2019)
============================================

 * Bugfix: masking was not working on eddy.
 * Bugfix: static versioning was not workign in the container.
 * New graphics in the documentation.


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
