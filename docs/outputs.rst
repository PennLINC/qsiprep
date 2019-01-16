

.. _outputs:

-------------------
Outputs of qsiprep
-------------------

qsiprep generates three broad classes of outcomes:

  1. **Visual QA (quality assessment) reports**:
     one :abbr:`HTML (hypertext markup language)` per subject,
     depicting images that provide a sanity check for each step of the pipeline.

  2. **Pre-processed imaging data** such as anatomical segmentations, realigned and resampled
     diffusion weighted images and the corresponding corrected gradient files in FSL and MRTrix
     format.

  3. **Additional data for subsequent analysis**, for instance the transformations
     between different spaces or the estimated head motion and model fit quality calculated
     during model-based head motion correction.


Visual Reports
--------------

qsiprep outputs summary reports, written to ``<output dir>/qsiprep/sub-<subject_label>.html``.
These reports provide a quick way to make visual inspection of the results easy.  One useful
graphic is the animation of the q-space sampling scheme before and after the pipeline. Here is
a sampling scheme from a DSI scan:

.. figure:: _static/sampling_scheme.gif
    :scale: 75%

    A Q5 DSI sampling scheme before (left) and after (right) preprocessing. This is useful to
    confirm that the gradients have indeed been rotated and that head motion correction has not
    disrupted the scheme extensively.


Preprocessed data (qsiprep *derivatives*)
------------------------------------------

There are additional files, called "Derivatives", written to
``<output dir>/qsiprep/sub-<subject_label>/``.

Derivatives related to T1w files are nearly identical to those produced by FMRIPREP and
can be found in the ``anat`` subfolder:

- ``*T1w_brainmask.nii.gz`` Brain mask derived using ANTs' ``antsBrainExtraction.sh``.
- ``*T1w_class-CSF_probtissue.nii.gz``
- ``*T1w_class-GM_probtissue.nii.gz``
- ``*T1w_class-WM_probtissue.nii.gz`` tissue-probability maps.
- ``*T1w_dtissue.nii.gz`` Tissue class map derived using FAST.
- ``*T1w_preproc.nii.gz`` Bias field corrected T1w file, using ANTS' N4BiasFieldCorrection
- ``*T1w_space-MNI152NLin2009cAsym_brainmask.nii.gz`` Same as ``_brainmask`` above, but in MNI space.
- ``*T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-WM_probtissue.nii.gz`` Probability tissue maps, transformed into MNI space
- ``*T1w_space-MNI152NLin2009cAsym_dtissue.nii.gz`` Same as ``_dtissue`` above, but in MNI space
- ``*T1w_space-MNI152NLin2009cAsym_preproc.nii.gz`` Same as ``_preproc`` above, but in MNI space
- ``*T1w_space-MNI152NLin2009cAsym_target-T1w_warp.h5`` Composite (warp and affine) transform to map from MNI to T1 space
- ``*T1w_target-MNI152NLin2009cAsym_warp.h5`` Composite (warp and affine) transform to transform T1w into MNI space

.. Note:
  These are in LPS+ orientation, so are not identical to FMRIPREP's anatomical outputs

Derivatives related to diffusion images are in the ``dwi`` subfolder.

- ``*_confounds.tsv`` A tab-separated value file with one column per calculated confound and one row per timepoint/volume

Volumetric output spaces include ``T1w`` and ``MNI152NLin2009cAsym`` (default).

- ``*dwiref.nii.gz`` The b0 template
- ``*desc-brain_mask.nii.gz`` The generous brain mask that should be reduced probably
- ``*desc-preproc_dwi.nii.gz`` Resampled DWI series including all b0 images.
- ``*desc-preproc_dwi.bval``, ``*desc-preproc_dwi.bvec`` FSL-style bvals and bvecs files.
  *These will be incorrectly interpreted by MRTrix, but will work with DSI Studio.* Use the
  ``.b`` file for MRTrix.
- ``desc-preproc_dwi.b`` The gradient table to import data into MRTrix. This and the
  ``_dwi.nii.gz`` can be converted directly to a ``.mif`` file using the ``mrconvert -grad _dwi.b``
  command.
- ``*b0series.nii.gz`` The b0 images from the series in a 4d image. Useful to see how much the
  images are impacted by Eddy currents.
- ``*bvecs.nii.gz`` Each voxel contains a gradient table that has been adjusted for local
  rotations introduced by spatial warping.


Confounds
---------

See implementation on :mod:`~qsiprep.workflows.dwi.confounds.init_dwi_confs_wf`.


For each DWI processed by qsiprep, a
``<output_folder>/qsiprep/sub-<sub_id>/func/sub-<sub_id>_task-<task_id>_run-<run_id>_confounds.tsv``
file will be generated.
These are :abbr:`TSV (tab-separated values)` tables, which look like the example below: ::

framewise_displacement	trans_x	trans_y	trans_z	rot_x	rot_y	rot_z	hmc_r2	hmc_xcorr	original_file	grad_x	grad_y	grad_z	bval

n/a	-0.705	-0.002	0.133	0.119	0.350	0.711	0.941	0.943	sub-abcd_dwi.nii.gz	0.000	0.000	0.000	0.000
16.343	-0.711	-0.075	0.220	0.067	0.405	0.495	0.945	0.946	sub-abcd_dwi.nii.gz	0.000	0.000	0.000	0.000
35.173	-0.672	-0.415	0.725	0.004	0.468	1.055	0.756	0.766	sub-abcd_dwi.nii.gz	-0.356	0.656	0.665	3000.000
45.131	0.021	-0.498	1.046	0.403	0.331	1.400	0.771	0.778	sub-abcd_dwi.nii.gz	-0.935	0.272	0.229	3000.000
37.506	-0.184	0.117	0.723	0.305	0.138	0.964	0.895	0.896	sub-abcd_dwi.nii.gz	-0.187	-0.957	-0.223	2000.000
16.388	-0.447	0.020	0.847	0.217	0.129	0.743	0.792	0.800	sub-abcd_dwi.nii.gz	-0.111	-0.119	0.987	3000.000

The motion parameters come from the model-based head motion estimation workflow. The ``hmc_r2`` and
``hmc_xcorr`` are whole-brain r^2 values and cross correlation scores (using the ANTs definition)
between the model-generated target image and the motion-corrected empirical image. The final
columns are not really confounds, but book-keeping information that reminds us which 4d DWI series
the image originally came from and what gradient direction (``grad_x``, ``grad_y``, ``grad_z``)
and gradient strength ``bval`` the image came from. This can be useful for tracking down
mostly-corrupted scans and can indicate if the head motion model isn't working on specific
gradient strengths or directions.

Confounds and "carpet"-plot on the visual reports
-------------------------------------------------

fMRI has been using a "carpet" visualization of the
:abbr:`BOLD (blood-oxygen level-dependant)` time-series (see [Power2016]_),
but this type of plot does not make sense for DWI data. Instead, we plot
the cross-correlation value between each raw slice and the HMC model signal
resampled into that slice.
This plot is included for each run within the corresponding visual report.
An example of these plots follows:


.. figure:: _static/sub-abcd_carpetplot.svg
    :scale: 100%

    Higher scores appear more yellow, while lower scores
    are more blue. Not all slices contain the same number of voxels,
    so the number of voxels in the slice is represented in the color bar
    to the left of the image plot. The more yellow the pixel, the more
    voxels are present in the slice. Purple pixels reflect slices with fewer
    brain voxels.


.. topic:: References

  .. [Power2016] Power JD, A simple but useful way to assess fMRI scan qualities.
     NeuroImage. 2016. doi: `10.1016/j.neuroimage.2016.08.009 <http://doi.org/10.1016/j.neuroimage.2016.08.009>`_
