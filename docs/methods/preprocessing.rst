.. include:: ../links.rst

.. _preprocessing_methods:

########################
Per-series preprocessing
########################

:func:`qsiprep.workflows.dwi.pre_hmc.init_dwi_pre_hmc_wf`

Each DWI series of an output is processed on its own before the series are
concatenated and handed to the head motion backend. Denoising has
assumptions about its input, so the order of these steps is fixed. A series
that belongs to several outputs (a ``MultipartID`` list) is denoised once;
every output that includes it reuses that result.

1. **Denoising** on the raw series: MP-PCA with ``dwidenoise``
   :footcite:p:`dwidenoise1`, its successor ``dwidenoise2``
   :footcite:p:`dwidenoise2`, or ``patch2self`` :footcite:p:`patch2self`
   (``--denoise-method``). With ``part-phase`` data, ``dwidenoise`` and
   ``dwidenoise2`` denoise the complex signal :footcite:p:`cordero2019complex`,
   which removes the noise floor.
2. **Gibbs unringing**, off by default: ``mrdegibbs`` :footcite:p:`mrdegibbs`
   for full Fourier acquisitions, or TORTOISE's ``rpg`` :footcite:p:`pfgibbs`
   for partial Fourier. Complex-valued ``mrdegibbs`` needs
   ``--mrtrix-version dev``; otherwise the data are reduced to magnitude
   after denoising.
3. **Concatenation** of the series, with their gradient tables, into one
   file. The b=0 intensities of the series are rescaled to a common level
   (``--no-b0-harmonization`` skips this).
4. **Quality measures** on the raw concatenated data, computed with DSI
   Studio :footcite:p:`yeh2019` and written to the ``raw_`` columns of the
   image QC table.

B1 bias field correction is not part of this stage. It runs after
resampling, with N4 :footcite:p:`n4` on the b=0 images and the resulting
field applied to the whole series (``--dwi-biascorrect``).

The residuals of denoising and unringing are shown in the visual report, so
you can check that they contain noise and ringing rather than anatomy.

.. raw:: html

    <object type="image/svg+xml" data="../_static/denoising.svg" style="width: 100%; aspect-ratio: 1108 / 939"></object>

The report's denoising figure: a low-b volume before and after denoising,
and what was removed, alternating while the pointer is over it.

.. raw:: html

    <object type="image/svg+xml" data="../_static/unringing.svg" style="width: 100%; aspect-ratio: 1108 / 939"></object>

The unringing figure works the same way: the unrung low-b and high-b
volumes flicker against the ringing that ``mrdegibbs`` or ``rpg``
estimated and removed, which should look like edges, not anatomy.

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from qsiprep_docs import example_unit, AP
    from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf

    wf = init_dwi_pre_hmc_wf(
        example_unit('single'),
        orientation='LPS',
        source_file=AP,
        do_biascorr=True,
    )


.. _gradwarp_methods:

********************************
Gradient nonlinearity correction
********************************

:func:`qsiprep.workflows.dwi.gradwarp.init_gradwarp_wf`

With ``--gradient-file``, a displacement field is generated once per DWI
run, directly from the raw DWI grid, with TORTOISE's
``CreateNonlinearityDisplacementMap``. It is never applied as a resampling
step of its own. It is folded into the transform that resamples the output
at the end of the pipeline, in the order head motion, gradient
nonlinearity, susceptibility distortion, coregistration.

Head motion estimation never sees the field; motion is estimated on the raw
grid, as TORTOISE does. Susceptibility distortion is estimated on
gradient-corrected b=0 and FA images whenever the estimated warp is applied
after the gradient correction, which is the case for DRBUDDI, T2Wreg and SyN
on every backend. A GRE fieldmap is acquired with the same gradients as the
DWI, so its warp is estimated on the raw b=0 and then composed with the
gradient field and its inverse. The exception is the field ``eddy`` applies
itself (from TOPUP or a GRE fieldmap): ``eddy`` resamples the raw data and
applies the field internally, so the field is estimated on raw b=0 images and
the gradient correction is applied afterwards, at the final resampling.

The b=0 image that coregistration is estimated from is gradient-corrected on
every backend, because the coregistration is applied after the gradient
correction.

The voxelwise gradient deviation map (see :ref:`outputs`) is computed
separately, with ``CreateGradientNonlinearityBMatrix``, and does not depend
on whether spatial correction was applied.

Known limitations:

* GE coefficient files are not accepted for spatial correction (see
  :ref:`gradient_files`).
* No gradient deviation map is written for outputs assembled by
  ``--distortion-group-merge``; the spatial correction is still applied.
* The deviation map is oriented by a rigid registration TORTOISE estimates
  itself, not by the coregistration *QSIPrep* applied. The two are close but
  not identical, and this is recorded in the sidecar.


**********
References
**********

.. footbibliography::
