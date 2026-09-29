.. include:: ../links.rst

.. _hmc_tortoise:

#################
TORTOISE DIFFPREP
#################

:func:`qsiprep.workflows.dwi.diffprep.init_diffprep_hmc_wf`

``--hmc-method tortoise`` runs DIFFPREP from TORTOISE v4
:footcite:p:`tortoisev4`. DIFFPREP fits a signal model over arbitrary
q-space, so it does not need shells: it is the method for Cartesian DSI,
compressed-sensing DSI and other non-shelled schemes, and it works on shelled
data too. It corrects rigid head motion together with eddy currents, using a
24-parameter quadratic transform by default.

What runs
=========

1. The concatenated series is converted to TORTOISE's format.
2. If the unit contains more than one phase encoding direction, the series
   is split by distortion group, because one DIFFPREP run models one phase
   axis for the whole file. DIFFPREP runs once per group.
3. DIFFPREP fits a SHORE/MAPMRI model, iteratively registers each volume to
   the model's prediction, and writes the corrected volumes with the
   rotated b-matrix.
4. Susceptibility distortion is corrected with TORTOISE's own tools where
   they apply (below), and the groups are recombined.

Like ``eddy``, DIFFPREP writes its own corrected volumes, so the output is
interpolated twice. The eddy-current Jacobian modulation of the DIFFPREP
transform is applied by *QSIPrep* at the final resampling (see
:ref:`jacobian_methods`), and the per-volume eddy parameters go to the
confounds file. The per-slice model fit is shown in the carpet plot.

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep_docs import example_unit, configure, AP
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    configure(workflow__hmc_method='tortoise', workflow__sdc_method='drbuddi')
    wf = init_diffprep_hmc_wf(example_unit('pepolar'), source_file=AP, t2w_sdc=False)

Distortion correction cases
===========================

**Reverse phase encoding (DRBUDDI).** After DIFFPREP has corrected each
phase encoding direction, DRBUDDI :footcite:p:`drbuddi` estimates the field
from the two directions and applies it to every volume. It is described in
:ref:`drbuddi_tool`; a T2w, if present, is included in its registration. For
non-shelled data, DRBUDDI's tensor fit for its registration target may be
poor; set ``"drbuddi_synth_shell_bval"`` (for example ``1000``) in
``--diffprep-config`` to have TORTOISE synthesize a tensor-fittable shell
per phase encoding direction from the model.

**GRE fieldmap.** The fieldmap is applied to DIFFPREP's output by *QSIPrep*'s
fieldmap workflow (see :ref:`sdc_gre`).

**T2Wreg.** With no fieldmap and ``--sdc-anat-reference t2w`` (or ``auto``
with a T2w), DIFFPREP's own T2Wreg stage nonlinearly registers the b=0 to the
T2w in the same TORTOISE run, and the correction is baked into its output.
With ``--sdc-anat-reference synb0``, the target is the synthetic b=0 instead.
When the anatomical reference is forced over a GRE fieldmap (``--force
sdc-anat-reference``), T2Wreg starts from the GRE-derived field
(:ref:`sdc_gre_init`).

**SyN.** ``--sdc-anat-reference invt1w`` is applied to DIFFPREP's output, as
with ``eddy``.

.. _configure_diffprep:

Configuring DIFFPREP
====================

``--diffprep-config`` is a JSON file; the default is `diffprep_params.json
<https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/diffprep_params.json>`__.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``correction_mode``
     - ``"motion"`` (rigid head motion only), ``"quadratic"`` (the default,
       rigid motion plus 24-parameter quadratic eddy currents) or
       ``"cubic"``.
   * - ``drbuddi_synth_shell_bval``
     - b-value of a shell to synthesize for DRBUDDI's registration target
       (``null``, the default, is off). ``drbuddi_synth_shell_ndirs`` sets
       its number of directions (default 30).
   * - ``b0_id``, ``is_human_brain``, ``rot_eddy_center``
     - Passed through to DIFFPREP.
   * - ``extra_args``
     - A list of extra command-line arguments for DIFFPREP.
   * - ``use_cuda``
     - Run the GPU build. ``--gpu`` overrides it.

``--tortoise-gpu-cpu-ratio`` tunes how DIFFPREP shares volumes between the
GPU and the CPU threads when both are used.

Limitations
===========

* GE gradient coefficient files are not accepted (see
  :ref:`gradient_files`).
* ``--sdc-method topup`` and ``topup+drbuddi`` are not available; TOPUP is an
  ``eddy`` tool.


**********
References
**********

.. footbibliography::
