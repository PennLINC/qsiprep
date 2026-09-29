.. include:: ../links.rst

.. _hmc_eddy:

##########
eddy (FSL)
##########

:func:`qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf`

``--hmc-method eddy`` (the default) runs FSL's ``eddy``
:footcite:p:`anderssoneddy`, which corrects head motion, eddy-current
distortion and, when it is given a field, susceptibility distortion in one
model. It requires a shelled sampling scheme.

What runs
=========

1. The concatenated series is reoriented to LAS+, which the FSL tools expect.
2. If there is reverse phase-encoded data, TOPUP :footcite:p:`topup`
   estimates the susceptibility field from b=0 images (see below).
3. ``eddy`` estimates and corrects head motion and eddy currents, and applies
   the TOPUP field or a GRE fieldmap in the same resampling. Its outlier
   replacement :footcite:p:`eddyrepol` is on by default.
4. The corrected series is converted back to LPS+. Any remaining distortion
   correction (DRBUDDI, SyN) is applied to these outputs.

``eddy`` writes its own corrected volumes, so the output data are
interpolated twice: once by ``eddy`` and once by *QSIPrep*'s final resampling
into ``ACPC`` space. ``eddy`` also applies Jacobian intensity modulation to
its own corrections whenever its resampling method is ``jac`` (the default);
see :ref:`jacobian_methods`.

The motion parameters, the eddy-current coefficients and the outlier
counts per slice go to the confounds file and the carpet plot. The
contrast-to-noise ratio maps ``eddy`` computes per shell are written as
``stat-cnr`` derivatives.

Distortion correction cases
===========================

**No fieldmap.** Only head motion and eddy currents are corrected.

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep_docs import example_unit, AP
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    wf = init_fsl_hmc_wf(example_unit('single'), source_file=AP, t2w_sdc=False)

**Reverse phase encoding (TOPUP).** b=0 images are selected from each
distortion group of the correction unit, whether they come from the DWI
series or from ``epi`` fieldmaps; images native to the series being
corrected are preferred over borrowed ones. TOPUP estimates the field from
them, and ``eddy`` applies it with ``--topup``. With
``--sdc-method topup+drbuddi``, DRBUDDI then refines ``eddy``'s output; this
needs a single matched pair of reverse phase-encoded DWI *series*, since with
a lone reverse b=0 the output is already unwarped and a second pass would
correct it twice. With ``--sdc-method drbuddi``, TOPUP is skipped and DRBUDDI
corrects the ``eddy`` output on its own.

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep_docs import example_unit, AP
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    wf = init_fsl_hmc_wf(example_unit('pepolar'), source_file=AP, t2w_sdc=False)

**GRE fieldmap.** The fieldmap is converted to Hz, rigidly aligned to
``eddy``'s first volume, and passed to ``eddy`` with ``--field``, which
applies it inside its model the way it applies a TOPUP field. Setting
``"estimate_move_by_susceptibility"`` in ``--eddy-config`` additionally lets
``eddy`` estimate how the field changes with head position
:footcite:p:`eddysus`. Before 26.1 the fieldmap was applied to ``eddy``'s
outputs instead; ``--force gre-sdc-after-eddy`` restores that, for
comparison, and is deprecated.

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from qsiprep_docs import example_unit, AP
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    wf = init_fsl_hmc_wf(example_unit('phasediff'), source_file=AP, t2w_sdc=False)

**SynB0.** With ``--sdc-anat-reference synb0``, the synthetic distortion-free
b=0 (see :ref:`sdc_synb0`) joins the TOPUP inputs as a volume with zero
readout time, and ``eddy`` uses the resulting field like any other. This
needs ``topup`` in ``--sdc-method``.

**SyN.** With ``--sdc-anat-reference invt1w``, the SyN correction is
estimated and applied on ``eddy``'s outputs.

``--sdc-anat-reference t2w`` has no effect with ``eddy``; T2Wreg is a
TORTOISE tool.

.. _configure_eddy:

Configuring eddy
================

``eddy`` has many options. Rather than exposing them on the command line,
*QSIPrep* reads them from a JSON file given with ``--eddy-config``. The
default file is `eddy_params.json
<https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/eddy_params.json>`__;
the keys are ``eddy``'s own option names. The file is validated when the
command line is parsed. ``--gpu eddy`` runs ``eddy_cuda`` and overrides the
file's ``use_cuda``.


**********
References
**********

.. footbibliography::
