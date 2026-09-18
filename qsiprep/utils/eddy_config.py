"""Questions about the effective FSL ``eddy`` configuration.

``eddy``'s resampling method decides whether it applied its own Jacobian
modulation for eddy-current and TOPUP susceptibility distortions. QSIPrep ships
``jac``, but ``--eddy-config`` lets a user supply any JSON, so this is a default
and not an invariant. Two places need the answer -- ``init_fsl_hmc_wf`` for the
warning and the methods boilerplate, and ``init_dwi_trans_wf`` for the
weighting decision -- so it is resolved once here, from the already-loaded
dict, rather than parsed twice.
"""

#: What ``eddy`` does when ``method`` is absent from the config. Nipype's trait
#: maps ``method`` to ``--resamp``, whose own default is ``jac``.
DEFAULT_RESAMPLING_METHOD = 'jac'


def effective_eddy_resampling_method(eddy_args):
    """The ``--resamp`` value ``eddy`` will actually run with."""
    return eddy_args.get('method') or DEFAULT_RESAMPLING_METHOD


def eddy_modulates_distortion(eddy_args):
    """Whether ``eddy`` Jacobian-modulates eddy-current and susceptibility.

    True for ``--resamp=jac``. False for ``lsr``, which is a different
    resampling model; the claim is deliberately narrow -- it says only that the
    Jacobian modulation this feature is about did not happen, not anything
    broader about least-squares restoration's intensity semantics.
    """
    return effective_eddy_resampling_method(eddy_args) == 'jac'
