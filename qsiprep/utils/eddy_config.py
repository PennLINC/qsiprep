"""Questions about the effective FSL ``eddy`` configuration.

``eddy``'s resampling method decides whether it applied its own Jacobian
modulation for eddy-current and TOPUP susceptibility distortions. QSIPrep ships
``jac``, but ``--eddy-config`` lets a user supply any JSON, so this is a default
and not an invariant. Two places need the answer -- ``init_fsl_hmc_wf`` for the
warning and the methods boilerplate, and ``init_dwi_trans_wf`` for the
weighting decision -- so it is resolved once here, from the already-loaded
dict, rather than parsed twice. The file itself is loaded once too, by
``load_eddy_args``, for the same reason.
"""

import json

from .. import config
from ..data import load as load_data

#: What ``eddy`` does when ``method`` is absent from the config. Nipype's trait
#: maps ``method`` to ``--resamp``, whose own default is ``jac``.
DEFAULT_RESAMPLING_METHOD = 'jac'


def effective_eddy_resampling_method(eddy_args):
    """Return the ``--resamp`` value ``eddy`` will actually run with."""
    return eddy_args.get('method') or DEFAULT_RESAMPLING_METHOD


def eddy_modulates_distortion(eddy_args):
    """Check whether ``eddy`` Jacobian-modulates eddy-current and susceptibility.

    True for ``--resamp=jac``. False for ``lsr``, which is a different
    resampling model; the claim is deliberately narrow -- it says only that the
    Jacobian modulation this feature is about did not happen, not anything
    broader about least-squares restoration's intensity semantics.
    """
    return effective_eddy_resampling_method(eddy_args) == 'jac'


def eddy_applies_gre(unit):
    """Check whether ``eddy`` applies this unit's GRE fieldmap itself (``--field``).

    That is, whether ``eddy`` applies the GRE fieldmap the way it applies TOPUP's
    field, rather than the warp being applied after ``eddy`` (the deprecated
    ``--force gre-sdc-after-eddy``).
    """
    return (
        unit.is_gre
        and unit.run.hmc_stage.tool == 'eddy'
        and 'gre-sdc-after-eddy' not in (config.workflow.force or [])
    )


def load_eddy_args():
    """Load the effective ``--eddy-config`` JSON, or the shipped default.

    Shared by ``init_fsl_hmc_wf`` (which needs the dict to build ``eddy``'s
    node and to warn/describe boilerplate) and
    :func:`qsiprep.utils.jacobian_provenance.jacobian_provenance_for` (which needs only
    ``eddy_modulates_distortion`` of it) so the two never parse the file
    independently and risk disagreeing about what it says.
    """
    if config.workflow.eddy_config is None:
        eddy_cfg_file = str(load_data('eddy_params.json'))
    else:
        eddy_cfg_file = config.workflow.eddy_config
    with open(eddy_cfg_file) as f:
        return json.load(f)
