"""qsiprep's grouping/compiler layer, now the standalone ``qsiplan`` package.

This module re-exports qsiplan's public API so qsiprep-internal imports
keep working, and holds :func:`method_selection_from_config` - the single
bridge from :mod:`qsiprep.config` to a :class:`~qsiplan.MethodSelection`,
which belongs on the qsiprep side of the boundary.
"""

from qsiplan import *  # noqa: F403
from qsiplan import __all__ as _qsiplan_all
from qsiplan import build_dwi_grouping, selection_for_config  # noqa: F401
from qsiplan.methods import MethodSelection

__all__ = [*_qsiplan_all, 'method_selection_from_config']


def method_selection_from_config() -> MethodSelection:
    """The selection the current :mod:`qsiprep.config` asks for.

    The single config-to-selection conversion point. Falls back to the legacy
    keys so a config object loaded from an older file still resolves.
    """
    from qsiprep import config

    hmc = config.workflow.hmc_method or config.workflow.hmc_model
    if config.workflow.hmc_method == 'shoreline' and config.workflow.shoreline_model:
        # 'shoreline' alone defaults to 3dshore; honor the configured model.
        hmc = config.workflow.shoreline_model
    return selection_for_config(
        hmc,
        config.workflow.sdc_method or config.workflow.pepolar_method,
        use_syn=bool(config.workflow.use_syn_sdc),
        force_t2wreg='t2wreg' in (config.workflow.force or ()),
    )
