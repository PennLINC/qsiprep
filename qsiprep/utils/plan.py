"""QSIPrep configuration adapters for QSIPlan."""

from qsiplan.methods import MethodSelection, selection_for_config


def method_selection_from_config() -> MethodSelection:
    """Resolve the current QSIPrep config to a QSIPlan method selection.

    Fall back to the deprecated keys so configuration files written by older
    QSIPrep versions remain loadable.
    """
    from qsiprep import config

    hmc = config.workflow.hmc_method or config.workflow.hmc_model
    if config.workflow.hmc_method == 'shoreline' and config.workflow.shoreline_model:
        hmc = config.workflow.shoreline_model
    return selection_for_config(
        hmc,
        config.workflow.sdc_method or config.workflow.pepolar_method,
    )
