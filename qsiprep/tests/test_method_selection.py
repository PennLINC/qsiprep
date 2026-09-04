"""The config-to-MethodSelection bridge (qsiprep side of the qsiplan boundary)."""

from qsiplan.methods import HmcMethod, SdcTool


def test_method_selection_from_config_reads_new_and_legacy_keys():
    from qsiprep import config
    from qsiprep.utils.plan import method_selection_from_config

    keys = (
        'hmc_method',
        'shoreline_model',
        'sdc_method',
        'hmc_model',
        'pepolar_method',
        'sdc_anat_reference',
        'force',
    )
    saved = {key: getattr(config.workflow, key) for key in keys}
    try:
        config.workflow.hmc_method = 'shoreline'
        config.workflow.shoreline_model = 'tensor'
        config.workflow.sdc_method = 'drbuddi'
        config.workflow.hmc_model = 'tensor'
        config.workflow.pepolar_method = 'TOPUP'
        config.workflow.sdc_anat_reference = 'none'
        config.workflow.force = None
        selection = method_selection_from_config()
        assert selection.hmc is HmcMethod.SHORELINE
        assert selection.shoreline_model == 'tensor'
        assert selection.pepolar_tools == (SdcTool.DRBUDDI,)

        # A config loaded from an older file has only the legacy keys.
        config.workflow.hmc_method = None
        config.workflow.shoreline_model = None
        config.workflow.sdc_method = None
        config.workflow.hmc_model = '3dSHORE'
        config.workflow.pepolar_method = 'DRBUDDI'
        selection = method_selection_from_config()
        assert selection.hmc is HmcMethod.SHORELINE
        assert selection.shoreline_model == '3dshore'
        assert selection.pepolar_tools == (SdcTool.DRBUDDI,)
    finally:
        for key, value in saved.items():
            setattr(config.workflow, key, value)
