"""The config-to-MethodSelection bridge (qsiprep side of the qsiplan boundary)."""

import pytest
from qsiplan.methods import HmcMethod, SdcTool

_KEYS = ('hmc_method', 'shoreline_model', 'sdc_method', 'sdc_anat_reference', 'force')


@pytest.fixture
def restore_config():
    from qsiprep import config

    saved = {key: getattr(config.workflow, key) for key in _KEYS}
    yield config
    for key, value in saved.items():
        setattr(config.workflow, key, value)


def test_method_selection_from_config_reads_the_method_axes(restore_config):
    from qsiprep.utils.plan import method_selection_from_config

    config = restore_config
    config.workflow.hmc_method = 'shoreline'
    config.workflow.shoreline_model = 'tensor'
    config.workflow.sdc_method = 'drbuddi'
    config.workflow.sdc_anat_reference = 'none'
    config.workflow.force = None
    selection = method_selection_from_config()
    assert selection.hmc is HmcMethod.SHORELINE
    assert selection.shoreline_model == 'tensor'
    assert selection.pepolar_tools == (SdcTool.DRBUDDI,)
