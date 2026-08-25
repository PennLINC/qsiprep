"""The method axes, capability registries, and legacy-vocabulary mapping."""

import itertools

import pytest

from qsiprep.grouping.methods import (
    HMC_CAPABILITIES,
    SDC_CAPABILITIES,
    HmcMethod,
    MethodSelection,
    SdcTool,
    canonical_selection,
    selection_for_config,
)
from qsiprep.grouping.models import CorrectionMethod
from qsiprep.grouping.validation import BACKENDS

LEGACY_HMC_MODELS = ('eddy', 'tortoise', '3dSHORE', 'tensor', 'none')
LEGACY_PEPOLAR_METHODS = ('TOPUP', 'DRBUDDI', 'TOPUP+DRBUDDI')


@pytest.mark.parametrize(
    ('hmc_model', 'pepolar_method'),
    list(itertools.product(LEGACY_HMC_MODELS, LEGACY_PEPOLAR_METHODS)),
)
def test_legacy_backend_matches_the_original_truth_table(hmc_model, pepolar_method):
    """The historical backend_for_config rule, verbatim."""
    if hmc_model == 'eddy':
        expected = 'mixed' if 'DRBUDDI' in pepolar_method else 'fsl'
    else:
        expected = 'tortoise'
    assert selection_for_config(hmc_model, pepolar_method).legacy_backend == expected


@pytest.mark.parametrize(
    ('hmc_model', 'hmc', 'shoreline_model'),
    [
        ('eddy', HmcMethod.EDDY, None),
        ('tortoise', HmcMethod.TORTOISE, None),
        ('shoreline', HmcMethod.SHORELINE, '3dshore'),
        ('3dSHORE', HmcMethod.SHORELINE, '3dshore'),
        ('tensor', HmcMethod.SHORELINE, 'tensor'),
        ('none', HmcMethod.SHORELINE, 'none'),
    ],
)
def test_hmc_vocabulary_old_and_new(hmc_model, hmc, shoreline_model):
    selection = selection_for_config(hmc_model, 'auto')
    assert selection.hmc is hmc
    assert selection.shoreline_model == shoreline_model


@pytest.mark.parametrize('pepolar_method', [None, 'auto', 'AUTO'])
def test_auto_resolves_per_hmc_method(pepolar_method):
    assert selection_for_config('eddy', pepolar_method).pepolar_tools == (SdcTool.TOPUP,)
    assert selection_for_config('shoreline', pepolar_method).pepolar_tools == (SdcTool.DRBUDDI,)
    assert selection_for_config('tortoise', pepolar_method).pepolar_tools == (SdcTool.DRBUDDI,)


def test_pepolar_vocabulary_old_and_new():
    for value in ('TOPUP+DRBUDDI', 'topup+drbuddi'):
        assert selection_for_config('eddy', value).pepolar_tools == (
            SdcTool.TOPUP,
            SdcTool.DRBUDDI,
        )
    assert selection_for_config('eddy', 'drbuddi').pepolar_tools == (SdcTool.DRBUDDI,)


def test_legacy_values_round_trip():
    for hmc_model, pepolar_method in itertools.product(
        LEGACY_HMC_MODELS, LEGACY_PEPOLAR_METHODS
    ):
        selection = selection_for_config(hmc_model, pepolar_method)
        rebuilt = selection_for_config(
            selection.legacy_hmc_model, selection.legacy_pepolar_method
        )
        assert rebuilt == selection


def test_unknown_values_raise():
    with pytest.raises(ValueError, match='hmc'):
        selection_for_config('bogus', 'TOPUP')
    with pytest.raises(ValueError, match='pepolar'):
        selection_for_config('eddy', 'bogus')


def test_selection_validation():
    with pytest.raises(ValueError, match='shoreline_model'):
        MethodSelection(
            hmc=HmcMethod.EDDY, pepolar_tools=(SdcTool.TOPUP,), shoreline_model='3dshore'
        )
    with pytest.raises(ValueError, match='shoreline_model'):
        MethodSelection(hmc=HmcMethod.SHORELINE, pepolar_tools=(SdcTool.DRBUDDI,))
    with pytest.raises(ValueError, match='model'):
        MethodSelection(
            hmc=HmcMethod.SHORELINE, pepolar_tools=(SdcTool.DRBUDDI,), shoreline_model='dti'
        )
    with pytest.raises(ValueError, match='Duplicate'):
        MethodSelection(hmc=HmcMethod.EDDY, pepolar_tools=(SdcTool.TOPUP, SdcTool.TOPUP))
    with pytest.raises(ValueError, match='not a PEPOLAR tool'):
        MethodSelection(hmc=HmcMethod.EDDY, pepolar_tools=(SdcTool.FIELDMAP,))


@pytest.mark.parametrize(
    ('backend', 'expected_label'),
    [
        ('fsl', 'eddy + TOPUP'),
        ('mixed', 'eddy + TOPUP→DRBUDDI'),
        ('tortoise', 'TORTOISE + DRBUDDI'),
    ],
)
def test_canonical_selection_previews_its_backend(backend, expected_label):
    selection = canonical_selection(backend)
    assert selection.legacy_backend == backend
    assert selection.label() == expected_label


def test_canonical_selection_covers_all_backends():
    for backend in BACKENDS:
        assert canonical_selection(backend).legacy_backend == backend
    with pytest.raises(ValueError, match='backend'):
        canonical_selection('afni')


def test_shoreline_label_is_first_class():
    selection = selection_for_config('shoreline', 'drbuddi')
    assert selection.label() == 'SHORELine + DRBUDDI'


def test_registries_are_total():
    assert set(HMC_CAPABILITIES) == set(HmcMethod)
    assert set(SDC_CAPABILITIES) == set(SdcTool)
    consumable = frozenset().union(*(cap.consumes for cap in SDC_CAPABILITIES.values()))
    assert consumable == frozenset(CorrectionMethod)


def test_integrated_pepolar_is_a_capable_tool():
    for capabilities in HMC_CAPABILITIES.values():
        if capabilities.integrated_pepolar is not None:
            assert capabilities.integrated_pepolar in capabilities.pepolar_tools


def test_method_selection_from_config_reads_new_and_legacy_keys():
    from qsiprep import config
    from qsiprep.grouping.methods import method_selection_from_config

    keys = ('hmc_method', 'shoreline_model', 'sdc_method', 'hmc_model', 'pepolar_method',
            'use_syn_sdc', 'force')
    saved = {key: getattr(config.workflow, key) for key in keys}
    try:
        config.workflow.hmc_method = 'shoreline'
        config.workflow.shoreline_model = 'tensor'
        config.workflow.sdc_method = 'drbuddi'
        config.workflow.hmc_model = 'tensor'
        config.workflow.pepolar_method = 'TOPUP'
        config.workflow.use_syn_sdc = None
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
