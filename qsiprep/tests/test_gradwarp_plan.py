"""Resolution of the per-unit gradwarp plan from ImageType and the CLI flags."""

import pytest

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.workflows.dwi.gradwarp import _reset_plan_logging, resolve_gradwarp_plan

DWI = '/data/sub-01_dwi.nii.gz'
COEFF = '/opt/coeff.grad'


@pytest.fixture(autouse=True)
def _reset_config():
    # resolve_gradwarp_plan suppresses exact repeats of a rendered log line, so
    # that memory has to be cleared between tests or one test can silence
    # another's expected message.
    _reset_plan_logging()
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    yield
    _reset_plan_logging()
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []


def _unit(image_type=None, manufacturer='SIEMENS', files=(DWI,), per_file=None):
    metadata = {'Manufacturer': manufacturer}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit(list(files), metadata=metadata, per_file_metadata=per_file)


def test_no_gradient_file_means_no_plan():
    """ImageType is never consulted when the feature is off."""
    assert resolve_gradwarp_plan(_unit(['ORIGINAL', 'PRIMARY'])) is None


def test_ignore_gradients_disables_everything():
    config.workflow.gradient_file = COEFF
    config.workflow.ignore = ['gradients']
    assert resolve_gradwarp_plan(_unit()) is None


@pytest.mark.parametrize(
    ('image_type', 'expected'),
    [
        (None, '3D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND'], '3D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND', 'DIS2D'], '1D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND', 'DIS3D'], None),
        (['ORIGINAL', 'DIS2D', 'DIS3D'], None),
    ],
)
def test_warp_dim_from_image_type(image_type, expected):
    config.workflow.gradient_file = COEFF
    plan = resolve_gradwarp_plan(_unit(image_type))
    assert plan is not None
    assert plan.warp_dim == expected
    assert plan.basis == 'metadata'


def test_image_type_may_be_a_bare_string():
    """Some converters write ImageType as a backslash-joined string."""
    config.workflow.gradient_file = COEFF
    plan = resolve_gradwarp_plan(_unit('ORIGINAL\\PRIMARY\\M\\DIS2D'))
    assert plan.warp_dim == '1D'


def test_force_overrides_metadata():
    config.workflow.gradient_file = COEFF
    config.workflow.force = ['gradients']
    plan = resolve_gradwarp_plan(_unit(['ORIGINAL', 'DIS3D']))
    assert plan.warp_dim == '3D'
    assert plan.basis == 'forced'


def test_mixed_image_types_take_the_minimum_warp(caplog):
    """A unit is concatenated before HMC and shares one field, so the members
    must agree. Under-correcting is recoverable; double-correcting is not."""
    config.workflow.gradient_file = COEFF
    other = '/data/sub-01_run-2_dwi.nii.gz'
    plan = resolve_gradwarp_plan(
        _unit(
            ['ORIGINAL', 'PRIMARY'],
            files=(DWI, other),
            per_file={other: {'ImageType': ['ORIGINAL', 'DIS2D']}},
        )
    )
    assert plan.warp_dim == '1D'
    assert other in caplog.text


def test_consistent_image_types_do_not_warn(caplog):
    config.workflow.gradient_file = COEFF
    other = '/data/sub-01_run-2_dwi.nii.gz'
    resolve_gradwarp_plan(_unit(['ORIGINAL', 'DIS2D'], files=(DWI, other)))
    assert 'disagree' not in caplog.text


@pytest.mark.parametrize(
    ('manufacturer', 'expected'),
    [
        ('GE MEDICAL SYSTEMS', True),
        ('ge medical systems', True),
        ('  GE  ', True),
        ('SIEMENS', False),
        ('Philips Medical Systems', False),
        ('', False),
    ],
)
def test_is_ge_detection(manufacturer, expected):
    """Manufacturer is free text from DICOM, so variants must be handled.

    Resolved on a ``DIS3D`` unit so the detection can be observed on its own:
    a GE unit that would get a spatial field is refused outright (see
    ``test_ge_coefficients_are_refused``), but ``is_ge`` still has to be right,
    because ``grad_dev`` is produced either way and passes it to
    ``CreateGradientNonlinearityBMatrix --isGE``.
    """
    config.workflow.gradient_file = COEFF
    plan = resolve_gradwarp_plan(_unit(['ORIGINAL', 'DIS3D'], manufacturer=manufacturer))
    assert plan.is_ge is expected


def test_plan_carries_the_coefficient_file():
    config.workflow.gradient_file = COEFF
    assert resolve_gradwarp_plan(_unit()).coeff_file == COEFF


def test_resolving_the_same_unit_repeatedly_logs_once(caplog):
    """One unit is resolved three times (base, the HMC backend, finalize).

    Without suppression, one plan prints as three, and the mixed-``ImageType``
    warning prints as three separate problems.
    """
    config.workflow.gradient_file = COEFF
    unit = _unit()
    with caplog.at_level('INFO', logger='nipype.workflow'):
        for _ in range(3):
            resolve_gradwarp_plan(unit)

    assert caplog.text.count('spatial warp 3D') == 1


def test_a_different_unit_still_logs(caplog):
    """Suppression is per rendered message, and every message names the unit."""
    config.workflow.gradient_file = COEFF
    with caplog.at_level('INFO', logger='nipype.workflow'):
        resolve_gradwarp_plan(_unit())
        resolve_gradwarp_plan(_unit(files=('/data/sub-02_dwi.nii.gz',)))

    assert caplog.text.count('spatial warp 3D') == 2


def test_repeated_mixed_image_type_warnings_are_suppressed(caplog):
    config.workflow.gradient_file = COEFF
    other = '/data/sub-01_run-2_dwi.nii.gz'
    unit = _unit(
        ['ORIGINAL', 'PRIMARY'],
        files=(DWI, other),
        per_file={other: {'ImageType': ['ORIGINAL', 'DIS2D']}},
    )
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        for _ in range(3):
            resolve_gradwarp_plan(unit)

    assert caplog.text.count('disagree about scanner gradwarp correction') == 1
