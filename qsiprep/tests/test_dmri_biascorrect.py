"""The ``--dwi-biascorrect auto`` heuristic.

``auto`` decides from the BIDS ``ImageType`` metadata whether the DWIs were already
intensity-normalized on the console. It is deliberately conservative: N4 is skipped
only when *every* input is marked ``NORM``, because a mixed set is concatenated into
one output and must be corrected consistently, and because running N4 unnecessarily
is a milder error than skipping it when it was needed.
"""

import pytest

from qsiprep.workflows.dwi.biascorrect import dmri_biascorrect_enabled


class _FakeLayout:
    """Stands in for the BIDSLayout, exposing only ``get_metadata``."""

    def __init__(self, image_types):
        self._image_types = image_types

    def get_metadata(self, path):
        image_type = self._image_types[path]
        return {} if image_type is None else {'ImageType': image_type}


class _RaisingLayout:
    """A layout whose sidecars cannot be read."""

    def __init__(self, exc):
        self._exc = exc

    def get_metadata(self, path):
        raise self._exc


@pytest.fixture(autouse=True)
def _restore_config():
    """Put global config state back.

    These tests install a fake layout on the shared ``config.execution``. Leaving
    it there breaks any later test that builds a real one -- the suite is ordered
    alphabetically, so `test_utils_gradcal` picks up the fake and fails. The
    repo's ``_config()`` convention leaks plain values harmlessly; leaking a stub
    object is a different matter.
    """
    from qsiprep import config

    saved = (
        config.workflow.dwi_biascorrect,
        config.execution._layout,
        config.execution.layout,
    )
    yield
    (
        config.workflow.dwi_biascorrect,
        config.execution._layout,
        config.execution.layout,
    ) = saved


def _config(mode, layout=None):
    from qsiprep import config

    config.workflow.dwi_biascorrect = mode
    # `layout` is a plain class attribute assigned by execution.init() from
    # `_layout` (config.py:512), not a property, so both must be set.
    config.execution._layout = layout
    config.execution.layout = layout
    return config


@pytest.mark.parametrize(('mode', 'expected'), [('n4', True), ('none', False)])
def test_explicit_modes_never_consult_metadata(mode, expected):
    """n4 and none are unconditional; a raising layout must not reach them."""
    _config(mode, _RaisingLayout(OSError('should never be read')))
    assert dmri_biascorrect_enabled(['/a_dwi.nii.gz']) is expected


def test_auto_skips_when_every_file_is_norm():
    _config(
        'auto',
        _FakeLayout({'/a_dwi.nii.gz': ['ORIGINAL', 'NORM'], '/b_dwi.nii.gz': ['NORM']}),
    )
    assert dmri_biascorrect_enabled(['/a_dwi.nii.gz', '/b_dwi.nii.gz']) is False


def test_auto_runs_on_a_mixed_set_and_warns(caplog):
    """A mixed set is concatenated, so it must be corrected consistently."""
    _config(
        'auto',
        _FakeLayout({'/a_dwi.nii.gz': ['NORM'], '/b_dwi.nii.gz': ['ORIGINAL']}),
    )
    with caplog.at_level('WARNING', logger='nipype.workflow'):
        assert dmri_biascorrect_enabled(['/a_dwi.nii.gz', '/b_dwi.nii.gz']) is True
    assert '1 of 2' in caplog.text


def test_auto_treats_absent_image_type_as_unnormalized():
    """A missing ImageType key, not merely an empty list."""
    _config('auto', _FakeLayout({'/a_dwi.nii.gz': None}))
    assert dmri_biascorrect_enabled(['/a_dwi.nii.gz']) is True


@pytest.mark.parametrize('exc', [OSError('unreadable'), ValueError('bad'), KeyError('x')])
def test_auto_treats_unreadable_metadata_as_unnormalized(exc):
    """A layout that raises must not take down the workflow build."""
    _config('auto', _RaisingLayout(exc))
    assert dmri_biascorrect_enabled(['/a_dwi.nii.gz']) is True


def test_auto_runs_without_a_layout():
    _config('auto', None)
    assert dmri_biascorrect_enabled(['/a_dwi.nii.gz']) is True


def test_auto_runs_with_no_files():
    _config('auto', _FakeLayout({}))
    assert dmri_biascorrect_enabled([]) is True
    assert dmri_biascorrect_enabled(None) is True
