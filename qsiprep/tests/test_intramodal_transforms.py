"""The intramodal space must be reachable from BIDS and from ACPC.

Both transforms were computed and fed straight into ComposeTransforms for
resampling, then discarded. That made the intramodal space a dead end: a user
could not map a session's native b=0 into it, or the template back to ACPC --
the two hops that make a cross-session template interpretable at all.

It also meant the space of the written template could not be checked
independently, which is how a template got written in the wrong space while
carrying a correct-looking space-ACPC name.
"""

import json

from bids.layout.writing import build_path

from qsiprep.data import load as load_data


def _patterns():
    return json.loads(load_data('io_spec.json').read_text())['default_path_patterns']


def test_intramodal_to_acpc_path():
    """Subject level: the template's hop into ACPC."""
    out = build_path(
        dict(
            subject='01',
            datatype='anat',
            suffix='xfm',
            mode='image',
            extension='.mat',
            **{'from': 'intramodal', 'to': 'ACPC'},
        ),
        _patterns(),
        strict=False,
    )
    assert out == 'sub-01/anat/sub-01_from-intramodal_to-ACPC_mode-image_xfm.mat'


def test_orig_to_intramodal_path():
    """Session level: each session's b=0 into the template."""
    out = build_path(
        dict(
            subject='01',
            session='3',
            datatype='anat',
            suffix='xfm',
            mode='image',
            extension='.mat',
            **{'from': 'orig', 'to': 'intramodal'},
        ),
        _patterns(),
        strict=False,
    )
    assert out == 'sub-01/ses-3/anat/sub-01_ses-3_from-orig_to-intramodal_mode-image_xfm.mat'


def test_existing_transform_paths_unchanged():
    """The anatomical round trip must keep working."""
    pats = _patterns()
    for ents, expected in (
        (
            dict(subject='01', session='3', datatype='anat', suffix='xfm', mode='image',
                 extension='.mat', **{'from': 'orig', 'to': 'anat'}),
            'sub-01/ses-3/anat/sub-01_ses-3_from-orig_to-anat_mode-image_xfm.mat',
        ),
        (
            dict(subject='01', datatype='anat', suffix='xfm', mode='image',
                 extension='.h5', **{'from': 'ACPC', 'to': 'MNI152NLin2009cAsym'}),
            'sub-01/anat/sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
        ),
    ):
        assert build_path(ents, pats, strict=False) == expected


def test_both_sinks_are_wired_in_base():
    """Guard against the transforms silently going unwritten again."""
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base)
    assert 'ds_intramodal_to_acpc' in src
    assert 'ds_orig_to_intramodal' in src
    assert "'outputnode.intramodal_template_to_t1_affine', 'in_file'" in src
