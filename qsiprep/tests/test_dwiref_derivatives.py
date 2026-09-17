"""Rendered-path guards for the dwiref derivatives.

These assert the paths QSIPrep's own ``io_spec.json`` patterns produce, not the
entities handed to a datasink: entity-level assertions are blind to two names
collapsing onto one path.
"""

import json

import pytest
from bids.layout.writing import build_path

from qsiprep.data import load as load_data

PATTERNS = json.loads(load_data('io_spec.json').read_text())['default_path_patterns']


def _render(**entities):
    entities.setdefault('subject', '01')
    return build_path(entities, PATTERNS, strict=False)


@pytest.mark.parametrize(
    ('entities', 'expected'),
    [
        (
            dict(
                datatype='dwi',
                suffix='xfm',
                extension='.mat',
                **{'from': 'subject', 'to': 'ACPC', 'mode': 'image'},
            ),
            'sub-01/dwi/sub-01_from-subject_to-ACPC_mode-image_xfm.mat',
        ),
        (
            dict(datatype='dwi', suffix='dwiref', extension='.tsv', desc='templateQC'),
            'sub-01/dwi/sub-01_desc-templateQC_dwiref.tsv',
        ),
    ],
)
def test_new_dwi_patterns_render(entities, expected):
    """Before this task both return None: there is no dwi transform pattern, and
    the dwi dwiref pattern permits no tsv."""
    assert _render(**entities) == expected


def test_the_new_dwi_transform_pattern_does_not_shadow_the_anat_one():
    """pybids takes the first matching pattern, so ordering matters."""
    assert (
        _render(
            datatype='anat',
            suffix='xfm',
            extension='.mat',
            **{'from': 'orig', 'to': 'ACPC', 'mode': 'image'},
        )
        == 'sub-01/anat/sub-01_from-orig_to-ACPC_mode-image_xfm.mat'
    )


def test_existing_dwiref_paths_are_unchanged():
    """The tsv addition must not disturb the NIfTI references."""
    assert (
        _render(datatype='dwi', suffix='dwiref', extension='.nii.gz', space='ACPC', session='1')
        == 'sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_dwiref.nii.gz'
    )
