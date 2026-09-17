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


def _unit(output_name, dwi_dir):
    """A stand-in carrying only what the guard reads."""
    import types

    return types.SimpleNamespace(
        output_name=output_name,
        dwi_files=[f'{dwi_dir}/{output_name}_dwi.nii.gz'],
    )


def test_distinct_output_names_never_raise():
    from qsiprep.workflows.base import check_output_names_are_bids_unique

    check_output_names_are_bids_unique(
        [_unit('sub-01_acq-A', '/data/sub-01/dwi'), _unit('sub-01_acq-B', '/data/sub-01/dwi')]
    )


@pytest.mark.parametrize(
    ('names', 'dwi_dir'),
    [
        (['sub-01', 'sub-01+2'], '/data/sub-01/dwi'),
        (['sub-01_ses-1', 'sub-01_ses-1+2'], '/data/sub-01/ses-1/dwi'),
    ],
)
def test_plus_suffixed_output_names_always_raise(names, dwi_dir):
    """Must hold on BOTH sides of the pybids 0.19 entity-pattern change.

    QSIPlan uniquifies same-named correction units as ``<base>+N``, an in-memory
    key rather than a BIDS entity. The subject and session patterns are
    path-anchored, so the real input directory supplies those entities and the
    filename's ``+N`` is never read -- under pybids 0.15.6 and 0.19.0 alike.

    A synthetic probe directory would make the session case pass wrongly under
    0.19, leaving a silent overwrite unguarded. That is the bug this pins.
    """
    from qsiprep.workflows.base import check_output_names_are_bids_unique

    units = [_unit(name, dwi_dir) for name in names]
    with pytest.raises(RuntimeError, match=r'render to the same BIDS name'):
        check_output_names_are_bids_unique(units)


# --- the derivative table -----------------------------------------------------
# `space` carries the dwiref's LEVEL; `desc-coreg` marks its ROLE -- the one whose
# transform final resampling actually uses. The two are orthogonal.


@pytest.mark.parametrize('session', [None, '1'])
@pytest.mark.parametrize(
    ('entities', 'name'),
    [
        # Run-level reference, distortion-group resolved: it is the coreg target.
        (dict(desc='coreg', acquisition='A'), 'sub-01_acq-A_desc-coreg_dwiref.nii.gz'),
        # Run-level reference, subject resolved: the template is the coreg target.
        (dict(acquisition='A'), 'sub-01_acq-A_dwiref.nii.gz'),
        # The template, in its own midpoint space.
        (dict(space='subject', desc='coreg'), 'sub-01_space-subject_desc-coreg_dwiref.nii.gz'),
        # The template resampled into ACPC.
        (dict(space='ACPC'), 'sub-01_space-ACPC_dwiref.nii.gz'),
        # Per-output reference, after this rename.
        (
            dict(space='ACPC', desc='preproc', acquisition='A'),
            'sub-01_acq-A_space-ACPC_desc-preproc_dwiref.nii.gz',
        ),
        (dict(space='subject', desc='agreement'), 'sub-01_space-subject_desc-agreement_dwiref.nii.gz'),
    ],
)
def test_dwiref_derivative_paths(entities, name, session):
    """Asserted in both --subject-anatomical-reference modes.

    Subject-level products inherit ses-Y under sessionwise processing, which
    builds one workflow per session; dropping it would make two sessions collide.
    """
    expected = f'sub-01/dwi/{name}'
    if session:
        expected = f'sub-01/ses-{session}/dwi/{name}'.replace('sub-01_', f'sub-01_ses-{session}_')
        entities = dict(entities, session=session)
    assert _render(datatype='dwi', suffix='dwiref', extension='.nii.gz', **entities) == expected


def test_the_template_and_the_per_output_reference_do_not_collide():
    """An entity-free output group renders to bare `sub-01`, which before the
    desc-preproc rename was the template's own path."""
    template = _render(datatype='dwi', suffix='dwiref', extension='.nii.gz', space='ACPC')
    reference = _render(
        datatype='dwi', suffix='dwiref', extension='.nii.gz', space='ACPC', desc='preproc'
    )
    assert template == 'sub-01/dwi/sub-01_space-ACPC_dwiref.nii.gz'
    assert reference == 'sub-01/dwi/sub-01_space-ACPC_desc-preproc_dwiref.nii.gz'
    assert template != reference


def test_dwiref_transform_paths():
    assert (
        _render(
            datatype='dwi', suffix='xfm', extension='.mat',
            **{'from': 'subject', 'to': 'ACPC', 'mode': 'image'},
        )
        == 'sub-01/dwi/sub-01_from-subject_to-ACPC_mode-image_xfm.mat'
    )
    assert (
        _render(
            datatype='dwi', suffix='xfm', extension='.mat', acquisition='A',
            **{'from': 'orig', 'to': 'subject', 'mode': 'image'},
        )
        == 'sub-01/dwi/sub-01_acq-A_from-orig_to-subject_mode-image_xfm.mat'
    )
