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
            {
                'datatype': 'dwi',
                'suffix': 'xfm',
                'extension': '.mat',
                'from': 'subject',
                'to': 'ACPC',
                'mode': 'image',
                'desc': 'coreg',
            },
            'sub-01/dwi/sub-01_from-subject_to-ACPC_mode-image_desc-coreg_xfm.mat',
        ),
        (
            {'datatype': 'dwi', 'suffix': 'dwiref', 'extension': '.tsv', 'desc': 'templateQC'},
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


def _units(names, dwi_dir):
    """Stand-ins carrying only what the check reads.

    The input directory matters: get_source_file places the output name in the
    first input's own directory, and the subject and session entity patterns are
    path-anchored, so that directory is what supplies those entities.
    """
    import types

    return [
        types.SimpleNamespace(output_name=name, dwi_files=[f'{dwi_dir}/{name}_dwi.nii.gz'])
        for name in names
    ]


def test_distinct_output_names_never_raise():
    from qsiprep.utils.bids import check_output_names_are_bids_unique

    check_output_names_are_bids_unique(
        _units(['sub-01_acq-A', 'sub-01_acq-B'], '/data/sub-01/dwi')
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
    from qsiprep.utils.bids import check_output_names_are_bids_unique

    with pytest.raises(RuntimeError, match=r'render to the same BIDS name'):
        check_output_names_are_bids_unique(_units(names, dwi_dir))


# --- the derivative table -----------------------------------------------------
# `space` names the level a dwiref belongs to, following fMRIPrep 26.0.0's
# space-run / space-session / space-subject scheme. `desc-coreg` marks the
# transforms rather than the images.


@pytest.mark.parametrize('session', [None, '1'])
@pytest.mark.parametrize(
    ('entities', 'name'),
    [
        # Run-level reference. Written for every unit, under one name, whatever
        # --dwiref-definition resolves to.
        (
            {'space': 'distortiongroup', 'acquisition': 'A'},
            'sub-01_acq-A_space-distortiongroup_dwiref.nii.gz',
        ),
        # The template, in its own midpoint space.
        ({'space': 'subject'}, 'sub-01_space-subject_dwiref.nii.gz'),
        # The same template resampled into ACPC. `space` is taken, so the level
        # moves to `desc`.
        ({'space': 'ACPC', 'desc': 'subject'}, 'sub-01_space-ACPC_desc-subject_dwiref.nii.gz'),
        # Reference of the preprocessed series.
        (
            {'space': 'ACPC', 'desc': 'preproc', 'acquisition': 'A'},
            'sub-01_acq-A_space-ACPC_desc-preproc_dwiref.nii.gz',
        ),
        (
            {'space': 'subject', 'desc': 'agreement'},
            'sub-01_space-subject_desc-agreement_dwiref.nii.gz',
        ),
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


def test_the_two_acpc_space_images_do_not_collide():
    """An entity-free output group renders to bare `sub-01`, so both ACPC-space
    images need a desc to stay apart."""
    template = _render(
        datatype='dwi', suffix='dwiref', extension='.nii.gz', space='ACPC', desc='subject'
    )
    reference = _render(
        datatype='dwi', suffix='dwiref', extension='.nii.gz', space='ACPC', desc='preproc'
    )
    assert template == 'sub-01/dwi/sub-01_space-ACPC_desc-subject_dwiref.nii.gz'
    assert reference == 'sub-01/dwi/sub-01_space-ACPC_desc-preproc_dwiref.nii.gz'
    assert template != reference


@pytest.mark.parametrize(
    ('entities', 'expected'),
    [
        # Resolved level distortion-group: one hop into anatomy, and back.
        (
            {'from': 'distortiongroup', 'to': 'ACPC', 'acquisition': 'A'},
            'sub-01_acq-A_from-distortiongroup_to-ACPC_mode-image_desc-coreg_xfm.mat',
        ),
        (
            {'from': 'ACPC', 'to': 'distortiongroup', 'acquisition': 'A'},
            'sub-01_acq-A_from-ACPC_to-distortiongroup_mode-image_desc-coreg_xfm.mat',
        ),
        # Resolved level subject: two hops, group into template and template
        # into anatomy, each with an inverse.
        (
            {'from': 'distortiongroup', 'to': 'subject', 'acquisition': 'A'},
            'sub-01_acq-A_from-distortiongroup_to-subject_mode-image_desc-coreg_xfm.mat',
        ),
        (
            {'from': 'subject', 'to': 'ACPC'},
            'sub-01_from-subject_to-ACPC_mode-image_desc-coreg_xfm.mat',
        ),
        (
            {'from': 'ACPC', 'to': 'subject'},
            'sub-01_from-ACPC_to-subject_mode-image_desc-coreg_xfm.mat',
        ),
    ],
)
def test_dwiref_transform_paths(entities, expected):
    """desc-coreg goes on the transforms, following fMRIPrep 26.0.0."""
    assert (
        _render(
            datatype='dwi',
            suffix='xfm',
            extension='.mat',
            mode='image',
            desc='coreg',
            **entities,
        )
        == f'sub-01/dwi/{expected}'
    )
