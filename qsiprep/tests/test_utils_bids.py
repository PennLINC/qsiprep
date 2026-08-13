"""Tests for the BIDS inheritance-principle helpers in qsiprep.utils.bids."""

import pytest

from qsiprep.tests.utils import (
    COMPLEX_DWI_SKELETON,
    SHARED_DWI_GRADIENTS,
    build_test_dataset,
)
from qsiprep.utils.bids import (
    _parse_bids_name,
    find_associated_files,
    find_bids_root,
    find_bval,
    find_bvec,
    load_sidecar,
)

# A single subject with one DWI and no metadata of its own.
BARE_DWI = {'01': [{'dwi': [{'suffix': 'dwi'}]}]}


def test_parse_bids_name_splits_a_complex_valued_dwi(tmp_path):
    """Entities, suffix and extension are read off a part-mag filename."""
    entities, suffix, extension = _parse_bids_name('sub-01_ses-1_part-mag_dwi.nii.gz')

    assert entities == {'sub': '01', 'ses': '1', 'part': 'mag'}
    assert suffix == 'dwi'
    assert extension == '.nii.gz'


def test_parse_bids_name_splits_a_shared_gradient_file(tmp_path):
    """The shared gradient file parses to the same suffix with fewer entities."""
    entities, suffix, extension = _parse_bids_name('sub-01_ses-1_dwi.bval')

    assert entities == {'sub': '01', 'ses': '1'}
    assert suffix == 'dwi'
    assert extension == '.bval'


def test_find_bids_root_finds_dataset_description(tmp_path):
    """The root is the closest ancestor holding dataset_description.json."""
    root = build_test_dataset(
        tmp_path / 'ds', COMPLEX_DWI_SKELETON, extra_files=SHARED_DWI_GRADIENTS
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_part-mag_dwi.nii.gz'

    assert find_bids_root(dwi) == root.resolve()


def test_find_bids_root_returns_none_outside_a_dataset(tmp_path):
    """A file that is not inside a BIDS dataset has no root."""
    stray = tmp_path / 'work' / 'node' / 'sub-01_dwi.nii.gz'
    stray.parent.mkdir(parents=True)
    stray.touch()

    assert find_bids_root(stray) is None


def test_load_sidecar_reads_a_colocated_sidecar(tmp_path):
    """A sidecar next to the image is used when there is nothing to inherit."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'dwi': [{'suffix': 'dwi', 'metadata': {'PhaseEncodingDirection': 'j'}}]}]},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {'PhaseEncodingDirection': 'j'}


def test_load_sidecar_inherits_from_dataset_root(tmp_path):
    """A top-level sidecar applies to an image that has none of its own."""
    root = build_test_dataset(
        tmp_path / 'ds',
        BARE_DWI,
        extra_files={'dwi.json': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}


def test_load_sidecar_inherits_from_subject_directory(tmp_path):
    """A subject-level sidecar applies to images in that subject's session directories."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'session': '1', 'dwi': [{'suffix': 'dwi'}]}]},
        extra_files={'sub-01/sub-01_dwi.json': {'PhaseEncodingDirection': 'j-'}},
    )
    dwi = root / 'sub-01' / 'ses-1' / 'dwi' / 'sub-01_ses-1_dwi.nii.gz'

    assert load_sidecar(dwi) == {'PhaseEncodingDirection': 'j-'}


def test_load_sidecar_merges_levels_with_nearest_winning(tmp_path):
    """Keys merge across levels; the closest file wins on conflicts."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'dwi': [{'suffix': 'dwi', 'metadata': {'PhaseEncodingDirection': 'j-'}}]}]},
        extra_files={'dwi.json': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05}


def test_load_sidecar_ignores_files_with_extra_entities(tmp_path):
    """A more specific sidecar does not apply to a less specific image."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {
            '01': [
                {
                    'dwi': [
                        {'suffix': 'dwi'},
                        {
                            'acq': 'hi',
                            'suffix': 'dwi',
                            'metadata': {'PhaseEncodingDirection': 'j'},
                        },
                    ],
                },
            ],
        },
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {}


def test_load_sidecar_ignores_files_with_a_different_suffix(tmp_path):
    """Inheritance only applies within a suffix."""
    root = build_test_dataset(
        tmp_path / 'ds',
        BARE_DWI,
        extra_files={'sub-01/sub-01_epi.json': {'PhaseEncodingDirection': 'j'}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {}


def test_load_sidecar_ignores_conflicting_entity_values(tmp_path):
    """A sidecar for a different run does not apply."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {
            '01': [
                {
                    'dwi': [
                        {'run': '1', 'suffix': 'dwi'},
                        {'run': '2', 'suffix': 'dwi', 'metadata': {'PhaseEncodingDirection': 'j'}},
                    ],
                },
            ],
        },
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_run-1_dwi.nii.gz'

    assert load_sidecar(dwi) == {}


def test_load_sidecar_returns_empty_when_nothing_applies(tmp_path):
    """A missing sidecar is not an error."""
    root = build_test_dataset(tmp_path / 'ds', BARE_DWI)
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert load_sidecar(dwi) == {}


def test_load_sidecar_outside_a_dataset_uses_the_containing_directory(tmp_path):
    """Without a dataset root, only the file's own directory is searched."""
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'sub-01_dwi.nii.gz').touch()
    (work / 'sub-01_dwi.json').write_text('{"PhaseEncodingDirection": "j"}')
    (tmp_path / 'dwi.json').write_text('{"TotalReadoutTime": 0.05}')

    assert load_sidecar(work / 'sub-01_dwi.nii.gz') == {'PhaseEncodingDirection': 'j'}


def test_load_sidecar_is_shared_by_magnitude_and_phase(tmp_path):
    """Both parts of a complex-valued acquisition reach the same metadata (issue #685)."""
    metadata = {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={**SHARED_DWI_GRADIENTS, 'sub-01/dwi/sub-01_dwi.json': metadata},
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    assert load_sidecar(dwi_dir / 'sub-01_part-mag_dwi.nii.gz') == metadata
    assert load_sidecar(dwi_dir / 'sub-01_part-phase_dwi.nii.gz') == metadata


def test_find_associated_files_matches_shared_gradients_for_both_parts(tmp_path):
    """The shared bvec applies to each part of a complex-valued acquisition."""
    root = build_test_dataset(
        tmp_path / 'ds', COMPLEX_DWI_SKELETON, extra_files=SHARED_DWI_GRADIENTS
    )
    dwi_dir = root / 'sub-01' / 'dwi'
    shared_bvec = dwi_dir / 'sub-01_dwi.bvec'

    assert find_associated_files(dwi_dir / 'sub-01_part-mag_dwi.nii.gz', '.bvec') == [shared_bvec]
    assert find_associated_files(dwi_dir / 'sub-01_part-phase_dwi.nii.gz', '.bvec') == [
        shared_bvec
    ]


def test_find_associated_files_raises_on_same_level_ambiguity(tmp_path):
    """Two applicable files at one level is invalid BIDS."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'session': '1', 'dwi': [{'suffix': 'dwi'}]}]},
        extra_files={
            'sub-01/sub-01_dwi.json': {'PhaseEncodingDirection': 'j'},
            'sub-01/dwi.json': {'PhaseEncodingDirection': 'j-'},
        },
    )
    dwi = root / 'sub-01' / 'ses-1' / 'dwi' / 'sub-01_ses-1_dwi.nii.gz'

    with pytest.raises(ValueError, match='Multiple'):
        find_associated_files(dwi, '.json')


def test_find_associated_files_orders_from_root_to_leaf(tmp_path):
    """Applicable files are returned shallowest-first."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'dwi': [{'suffix': 'dwi', 'metadata': {'PhaseEncodingDirection': 'j-'}}]}]},
        extra_files={'dwi.json': {'PhaseEncodingDirection': 'j'}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert find_associated_files(dwi, '.json') == [
        root / 'dwi.json',
        root / 'sub-01' / 'dwi' / 'sub-01_dwi.json',
    ]


def test_find_bval_uses_the_colocated_file(tmp_path):
    """A bval next to the image wins."""
    root = build_test_dataset(
        tmp_path / 'ds',
        BARE_DWI,
        extra_files={'sub-01/dwi/sub-01_dwi.bval': '0 1000\n'},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert find_bval(dwi) == str(root / 'sub-01' / 'dwi' / 'sub-01_dwi.bval')


def test_find_bval_is_shared_by_magnitude_and_phase(tmp_path):
    """part-mag and part-phase images inherit the same bval file (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={'sub-01/dwi/sub-01_dwi.bval': '0 1000\n'},
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    assert find_bval(dwi_dir / 'sub-01_part-mag_dwi.nii.gz') == str(dwi_dir / 'sub-01_dwi.bval')
    assert find_bval(dwi_dir / 'sub-01_part-phase_dwi.nii.gz') == str(dwi_dir / 'sub-01_dwi.bval')


def test_find_bvec_is_shared_by_magnitude_and_phase(tmp_path):
    """part-mag and part-phase images inherit the same bvec file (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={'sub-01/dwi/sub-01_dwi.bvec': '0 0\n0 0\n0 1\n'},
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    assert find_bvec(dwi_dir / 'sub-01_part-mag_dwi.nii.gz') == str(dwi_dir / 'sub-01_dwi.bvec')
    assert find_bvec(dwi_dir / 'sub-01_part-phase_dwi.nii.gz') == str(dwi_dir / 'sub-01_dwi.bvec')


def test_find_bval_prefers_the_nearest_file(tmp_path):
    """A run-specific bval overrides an inherited one."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={
            'dwi.bval': '0 1000\n',
            'sub-01/dwi/sub-01_dwi.bval': '0 2000\n',
        },
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    assert find_bval(dwi_dir / 'sub-01_part-mag_dwi.nii.gz') == str(dwi_dir / 'sub-01_dwi.bval')


def test_find_bval_returns_none_when_absent(tmp_path):
    """An image with no gradient table resolves to None rather than raising."""
    root = build_test_dataset(tmp_path / 'ds', BARE_DWI)
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert find_bval(dwi) is None


def test_find_bval_applies_to_epi_fieldmaps(tmp_path):
    """An EPI fieldmap can inherit a 'secret' bval file."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'fmap': [{'dir': 'PA', 'run': '1', 'suffix': 'epi'}]}]},
        extra_files={'sub-01/fmap/sub-01_dir-PA_epi.bval': '0 0\n'},
    )
    fmap_dir = root / 'sub-01' / 'fmap'

    assert find_bval(fmap_dir / 'sub-01_dir-PA_run-1_epi.nii.gz') == str(
        fmap_dir / 'sub-01_dir-PA_epi.bval'
    )
