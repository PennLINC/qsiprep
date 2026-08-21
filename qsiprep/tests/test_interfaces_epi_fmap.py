"""Tests for the qsiprep.interfaces.epi_fmap module."""

import pytest

from qsiprep.interfaces.epi_fmap import (
    get_distortion_grouping,
    load_epi_dwi_fieldmaps,
    read_nifti_sidecar,
)
from qsiprep.tests.utils import (
    COMPLEX_DWI_SKELETON,
    COMPLEX_EPI_SKELETON,
    SHARED_DWI_GRADIENTS,
    SHARED_EPI_GRADIENTS,
    build_test_dataset,
)

BARE_DWI = {'01': [{'dwi': [{'suffix': 'dwi'}]}]}

# Two AP runs and one PA run, forming two distortion groups.
MULTI_PED_DWI = {
    '01': [
        {
            'dwi': [
                {'dir': 'AP', 'run': '1', 'suffix': 'dwi'},
                {'dir': 'AP', 'run': '2', 'suffix': 'dwi'},
                {'dir': 'PA', 'run': '1', 'suffix': 'dwi'},
            ],
        },
    ],
}


def test_read_nifti_sidecar_reads_a_colocated_sidecar(tmp_path):
    """Metadata beside the image is still found."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {
            '01': [
                {
                    'dwi': [
                        {
                            'suffix': 'dwi',
                            'metadata': {
                                'PhaseEncodingDirection': 'j-',
                                'TotalReadoutTime': 0.05,
                                'SliceTiming': [0.0, 0.5],
                            },
                        },
                    ],
                },
            ],
        },
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert read_nifti_sidecar(str(dwi)) == {
        'PhaseEncodingDirection': 'j-',
        'TotalReadoutTime': 0.05,
        'SliceTiming': [0.0, 0.5],
    }


def test_read_nifti_sidecar_inherits_metadata(tmp_path):
    """A top-level sidecar supplies metadata for an image that has none (issue #685)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        BARE_DWI,
        extra_files={'dwi.json': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.09}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert read_nifti_sidecar(str(dwi)) == {
        'PhaseEncodingDirection': 'j',
        'TotalReadoutTime': 0.09,
        'SliceTiming': None,
    }


def test_read_nifti_sidecar_merges_inherited_and_local_metadata(tmp_path):
    """Inherited keys fill in around the image's own sidecar."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'dwi': [{'suffix': 'dwi', 'metadata': {'PhaseEncodingDirection': 'j-'}}]}]},
        extra_files={'dwi.json': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.09}},
    )
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    assert read_nifti_sidecar(str(dwi)) == {
        'PhaseEncodingDirection': 'j-',
        'TotalReadoutTime': 0.09,
        'SliceTiming': None,
    }


def test_read_nifti_sidecar_shared_by_magnitude_and_phase(tmp_path):
    """A single sidecar covers both parts of a complex-valued acquisition."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={
            'sub-01/dwi/sub-01_dwi.json': {
                'PhaseEncodingDirection': 'j',
                'TotalReadoutTime': 0.1,
            },
        },
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    assert read_nifti_sidecar(str(dwi_dir / 'sub-01_part-mag_dwi.nii.gz')) == read_nifti_sidecar(
        str(dwi_dir / 'sub-01_part-phase_dwi.nii.gz')
    )


def test_read_nifti_sidecar_errors_when_no_metadata_applies(tmp_path):
    """An image with no applicable sidecar raises a message naming the file."""
    root = build_test_dataset(tmp_path / 'ds', BARE_DWI)
    dwi = root / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz'

    with pytest.raises(ValueError, match='No metadata'):
        read_nifti_sidecar(str(dwi))


def test_get_distortion_grouping_uses_inherited_metadata(tmp_path):
    """Distortion groups are found when PE direction is only in an inherited sidecar."""
    root = build_test_dataset(
        tmp_path / 'ds',
        MULTI_PED_DWI,
        # Split across two levels, since only one sidecar per level may apply.
        extra_files={
            'dwi.json': {'TotalReadoutTime': 0.05},
            'sub-01/sub-01_dir-AP_dwi.json': {'PhaseEncodingDirection': 'j-'},
            'sub-01/sub-01_dir-PA_dwi.json': {'PhaseEncodingDirection': 'j'},
        },
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    acqps, groups = get_distortion_grouping(
        [
            str(dwi_dir / 'sub-01_dir-AP_run-1_dwi.nii.gz'),
            str(dwi_dir / 'sub-01_dir-AP_run-2_dwi.nii.gz'),
            str(dwi_dir / 'sub-01_dir-PA_run-1_dwi.nii.gz'),
        ]
    )

    assert acqps == ['0 -1 0 0.050000', '0 1 0 0.050000']
    assert groups == [1, 1, 2]


def test_get_distortion_grouping_handles_complex_valued_dwi(tmp_path):
    """Both parts of a complex-valued run land in the same distortion group (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_DWI_SKELETON,
        extra_files={
            **SHARED_DWI_GRADIENTS,
            'sub-01/dwi/sub-01_dwi.json': {
                'PhaseEncodingDirection': 'j-',
                'TotalReadoutTime': 0.05,
            },
        },
    )
    dwi_dir = root / 'sub-01' / 'dwi'

    acqps, groups = get_distortion_grouping(
        [
            str(dwi_dir / 'sub-01_part-mag_dwi.nii.gz'),
            str(dwi_dir / 'sub-01_part-phase_dwi.nii.gz'),
        ]
    )

    assert acqps == ['0 -1 0 0.050000']
    assert groups == [1, 1]


def test_load_epi_dwi_fieldmaps_handles_complex_valued_fieldmaps(tmp_path):
    """A part-mag fieldmap inherits the shared, non-part-specific bval (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_EPI_SKELETON,
        extra_files=SHARED_EPI_GRADIENTS,
        n_volumes=3,
    )
    fmap_dir = root / 'sub-01' / 'fmap'

    for part in ('mag', 'phase'):
        fmap = fmap_dir / f'sub-01_dir-PA_part-{part}_epi.nii.gz'
        _, b0_indices, _ = load_epi_dwi_fieldmaps([str(fmap)], b0_threshold=100)
        assert b0_indices == [0, 2]


def test_load_epi_dwi_fieldmaps_uses_an_inherited_bval(tmp_path):
    """A 'secret' bval file is honored even when it is inherited (issue #990)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'fmap': [{'dir': 'PA', 'run': '1', 'suffix': 'epi'}]}]},
        extra_files={'sub-01/fmap/sub-01_dir-PA_epi.bval': '0 2000 0\n'},
        n_volumes=3,
    )
    fmap = root / 'sub-01' / 'fmap' / 'sub-01_dir-PA_run-1_epi.nii.gz'

    _, b0_indices, original_files = load_epi_dwi_fieldmaps([str(fmap)], b0_threshold=100)

    assert b0_indices == [0, 2]
    assert original_files == [str(fmap)] * 3


def test_load_epi_dwi_fieldmaps_without_a_bval_keeps_every_volume(tmp_path):
    """A fieldmap with no applicable bval file contributes all of its volumes."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'fmap': [{'dir': 'PA', 'suffix': 'epi'}]}]},
        n_volumes=2,
    )
    fmap = root / 'sub-01' / 'fmap' / 'sub-01_dir-PA_epi.nii.gz'

    _, b0_indices, _ = load_epi_dwi_fieldmaps([str(fmap)], b0_threshold=100)

    assert b0_indices == [0, 1]
