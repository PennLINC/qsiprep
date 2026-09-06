"""Tests for the qsiprep.interfaces.epi_fmap module."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.epi_fmap import (
    add_synthetic_b0_to_topup_inputs,
    get_distortion_grouping,
    load_epi_dwi_fieldmaps,
    read_nifti_sidecar,
    synb0_topup_config,
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


def test_load_epi_dwi_fieldmaps_mixes_3d_and_4d_images(tmp_path):
    """3D and 4D fieldmap files can be concatenated together, in either order."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'fmap': [{'dir': 'AP', 'suffix': 'epi'}, {'dir': 'PA', 'suffix': 'epi'}]}]},
        n_volumes=2,
    )
    fmap_dir = root / 'sub-01' / 'fmap'
    pa_file = str(fmap_dir / 'sub-01_dir-PA_epi.nii.gz')
    ap_file = str(fmap_dir / 'sub-01_dir-AP_epi.nii.gz')
    # Make the AP file a single-volume 3D image
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(ap_file)

    for fmap_list, expected_files in (
        ([pa_file, ap_file], [pa_file, pa_file, ap_file]),
        ([ap_file, pa_file], [ap_file, pa_file, pa_file]),
    ):
        concatenated, b0_indices, original_files = load_epi_dwi_fieldmaps(
            fmap_list, b0_threshold=100
        )

        assert concatenated.ndim == 4
        assert concatenated.shape[3] == 3
        assert b0_indices == [0, 1, 2]
        assert original_files == expected_files


def test_load_epi_dwi_fieldmaps_thresholds_a_3d_image_bval(tmp_path):
    """A 3D fieldmap with a one-entry bval file is kept or excluded by b0_threshold."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {
            '01': [
                {
                    'fmap': [
                        {'dir': 'AP', 'run': '1', 'suffix': 'epi'},
                        {'dir': 'AP', 'run': '2', 'suffix': 'epi'},
                    ],
                },
            ],
        },
        extra_files={
            'sub-01/fmap/sub-01_dir-AP_run-1_epi.bval': '0\n',
            'sub-01/fmap/sub-01_dir-AP_run-2_epi.bval': '1000\n',
        },
    )
    fmap_dir = root / 'sub-01' / 'fmap'
    b0_file = str(fmap_dir / 'sub-01_dir-AP_run-1_epi.nii.gz')
    highb_file = str(fmap_dir / 'sub-01_dir-AP_run-2_epi.nii.gz')

    concatenated, b0_indices, original_files = load_epi_dwi_fieldmaps(
        [b0_file, highb_file], b0_threshold=100
    )

    assert concatenated.shape[3] == 2
    # The b=1000 image is not usable as a b=0
    assert b0_indices == [0]
    assert original_files == [b0_file, highb_file]


def _write_synb0_topup_inputs(tmp_path, n_vols=2, readout='0.050000'):
    rng = np.random.default_rng(7)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = rng.uniform(100, 200, size=(8, 8, 6, n_vols)).astype('f4')
    imain = str(tmp_path / 'topup_imain.nii.gz')
    nb.Nifti1Image(data, affine).to_filename(imain)
    datain = str(tmp_path / 'topup_datain.txt')
    with open(datain, 'w') as f:
        f.write('\n'.join([f'0 1 0 {readout}'] * n_vols))
    synthetic = rng.uniform(100, 200, size=(8, 8, 6)).astype('f4')
    synth_file = str(tmp_path / 'synthetic_b0.nii.gz')
    nb.Nifti1Image(synthetic, affine).to_filename(synth_file)
    return imain, datain, synth_file, data, synthetic


def test_add_synthetic_b0_smooths_and_appends(tmp_path):
    imain, datain, synth_file, data, synthetic = _write_synb0_topup_inputs(tmp_path)
    cwd = tmp_path / 'work'
    cwd.mkdir()

    new_datain, new_imain = add_synthetic_b0_to_topup_inputs(
        topup_datain=datain,
        topup_imain=imain,
        synthetic_b0=synth_file,
        cwd=str(cwd),
    )

    out = nb.load(new_imain)
    assert out.shape[3] == 3
    # The synthetic volume is appended unchanged...
    assert np.allclose(out.dataobj[..., 2], synthetic, atol=1e-3)
    # ...while the real volumes are smoothed (their voxelwise noise shrinks).
    for volnum in range(2):
        smoothed = np.asarray(out.dataobj[..., volnum])
        assert not np.allclose(smoothed, data[..., volnum], atol=1e-3)
        assert smoothed.std() < data[..., volnum].std()
        assert np.isclose(smoothed.mean(), data[..., volnum].mean(), rtol=0.01)

    with open(new_datain) as f:
        lines = f.read().splitlines()
    assert lines == ['0 1 0 0.050000', '0 1 0 0.050000', '0 1 0 0.000000']


def test_add_synthetic_b0_resamples_off_grid(tmp_path):
    imain, datain, _, _, _ = _write_synb0_topup_inputs(tmp_path)
    # A synthetic image on a coarser grid must be resampled onto imain's grid
    rng = np.random.default_rng(8)
    coarse = nb.Nifti1Image(
        rng.uniform(100, 200, size=(4, 4, 3)).astype('f4'), np.diag([4.0, 4.0, 4.0, 1.0])
    )
    synth_file = str(tmp_path / 'synthetic_coarse.nii.gz')
    coarse.to_filename(synth_file)
    cwd = tmp_path / 'work'
    cwd.mkdir()

    _, new_imain = add_synthetic_b0_to_topup_inputs(
        topup_datain=datain,
        topup_imain=imain,
        synthetic_b0=synth_file,
        cwd=str(cwd),
    )

    out = nb.load(new_imain)
    assert out.shape == (8, 8, 6, 3)
    assert np.asarray(out.dataobj[..., 2]).max() > 0


def test_synb0_topup_config_resolves_the_distribution(monkeypatch, tmp_path):
    monkeypatch.delenv('SYNB0_ATLASES', raising=False)
    assert synb0_topup_config() == '/opt/synb0/synb0.cnf'
    monkeypatch.setenv('SYNB0_ATLASES', str(tmp_path / 'dist' / 'atlases'))
    assert synb0_topup_config() == str(tmp_path / 'dist' / 'synb0.cnf')
