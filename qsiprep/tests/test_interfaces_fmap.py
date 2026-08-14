"""Tests for the qsiprep.interfaces.fmap module."""

import json
from pathlib import Path

import nibabel as nb

from qsiprep.interfaces.fmap import B0RPEFieldmap
from qsiprep.tests.utils import (
    COMPLEX_EPI_SKELETON,
    SHARED_EPI_GRADIENTS,
    build_test_dataset,
)

PEPOLAR_METADATA = {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}

SINGLE_EPI_SKELETON = {'01': [{'fmap': [{'dir': 'PA', 'suffix': 'epi'}]}]}


def _run(interface, work_dir):
    """Run an interface in a fresh working directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    return interface.run(cwd=str(work_dir))


def test_b0rpe_fieldmap_writes_metadata_not_a_path(tmp_path):
    """The sidecar holds the fieldmap's metadata rather than a JSON file path."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {'01': [{'fmap': [{'dir': 'PA', 'suffix': 'epi', 'metadata': PEPOLAR_METADATA}]}]},
        n_volumes=2,
    )
    fmap = root / 'sub-01' / 'fmap' / 'sub-01_dir-PA_epi.nii.gz'

    result = _run(B0RPEFieldmap(b0_file=[str(fmap)]), tmp_path / 'work')

    assert json.loads(Path(result.outputs.fmap_info).read_text()) == PEPOLAR_METADATA


def test_b0rpe_fieldmap_uses_inherited_metadata(tmp_path):
    """Metadata reached only through inheritance still lands in the sidecar (issue #685)."""
    root = build_test_dataset(
        tmp_path / 'ds',
        SINGLE_EPI_SKELETON,
        extra_files={'epi.json': PEPOLAR_METADATA},
        n_volumes=2,
    )
    fmap = root / 'sub-01' / 'fmap' / 'sub-01_dir-PA_epi.nii.gz'

    result = _run(B0RPEFieldmap(b0_file=[str(fmap)]), tmp_path / 'work')

    assert json.loads(Path(result.outputs.fmap_info).read_text()) == PEPOLAR_METADATA


def test_b0rpe_fieldmap_handles_complex_valued_fieldmaps(tmp_path):
    """A part-mag fieldmap inherits both its metadata and its shared bval."""
    root = build_test_dataset(
        tmp_path / 'ds',
        COMPLEX_EPI_SKELETON,
        extra_files={
            **SHARED_EPI_GRADIENTS,
            'sub-01/fmap/sub-01_dir-PA_epi.json': PEPOLAR_METADATA,
        },
        n_volumes=3,
    )
    fmap = root / 'sub-01' / 'fmap' / 'sub-01_dir-PA_part-mag_epi.nii.gz'

    result = _run(B0RPEFieldmap(b0_file=[str(fmap)]), tmp_path / 'work')

    assert json.loads(Path(result.outputs.fmap_info).read_text()) == PEPOLAR_METADATA
    # The shared bval marks volume 1 as b=2000, leaving two b=0 volumes.
    assert nb.load(result.outputs.fmap_file).shape[3] == 2


def test_b0rpe_fieldmap_merges_two_fieldmaps(tmp_path):
    """Two consistent fieldmaps merge into one metadata object without error."""
    root = build_test_dataset(
        tmp_path / 'ds',
        {
            '01': [
                {
                    'fmap': [
                        {'dir': 'PA', 'run': '1', 'suffix': 'epi', 'metadata': PEPOLAR_METADATA},
                        {'dir': 'PA', 'run': '2', 'suffix': 'epi', 'metadata': PEPOLAR_METADATA},
                    ],
                },
            ],
        },
        n_volumes=2,
    )
    fmap_dir = root / 'sub-01' / 'fmap'

    result = _run(
        B0RPEFieldmap(
            b0_file=[
                str(fmap_dir / 'sub-01_dir-PA_run-1_epi.nii.gz'),
                str(fmap_dir / 'sub-01_dir-PA_run-2_epi.nii.gz'),
            ]
        ),
        tmp_path / 'work',
    )

    assert json.loads(Path(result.outputs.fmap_info).read_text()) == PEPOLAR_METADATA
