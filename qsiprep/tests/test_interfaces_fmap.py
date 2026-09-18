"""Tests for the qsiprep.interfaces.fmap module."""

import json
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.fmap import B0RPEFieldmap, DisplacementToFieldmap
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


def _write_disp(path, components, affine):
    """Write a uniform ANTs-style (X, Y, Z, 1, 3) displacement field (mm, LPS)."""
    data = np.zeros((3, 3, 3, 1, 3), dtype=np.float32)
    data[..., 0, :] = components
    nb.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def _first_hz(result):
    return float(nb.load(result.outputs.fieldmap_hz).get_fdata().flat[0])


# Grids used below. A NIfTI affine maps voxel -> RAS+; ANTs stores the
# displacement components in LPS, so a +2 mm LPS-P component is a 2 mm posterior
# (RAS -y) shift regardless of the grid orientation.
_RAS = np.diag([2.0, 2.0, 2.0, 1.0])  # voxel +j -> RAS +A (anterior)
_LPS = np.diag([-2.0, -2.0, 2.0, 1.0])  # voxel +j -> RAS -A (posterior)
_ISO1 = np.diag([1.0, 1.0, 1.0, 1.0])


def test_displacement_to_fieldmap_baseline_and_polarity(tmp_path):
    """A 2 mm shift over 2 mm voxels at TRT 0.05 s is 1 voxel = 20 Hz; j- flips it."""
    disp = _write_disp(tmp_path / 'd.nii.gz', [0.0, -2.0, 0.0], _RAS)
    up = DisplacementToFieldmap(displacement_field=disp, pe_dir='j', readout_time=0.05)
    down = DisplacementToFieldmap(displacement_field=disp, pe_dir='j-', readout_time=0.05)
    assert _first_hz(_run(up, tmp_path / 'up')) == pytest.approx(20.0)
    assert _first_hz(_run(down, tmp_path / 'down')) == pytest.approx(-20.0)


def test_displacement_to_fieldmap_voxel_scaling(tmp_path):
    """The same physical shift over smaller voxels is more voxels, so more Hz."""
    disp = _write_disp(tmp_path / 'd.nii.gz', [0.0, -2.0, 0.0], _ISO1)
    iface = DisplacementToFieldmap(displacement_field=disp, pe_dir='j', readout_time=0.05)
    assert _first_hz(_run(iface, tmp_path / 'w')) == pytest.approx(40.0)


def test_displacement_to_fieldmap_axis_selection(tmp_path):
    """'i' projects onto the x component; a purely-x shift is orthogonal to 'j'."""
    disp = _write_disp(tmp_path / 'd.nii.gz', [-2.0, 0.0, 0.0], _RAS)
    on_i = DisplacementToFieldmap(displacement_field=disp, pe_dir='i', readout_time=0.05)
    on_j = DisplacementToFieldmap(displacement_field=disp, pe_dir='j', readout_time=0.05)
    assert _first_hz(_run(on_i, tmp_path / 'i')) == pytest.approx(20.0)
    assert _first_hz(_run(on_j, tmp_path / 'j')) == pytest.approx(0.0)


def test_displacement_to_fieldmap_uses_grid_orientation(tmp_path):
    """The +PE axis is the image's own +j, so flipping the grid flips the sign.

    Same stored LPS components, but on a grid whose voxel-j points posterior; the
    result must invert relative to the RAS-oriented grid. This only holds if the
    affine column (converted to LPS) is actually used in the projection.
    """
    disp = _write_disp(tmp_path / 'd.nii.gz', [0.0, -2.0, 0.0], _LPS)
    iface = DisplacementToFieldmap(displacement_field=disp, pe_dir='j', readout_time=0.05)
    assert _first_hz(_run(iface, tmp_path / 'w')) == pytest.approx(-20.0)


def test_displacement_to_fieldmap_oblique(tmp_path):
    """An oblique PE axis projects by the true world direction, not a voxel axis."""
    theta = np.pi / 4
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rot * 2.0
    # Displace by exactly +2 mm along the (LPS) PE axis -> 1 voxel -> 20 Hz.
    col_lps = affine[:3, 1] * np.array([-1.0, -1.0, 1.0])
    unit = col_lps / np.linalg.norm(col_lps)
    disp = _write_disp(tmp_path / 'd.nii.gz', (2.0 * unit).tolist(), affine)
    iface = DisplacementToFieldmap(displacement_field=disp, pe_dir='j', readout_time=0.05)
    assert _first_hz(_run(iface, tmp_path / 'w')) == pytest.approx(20.0)


def test_displacement_to_fieldmap_antisymmetric_average(tmp_path):
    """With both blip fields, the output is the antisymmetric average.

    DRBUDDI relaxes the antisymmetry constraint late, so FINV and MINV share a
    non-antisymmetric residue that is not an off-resonance field. Here the pure B0
    part is +2 mm anterior ([0, -2, 0]) and both fields carry a +1 mm symmetric
    residue: FINV = [0, -1, 0], MINV = -B0 + residue = [0, 3, 0]. Averaging
    (blip-up at +j, blip-down at -j) cancels the residue and recovers the B0 field
    (20 Hz); the blip-up field alone keeps it (10 Hz).
    """
    up = _write_disp(tmp_path / 'up.nii.gz', [0.0, -1.0, 0.0], _RAS)
    down = _write_disp(tmp_path / 'down.nii.gz', [0.0, 3.0, 0.0], _RAS)
    both = DisplacementToFieldmap(
        displacement_field=up, opposite_displacement_field=down, pe_dir='j', readout_time=0.05
    )
    up_only = DisplacementToFieldmap(displacement_field=up, pe_dir='j', readout_time=0.05)
    assert _first_hz(_run(both, tmp_path / 'both')) == pytest.approx(20.0)
    assert _first_hz(_run(up_only, tmp_path / 'up_only')) == pytest.approx(10.0)


def test_displacement_to_fieldmap_component_fields(tmp_path):
    """Both fields also emit the per-blip QC components [up, down, asymmetry]."""
    from nipype.interfaces.base import isdefined

    up = _write_disp(tmp_path / 'up.nii.gz', [0.0, -1.0, 0.0], _RAS)
    down = _write_disp(tmp_path / 'down.nii.gz', [0.0, 3.0, 0.0], _RAS)
    both = DisplacementToFieldmap(
        displacement_field=up, opposite_displacement_field=down, pe_dir='j', readout_time=0.05
    )
    comps = _run(both, tmp_path / 'both').outputs.component_fieldmaps
    values = [float(nb.load(f).get_fdata().flat[0]) for f in comps]
    # up = +10, down = +30, asymmetry = (up - down) / 2 = -10.
    assert values == pytest.approx([10.0, 30.0, -10.0])

    up_only = DisplacementToFieldmap(displacement_field=up, pe_dir='j', readout_time=0.05)
    assert not isdefined(_run(up_only, tmp_path / 'up_only').outputs.component_fieldmaps)


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
