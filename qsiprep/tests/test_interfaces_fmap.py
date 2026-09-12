"""Tests for the qsiprep.interfaces.fmap module."""

import json
from pathlib import Path

import nibabel as nb
import numpy as np

from qsiprep.interfaces.fmap import (
    B0RPEFieldmap,
    CleanupEdgeFilter,
    DespikeFilter,
    FieldmapToVSM,
    MedianFilter,
    _despike_2d,
    _sphere_footprint,
)
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


# The fieldmap edge cleanup (denoise + edge blend) runs entirely in nibabel/scipy.


def _write(path, data, zooms=(2.0, 2.0, 2.0)):
    affine = np.diag([*zooms, 1.0])
    nb.Nifti1Image(data.astype('float32'), affine).to_filename(str(path))
    return str(path)


def test_sphere_footprint_geometry():
    """A 3 mm spherical kernel on 2 mm voxels keeps center+faces+edges, not corners."""
    fp = _sphere_footprint(3.0, (2.0, 2.0, 2.0))
    assert fp.shape == (3, 3, 3)
    assert fp[1, 1, 1]  # center
    assert fp[0, 1, 1]  # face (2 mm)
    assert fp[1, 1, 0]  # face (2 mm)
    assert fp[0, 0, 1]  # in-plane edge (2.83 mm <= 3)
    assert not fp[0, 0, 0]  # corner (3.46 mm > 3)
    assert fp.sum() == 19


def test_median_filter_removes_isolated_spike(tmp_path):
    """The median denoise kills a lone spike and preserves shape/affine."""
    data = np.zeros((9, 9, 9), dtype='float32')
    data[4, 4, 4] = 500.0  # isolated spike, outnumbered in any neighborhood
    in_file = _write(tmp_path / 'spiky.nii.gz', data)

    result = _run(MedianFilter(in_file=in_file, kernel_radius_mm=3), tmp_path / 'w')
    out = nb.load(result.outputs.out_file)

    assert out.shape == (9, 9, 9)
    assert np.allclose(out.affine, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert out.get_fdata()[4, 4, 4] == 0.0


def test_cleanup_edge_blends_despiked_rim_into_original_interior(tmp_path):
    """Interior keeps the original field; the eroded rim takes the despiked values."""
    # A 2-voxel-thick slab so erosion leaves a clear interior and a one-voxel rim.
    mask = np.zeros((7, 7, 7), dtype='float32')
    mask[2:5, 2:5, 2:5] = 1.0
    original = np.full((7, 7, 7), 10.0, dtype='float32')
    despiked = np.full((7, 7, 7), 99.0, dtype='float32')

    result = _run(
        CleanupEdgeFilter(
            in_file=_write(tmp_path / 'fmap.nii.gz', original),
            despiked_file=_write(tmp_path / 'despiked.nii.gz', despiked),
            in_mask=_write(tmp_path / 'mask.nii.gz', mask),
        ),
        tmp_path / 'w',
    )
    out = nb.load(result.outputs.out_file).get_fdata()

    from scipy.ndimage import grey_erosion

    eroded = grey_erosion(mask, footprint=np.ones((3, 3, 1), dtype=bool))
    interior = eroded > 0
    edge = (mask - eroded) >= 0.5

    assert np.all(out[interior] == 10.0)  # original field kept inside
    assert np.all(out[edge] == 99.0)  # rim replaced by despiked values
    assert np.all(out[~(interior | edge)] == 0.0)  # nothing outside mask ∪ rim


def test_fieldmap_to_vsm_uses_standard_shift_formula(tmp_path):
    """VSM = fmap[rad/s]/(2*pi) * ees * N_pe (the standard EPI voxel-shift formula).

    The ``*N_pe`` factor is what makes the shift full-readout rather than per-line.
    """
    shape = (10, 12, 8)
    y = np.linspace(-1, 1, shape[1])[None, :, None]
    field_rads = np.broadcast_to((300.0 * y).astype('float32'), shape).astype('float32')
    in_file = _write(tmp_path / 'field.nii.gz', field_rads)
    ees = 0.0006

    result = _run(FieldmapToVSM(in_file=in_file, dwell_time=ees, pe_dir='j'), tmp_path / 'wj')
    vsm = nb.load(result.outputs.shift_out_file).get_fdata()

    assert np.allclose(vsm, field_rads / (2 * np.pi) * ees * shape[1], atol=1e-5)

    # The axis (not the sign) sets N_pe: 'k' uses the 3rd dimension...
    res_k = _run(FieldmapToVSM(in_file=in_file, dwell_time=ees, pe_dir='k'), tmp_path / 'wk')
    vsm_k = nb.load(res_k.outputs.shift_out_file).get_fdata()
    assert np.allclose(vsm_k, field_rads / (2 * np.pi) * ees * shape[2], atol=1e-5)

    # ...and the polarity sign does not change the VSM (applied downstream).
    res_jm = _run(FieldmapToVSM(in_file=in_file, dwell_time=ees, pe_dir='j-'), tmp_path / 'wjm')
    assert np.allclose(nb.load(res_jm.outputs.shift_out_file).get_fdata(), vsm, atol=1e-6)


def test_despike_metric_is_relative_to_neighbor_range():
    """A spike is flagged only when |dev|/range(neighbors) exceeds the threshold.

    Same absolute deviation, different local spread: flagged in a flat neighborhood,
    left alone in a variable one.
    """
    # flat neighborhood (range ~0.2): a +1 spike is >>2.1x the range -> despiked
    flat = np.full((5, 5, 1), 10.0, dtype='float32')
    flat += np.linspace(0, 0.2, 5)[:, None, None]
    flat[2, 2, 0] += 1.0
    out = _despike_2d(flat, threshold=2.1)
    assert out[2, 2, 0] < 10.5  # replaced by the local (~10.1) lower median

    # variable neighborhood (range ~8): the same +1 bump is well under threshold
    var = np.zeros((5, 5, 1), dtype='float32')
    var += np.linspace(0, 8, 5)[:, None, None]
    before = var[2, 2, 0]
    var[2, 2, 0] += 1.0
    out = _despike_2d(var, threshold=2.1)
    assert out[2, 2, 0] == before + 1.0  # untouched


def test_despike_filter_replaces_spike_and_zeros_outside_mask(tmp_path):
    """DespikeFilter replaces an isolated spike with the local value and masks output."""
    data = np.zeros((7, 7, 3), dtype='float32')
    data += np.linspace(0, 6, 7)[:, None, None]  # gentle ramp along axis 0
    data[3, 3, 1] += 100.0  # isolated spike; true value there is 3.0
    mask = np.zeros((7, 7, 3), dtype='float32')
    mask[1:6, 1:6, :] = 1.0

    result = _run(
        DespikeFilter(
            in_file=_write(tmp_path / 'f.nii.gz', data),
            in_mask=_write(tmp_path / 'm.nii.gz', mask),
            threshold=2.1,
        ),
        tmp_path / 'w',
    )
    out = nb.load(result.outputs.out_file)
    o = out.get_fdata()

    assert out.get_data_dtype() == np.float32
    assert np.allclose(out.affine, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert np.isclose(o[3, 3, 1], 3.0, atol=1e-4)  # spike -> local lower median
    assert np.all(o[mask == 0] == 0.0)  # zeroed outside the mask
    # a non-spike in-mask voxel is untouched
    assert np.isclose(o[2, 4, 1], data[2, 4, 1], atol=1e-4)


def test_median_and_cleanup_write_float32_from_integer_input(tmp_path):
    """Filtering an int16 image yields float32, not a requantised int.

    Passing the source header through nibabel would otherwise keep the int16
    dtype and scale the float result into it.
    """
    data = (np.random.default_rng(0).normal(0, 300, (8, 8, 6))).astype('int16')
    src = tmp_path / 'int16.nii.gz'
    img = nb.Nifti1Image(data, np.eye(4))
    img.set_data_dtype(np.int16)
    img.to_filename(str(src))

    med = _run(MedianFilter(in_file=str(src), kernel_radius_mm=3), tmp_path / 'm')
    assert nb.load(med.outputs.out_file).get_data_dtype() == np.float32

    mask = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(np.ones((8, 8, 6), 'int16'), np.eye(4)).to_filename(str(mask))
    clean = _run(
        CleanupEdgeFilter(in_file=str(src), despiked_file=str(src), in_mask=str(mask)),
        tmp_path / 'c',
    )
    assert nb.load(clean.outputs.out_file).get_data_dtype() == np.float32
