"""In-brain summaries of the denoiser maps and Jacobian weights in ``desc-image_qc.tsv``."""

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from qsiprep import config
from qsiprep.interfaces.reports import (
    SeriesQC,
    get_denoiser_map_values,
    get_jacobian_values,
    get_tsnr_values,
)
from qsiprep.tests.test_noise_map import _edges, _finalize_wf

AFFINE = np.diag([-2.0, -2.0, 2.0, 1.0])
SHAPE = (12, 12, 12)
INTERIOR = (slice(2, -2),) * 3


def _write_image(path, data, affine=AFFINE):
    nb.Nifti1Image(np.asarray(data, dtype=np.float32), affine).to_filename(str(path))
    return str(path)


def _series_qc_row(tmp_path, monkeypatch, **inputs):
    import qsiprep.interfaces.reports as reports_mod

    # Motion summary is not what these tests are about; it needs a full
    # confounds table otherwise.
    monkeypatch.setattr(reports_mod, 'calculate_motion_summary', lambda _: {'mean_fd': [0.1]})
    confounds = tmp_path / 'confounds.tsv'
    confounds.touch()
    pre_qc = tmp_path / 'pre.csv'
    pd.DataFrame({'neighbor_corr': [0.9]}).to_csv(pre_qc, index=False)

    result = SeriesQC(
        pre_qc=str(pre_qc),
        confounds_file=str(confounds),
        output_file_name='sub-01_ses-1_dwi',
        **inputs,
    ).run(cwd=str(tmp_path))
    return pd.read_csv(result.outputs.series_qc_file, sep='\t').iloc[0]


# ---------------------------------------------------------------------------
# Denoiser maps


def test_denoiser_map_summary_is_over_the_brain_voxels(tmp_path):
    data = np.full(SHAPE, 100.0)
    data[INTERIOR] = np.arange(8**3).reshape((8,) * 3)
    data[2, 2, 2] = np.nan
    mask = np.zeros(SHAPE)
    mask[INTERIOR] = 1

    values = get_denoiser_map_values(
        _write_image(tmp_path / 'noise.nii.gz', data),
        _write_image(tmp_path / 'mask.nii.gz', mask),
        'noise_sigma',
    )

    # The NaN voxel is dropped and the 100s outside the mask never counted
    brain = np.arange(1, 8**3)
    assert values['noise_sigma_mean'] == [pytest.approx(brain.mean())]
    assert values['noise_sigma_standard_deviation'] == [pytest.approx(brain.std(), rel=1e-6)]
    assert values['noise_sigma_median'] == [pytest.approx(np.median(brain), abs=1)]
    assert values['noise_sigma_p05'][0] < values['noise_sigma_median'][0]
    assert values['noise_sigma_p95'][0] == pytest.approx(np.percentile(brain, 95), abs=1)


def test_denoiser_map_summary_weighs_series_by_their_output_volumes(tmp_path):
    """Test that a series counts once for each output volume its map describes."""
    data = np.stack([np.full(SHAPE, 10.0), np.full(SHAPE, 40.0)], axis=-1)
    map_file = _write_image(tmp_path / 'noise.nii.gz', data)
    mask_file = _write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE))

    # Three output volumes come from the first series, one from the second
    weighted = get_denoiser_map_values(map_file, mask_file, 'noise', volume_index=[0, 0, 0, 1])
    assert weighted['noise_mean'] == [pytest.approx(17.5)]
    assert weighted['noise_median'] == [10.0]
    assert weighted['noise_p95'] == [40.0]

    unweighted = get_denoiser_map_values(map_file, mask_file, 'noise')
    assert unweighted['noise_mean'] == [pytest.approx(25.0)]


def test_denoiser_map_summary_resamples_a_mask_on_another_grid(tmp_path):
    # The 4 mm mask covers 2 mm voxels 1-9, inside the nonzero map
    data = np.zeros(SHAPE)
    data[1:11, 1:11, 1:11] = 5.0
    mask = np.zeros((6, 6, 6))
    mask[1:5, 1:5, 1:5] = 1

    values = get_denoiser_map_values(
        _write_image(tmp_path / 'rank.nii.gz', data),
        _write_image(tmp_path / 'mask.nii.gz', mask, affine=np.diag([-4.0, -4.0, 4.0, 1.0])),
        'rank',
    )

    assert values['rank_mean'] == [pytest.approx(5.0)]


def test_series_qc_writes_the_denoiser_map_columns(tmp_path, monkeypatch):
    mask = _write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE))
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=mask,
        noise_map_file=_write_image(tmp_path / 'noise.nii.gz', np.full(SHAPE, 3.0)),
        noise_map_meta={'NoiseMapIndex': [0, 0]},
        noise_quantity='residual_rms',
        rank_density_file=_write_image(tmp_path / 'rankdensity.nii.gz', np.full(SHAPE, 0.5)),
    )

    assert row['noise_residual_rms_median'] == 3.0
    assert row['rank_density_mean'] == 0.5
    assert not any(column.startswith('rank_median') for column in row.index)


# ---------------------------------------------------------------------------
# Jacobian weights


def test_jacobian_summary_of_a_single_map(tmp_path):
    weights = np.ones(SHAPE)
    weights[INTERIOR] = np.linspace(0.5, 1.5, 8**3).reshape((8,) * 3)
    mask = np.zeros(SHAPE)
    mask[INTERIOR] = 1

    values = get_jacobian_values(
        [_write_image(tmp_path / 'weights.nii.gz', weights)],
        _write_image(tmp_path / 'mask.nii.gz', mask),
        weight_index=[0, 0, 0],
    )

    assert values['jacobian_weight_mean'] == [pytest.approx(1.0)]
    assert values['jacobian_weight_p05'][0] == pytest.approx(0.55, abs=0.01)
    assert values['jacobian_weight_p95'][0] == pytest.approx(1.45, abs=0.01)
    assert values['jacobian_weight_volume_deviation'] == [0.0]


def test_jacobian_summary_averages_distinct_maps_over_volumes(tmp_path):
    """Test that per-volume maps are averaged first, each counted once per volume."""
    mask = _write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE))
    maps = [
        _write_image(tmp_path / 'w0.nii.gz', np.full(SHAPE, 0.8)),
        _write_image(tmp_path / 'w1.nii.gz', np.full(SHAPE, 1.2)),
    ]

    values = get_jacobian_values(maps, mask, weight_index=[0, 0, 0, 1])

    # (3 * 0.8 + 1.2) / 4: one value per voxel, not a pooled 0.8/1.2 distribution
    assert values['jacobian_weight_median'] == [pytest.approx(0.9)]
    assert values['jacobian_weight_standard_deviation'] == [pytest.approx(0.0, abs=1e-6)]
    assert values['jacobian_weight_volume_deviation'] == [pytest.approx(1.2 / 0.9 - 1, abs=1e-5)]


def test_series_qc_writes_the_jacobian_columns(tmp_path, monkeypatch):
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=_write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE)),
        jacobian_weights=[_write_image(tmp_path / 'w.nii.gz', np.full(SHAPE, 1.1))],
        jacobian_weight_index=[0, 0],
    )

    assert row['jacobian_weight_mean'] == pytest.approx(1.1)
    assert row['jacobian_weight_volume_deviation'] == 0.0


def test_series_qc_writes_no_jacobian_columns_without_weights(tmp_path, monkeypatch):
    """Test that a run that applied no weights gets no columns, not a map of ones."""
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=_write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE)),
    )

    assert not any(column.startswith('jacobian') for column in row.index)


@pytest.mark.parametrize('ignore', [[], ['jacobian']])
def test_finalize_passes_the_jacobian_weights_to_series_qc(tmp_path, monkeypatch, ignore):
    monkeypatch.setattr(config.workflow, 'ignore', ignore)
    wf = _finalize_wf(tmp_path, monkeypatch, 'shoreline', 'drbuddi', 'dwidenoise')
    connections = _edges(wf)[('dwi_trans_wf', 'series_qc')]

    expected = {
        ('outputnode.jacobian_weights', 'jacobian_weights'),
        ('outputnode.jacobian_weight_index', 'jacobian_weight_index'),
    }
    assert (expected <= connections) == (not ignore)
    assert not ignore or not (expected & connections)


# ---------------------------------------------------------------------------
# b=0 temporal SNR


def test_tsnr_summary_leaves_out_voxels_without_an_estimate(tmp_path):
    """Test that DWITSNR's zeros (no variance, or outside its mask) are not counted."""
    tsnr = np.zeros(SHAPE)
    tsnr[INTERIOR] = 20.0
    tsnr[2, 2, 2:6] = 0.0
    tsnr[2, 3, 2:4] = 30.0
    mask = np.zeros(SHAPE)
    mask[1:-1, 1:-1, 1:-1] = 1

    values = get_tsnr_values(
        _write_image(tmp_path / 'tsnr.nii.gz', tsnr),
        _write_image(tmp_path / 'mask.nii.gz', mask),
    )

    n_values = 8**3 - 4
    assert values['b0_tsnr_median'] == [20.0]
    assert values['b0_tsnr_mean'] == [pytest.approx((20.0 * (n_values - 2) + 60.0) / n_values)]
    assert values['b0_tsnr_p95'] == [20.0]


def test_series_qc_writes_the_tsnr_columns(tmp_path, monkeypatch):
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=_write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE)),
        t1_tsnr_file=_write_image(tmp_path / 'tsnr.nii.gz', np.full(SHAPE, 12.5)),
    )

    assert row['b0_tsnr_median'] == 12.5
    assert row['b0_tsnr_standard_deviation'] == 0.0


def test_series_qc_writes_no_tsnr_columns_from_an_empty_map(tmp_path, monkeypatch):
    """Test that a series with fewer than two b=0 volumes (an all-zero map) gets no columns."""
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=_write_image(tmp_path / 'mask.nii.gz', np.ones(SHAPE)),
        t1_tsnr_file=_write_image(tmp_path / 'tsnr.nii.gz', np.zeros(SHAPE)),
    )

    assert not any(column.startswith('b0_tsnr') for column in row.index)


def test_finalize_passes_the_tsnr_map_to_series_qc(tmp_path, monkeypatch):
    wf = _finalize_wf(tmp_path, monkeypatch, 'shoreline', 'drbuddi', 'dwidenoise')

    assert _edges(wf)[('dwi_derivatives_wf', 'series_qc')] == {('tsnr.out_file', 't1_tsnr_file')}


def test_merge_passes_the_tsnr_map_to_series_qc(tmp_path, monkeypatch):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.distortion_group_merge import init_distortion_group_merge_wf

    monkeypatch.setattr(config.execution, 'output_dir', str(tmp_path / 'out'))
    monkeypatch.setattr(config.workflow, 'denoise_method', 'none')
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)
    wf = init_distortion_group_merge_wf(
        merging_strategy='concat',
        inputs_list=['dir-AP', 'dir-PA'],
        source_file='sub-01_dwi.nii.gz',
        output_prefix='sub-01',
        name='merge_wf',
        resolution=parse_output_spaces(['acpc:res-2mm'])[0].resolution,
    )

    edges = _edges(wf)
    assert edges[('dwi_derivatives_wf', 'series_qc')] == {('tsnr.out_file', 't1_tsnr_file')}
    assert edges[('merged_b0_ref', 'series_qc')] == {('outputnode.dwi_mask', 't1_mask_file')}
