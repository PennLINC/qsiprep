"""Test the in-brain gradient nonlinearity measures in desc-image_qc.tsv."""

from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.gradunwarp import graddev_rotation_degrees, gradient_nonlinearity_qc

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SHAPE = (6, 6, 6)


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _save(data, path, affine=AFFINE):
    nb.Nifti1Image(np.asarray(data), affine).to_filename(path)
    return str(path)


def _brain():
    brain = np.zeros(SHAPE, dtype=np.uint8)
    brain[1:5, 1:5, 1:5] = 1
    return brain


def _inputs(tmp_path):
    """Write a brain where everything inside rotates 2 deg and moves 1.5 mm; outside, far more."""
    brain = _brain()
    tensors = np.zeros(SHAPE + (9,))
    for idx in np.ndindex(SHAPE):
        tensors[idx] = _rotation([0, 0, 1], 2.0 if brain[idx] else 30.0).ravel()
    displacement = np.where(brain > 0, 1.5, 9.0)
    field = np.zeros(SHAPE + (1, 3))
    field[..., 0, 0] = displacement * 0.6
    field[..., 0, 1] = displacement * 0.8
    return {
        'grad_dev': _save(tensors, tmp_path / 'graddev.nii.gz'),
        'mask': _save(brain, tmp_path / 'mask.nii.gz'),
        'field': _save(field, tmp_path / 'field.nii.gz'),
    }


def test_rotation_angle_ignores_scaling_shear_and_transposition():
    rotation = _rotation([0.6, 0.0, 0.8], 3.0)
    stretch = np.array([[1.04, 0.01, 0.0], [0.01, 0.97, 0.02], [0.0, 0.02, 1.01]])
    tensors = np.stack([np.eye(3), rotation, rotation @ stretch, (rotation @ stretch).T])

    np.testing.assert_allclose(graddev_rotation_degrees(tensors), [0, 3, 3, 3], atol=1e-6)


def test_rotation_angle_of_an_improper_tensor_is_finite():
    reflected = np.diag([1.0, 1.0, -1.0])
    assert graddev_rotation_degrees(reflected[None])[0] == pytest.approx(0.0)


def test_measures_count_only_brain_voxels(tmp_path):
    files = _inputs(tmp_path)
    values = gradient_nonlinearity_qc(
        files['grad_dev'],
        files['mask'],
        gradwarp_field=files['field'],
        gradwarp_mask=files['mask'],
    )

    assert sorted(values) == sorted(
        f'{measure}_{stat}'
        for measure in ('gradwarp_displacement_mm', 'graddev_rotation_deg')
        for stat in ('median', 'p95', 'maximum')
    )
    for stat in ('median', 'p95', 'maximum'):
        assert values[f'gradwarp_displacement_mm_{stat}'][0] == pytest.approx(1.5)
        assert values[f'graddev_rotation_deg_{stat}'][0] == pytest.approx(2.0, abs=1e-4)


def test_no_field_gives_no_displacement_columns(tmp_path):
    files = _inputs(tmp_path)
    values = gradient_nonlinearity_qc(files['grad_dev'], files['mask'])

    assert not any(key.startswith('gradwarp_displacement') for key in values)


def test_a_mask_on_another_grid_is_resampled(tmp_path):
    tensors = np.tile(np.eye(3).ravel(), SHAPE + (1,))
    fine_mask = _save(np.ones((12, 12, 12), dtype=np.uint8), tmp_path / 'm.nii.gz', np.eye(4))

    values = gradient_nonlinearity_qc(_save(tensors, tmp_path / 'g.nii.gz'), fine_mask)

    assert values['graddev_rotation_deg_maximum'][0] == pytest.approx(0.0, abs=1e-6)


def _series_qc_row(tmp_path, monkeypatch, **inputs):
    import pandas as pd

    import qsiprep.interfaces.reports as reports_mod
    from qsiprep.interfaces.reports import SeriesQC

    monkeypatch.setattr(reports_mod, 'calculate_motion_summary', lambda _: {'mean_fd': [0.1]})
    pre_qc = tmp_path / 'pre.csv'
    pd.DataFrame({'neighbor_corr': [0.9]}).to_csv(pre_qc, index=False)
    confounds = tmp_path / 'confounds.tsv'
    confounds.touch()
    interface = SeriesQC(
        pre_qc=str(pre_qc),
        confounds_file=str(confounds),
        output_file_name='sub-01_dwi',
        **inputs,
    )
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    return pd.read_csv(interface._results['series_qc_file'], sep='\t').iloc[0]


def test_series_qc_adds_the_columns_with_a_deviation_map(tmp_path, monkeypatch):
    files = _inputs(tmp_path)
    row = _series_qc_row(
        tmp_path,
        monkeypatch,
        t1_mask_file=files['mask'],
        grad_dev_file=files['grad_dev'],
        gradwarp_field=files['field'],
        gradwarp_mask_file=files['mask'],
    )

    assert row['gradwarp_displacement_mm_p95'] == pytest.approx(1.5)
    assert row['graddev_rotation_deg_median'] == pytest.approx(2.0, abs=1e-4)


def test_series_qc_has_no_columns_without_a_deviation_map(tmp_path, monkeypatch):
    files = _inputs(tmp_path)
    row = _series_qc_row(tmp_path, monkeypatch, t1_mask_file=files['mask'])

    assert not any(key.startswith(('gradwarp_', 'graddev_')) for key in row.index)
