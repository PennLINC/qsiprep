"""Tests for cs_dmri's image QC and its place in the merged QC table."""

from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('cs_dmri')

from qsiprep.interfaces.cs_dmri import CsDmriQC, _voxel_frame_bvecs  # noqa: E402
from qsiprep.interfaces.dsi_studio import (  # noqa: E402
    QC_WARNINGS_COLUMN,
    SRC_QC_MEASURES,
    DSIStudioMergeQC,
)


def _write_series(tmp_path, affine=None):
    """Write a small single-shell series whose voxels share a smooth fiber field."""
    rng = np.random.default_rng(0)
    shape, ndirs = (12, 12, 8), 30
    t = np.pi * np.arange(ndirs) / ndirs
    dirs = np.stack([np.cos(t), np.sin(t), 0.1 * np.ones_like(t)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((2, 3)), dirs])
    bvals = np.r_[0, 0, np.full(ndirs, 1000.0)]
    x = np.linspace(0, np.pi / 2, shape[0])[:, None, None]
    angle = np.broadcast_to(x, shape)
    fiber = np.stack([np.cos(angle), np.sin(angle), np.zeros(shape)], -1)
    signal = 0.6 + 0.4 * np.exp(-3 * np.einsum('xyzk,nk->xyzn', fiber, bvecs) ** 2)
    data = 1000 * signal * (1 + 0.01 * rng.standard_normal(signal.shape))
    affine = np.diag([-2.0, -2.0, 2.0, 1.0]) if affine is None else affine
    dwi = tmp_path / 'dwi.nii.gz'
    nb.Nifti1Image(data.astype('float32'), affine).to_filename(dwi)
    np.savetxt(tmp_path / 'dwi.bval', bvals[None], fmt='%d')
    np.savetxt(tmp_path / 'dwi.bvec', bvecs.T, fmt='%.6f')
    mask = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(np.ones(shape, 'uint8'), affine).to_filename(mask)
    return dwi, tmp_path / 'dwi.bval', tmp_path / 'dwi.bvec', mask


def _run(tmp_path, **inputs):
    interface = CsDmriQC(**inputs)
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    table = pd.read_csv(interface._results['qc_file'])
    return table, interface._results.get('warning')


def test_cs_dmri_qc_measures_a_series(tmp_path):
    import cs_dmri as cs

    dwi, bval, bvec, mask = _write_series(tmp_path)
    table, warning = _run(
        tmp_path, dwi_file=str(dwi), bval_file=str(bval), bvec_file=str(bvec), mask_file=str(mask)
    )

    assert warning is None
    assert list(table.columns) == cs.qc.columns()
    assert table['ndc'][0] > 0.9
    assert table['n_dwi_volumes'][0] == 30
    assert table['n_b0_volumes'][0] == 2
    assert table['dimension_x'][0] == 12
    assert np.isfinite(table['fixel_coherence'][0])


def test_cs_dmri_qc_failure_is_na_with_every_column(tmp_path):
    """QC must not fail the run; the table keeps its columns."""
    import cs_dmri as cs

    dwi, bval, _, _ = _write_series(tmp_path)
    short = tmp_path / 'short.bvec'
    np.savetxt(short, np.zeros((3, 5)))
    table, warning = _run(tmp_path, dwi_file=str(dwi), bval_file=str(bval), bvec_file=str(short))

    assert warning.startswith('cs_dmri QC failed:')
    assert list(table.columns) == cs.qc.columns()
    assert table.isna().all(axis=None)


def test_fsl_bvecs_are_flipped_only_for_neurological_images():
    bvecs = np.array([[0.6, 0.8, 0.0]])
    neurological = np.diag([2.0, 2.0, 2.0, 1.0])
    radiological = np.diag([-2.0, 2.0, 2.0, 1.0])

    assert _voxel_frame_bvecs(bvecs, neurological, 'FSL')[0, 0] == pytest.approx(-0.6)
    assert _voxel_frame_bvecs(bvecs, radiological, 'FSL')[0, 0] == pytest.approx(0.6)
    assert _voxel_frame_bvecs(bvecs, neurological, 'DIPY')[0, 0] == pytest.approx(0.6)


#: A real SRC QC row, from the passing forrest_gump run.
_SRC_QC = (
    'file name\tdimension\tresolution\tdwi count(b0/dwi)\tmax b-value\t'
    'DWI contrast\tneighboring DWI correlation\t'
    'neighboring DWI correlation(masked)\t#bad slices\n'
    'scaled_merged\t32 39 34 \t5 5 5 \t1/32\t800.000000\t1.054243\t0.996735\t0.991714\t0\t\t\n'
)


def _merge(tmp_path, src_contents, cs_row, **inputs):
    src = tmp_path / 'src_qc.txt'
    src.write_text(src_contents)
    fib = tmp_path / 'fib_qc.txt'
    fib.write_text('header\ncoherence\t0.42\n')
    cs_qc = tmp_path / 'cs_dmri_qc.csv'
    pd.DataFrame({k: [v] for k, v in cs_row.items()}).to_csv(cs_qc, index=False)
    interface = DSIStudioMergeQC(src_qc=str(src), fib_qc=str(fib), cs_dmri_qc=str(cs_qc), **inputs)
    interface._run_interface(SimpleNamespace(cwd=str(tmp_path)))
    return pd.read_csv(interface._results['qc_file'], keep_default_na=False)


_CS_ROW = {
    'dimension_x': 99,
    'max_b': 999.0,
    'ndc': 0.95,
    'dwi_contrast_ratio': 1.4,
    'fixel_coherence': 0.6,
}


def test_merge_appends_cs_dmri_measures_and_keeps_dsi_studio_values(tmp_path):
    table = _merge(tmp_path, _SRC_QC, _CS_ROW)

    assert table['neighbor_corr'][0] == pytest.approx(0.996735)
    assert table['ndc'][0] == pytest.approx(0.95)
    assert table['fixel_coherence'][0] == pytest.approx(0.6)
    # Shared header facts: DSI Studio's value stands.
    assert table['dimension_x'][0] == 32
    assert table['max_b'][0] == pytest.approx(800.0)
    assert list(table.columns)[: len(SRC_QC_MEASURES)] == list(SRC_QC_MEASURES)
    assert table[QC_WARNINGS_COLUMN][0] == ''


def test_merge_fills_shared_columns_from_cs_dmri_when_dsi_studio_failed(tmp_path):
    table = _merge(tmp_path, '', _CS_ROW, src_qc_warning='DSI Studio exited with status 1.')

    assert table['dimension_x'][0] == 99
    assert table['neighbor_corr'][0] == ''


def test_merge_reports_a_cs_dmri_failure(tmp_path):
    table = _merge(
        tmp_path,
        _SRC_QC,
        dict.fromkeys(_CS_ROW, np.nan),
        cs_dmri_qc_warning='cs_dmri QC failed: ValueError: bad bvecs',
    )

    assert table[QC_WARNINGS_COLUMN][0] == 'cs_dmri QC failed: ValueError: bad bvecs'
    assert table['neighbor_corr'][0] == pytest.approx(0.996735)
    assert table['ndc'][0] == ''
