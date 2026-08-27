"""Tests for the TORTOISE SelectBestB0-based b=0 picking.

``get_best_b0_topup_inputs_from`` ranks candidates per distortion group with
the binary's registered windowed-correlation scores (higher = better) and
carries the recovered rigid parameters into the QC tsv.
"""

import shutil

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from qsiprep.interfaces import epi_fmap
from qsiprep.interfaces.epi_fmap import get_best_b0_topup_inputs_from, select_best_b0_report

# LAS, so the imain assembly's to_lps(..., ('L', 'A', 'S')) is a no-op
LAS_AFFINE = np.diag([-1.0, 1.0, 1.0, 1.0])
SHAPE = (6, 5, 4)
TRT = 0.05


def _write_nii(path, arrays):
    data = np.stack(arrays, axis=-1)
    nb.Nifti1Image(data.astype('f4'), LAS_AFFINE).to_filename(str(path))
    return str(path)


@pytest.fixture
def topup_inputs(tmp_path):
    rng = np.random.default_rng(1)
    dwi_vols = [rng.uniform(100, 200, size=SHAPE) for _ in range(4)]
    ap_dwi = _write_nii(tmp_path / 'sub-01_dir-AP_dwi.nii.gz', dwi_vols)
    bval_file = str(tmp_path / 'sub-01_dir-AP_dwi.bval')
    with open(bval_file, 'w') as f:
        f.write('0 1000 0 1000\n')

    pa_vols = [rng.uniform(100, 200, size=SHAPE) for _ in range(2)]
    pa_fmap = _write_nii(tmp_path / 'sub-01_dir-PA_epi.nii.gz', pa_vols)

    sidecars = {
        ap_dwi: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': TRT},
        pa_fmap: {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': TRT},
    }
    return {
        'dwi_file': ap_dwi,
        'bval_file': bval_file,
        'pa_fmap': pa_fmap,
        'sidecars': sidecars,
        'dwi_vols': dwi_vols,
        'pa_vols': pa_vols,
    }


def test_topup_inputs_ranked_by_tortoise_scores(topup_inputs, tmp_path, monkeypatch):
    def fake_report(b0_files, prefix, num_threads=1):
        n = len(b0_files)
        # Groups are scored separately; 'spec-00_' is the j- group (sorts
        # first), 'spec-01_' the j group. Make the j group prefer its SECOND
        # b=0 and the j- group its first.
        mean_cc = [0.3, 0.9] if 'spec-01_' in prefix else [0.8, 0.2]
        return pd.DataFrame(
            {
                'volume_index': range(n),
                'mean_cc': mean_cc[:n],
                'translation_total_mm': np.linspace(0.1, 0.3, n),
                'rotation_total_deg': np.linspace(0.2, 0.6, n),
                'selected': [int(i == int(np.argmax(mean_cc[:n]))) for i in range(n)],
            }
        )

    monkeypatch.setattr(epi_fmap, 'select_best_b0_report', fake_report)

    cwd = tmp_path / 'work'
    cwd.mkdir()
    datain_file, imain_file, _, b0_tsv, _, _ = get_best_b0_topup_inputs_from(
        dwi_file=topup_inputs['dwi_file'],
        bval_file=topup_inputs['bval_file'],
        b0_threshold=100,
        cwd=str(cwd),
        bids_origin_files=[topup_inputs['dwi_file']] * 4,
        epi_fmaps=[topup_inputs['pa_fmap']],
        max_per_spec=1,
        topup_requested=True,
        sidecars=topup_inputs['sidecars'],
    )

    # One image per spec; the first must share the first b=0's spec (j).
    with open(datain_file) as f:
        datain = f.read().splitlines()
    assert len(datain) == 2
    assert datain[0].startswith('0 1 0')
    assert datain[1].startswith('0 -1 0')

    # The j group's winner is its second b=0 (volume 2 of the series).
    imain = nb.load(imain_file)
    assert imain.shape[3] == 2
    assert np.allclose(imain.dataobj[..., 0], topup_inputs['dwi_vols'][2], atol=1e-3)
    assert np.allclose(imain.dataobj[..., 1], topup_inputs['pa_vols'][0], atol=1e-3)

    # Scores and rigid parameters land in the QC tsv, higher score = selected.
    qc = pd.read_csv(b0_tsv, sep='\t')
    assert len(qc) == 4
    assert {'qc_score', 'translation_total_mm', 'rotation_total_deg'} <= set(qc.columns)
    for _, spec_df in qc.groupby('fsl_spec'):
        assert spec_df.loc[spec_df.selected_for_sdc, 'qc_score'].max() == spec_df.qc_score.max()


@pytest.mark.skipif(
    shutil.which('SelectBestB0') is None, reason='TORTOISE SelectBestB0 not installed'
)
def test_select_best_b0_report_runs_binary(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.uniform(100, 200, size=(24, 24, 16))
    candidates = []
    for i, noise in enumerate([2.0, 2.0, 80.0]):
        img = nb.Nifti1Image((base + rng.normal(0, noise, base.shape)).astype('f4'), np.eye(4))
        path = tmp_path / f'b0_{i}.nii.gz'
        img.to_filename(str(path))
        candidates.append(str(path))

    report = select_best_b0_report(candidates, prefix=str(tmp_path / 'pick_'))
    assert len(report) == 3
    assert {'mean_cc', 'selected', 'translation_total_mm', 'rotation_total_deg'} <= set(
        report.columns
    )
    assert report['selected'].sum() == 1
    # The heavy-noise outlier scores lowest and is never the pick.
    assert report['mean_cc'].idxmin() == 2
    assert report.loc[2, 'selected'] == 0
