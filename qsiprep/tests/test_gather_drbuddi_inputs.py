"""Tests for GatherDRBUDDIInputs' epi (fieldmap b=0) mode.

The blip-up image must be built from the motion-corrected volumes (the frame
DRBUDDI's warps are applied in), and the blip-down image must come only from
opposite-PE fieldmap candidates -- borrowed same-PE b=0s belong to TOPUP.
"""

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from qsiprep.interfaces.tortoise import GatherDRBUDDIInputs

LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])
SHAPE = (5, 6, 4)


def _write_nii(path, arrays):
    data = np.stack(arrays, axis=-1) if isinstance(arrays, list) else arrays
    nb.Nifti1Image(data.astype('f4'), LPS_AFFINE).to_filename(str(path))
    return str(path)


def _write_gradients(path_stem, bval):
    bval_file = f'{path_stem}.bval'
    bvec_file = f'{path_stem}.bvec'
    with open(bval_file, 'w') as f:
        f.write(f'{bval}\n')
    with open(bvec_file, 'w') as f:
        f.write('0\n0\n0\n' if bval == 0 else '1\n0\n0\n')
    return bval_file, bvec_file


@pytest.fixture
def epi_inputs(tmp_path):
    rng = np.random.default_rng(0)
    vols = rng.uniform(100, 200, size=SHAPE + (4,)).astype('f4')

    dwi_files, bval_files, bvec_files = [], [], []
    for i, bval in enumerate([0, 0, 1000, 1000]):
        dwi_files.append(_write_nii(tmp_path / f'corrected{i}.nii.gz', vols[..., i]))
        bv, bvec = _write_gradients(tmp_path / f'corrected{i}', bval)
        bval_files.append(bv)
        bvec_files.append(bvec)

    ap_origin = _write_nii(tmp_path / 'sub-01_dir-AP_dwi.nii.gz', [vols[..., i] for i in range(4)])
    pa_fmap = _write_nii(
        tmp_path / 'sub-01_dir-PA_epi.nii.gz',
        [rng.uniform(300, 400, size=SHAPE), rng.uniform(300, 400, size=SHAPE)],
    )
    # 4D like the PA file: load_epi_dwi_fieldmaps chokes on mixed 3D/4D lists.
    ap_donor = _write_nii(
        tmp_path / 'sub-02_dir-AP_epi.nii.gz', [rng.uniform(500, 600, size=SHAPE)]
    )

    sidecars = {
        ap_origin: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05},
        pa_fmap: {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05},
        ap_donor: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05},
    }
    return {
        'dwi_files': dwi_files,
        'bval_files': bval_files,
        'bvec_files': bvec_files,
        'original_files': [ap_origin] * 4,
        'pa_fmap': pa_fmap,
        'ap_donor': ap_donor,
        'sidecars': sidecars,
        'expected_up': vols[..., :2].mean(axis=-1),
    }


def _gather(epi_inputs, tmp_path, epi_fmaps):
    iface = GatherDRBUDDIInputs(
        dwi_files=epi_inputs['dwi_files'],
        bval_files=epi_inputs['bval_files'],
        bvec_files=epi_inputs['bvec_files'],
        original_files=epi_inputs['original_files'],
        epi_fmaps=epi_fmaps,
        fieldmap_type='epi',
        dwi_series_pedir='j',
        sidecars=epi_inputs['sidecars'],
    )
    return iface.run(cwd=str(tmp_path / 'work'))


def test_blip_up_is_mean_of_corrected_b0s(epi_inputs, tmp_path, monkeypatch):
    # Stand in for the TORTOISE binary: pick the second down candidate.
    def fake_report(b0_files, prefix, num_threads=1):
        n = len(b0_files)
        return pd.DataFrame(
            {
                'volume_index': range(n),
                'mean_cc': np.linspace(0.4, 0.5, n),
                'translation_total_mm': 0.1,
                'rotation_total_deg': 0.2,
                'selected': [int(i == 1) for i in range(n)],
            }
        )

    monkeypatch.setattr('qsiprep.interfaces.tortoise.select_best_b0_report', fake_report)

    (tmp_path / 'work').mkdir()
    result = _gather(epi_inputs, tmp_path, [epi_inputs['pa_fmap'], epi_inputs['ap_donor']])

    up = nb.load(result.outputs.blip_up_image)
    assert np.allclose(up.get_fdata(), epi_inputs['expected_up'], atol=1e-3)

    # The down image is the PA candidate the picker selected (index 1),
    # never the same-PE donor.
    down = nb.load(result.outputs.blip_down_image).get_fdata()
    pa_vols = np.asanyarray(nb.load(epi_inputs['pa_fmap']).dataobj)
    assert np.allclose(down, pa_vols[..., 1], atol=1e-3)

    assert len(result.outputs.blip_assignments) == 4


def test_same_pe_donor_alone_is_rejected(epi_inputs, tmp_path):
    (tmp_path / 'work').mkdir()
    with pytest.raises(Exception, match='No j- b=0 images'):
        _gather(epi_inputs, tmp_path, [epi_inputs['ap_donor']])
