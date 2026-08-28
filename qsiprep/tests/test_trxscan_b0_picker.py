"""Ground-truth b=0 selection test on TRXScan-simulated data.

The borrowing scenario the picker was built for, with physics instead of
mocks: a noisy native AP run, a cleaner donor AP run simulated at a known
head pose (so its b=0s are mutually consistent and outscore the natives --
the exact eviction pressure the native-first guarantee exists to resist),
and a PA pair for the reverse blip group. Assertions grade the picker
against the simulation's ground truth: natives keep the slots, the donors'
recovered rigid parameters match the applied pose, and the PA candidates
fill the opposite spec group.

The simulation uses the bundled fieldmap scaled to 0.2x: at full HBCD-like
strength (~9 voxels of PE shift) a head moving under the scanner-fixed field
changes its apparent pose, and rigid registration honestly recovers only
60-80% of the applied motion. The weak field keeps that coupling below the
assertion tolerances while still exercising real EPI distortion.

The streamlines are subsampled to 400k (weight-proportional, seeded -- the
same subset in every run) to keep the three simulations under a minute; the
measured ground-truth cost of that subset is FA p99 |delta| ~= 0.08 against
the full tractogram.

Runs only where trxscan, SelectBestB0 and the reference kit are available
(the in-container CI jobs); skips elsewhere.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from qsiprep.interfaces.epi_fmap import get_best_b0_topup_inputs_from

KIT = Path(os.environ.get('TRXSCAN_DATA', '/opt/trxscan/data'))
TRT = 0.0917  # HBCD readout, per the kit's EPI timing

APPLIED_TRANS = (1.0, 2.5, 1.0)  # mm, donor pose
APPLIED_TRANS_NORM = float(np.linalg.norm(APPLIED_TRANS))
APPLIED_ROT_X_RAD = 0.034907  # 2.0 degrees
APPLIED_ROT_DEG = 2.0

pytestmark = pytest.mark.skipif(
    shutil.which('trxscan') is None or shutil.which('SelectBestB0') is None or not KIT.exists(),
    reason='TRXScan, SelectBestB0 and the reference kit are only in the container',
)


def _run(cmd):
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f'{cmd[0]} failed:\n{proc.stdout}\n{proc.stderr}'


@pytest.fixture(scope='session')
def borrowing_sim(tmp_path_factory):
    work = tmp_path_factory.mktemp('trxscan_b0_picker')
    grid = work / 'grid'
    _run(
        [
            sys.executable,
            KIT / 'scripts' / 'prepare_acquisition_grid.py',
            '--anat-dir',
            KIT / 'sub-0001a' / 'anat',
            '--prefix',
            'sub-0001a_space-ACPC',
            '--out',
            grid,
            '--voxel',
            '3.0',
        ]
    )
    fmap = nb.load(grid / 'fmap_hz.nii.gz')
    nb.Nifti1Image(np.asanyarray(fmap.dataobj) * 0.2, fmap.affine, fmap.header).to_filename(
        grid / 'fmap_weak.nii.gz'
    )

    (work / 'ap3.bval').write_text('0 1000 0\n')
    (work / 'ap3.bvec').write_text('0 1 0\n0 0 0\n0 0 0\n')
    (work / 'b0s.bval').write_text('0 0\n')
    (work / 'b0s.bvec').write_text('0 0\n0 0\n0 0\n')
    tx, ty, tz = APPLIED_TRANS
    pose = f'{tx}\t{ty}\t{tz}\t{APPLIED_ROT_X_RAD}\t0\t0\n'
    (work / 'donor_motion.tsv').write_text(
        'trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n' + pose + pose
    )

    common = [
        'trxscan',
        '--wm',
        grid / 'wm.nii.gz',
        '--gm',
        grid / 'gm.nii.gz',
        '--csf',
        grid / 'csf.nii.gz',
        '--mask',
        grid / 'mask.nii.gz',
        '--streamlines',
        KIT / 'sub-0001a' / 'tract' / 'sub-0001a_space-ACPC_desc-actsift2_tracks.trx',
        '--fmap',
        grid / 'fmap_weak.nii.gz',
        '--weights',
        'sift2_weights',
        '--kappa',
        '15',
        '--params',
        'adult',
        '--seed',
        '0',
        '--subsample',
        '400000',
    ]
    ap = ['--bval', work / 'ap3.bval', '--bvec', work / 'ap3.bvec']
    b0s = ['--bval', work / 'b0s.bval', '--bvec', work / 'b0s.bvec']
    _run(common + ap + ['--out', work / 'sub-01_dir-AP_run-01', '--noise', '9'])
    _run(
        common
        + b0s
        + [
            '--out',
            work / 'sub-01_dir-AP_run-05',
            '--noise',
            '1',
            '--motion',
            work / 'donor_motion.tsv',
        ]
    )
    _run(common + b0s + ['--out', work / 'sub-01_dir-PA_run-01', '--noise', '9', '--reverse-pe'])

    def mag(stem):
        return str(work / f'{stem}_part-mag_dwi.nii.gz')

    return {
        'work': work,
        'native': mag('sub-01_dir-AP_run-01'),
        'native_bval': str(work / 'sub-01_dir-AP_run-01_dwi.bval'),
        'donor': mag('sub-01_dir-AP_run-05'),
        'pa': mag('sub-01_dir-PA_run-01'),
    }


def test_native_first_against_simulated_ground_truth(borrowing_sim):
    sim = borrowing_sim
    sidecars = {
        sim['native']: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': TRT},
        sim['donor']: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': TRT},
        sim['pa']: {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': TRT},
    }
    cwd = sim['work'] / 'pick'
    cwd.mkdir(exist_ok=True)
    datain_file, _, _, b0_tsv, _, _ = get_best_b0_topup_inputs_from(
        dwi_file=sim['native'],
        bval_file=sim['native_bval'],
        b0_threshold=100,
        cwd=str(cwd),
        bids_origin_files=[sim['native']] * 3,
        epi_fmaps=[sim['donor'], sim['pa']],
        max_per_spec=2,
        topup_requested=True,
        sidecars=sidecars,
        num_threads=4,
    )

    qc = pd.read_csv(b0_tsv, sep='\t')
    natives = qc[qc.is_native]
    donors = qc[(~qc.is_native) & (qc.bids_origin_file == sim['donor'])]
    pa = qc[qc.bids_origin_file == sim['pa']]

    # The donors are cleaner and mutually consistent, so they outscore the
    # natives -- and must still lose the slots.
    assert donors.qc_score.min() > natives.qc_score.max()
    assert natives.selected_for_sdc.all()
    assert not donors.selected_for_sdc.any()
    assert pa.selected_for_sdc.all()

    # The donors' recovered rigid parameters match the simulated pose.
    assert np.allclose(donors.translation_total_mm, APPLIED_TRANS_NORM, atol=0.5)
    assert np.allclose(donors.rotation_total_deg, APPLIED_ROT_DEG, atol=0.5)
    # Everything that did not move stays at the registration noise floor.
    unmoved = pd.concat([natives, pa])
    assert (unmoved.translation_total_mm < 0.8).all()
    assert (unmoved.rotation_total_deg < 0.8).all()

    # TOPUP's first input shares the native spec.
    with open(datain_file) as f:
        assert f.readline().startswith('0 1 0')
