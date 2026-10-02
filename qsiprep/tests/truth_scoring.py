"""Score a qsiprep run against the ground truth a TRXScan fixture carries.

TRXScan (https://github.com/PennLINC/TRXScan) simulates DWI from a tractogram phantom and
writes, under ``derivatives/trxscan``, the artifact-free b=0 (``desc-cleanb0``), the applied
off-resonance field (``desc-fieldmap``, Hz) and its displacement (``desc-displacement``, RAS mm),
the fibre peaks (``desc-truthpeaks``), gradient-nonlinearity fields (``desc-gnlgraddev``), the
head poses applied per volume (``desc-motion_timeseries.tsv``) and, when the fixture moved the
subject between scans, the true rigid transforms (``*_desc-truth_xfm.txt``).

This module maps that truth into qsiprep's ACPC output space with qsiprep's own transforms and
compares. It needs only numpy, scipy and nibabel (h5py for ``.h5`` composite transforms), so
it runs inside the test image without TRXScan installed.

Conventions: ITK transform files map output (fixed) points to input (moving) points in LPS;
``from-distortiongroup_to-ACPC`` therefore maps ACPC points to the raw DWI frame.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nb
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

LPS = np.diag([-1.0, -1.0, 1.0])


# ─── transforms ─────────────────────────────────────────────────────────────


def _euler_zxy(ax, ay, az):
    """ITK ``Euler3DTransform`` with ``ComputeZYX`` off: ``R = Rz Rx Ry``."""
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


def _itk_matrix(name, params, fixed):
    p = np.asarray(params, dtype=np.float64).ravel()
    c = np.asarray(fixed, dtype=np.float64).ravel()[:3]
    if name.startswith('Euler3D') or p.size == 6:
        mat, t = _euler_zxy(*p[:3]), p[3:6]
    elif p.size == 12:
        mat, t = p[:9].reshape(3, 3), p[9:12]
    elif p.size == 3:
        mat, t = np.eye(3), p[:3]
    else:
        raise ValueError(f'unsupported ITK transform {name!r} with {p.size} parameters')
    out = np.eye(4)
    out[:3, :3] = mat
    out[:3, 3] = t + c - mat @ c
    return out


def read_itk_transform(path):
    """Read an ITK ``.txt``, ``.mat`` or ``.h5`` transform as a 4x4 LPS point map.

    The map takes output (fixed) points to input (moving) points, ITK's resampling convention.
    """
    path = Path(path)
    if path.suffix == '.txt':
        lines = path.read_text().splitlines()
        kinds = [ln.split(':', 1)[1].strip() for ln in lines if ln.startswith('Transform:')]
        params = [
            np.array(ln.split(':', 1)[1].split(), float)
            for ln in lines
            if ln.startswith('Parameters:')
        ]
        fixed = [
            np.array(ln.split(':', 1)[1].split(), float)
            for ln in lines
            if ln.startswith('FixedParameters:')
        ]
        out = np.eye(4)
        for k, p, f in zip(kinds, params, fixed, strict=True):
            out = out @ _itk_matrix(k, p, f)
        return out
    if path.suffix == '.mat':
        m = loadmat(str(path))
        key = next(k for k in m if not k.startswith('__') and k != 'fixed')
        return _itk_matrix(key, m[key], m.get('fixed', np.zeros(3)))
    if path.suffix == '.h5':
        import h5py

        out = np.eye(4)
        with h5py.File(path) as h:
            for g in sorted(k for k in h['TransformGroup'] if k != '0'):
                grp = h['TransformGroup'][g]
                kind = grp['TransformType'][()][0].decode()
                out = out @ _itk_matrix(
                    kind, grp['TransformParameters'][()], grp['TransformFixedParameters'][()]
                )
        return out
    raise ValueError(f'unknown ITK transform format {path.suffix!r}')


def world_to_lps_map(T_ras):
    """Re-express a 4x4 RAS point map (what the fixture records) in LPS."""
    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    return F @ np.asarray(T_ras, dtype=np.float64) @ F


def rigid_error(estimate, truth, center=(0.0, 0.0, 0.0)):
    """Compute the rotation (deg) of ``estimate @ inv(truth)`` and the shift (mm) of ``center``."""
    e, t = np.asarray(estimate, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    d = e[:3, :3] @ np.linalg.inv(t[:3, :3])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    c = np.append(np.asarray(center, dtype=np.float64), 1.0)
    return {
        'rotation_deg': ang,
        'translation_mm': float(np.linalg.norm((e @ c)[:3] - (t @ c)[:3])),
    }


# ─── helpers ────────────────────────────────────────────────────────────────


def _sample(img, pts_ras, order=1):
    vox = np.linalg.inv(img.affine) @ np.vstack([pts_ras, np.ones(pts_ras.shape[1])])
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim == 3:
        return map_coordinates(data, vox[:3], order=order, mode='constant')
    return np.stack(
        [
            map_coordinates(data[..., c], vox[:3], order=order, mode='constant')
            for c in range(data.shape[-1])
        ],
        -1,
    )


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def _corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def _find(folders, pattern, prefer=None):
    for d in folders:
        hits = sorted(Path(d).glob(pattern))
        if hits:
            return next((h for h in hits if prefer and prefer in h.name), hits[0])
    return None


# ─── scoring ────────────────────────────────────────────────────────────────


def score_run(bids_root, output_dir, series=None):
    """Score one qsiprep output against its TRXScan truth.

    Parameters
    ----------
    bids_root : path
        The fixture (raw BIDS root with ``derivatives/trxscan``).
    output_dir : path
        qsiprep's output directory (holding ``sub-*/``).
    series : str, optional
        The ``dir-`` label of the series whose truth to use (default: the series qsiprep took
        as its reference, the first in the confounds).

    Returns
    -------
    dict
        ``coreg_error`` (vs identity or the fixture's recorded movement), ``sdc`` (estimated
        vs true displacement along the phase-encode axis: corr, slope, rms residual and the
        uncorrected rms), ``b0_corrected_vs_clean`` / ``b0_uncorrected_vs_clean``,
        ``fd_mean_mm``, and when present ``gnl_graddev`` and ``motion``.
    """
    bids, out = Path(bids_root), Path(output_dir)
    sub_dir = next(out.glob('sub-*/dwi'))
    sub = sub_dir.parent.name
    truth_dir = next((bids / 'derivatives' / 'trxscan').glob(f'{sub}/**/dwi'))
    anat_truth = next((bids / 'derivatives' / 'trxscan').glob(f'{sub}/**/anat'), None)
    report = {}

    conf_files = sorted(sub_dir.glob('*_desc-confounds_timeseries.tsv'))
    if series is None:
        rows = _read_tsv(conf_files[0])
        series = 'AP' if 'dir-AP' in rows[0].get('original_file', '') else 'PA'
    report['series'] = series
    xfm_path = _find([sub_dir], '*from-distortiongroup_to-ACPC*_xfm.mat', prefer=f'dir-{series}')
    coreg = read_itk_transform(xfm_path)
    A, offset = coreg[:3, :3], coreg[:3, 3]

    ref = nb.load(
        _find(
            [sub_dir, sub_dir.parent / 'anat'],
            '*_space-ACPC_desc-preproc_dwiref.nii.gz',
            f'dir-{series}',
        )
    )
    mask = (
        np.asarray(
            nb.load(
                _find(
                    [sub_dir, sub_dir.parent / 'anat'],
                    '*_space-ACPC_desc-brain_mask.nii.gz',
                    f'dir-{series}',
                )
            ).dataobj
        )
        > 0
    )
    ijk = np.indices(ref.shape[:3]).reshape(3, -1)
    p_ras = ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4]
    q_ras = LPS @ (A @ (LPS @ p_ras) + offset[:, None])  # the same points in the raw DWI frame
    centre = LPS @ ref.affine[:3, :3] @ (np.array(ref.shape[:3]) / 2) + LPS @ ref.affine[:3, 3]
    m = mask.ravel()

    truths = sorted(truth_dir.glob('*desc-cleanb0_dwi.nii.gz'))
    clean_p = next((t for t in truths if f'dir-{series}' in t.name), truths[0])
    stem = clean_p.name.replace('_desc-cleanb0_dwi.nii.gz', '')
    report['stem'] = stem

    # coregistration against the true DWI->ACPC map: the T1w->ACPC map qsiprep found, composed
    # with the subject's recorded movements (T1w vs DWI, and the reference series vs the first run)
    anat_xfm = _find([sub_dir.parent / 'anat'], '*from-anat_to-ACPC_mode-image_xfm.mat')
    if anat_xfm is not None:
        acpc = read_itk_transform(anat_xfm)
        T = np.eye(4)
        t1_truth = (
            _find([anat_truth], '*from-T1w_to-dwi*desc-truth_xfm.json') if anat_truth else None
        )
        if t1_truth:
            T = world_to_lps_map(np.array(_load_json(t1_truth)['WorldTransformRAS']))
        series_truth = _find(
            [truth_dir], f'*dir-{series}_from-dir{series}_to-dwi*desc-truth_xfm.json'
        )
        if series_truth:
            T = T @ np.linalg.inv(
                world_to_lps_map(np.array(_load_json(series_truth)['WorldTransformRAS']))
            )
        true_map = np.linalg.inv(T) @ acpc
        report['coreg_error'] = {
            **rigid_error(coreg, true_map, centre),
            'truth': 'movement' if (t1_truth or series_truth) else 'identity',
        }

    # susceptibility displacement: qsiprep's export (ACPC, LPS vectors) vs truth (raw frame, RAS)
    est_p = _find([sub_dir], '*_space-ACPC_desc-sdc_displacement.nii.gz', f'dir-{series}')
    disp_p = truth_dir / f'{stem}_desc-displacement_dwi.nii.gz'
    if est_p is not None and disp_p.exists():
        est = np.asarray(nb.load(est_p).dataobj, dtype=np.float64).reshape(-1, 3)
        true_raw = _sample(nb.load(disp_p), q_ras)
        true_acpc = (np.linalg.inv(A) @ (LPS @ true_raw.T)).T
        pe = np.linalg.inv(A) @ (LPS @ nb.load(clean_p).affine[:3, 1])
        pe /= np.linalg.norm(pe)
        e, t = (est @ pe)[m], (true_acpc @ pe)[m]
        report['sdc'] = {
            'corr': _corr(e, t),
            'slope': float(e @ t / (t @ t)) if t @ t > 0 else float('nan'),
            'rms_residual': float(np.sqrt(np.mean((e - t) ** 2))),
            'rms_truth': float(np.sqrt(np.mean(t**2))),
        }

    # the corrected reference vs the artifact-free b=0, and the raw b=0 through the same map
    clean = _sample(nb.load(clean_p), q_ras)
    report['b0_corrected_vs_clean'] = _corr(np.asarray(ref.dataobj).ravel()[m], clean[m])
    raws = sorted((bids / sub / 'dwi').glob('*_dwi.nii.gz'))
    raw_img = nb.load(next((r for r in raws if f'dir-{series}' in r.name), raws[0]))
    raw_b0 = nb.Nifti1Image(np.asarray(raw_img.dataobj)[..., 0], raw_img.affine)
    report['b0_uncorrected_vs_clean'] = _corr(_sample(raw_b0, q_ras)[m], clean[m])

    # motion: framewise displacement on a static object, or the parameters vs the applied poses
    conf = next((c for c in conf_files if f'dir-{series}' in c.name), conf_files[0])
    rows = _read_tsv(conf)
    fd = [
        float(r['framewise_displacement'])
        for r in rows
        if r.get('framewise_displacement') not in (None, '', 'n/a')
    ]
    if fd:
        report['fd_mean_mm'] = float(np.mean(fd))
    motion_truth = truth_dir / f'{stem}_desc-motion_timeseries.tsv'
    cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    if motion_truth.exists() and all(c in rows[0] for c in cols):
        truth = np.loadtxt(motion_truth, skiprows=1)
        est = np.array(
            [[float(r[c]) if r[c] not in ('', 'n/a') else np.nan for c in cols] for r in rows]
        )
        n = min(len(truth), len(est))
        truth, est = truth[:n] - truth[0], est[:n] - est[0]
        report['motion'] = {
            c: {
                'corr': _corr(est[:, i], truth[:, i]),
                'amplitude_ratio': float(np.ptp(est[:, i]) / np.ptp(truth[:, i]))
                if np.ptp(truth[:, i]) > 0
                else float('nan'),
            }
            for i, c in enumerate(cols)
        }

    # gradient nonlinearity: qsiprep's deviation map (ACPC voxel axes) vs the truth (raw axes)
    gd_est = _find([sub_dir], '*_space-ACPC*graddev.nii.gz')
    gd_true = truth_dir / f'{stem}_desc-gnlgraddev_dwi.nii.gz'
    if gd_est is not None and gd_true.exists():
        E = np.asarray(nb.load(gd_est).dataobj, dtype=np.float64).reshape(-1, 9)
        Tm = _sample(nb.load(gd_true), q_ras).reshape(-1, 3, 3)
        raw_aff = nb.load(gd_true).affine
        D_raw = raw_aff[:3, :3] / np.linalg.norm(raw_aff[:3, :3], axis=0)
        D_acpc = ref.affine[:3, :3] / np.linalg.norm(ref.affine[:3, :3], axis=0)
        Q = np.linalg.inv(D_acpc) @ (LPS @ np.linalg.inv(A) @ LPS) @ D_raw
        Tq = np.einsum('ij,njk,lk->nil', Q, Tm, Q).reshape(-1, 9)
        ok = m & np.isfinite(E).all(1)
        dev_e, dev_t = (E - np.eye(3).ravel())[ok], (Tq - np.eye(3).ravel())[ok]
        report['gnl_graddev'] = {
            'corr': _corr(dev_e, dev_t),
            'slope': float(dev_e.ravel() @ dev_t.ravel() / (dev_t.ravel() @ dev_t.ravel())),
            'rms_truth_dev': float(np.sqrt(np.mean(dev_t**2))),
            'rms_residual': float(np.sqrt(np.mean((dev_e - dev_t) ** 2))),
        }
    return report
