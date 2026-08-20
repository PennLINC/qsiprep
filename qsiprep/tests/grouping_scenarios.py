"""Shared helpers for the qsiprep.grouping test suite.

Each scenario is a YAML BIDS skeleton in ``qsiprep/tests/data/`` named
``skeleton_grouping_<scenario>.yml``. :func:`load_scenario` materializes the
skeleton in a temporary directory and runs the grouping on it.

``generate_bids_skeleton`` can only touch empty data files, so scenarios that
need real b-value content declare it in :data:`SCENARIO_BVALS`; the loader
writes those ``.bval`` files after the skeleton is generated.
"""

import glob
import os.path as op

from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.grouping import build_dwi_grouping
from qsiprep.tests.utils import get_test_data_path

#: Every scenario skeleton that ships with the test suite.
SCENARIOS = [
    'abcd_style',
    'abcd_t2w',
    'acq_multipart',
    'b0only_fmap_with_bvals',
    'bidsuri_intendedfor',
    'cluster_multipart',
    'cluster_nomultipart',
    'cross_axis_b0field',
    'cross_axis_unpaired',
    'curated_b0field',
    'curated_t2wreg',
    'fieldmapless_t1w_only',
    'fieldmapless_t2w',
    'fov_grid',
    'fov_oblique',
    'fov_shift',
    'gre_phasediff',
    'hcp_style',
    'intendedfor_superseded',
    'maxb_mismatch',
    'missing_pedir',
    'mixed_trt',
    'multi_readout',
    'multi_session',
    'multi_session_b0field_reused',
    'multi_session_curated_multipart',
    'multi_session_shared_fmap',
    'multipart_splits_estimation',
    'name_collision',
    'name_collision_inferred',
    'nonshelled_pair',
    'partial_curation',
    'partial_curation_stranded',
    'partial_intendedfor',
    'partial_multipart',
    'partial_pair',
    'reshim',
    'same_ped_own_fmaps',
    'shell_mix',
    't2w_hcp',
    'two_gre_fmaps',
    'unlinked_fmap',
    'virtual_acq_isolated_fields',
    'virtual_acq_multipart',
]

#: A two-shell (b=1000/2000) scheme: unambiguously shelled.
SHELLED_BVALS = ' '.join(['0'] + ['1000'] * 6 + ['2000'] * 6)

#: A CS-DSI-style q-space grid: many sparse b-value clusters, non-shelled.
NONSHELLED_BVALS = ' '.join(map(str, [0] + list(range(200, 3000, 150))))

#: scenario -> {dwi nii basename: bval file content}
SCENARIO_BVALS = {
    'shell_mix': {
        'sub-01_dir-AP_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_dir-PA_dwi.nii.gz': NONSHELLED_BVALS,
    },
    'nonshelled_pair': {
        'sub-01_dir-AP_dwi.nii.gz': NONSHELLED_BVALS,
        'sub-01_dir-PA_dwi.nii.gz': NONSHELLED_BVALS,
    },
    'maxb_mismatch': {
        'sub-01_dir-AP_dwi.nii.gz': ' '.join(['0'] + ['1000'] * 6),
        'sub-01_dir-PA_dwi.nii.gz': ' '.join(['0'] + ['3000'] * 6),
    },
    'multi_readout': {
        'sub-01_acq-fast_dir-AP_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_acq-fast_dir-PA_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_acq-slow_dir-AP_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_acq-slow_dir-PA_dwi.nii.gz': SHELLED_BVALS,
    },
    'partial_pair': {
        'sub-01_acq-fast_dir-AP_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_acq-fast_dir-PA_dwi.nii.gz': SHELLED_BVALS,
        'sub-01_acq-slow_dir-AP_dwi.nii.gz': SHELLED_BVALS,
    },
}

#: scenario -> {dwi nii basename: grid spec}. Skeleton files are zero-byte
#: placeholders; scenarios exercising field-of-view checks replace them with
#: real (tiny) NIfTIs. Spec keys: shape, shift (mm), rot_x_deg.
SCENARIO_NIFTIS = {
    'fov_shift': {
        'sub-01_dir-AP_dwi.nii.gz': {},
        'sub-01_dir-PA_dwi.nii.gz': {'shift': (10.0, 0.0, 5.0)},
    },
    'fov_oblique': {
        'sub-01_dir-AP_dwi.nii.gz': {},
        'sub-01_dir-PA_dwi.nii.gz': {'rot_x_deg': 5.0},
    },
    'fov_grid': {
        'sub-01_dir-AP_dwi.nii.gz': {},
        'sub-01_dir-PA_dwi.nii.gz': {'shape': (4, 4, 6)},
    },
}


def write_test_nifti(path, spec):
    """Write a tiny real NIfTI with the grid described by ``spec``."""
    import math

    import nibabel as nb
    import numpy as np

    shape = tuple(spec.get('shape', (4, 4, 4)))
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -40.0
    if 'shift' in spec:
        affine[:3, 3] += np.asarray(spec['shift'], dtype=float)
    if 'rot_x_deg' in spec:
        theta = math.radians(spec['rot_x_deg'])
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta)],
                [0.0, math.sin(theta), math.cos(theta)],
            ]
        )
        affine[:3, :3] = rot_x @ affine[:3, :3]
    nb.Nifti1Image(np.zeros((*shape, 2), dtype=np.uint8), affine).to_filename(str(path))


def _find_generated(bids_dir, nii_name):
    (matched,) = glob.glob(str(bids_dir / 'sub-*' / '**' / nii_name), recursive=True)
    return matched


def build_layout(scenario, tmp_path):
    """Materialize a scenario skeleton and return (layout, subject_data)."""
    from qsiprep.grouping.metadata import sibling_bval

    bids_dir = tmp_path / scenario
    skeleton = op.join(get_test_data_path(), f'skeleton_grouping_{scenario}.yml')
    generate_bids_skeleton(str(bids_dir), skeleton)
    for nii_name, bval_content in SCENARIO_BVALS.get(scenario, {}).items():
        with open(sibling_bval(_find_generated(bids_dir, nii_name)), 'w') as fobj:
            fobj.write(bval_content + '\n')
    for nii_name, grid_spec in SCENARIO_NIFTIS.get(scenario, {}).items():
        write_test_nifti(_find_generated(bids_dir, nii_name), grid_spec)
    layout = BIDSLayout(str(bids_dir), validate=False)
    subject_data = {
        'dwi': sorted(
            layout.get(
                suffix='dwi',
                datatype='dwi',
                extension=['.nii', '.nii.gz'],
                return_type='file',
            )
        )
    }
    return layout, subject_data


def load_scenario(scenario, tmp_path, strict=True, **kwargs):
    """Materialize a scenario and run the grouping on it."""
    layout, subject_data = build_layout(scenario, tmp_path)
    return build_dwi_grouping(layout, subject_data, strict=strict, **kwargs)


def basenames(paths):
    """Map full paths to basenames for readable assertions."""
    return [op.basename(path) for path in paths]
