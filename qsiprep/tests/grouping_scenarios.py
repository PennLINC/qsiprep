"""Shared helpers for the qsiprep.grouping test suite.

Each scenario is a YAML BIDS skeleton in ``qsiprep/tests/data/`` named
``skeleton_grouping_<scenario>.yml``. :func:`load_scenario` materializes the
skeleton in a temporary directory and runs the grouping on it.
"""

import os.path as op

from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.grouping import build_dwi_grouping
from qsiprep.tests.utils import get_test_data_path

#: Every scenario skeleton that ships with the test suite.
SCENARIOS = [
    'abcd_style',
    'b0only_fmap_with_bvals',
    'bidsuri_intendedfor',
    'cluster_multipart',
    'cluster_nomultipart',
    'cross_axis_b0field',
    'curated_b0field',
    'gre_phasediff',
    'hcp_style',
    'missing_pedir',
    'mixed_trt',
    'multi_session',
    'multipart_splits_estimation',
    'name_collision',
    'partial_curation',
    'reshim',
    'same_ped_own_fmaps',
    'two_gre_fmaps',
    'unlinked_fmap',
]


def build_layout(scenario, tmp_path):
    """Materialize a scenario skeleton and return (layout, subject_data)."""
    bids_dir = tmp_path / scenario
    skeleton = op.join(get_test_data_path(), f'skeleton_grouping_{scenario}.yml')
    generate_bids_skeleton(str(bids_dir), skeleton)
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
