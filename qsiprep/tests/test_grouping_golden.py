"""Golden/characterization tests for scan grouping and field map selection.

These freeze the behavior of ``group_dwi_scans`` and the ``base.py`` adapter
(``_build_outputs_to_files``) so the value-object refactor can be proven not to
change behavior.

To (re)generate snapshots after an intentional behavior change::

    QSIPREP_REGEN_GOLDEN=1 .pixi/envs/test/bin/python -m pytest \
        qsiprep/tests/test_grouping_golden.py -p no:cacheprovider

Without the env var, the test compares against the committed snapshots.
"""

import json
import os
from itertools import product
from pathlib import Path

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.tests import test_utils_grouping as fixtures
from qsiprep.utils import grouping

try:
    from qsiprep.workflows.base import _build_outputs_to_files
except ModuleNotFoundError:  # pragma: no cover
    _build_outputs_to_files = None

GOLDEN_DIR = Path(__file__).parent / 'data' / 'grouping_golden'

FIXTURE_NAMES = sorted(name for name in dir(fixtures) if name.startswith('dset_'))

# Flag axes exercised for every fixture.
FLAG_COMBOS = list(
    product(
        [True, False],   # combine_scans
        [True, False],   # ignore_fieldmaps
        [True, False],   # estimate_per_axis
    )
)


def _case_id(name, combine_scans, ignore_fieldmaps, estimate_per_axis):
    return (
        f'{name}__combine-{int(combine_scans)}'
        f'__ignorefmaps-{int(ignore_fieldmaps)}'
        f'__peraxis-{int(estimate_per_axis)}'
    )


def _relativize(obj, root):
    """Replace absolute paths under *root* with root-relative paths."""
    if isinstance(obj, str):
        return os.path.relpath(obj, root) if obj.startswith(root) else obj
    if isinstance(obj, dict):
        return {k: _relativize(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_relativize(v, root) for v in obj]
    return obj


def _capture(dset_dict, name, tmp_path, combine_scans, ignore_fieldmaps, estimate_per_axis):
    bids_dir = tmp_path / name
    generate_bids_skeleton(str(bids_dir), dset_dict)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    root = str(bids_dir)
    try:
        dg, fme, fma, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=combine_scans,
            ignore_fieldmaps=ignore_fieldmaps,
            estimate_per_axis=estimate_per_axis,
        )
        outputs = _build_outputs_to_files(layout, dg, fme, fma)
        fieldmap_infos = {key: val['fieldmap_info'] for key, val in outputs.items()}
        result = {
            'distortion_groups': dg,
            'fmap_estimation_groups': fme,
            'fmap_application_groups': fma,
            'concatenation_groups': cg,
            'fieldmap_infos': fieldmap_infos,
        }
        return _relativize(result, root)
    except (ValueError, RuntimeError, TypeError, Exception) as err:  # noqa: BLE001
        # Record the raised behavior so error cases are part of the contract.
        return {'__error__': type(err).__name__, '__match__': str(err)[:120]}


@pytest.mark.parametrize('name', FIXTURE_NAMES)
@pytest.mark.parametrize(('combine_scans', 'ignore_fieldmaps', 'estimate_per_axis'), FLAG_COMBOS)
def test_grouping_golden(name, combine_scans, ignore_fieldmaps, estimate_per_axis, tmp_path):
    if _build_outputs_to_files is None:
        pytest.skip('qsiprep.workflows.base unavailable in this environment')

    dset_dict = getattr(fixtures, name)
    case_id = _case_id(name, combine_scans, ignore_fieldmaps, estimate_per_axis)
    actual = _capture(
        dset_dict, name, tmp_path, combine_scans, ignore_fieldmaps, estimate_per_axis
    )
    actual_str = json.dumps(actual, indent=2)  # insertion order + list order preserved
    golden_path = GOLDEN_DIR / f'{case_id}.json'

    if os.environ.get('QSIPREP_REGEN_GOLDEN'):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual_str + '\n')
        pytest.skip(f'regenerated {golden_path.name}')

    assert golden_path.is_file(), f'Missing golden snapshot: {golden_path}'
    expected_str = golden_path.read_text().rstrip('\n')
    assert actual_str == expected_str
