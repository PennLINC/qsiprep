# Scan Grouping & Field Map Selection Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `qsiprep`'s scan-grouping and field-map-selection code simpler and more interpretable by introducing SDCFlows-style value objects, without changing any behavior.

**Architecture:** A new `qsiprep/utils/fieldmaps.py` mirrors `sdcflows/fieldmaps.py` (`EstimatorType`, `FieldmapFile`, `FieldmapEstimation`, an instance-scoped registry) and owns all per-suffix field-map knowledge. A slimmed `qsiprep/utils/grouping.py` keeps only QSIPrep-specific concepts (`DwiSeries`, distortion/concatenation groups, a `find_estimators`-shaped discovery). `group_dwi_scans` becomes a thin orchestrator that serializes value objects to today's four dicts; the `base.py` adapter shrinks to call `FieldmapEstimation.to_fieldmap_info()`. A golden/characterization test suite, frozen against current `group-dwi-scans` HEAD, is the behavior contract.

**Tech Stack:** Python 3.10+, stdlib `dataclasses` (`slots=True`), `pybids` (`BIDSLayout`), `pytest`, `niworkflows.utils.testing.generate_bids_skeleton`. Test env interpreter: `.pixi/envs/test/bin/python`. Run pytest with `-p no:cacheprovider` (the repo filesystem rejects pytest cache writes).

**Spec:** `docs/superpowers/specs/2026-06-10-scan-grouping-fieldmap-refactor-design.md`

**Branch:** `refactor-scan-grouping` (already created off `group-dwi-scans`).

---

## Conventions used in every task

- **Interpreter / test command:** always
  `.pixi/envs/test/bin/python -m pytest <args> -p no:cacheprovider`
- **All paths are repo-relative** to `/mnt/c/Users/tsalo/Documents/linc/qsiprep`.
- **Commit after each task** (frequent commits). Commit messages end with the
  trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- The legacy `fieldmap_info` shapes that must be reproduced exactly:
  - PEPOLAR EPI: `{'suffix': 'epi', 'epi': [<paths>]}`
  - Reverse-PE DWI: `{'suffix': 'rpe_series', 'rpe_series': [<paths>]}`
    (optionally also `'epi': [<paths>]` when fmap EPIs are present too)
  - GRE phasediff: `{'suffix': 'phasediff', 'phasediff': <path>, 'magnitude1': <path>, 'magnitude2': <path>}`
  - GRE two-phase: `{'suffix': 'phase1', 'phase1': <path>, 'phase2': <path>, 'magnitude1': <path>, 'magnitude2': <path>}`
  - Direct fieldmap: `{'suffix': 'fieldmap', 'fieldmap': <path>, 'magnitude': <path>}`
  - None: `{'suffix': None}`

---

## Phase 1 — Golden / characterization tests (freeze current behavior)

No production code changes in this phase. Capture the current four-dict output
of `group_dwi_scans` and the per-group `fieldmap_info` for every fixture × flag
combination, freeze them as JSON, and assert equality on every subsequent run.

### Task 1: Add two missing-coverage fixtures

**Files:**
- Modify: `qsiprep/tests/test_utils_grouping.py` (append two module-level fixtures next to the existing `dset_*` definitions, e.g. after `dset_with_two_phase_fmaps`)

- [ ] **Step 1: Add a `B0FieldSource`-only fixture and a mixed EPI+GRE fixture**

Append to `qsiprep/tests/test_utils_grouping.py`:

```python
# A DWI with B0FieldSource but no B0FieldIdentifier on the DWI itself;
# the EPI fmap carries the matching B0FieldIdentifier.
dset_b0source_only = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'B0FieldIdentifier': 'pepolar01',
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.05,
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'TotalReadoutTime': 0.05,
                        'B0FieldSource': 'pepolar01',
                    },
                },
            ],
        },
    ],
}

# One fmap/ directory holding BOTH an EPI fieldmap and a GRE phasediff fieldmap,
# linked to the same DWI via IntendedFor.
dset_mixed_epi_gre = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'IntendedFor': ['dwi/sub-01_dir-AP_dwi.nii.gz'],
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.05,
                    },
                },
                {'suffix': 'magnitude1', 'metadata': {'EchoTime': 0.00492}},
                {'suffix': 'magnitude2', 'metadata': {'EchoTime': 0.00738}},
                {
                    'suffix': 'phasediff',
                    'metadata': {
                        'IntendedFor': ['dwi/sub-01_dir-AP_dwi.nii.gz'],
                        'EchoTime1': 0.00492,
                        'EchoTime2': 0.00738,
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'TotalReadoutTime': 0.05,
                    },
                },
            ],
        },
    ],
}
```

- [ ] **Step 2: Verify the fixtures import cleanly**

Run:
```bash
.pixi/envs/test/bin/python -c "from qsiprep.tests import test_utils_grouping as f; print(f.dset_b0source_only is not None, f.dset_mixed_epi_gre is not None)"
```
Expected: `True True`

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_utils_grouping.py
git commit -m "test: add B0FieldSource-only and mixed EPI+GRE grouping fixtures

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2: Write the golden-test harness

**Files:**
- Create: `qsiprep/tests/test_grouping_golden.py`
- Create (generated next task): `qsiprep/tests/data/grouping_golden/` (snapshot dir)

- [ ] **Step 1: Create the harness**

Create `qsiprep/tests/test_grouping_golden.py`:

```python
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
```

Notes for the implementer:
- The broad `except ... Exception` is intentional and unusual — this harness must
  record *whatever* the current code does, including unexpected raises, as the
  frozen contract. Keep the `# noqa: BLE001`.
- String comparison (not dict `==`) is deliberate: it pins dict key insertion
  order and list order, which are part of the behavior the spec requires
  preserving.

- [ ] **Step 2: Confirm the harness collects without errors**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py --collect-only -q -p no:cacheprovider 2>&1 | tail -5
```
Expected: a count of collected items (≈ 25 fixtures × 8 combos ≈ 200), no collection errors.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_grouping_golden.py
git commit -m "test: add golden harness for scan grouping (no snapshots yet)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Capture and commit the golden snapshots

**Files:**
- Create: `qsiprep/tests/data/grouping_golden/*.json` (one per case)

- [ ] **Step 1: Generate snapshots from current HEAD**

Run:
```bash
QSIPREP_REGEN_GOLDEN=1 .pixi/envs/test/bin/python -m pytest \
    qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -5
```
Expected: all cases reported as skipped (`regenerated ...`).

- [ ] **Step 2: Verify snapshots now pass in compare mode**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -5
```
Expected: all cases PASS (0 failed).

- [ ] **Step 3: Sanity-check a couple of snapshots by eye**

Run:
```bash
ls qsiprep/tests/data/grouping_golden | wc -l
cat qsiprep/tests/data/grouping_golden/dset_with_phasediff_fmaps__combine-1__ignorefmaps-0__peraxis-0.json
```
Expected: the file count matches the collected case count; the phasediff snapshot
shows `"fieldmap_infos"` with `"suffix": "phasediff"` (not `"epi"`), confirming
the harness captured the post-bugfix behavior.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/tests/data/grouping_golden
git commit -m "test: freeze golden snapshots for scan grouping behavior

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — `qsiprep/utils/fieldmaps.py` (value objects, not yet wired in)

Introduce the field-map model mirroring `sdcflows/fieldmaps.py`. Unit-test it in
isolation. Nothing in `grouping.py`/`base.py` uses it yet, so the golden suite is
unaffected.

### Task 4: `EstimatorType`, `MODALITIES`, and module skeleton

**Files:**
- Create: `qsiprep/utils/fieldmaps.py`
- Create: `qsiprep/tests/test_utils_fieldmaps.py`

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_utils_fieldmaps.py`:

```python
"""Tests for the field-map value-object model."""

import pytest

from qsiprep.utils import fieldmaps as fm


def test_estimator_types_exist():
    assert {t.name for t in fm.EstimatorType} >= {
        'UNKNOWN',
        'PEPOLAR',
        'PHASEDIFF',
        'MAPPED',
        'ANAT',
    }


def test_modalities_mapping():
    assert fm.MODALITIES['dwi'] is fm.EstimatorType.PEPOLAR
    assert fm.MODALITIES['epi'] is fm.EstimatorType.PEPOLAR
    assert fm.MODALITIES['phasediff'] is fm.EstimatorType.PHASEDIFF
    assert fm.MODALITIES['phase1'] is fm.EstimatorType.PHASEDIFF
    assert fm.MODALITIES['fieldmap'] is fm.EstimatorType.MAPPED
    assert fm.MODALITIES['T1w'] is fm.EstimatorType.ANAT
    assert fm.MODALITIES['magnitude1'] is None
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.utils.fieldmaps'`.

- [ ] **Step 3: Create the module skeleton**

Create `qsiprep/utils/fieldmaps.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Field-map value objects for QSIPrep.

This module mirrors the names and structure of ``sdcflows.fieldmaps`` so that a
future offload of distortion correction to SDCFlows is a deletion-and-delegation
rather than a rewrite. It uses stdlib dataclasses (rather than SDCFlows' attrs)
to avoid taking on a new direct dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto

from nipype.utils.filemanip import split_filename


class EstimatorType(Enum):
    """The kind of field-map estimation a set of sources supports."""

    UNKNOWN = auto()
    PEPOLAR = auto()
    PHASEDIFF = auto()
    MAPPED = auto()
    ANAT = auto()


MODALITIES = {
    'dwi': EstimatorType.PEPOLAR,
    'epi': EstimatorType.PEPOLAR,
    'fieldmap': EstimatorType.MAPPED,
    'magnitude': None,
    'magnitude1': None,
    'magnitude2': None,
    'phase1': EstimatorType.PHASEDIFF,
    'phase2': EstimatorType.PHASEDIFF,
    'phasediff': EstimatorType.PHASEDIFF,
    'T1w': EstimatorType.ANAT,
    'T2w': EstimatorType.ANAT,
}


def _suffix_of(path):
    """Return the BIDS suffix of a NIfTI path (e.g. 'phasediff')."""
    return split_filename(path)[1].split('_')[-1]
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/fieldmaps.py qsiprep/tests/test_utils_fieldmaps.py
git commit -m "feat: add EstimatorType and MODALITIES field-map model skeleton

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 5: `FieldmapFile` value object

**Files:**
- Modify: `qsiprep/utils/fieldmaps.py`
- Modify: `qsiprep/tests/test_utils_fieldmaps.py`

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_fieldmaps.py`:

```python
def _touch(path, content=''):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def test_fieldmapfile_reads_suffix_and_metadata(tmp_path):
    nii = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    f = fm.FieldmapFile(nii, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})
    assert f.suffix == 'phasediff'
    assert f.metadata['EchoTime1'] == 0.004
    assert f.path == nii


def test_fieldmapfile_finds_sibling_magnitudes(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    pd = _touch(fmap_dir / 'sub-01_phasediff.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude1.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude2.nii.gz')
    f = fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})
    siblings = f.find_siblings(('magnitude1', 'magnitude2'))
    assert sorted(os.path.basename(s) for s in siblings.values()) == [
        'sub-01_magnitude1.nii.gz',
        'sub-01_magnitude2.nii.gz',
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: FAIL — `AttributeError: module 'qsiprep.utils.fieldmaps' has no attribute 'FieldmapFile'`.

- [ ] **Step 3: Implement `FieldmapFile`**

Append to `qsiprep/utils/fieldmaps.py`:

```python
@dataclass(slots=True)
class FieldmapFile:
    """A single file usable in field-map estimation, with metadata read once.

    Parameters
    ----------
    path : str
        Path to a NIfTI file in a BIDS tree.
    metadata : dict, optional
        Metadata for the file. Caller is responsible for providing BIDS-resolved
        metadata (QSIPrep reads it from the pybids layout).
    """

    path: str
    metadata: dict = field(default_factory=dict)
    suffix: str = field(init=False)

    def __post_init__(self):
        self.suffix = _suffix_of(self.path)

    def find_siblings(self, suffixes):
        """Return ``{suffix: path}`` for sibling files that exist on disk.

        Siblings are located by replacing this file's suffix token in its
        basename, matching pybids' ``get_fieldmap`` filename convention.
        """
        dirname, basename = os.path.split(self.path)
        found = {}
        for sibling_suffix in suffixes:
            sibling_basename = basename.replace(
                f'_{self.suffix}.', f'_{sibling_suffix}.'
            )
            if sibling_basename == basename:
                continue
            candidate = os.path.join(dirname, sibling_basename)
            if os.path.exists(candidate):
                found[sibling_suffix] = candidate
        return found
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/fieldmaps.py qsiprep/tests/test_utils_fieldmaps.py
git commit -m "feat: add FieldmapFile value object with sibling discovery

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 6: `FieldmapEstimation` — method inference + `bids_id`

**Files:**
- Modify: `qsiprep/utils/fieldmaps.py`
- Modify: `qsiprep/tests/test_utils_fieldmaps.py`

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_fieldmaps.py`:

```python
def test_estimation_infers_pepolar(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    assert est.method is fm.EstimatorType.PEPOLAR


def test_estimation_infers_phasediff(tmp_path):
    pd = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    m1 = _touch(tmp_path / 'sub-01_magnitude1.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006}),
            fm.FieldmapFile(m1),
        ]
    )
    assert est.method is fm.EstimatorType.PHASEDIFF


def test_estimation_uses_b0fieldidentifier_as_id(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j', 'B0FieldIdentifier': 'pp1'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-', 'B0FieldIdentifier': 'pp1'}),
        ]
    )
    assert est.bids_id == 'pp1'


def test_estimation_auto_id_when_unnamed(tmp_path):
    pd = _touch(tmp_path / 'sub-01_phasediff.nii.gz')
    m1 = _touch(tmp_path / 'sub-01_magnitude1.nii.gz')
    est = fm.FieldmapEstimation(
        [fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006}), fm.FieldmapFile(m1)],
        auto_id='auto_00000',
    )
    assert est.bids_id == 'auto_00000'
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'FieldmapEstimation'`.

- [ ] **Step 3: Implement `FieldmapEstimation`**

Append to `qsiprep/utils/fieldmaps.py`:

```python
_PEPOLAR_SUFFIXES = ('dwi', 'epi')
_GRE_SUFFIXES = ('fieldmap', 'phasediff', 'phase1', 'phase2')


@dataclass(slots=True)
class FieldmapEstimation:
    """A set of :class:`FieldmapFile`s that together define one field map.

    The estimation ``method`` is inferred from the suffixes of the sources. The
    ``bids_id`` is the common ``B0FieldIdentifier`` of the sources, or the
    provided ``auto_id`` fallback when none is set.

    Parameters
    ----------
    sources : list[FieldmapFile]
    auto_id : str, optional
        Identifier to use when the sources carry no ``B0FieldIdentifier``.
    """

    sources: list
    auto_id: str | None = None
    method: EstimatorType = field(init=False, default=EstimatorType.UNKNOWN)
    bids_id: str | None = field(init=False, default=None)

    def __post_init__(self):
        suffixes = {f.suffix for f in self.sources}

        gre = suffixes.intersection(_GRE_SUFFIXES)
        pepolar = suffixes.intersection(_PEPOLAR_SUFFIXES)

        if gre:
            # phase1/phase2 are the same method; any other mix is invalid.
            if len(gre) > 1 and gre - {'phase1', 'phase2'}:
                raise ValueError(f'Incompatible field-map suffixes: {sorted(gre)}')
            self.method = MODALITIES[sorted(gre)[0]]
        elif pepolar:
            self.method = EstimatorType.PEPOLAR
        else:
            raise ValueError('Insufficient sources to estimate a field map.')

        b0_ids = {
            f.metadata['B0FieldIdentifier']
            for f in self.sources
            if f.metadata.get('B0FieldIdentifier') is not None
        }
        # Flatten list-valued identifiers.
        flat_ids = set()
        for value in b0_ids:
            flat_ids.update(value if isinstance(value, list) else [value])

        if flat_ids:
            if len(flat_ids) > 1:
                raise ValueError(f'Conflicting B0FieldIdentifiers: {sorted(flat_ids)}')
            self.bids_id = flat_ids.pop()
        else:
            self.bids_id = self.auto_id

    def paths(self):
        """Sorted tuple of source paths."""
        return tuple(sorted(str(f.path) for f in self.sources))
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/fieldmaps.py qsiprep/tests/test_utils_fieldmaps.py
git commit -m "feat: add FieldmapEstimation with method inference and bids_id

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 7: `FieldmapEstimation.to_fieldmap_info()` (legacy serializer)

This is the single place per-suffix `fieldmap_info` is built. It replaces the
suffix branching in `base._build_outputs_to_files` and `grouping.get_highest_priority_fieldmap` / `classify_fmap_files`. The PEPOLAR branch takes the
caller-resolved EPI paths and reverse-PE DWI paths (which depend on the target
distortion group), because that context lives in the grouping layer.

**Files:**
- Modify: `qsiprep/utils/fieldmaps.py`
- Modify: `qsiprep/tests/test_utils_fieldmaps.py`

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_fieldmaps.py`:

```python
def test_fieldmap_info_phasediff(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    pd = _touch(fmap_dir / 'sub-01_phasediff.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude1.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude2.nii.gz')
    est = fm.FieldmapEstimation(
        [fm.FieldmapFile(pd, metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006})]
    )
    info = est.to_fieldmap_info()
    assert info['suffix'] == 'phasediff'
    assert info['phasediff'].endswith('sub-01_phasediff.nii.gz')
    assert info['magnitude1'].endswith('sub-01_magnitude1.nii.gz')
    assert info['magnitude2'].endswith('sub-01_magnitude2.nii.gz')
    assert 'epi' not in info


def test_fieldmap_info_fieldmap(tmp_path):
    fmap_dir = tmp_path / 'sub-01' / 'fmap'
    fmapf = _touch(fmap_dir / 'sub-01_fieldmap.nii.gz')
    _touch(fmap_dir / 'sub-01_magnitude.nii.gz')
    est = fm.FieldmapEstimation([fm.FieldmapFile(fmapf, metadata={'Units': 'Hz'})])
    info = est.to_fieldmap_info()
    assert info['suffix'] == 'fieldmap'
    assert info['fieldmap'].endswith('sub-01_fieldmap.nii.gz')
    assert info['magnitude'].endswith('sub-01_magnitude.nii.gz')


def test_fieldmap_info_pepolar_epi_only(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    pa = _touch(tmp_path / 'sub-01_dir-PA_epi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(pa, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    info = est.to_fieldmap_info(epi_files=[ap, pa], rpe_files=[])
    assert info == {'suffix': 'epi', 'epi': sorted([ap, pa])}


def test_fieldmap_info_pepolar_rpe_series(tmp_path):
    ap = _touch(tmp_path / 'sub-01_dir-AP_epi.nii.gz')
    rpe = _touch(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    est = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(ap, metadata={'PhaseEncodingDirection': 'j'}),
            fm.FieldmapFile(rpe, metadata={'PhaseEncodingDirection': 'j-'}),
        ]
    )
    info = est.to_fieldmap_info(epi_files=[ap], rpe_files=[rpe])
    assert info['suffix'] == 'rpe_series'
    assert info['rpe_series'] == [rpe]
    assert info['epi'] == [ap]
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -k fieldmap_info -p no:cacheprovider -q
```
Expected: FAIL — `TypeError: to_fieldmap_info() ...` / `AttributeError`.

- [ ] **Step 3: Implement `to_fieldmap_info`**

Append as a method on `FieldmapEstimation` (inside the class):

```python
    def to_fieldmap_info(self, epi_files=None, rpe_files=None):
        """Build the legacy ``fieldmap_info`` dict for downstream workflows.

        For GRE field maps (MAPPED / PHASEDIFF) the dict is built from the
        sources and their sibling magnitude/phase files. For PEPOLAR field maps
        the caller supplies the target-specific ``epi_files`` (EPIs from
        ``fmap/``) and ``rpe_files`` (reverse-PE DWI series), because which
        series are sources vs. targets depends on the distortion group being
        corrected.
        """
        if self.method is EstimatorType.MAPPED:
            src = next(f for f in self.sources if f.suffix == 'fieldmap')
            info = {'suffix': 'fieldmap', 'fieldmap': str(src.path)}
            mag = self._sibling_path(src, 'magnitude')
            if mag is not None:
                info['magnitude'] = mag
            return info

        if self.method is EstimatorType.PHASEDIFF:
            phasediff = next((f for f in self.sources if f.suffix == 'phasediff'), None)
            if phasediff is not None:
                info = {'suffix': 'phasediff', 'phasediff': str(phasediff.path)}
                for mag in ('magnitude1', 'magnitude2'):
                    sib = self._sibling_path(phasediff, mag)
                    if sib is not None:
                        info[mag] = sib
                return info
            # phase1/phase2
            phase1 = next(f for f in self.sources if f.suffix == 'phase1')
            info = {'suffix': 'phase1', 'phase1': str(phase1.path)}
            for sib_suffix in ('phase2', 'magnitude1', 'magnitude2'):
                sib = self._sibling_path(phase1, sib_suffix)
                if sib is not None:
                    info[sib_suffix] = sib
            return info

        # PEPOLAR
        epi_files = sorted(epi_files or [])
        rpe_files = sorted(rpe_files or [])
        if rpe_files:
            info = {'suffix': 'rpe_series', 'rpe_series': rpe_files}
            if epi_files:
                info['epi'] = epi_files
            return info
        if epi_files:
            return {'suffix': 'epi', 'epi': epi_files}
        return {'suffix': None}

    def _sibling_path(self, source, suffix):
        """Return a sibling path already among sources, else discover on disk."""
        for f in self.sources:
            if f.suffix == suffix:
                return str(f.path)
        found = source.find_siblings((suffix,))
        return found.get(suffix)
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q
```
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/fieldmaps.py qsiprep/tests/test_utils_fieldmaps.py
git commit -m "feat: add FieldmapEstimation.to_fieldmap_info legacy serializer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — `DwiSeries` + discovery in `grouping.py` (added, not yet wired)

Add the QSIPrep-specific value object and a discovery function alongside the
existing code. Do **not** change `group_dwi_scans` yet — the golden suite must
stay green throughout this phase.

### Task 8: `DwiSeries` value object

**Files:**
- Modify: `qsiprep/utils/grouping.py` (add near the top, after imports)
- Modify: `qsiprep/tests/test_utils_grouping.py` (add a unit test)

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_grouping.py`:

```python
def test_dwiseries_from_layout(tmpdir):
    layout, subject_data = _make_layout(tmpdir, dset_single_dwi, 'dwiseries_single')
    dwi_path = subject_data['dwi'][0]
    series = grouping.DwiSeries.from_layout(layout, dwi_path)
    assert series.path == dwi_path
    assert series.pe_dir is not None
    assert series.session == layout.get_file(dwi_path).entities.get('session')
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_grouping.py::test_dwiseries_from_layout -p no:cacheprovider -q
```
Expected: FAIL — `AttributeError: module 'qsiprep.utils.grouping' has no attribute 'DwiSeries'`.

- [ ] **Step 3: Implement `DwiSeries`**

Add to `qsiprep/utils/grouping.py` (after the existing imports and `LOGGER`/`_DISTORTION_FIELDS` block):

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class DwiSeries:
    """A DWI file plus its distortion-relevant metadata, read once."""

    path: str
    session: object
    pe_dir: object
    shim: object
    trt: object
    b0_identifier: object
    b0_source: object
    multipart_id: object

    @classmethod
    def from_layout(cls, layout, path):
        bidsfile = layout.get_file(path)
        metadata = bidsfile.get_metadata()
        shim = metadata.get('ShimSetting')
        b0fi = metadata.get('B0FieldIdentifier')
        return cls(
            path=path,
            session=bidsfile.entities.get('session'),
            pe_dir=metadata.get('PhaseEncodingDirection'),
            shim=tuple(shim) if isinstance(shim, list) else shim,
            trt=metadata.get('TotalReadoutTime'),
            b0_identifier=tuple(b0fi) if isinstance(b0fi, list) else b0fi,
            b0_source=metadata.get('B0FieldSource'),
            multipart_id=metadata.get('MultipartID'),
        )

    @property
    def distortion_signature(self):
        """The physical-distortion key (no estimator constraint yet)."""
        return (self.session, self.pe_dir, self.shim, self.trt, self.b0_identifier)
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_grouping.py::test_dwiseries_from_layout -p no:cacheprovider -q
```
Expected: PASS.

- [ ] **Step 5: Run the golden suite to confirm no behavior drift**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -3
```
Expected: all PASS (no production behavior touched).

- [ ] **Step 6: Commit**

```bash
git add qsiprep/utils/grouping.py qsiprep/tests/test_utils_grouping.py
git commit -m "feat: add DwiSeries value object (not yet wired in)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 9: `find_estimators` discovery function

This wraps the existing estimation-priority logic (`build_fmap_estimation_groups`
in `qsiprep/utils/grouping.py`) into a function that returns
`FieldmapEstimation` objects plus a per-series estimator-id assignment. **Port the
priority and `estimate_per_axis` logic from the existing functions** — do not
invent new behavior:

- B0FieldIdentifier path → existing `build_fmap_estimation_groups` Path 1 + `_check_b0field_axis_conflict`.
- IntendedFor path → existing `_build_intendedfor_groups`.
- Heuristic path → existing `_build_heuristic_estimation_groups`.

**Files:**
- Modify: `qsiprep/utils/grouping.py` (add `find_estimators`; keep existing
  functions in place for now)
- Modify: `qsiprep/tests/test_utils_grouping.py`

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_grouping.py`:

```python
def test_find_estimators_b0field(tmpdir):
    layout, subject_data = _make_layout(tmpdir, dset_with_b0field_fmaps, 'find_est_b0')
    series = [grouping.DwiSeries.from_layout(layout, p) for p in subject_data['dwi']]
    estimators, series_to_id = grouping.find_estimators(
        layout=layout,
        series=series,
        ignore_fieldmaps=False,
        estimate_per_axis=False,
    )
    assert estimators
    # Every estimator has an inferred method and an id.
    assert all(e.method is not None and e.bids_id for e in estimators)
    # The B0FieldIdentifier becomes the estimator id.
    assert any(e.bids_id == 'pepolar01' for e in estimators)


def test_find_estimators_ignore_fieldmaps(tmpdir):
    layout, subject_data = _make_layout(tmpdir, dset_with_b0field_fmaps, 'find_est_ign')
    series = [grouping.DwiSeries.from_layout(layout, p) for p in subject_data['dwi']]
    estimators, series_to_id = grouping.find_estimators(
        layout=layout,
        series=series,
        ignore_fieldmaps=True,
        estimate_per_axis=False,
    )
    # fmap/ EPIs are ignored; only reverse-PE DWI estimators (if any) remain.
    assert all('epi' not in [s.suffix for s in e.sources] for e in estimators)
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_grouping.py -k find_estimators -p no:cacheprovider -q
```
Expected: FAIL — `AttributeError: ... has no attribute 'find_estimators'`.

- [ ] **Step 3: Implement `find_estimators`**

`find_estimators` operates at the **series level** (not the distortion-group
level today's `build_fmap_estimation_groups` uses). This is the key change that
lets Phase 4 build final distortion groups in one pass and delete refine/remap:
because each series is assigned to an estimator here, the estimator id can be
folded directly into the distortion signature later.

The function returns `(list[FieldmapEstimation], dict[str, str])`, where the
second value maps each DWI series path to the `bids_id` of the estimator that
corrects it. **This returned dict is the spec's "instance-scoped estimator/intent
registry"** — realized as a per-call return value rather than a module-global
class, which is exactly the no-global-state behavior the spec requires.

Add to `qsiprep/utils/grouping.py`:

```python
from collections import defaultdict

from .fieldmaps import FieldmapEstimation, FieldmapFile


def _subject_fmap_files(layout, series):
    """Sorted fmap/ NIfTI paths for the subject owning these series."""
    if not series:
        return []
    sub_id = layout.get_file(series[0].path).entities.get('subject')
    try:
        files = layout.get(
            subject=sub_id, datatype='fmap',
            extension=['.nii.gz', '.nii'], return_type='file',
        )
    except Exception:  # noqa: BLE001
        files = []
    return sorted(files)


def find_estimators(*, layout, series, ignore_fieldmaps, estimate_per_axis):
    """Discover field-map estimators for a set of DWI series.

    Priority (highest first): B0FieldIdentifier -> IntendedFor -> PE-direction
    heuristic. Mirrors sdcflows' ``find_estimators`` shape but returns QSIPrep
    estimator objects plus a per-series application assignment.

    Returns
    -------
    estimators : list[FieldmapEstimation]
    series_to_estimator : dict[str, str]
        Maps a DWI series path to the ``bids_id`` of the estimator that corrects
        it. (Application mapping; mirrors today's fmap_application_groups.)
    """
    estimators = []
    series_to_estimator = {}
    auto_counter = 0

    by_path = {s.path: s for s in series}
    fmap_files = [] if ignore_fieldmaps else _subject_fmap_files(layout, series)

    def _fieldmap_file(path):
        return FieldmapFile(path, metadata=layout.get_metadata(path))

    # ----- Priority 1: B0FieldIdentifier on any DWI or fmap file -----
    # Reproduce build_fmap_estimation_groups Path 1: collect every file (DWI +
    # fmap) carrying B0FieldIdentifier, grouped by identifier value. Each group
    # becomes one FieldmapEstimation whose bids_id == the identifier. Apply the
    # per-axis conflict check (port _check_b0field_axis_conflict) when
    # estimate_per_axis is True. Application: a DWI's B0FieldSource names its
    # estimator; if absent, every DWI in the group is a target.
    b0_members = defaultdict(list)   # identifier -> [paths]
    for s in series:
        for ident in _as_list(s.b0_identifier):
            if ident is not None:
                b0_members[ident].append(s.path)
    for fpath in fmap_files:
        ident = layout.get_metadata(fpath).get('B0FieldIdentifier')
        for value in _as_list(ident):
            if value is not None:
                b0_members[value].append(fpath)

    if b0_members:
        for ident in sorted(b0_members):
            member_paths = b0_members[ident]
            sources = [_fieldmap_file(p) for p in sorted(set(member_paths))]
            est = FieldmapEstimation(sources)  # bids_id resolves to `ident`
            _check_axis_conflict(est, estimate_per_axis)   # ports _check_b0field_axis_conflict
            estimators.append(est)
            for s in series:
                src = s.b0_source if s.b0_source is not None else None
                targets = _as_list(src) if src is not None else (
                    [ident] if s.path in member_paths else []
                )
                if ident in targets:
                    series_to_estimator[s.path] = est.bids_id
        return estimators, series_to_estimator

    # ----- Priority 2: IntendedFor on fmap files -----
    # Reproduce _build_intendedfor_groups: bucket fmap files by the set of DWI
    # series they target (resolved from IntendedFor, BIDS-URI or relpath). Each
    # bucket is one FieldmapEstimation(auto_id=f'auto_{auto_counter:05d}').
    if fmap_files:
        intended = _intendedfor_buckets(layout, fmap_files, by_path)  # ports _resolve_intended_for
        for target_paths, fmap_paths in intended:   # deterministic order: sorted by target tuple
            sources = [_fieldmap_file(p) for p in sorted(fmap_paths)]
            est = FieldmapEstimation(sources, auto_id=f'auto_{auto_counter:05d}')
            auto_counter += 1
            estimators.append(est)
            for tpath in target_paths:
                series_to_estimator[tpath] = est.bids_id
        if estimators:
            return estimators, series_to_estimator

    # ----- Priority 3: PE-direction heuristic -----
    # Reproduce _build_heuristic_estimation_groups exactly, but over series:
    # iterate sorted(sessions), then MultipartID partitions, then PE axes. With
    # estimate_per_axis, pair only opposite directions on the same axis (j/j-);
    # otherwise combine all series in the partition when >=2 distinct PE dirs
    # exist. Each group is FieldmapEstimation([dwi FieldmapFiles],
    # auto_id=f'auto_{auto_counter:05d}'); every member series is also a target.
    for group_paths in _heuristic_series_groups(series, estimate_per_axis):
        sources = [_fieldmap_file(p) for p in sorted(group_paths)]
        est = FieldmapEstimation(sources, auto_id=f'auto_{auto_counter:05d}')
        auto_counter += 1
        estimators.append(est)
        for p in group_paths:
            series_to_estimator[p] = est.bids_id

    return estimators, series_to_estimator
```

Implementer guidance:
- Add the small helpers used above by **porting the bodies of the existing
  functions** (then those originals are deleted in Task 11):
  - `_as_list(value)` — the existing `_ensure_list`.
  - `_check_axis_conflict(est, estimate_per_axis)` — the existing
    `_check_b0field_axis_conflict`, but reading PE direction from the estimator's
    sources.
  - `_intendedfor_buckets(layout, fmap_files, by_path)` — the existing
    `_build_intendedfor_groups` + `_resolve_intended_for`, returning
    `[(sorted target_paths tuple, set of fmap paths)]` in the same deterministic
    order (sorted by target tuple) so `auto_` ids match.
  - `_heuristic_series_groups(series, estimate_per_axis)` — the existing
    `_build_heuristic_estimation_groups`, yielding lists of series paths in the
    same iteration order (sorted sessions → MultipartID partitions → sorted PE
    axes) so `auto_` ids match.
- The shim/TRT compatibility error (existing
  `_check_distortion_metadata_compatibility`) must still fire when a
  metadata-defined estimation group spans incompatible `ShimSetting`/
  `TotalReadoutTime`. Port it as a check inside the B0FieldIdentifier and
  IntendedFor branches (raise the same `ValueError`).
- **The `auto_xxxxx` numbering order is load-bearing** — the golden snapshots
  encode it. The conflict fixtures (`dset_b0field_shim_conflict`,
  `dset_intendedfor_trt_conflict`, `dset_b0field_cross_axis`) verify the error
  paths; the multi-PED fixtures verify the heuristic ordering.

- [ ] **Step 4: Run to verify the new tests pass**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_grouping.py -k find_estimators -p no:cacheprovider -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Confirm golden suite still green (nothing wired in yet)**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -3
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/utils/grouping.py qsiprep/tests/test_utils_grouping.py
git commit -m "feat: add find_estimators discovery returning FieldmapEstimation objects

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 4 — Swap `group_dwi_scans` internals; delete refine/remap

Rewire `group_dwi_scans` to the new ordering and delete the post-hoc dance. The
**golden suite is the test** — it must stay green after every step. Work in small
commits and run the golden suite each time.

### Task 10: Rebuild distortion groups from `DwiSeries` with estimator constraint

**Files:**
- Modify: `qsiprep/utils/grouping.py` (`group_dwi_scans` + `build_distortion_groups`)

- [ ] **Step 1: Add an estimator-aware final distortion grouping**

In `qsiprep/utils/grouping.py`, add a helper that builds final distortion groups
from `DwiSeries` using `(distortion_signature, estimator_id)`:

```python
def _final_distortion_groups(series_list, series_to_estimator):
    """Group DwiSeries by (physical signature, estimator id).

    Folding the estimator id into the signature means a distortion group never
    spans two estimators, so no post-hoc split/remap is needed.
    """
    groups = []
    keys = []
    for s in series_list:
        key = (s.distortion_signature, series_to_estimator.get(s.path))
        if key in keys:
            groups[keys.index(key)].append(s.path)
        else:
            keys.append(key)
            groups.append([s.path])
    names = _get_unique_concatenated_bids_names(groups)
    return dict(zip(names, groups, strict=False))
```

- [ ] **Step 2: Rewrite `group_dwi_scans` to the new ordering**

Replace the body of `group_dwi_scans` with the orchestration below (keep the
exact signature and return tuple). The code is concrete; it reuses the helpers
already built in earlier tasks:

```python
def group_dwi_scans(layout, subject_data, combine_scans=True,
                    ignore_fieldmaps=False, estimate_per_axis=False):
    config.loggers.workflow.info('Grouping DWI scans')

    series_list = [DwiSeries.from_layout(layout, p) for p in subject_data['dwi']]

    if ignore_fieldmaps:
        estimators, series_to_estimator = ([], {})
    else:
        estimators, series_to_estimator = find_estimators(
            layout=layout, series=series_list,
            ignore_fieldmaps=ignore_fieldmaps, estimate_per_axis=estimate_per_axis,
        )

    if combine_scans:
        distortion_groups = _final_distortion_groups(series_list, series_to_estimator)
    else:
        # Each file is its own distortion group (separate-all-dwis).
        names = _get_unique_concatenated_bids_names([[s.path] for s in series_list])
        distortion_groups = {n: [s.path] for n, s in zip(names, series_list, strict=False)}

    fmap_estimation_groups, fmap_application_groups = _serialize_estimators(
        estimators, series_to_estimator, distortion_groups
    ) if not ignore_fieldmaps else (None, None)

    concatenation_groups = build_concatenation_groups(
        layout, subject_data, distortion_groups, combine_scans
    )

    validate_group_consistency(
        distortion_groups, fmap_estimation_groups, concatenation_groups, combine_scans
    )

    config.loggers.workflow.info('Finished grouping DWI scans')
    return distortion_groups, fmap_estimation_groups, fmap_application_groups, concatenation_groups
```

Implementer notes:
- `_serialize_estimators` (add it) turns the `FieldmapEstimation` list +
  application map into today's two dicts: `fmap_estimation_groups`
  (`bids_id -> [dg_ids and/or fmap paths]`) and `fmap_application_groups`
  (`bids_id -> [dg_ids]`). Map a DWI source path to its dg id via a
  path→dg lookup over `distortion_groups`; keep `fmap/` paths as paths.
- Keep `build_concatenation_groups` and `validate_group_consistency` as they are
  (they already operate on the distortion-group dict).

- [ ] **Step 3: Run the golden suite**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -8
```
Expected: all PASS. If any case differs, diff the failing snapshot against the
captured `actual` and adjust the **serializer/ordering** (not the snapshot) until
identical — per the spec, snapshots are the contract. Common culprits: `auto_`
counter order, list sort order, or key insertion order.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/utils/grouping.py
git commit -m "refactor: rebuild group_dwi_scans on DwiSeries + find_estimators

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 11: Delete the refine/remap dance and superseded helpers

**Files:**
- Modify: `qsiprep/utils/grouping.py` (delete dead functions)

- [ ] **Step 1: Delete functions now unused**

Remove from `qsiprep/utils/grouping.py`:
- `refine_distortion_groups`
- `_remap_groups_after_refinement`
- `_remap_group_members_with_new_dg_ids`
- Any of these now fully superseded by the value objects (verify each has no
  remaining references with grep before deleting):
  `classify_fmap_files`, `_find_sibling_fmap_file`, `get_highest_priority_fieldmap`,
  `FMAP_PRIORITY`, `build_fmap_estimation_groups`, `build_fmap_application_groups`,
  `_build_intendedfor_groups`, `_build_heuristic_estimation_groups`,
  `_check_b0field_axis_conflict`, `_dg_pe_direction`, `_get_distortion_group_signature`,
  `_get_member_session`, `_get_metadata_field` (if no longer used),
  `build_distortion_groups` (replaced by `_final_distortion_groups`).

- [ ] **Step 2: Verify no dangling references**

Run:
```bash
for fn in refine_distortion_groups _remap_groups_after_refinement classify_fmap_files get_highest_priority_fieldmap FMAP_PRIORITY build_fmap_estimation_groups; do
  echo "== $fn =="; grep -rn "$fn" qsiprep --include=*.py | grep -v "test_grouping_golden\|test_utils_fieldmaps"
done
```
Expected: only definition-site lines (about to be removed) or none; **no live
call sites** outside tests slated for porting in Phase 6. If `base.py` still
references a to-be-deleted function, that is handled in Phase 5 — sequence Phase 5
before deleting that specific function if needed.

- [ ] **Step 3: Run golden suite + fieldmaps unit tests**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py qsiprep/tests/test_utils_fieldmaps.py -p no:cacheprovider -q 2>&1 | tail -4
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/utils/grouping.py
git commit -m "refactor: delete refine/remap dance and superseded grouping helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 5 — Shrink the `base.py` adapter

### Task 12: Use `FieldmapEstimation.to_fieldmap_info()` in the adapter

**Files:**
- Modify: `qsiprep/workflows/base.py` (`_build_outputs_to_files`, imports)

- [ ] **Step 1: Rewrite `_build_outputs_to_files` to delegate fieldmap typing**

Today the adapter re-derives the suffix via `classify_fmap_files` /
`get_highest_priority_fieldmap`. Change it so that, for each distortion group, it
finds the owning `FieldmapEstimation` and calls `to_fieldmap_info(epi_files=...,
rpe_files=...)` with the target-specific EPI and reverse-PE DWI paths (computed as
today: `rpe_files` = DWI files from other dgs in the estimation group;
`epi_files` = the estimation group's `fmap/` EPI paths).

Concretely, `group_dwi_scans` should additionally expose the estimator objects so
the adapter can call `to_fieldmap_info`. Two acceptable shapes (pick one; the
golden suite verifies the resulting `fieldmap_info` either way):

- (a) Have `group_dwi_scans` stash `{bids_id: FieldmapEstimation}` on a returned
  structure the adapter can read, **or**
- (b) Keep the four-dict return unchanged and have the adapter reconstruct the
  `FieldmapEstimation` from the `fmap_estimation_groups` entry (build
  `FieldmapFile`s from the fmap paths; the GRE branch only needs the fmap paths +
  sibling discovery, which `to_fieldmap_info` already does).

Recommended: **(b)** — it keeps `group_dwi_scans`' return exactly as the golden
suite froze it. The adapter becomes:

```python
from ..utils.fieldmaps import EstimatorType, FieldmapEstimation, FieldmapFile


def _build_outputs_to_files(layout, distortion_groups, fmap_estimation_groups,
                            fmap_application_groups):
    fmap_estimation_groups = fmap_estimation_groups or {}
    fmap_application_groups = fmap_application_groups or {}

    dg_to_fmap_key = {}
    for fmap_key, dg_ids in fmap_application_groups.items():
        for dg_id in dg_ids:
            dg_to_fmap_key[dg_id] = fmap_key

    outputs_to_files = {}
    for dg_id, files in distortion_groups.items():
        pe_dir = layout.get_metadata(files[0]).get('PhaseEncodingDirection', '')
        fmap_key = dg_to_fmap_key.get(dg_id)
        if fmap_key is None:
            fieldmap_info = {'suffix': None}
        else:
            members = fmap_estimation_groups.get(fmap_key, [])
            fmap_paths = [m for m in members if m not in distortion_groups]
            rpe_dg_ids = [m for m in members if m in distortion_groups and m != dg_id]
            rpe_files = []
            for rpe_id in rpe_dg_ids:
                rpe_files.extend(distortion_groups[rpe_id])

            est = FieldmapEstimation(
                [FieldmapFile(p, metadata=layout.get_metadata(p)) for p in fmap_paths]
            ) if fmap_paths else None

            if est is not None and est.method in (EstimatorType.MAPPED, EstimatorType.PHASEDIFF):
                fieldmap_info = est.to_fieldmap_info()
            else:
                epi_files = [p for p in fmap_paths]  # PEPOLAR fmap EPIs
                pepolar = FieldmapEstimation(
                    [FieldmapFile(p, metadata=layout.get_metadata(p))
                     for p in fmap_paths + rpe_files]
                ) if (fmap_paths or rpe_files) else None
                fieldmap_info = (
                    pepolar.to_fieldmap_info(epi_files=epi_files, rpe_files=rpe_files)
                    if pepolar is not None else {'suffix': None}
                )

        outputs_to_files[dg_id] = {
            'concatenated_bids_name': dg_id,
            'dwi_series': files,
            'dwi_series_pedir': pe_dir,
            'fieldmap_info': fieldmap_info,
        }
    return outputs_to_files
```

Implementer note: this must reproduce the current `fieldmap_info` exactly —
including the `epi`-only vs `rpe_series` vs GRE branching. The golden snapshots
are the arbiter. Keep the `force_syn` override (which sets `fieldmap_info =
{'suffix': 'syn'}`) wherever it currently lives in `base.py`.

- [ ] **Step 2: Update imports and remove now-unused ones**

Remove the `from ..utils.grouping import (classify_fmap_files,
get_highest_priority_fieldmap, group_dwi_scans)` extras; keep `group_dwi_scans`.
Add the `fieldmaps` import shown above.

- [ ] **Step 3: Run golden suite + adapter regression tests**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py qsiprep/tests/test_utils_grouping.py -k "Golden or LegacyOutputAdapter or golden" -p no:cacheprovider -q 2>&1 | tail -5
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -3
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/workflows/base.py
git commit -m "refactor: build fieldmap_info via FieldmapEstimation in base adapter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 6 — Port helper-level tests; final verification

### Task 13: Port/remove tests that referenced deleted helpers

**Files:**
- Modify: `qsiprep/tests/test_utils_grouping.py`

- [ ] **Step 1: Inventory tests referencing deleted symbols**

Run:
```bash
grep -nE "classify_fmap_files|get_highest_priority_fieldmap|refine_distortion_groups|build_fmap_estimation_groups|build_fmap_application_groups|_build_heuristic_estimation_groups|_build_intendedfor_groups|_check_b0field_axis_conflict|_remap_" qsiprep/tests/test_utils_grouping.py
```
Expected: a list of test functions/asserts to port or remove.

- [ ] **Step 2: Port behavior-level intent; delete pure-internal tests**

For each hit:
- If the test asserts a **behavior** (e.g. "phasediff fmaps are typed", "estimate
  per axis requires opposite PEDs", "shim conflict raises"), rewrite it to call
  `group_dwi_scans` / `find_estimators` / `to_fieldmap_info` (the surviving API).
  The `TestLegacyOutputAdapter` tests already do this and should keep passing
  unchanged.
- If the test only checked a deleted private helper's mechanics (now covered by
  `test_utils_fieldmaps.py` + golden), delete it.

Keep `get_concatenated_bids_name`, `_get_unique_concatenated_bids_names`,
`_add_acq_entity`, `_get_common_bids_fields` and their tests — they survive.

- [ ] **Step 3: Run the full grouping + fieldmaps test set**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep/tests/test_utils_grouping.py qsiprep/tests/test_utils_fieldmaps.py qsiprep/tests/test_grouping_golden.py -p no:cacheprovider -q 2>&1 | tail -5
```
Expected: all PASS, no collection errors, no references to deleted symbols.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/tests/test_utils_grouping.py
git commit -m "test: port grouping tests onto value-object API; drop dead-helper tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 14: Lint, format, and full non-integration suite

**Files:** none (verification only)

- [ ] **Step 1: Ruff lint + format the changed files**

Run:
```bash
~/.local/bin/uvx "ruff@0.4.10" check qsiprep/utils/fieldmaps.py qsiprep/utils/grouping.py qsiprep/workflows/base.py qsiprep/tests/test_grouping_golden.py qsiprep/tests/test_utils_fieldmaps.py qsiprep/tests/test_utils_grouping.py
~/.local/bin/uvx "ruff@0.4.10" format --check qsiprep/utils/fieldmaps.py qsiprep/utils/grouping.py qsiprep/workflows/base.py qsiprep/tests/test_grouping_golden.py qsiprep/tests/test_utils_fieldmaps.py qsiprep/tests/test_utils_grouping.py
```
Expected: `All checks passed!` and `... already formatted`. (The pre-existing
`PT001` fixture-decorator warnings in `test_utils_grouping.py` at the old
`@pytest.fixture` lines are not introduced by this work; leave them unless the
project's style job flags them.)

- [ ] **Step 2: Run the full non-integration suite**

Run:
```bash
.pixi/envs/test/bin/python -m pytest qsiprep -m "not integration" -p no:cacheprovider -q 2>&1 | tail -6
```
Expected: same pass/skip/xfail profile as before the refactor, plus the new
tests. The 4 pre-existing ERRORs in `test_interfaces_dipy/freesurfer/mrtrix3`
(external-binary/`TypeError` setup issues) are unrelated and expected to remain.

- [ ] **Step 3: Confirm line-count reduction (interpretability goal)**

Run:
```bash
wc -l qsiprep/utils/grouping.py qsiprep/utils/fieldmaps.py
```
Expected: `grouping.py` is substantially smaller than its pre-refactor ~1185
lines; the two files together are smaller and each has a single clear
responsibility.

- [ ] **Step 4: Final commit (if any formatting changes)**

```bash
git add -A
git commit -m "style: ruff format/lint for grouping refactor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 15: Open the PR

- [ ] **Step 1: Push and open a PR targeting `group-dwi-scans` (or `main` if #967 has merged)**

Run:
```bash
git push -u origin refactor-scan-grouping
gh pr create --base group-dwi-scans --title "Refactor scan grouping & field map selection (value objects)" \
  --body "$(cat <<'EOF'
Refactors scan grouping and field map selection into SDCFlows-style value objects
(`qsiprep/utils/fieldmaps.py`: FieldmapFile/FieldmapEstimation/EstimatorType) and a
slimmed `grouping.py` (DwiSeries + find_estimators), with the four-dict output and
legacy `fieldmap_info` preserved exactly.

Behavior is frozen by a golden/characterization suite captured from `group-dwi-scans`
HEAD (`qsiprep/tests/test_grouping_golden.py`). The post-hoc refine/remap dance is
deleted; per-suffix field-map knowledge is centralized in one place.

Design: `docs/superpowers/specs/2026-06-10-scan-grouping-fieldmap-refactor-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL printed. Confirm CircleCI integration jobs (forrest_gump,
maternal_brain_project, drbuddi_*) are triggered — they are the end-to-end check
that the preserved `fieldmap_info` drives the same workflows.

---

## Notes for the executor

- **The golden suite is the contract.** If a snapshot diff appears during Phases
  4–5, the default assumption is a bug in the new code, not a stale snapshot.
  Only regenerate snapshots (`QSIPREP_REGEN_GOLDEN=1`) if you have *deliberately
  and knowingly* changed behavior — which this refactor must not do.
- **Order matters between Phase 4 Task 11 and Phase 5.** If `base.py` still
  imports a helper you are about to delete, do Phase 5 (Task 12) first, or delete
  that specific symbol only after Task 12. The grep in Task 11 Step 2 catches
  this.
- **`find_estimators` (Task 9) is the porting-heavy task.** Budget the most time
  there; it must reproduce the existing priority dispatch, `estimate_per_axis`
  behavior, auto-id ordering, and the shim/TRT and per-axis error conditions.
  The golden suite covers all of these via the existing conflict fixtures
  (`dset_b0field_shim_conflict`, `dset_intendedfor_trt_conflict`,
  `dset_b0field_cross_axis`, `dset_multirun_multiped_conflicting_concat`).
```
