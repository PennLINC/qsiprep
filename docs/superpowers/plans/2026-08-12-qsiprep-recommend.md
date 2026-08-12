# qsiprep-recommend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `qsiprep-recommend` command that reads a BIDS dataset and prints the QSIPrep flags the documentation recommends for that data, with the reasoning for each.

**Architecture:** A new `qsiprep/recommend/` package with a one-way pipeline: `probe.py` turns BIDS into `SubjectFacts` dataclasses (the only pybids-aware module, reusing `collect_data` and `group_dwi_scans`), `profiles.py` groups subjects by acquisition signature, `rules.py` is an ordered registry of pure functions from facts to `Recommendation`s, and `report.py` renders text. A thin CLI in `qsiprep/cli/recommend.py` orchestrates them.

**Tech Stack:** Python 3.10+, pybids (`<0.16`), nibabel, numpy, pytest, Sphinx (`sphinxarg.ext`, custom directive). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-12-qsiprep-recommend-design.md`

**Issue:** https://github.com/PennLINC/qsiprep/issues/1081

## Global Constraints

- Environment for every command: `micromamba run -n lincapps <command>`. Not pixi, not conda.
- Line length 99 (`[tool.ruff] line-length = 99`). Single quotes (`inline-quotes = "single"`).
- Every new Python module starts with the repo's two-line editor header:
  ```python
  # emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
  # vi: set ft=python sts=4 ts=4 sw=4 et:
  ```
  followed by a module docstring.
- Ruff has `BLE` (blind-except) enabled. Any `except Exception` needs `# noqa: BLE001`.
- Unit tests must not be marked `integration` — `addopts = '-m "not integration"'` means unmarked tests run by default, which is what we want.
- Constants fixed by the spec: b=0 threshold 100 s/mm², shell clustering tolerance 100 s/mm², at most 4 shells, at least 6 directions per shell, fewer than 30 volumes triggers the denoising warning, voxel sizes round to 2 decimal places, infant cutoff 60 months.
- Recommendation severities are exactly: `recommended`, `consider`, `note`, `warning`, `undetermined`.
- No rule may emit a flag that `qsiprep.cli.parser._build_parser()` does not define, and none may emit `--dwi-only`.

---

## File Structure

| File | Responsibility |
|---|---|
| `qsiprep/recommend/__init__.py` | Package marker, exports `probe_dataset`, `build_profiles`, `evaluate`, `render_report` |
| `qsiprep/recommend/facts.py` | The vocabulary: `SeriesFacts`, `GroupFacts`, `SubjectFacts`, `Recommendation`, plus `classify_sampling_scheme` |
| `qsiprep/recommend/probe.py` | BIDS ingress; the only pybids-aware module |
| `qsiprep/recommend/profiles.py` | Acquisition signature and subject grouping |
| `qsiprep/recommend/rules.py` | Ordered rule registry and `evaluate()` |
| `qsiprep/recommend/report.py` | Text rendering and command construction |
| `qsiprep/cli/recommend.py` | argparse + orchestration + `main()` |
| `qsiprep/tests/recommend_fixtures.py` | Fixture factory building real BIDS trees with valid NIfTI/bval/bvec |
| `qsiprep/tests/test_recommend_facts.py` | Scheme classification tests |
| `qsiprep/tests/test_recommend_probe.py` | Ingress tests against generated trees |
| `qsiprep/tests/test_recommend_profiles.py` | Signature and grouping tests |
| `qsiprep/tests/test_recommend_rules.py` | Per-rule tests over hand-built facts |
| `qsiprep/tests/test_recommend_report.py` | Rendering tests |
| `qsiprep/tests/test_recommend_cli.py` | End-to-end CLI tests and the parser invariant |
| `docs/sphinxext/recommend_rules.py` | Sphinx directive rendering the rule registry |
| `docs/recommend.rst` | User-facing page |

**Note on the spec:** the spec lists four modules (`probe`, `profiles`, `rules`, `report`). This plan adds a fifth, `facts.py`, holding the dataclasses that all four share. Without it, `rules.py` would have to import from `probe.py` purely for type definitions, which would drag pybids into the pure-function layer and make the rule tests slow to import.

---

### Task 1: Facts vocabulary and sampling-scheme classification

**Files:**
- Create: `qsiprep/recommend/__init__.py`
- Create: `qsiprep/recommend/facts.py`
- Test: `qsiprep/tests/test_recommend_facts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Recommendation(flag: str | None, value: str | None, rationale: str, docs_anchor: str, severity: str)`; `SeriesFacts`; `GroupFacts`; `SubjectFacts`; `classify_sampling_scheme(bvals, b0_threshold=100, tolerance=100) -> tuple[bool, tuple[int, ...], int]` returning `(is_shelled, shell_centers, n_unique_bvals)`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_facts.py`:

```python
"""Tests for the recommender's fact vocabulary."""

import numpy as np

from qsiprep.recommend.facts import classify_sampling_scheme


def test_single_shell_is_shelled():
    bvals = [0, 0, 0] + [1000] * 30
    is_shelled, shells, n_unique = classify_sampling_scheme(bvals)
    assert is_shelled is True
    assert shells == (1000,)
    assert n_unique == 1


def test_shell_jitter_within_tolerance_is_one_shell():
    rng = np.random.default_rng(0)
    bvals = [0, 0] + list(1000 + rng.integers(-40, 40, size=30))
    is_shelled, shells, _ = classify_sampling_scheme(bvals)
    assert is_shelled is True
    assert len(shells) == 1


def test_multi_shell_is_shelled():
    bvals = [0] * 5 + [1000] * 30 + [2000] * 30 + [3000] * 30
    is_shelled, shells, n_unique = classify_sampling_scheme(bvals)
    assert is_shelled is True
    assert shells == (1000, 2000, 3000)
    assert n_unique == 3


def test_cs_dsi_is_not_shelled():
    bvals = [0] * 5 + list(range(300, 5000, 85))
    is_shelled, _, n_unique = classify_sampling_scheme(bvals)
    assert is_shelled is False
    assert n_unique > 4


def test_five_shells_is_not_shelled():
    bvals = [0] * 5
    for center in (1000, 2000, 3000, 4000, 5000):
        bvals += [center] * 10
    is_shelled, shells, _ = classify_sampling_scheme(bvals)
    assert is_shelled is False
    assert len(shells) == 5


def test_shell_with_too_few_directions_is_not_shelled():
    bvals = [0] * 5 + [1000] * 30 + [2500] * 4
    is_shelled, shells, _ = classify_sampling_scheme(bvals)
    assert is_shelled is False
    assert len(shells) == 2


def test_b0_only_series_is_not_shelled():
    is_shelled, shells, n_unique = classify_sampling_scheme([0, 0, 0])
    assert is_shelled is False
    assert shells == ()
    assert n_unique == 0


def test_values_below_threshold_are_treated_as_b0():
    bvals = [5, 20, 99] + [1000] * 30
    _, shells, n_unique = classify_sampling_scheme(bvals)
    assert shells == (1000,)
    assert n_unique == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_facts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.recommend'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/recommend/__init__.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Recommend QSIPrep command-line flags based on a BIDS dataset."""
```

Create `qsiprep/recommend/facts.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Data structures describing a BIDS dataset for the flag recommender.

Everything in this module is pure: it derives facts from arrays and metadata
without touching the filesystem or pybids, so the rules that consume these
structures can be tested without a dataset on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Any b-value below this is treated as a b=0 image.
B0_THRESHOLD = 100
#: b-values within this distance of one another belong to the same shell.
SHELL_TOLERANCE = 100
#: More clusters than this means the scheme is not shelled.
MAX_SHELLS = 4
#: A shell with fewer directions than this does not count as a shell.
MIN_DIRECTIONS_PER_SHELL = 6
#: The oldest age with an MNIInfant cohort, in months. Used by both the profile
#: signature and the ``infant`` rule, so it lives here rather than in either.
INFANT_MAX_MONTHS = 60

Severity = Literal['recommended', 'consider', 'note', 'warning', 'undetermined']


@dataclass(frozen=True)
class Recommendation:
    """A single piece of advice about one QSIPrep flag.

    ``flag`` is ``None`` for advice that does not correspond to a flag, which is
    the case for ``note`` and most ``warning`` entries. ``value`` is ``None``
    for store-true flags such as ``--infant``.
    """

    rationale: str
    severity: Severity
    flag: str | None = None
    value: str | None = None
    docs_anchor: str = ''


@dataclass(frozen=True)
class SeriesFacts:
    """What we know about one DWI series."""

    path: str
    n_volumes: int
    n_b0s: int
    shells: tuple[int, ...]
    is_shelled: bool
    n_unique_bvals: int
    voxel_size: tuple[float, float, float]
    pe_direction: str | None = None
    partial_fourier: float | None = None
    image_type: tuple[str, ...] = ()
    multipart_id: str | None = None


@dataclass(frozen=True)
class GroupFacts:
    """One group of DWI series that QSIPrep would process together."""

    concatenated_bids_name: str
    n_series: int
    fieldmap_suffix: str | None
    pe_direction: str


@dataclass(frozen=True)
class SubjectFacts:
    """Everything the rules are allowed to know about one subject."""

    subject_id: str
    sessions: tuple[str, ...] = ()
    series: tuple[SeriesFacts, ...] = ()
    groups: tuple[GroupFacts, ...] = ()
    has_t1w: bool = False
    has_t2w: bool = False
    has_phase_data: bool = False
    age_months: int | None = None


def classify_sampling_scheme(
    bvals,
    b0_threshold: int = B0_THRESHOLD,
    tolerance: int = SHELL_TOLERANCE,
) -> tuple[bool, tuple[int, ...], int]:
    """Decide whether a set of b-values forms shells.

    b-values are sorted, b=0 volumes are dropped, and a new cluster starts
    wherever the gap between consecutive b-values exceeds ``tolerance``. The
    scheme is shelled when there are at most :data:`MAX_SHELLS` clusters and
    every cluster holds at least :data:`MIN_DIRECTIONS_PER_SHELL` directions.

    Parameters
    ----------
    bvals : array-like
        The b-values of one DWI series.
    b0_threshold : :obj:`int`
        b-values below this are considered b=0 and excluded.
    tolerance : :obj:`int`
        Maximum gap within a single shell.

    Returns
    -------
    is_shelled : :obj:`bool`
    shells : :obj:`tuple` of :obj:`int`
        The rounded center of each cluster, in ascending order.
    n_unique_bvals : :obj:`int`
        How many distinct non-b=0 b-values are present, after rounding.
    """
    bvals = np.asarray(bvals, dtype=float)
    dwi_bvals = np.sort(bvals[bvals >= b0_threshold])
    if dwi_bvals.size == 0:
        return False, (), 0

    n_unique = int(np.unique(np.round(dwi_bvals)).size)
    breaks = np.nonzero(np.diff(dwi_bvals) > tolerance)[0]
    clusters = np.split(dwi_bvals, breaks + 1)
    shells = tuple(int(round(float(cluster.mean()))) for cluster in clusters)
    is_shelled = len(clusters) <= MAX_SHELLS and all(
        cluster.size >= MIN_DIRECTIONS_PER_SHELL for cluster in clusters
    )
    return is_shelled, shells, n_unique
```

Note: every dataclass here has immutable defaults, so `dataclasses.field` is deliberately not imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_facts.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Lint**

Run: `micromamba run -n lincapps python -m ruff check qsiprep/recommend/ qsiprep/tests/test_recommend_facts.py`
Expected: no errors (remove the unused `field` import if F401 is reported)

- [ ] **Step 6: Commit**

```bash
git add qsiprep/recommend/__init__.py qsiprep/recommend/facts.py qsiprep/tests/test_recommend_facts.py
git commit -m "Add fact vocabulary and sampling-scheme classification for the recommender"
```

---

### Task 2: BIDS fixture factory

**Files:**
- Create: `qsiprep/tests/recommend_fixtures.py`
- Test: `qsiprep/tests/test_recommend_probe.py` (first test only; the rest arrives in Task 3)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `make_dataset(root, subjects, ...) -> Path` and the `DwiSpec` dataclass used by every ingress test.

**Why this exists:** `generate_bids_skeleton` writes zero-length `.nii.gz` files. The probe reads voxel sizes from NIfTI headers and b-values from `.bval` files, and `collect_data` raises `RuntimeError` on any DWI that is not 4D. So the fixtures need real, tiny images.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_probe.py`:

```python
"""Tests for the recommender's BIDS ingress."""

import json

import nibabel as nb
import numpy as np

from qsiprep.tests.recommend_fixtures import DwiSpec, make_dataset


def test_fixture_factory_writes_loadable_4d_images(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[DwiSpec(entities={'dir': 'AP'}, bvals=[0] * 3 + [1000] * 30, voxel_size=2.0)],
    )

    dwi = bids_dir / 'sub-01' / 'dwi' / 'sub-01_dir-AP_dwi.nii.gz'
    img = nb.load(dwi)
    assert img.ndim == 4
    assert img.shape[3] == 33
    assert img.header.get_zooms()[:3] == (2.0, 2.0, 2.0)

    bvals = np.loadtxt(dwi.parent / 'sub-01_dir-AP_dwi.bval', ndmin=1)
    assert bvals.size == 33

    bvecs = np.loadtxt(dwi.parent / 'sub-01_dir-AP_dwi.bvec', ndmin=2)
    assert bvecs.shape == (3, 33)

    sidecar = json.loads((dwi.parent / 'sub-01_dir-AP_dwi.json').read_text())
    assert sidecar['PhaseEncodingDirection'] == 'j'

    assert (bids_dir / 'dataset_description.json').exists()
    assert (bids_dir / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz').exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.tests.recommend_fixtures'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/tests/recommend_fixtures.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Build small but valid BIDS datasets for recommender tests.

``generate_bids_skeleton`` writes empty NIfTI files, which the recommender's
probe cannot read: it needs voxel sizes from image headers and b-values from
``.bval`` files. These helpers write real 4x4x4xN images instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nb
import numpy as np

DATASET_DESCRIPTION = {
    'Name': 'recommender test dataset',
    'BIDSVersion': '1.8.0',
    'DatasetType': 'raw',
}


@dataclass
class DwiSpec:
    """Description of one DWI series to write."""

    bvals: list[float]
    entities: dict[str, str] = field(default_factory=dict)
    voxel_size: float | tuple[float, float, float] = 2.0
    pe_direction: str = 'j'
    metadata: dict = field(default_factory=dict)
    suffix: str = 'dwi'
    datatype: str = 'dwi'


def _entity_string(entities: dict[str, str]) -> str:
    order = ['ses', 'acq', 'dir', 'run', 'part']
    parts = [f'{key}-{entities[key]}' for key in order if key in entities]
    return '_'.join(parts)


def _write_image(path: Path, n_volumes: int, voxel_size) -> None:
    if isinstance(voxel_size, int | float):
        voxel_size = (voxel_size, voxel_size, voxel_size)
    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    shape = (4, 4, 4) if n_volumes is None else (4, 4, 4, n_volumes)
    data = np.zeros(shape, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(data, affine).to_filename(str(path))


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))


def _write_gradients(stem: Path, bvals: list[float]) -> None:
    bvals = np.asarray(bvals, dtype=float)
    np.savetxt(str(stem.with_suffix('.bval')), bvals[None, :], fmt='%g')

    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(3, bvals.size))
    bvecs /= np.linalg.norm(bvecs, axis=0, keepdims=True)
    bvecs[:, bvals < 100] = 0.0
    np.savetxt(str(stem.with_suffix('.bvec')), bvecs, fmt='%.6f')


def make_dataset(
    root,
    subjects,
    dwis,
    sessions=None,
    anat=('T1w',),
    fieldmaps=(),
    ages=None,
) -> Path:
    """Write a BIDS dataset and return its root.

    Parameters
    ----------
    root : :obj:`os.PathLike`
        Where to create the dataset.
    subjects : :obj:`list` of :obj:`str`
        Subject labels without the ``sub-`` prefix.
    dwis : :obj:`list` of :obj:`DwiSpec`
        DWI series written for every subject and session.
    sessions : :obj:`list` of :obj:`str` or :obj:`None`
        Session labels without the ``ses-`` prefix. ``None`` means no session
        level.
    anat : :obj:`tuple` of :obj:`str`
        Anatomical suffixes to write, for example ``('T1w', 'T2w')``. An empty
        tuple writes no anatomical data.
    fieldmaps : :obj:`tuple` of :obj:`dict`
        Each dict needs ``suffix`` (for example ``'epi'``), ``entities``, and
        ``metadata``. ``IntendedFor`` is filled in automatically when absent.
    ages : :obj:`dict` or :obj:`None`
        Maps subject label to age in months, written to ``participants.tsv``.

    Returns
    -------
    :obj:`pathlib.Path`
        The dataset root.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / 'dataset_description.json', DATASET_DESCRIPTION)

    for subject in subjects:
        for session in sessions or [None]:
            subject_dir = root / f'sub-{subject}'
            base = subject_dir / f'ses-{session}' if session else subject_dir
            prefix = f'sub-{subject}' + (f'_ses-{session}' if session else '')

            for suffix in anat:
                _write_image(base / 'anat' / f'{prefix}_{suffix}.nii.gz', None, 1.0)
                _write_json(base / 'anat' / f'{prefix}_{suffix}.json', {})

            intended_for = []
            for spec in dwis:
                entities = dict(spec.entities)
                entities.pop('ses', None)
                name = '_'.join(filter(None, [prefix, _entity_string(entities), spec.suffix]))
                target = base / spec.datatype / f'{name}.nii.gz'
                _write_image(target, len(spec.bvals), spec.voxel_size)
                _write_gradients(target.parent / name, spec.bvals)

                metadata = {
                    'PhaseEncodingDirection': spec.pe_direction,
                    'TotalReadoutTime': 0.05,
                }
                metadata.update(spec.metadata)
                _write_json(base / spec.datatype / f'{name}.json', metadata)

                relative = str(target.relative_to(subject_dir))
                intended_for.append(relative.replace('\\', '/'))

            for fmap in fieldmaps:
                entities = dict(fmap.get('entities', {}))
                name = '_'.join(
                    filter(None, [prefix, _entity_string(entities), fmap['suffix']])
                )
                target = base / 'fmap' / f'{name}.nii.gz'
                _write_image(target, fmap.get('n_volumes', 3), fmap.get('voxel_size', 2.0))
                metadata = {
                    'PhaseEncodingDirection': fmap.get('pe_direction', 'j-'),
                    'TotalReadoutTime': 0.05,
                    'IntendedFor': intended_for,
                }
                metadata.update(fmap.get('metadata', {}))
                _write_json(base / 'fmap' / f'{name}.json', metadata)

    if ages:
        lines = ['participant_id\tage']
        lines += [f'sub-{subject}\t{ages[subject]}' for subject in subjects if subject in ages]
        (root / 'participants.tsv').write_text('\n'.join(lines) + '\n')
        _write_json(
            root / 'participants.json',
            {'age': {'Description': 'age of the participant', 'Units': 'months'}},
        )

    return root
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_probe.py -v`
Expected: PASS, 1 test

- [ ] **Step 5: Commit**

```bash
git add qsiprep/tests/recommend_fixtures.py qsiprep/tests/test_recommend_probe.py
git commit -m "Add BIDS fixture factory for recommender tests"
```

---

### Task 3: BIDS ingress

**Files:**
- Create: `qsiprep/recommend/probe.py`
- Modify: `qsiprep/tests/test_recommend_probe.py` (append tests)

**Interfaces:**
- Consumes: `SeriesFacts`, `GroupFacts`, `SubjectFacts`, `classify_sampling_scheme` from Task 1; `make_dataset`, `DwiSpec` from Task 2.
- Produces: `probe_dataset(bids_dir, participant_label=None, session_id=None, filters=None, bids_validate=True, database_dir=None) -> tuple[list[SubjectFacts], list[tuple[str, str]]]`, returning facts and a list of `(subject_id, reason)` skips.

**Reference — shapes this task depends on:**
- `collect_data(bids_dir_or_layout, participant_label, session_id=None, filters=None, bids_validate=True, ignore=None)` returns `(subj_data, layout)`; `subj_data` has keys `fmap`, `t2w`, `t1w`, `roi`, `dwi`, each a sorted list of paths.
- `group_dwi_scans(layout, subject_data, combine_scans=True, ignore_fieldmaps=False)` returns `(groups, concatenation_grouping)`; each group dict has `dwi_series`, `fieldmap_info`, `dwi_series_pedir`, `concatenated_bids_name`. `fieldmap_info['suffix']` is `None`, `'epi'`, `'dwi'`, `'rpe_series'`, `'phasediff'`, etc.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_recommend_probe.py`:

```python
from qsiprep.recommend.probe import probe_dataset


def test_probe_reads_single_shell_subject(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[
            DwiSpec(
                entities={'dir': 'AP'},
                bvals=[0] * 3 + [1000] * 30,
                voxel_size=2.0,
                metadata={'PartialFourier': 0.75, 'ImageType': ['ORIGINAL', 'PRIMARY', 'NORM']},
            )
        ],
        fieldmaps=({'suffix': 'epi', 'entities': {'dir': 'PA'}},),
    )

    facts, skipped = probe_dataset(bids_dir, bids_validate=False)

    assert skipped == []
    assert len(facts) == 1
    subject = facts[0]
    assert subject.subject_id == '01'
    assert subject.has_t1w is True
    assert subject.has_t2w is False
    assert len(subject.series) == 1

    series = subject.series[0]
    assert series.n_volumes == 33
    assert series.n_b0s == 3
    assert series.is_shelled is True
    assert series.shells == (1000,)
    assert series.voxel_size == (2.0, 2.0, 2.0)
    assert series.pe_direction == 'j'
    assert series.partial_fourier == 0.75
    assert 'NORM' in series.image_type

    assert len(subject.groups) == 1
    assert subject.groups[0].fieldmap_suffix == 'epi'


def test_probe_reports_non_shelled_scheme(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[DwiSpec(entities={'acq': 'csdsi'}, bvals=[0] * 5 + list(range(300, 5000, 85)))],
    )

    facts, _ = probe_dataset(bids_dir, bids_validate=False)

    assert facts[0].series[0].is_shelled is False
    assert facts[0].series[0].n_unique_bvals > 4


def test_probe_groups_multiple_runs(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[
            DwiSpec(entities={'run': '1'}, bvals=[0] * 3 + [1000] * 30),
            DwiSpec(entities={'run': '2'}, bvals=[0] * 3 + [1000] * 30),
        ],
    )

    facts, _ = probe_dataset(bids_dir, bids_validate=False)

    assert len(facts[0].series) == 2
    assert len(facts[0].groups) == 1
    assert facts[0].groups[0].n_series == 2


def test_probe_records_sessions_and_age(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        sessions=['A', 'B'],
        dwis=[DwiSpec(bvals=[0] * 3 + [1000] * 30)],
        ages={'01': 9},
    )

    facts, _ = probe_dataset(bids_dir, bids_validate=False)

    assert facts[0].sessions == ('A', 'B')
    assert facts[0].age_months == 9


def test_probe_detects_missing_anatomicals(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[DwiSpec(bvals=[0] * 3 + [1000] * 30)],
        anat=('T2w',),
    )

    facts, _ = probe_dataset(bids_dir, bids_validate=False)

    assert facts[0].has_t1w is False
    assert facts[0].has_t2w is True


def test_probe_skips_subjects_without_dwi(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[DwiSpec(bvals=[0] * 3 + [1000] * 30)],
    )
    # sub-02 has an anatomical but no DWI data.
    make_dataset(bids_dir, subjects=['02'], dwis=[], anat=('T1w',))

    facts, skipped = probe_dataset(bids_dir, bids_validate=False)

    assert [subject.subject_id for subject in facts] == ['01']
    assert skipped == [('02', 'no DWI data')]


def test_probe_honors_participant_label(tmp_path):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01', '02'],
        dwis=[DwiSpec(bvals=[0] * 3 + [1000] * 30)],
    )

    facts, _ = probe_dataset(bids_dir, participant_label=['02'], bids_validate=False)

    assert [subject.subject_id for subject in facts] == ['02']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.recommend.probe'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/recommend/probe.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Turn a BIDS dataset into the facts the recommender's rules consume.

This is the only module in :mod:`qsiprep.recommend` that knows about pybids.
It reuses QSIPrep's own ingress -- :func:`~qsiprep.utils.bids.collect_data` and
:func:`~qsiprep.utils.grouping.group_dwi_scans` -- so that anything the report
says about scan grouping is what QSIPrep will actually do.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout

from ..utils.bids import collect_data, collect_participants, parse_bids_for_age_months
from ..utils.grouping import group_dwi_scans
from .facts import GroupFacts, SeriesFacts, SubjectFacts, classify_sampling_scheme


def probe_dataset(
    bids_dir,
    participant_label=None,
    session_id=None,
    filters=None,
    bids_validate=True,
    database_dir=None,
) -> tuple[list[SubjectFacts], list[tuple[str, str]]]:
    """Collect facts about every subject in a BIDS dataset.

    Parameters
    ----------
    bids_dir : :obj:`os.PathLike`
        Root of the BIDS dataset.
    participant_label : :obj:`list` of :obj:`str` or :obj:`None`
        Restrict the analysis to these subjects.
    session_id : :obj:`str` or :obj:`None`
        Restrict the analysis to one session.
    filters : :obj:`dict` or :obj:`None`
        A BIDS filter dictionary, as accepted by ``collect_data``.
    bids_validate : :obj:`bool`
        Whether pybids should validate the dataset while indexing.
    database_dir : :obj:`os.PathLike` or :obj:`None`
        Where to read or write the pybids database, to avoid re-indexing.

    Returns
    -------
    facts : :obj:`list` of :obj:`~qsiprep.recommend.facts.SubjectFacts`
    skipped : :obj:`list` of :obj:`tuple`
        ``(subject_id, reason)`` for every subject that was passed over.
    """
    bids_dir = Path(bids_dir)
    if not bids_dir.exists():
        raise FileNotFoundError(f'BIDS directory does not exist: {bids_dir}')

    layout = BIDSLayout(
        str(bids_dir),
        validate=bids_validate,
        database_path=str(database_dir) if database_dir else None,
    )
    subjects = collect_participants(layout, participant_label=participant_label)

    facts = []
    skipped = []
    for subject in subjects:
        try:
            subject_facts = probe_subject(
                layout,
                bids_dir,
                subject,
                session_id=session_id,
                filters=filters,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append((subject, f'{type(exc).__name__}: {exc}'))
            continue

        if subject_facts is None:
            skipped.append((subject, 'no DWI data'))
            continue

        facts.append(subject_facts)

    return facts, skipped


def probe_subject(layout, bids_dir, subject, session_id=None, filters=None):
    """Collect facts about one subject, or return ``None`` when it has no DWI data."""
    subject_data, _ = collect_data(
        layout,
        subject,
        session_id=session_id,
        filters=filters,
        bids_validate=False,
    )
    if not subject_data['dwi']:
        return None

    groups, _ = group_dwi_scans(
        layout,
        subject_data,
        combine_scans=True,
        ignore_fieldmaps=False,
    )

    series = tuple(_series_facts(path, layout) for path in subject_data['dwi'])
    group_facts = tuple(
        GroupFacts(
            concatenated_bids_name=group.get('concatenated_bids_name', ''),
            n_series=len(group.get('dwi_series', [])),
            fieldmap_suffix=(group.get('fieldmap_info') or {}).get('suffix'),
            pe_direction=group.get('dwi_series_pedir', ''),
        )
        for group in groups
    )

    sessions = tuple(sorted(layout.get_sessions(subject=subject)))
    phase_files = layout.get(
        subject=subject,
        part='phase',
        suffix='dwi',
        extension=['nii', 'nii.gz'],
        return_type='file',
    )

    return SubjectFacts(
        subject_id=subject,
        sessions=sessions,
        series=series,
        groups=group_facts,
        has_t1w=bool(subject_data['t1w']),
        has_t2w=bool(subject_data['t2w']),
        has_phase_data=bool(phase_files),
        age_months=parse_bids_for_age_months(bids_dir, subject, session_id=session_id),
    )


def _series_facts(path, layout) -> SeriesFacts:
    """Reduce one DWI file to a :class:`~qsiprep.recommend.facts.SeriesFacts`."""
    img = nb.load(path)
    zooms = img.header.get_zooms()
    voxel_size = tuple(round(float(zoom), 2) for zoom in zooms[:3])
    n_volumes = int(img.shape[3]) if img.ndim == 4 else 1

    bvals = _load_bvals(path, layout)
    if bvals is None:
        is_shelled, shells, n_unique, n_b0s = False, (), 0, 0
    else:
        is_shelled, shells, n_unique = classify_sampling_scheme(bvals)
        n_b0s = int(np.sum(np.asarray(bvals, dtype=float) < 100))

    metadata = layout.get_metadata(path)
    image_type = metadata.get('ImageType') or []

    return SeriesFacts(
        path=str(path),
        n_volumes=n_volumes,
        n_b0s=n_b0s,
        shells=shells,
        is_shelled=is_shelled,
        n_unique_bvals=n_unique,
        voxel_size=voxel_size,
        pe_direction=metadata.get('PhaseEncodingDirection'),
        partial_fourier=metadata.get('PartialFourier'),
        image_type=tuple(str(item) for item in image_type),
        multipart_id=metadata.get('MultipartID'),
    )


def _load_bvals(dwi_file, layout):
    """Return the b-values for ``dwi_file``, or ``None`` when they cannot be found."""
    bval_file = layout.get_bval(dwi_file)
    if not bval_file or not os.path.exists(bval_file):
        return None
    return np.loadtxt(bval_file, ndmin=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_probe.py -v`
Expected: PASS, 8 tests

If `collect_participants(layout, ...)` rejects a layout argument, pass `bids_dir` instead — check its signature at `qsiprep/utils/bids.py:106` and adapt. If `layout.get_bval` does not exist in the installed pybids, replace `_load_bvals` with a sibling-file lookup that swaps the `.nii.gz`/`.nii` extension for `.bval`.

- [ ] **Step 5: Lint**

Run: `micromamba run -n lincapps python -m ruff check qsiprep/recommend/ qsiprep/tests/`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add qsiprep/recommend/probe.py qsiprep/tests/test_recommend_probe.py
git commit -m "Add BIDS ingress for the recommender"
```

---

### Task 4: Acquisition profiles

**Files:**
- Create: `qsiprep/recommend/profiles.py`
- Test: `qsiprep/tests/test_recommend_profiles.py`

**Interfaces:**
- Consumes: `SubjectFacts`, `SeriesFacts`, `GroupFacts` from Task 1.
- Produces: `AcquisitionProfile(signature, subjects, facts)`; `acquisition_signature(facts) -> tuple`; `build_profiles(all_facts) -> list[AcquisitionProfile]`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_profiles.py`:

```python
"""Tests for grouping subjects into acquisition profiles."""

from qsiprep.recommend.facts import GroupFacts, SeriesFacts, SubjectFacts
from qsiprep.recommend.profiles import acquisition_signature, build_profiles


def _series(**overrides):
    defaults = {
        'path': '/data/sub-01_dwi.nii.gz',
        'n_volumes': 33,
        'n_b0s': 3,
        'shells': (1000,),
        'is_shelled': True,
        'n_unique_bvals': 1,
        'voxel_size': (2.0, 2.0, 2.0),
        'pe_direction': 'j',
    }
    defaults.update(overrides)
    return SeriesFacts(**defaults)


def _subject(subject_id='01', **overrides):
    defaults = {
        'sessions': (),
        'series': (_series(),),
        'groups': (GroupFacts('sub-01', 1, 'epi', 'j'),),
        'has_t1w': True,
        'has_t2w': False,
    }
    defaults.update(overrides)
    return SubjectFacts(subject_id=subject_id, **defaults)


def test_identical_subjects_form_one_profile():
    profiles = build_profiles([_subject('01'), _subject('02'), _subject('03')])

    assert len(profiles) == 1
    assert profiles[0].subjects == ['01', '02', '03']


def test_different_schemes_form_separate_profiles():
    shelled = _subject('01')
    non_shelled = _subject(
        '02',
        series=(_series(is_shelled=False, shells=(300, 900, 2500), n_unique_bvals=40),),
    )

    profiles = build_profiles([shelled, non_shelled])

    assert len(profiles) == 2
    assert [profile.subjects for profile in profiles] == [['01'], ['02']]


def test_different_voxel_sizes_form_separate_profiles():
    coarse = _subject('01')
    fine = _subject('02', series=(_series(voxel_size=(1.5, 1.5, 1.5)),))

    assert len(build_profiles([coarse, fine])) == 2


def test_different_fieldmaps_form_separate_profiles():
    with_fmap = _subject('01')
    without = _subject('02', groups=(GroupFacts('sub-02', 1, None, 'j'),))

    assert len(build_profiles([with_fmap, without])) == 2


def test_signature_ignores_subject_identity():
    assert acquisition_signature(_subject('01')) == acquisition_signature(_subject('99'))


def test_profiles_preserve_first_seen_order():
    first = _subject('05')
    second = _subject('02', groups=(GroupFacts('sub-02', 1, None, 'j'),))
    third = _subject('01')

    profiles = build_profiles([first, second, third])

    assert profiles[0].subjects == ['01', '05']
    assert profiles[1].subjects == ['02']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.recommend.profiles'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/recommend/profiles.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Group subjects that share an acquisition into a single recommendation block."""

from __future__ import annotations

from dataclasses import dataclass, field

from .facts import INFANT_MAX_MONTHS, SubjectFacts


@dataclass
class AcquisitionProfile:
    """A set of subjects whose data warrant the same recommendations."""

    signature: tuple
    facts: SubjectFacts
    subjects: list[str] = field(default_factory=list)


def acquisition_signature(facts: SubjectFacts) -> tuple:
    """Reduce a subject to the acquisition properties that drive recommendations.

    Subject identity, run names, and file paths are deliberately excluded, so
    that subjects scanned the same way collapse into one profile.
    """
    series = facts.series
    n_sessions = max(len(facts.sessions), 1)
    partial_fourier = sorted({s.partial_fourier for s in series if s.partial_fourier is not None})

    return (
        tuple(sorted({s.is_shelled for s in series})),
        tuple(sorted({s.shells for s in series})),
        tuple(sorted({s.pe_direction or '' for s in series})),
        tuple(sorted({s.voxel_size for s in series})),
        tuple(sorted({g.fieldmap_suffix or '' for g in facts.groups})),
        (facts.has_t1w, facts.has_t2w),
        len(series) // n_sessions,
        len(facts.sessions),
        tuple(partial_fourier),
        any('NORM' in s.image_type for s in series),
        facts.age_months is not None and facts.age_months <= INFANT_MAX_MONTHS,
    )


def build_profiles(all_facts) -> list[AcquisitionProfile]:
    """Group subjects by acquisition signature, preserving first-seen order.

    Parameters
    ----------
    all_facts : :obj:`list` of :obj:`~qsiprep.recommend.facts.SubjectFacts`

    Returns
    -------
    :obj:`list` of :obj:`AcquisitionProfile`
        Profiles in the order their first subject was seen, each holding a
        sorted list of subject labels.
    """
    profiles: dict[tuple, AcquisitionProfile] = {}
    for facts in all_facts:
        signature = acquisition_signature(facts)
        if signature not in profiles:
            profiles[signature] = AcquisitionProfile(signature=signature, facts=facts)
        profiles[signature].subjects.append(facts.subject_id)

    for profile in profiles.values():
        profile.subjects.sort()

    return list(profiles.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_profiles.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/profiles.py qsiprep/tests/test_recommend_profiles.py
git commit -m "Group subjects into acquisition profiles for the recommender"
```

---

### Task 5: Rule registry

**Files:**
- Create: `qsiprep/recommend/rules.py`
- Test: `qsiprep/tests/test_recommend_rules.py`

**Interfaces:**
- Consumes: `Recommendation`, `SubjectFacts` from Task 1.
- Produces: `Rule(name, condition, docs_anchor, flags, func)`; the `@rule(...)` decorator; `RULES: list[Rule]`; `evaluate(facts) -> list[Recommendation]`.

**Design notes for the implementer:** Registration order is evaluation order — rules must be *defined* in dependency order in the file. Each rule declares the flags it may emit; `evaluate` raises if a rule emits an undeclared flag, which keeps the declaration honest and lets the docs table and the parser invariant test rely on it.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_rules.py`:

```python
"""Tests for the recommender's rule registry."""

from qsiprep.recommend.facts import Recommendation, SubjectFacts
from qsiprep.recommend.rules import Rule, evaluate_rules


def _noop_rule(name, flags, produced):
    def func(facts, decisions):
        return list(produced)

    return Rule(name=name, condition='always', docs_anchor='', flags=flags, func=func)


def test_evaluate_returns_recommendations_in_rule_order():
    rules = [
        _noop_rule('first', ('--a',), [Recommendation('because a', 'recommended', '--a', '1')]),
        _noop_rule('second', ('--b',), [Recommendation('because b', 'consider', '--b', '2')]),
    ]

    result = evaluate_rules(SubjectFacts(subject_id='01'), rules=rules)

    assert [rec.flag for rec in result] == ['--a', '--b']


def test_later_rules_see_earlier_decisions():
    seen = {}

    def reader(facts, decisions):
        seen.update(decisions)
        return []

    rules = [
        _noop_rule('first', ('--a',), [Recommendation('because a', 'recommended', '--a', '1')]),
        Rule(name='reader', condition='always', docs_anchor='', flags=(), func=reader),
    ]

    evaluate_rules(SubjectFacts(subject_id='01'), rules=rules)

    assert seen == {'--a': '1'}


def test_notes_do_not_become_decisions():
    seen = {}

    def reader(facts, decisions):
        seen.update(decisions)
        return []

    rules = [
        _noop_rule('first', ('--a',), [Recommendation('just so you know', 'note', '--a', '1')]),
        Rule(name='reader', condition='always', docs_anchor='', flags=(), func=reader),
    ]

    evaluate_rules(SubjectFacts(subject_id='01'), rules=rules)

    assert seen == {}


def test_failing_rule_becomes_an_undetermined_entry():
    def broken(facts, decisions):
        raise ValueError('bad metadata')

    rules = [
        Rule(name='broken', condition='always', docs_anchor='', flags=('--c',), func=broken),
        _noop_rule('after', ('--d',), [Recommendation('because d', 'recommended', '--d', '4')]),
    ]

    result = evaluate_rules(SubjectFacts(subject_id='01'), rules=rules)

    assert result[0].severity == 'undetermined'
    assert 'broken' in result[0].rationale
    assert 'bad metadata' in result[0].rationale
    # A failing rule must not stop the ones after it.
    assert result[1].flag == '--d'


def test_emitting_an_undeclared_flag_is_an_error():
    rules = [
        _noop_rule('sneaky', ('--a',), [Recommendation('surprise', 'recommended', '--z', '1')]),
    ]

    result = evaluate_rules(SubjectFacts(subject_id='01'), rules=rules)

    assert result[0].severity == 'undetermined'
    assert '--z' in result[0].rationale
```

Every test here passes an explicit `rules=` list, so this task is green with an empty registry. The test that the registry itself is populated and free of duplicate names arrives in Task 6, once there are rules to assert on.

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.recommend.rules'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/recommend/rules.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The recommendation knowledge base.

Every rule is a pure function of :class:`~qsiprep.recommend.facts.SubjectFacts`
and the decisions earlier rules reached, so the whole knowledge base can be
tested without a BIDS dataset on disk.

Rules are evaluated in the order they are defined in this file. Two orderings
matter: ``pepolar_method`` and ``sdc_fallback`` must follow ``hmc_model``, and
``anat_modality`` must follow ``infant``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .facts import INFANT_MAX_MONTHS, Recommendation, SubjectFacts

#: Severities that represent a decision later rules can read.
DECISIVE = ('recommended', 'consider')


@dataclass
class Rule:
    """One recommendation rule.

    Attributes
    ----------
    name : :obj:`str`
        Identifier, also shown in the documentation table.
    condition : :obj:`str`
        Human-readable description of when this rule fires, rendered into the
        documentation.
    docs_anchor : :obj:`str`
        Where in the documentation the advice comes from.
    flags : :obj:`tuple` of :obj:`str`
        Every flag this rule may emit. Emitting anything else is an error.
    func : :obj:`callable`
        ``func(facts, decisions) -> list[Recommendation]``.
    """

    name: str
    condition: str
    docs_anchor: str
    flags: tuple[str, ...]
    func: Callable[[SubjectFacts, dict], list[Recommendation]]


RULES: list[Rule] = []


def rule(name: str, condition: str, docs_anchor: str, flags: tuple[str, ...] = ()):
    """Register a rule. Definition order is evaluation order."""

    def decorator(func):
        RULES.append(
            Rule(name=name, condition=condition, docs_anchor=docs_anchor, flags=flags, func=func)
        )
        return func

    return decorator


def evaluate_rules(facts: SubjectFacts, rules=None) -> list[Recommendation]:
    """Run every rule against one subject's facts.

    A rule that raises, or that emits a flag it did not declare, is converted
    into an ``undetermined`` entry so that one broken rule degrades the report
    instead of destroying it.

    Parameters
    ----------
    facts : :obj:`~qsiprep.recommend.facts.SubjectFacts`
    rules : :obj:`list` of :obj:`Rule` or :obj:`None`
        Defaults to the module-level registry.

    Returns
    -------
    :obj:`list` of :obj:`~qsiprep.recommend.facts.Recommendation`
    """
    rules = RULES if rules is None else rules
    decisions: dict[str, str | None] = {}
    results: list[Recommendation] = []

    for entry in rules:
        try:
            produced = list(entry.func(facts, decisions) or [])
            for recommendation in produced:
                if recommendation.flag is not None and recommendation.flag not in entry.flags:
                    raise ValueError(
                        f'emitted {recommendation.flag}, which it does not declare'
                    )
        except Exception as exc:  # noqa: BLE001
            results.append(
                Recommendation(
                    rationale=f'Rule "{entry.name}" failed: {exc}',
                    severity='undetermined',
                    docs_anchor=entry.docs_anchor,
                )
            )
            continue

        for recommendation in produced:
            if recommendation.flag is not None and recommendation.severity in DECISIVE:
                decisions[recommendation.flag] = recommendation.value

        results.extend(produced)

    return results


def evaluate(facts: SubjectFacts) -> list[Recommendation]:
    """Run the registered rules against one subject's facts."""
    return evaluate_rules(facts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/rules.py qsiprep/tests/test_recommend_rules.py
git commit -m "Add rule registry and evaluation for the recommender"
```

---

### Task 6: Core rules — hmc, sdc, resolution

**Files:**
- Modify: `qsiprep/recommend/rules.py` (append rules)
- Modify: `qsiprep/tests/test_recommend_rules.py` (append tests, remove the xfail marker)

**Interfaces:**
- Consumes: `Rule`, `rule`, `evaluate` from Task 5; facts from Task 1.
- Produces: registered rules `hmc_model`, `pepolar_method`, `sdc_fallback`, `output_resolution`, `distortion_group_merge`.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_recommend_rules.py`:

```python
from qsiprep.recommend.facts import GroupFacts, SeriesFacts
from qsiprep.recommend.rules import evaluate


def series(**overrides):
    defaults = {
        'path': '/data/sub-01_dwi.nii.gz',
        'n_volumes': 33,
        'n_b0s': 3,
        'shells': (1000,),
        'is_shelled': True,
        'n_unique_bvals': 1,
        'voxel_size': (2.0, 2.0, 2.0),
        'pe_direction': 'j',
    }
    defaults.update(overrides)
    return SeriesFacts(**defaults)


def subject(**overrides):
    defaults = {
        'subject_id': '01',
        'series': (series(),),
        'groups': (GroupFacts('sub-01', 1, 'epi', 'j'),),
        'has_t1w': True,
    }
    defaults.update(overrides)
    return SubjectFacts(**defaults)


def flags_from(recommendations):
    return {rec.flag: rec.value for rec in recommendations if rec.flag}


def test_shelled_data_gets_no_hmc_recommendation():
    result = evaluate(subject())
    assert '--hmc-model' not in flags_from(result)


def test_non_shelled_data_recommends_diffprep_quadratic():
    facts = subject(series=(series(is_shelled=False, shells=(300, 2500), n_unique_bvals=40),))

    result = evaluate(facts)

    assert flags_from(result)['--hmc-model'] == 'diffprep_quadratic'


def test_unreadable_bvals_leave_hmc_undetermined():
    facts = subject(series=(series(shells=(), is_shelled=False, n_unique_bvals=0),))

    undetermined = [rec for rec in evaluate(facts) if rec.severity == 'undetermined']

    assert any(rec.flag == '--hmc-model' for rec in undetermined)


def test_diffprep_with_reverse_pe_recommends_drbuddi():
    facts = subject(series=(series(is_shelled=False, shells=(300, 2500), n_unique_bvals=40),))

    result = flags_from(evaluate(facts))

    assert result['--pepolar-method'] == 'DRBUDDI'


def test_eddy_with_reverse_pe_keeps_topup_default():
    assert '--pepolar-method' not in flags_from(evaluate(subject()))


def test_no_fieldmap_recommends_syn_sdc():
    facts = subject(groups=(GroupFacts('sub-01', 1, None, 'j'),))

    assert flags_from(evaluate(facts))['--use-syn-sdc'] == 'warn'


def test_no_fieldmap_with_t2w_and_diffprep_is_a_note():
    facts = subject(
        series=(series(is_shelled=False, shells=(300, 2500), n_unique_bvals=40),),
        groups=(GroupFacts('sub-01', 1, None, 'j'),),
        has_t2w=True,
    )

    result = evaluate(facts)

    assert '--use-syn-sdc' not in flags_from(result)
    assert any(rec.severity == 'note' and 'T2Wreg' in rec.rationale for rec in result)


def test_output_resolution_uses_largest_voxel_dimension():
    facts = subject(series=(series(voxel_size=(1.8, 1.8, 2.4)),))

    assert flags_from(evaluate(facts))['--output-resolution'] == '2.4'


def test_anisotropic_voxels_produce_a_warning():
    facts = subject(series=(series(voxel_size=(1.8, 1.8, 2.4)),))

    assert any(rec.severity == 'warning' for rec in evaluate(facts))


def test_mixed_voxel_sizes_produce_a_warning():
    facts = subject(series=(series(), series(voxel_size=(1.5, 1.5, 1.5))))

    warnings = [rec for rec in evaluate(facts) if rec.severity == 'warning']

    assert any('different voxel sizes' in rec.rationale for rec in warnings)


def test_rpe_series_recommends_distortion_group_merge():
    facts = subject(groups=(GroupFacts('sub-01', 1, 'rpe_series', 'j'),))

    assert flags_from(evaluate(facts))['--distortion-group-merge'] == 'average'


def test_registry_is_populated_and_ordered():
    from qsiprep.recommend.rules import RULES

    names = [rule.name for rule in RULES]
    assert names, 'no rules registered'
    assert len(names) == len(set(names)), 'duplicate rule names'
    # pepolar_method and sdc_fallback read the hmc_model decision.
    assert names.index('hmc_model') < names.index('pepolar_method')
    assert names.index('hmc_model') < names.index('sdc_fallback')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: FAIL — `KeyError: '--hmc-model'` and similar, because no rules are registered yet

- [ ] **Step 3: Write minimal implementation**

Append to `qsiprep/recommend/rules.py`:

```python
#: Fieldmap suffixes that indicate reverse phase-encoded data is available.
REVERSE_PE_SUFFIXES = ('epi', 'dwi', 'rpe_series')


@rule(
    name='hmc_model',
    condition='DWI b-values do not form shells (DSI or CS-DSI sampling)',
    docs_anchor='quickstart: head motion correction model',
    flags=('--hmc-model',),
)
def hmc_model(facts, decisions):
    """Non-shelled schemes cannot use FSL eddy."""
    if not facts.series:
        return []

    if any(series.n_unique_bvals == 0 for series in facts.series):
        return [
            Recommendation(
                rationale=(
                    'b-values could not be read for at least one DWI series, so the '
                    'sampling scheme could not be determined.'
                ),
                severity='undetermined',
                flag='--hmc-model',
                docs_anchor='quickstart: head motion correction model',
            )
        ]

    if all(series.is_shelled for series in facts.series):
        return []

    n_unique = max(series.n_unique_bvals for series in facts.series)
    return [
        Recommendation(
            rationale=(
                f'The sampling scheme is not shelled ({n_unique} unique b-values), and FSL '
                'eddy requires shells. TORTOISE DIFFPREP fits a signal model over arbitrary '
                'q-space and adds 24-parameter quadratic eddy-current correction. Use '
                '"3dSHORE" instead if you want motion correction without eddy correction.'
            ),
            severity='recommended',
            flag='--hmc-model',
            value='diffprep_quadratic',
            docs_anchor='quickstart: head motion correction model',
        )
    ]


@rule(
    name='pepolar_method',
    condition='reverse phase-encoded data with a diffprep_* head-motion model',
    docs_anchor='quickstart: head motion correction model',
    flags=('--pepolar-method',),
)
def pepolar_method(facts, decisions):
    """TOPUP is not supported with the DIFFPREP backends."""
    hmc = decisions.get('--hmc-model') or ''
    if not hmc.startswith('diffprep'):
        return []

    if not any(group.fieldmap_suffix in REVERSE_PE_SUFFIXES for group in facts.groups):
        return []

    return [
        Recommendation(
            rationale=(
                'Reverse phase-encoded data is present and the DIFFPREP backends do not '
                'support TOPUP; DRBUDDI is TORTOISE\'s native distortion correction.'
            ),
            severity='recommended',
            flag='--pepolar-method',
            value='DRBUDDI',
            docs_anchor='quickstart: head motion correction model',
        )
    ]


@rule(
    name='sdc_fallback',
    condition='no fieldmap and no reverse phase-encoded data',
    docs_anchor='quickstart: head motion correction model',
    flags=('--use-syn-sdc',),
)
def sdc_fallback(facts, decisions):
    """Without a fieldmap, distortion correction needs another source."""
    if any(group.fieldmap_suffix for group in facts.groups):
        return []

    hmc = decisions.get('--hmc-model') or 'eddy'
    if hmc.startswith('diffprep') and facts.has_t2w:
        return [
            Recommendation(
                rationale=(
                    'No fieldmap was found, but a T2w image is available, so TORTOISE will '
                    'apply its "--epi T2Wreg" structural correction automatically. No flag '
                    'is needed.'
                ),
                severity='note',
                docs_anchor='quickstart: head motion correction model',
            )
        ]

    return [
        Recommendation(
            rationale=(
                'No fieldmap and no reverse phase-encoded data were found. Fieldmap-less '
                'SyN correction uses the anatomical image instead; "warn" continues when it '
                'cannot be applied, while the default "error" stops the run.'
            ),
            severity='consider',
            flag='--use-syn-sdc',
            value='warn',
            docs_anchor='quickstart: head motion correction model',
        )
    ]


@rule(
    name='output_resolution',
    condition='always; derived from the input voxel size',
    docs_anchor='quickstart: output resolution and resampling',
    flags=('--output-resolution',),
)
def output_resolution(facts, decisions):
    """``--output-resolution`` is required, so always recommend a value."""
    if not facts.series:
        return []

    sizes = {series.voxel_size for series in facts.series}
    largest = max(max(size) for size in sizes)
    value = f'{round(largest, 2):g}'

    results = [
        Recommendation(
            rationale=(
                f'The largest input voxel dimension is {value} mm. Upsampling by more than '
                '10% switches interpolation from Lanczos to Linear; some pipelines, such as '
                'fixel-based analysis, want at least 1.3 mm.'
            ),
            severity='recommended',
            flag='--output-resolution',
            value=value,
            docs_anchor='quickstart: output resolution and resampling',
        )
    ]

    if len(sizes) > 1:
        listing = ', '.join(
            ' x '.join(f'{dim:g}' for dim in size) for size in sorted(sizes)
        )
        results.append(
            Recommendation(
                rationale=(
                    f'DWI series have different voxel sizes ({listing}). All of them will be '
                    'resampled to the single output resolution.'
                ),
                severity='warning',
                docs_anchor='quickstart: output resolution and resampling',
            )
        )
    elif len(set(next(iter(sizes)))) > 1:
        size = ' x '.join(f'{dim:g}' for dim in next(iter(sizes)))
        results.append(
            Recommendation(
                rationale=(
                    f'The input voxels are anisotropic ({size} mm), so the isotropic output '
                    'will upsample at least one axis.'
                ),
                severity='warning',
                docs_anchor='quickstart: output resolution and resampling',
            )
        )

    return results


@rule(
    name='distortion_group_merge',
    condition='a complete DWI series was acquired in both phase-encoding directions',
    docs_anchor='preprocessing: preprocessing HCP-style',
    flags=('--distortion-group-merge',),
)
def distortion_group_merge(facts, decisions):
    """HCP-style dual phase-encoding acquisitions can be averaged after correction."""
    if not any(group.fieldmap_suffix == 'rpe_series' for group in facts.groups):
        return []

    return [
        Recommendation(
            rationale=(
                'A complete DWI series was acquired in both phase-encoding directions. '
                '"average" averages the corrected images at matching q-space coordinates, '
                'as the HCP pipelines do; "concat" keeps twice as many volumes instead.'
            ),
            severity='consider',
            flag='--distortion-group-merge',
            value='average',
            docs_anchor='preprocessing: preprocessing HCP-style',
        )
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: PASS, all tests including `test_registry_is_populated_and_ordered`

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/rules.py qsiprep/tests/test_recommend_rules.py
git commit -m "Add core hmc, sdc, and resolution recommendation rules"
```

---

### Task 7: Denoising rules

**Files:**
- Modify: `qsiprep/recommend/rules.py` (append)
- Modify: `qsiprep/tests/test_recommend_rules.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 5 and 6.
- Produces: registered rules `unringing_method`, `b1_biascorrect_stage`, `denoise_volume_count`.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_recommend_rules.py`:

```python
def test_partial_fourier_recommends_rpg():
    facts = subject(series=(series(partial_fourier=0.75),))

    assert flags_from(evaluate(facts))['--unringing-method'] == 'rpg'


def test_full_fourier_suggests_mrdegibbs():
    facts = subject(series=(series(partial_fourier=1.0),))

    result = [rec for rec in evaluate(facts) if rec.flag == '--unringing-method']

    assert result[0].value == 'mrdegibbs'
    assert result[0].severity == 'consider'


def test_missing_partial_fourier_is_undetermined():
    result = [
        rec
        for rec in evaluate(subject())
        if rec.flag == '--unringing-method' and rec.severity == 'undetermined'
    ]

    assert len(result) == 1


def test_norm_image_type_disables_b1_biascorrection():
    facts = subject(series=(series(image_type=('ORIGINAL', 'PRIMARY', 'M', 'NORM')),))

    assert flags_from(evaluate(facts))['--b1-biascorrect-stage'] == 'none'


def test_image_type_without_norm_recommends_nothing():
    facts = subject(series=(series(image_type=('ORIGINAL', 'PRIMARY', 'M')),))

    assert '--b1-biascorrect-stage' not in flags_from(evaluate(facts))


def test_missing_image_type_is_undetermined():
    result = [
        rec
        for rec in evaluate(subject())
        if rec.flag == '--b1-biascorrect-stage' and rec.severity == 'undetermined'
    ]

    assert len(result) == 1


def test_short_series_warns_about_denoising():
    facts = subject(series=(series(n_volumes=20),))

    warnings = [rec for rec in evaluate(facts) if rec.severity == 'warning']

    assert any('fewer than 30 volumes' in rec.rationale for rec in warnings)


def test_long_series_does_not_warn_about_denoising():
    warnings = [rec for rec in evaluate(subject()) if rec.severity == 'warning']

    assert not any('fewer than 30 volumes' in rec.rationale for rec in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -k "fourier or norm or image_type or denoising" -v`
Expected: FAIL — `KeyError: '--unringing-method'` and similar

- [ ] **Step 3: Write minimal implementation**

Append to `qsiprep/recommend/rules.py`:

```python
#: Series shorter than this give MP-PCA little to work with.
MIN_DENOISE_VOLUMES = 30


@rule(
    name='unringing_method',
    condition='the PartialFourier field is present in the DWI sidecars',
    docs_anchor='usage: --unringing-method',
    flags=('--unringing-method',),
)
def unringing_method(facts, decisions):
    """mrdegibbs assumes full Fourier sampling; rpg handles partial Fourier."""
    known = {
        series.partial_fourier
        for series in facts.series
        if series.partial_fourier is not None
    }
    if not known:
        return [
            Recommendation(
                rationale=(
                    'No PartialFourier field was found in the DWI sidecars, so Gibbs '
                    'unringing could not be recommended either way.'
                ),
                severity='undetermined',
                flag='--unringing-method',
                docs_anchor='usage: --unringing-method',
            )
        ]

    if any(value < 1 for value in known):
        smallest = min(known)
        return [
            Recommendation(
                rationale=(
                    f'PartialFourier is {smallest:g}. mrdegibbs is only supposed to run on '
                    'full Fourier acquisitions, while rpg (from TORTOISE) is suggested for '
                    'partial Fourier data.'
                ),
                severity='recommended',
                flag='--unringing-method',
                value='rpg',
                docs_anchor='usage: --unringing-method',
            )
        ]

    return [
        Recommendation(
            rationale=(
                'These are full Fourier acquisitions, which is what mrdegibbs expects. '
                'Gibbs unringing is off by default, so enable it only if you want it.'
            ),
            severity='consider',
            flag='--unringing-method',
            value='mrdegibbs',
            docs_anchor='usage: --unringing-method',
        )
    ]


@rule(
    name='b1_biascorrect_stage',
    condition='the DWI ImageType metadata contains NORM (prescan normalization)',
    docs_anchor='preprocessing: denoising and merging images',
    flags=('--b1-biascorrect-stage',),
)
def b1_biascorrect_stage(facts, decisions):
    """Bias correction can introduce artifacts on prescan-normalized data."""
    if not any(series.image_type for series in facts.series):
        return [
            Recommendation(
                rationale=(
                    'No ImageType field was found in the DWI sidecars, so prescan '
                    'normalization could not be detected.'
                ),
                severity='undetermined',
                flag='--b1-biascorrect-stage',
                docs_anchor='preprocessing: denoising and merging images',
            )
        ]

    if not any('NORM' in series.image_type for series in facts.series):
        return []

    return [
        Recommendation(
            rationale=(
                'The DWI ImageType contains "NORM", which is how scanners flag '
                'console-applied prescan normalization. B1 bias field correction may '
                'introduce artifacts on already-normalized data.'
            ),
            severity='recommended',
            flag='--b1-biascorrect-stage',
            value='none',
            docs_anchor='preprocessing: denoising and merging images',
        )
    ]


@rule(
    name='denoise_volume_count',
    condition=f'a DWI series has fewer than {MIN_DENOISE_VOLUMES} volumes',
    docs_anchor='preprocessing: denoising and merging images',
)
def denoise_volume_count(facts, decisions):
    """MP-PCA needs volumes; warn when there are few."""
    short = [series for series in facts.series if series.n_volumes < MIN_DENOISE_VOLUMES]
    if not short:
        return []

    return [
        Recommendation(
            rationale=(
                f'{len(short)} of {len(facts.series)} DWI series have fewer than '
                f'{MIN_DENOISE_VOLUMES} volumes. MP-PCA denoising has little data to work '
                'with at this length; check the denoising reportlets, and consider '
                '--denoise-method none if the results look poor.'
            ),
            severity='warning',
            docs_anchor='preprocessing: denoising and merging images',
        )
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/rules.py qsiprep/tests/test_recommend_rules.py
git commit -m "Add denoising recommendation rules"
```

---

### Task 8: Anatomical and cohort rules

**Files:**
- Modify: `qsiprep/recommend/rules.py` (append)
- Modify: `qsiprep/tests/test_recommend_rules.py` (append)

**Interfaces:**
- Consumes: Tasks 5–7; `cohort_by_months` from `qsiprep.utils.bids`.
- Produces: registered rules `infant`, `anat_modality`, `anatomical_reference`. `infant` **must** be defined before `anat_modality`.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_recommend_rules.py`:

```python
def test_missing_t1w_with_t2w_recommends_t2w_modality():
    facts = subject(has_t1w=False, has_t2w=True)

    assert flags_from(evaluate(facts))['--anat-modality'] == 'T2w'


def test_no_anatomicals_recommends_no_modality():
    facts = subject(has_t1w=False, has_t2w=False)

    assert flags_from(evaluate(facts))['--anat-modality'] == 'none'


def test_deprecated_dwi_only_is_never_emitted():
    facts = subject(has_t1w=False, has_t2w=False)

    assert '--dwi-only' not in flags_from(evaluate(facts))


def test_young_subject_recommends_infant():
    facts = subject(age_months=9)

    result = flags_from(evaluate(facts))

    assert '--infant' in result
    assert result['--infant'] is None


def test_infant_suppresses_anat_modality():
    facts = subject(age_months=9, has_t1w=False, has_t2w=True)

    result = flags_from(evaluate(facts))

    assert '--infant' in result
    assert '--anat-modality' not in result


def test_adult_subject_does_not_recommend_infant():
    assert '--infant' not in flags_from(evaluate(subject(age_months=360)))


def test_missing_age_does_not_recommend_infant():
    assert '--infant' not in flags_from(evaluate(subject()))


def test_multiple_sessions_produce_a_note():
    facts = subject(sessions=('A', 'B'))

    notes = [rec for rec in evaluate(facts) if rec.severity == 'note']

    assert any('sessionwise' in rec.rationale for rec in notes)


def test_single_session_produces_no_reference_note():
    notes = [rec for rec in evaluate(subject()) if rec.severity == 'note']

    assert not any('sessionwise' in rec.rationale for rec in notes)


def test_infant_is_registered_before_anat_modality():
    from qsiprep.recommend.rules import RULES

    names = [rule.name for rule in RULES]

    assert names.index('infant') < names.index('anat_modality')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -k "modality or infant or session" -v`
Expected: FAIL — `KeyError: '--anat-modality'` and similar

- [ ] **Step 3: Write minimal implementation**

Append to `qsiprep/recommend/rules.py`. Keep this order — `infant` first. `INFANT_MAX_MONTHS` is already imported from `facts.py` at the top of the module; do not redefine it here:

```python
@rule(
    name='infant',
    condition=f'participant age is {INFANT_MAX_MONTHS} months or younger',
    docs_anchor='usage: --infant',
    flags=('--infant',),
)
def infant(facts, decisions):
    """Infant brains need the MNIInfant template and its cohorts."""
    if facts.age_months is None or facts.age_months > INFANT_MAX_MONTHS:
        return []

    from ..utils.bids import cohort_by_months

    cohort = cohort_by_months('MNIInfant', facts.age_months)
    return [
        Recommendation(
            rationale=(
                f'The participant is {facts.age_months} months old. --infant switches the '
                f'anatomical template to MNIInfant (cohort {cohort} at this age) and forces '
                'a T2w anatomical reference.'
            ),
            severity='recommended',
            flag='--infant',
            docs_anchor='usage: --infant',
        )
    ]


@rule(
    name='anat_modality',
    condition='no T1w image is present',
    docs_anchor='usage: --anat-modality',
    flags=('--anat-modality',),
)
def anat_modality(facts, decisions):
    """Pick an anatomical reference when the default T1w is unavailable.

    Runs after ``infant``, which already forces a T2w reference.
    """
    if facts.has_t1w:
        return []

    if '--infant' in decisions:
        return []

    if facts.has_t2w:
        return [
            Recommendation(
                rationale=(
                    'No T1w image was found, but a T2w image is available. It will be skull '
                    'stripped and segmented as the anatomical reference.'
                ),
                severity='recommended',
                flag='--anat-modality',
                value='T2w',
                docs_anchor='usage: --anat-modality',
            )
        ]

    return [
        Recommendation(
            rationale=(
                'No T1w or T2w images were found, so QSIPrep must run without an anatomical '
                'reference and align to an AC-PC b=0 template instead.'
            ),
            severity='recommended',
            flag='--anat-modality',
            value='none',
            docs_anchor='usage: --anat-modality',
        )
    ]


@rule(
    name='anatomical_reference',
    condition='a subject has more than one session',
    docs_anchor='usage: --subject-anatomical-reference',
)
def anatomical_reference(facts, decisions):
    """Multi-session data forces a choice the data cannot make."""
    if len(facts.sessions) < 2:
        return []

    return [
        Recommendation(
            rationale=(
                f'Subjects have {len(facts.sessions)} sessions. '
                '--subject-anatomical-reference decides whether they share one anatomical '
                'space ("first-lex", the default, or "unbiased") or get one per session '
                '("sessionwise"). This is a study-design decision the data cannot settle.'
            ),
            severity='note',
            docs_anchor='usage: --subject-anatomical-reference',
        )
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/rules.py qsiprep/tests/test_recommend_rules.py
git commit -m "Add anatomical and cohort recommendation rules"
```

---

### Task 9: Grouping rules

**Files:**
- Modify: `qsiprep/recommend/rules.py` (append)
- Modify: `qsiprep/tests/test_recommend_rules.py` (append)

**Interfaces:**
- Consumes: Tasks 5–8.
- Produces: registered rules `concatenation_preview`, `denoise_after_combining`, `multipart_id_missing`.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_recommend_rules.py`:

```python
def test_multi_run_group_produces_a_concatenation_note():
    facts = subject(
        series=(series(multipart_id='abc'), series(multipart_id='abc')),
        groups=(GroupFacts('sub-01_acq-multi', 2, 'epi', 'j'),),
    )

    notes = [rec for rec in evaluate(facts) if rec.severity == 'note']

    assert any('sub-01_acq-multi' in rec.rationale for rec in notes)


def test_multi_run_group_suggests_denoise_after_combining():
    facts = subject(
        series=(series(multipart_id='abc'), series(multipart_id='abc')),
        groups=(GroupFacts('sub-01_acq-multi', 2, 'epi', 'j'),),
    )

    assert '--denoise-after-combining' in flags_from(evaluate(facts))


def test_single_run_group_says_nothing_about_grouping():
    result = evaluate(subject())

    assert '--denoise-after-combining' not in flags_from(result)


def test_missing_multipart_id_warns():
    facts = subject(
        series=(series(), series()),
        groups=(GroupFacts('sub-01', 2, 'epi', 'j'),),
    )

    warnings = [rec for rec in evaluate(facts) if rec.severity == 'warning']

    assert any('MultipartID' in rec.rationale for rec in warnings)


def test_present_multipart_id_does_not_warn():
    facts = subject(
        series=(series(multipart_id='abc'), series(multipart_id='abc')),
        groups=(GroupFacts('sub-01', 2, 'epi', 'j'),),
    )

    warnings = [rec for rec in evaluate(facts) if rec.severity == 'warning']

    assert not any('MultipartID' in rec.rationale for rec in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -k "concatenation or combining or multipart or grouping" -v`
Expected: FAIL — no notes or flags produced

- [ ] **Step 3: Write minimal implementation**

Append to `qsiprep/recommend/rules.py`:

```python
@rule(
    name='concatenation_preview',
    condition='two or more DWI runs will be concatenated into one group',
    docs_anchor='quickstart: grouping scans',
)
def concatenation_preview(facts, decisions):
    """Show the grouping QSIPrep computed so the user can accept or override it."""
    multi = [group for group in facts.groups if group.n_series > 1]
    if not multi:
        return []

    listing = '; '.join(f'{group.concatenated_bids_name} ({group.n_series} runs)' for group in multi)
    return [
        Recommendation(
            rationale=(
                f'These DWI runs share a warped space and will be concatenated before head '
                f'motion correction: {listing}. Pass --separate-all-dwis to process each run '
                'on its own instead.'
            ),
            severity='note',
            docs_anchor='quickstart: grouping scans',
        )
    ]


@rule(
    name='denoise_after_combining',
    condition='two or more DWI runs will be concatenated into one group',
    docs_anchor='preprocessing: denoising and merging images',
    flags=('--denoise-after-combining',),
)
def denoise_after_combining(facts, decisions):
    """More volumes help MP-PCA, but between-scan motion can hurt it."""
    if not any(group.n_series > 1 for group in facts.groups):
        return []

    return [
        Recommendation(
            rationale=(
                'Runs will be concatenated, so denoising could run on the combined series '
                'instead of each run separately. More volumes give MP-PCA more to work with, '
                'but large between-scan head motion can hurt it. There is little data to '
                'guide this choice; check Framewise Displacement where a new series begins.'
            ),
            severity='consider',
            flag='--denoise-after-combining',
            docs_anchor='preprocessing: denoising and merging images',
        )
    ]


@rule(
    name='multipart_id_missing',
    condition='multiple runs will be concatenated but MultipartID is not set',
    docs_anchor='usage: MultipartID',
)
def multipart_id_missing(facts, decisions):
    """Without MultipartID, grouping falls back to BIDS entities."""
    if not any(group.n_series > 1 for group in facts.groups):
        return []

    if all(series.multipart_id for series in facts.series):
        return []

    return [
        Recommendation(
            rationale=(
                'Multiple DWI runs will be concatenated, but MultipartID is not set on all '
                'of them, so grouping falls back to BIDS entities. Check that the groups '
                'above match what you expect.'
            ),
            severity='warning',
            docs_anchor='usage: MultipartID',
        )
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_rules.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/rules.py qsiprep/tests/test_recommend_rules.py
git commit -m "Add scan-grouping recommendation rules"
```

---

### Task 10: Report rendering

**Files:**
- Create: `qsiprep/recommend/report.py`
- Test: `qsiprep/tests/test_recommend_report.py`

**Interfaces:**
- Consumes: `AcquisitionProfile` from Task 4; `Recommendation` from Task 1; `evaluate` from Task 5.
- Produces: `build_command(bids_dir, output_dir, recommendations, participant_labels=None) -> str`; `render_report(bids_dir, profiles, skipped=(), output_dir=None, verbose=False) -> str`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_report.py`:

```python
"""Tests for the recommender's text output."""

from qsiprep.recommend.facts import GroupFacts, Recommendation, SeriesFacts, SubjectFacts
from qsiprep.recommend.profiles import build_profiles
from qsiprep.recommend.report import build_command, render_report


def _series(**overrides):
    defaults = {
        'path': '/data/sub-01_dwi.nii.gz',
        'n_volumes': 33,
        'n_b0s': 3,
        'shells': (1000,),
        'is_shelled': True,
        'n_unique_bvals': 1,
        'voxel_size': (2.0, 2.0, 2.0),
        'pe_direction': 'j',
    }
    defaults.update(overrides)
    return SeriesFacts(**defaults)


def _subject(subject_id='01', **overrides):
    defaults = {
        'series': (_series(),),
        'groups': (GroupFacts('sub-01', 1, 'epi', 'j'),),
        'has_t1w': True,
    }
    defaults.update(overrides)
    return SubjectFacts(subject_id=subject_id, **defaults)


def test_command_includes_recommended_and_consider_flags():
    recommendations = [
        Recommendation('a', 'recommended', '--hmc-model', 'diffprep_quadratic'),
        Recommendation('b', 'consider', '--use-syn-sdc', 'warn'),
        Recommendation('c', 'note'),
        Recommendation('d', 'warning'),
    ]

    command = build_command('/in', '/out', recommendations)

    assert command.startswith('qsiprep /in /out participant')
    assert '--hmc-model diffprep_quadratic' in command
    assert '--use-syn-sdc warn' in command


def test_command_renders_store_true_flags_without_a_value():
    command = build_command('/in', '/out', [Recommendation('a', 'recommended', '--infant')])

    assert '--infant' in command
    assert '--infant None' not in command


def test_command_appends_participant_labels_when_given():
    command = build_command('/in', '/out', [], participant_labels=['01', '02'])

    assert '--participant-label 01 02' in command


def test_command_uses_a_placeholder_without_an_output_directory():
    command = build_command('/in', None, [])

    assert '/path/to/outputs' in command


def test_report_lists_every_bucket_that_has_entries():
    report = render_report('/data', build_profiles([_subject()]))

    assert 'Recommended' in report
    assert 'Detected' in report


def test_report_omits_empty_buckets():
    report = render_report('/data', build_profiles([_subject()]))

    # No rule fires a note for this subject, so the heading must not appear.
    assert '\nNotes\n' not in report


def test_report_truncates_long_subject_lists():
    subjects = [_subject(f'{index:02d}') for index in range(20)]

    report = render_report('/data', build_profiles(subjects))

    assert '+15 more' in report


def test_verbose_report_lists_every_subject():
    subjects = [_subject(f'{index:02d}') for index in range(20)]

    report = render_report('/data', build_profiles(subjects), verbose=True)

    assert '+15 more' not in report
    assert 'sub-19' in report or '19' in report


def test_report_numbers_multiple_profiles():
    shelled = _subject('01')
    non_shelled = _subject(
        '02',
        series=(_series(is_shelled=False, shells=(300, 2500), n_unique_bvals=40),),
    )

    report = render_report('/data', build_profiles([shelled, non_shelled]))

    assert 'Profile 1 of 2' in report
    assert 'Profile 2 of 2' in report
    assert '--participant-label' in report


def test_single_profile_omits_participant_label():
    report = render_report('/data', build_profiles([_subject()]))

    assert '--participant-label' not in report


def test_report_lists_skipped_subjects():
    report = render_report('/data', build_profiles([_subject()]), skipped=[('09', 'no DWI data')])

    assert 'Skipped' in report
    assert '09' in report
    assert 'no DWI data' in report


def test_report_lines_fit_in_eighty_columns():
    report = render_report('/data', build_profiles([_subject()]))

    overlong = [line for line in report.splitlines() if len(line) > 79]

    assert overlong == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.recommend.report'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/recommend/report.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Render recommendations as plain text.

The output is deliberately plain ASCII wrapped at 79 columns so that it
survives being pasted into an issue or a cluster log.
"""

from __future__ import annotations

import textwrap

from .rules import evaluate

WIDTH = 79
#: Bucket order and headings.
BUCKETS = (
    ('recommended', 'Recommended'),
    ('consider', 'Consider'),
    ('note', 'Notes'),
    ('warning', 'Warnings'),
    ('undetermined', 'Not determined'),
)
#: Severities whose flags belong in the printed command.
IN_COMMAND = ('recommended', 'consider')
OUTPUT_PLACEHOLDER = '/path/to/outputs'
MAX_LISTED_SUBJECTS = 5


def build_command(bids_dir, output_dir, recommendations, participant_labels=None) -> str:
    """Build a runnable ``qsiprep`` invocation from a profile's recommendations."""
    parts = [f'qsiprep {bids_dir} {output_dir or OUTPUT_PLACEHOLDER} participant']
    for recommendation in recommendations:
        if recommendation.flag is None or recommendation.severity not in IN_COMMAND:
            continue
        if recommendation.value is None:
            parts.append(recommendation.flag)
        else:
            parts.append(f'{recommendation.flag} {recommendation.value}')

    if participant_labels:
        parts.append('--participant-label ' + ' '.join(participant_labels))

    return ' \\\n  '.join(parts)


def render_report(bids_dir, profiles, skipped=(), output_dir=None, verbose=False) -> str:
    """Render the full report for a dataset.

    Parameters
    ----------
    bids_dir : :obj:`os.PathLike`
    profiles : :obj:`list` of :obj:`~qsiprep.recommend.profiles.AcquisitionProfile`
    skipped : :obj:`list` of :obj:`tuple`
        ``(subject_id, reason)`` pairs.
    output_dir : :obj:`os.PathLike` or :obj:`None`
    verbose : :obj:`bool`
        List every subject instead of truncating.

    Returns
    -------
    :obj:`str`
    """
    from qsiprep import __version__

    n_subjects = sum(len(profile.subjects) for profile in profiles)
    lines = [
        f'qsiprep-recommend {__version__} -- {bids_dir}',
        f'Indexed {n_subjects} subjects; {len(profiles)} acquisition profile'
        f'{"s" if len(profiles) != 1 else ""}.',
    ]

    for index, profile in enumerate(profiles, start=1):
        lines.append('')
        lines.extend(
            _render_profile(
                bids_dir,
                profile,
                index,
                len(profiles),
                output_dir=output_dir,
                verbose=verbose,
            )
        )

    if skipped:
        lines.append('')
        lines.append('Skipped')
        for subject_id, reason in skipped:
            lines.extend(_wrap(f'sub-{subject_id}: {reason}', indent=2))

    return '\n'.join(lines) + '\n'


def _render_profile(bids_dir, profile, index, total, output_dir=None, verbose=False):
    """Render one acquisition profile block."""
    header = f'=== Profile {index} of {total} -- {len(profile.subjects)} subjects '
    lines = [header + '=' * max(WIDTH - len(header), 0)]

    listed = profile.subjects if verbose else profile.subjects[:MAX_LISTED_SUBJECTS]
    names = ', '.join(f'sub-{subject}' for subject in listed)
    hidden = len(profile.subjects) - len(listed)
    if hidden > 0:
        names += f' (+{hidden} more; -v for all)'
    lines.extend(_wrap(f'Subjects: {names}', indent=0))

    lines.append('')
    lines.append('Detected')
    lines.extend(_render_detected(profile.facts))

    recommendations = evaluate(profile.facts)
    for severity, heading in BUCKETS:
        entries = [rec for rec in recommendations if rec.severity == severity]
        if not entries:
            continue
        lines.append('')
        lines.append(heading)
        for entry in entries:
            lines.extend(_render_entry(entry))

    lines.append('')
    labels = profile.subjects if total > 1 else None
    lines.append(build_command(bids_dir, output_dir, recommendations, participant_labels=labels))
    return lines


def _render_detected(facts):
    """Render the facts block for one profile."""
    rows = []
    series = facts.series
    if series:
        first = series[0]
        scheme = (
            f'shelled, b = {", ".join(str(shell) for shell in first.shells)}'
            if first.is_shelled
            else f'non-shelled, {first.n_unique_bvals} unique b-values'
        )
        rows.append(('Sampling scheme', f'{scheme}, {first.n_volumes} volumes'))
        rows.append(
            ('Voxel size', ' x '.join(f'{dim:g}' for dim in first.voxel_size) + ' mm')
        )
        rows.append(
            (
                'DWI runs',
                f'{len(series)} run{"s" if len(series) != 1 else ""}, '
                f'{max(len(facts.sessions), 1)} session'
                f'{"s" if max(len(facts.sessions), 1) != 1 else ""}',
            )
        )
        if first.partial_fourier is not None:
            rows.append(('Partial Fourier', f'{first.partial_fourier:g}'))

    fieldmaps = sorted({group.fieldmap_suffix for group in facts.groups if group.fieldmap_suffix})
    rows.append(('Fieldmaps', ', '.join(fieldmaps) if fieldmaps else 'none'))

    anatomicals = [name for name, present in (('T1w', facts.has_t1w), ('T2w', facts.has_t2w)) if present]
    rows.append(('Anatomicals', ', '.join(anatomicals) if anatomicals else 'none'))

    if facts.age_months is not None:
        rows.append(('Age', f'{facts.age_months} months'))

    label_width = max((len(label) for label, _ in rows), default=0)
    return [f'  {label.ljust(label_width)}   {value}' for label, value in rows]


def _render_entry(recommendation):
    """Render one recommendation: an optional flag line plus wrapped rationale."""
    lines = []
    if recommendation.flag is not None:
        value = '' if recommendation.value is None else f' {recommendation.value}'
        lines.append(f'  {recommendation.flag}{value}')
        indent = 6
    else:
        indent = 2

    rationale = recommendation.rationale
    if recommendation.docs_anchor:
        rationale = f'{rationale}  [{recommendation.docs_anchor}]'
    lines.extend(_wrap(rationale, indent=indent))
    return lines


def _wrap(text, indent=0):
    """Wrap ``text`` to the report width with a hanging indent."""
    prefix = ' ' * indent
    return textwrap.wrap(
        text,
        width=WIDTH,
        initial_indent=prefix,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix.rstrip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_report.py -v`
Expected: PASS, 12 tests

The 79-column test also covers the generated command line; if a long `--participant-label` list overflows, that is acceptable and the test should exclude lines that begin with `  --participant-label`. Adjust the test rather than mangling the command.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/recommend/report.py qsiprep/tests/test_recommend_report.py
git commit -m "Render recommender output as plain text"
```

---

### Task 11: CLI entry point and the parser invariant

**Files:**
- Create: `qsiprep/cli/recommend.py`
- Modify: `pyproject.toml:105-107` (the `[project.scripts]` table)
- Test: `qsiprep/tests/test_recommend_cli.py`

**Interfaces:**
- Consumes: `probe_dataset`, `build_profiles`, `render_report`, `RULES`.
- Produces: `main(argv=None) -> int` and the console script `qsiprep-recommend`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_recommend_cli.py`:

```python
"""End-to-end tests for the qsiprep-recommend CLI."""

import pytest

from qsiprep.cli.parser import _build_parser
from qsiprep.cli.recommend import main
from qsiprep.recommend.rules import RULES
from qsiprep.tests.recommend_fixtures import DwiSpec, make_dataset

#: Flags no rule may emit, with the reason.
NEVER_EMIT = {
    # Listed in the parser's `deprecations` dict, but registered as a plain
    # store_true, so it is not caught by the DeprecatedAction check below.
    '--dwi-only': 'deprecated in favor of --anat-modality none',
}


def _parser_actions():
    parser = _build_parser()
    return {
        option: action for action in parser._actions for option in action.option_strings
    }


def test_every_rule_flag_exists_in_the_qsiprep_parser():
    known = _parser_actions()

    for rule in RULES:
        for flag in rule.flags:
            assert flag in known, f'rule "{rule.name}" emits unknown flag {flag}'


def test_no_rule_emits_a_deprecated_flag():
    actions = _parser_actions()

    for rule in RULES:
        for flag in rule.flags:
            assert flag not in NEVER_EMIT, f'rule "{rule.name}" emits {flag}: {NEVER_EMIT[flag]}'
            action_name = type(actions[flag]).__name__
            assert action_name != 'DeprecatedAction', (
                f'rule "{rule.name}" emits deprecated flag {flag}'
            )


def test_cli_prints_a_report_and_exits_zero(tmp_path, capsys):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[DwiSpec(entities={'dir': 'AP'}, bvals=[0] * 5 + list(range(300, 5000, 85)))],
        fieldmaps=({'suffix': 'epi', 'entities': {'dir': 'PA'}},),
    )

    status = main([str(bids_dir), '--skip-bids-validation'])
    output = capsys.readouterr().out

    assert status == 0
    assert '--hmc-model diffprep_quadratic' in output
    assert 'qsiprep' in output


def test_printed_command_parses_with_the_qsiprep_parser(tmp_path, capsys):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01'],
        dwis=[
            DwiSpec(
                entities={'dir': 'AP'},
                bvals=[0] * 5 + list(range(300, 5000, 85)),
                metadata={'PartialFourier': 0.75, 'ImageType': ['ORIGINAL', 'PRIMARY', 'NORM']},
            )
        ],
        fieldmaps=({'suffix': 'epi', 'entities': {'dir': 'PA'}},),
    )

    main([str(bids_dir), '--output-dir', str(tmp_path / 'out'), '--skip-bids-validation'])
    output = capsys.readouterr().out

    command = output.split('qsiprep ')[-1].replace('\\\n', ' ')
    args = command.split()
    # Drop the positional bids_dir, output_dir, and analysis level.
    parsed = _build_parser().parse_args(args[:2] + ['participant'] + args[3:])

    assert parsed.hmc_model == 'diffprep_quadratic'


def test_missing_bids_dir_exits_one(tmp_path, capsys):
    status = main([str(tmp_path / 'nope')])

    assert status == 1
    assert 'does not exist' in capsys.readouterr().err


def test_dataset_without_dwi_exits_one(tmp_path, capsys):
    bids_dir = make_dataset(tmp_path / 'ds', subjects=['01'], dwis=[])

    status = main([str(bids_dir), '--skip-bids-validation'])

    assert status == 1
    assert 'no subjects' in capsys.readouterr().err


def test_participant_label_restricts_the_report(tmp_path, capsys):
    bids_dir = make_dataset(
        tmp_path / 'ds',
        subjects=['01', '02'],
        dwis=[DwiSpec(bvals=[0] * 3 + [1000] * 30)],
    )

    main([str(bids_dir), '--participant-label', '02', '--skip-bids-validation'])
    output = capsys.readouterr().out

    assert 'Indexed 1 subjects' in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.cli.recommend'`

- [ ] **Step 3: Write minimal implementation**

Create `qsiprep/cli/recommend.py`:

```python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Recommend QSIPrep flags for a BIDS dataset."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


def _build_parser() -> ArgumentParser:
    """Build the ``qsiprep-recommend`` argument parser."""
    from qsiprep import __version__

    parser = ArgumentParser(
        prog='qsiprep-recommend',
        description=(
            'Inspect a BIDS dataset and recommend QSIPrep command-line flags for it. '
            'This reads only sidecar metadata, gradient tables, and image headers; it '
            'does not process any image data.'
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument('bids_dir', type=Path, help='the root of the BIDS dataset to inspect')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=(
            'the output directory to put in the suggested command. If omitted, a '
            'placeholder path is used.'
        ),
    )
    parser.add_argument(
        '--participant-label',
        nargs='+',
        default=None,
        help='a space delimited list of participant identifiers, without the "sub-" prefix',
    )
    parser.add_argument(
        '--session-id',
        default=None,
        help='restrict the analysis to a single session, without the "ses-" prefix',
    )
    parser.add_argument(
        '--bids-filter-file',
        type=Path,
        default=None,
        help='a JSON file describing custom BIDS input filters, as used by qsiprep itself',
    )
    parser.add_argument(
        '--bids-database-dir',
        type=Path,
        default=None,
        help='a PyBIDS database directory to reuse or create, to avoid re-indexing',
    )
    parser.add_argument(
        '--skip-bids-validation',
        action='store_true',
        default=False,
        help='assume the input dataset is BIDS compliant and skip validation',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='list every subject in each profile instead of truncating',
    )
    parser.add_argument('--version', action='version', version=f'qsiprep-recommend {__version__}')
    return parser


def main(argv=None) -> int:
    """Entry point. Returns the process exit status."""
    from qsiprep.recommend.probe import probe_dataset
    from qsiprep.recommend.profiles import build_profiles
    from qsiprep.recommend.report import render_report

    opts = _build_parser().parse_args(argv)

    filters = None
    if opts.bids_filter_file is not None:
        filters = json.loads(opts.bids_filter_file.read_text())

    try:
        facts, skipped = probe_dataset(
            opts.bids_dir,
            participant_label=opts.participant_label,
            session_id=opts.session_id,
            filters=filters,
            bids_validate=not opts.skip_bids_validation,
            database_dir=opts.bids_database_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f'qsiprep-recommend: {exc}', file=sys.stderr)
        return 1

    if not facts:
        detail = ''
        if skipped:
            detail = ' Skipped: ' + '; '.join(
                f'sub-{subject} ({reason})' for subject, reason in skipped
            )
        print(
            f'qsiprep-recommend: no subjects with DWI data were found.{detail}',
            file=sys.stderr,
        )
        return 1

    print(
        render_report(
            opts.bids_dir,
            build_profiles(facts),
            skipped=skipped,
            output_dir=opts.output_dir,
            verbose=opts.verbose,
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Register the console script**

Edit `pyproject.toml`, replacing the `[project.scripts]` table at lines 105-107:

```toml
[project.scripts]
qsiprep = "qsiprep.cli.run:main"
qsiprep-recommend = "qsiprep.cli.recommend:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/test_recommend_cli.py -v`
Expected: PASS, 7 tests

If `test_printed_command_parses_with_the_qsiprep_parser` fails because the output contains more than one occurrence of `qsiprep `, extract the command with a regex anchored to the last line block instead of `split`.

- [ ] **Step 6: Run the whole recommender suite**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/ -k recommend -v`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add qsiprep/cli/recommend.py qsiprep/tests/test_recommend_cli.py pyproject.toml
git commit -m "Add the qsiprep-recommend command-line entry point"
```

---

### Task 12: Documentation

**Files:**
- Create: `docs/sphinxext/recommend_rules.py`
- Create: `docs/recommend.rst`
- Modify: `docs/conf.py:45-58` (extensions list)
- Modify: `docs/index.rst` (toctree)
- Modify: `docs/quickstart.rst` (pointer near the head-motion section)
- Modify: `docs/preprocessing.rst:200` (the `--combine-all-dwis` fix)

**Interfaces:**
- Consumes: `RULES` from `qsiprep.recommend.rules`.
- Produces: the `qsiprep-recommendation-rules` directive.

- [ ] **Step 1: Write the directive**

Create `docs/sphinxext/recommend_rules.py`:

```python
"""Render the recommender's rule registry as a documentation table.

The rules are the source of truth: this directive reads them directly, so the
documentation and ``qsiprep-recommend`` cannot disagree.
"""

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


class RecommendationRules(SphinxDirective):
    """Emit a table of every registered recommendation rule."""

    has_content = False

    def run(self):
        from qsiprep.recommend.rules import RULES

        lines = [
            '.. list-table::',
            '   :header-rows: 1',
            '   :widths: 20 45 35',
            '',
            '   * - Rule',
            '     - When it applies',
            '     - Flags it may recommend',
        ]
        for rule in RULES:
            flags = ', '.join(f'``{flag}``' for flag in rule.flags) or '*(note only)*'
            lines.extend(
                [
                    f'   * - ``{rule.name}``',
                    f'     - {rule.condition}',
                    f'     - {flags}',
                ]
            )

        container = nodes.section()
        container.document = self.state.document
        self.state.nested_parse(StringList(lines), self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive('qsiprep-recommendation-rules', RecommendationRules)
    return {'version': '1.0', 'parallel_read_safe': True}
```

- [ ] **Step 2: Register the extension**

In `docs/conf.py`, add `'recommend_rules'` to the `extensions` list (the `sphinxext` directory is already on `sys.path` via line 32):

```python
extensions = [
    'nbsphinx',
    'nipype.sphinxext.apidoc',
    'nipype.sphinxext.plot_workflow',
    'recommend_rules',
    'recommonmark',
    ...
]
```

- [ ] **Step 3: Write the documentation page**

Create `docs/recommend.rst`:

```rst
.. include:: links.rst

##################
Recommending flags
##################

*QSIPrep* has many options, and the right ones depend on how your data were
acquired. ``qsiprep-recommend`` reads a BIDS dataset and prints the flags this
documentation recommends for it, along with the reasoning for each one::

  qsiprep-recommend /path/to/bids --output-dir /path/to/outputs

It reads only JSON sidecars, ``.bval``/``.bvec`` files, image headers, and
``participants.tsv``. No image data is processed, so it finishes in seconds and
needs none of the external neuroimaging tools that a real run does.

Subjects whose data were acquired the same way are grouped into a single
*acquisition profile*, so a homogeneous study produces one block of advice no
matter how many participants it has, while a study that mixes acquisitions
produces one block per acquisition.

Advice is sorted into five groups: **Recommended** (this documentation says to
do it), **Consider** (a real choice your data makes relevant, which the
documentation does not settle), **Notes**, **Warnings**, and **Not determined**
(a rule could not decide, and why). The last group usually means a metadata
field is missing from your sidecars.

.. important::

   The recommendations are a starting point, not a substitute for looking at
   your data and the visual reports.


*****************
Command-line help
*****************

.. argparse::
   :ref: qsiprep.cli.recommend._build_parser
   :prog: qsiprep-recommend
   :nodefault:
   :nodefaultconst:


**********
Rule table
**********

.. qsiprep-recommendation-rules::
```

- [ ] **Step 4: Add the page to the toctree**

In `docs/index.rst`, add `recommend` to the toctree, after `quickstart`.

- [ ] **Step 5: Fix the stale flag in preprocessing.rst**

At `docs/preprocessing.rst:199-200`, replace:

```rst
  --distortion-group-merge average \
  --combine-all-dwis \
```

with:

```rst
  --distortion-group-merge average \
```

and add a sentence after the block: `Combining all DWIs is the default, so no additional flag is needed; pass ``--separate-all-dwis`` if you want the runs kept apart.`

- [ ] **Step 6: Add a pointer from quickstart.rst**

In `docs/quickstart.rst`, after the opening paragraph (around line 8), add:

```rst
.. tip::

   ``qsiprep-recommend /path/to/bids`` inspects your dataset and suggests the
   flags described on this page. See :doc:`recommend`.
```

- [ ] **Step 7: Build the documentation**

Run: `micromamba run -n lincapps python -m sphinx -b html docs docs/_build/test -q`
Expected: builds with no errors; the rule table and the CLI reference appear in `docs/_build/test/recommend.html`

If sphinx is not installed in the environment, install the doc extras first: `micromamba run -n lincapps python -m pip install -e '.[doc]'`

- [ ] **Step 8: Run the full recommender suite one more time**

Run: `micromamba run -n lincapps python -m pytest qsiprep/tests/ -k recommend -v`
Expected: PASS

- [ ] **Step 9: Lint everything**

Run: `micromamba run -n lincapps python -m ruff check qsiprep/recommend/ qsiprep/cli/recommend.py qsiprep/tests/ docs/sphinxext/recommend_rules.py`
Expected: no errors

- [ ] **Step 10: Commit**

```bash
git add docs/sphinxext/recommend_rules.py docs/recommend.rst docs/conf.py docs/index.rst \
        docs/quickstart.rst docs/preprocessing.rst
git commit -m "Document qsiprep-recommend and generate its rule table from the registry"
```

---

## Out of scope

Recorded so the implementer does not add them on impulse:

- **Image-based probes**, including running N4 on anatomicals and b=0 images to decide on B1 bias correction (requested in the issue comments). `probe.py` is the seam where this would attach later.
- **A `dwidenoise2` recommendation** based on complex or phase data. No documentation supports one.
- **`--anat-biascorrect auto`** on NORM anatomicals. The parser's own help says scanner-side normalization does not remove the need for N4.
- **Validating a user-supplied command line.**
- **Fixing the `--dwi-only` deprecation inconsistency** in `qsiprep/cli/parser.py`. The flag is listed in the `deprecations` dict but registered as a plain `store_true`, so it never warns. Worth a separate issue; the recommender simply never emits it.
