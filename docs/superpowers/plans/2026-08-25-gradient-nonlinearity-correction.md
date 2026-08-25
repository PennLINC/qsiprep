# Gradient Nonlinearity Distortion Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct gradient nonlinearity in QSIPrep — spatially, by adding a TORTOISE-generated gradwarp field to the existing composed transform chain, and in the diffusion encoding, by writing a voxelwise `grad_dev` derivative.

**Architecture:** Two TORTOISE V4 standalone binaries generate a displacement field and a 9-component gradient deviation map from a scanner coefficient file. The field enters `ComposeTransforms` between HMC and SDC, preserving QSIPrep's single-resample guarantee. Whether the spatial warp applies, and in which dimensions, is resolved per correction unit from the DICOM `ImageType` tag, overridable with `--force gradients` / `--ignore gradients`.

**Tech Stack:** Python 3.11+, nipype, nibabel, numpy, pytest, TORTOISE V4 binaries (`CreateNonlinearityDisplacementMap`, `CreateGradientNonlinearityBMatrix`).

**Spec:** `docs/superpowers/specs/2026-08-25-gradient-nonlinearity-correction-design.md`

## Global Constraints

- Run every command through micromamba: `micromamba run -n lincapps <command>`. Never `pixi`, never bare `pytest`.
- Binary-backed tests use the `_require()` + `shutil.which` skip pattern from `qsiprep/tests/test_interfaces_diffprep.py:23-27`. They are **not** permanently skipped: CI's `unit_tests` job runs inside `pennlinc/qsiprep:test`, which ships the TORTOISE tools.
- `CreateNonlinearityDisplacementMap` argument order is `<coeff_file> <base_nifti> <output.nii> [is_GE]`. Coefficient file **first**.
- `CreateNonlinearityDisplacementMap`'s `is_GE` is `(bool)(argv[4])` — a **pointer** cast. Passing `0` yields **true**. The argument must be omitted entirely for non-GE data.
- `CreateGradientNonlinearityBMatrix`'s `getIsGE()` uses `atoi()`, so `--isGE 0` correctly means false. The two binaries are wrapped differently on purpose.
- The direct output of `CreateNonlinearityDisplacementMap` is the field that gets composed and that resamples b0s. Do **not** invert it.
- Output files of `CreateGradientNonlinearityBMatrix` are written **beside its `-f` input**, not in the working directory. Stage that input with `copyfile=True`.
- Never write the coefficient file's full host path into a derivative sidecar. Basename only.
- Ruff must pass: `micromamba run -n lincapps ruff check --diff && micromamba run -n lincapps ruff format --diff`.

---

## File Structure

**Create:**
- `qsiprep/tests/gradient_fixtures.py` — synthetic Siemens `.grad` and ITK field writers
- `qsiprep/interfaces/gradunwarp.py` — the three interfaces
- `qsiprep/workflows/dwi/gradwarp.py` — `GradwarpPlan`, `resolve_gradwarp_plan`, `init_gradwarp_wf`
- `qsiprep/tests/test_gradient_fixtures.py`
- `qsiprep/tests/test_gradwarp_plan.py`
- `qsiprep/tests/test_interfaces_gradunwarp.py`
- `qsiprep/tests/test_workflows_gradwarp.py`

**Modify:**
- `qsiprep/cli/parser.py` — `--ignore` choices, new `--force`, new `--gradient-file`, validation
- `qsiprep/config.py` — `force`, `gradient_file` keys
- `qsiprep/tests/preproc_factory.py` — `per_file_metadata` kwarg
- `qsiprep/interfaces/gradients.py:445,497` — `gradwarp` slot in `transform_order`
- `qsiprep/workflows/dwi/resampling.py` — `gradwarp_field` input
- `qsiprep/workflows/dwi/base.py` — instantiate `gradwarp_wf`
- `qsiprep/workflows/dwi/fsl.py`, `diffprep.py`, `hmc_sdc.py` — gradwarp SDC estimation inputs
- `qsiprep/workflows/dwi/finalize.py`, `derivatives.py` — grad_dev node and sink
- `qsiprep/data/io_spec.json` — `graddev` suffix
- `qsiprep/interfaces/reports.py` — `gradient_correction` field
- `docs/usage.rst` — flag documentation

---

## Task 1: Simulated coefficient file fixtures

The Siemens `.grad` grammar in `gradcal.cxx:99-183` is narrow, and a fixture that
does not parse would make every downstream binary test silently meaningless.
Build the generator first and prove it satisfies the grammar.

Three constraints, all derived from the C++ reader:
1. The axis letter must be the **last character on the line**. The coefficient is
   `stof(input.substr(posA3+1, input.size() - posA3 - 2))`, which drops the final
   character; if the axis letter is not last, it lands inside the parsed float.
2. The axis letter must be the **only** `x`/`y`/`z` on the line. The axis is chosen
   by `input.find("x")` over the whole string.
3. `R0` is read as `atof(input.substr(1, 5)) * 1000`, so the value must occupy
   columns 1–5 of a line containing `= R0`.
4. `(` must appear at index >= 3 and < 10 (`find_first_of("(", 3, 3)`).

**Files:**
- Create: `qsiprep/tests/gradient_fixtures.py`
- Test: `qsiprep/tests/test_gradient_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `write_siemens_grad(path, terms=None, r0_m=0.250) -> pathlib.Path`, where
    `terms` is a list of `(axis, l, m, coefficient)` tuples with `axis in {'x','y','z'}`.
    Defaults to a small realistic third-order set.
  - `write_itk_field(path, shape=(8, 8, 8), amplitude=0.5) -> pathlib.Path`,
    writing a 5D `(X, Y, Z, 1, 3)` NIfTI vector field.
  - `write_dwi_with_gradients(path, nvols=6) -> str`, a tiny 4D DWI with
    sibling `.bval`/`.bvec`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_gradient_fixtures.py`:

```python
"""The synthetic coefficient fixtures must satisfy TORTOISE's .grad grammar.

Every binary-backed test downstream feeds these files to the real TORTOISE
tools. A fixture that silently fails to parse would turn those tests into
no-ops, so the grammar is asserted here directly against the rules in
``src/tools/gradnonlin/gradcal.cxx:99-183``.
"""

import nibabel as nb
import numpy as np

from qsiprep.tests.gradient_fixtures import (
    write_dwi_with_gradients,
    write_itk_field,
    write_siemens_grad,
)


def _coefficient_lines(text):
    return [ln for ln in text.splitlines() if '(' in ln and ')' in ln]


def test_siemens_grad_axis_letter_is_last_character(tmp_path):
    """TORTOISE drops the final character when parsing the coefficient."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert line[-1] in 'xyz', line


def test_siemens_grad_axis_letter_appears_exactly_once(tmp_path):
    """The axis is chosen by searching the whole line for x, then y, then z."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert sum(line.count(axis) for axis in 'xyz') == 1, line


def test_siemens_grad_open_paren_position(tmp_path):
    """find_first_of("(", 3, 3) requires the paren at index >= 3, and < 10."""
    text = write_siemens_grad(tmp_path / 'coeff.grad').read_text()
    for line in _coefficient_lines(text):
        assert 3 <= line.index('(') < 10, line


def test_siemens_grad_r0_occupies_columns_one_to_five(tmp_path):
    """R0 = atof(substr(1, 5)) * 1000, so 0.250 must sit at columns 1-5."""
    text = write_siemens_grad(tmp_path / 'coeff.grad', r0_m=0.250).read_text()
    r0_lines = [ln for ln in text.splitlines() if '= R0' in ln]
    assert len(r0_lines) == 1
    assert float(r0_lines[0][1:6]) * 1000 == 250.0


def test_siemens_grad_records_requested_terms(tmp_path):
    path = write_siemens_grad(
        tmp_path / 'coeff.grad', terms=[('x', 3, 1, -0.0234), ('z', 5, 0, 0.0011)]
    )
    lines = _coefficient_lines(path.read_text())
    assert len(lines) == 2
    assert lines[0].endswith('x') and '( 3, 1)' in lines[0]
    assert lines[1].endswith('z') and '( 5, 0)' in lines[1]


def test_itk_field_is_five_dimensional_vector_image(tmp_path):
    img = nb.load(str(write_itk_field(tmp_path / 'field.nii', shape=(4, 5, 6))))
    assert img.shape == (4, 5, 6, 1, 3)


def test_itk_field_is_nonzero(tmp_path):
    """A zero field would make a warp test pass for the wrong reason."""
    img = nb.load(str(write_itk_field(tmp_path / 'field.nii')))
    assert np.abs(np.asanyarray(img.dataobj)).max() > 0


def test_write_dwi_with_gradients_makes_siblings(tmp_path):
    path = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz', nvols=5)
    stem = str(path).split('.nii')[0]
    assert nb.load(path).shape == (8, 8, 8, 5)
    assert np.loadtxt(stem + '.bval').size == 5
    assert np.loadtxt(stem + '.bvec').shape == (3, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_gradient_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.tests.gradient_fixtures'`

- [ ] **Step 3: Write the implementation**

Create `qsiprep/tests/gradient_fixtures.py`:

```python
"""Synthetic gradient nonlinearity inputs for tests.

The Siemens ``.grad`` writer targets the reader at
``src/tools/gradnonlin/gradcal.cxx:99-183`` in TORTOISEV4, which is stricter
than it looks: the axis letter must be the last character on the line (the
coefficient substring drops the final character) and the only x/y/z on it (the
axis is found by searching the whole line), and R0 is read from columns 1-5.
``test_gradient_fixtures.py`` asserts each of these.
"""

from pathlib import Path

import nibabel as nb
import numpy as np

#: A small, physically plausible third-order set. Real coefficient files hold
#: dozens of terms; these are enough to produce a non-trivial field.
DEFAULT_TERMS = [
    ('x', 3, 1, -0.0234),
    ('y', 3, 1, 0.0198),
    ('z', 3, 0, 0.0456),
    ('z', 5, 0, 0.0011),
]


def write_siemens_grad(path, terms=None, r0_m=0.250):
    """Write a Siemens-format ``.grad`` coefficient file.

    ``terms`` is a list of ``(axis, l, m, coefficient)``; ``r0_m`` is the
    reference radius in metres (TORTOISE multiplies it by 1000).
    """
    terms = DEFAULT_TERMS if terms is None else terms
    path = Path(path)
    lines = [
        ' Synthetic gradient coefficients for tests',
        f' {r0_m:.3f} = R0',
        '',
    ]
    for index, (axis, l_val, m_val, coefficient) in enumerate(terms, start=1):
        # Two leading columns put "(" at index >= 3. The coefficient is written
        # with no trailing content except the axis letter, which must be last.
        lines.append(f'{index:>3d} A({l_val:>2d},{m_val:>2d}) {coefficient: .6f} {axis}')
    lines.append('')
    path.write_text('\n'.join(lines))
    return path


def write_itk_field(path, shape=(8, 8, 8), amplitude=0.5):
    """Write a smooth, non-zero ITK displacement field as a 5D vector NIfTI."""
    path = Path(path)
    grid = np.meshgrid(
        *[np.linspace(-1.0, 1.0, n) for n in shape],
        indexing='ij',
    )
    data = np.zeros(shape + (1, 3), dtype='float32')
    for component in range(3):
        data[..., 0, component] = amplitude * grid[component] ** 2
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return path


def write_dwi_with_gradients(path, nvols=6):
    """Write a tiny 4D DWI plus sibling .bval/.bvec, and return its path."""
    path = Path(path)
    nb.Nifti1Image(
        np.random.default_rng(0).random((8, 8, 8, nvols)).astype('float32'), np.eye(4)
    ).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    bvecs = np.zeros((3, nvols))
    bvecs[0, 1:] = 1.0
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', bvecs, fmt='%.6f')
    return str(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_gradient_fixtures.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add qsiprep/tests/gradient_fixtures.py qsiprep/tests/test_gradient_fixtures.py
git commit -m "test: add synthetic gradient coefficient fixtures

The Siemens .grad grammar is strict about column positions and the
trailing axis letter. Assert those rules directly so downstream binary
tests cannot silently degrade into no-ops."
```

---

## Task 2: Plan resolution from ImageType

**Files:**
- Create: `qsiprep/workflows/dwi/gradwarp.py`
- Modify: `qsiprep/tests/preproc_factory.py`
- Test: `qsiprep/tests/test_gradwarp_plan.py`

**Interfaces:**
- Consumes: `PreprocUnit.dwi_records` (`qsiprep/grouping/adapters.py:76`), each a
  `FileRecord` with `.path` and `.metadata`.
- Produces:
  - `GradwarpPlan(coeff_file: str, warp_dim: str | None, is_ge: bool, basis: str)`,
    frozen dataclass. `warp_dim` is `'3D'`, `'1D'`, or `None`. `basis` is
    `'metadata'` or `'forced'`.
  - `resolve_gradwarp_plan(unit) -> GradwarpPlan | None`
  - `make_preproc_unit(..., per_file_metadata: dict[str, dict] | None = None)`

Precedence rule to encode: when a single run carries **both** `DIS2D` and
`DIS3D`, `DIS3D` wins. It is the more-corrected claim, so it leaves the smaller
residual — the same "minimum warp" principle used across files in a unit.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_gradwarp_plan.py`:

```python
"""Resolution of the per-unit gradwarp plan from ImageType and the CLI flags."""

import pytest

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.workflows.dwi.gradwarp import resolve_gradwarp_plan

DWI = '/data/sub-01_dwi.nii.gz'
COEFF = '/opt/coeff.grad'


@pytest.fixture(autouse=True)
def _reset_config():
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    yield
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []


def _unit(image_type=None, manufacturer='SIEMENS', files=(DWI,), per_file=None):
    metadata = {'Manufacturer': manufacturer}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit(list(files), metadata=metadata, per_file_metadata=per_file)


def test_no_gradient_file_means_no_plan():
    """ImageType is never consulted when the feature is off."""
    assert resolve_gradwarp_plan(_unit(['ORIGINAL', 'PRIMARY'])) is None


def test_ignore_gradients_disables_everything():
    config.workflow.gradient_file = COEFF
    config.workflow.ignore = ['gradients']
    assert resolve_gradwarp_plan(_unit()) is None


@pytest.mark.parametrize(
    ('image_type', 'expected'),
    [
        (None, '3D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND'], '3D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND', 'DIS2D'], '1D'),
        (['ORIGINAL', 'PRIMARY', 'M', 'ND', 'DIS3D'], None),
        (['ORIGINAL', 'DIS2D', 'DIS3D'], None),
    ],
)
def test_warp_dim_from_image_type(image_type, expected):
    config.workflow.gradient_file = COEFF
    plan = resolve_gradwarp_plan(_unit(image_type))
    assert plan is not None
    assert plan.warp_dim == expected
    assert plan.basis == 'metadata'


def test_image_type_may_be_a_bare_string():
    """Some converters write ImageType as a backslash-joined string."""
    config.workflow.gradient_file = COEFF
    plan = resolve_gradwarp_plan(_unit('ORIGINAL\\PRIMARY\\M\\DIS2D'))
    assert plan.warp_dim == '1D'


def test_force_overrides_metadata():
    config.workflow.gradient_file = COEFF
    config.workflow.force = ['gradients']
    plan = resolve_gradwarp_plan(_unit(['ORIGINAL', 'DIS3D']))
    assert plan.warp_dim == '3D'
    assert plan.basis == 'forced'


def test_mixed_image_types_take_the_minimum_warp(caplog):
    """A unit is concatenated before HMC and shares one field, so the members
    must agree. Under-correcting is recoverable; double-correcting is not."""
    config.workflow.gradient_file = COEFF
    other = '/data/sub-01_run-2_dwi.nii.gz'
    plan = resolve_gradwarp_plan(
        _unit(
            ['ORIGINAL', 'PRIMARY'],
            files=(DWI, other),
            per_file={other: {'ImageType': ['ORIGINAL', 'DIS2D']}},
        )
    )
    assert plan.warp_dim == '1D'
    assert other in caplog.text


def test_consistent_image_types_do_not_warn(caplog):
    config.workflow.gradient_file = COEFF
    other = '/data/sub-01_run-2_dwi.nii.gz'
    resolve_gradwarp_plan(_unit(['ORIGINAL', 'DIS2D'], files=(DWI, other)))
    assert 'disagree' not in caplog.text


@pytest.mark.parametrize(
    ('manufacturer', 'expected'),
    [
        ('GE MEDICAL SYSTEMS', True),
        ('ge medical systems', True),
        ('  GE  ', True),
        ('SIEMENS', False),
        ('Philips Medical Systems', False),
        ('', False),
    ],
)
def test_is_ge_detection(manufacturer, expected):
    """Manufacturer is free text from DICOM, so variants must be handled."""
    config.workflow.gradient_file = COEFF
    assert resolve_gradwarp_plan(_unit(manufacturer=manufacturer)).is_ge is expected


def test_plan_carries_the_coefficient_file():
    config.workflow.gradient_file = COEFF
    assert resolve_gradwarp_plan(_unit()).coeff_file == COEFF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_gradwarp_plan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.workflows.dwi.gradwarp'`

- [ ] **Step 3: Add the per-file metadata override to the factory**

In `qsiprep/tests/preproc_factory.py`, add the keyword to the signature after
`metadata`, mirroring the existing per-file `pe_dirs` pattern:

```python
    metadata: dict | None = None,
    per_file_metadata: dict[str, dict] | None = None,
```

and inside the record loop, after `record_meta.update(metadata)`:

```python
        record_meta.update((per_file_metadata or {}).get(path, {}))
```

Extend the docstring's `pe_dirs` sentence to mention it:

```
    ``per_file_metadata`` overrides sidecar keys for individual files (keyed by
    path), layered on top of ``metadata``.
```

- [ ] **Step 4: Write the plan resolver**

Create `qsiprep/workflows/dwi/gradwarp.py`:

```python
"""Gradient nonlinearity correction.

Resolves, per correction unit, whether the images need spatial gradwarp
correction and in which dimensions, then builds the field with TORTOISE's
standalone tools.

The spatial correction and the voxelwise b-matrix (``grad_dev``) are separable:
a scanner that tags a run ``DIS3D`` has already corrected the geometry, but no
scanner can correct the diffusion encoding, because the bval/bvec table holds
one value per volume and cannot express a spatially varying encoding.
"""

import dataclasses

from ... import config

#: Ordering used to reconcile disagreeing runs in one unit. Lower is less
#: correction, and less correction is the recoverable error.
_WARP_RANK = {None: 0, '1D': 1, '3D': 2}
_RANK_TO_WARP = {rank: warp for warp, rank in _WARP_RANK.items()}


@dataclasses.dataclass(frozen=True)
class GradwarpPlan:
    """What gradient correction to apply to one correction unit.

    ``warp_dim`` is ``'3D'`` (full spatial correction), ``'1D'`` (through-plane
    residual only, for scanner-corrected DIS2D data), or ``None`` (no spatial
    correction; the grad_dev map is still produced).
    """

    coeff_file: str
    warp_dim: str | None
    is_ge: bool
    basis: str  # 'metadata' | 'forced'


def _image_type_tags(metadata):
    """Normalise ImageType, which may be a list or a backslash-joined string."""
    image_type = metadata.get('ImageType') or ()
    if isinstance(image_type, str):
        image_type = image_type.split('\\')
    return {str(tag).strip().upper() for tag in image_type}


def _warp_dim_for(metadata):
    """Residual spatial distortion implied by one run's ImageType.

    DIS3D wins over DIS2D when both are present: it is the more-corrected
    claim, so it leaves the smaller residual.
    """
    tags = _image_type_tags(metadata)
    if 'DIS3D' in tags:
        return None
    if 'DIS2D' in tags:
        return '1D'
    return '3D'


def _is_ge(metadata):
    """True when the DICOM Manufacturer field names GE.

    Free text from DICOM: 'GE MEDICAL SYSTEMS', 'SIEMENS', 'Philips Medical
    Systems'. Drives the z-origin shift TORTOISE applies for GE coefficients.
    """
    return str(metadata.get('Manufacturer', '')).strip().upper().startswith('GE')


def resolve_gradwarp_plan(unit):
    """Decide the gradient correction for one PreprocUnit, or None."""
    coeff_file = config.workflow.gradient_file
    if not coeff_file or 'gradients' in (config.workflow.ignore or []):
        return None

    records = unit.dwi_records
    is_ge = any(_is_ge(record.metadata) for record in records)

    if 'gradients' in (config.workflow.force or []):
        config.loggers.workflow.info(
            'Gradient correction: forced 3D spatial warp for %s (--force gradients).',
            unit.output_name,
        )
        return GradwarpPlan(str(coeff_file), '3D', is_ge, 'forced')

    per_file = {record.path: _warp_dim_for(record.metadata) for record in records}
    ranks = {path: _WARP_RANK[warp] for path, warp in per_file.items()}
    warp_dim = _RANK_TO_WARP[min(ranks.values())]

    if len(set(ranks.values())) > 1:
        config.loggers.workflow.warning(
            'Runs in %s disagree about scanner gradwarp correction (%s). These '
            'series are concatenated before head motion correction and share one '
            'field, so the least-correcting value (%s) is used to avoid '
            'double-correcting an already-corrected run.',
            unit.output_name,
            ', '.join(f'{path}: {warp or "none"}' for path, warp in sorted(per_file.items())),
            warp_dim or 'none',
        )
    else:
        config.loggers.workflow.info(
            'Gradient correction for %s: spatial warp %s (from ImageType).',
            unit.output_name,
            warp_dim or 'disabled',
        )

    return GradwarpPlan(str(coeff_file), warp_dim, is_ge, 'metadata')
```

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_gradwarp_plan.py -v`
Expected: PASS (18 tests)

- [ ] **Step 6: Verify the factory change broke nothing**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_grouping_adapters.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add qsiprep/workflows/dwi/gradwarp.py qsiprep/tests/test_gradwarp_plan.py qsiprep/tests/preproc_factory.py
git commit -m "feat: resolve gradwarp plan from ImageType

DIS2D leaves through-plane distortion, DIS3D leaves only the voxelwise
b-matrix. Runs in one unit share a field, so disagreement resolves to the
least-correcting value."
```

---

## Task 3: MaskWarpDimensions interface

Reproduces `TORTOISE.cxx:1994-2012`. Pure nibabel/numpy — no binary, so this is
fully tested locally.

**Files:**
- Create: `qsiprep/interfaces/gradunwarp.py`
- Test: `qsiprep/tests/test_interfaces_gradunwarp.py`

**Interfaces:**
- Consumes: `write_itk_field` from Task 1.
- Produces: `MaskWarpDimensions(in_file, warp_dim) -> out_file`, with
  `warp_dim in {'3D', '2D', '1D'}`.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_interfaces_gradunwarp.py`:

```python
"""Tests for the TORTOISE gradient nonlinearity interfaces.

Pure-Python behaviour -- command-line construction and field masking -- is
tested unconditionally. Tests that exercise the real TORTOISE binaries are
guarded with ``shutil.which`` and skip when those binaries are absent. They are
*not* permanently offline: CircleCI's ``unit_tests`` job runs pytest inside the
``pennlinc/qsiprep:test`` image, which ships the TORTOISE tools.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep.tests.gradient_fixtures import (
    write_dwi_with_gradients,
    write_itk_field,
    write_siemens_grad,
)


def _require(*binaries):
    """Skip unless every named TORTOISE binary is on PATH."""
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        pytest.skip(f'{", ".join(missing)} required for this test')


def _components(path):
    """Return the (X, Y, Z, 3) displacement components of an ITK field."""
    data = np.asanyarray(nb.load(str(path)).dataobj)
    return data.reshape(data.shape[:3] + (3,))


@pytest.mark.parametrize(
    ('warp_dim', 'zeroed'),
    [('3D', ()), ('2D', (2,)), ('1D', (0, 1))],
)
def test_mask_warp_dimensions(tmp_path, warp_dim, zeroed):
    """2D zeroes the through-plane component; 1D keeps only that component."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim=warp_dim).run(cwd=str(tmp_path))

    before = _components(field)
    after = _components(result.outputs.out_file)
    for component in range(3):
        if component in zeroed:
            assert np.all(after[..., component] == 0)
        else:
            assert np.allclose(after[..., component], before[..., component])


def test_mask_warp_dimensions_preserves_geometry(tmp_path):
    """The masked field must stay on the same grid to compose correctly."""
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    result = MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))

    original, masked = nb.load(str(field)), nb.load(result.outputs.out_file)
    assert masked.shape == original.shape
    assert np.allclose(masked.affine, original.affine)


def test_mask_warp_dimensions_does_not_modify_input(tmp_path):
    from qsiprep.interfaces.gradunwarp import MaskWarpDimensions

    field = write_itk_field(tmp_path / 'field.nii')
    before = _components(field).copy()
    MaskWarpDimensions(in_file=str(field), warp_dim='1D').run(cwd=str(tmp_path))
    assert np.allclose(_components(field), before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.interfaces.gradunwarp'`

- [ ] **Step 3: Write the implementation**

Create `qsiprep/interfaces/gradunwarp.py`:

```python
"""Interfaces to TORTOISE V4's gradient nonlinearity tools.

Three hazards in the upstream binaries drive the shape of these wrappers, all
verified against TORTOISEV4 at ``main``:

1. ``CreateNonlinearityDisplacementMap`` takes the coefficient file as its
   *first* positional argument (``mk_displacement(argv[1], img, is_GE)`` in
   ``src/tools/gradnonlin/mk_displacementMaps.cxx``). A stale, unbuilt copy of
   that file elsewhere in the tree has the arguments reversed.
2. That tool reads ``is_GE`` as ``(bool)(argv[4])`` -- a cast of the *pointer*,
   not its contents -- so passing ``0`` yields **true**. The argument is
   appended only when GE. ``CreateGradientNonlinearityBMatrix`` is different:
   its ``getIsGE()`` uses ``atoi()``, so ``--isGE 0`` is correctly false.
3. ``mk_displacement`` returns the field TORTOISE names ``gradwarp_field_inv``,
   and that is the file ``FINALDATA.cxx:548`` composes and ``DRBUDDI.cxx:141``
   resamples with. It must **not** be inverted here.
"""

import os
import os.path as op

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

#: Displacement components zeroed for each warp dimensionality, mirroring
#: ``TORTOISE.cxx:1994-2012``.
_ZEROED_COMPONENTS = {'3D': (), '2D': (2,), '1D': (0, 1)}


class _MaskWarpDimensionsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='ITK displacement field')
    warp_dim = traits.Enum(
        '3D',
        '2D',
        '1D',
        usedefault=True,
        desc='Which displacement components to keep. "3D" keeps all; "2D" '
        'zeroes the through-plane component; "1D" keeps only it.',
    )


class _MaskWarpDimensionsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Displacement field with components zeroed')


class MaskWarpDimensions(SimpleInterface):
    """Zero displacement components a scanner has already corrected."""

    input_spec = _MaskWarpDimensionsInputSpec
    output_spec = _MaskWarpDimensionsOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.array(img.dataobj, dtype='float32')
        # ITK vector fields are (X, Y, Z, 1, 3); indexing the last axis works
        # whether or not the singleton dimension is present.
        for component in _ZEROED_COMPONENTS[self.inputs.warp_dim]:
            data[..., component] = 0
        out_file = op.join(runtime.cwd, 'gradwarp_field_masked.nii')
        nb.Nifti1Image(data, img.affine, img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/gradunwarp.py qsiprep/tests/test_interfaces_gradunwarp.py
git commit -m "feat: add MaskWarpDimensions for scanner-corrected dimensions"
```

---

## Task 4: CreateNonlinearityDisplacementMap interface

**Files:**
- Modify: `qsiprep/interfaces/gradunwarp.py`
- Test: `qsiprep/tests/test_interfaces_gradunwarp.py`

**Interfaces:**
- Consumes: `write_siemens_grad`, `write_dwi_with_gradients` from Task 1.
- Produces: `CreateNonlinearityDisplacementMap(coeff_file, ref_image, out_field, is_ge) -> out_field`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_gradunwarp.py`:

```python
def test_displacement_map_puts_coefficients_first(tmp_path):
    """mk_displacement(argv[1], img, is_GE): coefficient file, then NIfTI.

    A stale unbuilt copy of that source has the arguments reversed; getting
    this backwards produces a plausible-looking wrong field, not an error.
    """
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(coeff_file=str(coeff), ref_image=ref)

    args = iface.cmdline.split()
    assert args[0] == 'CreateNonlinearityDisplacementMap'
    assert args[1] == str(coeff)
    assert args[2] == ref


def test_displacement_map_omits_is_ge_when_false(tmp_path):
    """is_GE=(bool)(argv[4]) casts the pointer, so "0" would read as true.

    The only way to express false is to pass no fourth argument at all.
    """
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(
        coeff_file=str(coeff), ref_image=ref, is_ge=False
    )
    assert len(iface.cmdline.split()) == 4


def test_displacement_map_appends_is_ge_when_true(tmp_path):
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    iface = CreateNonlinearityDisplacementMap(
        coeff_file=str(coeff), ref_image=ref, is_ge=True
    )
    args = iface.cmdline.split()
    assert len(args) == 5
    assert args[4] == '1'


def test_displacement_map_runs_on_synthetic_coefficients(tmp_path):
    """End-to-end against the real binary, in CI's container."""
    from qsiprep.interfaces.gradunwarp import CreateNonlinearityDisplacementMap

    _require('CreateNonlinearityDisplacementMap')
    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    ref = write_dwi_with_gradients(tmp_path / 'ref.nii.gz')
    result = CreateNonlinearityDisplacementMap(
        coeff_file=str(coeff), ref_image=ref
    ).run(cwd=str(tmp_path))

    field = nb.load(result.outputs.out_field)
    assert field.shape[:3] == (8, 8, 8)
    # A parsed coefficient file must produce a non-trivial field; an all-zero
    # result means the .grad fixture did not parse.
    assert np.abs(np.asanyarray(field.dataobj)).max() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -k displacement_map -v`
Expected: FAIL — `ImportError: cannot import name 'CreateNonlinearityDisplacementMap'`

- [ ] **Step 3: Write the implementation**

Append to `qsiprep/interfaces/gradunwarp.py`:

```python
class _CreateNonlinearityDisplacementMapInputSpec(CommandLineInputSpec):
    coeff_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Scanner gradient coefficient file (.grad, .dat, .gc) or gcal file',
    )
    ref_image = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='NIfTI defining the grid the field is generated on',
    )
    out_field = traits.Str(
        'gradwarp_field.nii',
        usedefault=True,
        argstr='%s',
        position=2,
        desc='Output displacement field name. Must end in .nii for the ITK writer.',
    )
    # No argstr: appended in _parse_inputs only when True. See module docstring.
    is_ge = traits.Bool(False, usedefault=True, desc='Scanner is GE')


class _CreateNonlinearityDisplacementMapOutputSpec(TraitedSpec):
    out_field = File(exists=True, desc='Gradwarp displacement field, native space')


class CreateNonlinearityDisplacementMap(CommandLine):
    """Expand gradient coefficients into a displacement field.

    The output is TORTOISE's ``gradwarp_field_inv``, which is what gets
    composed and what resamples b=0 images. Do not invert it.
    """

    input_spec = _CreateNonlinearityDisplacementMapInputSpec
    output_spec = _CreateNonlinearityDisplacementMapOutputSpec
    _cmd = 'CreateNonlinearityDisplacementMap'

    def _parse_inputs(self, skip=None):
        parsed = super()._parse_inputs(skip=skip)
        # ``is_GE=(bool)(argv[4])`` casts the pointer: ANY fourth argument is
        # true. Omitting it is the only way to say false.
        if self.inputs.is_ge:
            parsed.append('1')
        return parsed

    def _list_outputs(self):
        return {'out_field': op.abspath(self.inputs.out_field)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -v`
Expected: PASS. The `runs_on_synthetic_coefficients` test SKIPs locally and runs in CI.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/gradunwarp.py qsiprep/tests/test_interfaces_gradunwarp.py
git commit -m "feat: wrap CreateNonlinearityDisplacementMap

The is_GE argument is a pointer cast upstream, so passing 0 means true.
Append it only when GE, and assert that in a test."
```

---

## Task 5: CreateGradientNonlinearityBMatrix interface

Two subtleties: output files land **beside the `-f` input** (so it must be
staged into the node directory with `copyfile=True`), and the grad_dev filename
suffix depends on whether `-g` was a coefficient file (`_graddev_c.nii`) or a
displacement field (`_graddev_f.nii`).

**Files:**
- Modify: `qsiprep/interfaces/gradunwarp.py`
- Test: `qsiprep/tests/test_interfaces_gradunwarp.py`

**Interfaces:**
- Produces: `CreateGradientNonlinearityBMatrix(final_image, nonlinearity, initial_image, is_ge) -> grad_dev, gradwarp_field`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_gradunwarp.py`:

```python
def test_bmatrix_cmdline_flags(tmp_path):
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')
    initial = write_dwi_with_gradients(tmp_path / 'initial_b0.nii.gz')
    cmd = CreateGradientNonlinearityBMatrix(
        final_image=final, initial_image=initial, nonlinearity=str(coeff)
    ).cmdline

    assert f'-f {final}' in cmd
    assert f'-i {initial}' in cmd
    assert f'-g {coeff}' in cmd


def test_bmatrix_is_ge_uses_a_value_not_omission(tmp_path):
    """Unlike CreateNonlinearityDisplacementMap, this tool's getIsGE() uses
    atoi(), so --isGE 0 correctly means false."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')
    iface = CreateGradientNonlinearityBMatrix(final_image=final, nonlinearity=str(coeff))

    assert '--isGE 0' in iface.cmdline
    iface.inputs.is_ge = True
    assert '--isGE 1' in iface.cmdline


def test_bmatrix_output_suffix_depends_on_nonlinearity_type(tmp_path):
    """Coefficients produce _graddev_c.nii; a field produces _graddev_f.nii."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz')

    from_coeffs = CreateGradientNonlinearityBMatrix(
        final_image=final, nonlinearity=str(write_siemens_grad(tmp_path / 'coeff.grad'))
    )
    assert from_coeffs._graddev_suffix() == '_graddev_c.nii'

    from_field = CreateGradientNonlinearityBMatrix(
        final_image=final, nonlinearity=str(write_itk_field(tmp_path / 'field.nii'))
    )
    assert from_field._graddev_suffix() == '_graddev_f.nii'


def test_bmatrix_runs_on_synthetic_coefficients(tmp_path):
    """End-to-end against the real binary, in CI's container."""
    from qsiprep.interfaces.gradunwarp import CreateGradientNonlinearityBMatrix

    _require('CreateGradientNonlinearityBMatrix')
    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    final = write_dwi_with_gradients(tmp_path / 'final_b0.nii.gz', nvols=1)
    result = CreateGradientNonlinearityBMatrix(
        final_image=final, nonlinearity=str(coeff)
    ).run(cwd=str(tmp_path))

    grad_dev = nb.load(result.outputs.grad_dev)
    # Nine components: the row-major 3x3 L matrix per voxel.
    assert grad_dev.shape[-1] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -k bmatrix -v`
Expected: FAIL — `ImportError: cannot import name 'CreateGradientNonlinearityBMatrix'`

- [ ] **Step 3: Write the implementation**

Append to `qsiprep/interfaces/gradunwarp.py`:

```python
class _CreateGradientNonlinearityBMatrixInputSpec(CommandLineInputSpec):
    final_image = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        argstr='-f %s',
        desc='Final preprocessed b=0, in the output space. The tool writes its '
        'outputs beside this file, so it is staged into the node directory.',
    )
    nonlinearity = File(
        exists=True,
        mandatory=True,
        argstr='-g %s',
        desc='Coefficient file or ITK gradwarp displacement field',
    )
    initial_image = File(
        exists=True,
        argstr='-i %s',
        desc='Raw native-space b=0. Omitted means the final image is native.',
    )
    is_ge = traits.Bool(False, usedefault=True, argstr='--isGE %d', desc='Scanner is GE')


class _CreateGradientNonlinearityBMatrixOutputSpec(TraitedSpec):
    grad_dev = File(exists=True, desc='9-component gradient deviation (L) map')
    gradwarp_field = File(exists=True, desc='Gradwarp displacement field')


class CreateGradientNonlinearityBMatrix(CommandLine):
    """Compute the voxelwise gradient deviation tensor.

    Emits the HCP-style 9-component L matrix per voxel. Applied downstream as
    ``L @ g``: because L carries scaling and shear, not just rotation, both the
    b-vector and the b-value deviate per voxel.
    """

    input_spec = _CreateGradientNonlinearityBMatrixInputSpec
    output_spec = _CreateGradientNonlinearityBMatrixOutputSpec
    _cmd = 'CreateGradientNonlinearityBMatrix'

    def _graddev_suffix(self):
        """The tool names its output for how the nonlinearity was supplied."""
        if '.nii' in op.basename(self.inputs.nonlinearity):
            return '_graddev_f.nii'
        return '_graddev_c.nii'

    def _list_outputs(self):
        # Outputs are named from the -f input's stem, in its directory. Because
        # final_image is copyfile=True, that directory is this node's cwd.
        staged = op.abspath(op.basename(self.inputs.final_image))
        stem = staged[: staged.rfind('.nii')]
        return {
            'grad_dev': stem + self._graddev_suffix(),
            'gradwarp_field': stem + '_gradwarp_field.nii',
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -v`
Expected: PASS (12 tests; 2 SKIP locally)

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/gradunwarp.py qsiprep/tests/test_interfaces_gradunwarp.py
git commit -m "feat: wrap CreateGradientNonlinearityBMatrix

Outputs land beside the -f input, so it is staged with copyfile=True, and
the grad_dev suffix depends on whether -g was coefficients or a field."
```

---

## Task 6: init_gradwarp_wf

**Files:**
- Modify: `qsiprep/workflows/dwi/gradwarp.py`
- Test: `qsiprep/tests/test_workflows_gradwarp.py`

**Interfaces:**
- Consumes: `resolve_gradwarp_plan` (Task 2), `CreateNonlinearityDisplacementMap`
  and `MaskWarpDimensions` (Tasks 3–4).
- Produces: `init_gradwarp_wf(unit, name='gradwarp_wf') -> Workflow | None`, with
  `inputnode.ref_image` and `outputnode.gradwarp_field`. Returns `None` when the
  plan is `None`. The workflow carries `.plan` (the `GradwarpPlan`) as an
  attribute so callers can branch without re-resolving.

The field node runs even when `warp_dim is None`, because
`CreateGradientNonlinearityBMatrix` needs a field. Only the wiring into
`ComposeTransforms` is suppressed, in Task 9.

- [ ] **Step 1: Write the failing test**

Create `qsiprep/tests/test_workflows_gradwarp.py`:

```python
"""Construction tests for the gradwarp workflow and its wiring."""

import pytest

from qsiprep import config
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients, write_siemens_grad
from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.fixture(autouse=True)
def _reset_config():
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    config.nipype.omp_nthreads = 1
    yield
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []


def _unit(tmp_path, image_type=None):
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    return make_preproc_unit([dwi], metadata=metadata)


def test_gradwarp_wf_is_none_without_a_coefficient_file(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    assert init_gradwarp_wf(_unit(tmp_path)) is None


def test_gradwarp_wf_builds_field_and_mask_nodes(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path))

    assert wf.get_node('make_field') is not None
    assert wf.get_node('mask_field') is not None
    assert wf.get_node('outputnode') is not None


def test_gradwarp_wf_masks_to_through_plane_for_dis2d(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, ['ORIGINAL', 'DIS2D']))

    assert wf.get_node('mask_field').inputs.warp_dim == '1D'
    assert wf.plan.warp_dim == '1D'


def test_gradwarp_wf_still_builds_a_field_for_dis3d(tmp_path):
    """grad_dev needs a field even when no spatial correction is applied."""
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    wf = init_gradwarp_wf(_unit(tmp_path, ['ORIGINAL', 'DIS3D']))

    assert wf.get_node('make_field') is not None
    assert wf.plan.warp_dim is None


def test_gradwarp_wf_passes_is_ge_through(tmp_path):
    from qsiprep.workflows.dwi.gradwarp import init_gradwarp_wf

    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    unit = make_preproc_unit([dwi], metadata={'Manufacturer': 'GE MEDICAL SYSTEMS'})

    assert init_gradwarp_wf(unit).get_node('make_field').inputs.is_ge is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -v`
Expected: FAIL — `ImportError: cannot import name 'init_gradwarp_wf'`

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `qsiprep/workflows/dwi/gradwarp.py`:

```python
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...interfaces.gradunwarp import CreateNonlinearityDisplacementMap, MaskWarpDimensions
```

and append:

```python
#: Boilerplate fragments, keyed by the resolved warp dimensionality. The text
#: must track the plan: claiming 3D correction on DIS2D data would be a
#: methods-section error.
_BOILERPLATE = {
    '3D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4. The full three-dimensional gradwarp '
        'displacement field was combined with the head motion, eddy current, '
        'and susceptibility distortion transforms, so the data were resampled '
        'only once.'
    ),
    '1D': (
        'Gradient nonlinearity was corrected using the scanner gradient '
        'coefficients with TORTOISE V4. Because the scanner had already applied '
        'in-plane gradwarp correction (DIS2D), only the residual through-plane '
        'component was applied here, combined with the head motion, eddy '
        'current, and susceptibility distortion transforms so the data were '
        'resampled only once.'
    ),
    None: (
        'The scanner had already applied full three-dimensional gradwarp '
        'correction (DIS3D), so no further spatial correction was performed. '
        'A voxelwise gradient deviation map was computed with TORTOISE V4 to '
        'account for the spatially varying diffusion encoding.'
    ),
}


def init_gradwarp_wf(unit, name='gradwarp_wf'):
    """Build the gradwarp displacement field for one correction unit.

    Returns ``None`` when no gradient correction was requested. The field node
    runs even when ``warp_dim`` is ``None``: the grad_dev map needs a field, and
    only the wiring into the composed transform chain is suppressed.
    """
    plan = resolve_gradwarp_plan(unit)
    if plan is None:
        return None

    workflow = Workflow(name=name)
    workflow.__desc__ = _BOILERPLATE[plan.warp_dim]
    workflow.plan = plan

    inputnode = pe.Node(niu.IdentityInterface(fields=['ref_image']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['gradwarp_field']), name='outputnode'
    )

    make_field = pe.Node(
        CreateNonlinearityDisplacementMap(coeff_file=plan.coeff_file, is_ge=plan.is_ge),
        name='make_field',
    )
    # '3D' is a passthrough, but keeping the node unconditional means the graph
    # shape does not depend on the plan.
    mask_field = pe.Node(
        MaskWarpDimensions(warp_dim=plan.warp_dim or '3D'), name='mask_field'
    )

    workflow.connect([
        (inputnode, make_field, [('ref_image', 'ref_image')]),
        (make_field, mask_field, [('out_field', 'in_file')]),
        (mask_field, outputnode, [('out_file', 'gradwarp_field')]),
    ])  # fmt:skip

    return workflow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add qsiprep/workflows/dwi/gradwarp.py qsiprep/tests/test_workflows_gradwarp.py
git commit -m "feat: add init_gradwarp_wf"
```

---

## Task 7: CLI flags, config keys, and validation

**Files:**
- Modify: `qsiprep/cli/parser.py:432-445` (`--ignore`), `parser.py:967-983` (validation)
- Modify: `qsiprep/config.py:600` region
- Test: `qsiprep/tests/test_cli.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `config.workflow.force: list[str]`, `config.workflow.gradient_file: Path | None`.
  `config.from_dict(vars(opts))` at `parser.py:986` picks both up automatically
  once the class attributes exist.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_cli.py`:

```python
def test_gradient_flags_reach_config(tmp_path, minimal_cli_args):
    """--gradient-file and --force gradients land in config.workflow."""
    from qsiprep.cli.parser import parse_args

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    parse_args([*minimal_cli_args, '--gradient-file', str(coeff), '--force', 'gradients'])
    assert str(config.workflow.gradient_file) == str(coeff)
    assert config.workflow.force == ['gradients']


def test_force_and_ignore_gradients_conflict(tmp_path, minimal_cli_args):
    from qsiprep.cli.parser import parse_args

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    with pytest.raises(SystemExit):
        parse_args([
            *minimal_cli_args,
            '--gradient-file', str(coeff),
            '--force', 'gradients',
            '--ignore', 'gradients',
        ])


def test_force_gradients_requires_a_gradient_file(minimal_cli_args):
    from qsiprep.cli.parser import parse_args

    with pytest.raises(SystemExit):
        parse_args([*minimal_cli_args, '--force', 'gradients'])


def test_gradient_file_rejects_unknown_extension(tmp_path, minimal_cli_args):
    """TORTOISE only warns and silently disables correction. Silently producing
    uncorrected output is the wrong default for a batch pipeline."""
    from qsiprep.cli.parser import parse_args

    bogus = tmp_path / 'coeff.txt'
    bogus.write_text('not a coefficient file')
    with pytest.raises(SystemExit):
        parse_args([*minimal_cli_args, '--gradient-file', str(bogus)])


@pytest.mark.parametrize('extension', ['.grad', '.dat', '.gc', '.nii', '.nii.gz'])
def test_gradient_file_accepts_every_tortoise_extension(
    tmp_path, minimal_cli_args, extension
):
    from qsiprep.cli.parser import parse_args

    path = tmp_path / f'coeff{extension}'
    path.write_bytes(b'\x00')
    parse_args([*minimal_cli_args, '--gradient-file', str(path)])
    assert config.workflow.gradient_file is not None


def test_ignore_gradients_with_a_file_warns(tmp_path, minimal_cli_args, caplog):
    from qsiprep.cli.parser import parse_args

    coeff = write_siemens_grad(tmp_path / 'coeff.grad')
    parse_args([*minimal_cli_args, '--gradient-file', str(coeff), '--ignore', 'gradients'])
    assert 'unused' in caplog.text.lower()
```

Add to that file's imports if not already present:

```python
from qsiprep.tests.gradient_fixtures import write_siemens_grad
```

If `test_cli.py` has no `minimal_cli_args` fixture, add one that supplies the
smallest valid positional set for `parse_args` (bids_dir, output_dir,
`participant`, plus `--skip-bids-validation`), matching how the existing tests in
that file construct arguments.

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_cli.py -k gradient -v`
Expected: FAIL — `unrecognized arguments: --gradient-file`

- [ ] **Step 3: Add the flags**

In `qsiprep/cli/parser.py`, extend the `--ignore` choices at line 437:

```python
        choices=['fieldmaps', 't2w', 'phase', 'sdc', 'gradients'],
```

and extend its help text with:

```
            '"gradients" disables gradient nonlinearity correction entirely, '
            'including the voxelwise gradient deviation map.'
```

Immediately after the `--ignore` block, add:

```python
    g_conf.add_argument(
        '--force',
        required=False,
        action='store',
        nargs='+',
        default=[],
        choices=['gradients'],
        help=(
            'Force selected corrections on, overriding what the input metadata '
            'implies (a space delimited list). "gradients" applies the full 3D '
            'gradient nonlinearity correction to every DWI run regardless of the '
            'ImageType field, for data whose DIS2D/DIS3D tags are absent or '
            'untrustworthy. Requires --gradient-file.'
        ),
    )
    g_conf.add_argument(
        '--gradient-file',
        required=False,
        action='store',
        type=IsFile,
        help=(
            'Path to a gradient nonlinearity information file, matching '
            "TORTOISE's --grad_nonlin: a scanner coefficient file (.grad for "
            'Siemens, .dat for GE, .gc for the TORTOISE binary format) or an ITK '
            'displacement field (.nii/.nii.gz). Applies to every DWI run in the '
            'dataset. Whether the spatial correction is applied to a given run, '
            'and in which dimensions, is decided from that run\'s ImageType '
            'field unless --force/--ignore gradients says otherwise. The '
            'voxelwise gradient deviation map is written whenever this is given.'
        ),
    )
```

- [ ] **Step 4: Add the config keys**

In `qsiprep/config.py`, in the `workflow` class, alphabetically between
`fmap_demean` and `force_syn`:

```python
    force = None
    """Corrections to force on regardless of input metadata."""
```

and between `fmap_demean`/`force_syn` and `gpu`:

```python
    gradient_file = None
    """Gradient nonlinearity coefficient file or displacement field."""
```

- [ ] **Step 5: Add the validation**

In `parse_args`, after the `--diffprep-config` block ending at line 975 and
before the `--gpu` block:

```python
    gradient_extensions = ('.grad', '.dat', '.gc', '.nii', '.nii.gz')
    forcing_gradients = 'gradients' in opts.force
    ignoring_gradients = 'gradients' in opts.ignore

    if forcing_gradients and ignoring_gradients:
        parser.error('"--force gradients" and "--ignore gradients" are contradictory.')

    if forcing_gradients and not opts.gradient_file:
        parser.error('"--force gradients" requires --gradient-file.')

    if opts.gradient_file:
        if not str(opts.gradient_file).endswith(gradient_extensions):
            parser.error(
                f'--gradient-file must be one of {", ".join(gradient_extensions)}: '
                f'<{opts.gradient_file}>. TORTOISE silently disables gradient '
                'correction for unrecognized extensions, so this is rejected here.'
            )
        if ignoring_gradients:
            config.loggers.cli.warning(
                '--gradient-file is unused because "--ignore gradients" was given.'
            )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_cli.py -k gradient -v`
Expected: PASS (10 tests)

- [ ] **Step 7: Verify no regression in the rest of the CLI**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_cli.py qsiprep/tests/test_cli_run.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/config.py qsiprep/tests/test_cli.py
git commit -m "feat: add --gradient-file and --force gradients

Unknown extensions are a hard error: TORTOISE only warns and silently
disables correction, which is the wrong default for a batch pipeline."
```

---

## Task 8: The gradwarp slot in ComposeTransforms

TORTOISE's composite order (`FINALDATA.cxx:930-976`) is motion/eddy, then
gradwarp, then EPI/SDC. `transform_order` at `gradients.py:445` is written
native-to-target and reversed for ANTs at line 478, so `gradwarp` goes
immediately after `hmc`.

**Files:**
- Modify: `qsiprep/interfaces/gradients.py:342-372,445-451,497-516`
- Test: `qsiprep/tests/test_interfaces_gradients.py`

**Interfaces:**
- Produces: `ComposeTransforms(gradwarp=<list of paths>)`. Like `fieldwarps`, a
  single-element list is broadcast to every DWI.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_gradients.py`:

```python
def test_compose_transforms_places_gradwarp_between_hmc_and_sdc():
    """TORTOISE composes motion/eddy, then gradwarp, then SDC.

    transform_order is native-to-target and reversed for ANTs, so gradwarp must
    sit immediately after hmc in the list.
    """
    from qsiprep.interfaces.gradients import ComposeTransforms

    order = ComposeTransforms._transform_order_names()
    assert order.index('gradwarp') == order.index('hmc') + 1
    assert order.index('gradwarp') < order.index('fieldwarp')


def test_compose_transforms_stage_names_match_the_runtime_lookup():
    """Every stage name must have an entry in _run_interface's by_name dict.

    A stage present in _TRANSFORM_STAGES but missing from that dict raises
    KeyError at runtime, long after the graph is built.
    """
    import inspect

    from qsiprep.interfaces.gradients import ComposeTransforms

    source = inspect.getsource(ComposeTransforms._run_interface)
    for stage in ComposeTransforms._TRANSFORM_STAGES:
        assert f"'{stage}':" in source, stage


def test_compose_transforms_gradwarp_is_not_forwarded_to_apply_transforms():
    """Every custom input must be popped before ifargs reaches ApplyTransforms."""
    from qsiprep.interfaces.gradients import ComposeTransforms

    assert 'gradwarp' in ComposeTransforms._popped_keys()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradients.py -k gradwarp -v`
Expected: FAIL — `AttributeError: type object 'ComposeTransforms' has no attribute '_transform_order_names'`

- [ ] **Step 3: Add the input to the spec**

In `ComposeTransformsInputSpec` (`gradients.py:342`), after `hmc_affines`:

```python
    gradwarp = InputMultiObject(
        File(exists=True),
        mandatory=False,
        desc='gradient nonlinearity displacement field, in native DWI space',
    )
```

- [ ] **Step 4: Introduce the two class helpers and use them**

Replace the literal `transform_order` list at `gradients.py:445-451` and the
literal pop list at `gradients.py:497-516` with class-level definitions, so both
are inspectable from tests and cannot drift from `_run_interface`.

Add these as class attributes on `ComposeTransforms`, above `_run_interface`:

```python
    #: Transform stages, native-to-target. Reversed for ANTs before use.
    #: gradwarp sits between hmc and fieldwarp, matching TORTOISE's
    #: FINALDATA.cxx:930-976 composite order.
    _TRANSFORM_STAGES = (
        'hmc',
        'gradwarp',
        'fieldwarp',
        'to b=0 affine',
        'to b=0 warp',
        'b=0 to T1w',
    )

    #: Inputs consumed here that must not be forwarded to ApplyTransforms.
    _POPPED_KEYS = (
        'environ',
        'ignore_exception',
        'print_out_composite_warp_file',
        'terminal_output',
        'output_image',
        'input_image',
        'transforms',
        'dwi_files',
        'original_b0_indices',
        'hmc_affines',
        'gradwarp',
        'b0_to_intramodal_template_transforms',
        'intramodal_template_to_t1_affine',
        'intramodal_template_to_t1_warp',
        'fieldwarps',
        'hmcsdc_dwi_ref_to_t1w_affine',
        'interpolation',
        't1_2_mni_forward_transform',
        'copy_dtype',
    )

    @classmethod
    def _transform_order_names(cls):
        return list(cls._TRANSFORM_STAGES)

    @classmethod
    def _popped_keys(cls):
        return list(cls._POPPED_KEYS)
```

In `_run_interface`, add gradwarp broadcasting beside the existing `fieldwarps`
handling (which begins at `gradients.py:405`):

```python
        gradwarp = self.inputs.gradwarp
        if isdefined(gradwarp) and len(gradwarp) == 1:
            LOGGER.info('using a single gradwarp field for all DWI files')
            gradwarp = gradwarp * num_dwis
```

Replace the `transform_order` literal with a lookup keyed by
`_TRANSFORM_STAGES`:

```python
        by_name = {
            'hmc': hmc_affines,
            'gradwarp': gradwarp,
            'fieldwarp': fieldwarps,
            'to b=0 affine': intramodal_affine,
            'to b=0 warp': intramodal_warp,
            'b=0 to T1w': coreg_to_t1,
        }
        transform_order = [(by_name[name], name) for name in self._TRANSFORM_STAGES]
```

Replace the `for key in [...]` literal at line 497 with:

```python
        for key in self._POPPED_KEYS:
```

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradients.py -v`
Expected: PASS

- [ ] **Step 6: Verify existing transform composition still works**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_intramodal_transforms.py qsiprep/tests/test_interfaces_gradients.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add qsiprep/interfaces/gradients.py qsiprep/tests/test_interfaces_gradients.py
git commit -m "feat: add gradwarp slot to ComposeTransforms

Between hmc and fieldwarp, matching TORTOISE's FINALDATA composite order.
The stage list and the popped-key list become class attributes so tests
can assert them without duplicating the literals."
```

---

## Task 9: Thread the field through resampling and base

**Files:**
- Modify: `qsiprep/workflows/dwi/resampling.py:140-200`
- Modify: `qsiprep/workflows/dwi/base.py`
- Test: `qsiprep/tests/test_workflows_gradwarp.py`

**Interfaces:**
- Consumes: `init_gradwarp_wf` (Task 6), `ComposeTransforms.gradwarp` (Task 8).
- Produces: `init_dwi_trans_wf` gains `inputnode.gradwarp_field`, connected to
  `compose_transforms.gradwarp` through a `_listify` (already defined at
  `resampling.py:51`).

When `plan.warp_dim is None`, `base.py` must **not** connect the field into
`dwi_trans_wf`, while still building the workflow for grad_dev.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_gradwarp.py`:

```python
def _trans_wf_gradwarp_sources(wf):
    """Names of nodes feeding compose_transforms.gradwarp, if any."""
    compose = wf.get_node('compose_transforms')
    return [
        edge[0].name
        for edge in wf._graph.in_edges(compose)
        if any(dest == 'gradwarp' for _, dest in wf._graph.get_edge_data(*edge)['connect'])
    ]


def test_dwi_trans_wf_exposes_a_gradwarp_field_input():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(name='trans_wf', use_compression=False)
    assert 'gradwarp_field' in wf.get_node('inputnode').inputs.trait_get()


def test_dwi_trans_wf_connects_gradwarp_to_compose_transforms():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    wf = init_dwi_trans_wf(name='trans_wf', use_compression=False)
    assert _trans_wf_gradwarp_sources(wf)
```

Match the actual `init_dwi_trans_wf` signature at `resampling.py:32-40` when
writing these calls; pass whatever positional or keyword arguments the existing
construction tests in `test_workflows_native.py` use.

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -k trans_wf -v`
Expected: FAIL — `gradwarp_field` not in the inputnode traits

- [ ] **Step 3: Add the input and connection in resampling.py**

Add `'gradwarp_field'` to the `inputnode` field list at `resampling.py:140-166`,
and add to the workflow's connect block:

```python
        (inputnode, compose_transforms, [(('gradwarp_field', _listify), 'gradwarp')]),
```

- [ ] **Step 4: Wire base.py**

In `qsiprep/workflows/dwi/base.py`, add the import:

```python
from .gradwarp import init_gradwarp_wf
```

After the hmc workflow is selected (the `if/elif` chain at `base.py:244-274`),
build the gradwarp workflow once per unit and feed it the pre-HMC reference:

```python
    gradwarp_wf = init_gradwarp_wf(unit)
    if gradwarp_wf is not None:
        workflow.connect([
            (pre_hmc_wf, gradwarp_wf, [('outputnode.dwi_file', 'inputnode.ref_image')]),
        ])  # fmt:skip
        # A DIS3D unit still builds a field, because grad_dev needs one, but no
        # spatial correction is applied to the images.
        if gradwarp_wf.plan.warp_dim is not None:
            workflow.connect([
                (gradwarp_wf, dwi_finalize_wf, [
                    ('outputnode.gradwarp_field', 'inputnode.gradwarp_field'),
                ]),
            ])  # fmt:skip
```

Use the node names actually present in `base.py` for the pre-HMC and finalize
workflows; the connection targets are `inputnode.ref_image` on `gradwarp_wf` and
`inputnode.gradwarp_field` on the finalize workflow.

- [ ] **Step 5: Add the pass-through in finalize.py**

Add `'gradwarp_field'` to the `inputnode` field list in `init_dwi_finalize_wf`
(`finalize.py:220` region) and connect it to `dwi_trans_wf`:

```python
        (inputnode, dwi_trans_wf, [('gradwarp_field', 'inputnode.gradwarp_field')]),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -v`
Expected: PASS

- [ ] **Step 7: Verify the default path is untouched**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_interfaces_diffprep.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add qsiprep/workflows/dwi/resampling.py qsiprep/workflows/dwi/base.py qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_workflows_gradwarp.py
git commit -m "feat: thread the gradwarp field into the composed chain

A DIS3D unit still builds a field, because grad_dev needs one, but the
field is not connected into the resampling chain."
```

---

## Task 10: Gradwarp-correct SDC estimation inputs

TORTOISE resamples the b0/FA images through the gradwarp field before estimating
the susceptibility field (`DRBUDDI.cxx:133-183`, `EPIREG.cxx:68-88`). QSIPrep
backends differ in where SDC lands, so the rule is conditional:

> Gradwarp-correct the SDC estimation inputs exactly when that SDC field is
> applied downstream of gradwarp in the composed chain.

- **eddy + TOPUP**: leave raw. Eddy applies the TOPUP field to raw data via
  `--field` and resamples once itself (`fsl.py:255-273`), so the field is baked
  in *upstream* of `ComposeTransforms`. Estimating it on raw b0s is the
  internally consistent choice.
- **eddy + DRBUDDI, GRE, SyN; TORTOISE; SHORELine**: gradwarp-correct, because
  those SDC warps stay in `to_dwi_ref_warps` and are applied downstream.

**Files:**
- Modify: `qsiprep/workflows/dwi/fsl.py:420-495`
- Modify: `qsiprep/workflows/dwi/diffprep.py`, `qsiprep/workflows/dwi/hmc_sdc.py`
- Test: `qsiprep/tests/test_workflows_gradwarp.py`

**Interfaces:**
- Consumes: `outputnode.gradwarp_field` from Task 6.
- Produces: each HMC workflow gains `inputnode.gradwarp_field`, and a
  `gradwarp_sdc_inputs` `ApplyTransforms` MapNode in the branches that use it.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_gradwarp.py`:

```python
def _rpe_unit(tmp_path, image_type=None):
    main = write_dwi_with_gradients(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = write_dwi_with_gradients(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    metadata = {'Manufacturer': 'SIEMENS'}
    if image_type is not None:
        metadata['ImageType'] = image_type
    from qsiprep.grouping.models import CorrectionMethod

    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
        metadata=metadata,
    )


def _cfg_for_fsl(tmp_path, pepolar_method):
    config.workflow.gradient_file = str(write_siemens_grad(tmp_path / 'coeff.grad'))
    config.workflow.hmc_model = 'eddy'
    config.workflow.pepolar_method = pepolar_method
    config.workflow.b0_threshold = 100
    config.workflow.eddy_config = None
    config.workflow.denoise_method = 'dwidenoise'
    config.execution.sloppy = False


def test_topup_branch_does_not_gradwarp_sdc_inputs(tmp_path):
    """Eddy applies the TOPUP field to raw data, so the field must be
    estimated on raw data too."""
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    _cfg_for_fsl(tmp_path, 'TOPUP')
    wf = init_fsl_hmc_wf(_rpe_unit(tmp_path), source_file='/data/x_dwi.nii.gz', t2w_sdc=False)
    assert wf.get_node('topup') is not None
    assert wf.get_node('gradwarp_sdc_inputs') is None


def test_drbuddi_branch_gradwarps_sdc_inputs(tmp_path):
    """DRBUDDI's warp is applied downstream of gradwarp, so its inputs must be
    corrected first -- matching DRBUDDI::Step0_CreateImages."""
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = init_fsl_hmc_wf(_rpe_unit(tmp_path), source_file='/data/x_dwi.nii.gz', t2w_sdc=False)
    assert wf.get_node('gradwarp_sdc_inputs') is not None


def test_no_gradwarp_node_without_a_coefficient_file(tmp_path):
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    config.workflow.gradient_file = None
    wf = init_fsl_hmc_wf(_rpe_unit(tmp_path), source_file='/data/x_dwi.nii.gz', t2w_sdc=False)
    assert wf.get_node('gradwarp_sdc_inputs') is None


def test_dis3d_does_not_gradwarp_sdc_inputs(tmp_path):
    """No spatial correction means nothing to apply before SDC estimation."""
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    _cfg_for_fsl(tmp_path, 'DRBUDDI')
    wf = init_fsl_hmc_wf(
        _rpe_unit(tmp_path, ['ORIGINAL', 'DIS3D']),
        source_file='/data/x_dwi.nii.gz',
        t2w_sdc=False,
    )
    assert wf.get_node('gradwarp_sdc_inputs') is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -k sdc_inputs -v`
Expected: FAIL — `test_drbuddi_branch_gradwarps_sdc_inputs` fails; node is None

- [ ] **Step 3: Implement in fsl.py**

Add near the top of `init_fsl_hmc_wf`, after the unit's SDC branch is known:

```python
    from .gradwarp import resolve_gradwarp_plan

    gradwarp_plan = resolve_gradwarp_plan(unit)
    # eddy applies a TOPUP field to raw data and resamples once itself, so that
    # field is baked in upstream of ComposeTransforms. Estimating it on
    # gradwarp-corrected b0s would apply it in a space it was not measured in.
    gradwarp_before_sdc = (
        gradwarp_plan is not None
        and gradwarp_plan.warp_dim is not None
        and not run_topup
    )
```

Place this after `run_topup` is computed (`fsl.py:311`). Add
`'gradwarp_field'` to the workflow's `inputnode` field list, and inside the
DRBUDDI / GRE / SyN branches, insert before the SDC node:

```python
    if gradwarp_before_sdc:
        gradwarp_sdc_inputs = pe.MapNode(
            ants.ApplyTransforms(
                dimension=3,
                interpolation='LanczosWindowedSinc',
            ),
            iterfield=['input_image'],
            name='gradwarp_sdc_inputs',
        )
        workflow.connect([
            (inputnode, gradwarp_sdc_inputs, [
                (('gradwarp_field', _listify), 'transforms'),
            ]),
        ])  # fmt:skip
```

Then route the b0 images that currently feed the SDC node through
`gradwarp_sdc_inputs` instead. Use the existing `_listify` helper if `fsl.py`
has one; otherwise import it from `.resampling`.

- [ ] **Step 4: Mirror in diffprep.py and hmc_sdc.py**

Apply the same pattern in `init_diffprep_hmc_wf` (`diffprep.py`) and
`init_qsiprep_hmcsdc_wf` (`hmc_sdc.py`). Neither runs TOPUP, so
`gradwarp_before_sdc` is simply:

```python
    gradwarp_plan = resolve_gradwarp_plan(unit)
    gradwarp_before_sdc = gradwarp_plan is not None and gradwarp_plan.warp_dim is not None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -v`
Expected: PASS

- [ ] **Step 6: Verify no regression**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_interfaces_diffprep.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add qsiprep/workflows/dwi/fsl.py qsiprep/workflows/dwi/diffprep.py qsiprep/workflows/dwi/hmc_sdc.py qsiprep/tests/test_workflows_gradwarp.py
git commit -m "feat: gradwarp SDC estimation inputs where the warp applies downstream

eddy+TOPUP is carved out: eddy bakes that field into raw data upstream of
ComposeTransforms, so estimating it on corrected b0s would misplace it."
```

---

## Task 11: The grad_dev derivative

**Files:**
- Modify: `qsiprep/data/io_spec.json`
- Modify: `qsiprep/workflows/dwi/finalize.py`, `qsiprep/workflows/dwi/derivatives.py`
- Test: `qsiprep/tests/test_workflows_gradwarp.py`

**Interfaces:**
- Consumes: `CreateGradientNonlinearityBMatrix` (Task 5), `GradwarpPlan` (Task 2).
- Produces: `sub-X_..._space-ACPC_graddev.nii.gz` plus its JSON sidecar.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_gradwarp.py`:

```python
def test_io_spec_has_a_graddev_pattern():
    """grad_dev is neither a spatial transform nor a tissue map: it needs its
    own suffix rather than xfm or dwimap."""
    import json

    from qsiprep.data import load as load_data

    with open(load_data('io_spec.json')) as handle:
        spec = json.load(handle)

    assert any('graddev' in pattern for pattern in spec['default_path_patterns'])


def test_graddev_filename_renders_with_space_entity(tmp_path):
    from qsiprep.interfaces.bids import DerivativesDataSink

    payload = tmp_path / 'graddev.nii.gz'
    payload.write_bytes(b'\x00')
    sink = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file='/data/sub-01/dwi/sub-01_dwi.nii.gz',
        space='ACPC',
        suffix='graddev',
        extension='.nii.gz',
        in_file=str(payload),
    ).run()

    out = sink.outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert out.endswith('sub-01_space-ACPC_graddev.nii.gz')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -k graddev -v`
Expected: FAIL — no `graddev` pattern in `io_spec.json`

- [ ] **Step 3: Add the io_spec pattern**

In `qsiprep/data/io_spec.json`, add to `default_path_patterns`, immediately after
the existing `bvecs` pattern:

```
sub-{subject}[/ses-{session}]/{datatype<dwi>|dwi}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_rec-{reconstruction}][_dir-{direction}][_run-{run}][_part-{part}][_chunk-{chunk}][_space-{space}][_desc-{desc}]_{suffix<graddev>|graddev}.{extension<nii|nii.gz|json>|nii.gz}
```

- [ ] **Step 4: Add the grad_dev node and sink**

In `qsiprep/workflows/dwi/finalize.py`, import:

```python
from ...interfaces.gradunwarp import CreateGradientNonlinearityBMatrix
```

and, guarded on the unit having a plan, add after the final b0 reference is
produced:

```python
    if gradwarp_plan is not None:
        grad_dev = pe.Node(
            CreateGradientNonlinearityBMatrix(
                nonlinearity=gradwarp_plan.coeff_file,
                is_ge=gradwarp_plan.is_ge,
            ),
            name='grad_dev',
        )
        ds_grad_dev = pe.Node(
            DerivativesDataSink(
                source_file=source_file,
                base_directory=output_dir,
                space='ACPC',
                suffix='graddev',
                extension='.nii.gz',
                compress=True,
                meta_dict={
                    'Description': (
                        'Voxelwise gradient deviation tensor (row-major 3x3 L '
                        'matrix per voxel). The effective diffusion gradient at '
                        'a voxel is L @ g; because L carries scaling and shear, '
                        'both the b-vector and the b-value deviate per voxel.'
                    ),
                    'GradientCoefficientFile': os.path.basename(gradwarp_plan.coeff_file),
                    'GradientWarpDimensions': gradwarp_plan.warp_dim or 'none',
                    'GradientCoefficientManufacturer': 'GE' if gradwarp_plan.is_ge else 'non-GE',
                    'GradientCorrectionBasis': gradwarp_plan.basis,
                },
            ),
            name='ds_grad_dev',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (dwi_trans_wf, grad_dev, [('outputnode.dwi_ref', 'final_image')]),
            (inputnode, grad_dev, [('dwi_file', 'initial_image')]),
            (grad_dev, ds_grad_dev, [('grad_dev', 'in_file')]),
        ])  # fmt:skip
```

Resolve `gradwarp_plan` at the top of `init_dwi_finalize_wf` with
`resolve_gradwarp_plan(unit)`, and use the actual node/field names present in
that function for the final b0 reference and the raw DWI.

Add `GradientWarpDimensions` to the main `desc-preproc_dwi.json` through the
existing `DerivativesSidecar` node in the same function.

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_workflows_gradwarp.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add qsiprep/data/io_spec.json qsiprep/workflows/dwi/finalize.py qsiprep/workflows/dwi/derivatives.py qsiprep/tests/test_workflows_gradwarp.py
git commit -m "feat: write the grad_dev derivative

New graddev suffix: grad_dev maps gradient vectors, not coordinates, so
it is not an xfm, and it describes the scanner rather than the tissue, so
it is not a dwimap."
```

---

## Task 12: Report line and boilerplate

**Files:**
- Modify: `qsiprep/interfaces/reports.py:56-70,203-258`
- Modify: `qsiprep/workflows/dwi/base.py`
- Test: `qsiprep/tests/test_reports.py`

**Interfaces:**
- Consumes: `GradwarpPlan` (Task 2).
- Produces: `DiffusionSummary(gradient_correction: str)`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_reports.py`:

```python
@pytest.mark.parametrize(
    ('warp_dim', 'basis', 'expected'),
    [
        ('3D', 'metadata', '3D (from ImageType)'),
        ('1D', 'metadata', 'through-plane only (ImageType: DIS2D)'),
        (None, 'metadata', 'b-matrix only (ImageType: DIS3D)'),
        ('3D', 'forced', 'forced 3D'),
    ],
)
def test_describe_gradient_correction(warp_dim, basis, expected):
    from qsiprep.workflows.dwi.gradwarp import GradwarpPlan, describe_gradient_correction

    plan = GradwarpPlan(coeff_file='/opt/c.grad', warp_dim=warp_dim, is_ge=False, basis=basis)
    assert describe_gradient_correction(plan) == expected


def test_describe_gradient_correction_without_a_plan():
    from qsiprep.workflows.dwi.gradwarp import describe_gradient_correction

    assert describe_gradient_correction(None) == 'none'


def test_diffusion_summary_renders_gradient_correction():
    from qsiprep.interfaces.reports import DiffusionSummary

    summary = DiffusionSummary(
        distortion_correction='TOPUP',
        pe_direction='j',
        hmc_transform='Affine',
        hmc_model='eddy',
        b0_to_anat_transform='Rigid',
        denoise_method='dwidenoise',
        dwi_denoise_window=5,
        gradient_correction='through-plane only (ImageType: DIS2D)',
    )
    assert 'through-plane only' in summary._generate_segment()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_reports.py -k gradient -v`
Expected: FAIL — `ImportError: cannot import name 'describe_gradient_correction'`

- [ ] **Step 3: Add the describer**

Append to `qsiprep/workflows/dwi/gradwarp.py`:

```python
#: Report phrasing for each resolved state.
_REPORT_TEXT = {
    '3D': '3D (from ImageType)',
    '1D': 'through-plane only (ImageType: DIS2D)',
    None: 'b-matrix only (ImageType: DIS3D)',
}


def describe_gradient_correction(plan):
    """One-line description of the resolved plan, for the HTML report."""
    if plan is None:
        return 'none'
    if plan.basis == 'forced':
        return 'forced 3D'
    return _REPORT_TEXT[plan.warp_dim]
```

- [ ] **Step 4: Add the report field**

In `qsiprep/interfaces/reports.py`, add to `DiffusionSummaryInputSpec`:

```python
    gradient_correction = traits.Str(
        'none', usedefault=True, desc='Gradient nonlinearity correction applied'
    )
```

Add to the template near line 64:

```
\t\t\t<li>Gradient correction: {gradient_correction}</li>
```

and to the `_generate_segment` format call:

```python
            gradient_correction=self.inputs.gradient_correction,
```

- [ ] **Step 5: Wire it in base.py**

Where the `summary` node is configured (`base.py:419` region), add:

```python
        gradient_correction=describe_gradient_correction(
            gradwarp_wf.plan if gradwarp_wf is not None else None
        ),
```

with the import:

```python
from .gradwarp import describe_gradient_correction, init_gradwarp_wf
```

- [ ] **Step 6: Run test to verify it passes**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_reports.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add qsiprep/interfaces/reports.py qsiprep/workflows/dwi/gradwarp.py qsiprep/workflows/dwi/base.py qsiprep/tests/test_reports.py
git commit -m "feat: report the resolved gradient correction"
```

---

## Task 13: Documentation and full-suite verification

**Files:**
- Modify: `docs/usage.rst`
- Modify: `docs/preprocessing.rst` (or the equivalent methods page)

- [ ] **Step 1: Document the flags in docs/usage.rst**

Add to the workflow configuration section, matching the surrounding style:

```rst
Gradient nonlinearity correction
--------------------------------

Gradient coils deviate from their nominal linear field, displacing voxels and
altering the diffusion encoding. Pass a scanner coefficient file with
``--gradient-file`` to correct both::

    --gradient-file /path/to/coeff.grad

Accepted formats are ``.grad`` (Siemens), ``.dat`` (GE), ``.gc`` (TORTOISE
binary), and ``.nii``/``.nii.gz`` (an ITK displacement field). The file applies
to every DWI run in the dataset; process multi-site data one site at a time.

Whether the *spatial* correction is applied to a given run is decided from that
run's ``ImageType`` field:

===================  ======================================================
``ImageType`` tag    Behavior
===================  ======================================================
(no ``DIS`` tag)     Full 3D gradwarp correction
``DIS2D``            Through-plane residual only; the scanner already
                     corrected in-plane distortion
``DIS3D``            No spatial correction; the scanner corrected it
===================  ======================================================

In every case a voxelwise gradient deviation map is written as
``*_space-ACPC_graddev.nii.gz``. No scanner can correct the diffusion encoding,
because the bval/bvec table holds one value per volume and cannot express a
spatially varying encoding, so this map is produced even for ``DIS3D`` data.

Use ``--force gradients`` to apply the full 3D correction regardless of
``ImageType``, for data whose tags are absent or untrustworthy. Use
``--ignore gradients`` to disable gradient correction entirely, including the
deviation map.
```

- [ ] **Step 2: Run the full test suite**

Run: `micromamba run -n lincapps pytest qsiprep/tests -q -x`
Expected: PASS, with the TORTOISE-binary tests SKIPping locally

- [ ] **Step 3: Run style checks**

Run: `micromamba run -n lincapps ruff check --diff && micromamba run -n lincapps ruff format --diff`
Expected: no diff

- [ ] **Step 4: Confirm which tests skipped**

Run: `micromamba run -n lincapps pytest qsiprep/tests/test_interfaces_gradunwarp.py -v -rs`
Expected: the two `_require`-guarded tests report SKIP with the missing binary
named. Confirm they are the *only* skips in that file — a skip anywhere else
means a pure-Python test is accidentally guarded.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "docs: document gradient nonlinearity correction"
```

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: TORTOISE findings → Tasks
1, 4, 5 (all three hazards get dedicated tests); decisions table → Tasks 2, 7;
architecture → Tasks 6, 8, 9; SDC placement rule → Task 10; grad_dev fidelity →
Task 5; CLI and config → Task 7; plan resolution → Task 2; interfaces → Tasks
3–5; wiring table → Tasks 8–10; outputs → Task 11; report and boilerplate →
Tasks 6, 12; testing → distributed across all tasks; docs → Task 13.

**Two deviations from the spec, both improvements found while planning:**

1. The spec proposed a container-only integration marker. The repo already has a
   better convention (`_require()` + `shutil.which` in
   `test_interfaces_diffprep.py`), where CI's `unit_tests` job runs inside
   `pennlinc/qsiprep:test` and executes those tests for real. Tasks 4 and 5 use
   it, so the simulated `.grad` fixtures are exercised against the actual
   binaries in CI rather than being skipped everywhere.
2. Task 8 lifts `transform_order` and the popped-key list out of `_run_interface`
   into class attributes. The spec cited both as line-number edits; making them
   inspectable lets the tests assert ordering without duplicating the literals,
   which is what keeps the gradwarp slot from silently drifting.

**Open item for the executor.** Tasks 9, 10 and 11 give connection targets but
tell the implementer to use the node names actually present in `base.py`,
`fsl.py` and `finalize.py`. Those functions are long and their internal node
names were not all read during planning. Verify each against the file before
connecting; a wrong name fails loudly at graph construction, so this is
self-checking.
