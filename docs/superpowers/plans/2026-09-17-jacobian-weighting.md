# Jacobian Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Jacobian intensity modulation for gradient-nonlinearity, susceptibility and eddy-current distortion corrections on every QSIPrep backend, so preprocessed DWI intensities are comparable across `--hmc-method` and `--sdc-method` choices.

**Architecture:** One new interface (`ComposeJacobianWeights`) computes the analytic Jacobian determinant of the composed native-space distortion warps (gradwarp ∘ SDC), keyed over unique warp combinations so cost is 1–2 ANTs calls per run. The existing `ApplyScalingImages` node — renamed `ApplyJacobianWeights` — transports those maps to the output grid through the intramodal/coregistration transforms it already applies, and multiplies them into the resampled DWIs. Because DRBUDDI, GRE, SyN and T2Wreg warps all already arrive as `fieldwarps`, they need no per-backend code. The single-interpolation contract is unchanged.

**Tech Stack:** Python 3.11, Nipype workflows/interfaces, ANTs (`antsApplyTransforms`, `CreateJacobianDeterminantImage`), nibabel, nilearn, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-jacobian-weighting-design.md` — read it before starting. It carries the derivations (correction direction, HMC coordinate safety), the per-backend contract table, and the coordinate-domain table that several tasks implement guards for.

## Global Constraints

- **Environment:** every command runs through micromamba: `micromamba run -n linc311 <command>`. Never `conda`/`venv`/`pip venv`.
- **PYTHONPATH:** `linc311` imports a stale editable QSIPlan, so ~18 unit tests fail at baseline. Run tests with QSIPlan 0.4.0 on `PYTHONPATH`. Establish the baseline failure set in Task 0 and compare against it; never claim "tests pass" against an unmeasured baseline.
- **Branch:** `jacobian-weighting` (already checked out, already branched off main).
- **Line-endings:** the working tree is CRLF, so `git status` shows every file modified. **Stage by explicit filename. Never `git add -A` or `git add .`**
- **Do not edit `docs/changes.md`** — it is generated from PR titles at release. Change descriptions go in the PR body.
- **Do not stage `docs/superpowers/`** — specs and plans stay untracked.
- **Commit trailer**, verbatim, on every commit: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **Correction direction:** multiply by `|det ∇φ|` of the ANTs pull-back composite. Never divide. The conservation oracle (Task 13) is the arbiter.
- **Transform ordering:** for `-t T_1 ... -t T_n`, the composite point map is `φ = T_n ∘ … ∘ T_1` (first-listed is innermost/applied-first-to-the-point). Task 1 pins this with an executable test and **must complete before Task 4**.
- **ANTs-dependent tests** are guarded with `shutil.which` and skip locally; they run in CircleCI's `unit_tests` job, which uses the `pennlinc/qsiprep:test` image. Follow the docstring pattern in `qsiprep/tests/test_interfaces_gradunwarp.py`.
- **Derivative naming** is asserted by rendering `build_path` against `io_spec.json`, never by inspecting datasink node inputs.

## File Structure

**Create:**
- `qsiprep/interfaces/jacobian.py` — `ComposeJacobianWeights`, `OkanQuadraticJacobian`, and their pure helpers. New module rather than growing `gradients.py` (1087 lines) or `fmap.py` (1430 lines).
- `qsiprep/utils/eddy_config.py` — `effective_eddy_resampling_method()`, the single parse shared by `fsl.py` and `init_dwi_trans_wf`.
- `qsiprep/tests/test_ants_transform_order.py` — the ordering-convention gate.
- `qsiprep/tests/test_interfaces_jacobian.py` — interface unit tests.
- `qsiprep/tests/test_workflows_jacobian.py` — wiring tests per backend × fieldmap branch.
- `qsiprep/tests/test_jacobian_conservation.py` — the conservation oracle.

**Modify:**
- `qsiprep/interfaces/fmap.py:1017-1135` — rename `ApplyScalingImages` → `ApplyJacobianWeights`, add `resampled_weight_images` output.
- `qsiprep/workflows/dwi/resampling.py` — add `compose_jacobian` node, new inputnode fields.
- `qsiprep/workflows/fieldmap/unwarp.py:177,236` — delete dead `jac_dfm` node and `out_jacobian` output.
- `qsiprep/interfaces/tortoise.py:499-541` — delete ratio images (Task 10, gated).
- `qsiprep/workflows/dwi/{fsl,diffprep,hmc_sdc,base,finalize}.py` — retire `sdc_scaling_images`, add `ec_jacobian_images`.
- `qsiprep/workflows/dwi/derivatives.py` — the new sink.
- `qsiprep/cli/parser.py`, `qsiprep/config.py` — the flag.
- `docs/preprocessing.rst`, `docs/usage.rst`, `docs/api.rst`, `qsiprep/data/boilerplate.bib`.
- `qsiprep/tests/data/*_outputs.txt` — expected-file lists.

---

### Task 0: Establish the test baseline

**Files:**
- Create: `/tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep/eb63e21c-169c-4711-ba87-06adc495619e/scratchpad/baseline.txt` (scratchpad, not the repo)

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded list of pre-existing test failures that every later task compares against.

- [ ] **Step 1: Record the baseline failure set**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
SP=/tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep/eb63e21c-169c-4711-ba87-06adc495619e/scratchpad
micromamba run -n linc311 python -m pytest qsiprep/tests -m "not integration" -q 2>&1 | tail -40 | tee $SP/baseline.txt
```

- [ ] **Step 2: Confirm the QSIPlan situation**

If the tail shows import errors or ~18 failures mentioning QSIPlan, locate a 0.4.0 checkout and re-run with it on `PYTHONPATH`:

```bash
micromamba run -n linc311 env PYTHONPATH=/path/to/qsiplan-0.4.0 \
  python -m pytest qsiprep/tests -m "not integration" -q 2>&1 | tail -40 | tee $SP/baseline.txt
```

Record the exact command that produced the cleanest baseline. Every later "run the tests" step uses that same command. **Do not proceed until you can state the baseline pass/fail counts.**

- [ ] **Step 3: No commit**

Nothing in the repo changed. Do not commit.

---

### Task 1: Pin the ANTs transform-ordering convention

This is the gate for the whole composition design. The spec's derivation says the first-listed transform is innermost; ANTs' and Nipype's own docs say "the last specified transform will be applied first," which describes image-warp order and reads the opposite way. Two adversarial review rounds disagreed about this. Settle it with code.

**Files:**
- Create: `qsiprep/tests/test_ants_transform_order.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a green test asserting `transforms=[A, B]` realizes `φ(x) = B(A(x))`. Task 4 depends on this result.

- [ ] **Step 1: Write the failing test**

```python
"""Pin the ANTs transform-list ordering convention.

``ComposeJacobianWeights`` composes only the gradwarp and SDC stages out of
QSIPrep's full transform chain, which is sound only if the stages it drops sit
at the ends of that chain rather than in the middle. Working out which end
requires knowing what ANTs does with a transform list -- and ANTs and Nipype
document it as "the last specified transform will be applied first", which
describes *image warp* order and is the exact opposite of *point map* order.

Two non-commuting transforms discriminate the two candidate composites, so
this test answers the question instead of arguing about it. It also fails
loudly if a future ANTs or Nipype release changes the convention.

Guarded with ``shutil.which``: skips locally, runs in CircleCI's ``unit_tests``
job, which uses the ``pennlinc/qsiprep:test`` image.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces import ants

# Deliberately non-commuting, and discriminating in a way that survives the
# RAS/LPS sign flip between NIfTI and ITK: a uniform 2x scaling and a 10mm
# translation along one axis.
#
#   phi = B . A  (first-listed innermost)  -> phi(0) = 2*0 + 10 = 10
#   phi = A . B  (last-listed innermost)   -> phi(0) = 2*(0 + 10) = 20
#
# The predictions differ in *magnitude* (10 vs 20), so a sign flip cannot turn
# one into the other.
_SCALE_2X = 'Parameters: 2 0 0 0 2 0 0 0 2 0 0 0'
_TRANSLATE_10 = 'Parameters: 1 0 0 0 1 0 0 0 1 10 0 0'


def _write_itk_affine(path, parameters_line):
    """Write a 3D ITK affine transform in the text format ANTs reads."""
    path.write_text(
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: MatrixOffsetTransformBase_double_3_3\n'
        f'{parameters_line}\n'
        'FixedParameters: 0 0 0\n'
    )
    return str(path)


def test_first_listed_transform_is_applied_first_to_the_point(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')

    reference = tmp_path / 'ref.nii.gz'
    nb.Nifti1Image(np.zeros((8, 8, 8), dtype='float32'), np.eye(4)).to_filename(
        str(reference)
    )
    scale = _write_itk_affine(tmp_path / 'scale.txt', _SCALE_2X)
    translate = _write_itk_affine(tmp_path / 'translate.txt', _TRANSLATE_10)

    composite = tmp_path / 'composite.nii.gz'
    xfm = ants.ApplyTransforms(
        input_image=str(reference),
        reference_image=str(reference),
        transforms=[scale, translate],
        output_image=str(composite),
        print_out_composite_warp_file=True,
        interpolation='LanczosWindowedSinc',
        dimension=3,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    xfm.run()

    # The composite warp stores phi(x) - x at each reference voxel. Voxel
    # (0, 0, 0) is world origin under the identity affine, so the stored
    # displacement is phi(0) itself.
    field = np.asanyarray(nb.load(str(composite)).dataobj)
    displacement = field.reshape(field.shape[:3] + (3,))[0, 0, 0]
    magnitude = np.abs(displacement).max()

    # Separate "the other convention" from "something else entirely" (an
    # affine collapse, a zero field, a units problem). Only 10 and 20 are
    # convention answers; anything else means the experiment itself is wrong
    # and neither party's prediction has been tested.
    assert magnitude == pytest.approx(10.0, abs=0.5) or magnitude == pytest.approx(
        20.0, abs=0.5
    ), (
        f'Composite displacement at the origin was {magnitude:.3f}mm, which is '
        'neither 10mm nor 20mm. This test has not discriminated anything -- the '
        'experiment is broken (collapsed affines, an empty field, or a units '
        'mismatch), not the convention. Investigate before reading anything '
        'into it.'
    )

    assert magnitude == pytest.approx(10.0, abs=0.5), (
        'transforms=[scale, translate] produced a displacement of '
        f'{magnitude:.3f}mm. 10mm means phi = translate . scale, i.e. the '
        'FIRST-listed transform is innermost (applied first to the point), '
        'which is what ComposeJacobianWeights assumes. 20mm means the opposite '
        'convention: STOP, and rework the composition to include the '
        'per-volume HMC affine so the gradwarp and SDC determinants are '
        'evaluated at the right coordinates. Do not work around it. See '
        '"Disambiguating \'applied first\'" in the design spec, and note that '
        'under the 20mm reading QSIPrep\'s existing resampling would be '
        'sampling native per-volume data at b=0-reference coordinates, so '
        'check that too.'
    )
```

- [ ] **Step 2: Run the test**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_ants_transform_order.py -v
```

Expected locally: `SKIPPED (antsApplyTransforms required for this test)`.

- [ ] **Step 3: Run it where ANTs exists**

```bash
docker run --rm -v /mnt/c/Users/tsalo/Documents/linc/qsiprep:/src \
  --entrypoint pytest pennlinc/qsiprep:test \
  /src/qsiprep/tests/test_ants_transform_order.py -v
```

Expected: PASS. **If it fails with a 20mm displacement, stop and report.** The spec's HMC coordinate-safety argument inverts, the composition in Task 4 must include the HMC affine per-volume, and the spec needs rewriting before any further task runs. Do not work around it.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/tests/test_ants_transform_order.py
git commit -m "test: pin the ANTs transform-list ordering convention

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Determine what DRBUDDI's `_JAC` outputs contain

> **Execution order: run this task AFTER Task 3 and BEFORE Task 4.** It needs
> `jacobian_determinant`, which Task 3 creates. It is placed here in the
> document because it is a gate, not because it runs second.
>
> **Why it must precede Task 4 and not merely Task 10:** Tasks 5 and 7 retire
> the legacy scaling-image path. If this gate ran after them and failed, the
> ratio images would already be disconnected from the renamed interface and the
> spec's option-(2) fallback would be unreachable. A failure here means the
> plan shape changes — analytic for the other backends, ratios retained for
> DRBUDDI — so **stop and re-plan** rather than carrying a conditional branch
> through eight tasks.

`DRBUDDIAggregateOutputs` already receives `blip_up_b0_corrected_jac` and `blip_down_b0_corrected_jac` (`qsiprep/interfaces/tortoise.py:400-442`, wired at `qsiprep/workflows/fieldmap/drbuddi.py:241`) and does nothing with them. If they are the Jacobian-modulated corrected b=0 images, then `_JAC / b0_corrected` is TORTOISE's own determinant and gives a real oracle for Task 10. This task finds out. It gates Task 10 only.

**Files:**
- Create: `qsiprep/tests/test_drbuddi_jac_semantics.py`

**Interfaces:**
- Consumes: `jacobian_determinant` (Task 3).
- Produces: a recorded finding (in the test docstring) plus the gate result that Tasks 4–12 proceed on.

- [ ] **Step 1: Write the investigation test**

```python
"""What DRBUDDI's ``*_b0_corrected_JAC`` images actually contain.

DRBUDDI emits both ``blip_*_b0_corrected`` and ``blip_*_b0_corrected_JAC``.
If the JAC variant is the Jacobian-modulated version of the same image, their
voxelwise ratio is TORTOISE's own determinant for that warp, which is a far
better oracle for replacing the empirical ratio images than correlating
against those ratios.

Requires a real DRBUDDI run, so this is integration-marked.
"""

import nibabel as nb
import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.drbuddi_rpe
@pytest.mark.parametrize(
    ('blip', 'warp_name'),
    [
        ('up', 'deformation_FINV.nii.gz'),
        # The down direction's warp is the composite that already carries the
        # bdown_to_bup rigid (qsiprep/interfaces/tortoise.py:508), so testing
        # it checks that the rigid composition is handled too -- the up case
        # alone would not.
        ('down', 'blip_down_composite.nii.gz'),
    ],
)
def test_jac_over_corrected_matches_analytic_determinant(working_dir, blip, warp_name):
    """``_JAC / _corrected`` should equal the determinant of the DRBUDDI warp."""
    from pathlib import Path

    from qsiprep.interfaces.jacobian import jacobian_determinant

    work = Path(working_dir)
    corrected = next(work.rglob(f'blip_{blip}_b0_corrected.nii'))
    jac = next(work.rglob(f'blip_{blip}_b0_corrected_JAC.nii'))
    warp = next(work.rglob(warp_name))

    corrected_data = np.asanyarray(nb.load(str(corrected)).dataobj)
    jac_data = np.asanyarray(nb.load(str(jac)).dataobj)
    inside = corrected_data > np.percentile(corrected_data, 60)

    tortoise_ratio = np.zeros_like(corrected_data)
    tortoise_ratio[inside] = jac_data[inside] / corrected_data[inside]

    analytic = np.asanyarray(
        nb.load(str(jacobian_determinant(str(warp), str(work / 'analytic.nii.gz')))).dataobj
    )

    # Correlation alone is not an equivalence check: a map that is globally
    # mis-scaled, or biased in a way that varies smoothly across the brain,
    # correlates strongly while being wrong everywhere. Require voxelwise
    # agreement, and report the correlation only as context.
    r = np.corrcoef(tortoise_ratio[inside], analytic[inside])[0, 1]
    relative_error = np.abs(analytic[inside] - tortoise_ratio[inside]) / tortoise_ratio[inside]
    p95 = float(np.percentile(relative_error, 95))

    assert p95 < 0.05, (
        f'95th-percentile voxelwise relative error between TORTOISE '
        f'_JAC/_corrected and the analytic determinant is {p95:.4f} (r={r:.4f}). '
        'Above 5% means _JAC is not simple Jacobian modulation, or our '
        'determinant is wrong. STOP and re-plan on the spec option-(2) path: '
        'analytic Jacobians for the other backends, DRBUDDI keeps its ratio '
        'images, and the sdc_scaling_images deletion in Task 10 does not '
        'happen. Record here what _JAC actually turned out to be.'
    )
```

- [ ] **Step 2: Run it (after Task 3 has landed)**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_drbuddi_jac_semantics.py -v -m drbuddi_rpe
```

Expected: PASS for both blip directions. **If either fails, stop and re-plan** —
do not continue to Task 4. Record the measured 95th-percentile relative error
and the correlation in the commit message either way, so the decision is
evidenced rather than remembered.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_drbuddi_jac_semantics.py
git commit -m "test: investigate DRBUDDI _JAC output semantics

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `jacobian_determinant` and the pure helpers

Split so that everything testable without ANTs is a pure function, and only the two ANTs shellouts are guarded.

**Files:**
- Create: `qsiprep/interfaces/jacobian.py`
- Test: `qsiprep/tests/test_interfaces_jacobian.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `jacobian_determinant(field_path: str, out_path: str) -> str`
  - `compose_fields(field_paths: list[str], reference: str, out_path: str) -> str`
  - `multiply_maps(paths: list[str], out_path: str) -> str`
  - `validate_field_geometry(field_path: str, reference_path: str) -> None` (raises `ValueError`)
  - `check_weight_map(map_path: str, mask_path: str) -> None` (raises `ValueError`; warns via `LOGGER`)
  - `weight_key(gradwarp: str | None, fieldwarp: str | None) -> tuple`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the Jacobian weighting helpers.

Pure-Python behaviour -- geometry validation, dedup keying, map arithmetic and
the positivity guard -- is tested unconditionally. Tests that shell out to ANTs
are guarded with ``shutil.which`` and skip when the binaries are absent. They
are not permanently offline: CircleCI's ``unit_tests`` job runs pytest inside
the ``pennlinc/qsiprep:test`` image, which ships ANTs.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.jacobian import (
    check_weight_map,
    compose_fields,
    jacobian_determinant,
    multiply_maps,
    validate_field_geometry,
    weight_key,
)
from qsiprep.tests.gradient_fixtures import write_itk_field


def _write_map(path, value, shape=(8, 8, 8), affine=None):
    affine = np.eye(4) if affine is None else affine
    data = np.full(shape, value, dtype='float32')
    nb.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def _write_linear_field(path, matrix, shape=(8, 8, 8)):
    """Write an ITK displacement field encoding phi(x) = matrix @ x."""
    coords = np.stack(
        np.meshgrid(*[np.arange(n, dtype='float32') for n in shape], indexing='ij'),
        axis=-1,
    )
    mapped = coords @ np.asarray(matrix, dtype='float32').T
    data = (mapped - coords).reshape(shape + (1, 3)).astype('float32')
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


# --- pure helpers ----------------------------------------------------------


def test_weight_key_dedups_identical_pairs():
    assert weight_key('g.nii.gz', 'f.nii.gz') == weight_key('g.nii.gz', 'f.nii.gz')


def test_weight_key_separates_different_fieldwarps():
    assert weight_key('g.nii.gz', 'up.nii.gz') != weight_key('g.nii.gz', 'down.nii.gz')


def test_weight_key_of_nothing_is_falsy():
    """No gradwarp and no fieldwarp means a unity weight, not a cache entry."""
    assert not weight_key(None, None)


def test_multiply_maps_multiplies_voxelwise(tmp_path):
    a = _write_map(tmp_path / 'a.nii.gz', 2.0)
    b = _write_map(tmp_path / 'b.nii.gz', 3.0)
    out = multiply_maps([a, b], str(tmp_path / 'out.nii.gz'))
    assert np.asanyarray(nb.load(out).dataobj)[0, 0, 0] == pytest.approx(6.0)


def test_multiply_maps_with_one_input_returns_it_unchanged(tmp_path):
    a = _write_map(tmp_path / 'a.nii.gz', 2.0)
    assert multiply_maps([a], str(tmp_path / 'out.nii.gz')) == a


def test_validate_field_geometry_accepts_matching_grid(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(8, 8, 8))
    validate_field_geometry(str(field), reference)


def test_validate_field_geometry_rejects_wrong_shape(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(6, 6, 6))
    with pytest.raises(ValueError, match='shape'):
        validate_field_geometry(str(field), reference)


def test_validate_field_geometry_rejects_wrong_affine(tmp_path):
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0, affine=np.diag([2.0, 2, 2, 1]))
    field = write_itk_field(tmp_path / 'field.nii.gz', shape=(8, 8, 8))
    with pytest.raises(ValueError, match='affine'):
        validate_field_geometry(str(field), reference)


def test_check_weight_map_rejects_nonpositive_in_mask(tmp_path):
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[4, 4, 4] = -0.5
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    with pytest.raises(ValueError, match='non-positive'):
        check_weight_map(weights, mask)


def test_check_weight_map_rejects_nonfinite(tmp_path):
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[0, 0, 0] = np.nan
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    with pytest.raises(ValueError, match='finite'):
        check_weight_map(weights, mask)


def test_check_weight_map_warns_on_far_from_unity_median(tmp_path, caplog):
    weights = _write_map(tmp_path / 'w.nii.gz', 4.0)
    mask = _write_map(tmp_path / 'm.nii.gz', 1.0)
    check_weight_map(weights, mask)
    assert 'median' in caplog.text


def test_check_weight_map_ignores_nonpositive_outside_mask(tmp_path):
    """Determinants outside the brain are not the guard's business."""
    weights = _write_map(tmp_path / 'w.nii.gz', 1.0)
    data = np.asanyarray(nb.load(weights).dataobj).copy()
    data[0, 0, 0] = -1.0
    nb.Nifti1Image(data, np.eye(4)).to_filename(weights)
    mask_data = np.zeros((8, 8, 8), dtype='uint8')
    mask_data[2:6, 2:6, 2:6] = 1
    mask = str(tmp_path / 'm.nii.gz')
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask)
    check_weight_map(weights, mask)


# --- ANTs-backed helpers ---------------------------------------------------


def test_jacobian_determinant_of_translation_is_unity(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    shape = (8, 8, 8)
    data = np.zeros(shape + (1, 3), dtype='float32')
    data[..., 0, 0] = 3.0
    field = tmp_path / 'translation.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(field))

    out = jacobian_determinant(str(field), str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=1e-3)


def test_jacobian_determinant_of_anisotropic_scaling(tmp_path):
    """phi(x) = diag(2, 1, 1) @ x has det = 2 everywhere."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    field = _write_linear_field(tmp_path / 'scale.nii.gz', np.diag([2.0, 1.0, 1.0]))

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 2.0, atol=5e-2)


def test_jacobian_determinant_of_shear_is_unity(tmp_path):
    """A shear moves voxels without changing volume, so det = 1."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    field = _write_linear_field(tmp_path / 'shear.nii.gz', shear)

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'))
    interior = np.asanyarray(nb.load(out).dataobj)[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=5e-2)


def test_jacobian_determinant_rejects_a_fold_in_mask(tmp_path):
    """A folded warp has a negative determinant and must not be absolute-valued.

    Without this case the three positive-determinant tests above all pass while
    a fold at -0.5 silently becomes a weight of 0.5.
    """
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    # phi(x) = diag(-1, 1, 1) @ x is an orientation reversal: det = -1.
    field = _write_linear_field(tmp_path / 'fold.nii.gz', np.diag([-1.0, 1.0, 1.0]))
    mask = _write_map(tmp_path / 'mask.nii.gz', 1.0)

    with pytest.raises(ValueError, match='non-positive'):
        jacobian_determinant(field, str(tmp_path / 'det.nii.gz'), mask_path=mask)


def test_jacobian_determinant_tolerates_edge_negatives_outside_mask(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    field = _write_linear_field(tmp_path / 'scale.nii.gz', np.diag([2.0, 1.0, 1.0]))
    mask_data = np.zeros((8, 8, 8), dtype='uint8')
    mask_data[3:5, 3:5, 3:5] = 1
    mask = str(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask)

    out = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'), mask_path=mask)
    assert (np.asanyarray(nb.load(out).dataobj) >= 0).all()


def test_validate_scalar_geometry_rejects_a_wrong_grid(tmp_path):
    from qsiprep.interfaces.jacobian import validate_scalar_geometry

    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    other = _write_map(tmp_path / 'other.nii.gz', 1.0, shape=(6, 6, 6))
    with pytest.raises(ValueError, match='shape'):
        validate_scalar_geometry(other, reference)


def test_compose_fields_returns_a_single_field(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')
    reference = _write_map(tmp_path / 'ref.nii.gz', 1.0)
    first = write_itk_field(tmp_path / 'a.nii.gz', amplitude=0.4)
    second = write_itk_field(tmp_path / 'b.nii.gz', amplitude=0.2)

    out = compose_fields([str(first), str(second)], reference, str(tmp_path / 'c.nii.gz'))
    composed = nb.load(out)
    assert composed.shape[:3] == (8, 8, 8)
    assert np.isfinite(np.asanyarray(composed.dataobj)).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'qsiprep.interfaces.jacobian'`.

- [ ] **Step 3: Write the implementation**

```python
"""Jacobian intensity modulation for QSIPrep's spatial distortion corrections.

The weight applied to the resampled DWIs is ``|det grad phi|`` of the composed
*native-space* distortion warps -- gradient nonlinearity, susceptibility, and
TORTOISE eddy current -- and of nothing else. See
``docs/superpowers/specs/2026-09-17-jacobian-weighting-design.md`` for why head
motion, coregistration and the intramodal/template warps are excluded, and for
the derivation showing that excluding them does not move the coordinates at
which the remaining determinants are evaluated.
"""

import os

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype import logging
from nipype.interfaces import ants

LOGGER = logging.getLogger('nipype.interface')

#: Tolerances for comparing a displacement field's affine against the
#: composition reference. Loose enough for float32 header round-trips, tight
#: enough that a different lattice or orientation fails.
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4

#: Floor for a resampled weight map. Lanczos ringing can push a positive
#: scalar map slightly negative; a weight can be small but never zero or
#: negative.
WEIGHT_FLOOR = 1e-3

#: In-mask median outside this range is a smell, not an error: a correct
#: distortion field redistributes signal without changing its total, so the
#: median determinant should sit near unity.
MEDIAN_WARN_RANGE = (0.9, 1.1)


def weight_key(gradwarp, fieldwarp):
    """Cache key for one unique (gradwarp, fieldwarp) combination.

    Falsy when neither field is present, which means a unity weight rather
    than a cache entry.
    """
    return tuple(path for path in (gradwarp, fieldwarp) if path)


def validate_field_geometry(field_path, reference_path):
    """Raise unless ``field_path`` is sampled on ``reference_path``'s lattice.

    This is a cheap structural screen. It cannot prove that a field encodes the
    intended coordinate domain, and it cannot detect an inverted field -- an
    inverse has identical headers. Direction is established behaviourally, by
    the conservation oracle and the positivity guard.
    """
    field = nb.load(field_path)
    reference = nb.load(reference_path)

    if field.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f'Displacement field {field_path} has spatial shape {field.shape[:3]}, '
            f'but the composition reference {reference_path} has '
            f'{reference.shape[:3]}. A field on a different lattice cannot be '
            'composed here; see the coordinate-domain table in the design spec.'
        )
    if not np.allclose(field.affine, reference.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
        raise ValueError(
            f'Displacement field {field_path} has an affine that does not match '
            f'the composition reference {reference_path}. QSIPrep deliberately '
            'requires exact agreement here rather than resampling: telling a '
            'world-compatible-but-differently-sampled field apart from one in '
            'the wrong coordinate domain needs case work no current input '
            'exercises.'
        )

    components = field.shape[4] if field.ndim == 5 else field.shape[-1]
    if components != 3:
        raise ValueError(
            f'Displacement field {field_path} has {components} vector components, '
            'expected 3.'
        )


def check_weight_map(map_path, mask_path):
    """Raise on a non-finite or non-positive in-mask weight; warn if off-unity.

    A folded warp produces a non-positive determinant, and silently multiplying
    DWI data by a negative number is worse than failing.
    """
    weights = np.asanyarray(nb.load(map_path).dataobj)
    mask = np.asanyarray(nb.load(mask_path).dataobj) > 0

    if not np.isfinite(weights).all():
        raise ValueError(
            f'Jacobian weight map {map_path} contains non-finite values.'
        )

    inside = weights[mask]
    if inside.size and inside.min() <= 0:
        raise ValueError(
            f'Jacobian weight map {map_path} has non-positive values inside the '
            f'brain mask (minimum {inside.min():.4f}). This means the composed '
            'warp folds. Applying it would multiply DWI signal by a negative '
            'number.'
        )

    if inside.size:
        median = float(np.median(inside))
        if not MEDIAN_WARN_RANGE[0] <= median <= MEDIAN_WARN_RANGE[1]:
            LOGGER.warning(
                'Jacobian weight map %s has an in-mask median of %.4f, outside '
                '%s. A correct distortion field redistributes signal without '
                'changing its total, so this may indicate a wrong field, an '
                'inverted field, or a wrong composition order.',
                map_path,
                median,
                MEDIAN_WARN_RANGE,
            )


def multiply_maps(paths, out_path):
    """Voxelwise product of one or more scalar maps."""
    if len(paths) == 1:
        return paths[0]
    product = nim.load_img(paths[0])
    for path in paths[1:]:
        product = nim.math_img('a*b', a=product, b=path)
    product.to_filename(out_path)
    return out_path


def compose_fields(field_paths, reference, out_path):
    """Compose displacement fields into one, on ``reference``'s grid.

    ``field_paths`` is given in QSIPrep's native-to-target chain order; it is
    reversed here for ANTs, matching ``ComposeTransforms``. The reference is a
    *native* lattice: the composite's domain is undistorted b=0-reference
    space, and materialising it on the output grid would evaluate a
    native-domain function at output coordinates.
    """
    xfm = ants.ApplyTransforms(
        # input_image is ignored because print_out_composite_warp_file is True
        input_image=reference,
        reference_image=reference,
        transforms=list(field_paths)[::-1],
        output_image=out_path,
        print_out_composite_warp_file=True,
        interpolation='LanczosWindowedSinc',
        dimension=3,
        float=True,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime
    LOGGER.info(runtime.cmdline)
    return out_path


def jacobian_determinant(field_path, out_path, mask_path=None):
    """``|det grad phi|`` of a displacement field, on the field's own grid.

    ``CreateJacobianDeterminantImage`` takes no reference image; it emits on
    the deformation field's grid. ``doLogJacobian=0`` because the weight is
    multiplicative, and ``useGeometric=0`` for the plain determinant.

    A folded warp produces a *negative* determinant, so the sign is inspected
    before any absolute value is taken -- otherwise a fold at -0.5 would become
    an innocuous-looking weight of 0.5 and no later check could recover it.
    Pass ``mask_path`` to get that check; without it the sign is only logged.
    """
    jac = ants.CreateJacobianDeterminantImage(
        imageDimension=3,
        deformationField=field_path,
        outputImage=out_path,
        doLogJacobian=0,
        useGeometric=0,
    )
    jac.terminal_output = 'allatonce'
    jac.resource_monitor = False
    runtime = jac.run().runtime
    LOGGER.info(runtime.cmdline)

    img = nb.load(out_path)
    signed = np.asanyarray(img.dataobj)

    # Inspect the SIGN before taking any absolute value. In-mask negatives mean
    # the composed warp folds, which is a hard error; outside the mask,
    # CreateJacobianDeterminantImage legitimately emits small negatives at the
    # field's edge from one-sided differences, and those are harmless.
    if mask_path is not None:
        mask = np.asanyarray(nb.load(mask_path).dataobj) > 0
        inside = signed[mask]
        if inside.size and inside.min() <= 0:
            raise ValueError(
                f'Jacobian determinant of {field_path} is non-positive inside '
                f'the brain mask (minimum {inside.min():.4f}), i.e. the composed '
                'warp folds there. Refusing to build a weight map from it.'
            )
    elif np.nanmin(signed) <= 0:
        LOGGER.warning(
            'Jacobian determinant of %s contains non-positive values and no '
            'mask was supplied to localise them. Edge negatives are expected; '
            'interior ones are a fold.',
            field_path,
        )

    nb.Nifti1Image(
        np.abs(signed).astype('float32'), img.affine, img.header
    ).to_filename(out_path)
    return out_path


def validate_scalar_geometry(image_path, reference_path):
    """Raise unless a scalar map shares ``reference_path``'s sampling grid.

    ``validate_field_geometry`` covers displacement fields. Scalar inputs --
    the eddy-current Jacobians, the brain mask, and every map that goes into a
    product -- need the same check, or a mislatticed input is multiplied
    elementwise against the wrong voxels or silently moves which voxels the
    positivity guard inspects.
    """
    image = nb.load(image_path)
    reference = nb.load(reference_path)
    if image.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f'{image_path} has spatial shape {image.shape[:3]}, but '
            f'{reference_path} has {reference.shape[:3]}.'
        )
    if not np.allclose(image.affine, reference.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
        raise ValueError(
            f'{image_path} has an affine that does not match {reference_path}.'
        )


def _abspath(path, cwd):
    return path if os.path.isabs(path) else os.path.join(cwd, path)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -v
```

Expected: the pure-helper tests PASS; the four ANTs-backed tests SKIP locally. Then run the ANTs ones in the image:

```bash
docker run --rm -v /mnt/c/Users/tsalo/Documents/linc/qsiprep:/src \
  --entrypoint pytest pennlinc/qsiprep:test \
  /src/qsiprep/tests/test_interfaces_jacobian.py -v
```

Expected: all PASS. If the scaling test gives `0.5` where `2.0` was expected, the field-writing helper has the direction inverted; fix the test helper, not `jacobian_determinant`.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/jacobian.py qsiprep/tests/test_interfaces_jacobian.py
git commit -m "feat: add Jacobian determinant helpers for distortion weighting

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The `ComposeJacobianWeights` interface

**Prerequisite: Task 1 must be green.** This task's composition order depends on its result.

**Files:**
- Modify: `qsiprep/interfaces/jacobian.py`
- Test: `qsiprep/tests/test_interfaces_jacobian.py`

**Interfaces:**
- Consumes: `jacobian_determinant`, `compose_fields`, `multiply_maps`, `validate_field_geometry`, `check_weight_map`, `weight_key` from Task 3.
- Produces: `ComposeJacobianWeights`, a Nipype `SimpleInterface` with inputs `dwi_files` (N), `b0_ref_image`, `mask`, `gradwarp_field` (0–1), `fieldwarps` (0/1/N), `ec_jacobian_images` (0/N), and output `jacobian_weight_images` (N paths with repeats, or `Undefined`).

- [ ] **Step 1: Write the failing tests**

```python
# Append to qsiprep/tests/test_interfaces_jacobian.py

from nipype.interfaces.base import isdefined

from qsiprep.interfaces.jacobian import ComposeJacobianWeights


def _dwi_volumes(tmp_path, count):
    return [_write_map(tmp_path / f'dwi{i}.nii.gz', 1.0) for i in range(count)]


def test_compose_weights_with_no_fields_is_undefined(tmp_path):
    """Nothing to modulate means no weight, not a map of ones."""
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
    )
    result = interface.run()
    assert not isdefined(result.outputs.jacobian_weight_images)


def test_compose_weights_returns_one_path_per_volume(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        gradwarp_field=[str(write_itk_field(tmp_path / 'g.nii.gz'))],
    )
    weights = interface.run().outputs.jacobian_weight_images
    assert len(weights) == 4


def test_compose_weights_dedups_a_single_shared_field(tmp_path):
    """One gradwarp field for the whole run costs one determinant, not N."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        gradwarp_field=[str(write_itk_field(tmp_path / 'g.nii.gz'))],
    )
    weights = interface.run().outputs.jacobian_weight_images
    assert len(set(weights)) == 1


def test_compose_weights_keeps_two_blip_directions_distinct(tmp_path):
    """DRBUDDI rpe_series has one warp per blip direction, so two maps."""
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    up = str(write_itk_field(tmp_path / 'up.nii.gz', amplitude=0.4))
    down = str(write_itk_field(tmp_path / 'down.nii.gz', amplitude=0.2))
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[up, up, down, down],
    )
    weights = interface.run().outputs.jacobian_weight_images
    assert len(set(weights)) == 2
    assert weights[0] == weights[1]
    assert weights[2] == weights[3]


def test_compose_weights_broadcasts_a_single_fieldwarp(tmp_path):
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[str(write_itk_field(tmp_path / 'f.nii.gz'))],
    )
    weights = interface.run().outputs.jacobian_weight_images
    assert len(weights) == 3
    assert len(set(weights)) == 1


def test_compose_weights_rejects_a_mislatticed_field(tmp_path):
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 2),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        fieldwarps=[str(write_itk_field(tmp_path / 'f.nii.gz', shape=(6, 6, 6)))],
    )
    with pytest.raises(ValueError, match='shape'):
        interface.run()


def test_compose_weights_rejects_mismatched_ec_count(tmp_path):
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 4),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        ec_jacobian_images=[_write_map(tmp_path / 'ec0.nii.gz', 1.0)],
    )
    with pytest.raises(ValueError, match='eddy-current'):
        interface.run()


def test_compose_weights_applies_ec_only(tmp_path):
    """TORTOISE EC with no gradwarp and no SDC still produces weights."""
    ec = [_write_map(tmp_path / f'ec{i}.nii.gz', 1.0 + 0.1 * i) for i in range(3)]
    interface = ComposeJacobianWeights(
        dwi_files=_dwi_volumes(tmp_path, 3),
        b0_ref_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        mask=_write_map(tmp_path / 'mask.nii.gz', 1.0),
        ec_jacobian_images=ec,
    )
    weights = interface.run().outputs.jacobian_weight_images
    assert len(weights) == 3
    assert len(set(weights)) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -k compose_weights -v
```

Expected: `ImportError: cannot import name 'ComposeJacobianWeights'`.

- [ ] **Step 3: Write the implementation**

```python
# Append to qsiprep/interfaces/jacobian.py

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
)
from nipype.utils.filemanip import fname_presuffix


class _ComposeJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='split DWI volumes, in their native grid; supplies the volume count',
    )
    b0_ref_image = File(
        exists=True,
        mandatory=True,
        desc='undistorted b=0 reference; the lattice the weight maps live on and '
        'the reference for composing two fields',
    )
    # NOTE for the implementer: only this image's *grid* is used -- as the
    # ``-r`` reference for composition and as the geometry the input fields are
    # validated against. Its voxel content is never read. That matters because
    # on the DRBUDDI-with-T2w path ``b0_ref_image`` is the *structural* image
    # rather than a b=0 (``DRBUDDIAggregateOutputs`` returns
    # ``structural_image`` as ``b0_ref`` when one exists,
    # ``qsiprep/interfaces/tortoise.py:503-507``). That is still correct here:
    # ``init_structural_to_b0_alignment_wf`` resamples the T2w with ``b0_ref``
    # as its ``reference_image`` (``qsiprep/workflows/dwi/registration.py:157``),
    # so the structural is on the b=0 lattice. Do not "fix" this by reaching for
    # a different input.
    mask = File(
        exists=True,
        mandatory=True,
        desc='native-space brain mask, for the positivity guard',
    )
    gradwarp_field = InputMultiObject(
        File(exists=True),
        desc='gradient nonlinearity displacement field (one, shared by every volume)',
    )
    fieldwarps = InputMultiObject(
        File(exists=True),
        desc='SDC displacement field(s): one shared, or one per DWI volume',
    )
    ec_jacobian_images = InputMultiObject(
        File(exists=True),
        desc='per-volume eddy-current Jacobian determinants (TORTOISE DIFFPREP)',
    )


class _ComposeJacobianWeightsOutputSpec(TraitedSpec):
    jacobian_weight_images = OutputMultiObject(
        File(exists=True),
        desc='one weight map per DWI volume, with repeats where volumes share one',
    )


class ComposeJacobianWeights(SimpleInterface):
    """Build per-volume Jacobian weight maps for the native distortion warps.

    The weight for a volume is ``|det grad(gradwarp . fieldwarp)|`` evaluated in
    undistorted b=0-reference space, times that volume's eddy-current Jacobian
    when one exists. Head motion, coregistration and the intramodal/template
    warps are excluded by policy; see the design spec.

    Unique ``(gradwarp, fieldwarp)`` combinations are computed once and shared,
    so a run with one gradwarp field and one SDC warp costs a single ANTs call
    even with hundreds of volumes.
    """

    input_spec = _ComposeJacobianWeightsInputSpec
    output_spec = _ComposeJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        num_dwis = len(self.inputs.dwi_files)
        reference = self.inputs.b0_ref_image

        gradwarp = None
        if isdefined(self.inputs.gradwarp_field) and self.inputs.gradwarp_field:
            if len(self.inputs.gradwarp_field) != 1:
                raise ValueError(
                    'Expected a single gradwarp field, got '
                    f'{len(self.inputs.gradwarp_field)}.'
                )
            gradwarp = self.inputs.gradwarp_field[0]
            validate_field_geometry(gradwarp, reference)

        fieldwarps = [None] * num_dwis
        if isdefined(self.inputs.fieldwarps) and self.inputs.fieldwarps:
            supplied = list(self.inputs.fieldwarps)
            if len(supplied) == 1:
                LOGGER.info('Using a single SDC warp for all DWI volumes')
                fieldwarps = supplied * num_dwis
            elif len(supplied) == num_dwis:
                LOGGER.info('Using per-volume SDC warps')
                fieldwarps = supplied
            else:
                raise ValueError(
                    f'Got {len(supplied)} SDC warps for {num_dwis} DWI volumes; '
                    'expected 1 or one per volume.'
                )
            for warp in set(fieldwarps):
                validate_field_geometry(warp, reference)

        validate_scalar_geometry(self.inputs.mask, reference)

        ec_images = [None] * num_dwis
        if isdefined(self.inputs.ec_jacobian_images) and self.inputs.ec_jacobian_images:
            supplied = list(self.inputs.ec_jacobian_images)
            if len(supplied) != num_dwis:
                raise ValueError(
                    f'Got {len(supplied)} eddy-current Jacobians for {num_dwis} '
                    'DWI volumes; expected one per volume.'
                )
            for ec_image in set(supplied):
                validate_scalar_geometry(ec_image, reference)
            ec_images = supplied

        if gradwarp is None and not any(fieldwarps) and not any(ec_images):
            LOGGER.info('No distortion transforms to modulate; no weights produced')
            return runtime

        # One determinant per unique (gradwarp, fieldwarp) pair.
        determinants = {}
        for fieldwarp in dict.fromkeys(fieldwarps):
            key = weight_key(gradwarp, fieldwarp)
            if not key or key in determinants:
                continue
            fields = [path for path in (gradwarp, fieldwarp) if path]
            if len(fields) == 2:
                composed = compose_fields(
                    fields, reference, os.path.join(runtime.cwd, f'composite{len(determinants)}.nii.gz')
                )
            else:
                composed = fields[0]
            determinants[key] = jacobian_determinant(
                composed,
                os.path.join(runtime.cwd, f'jacobian{len(determinants)}.nii.gz'),
                mask_path=self.inputs.mask,
            )

        # Per-volume weight = shared determinant x that volume's EC Jacobian.
        weights = []
        cache = {}
        for index, (fieldwarp, ec_image) in enumerate(zip(fieldwarps, ec_images, strict=True)):
            factors = []
            key = weight_key(gradwarp, fieldwarp)
            if key:
                factors.append(determinants[key])
            if ec_image:
                factors.append(ec_image)

            # Tag the roles rather than collapsing missing factors into an
            # untagged tuple: (None, 'f.nii.gz') and ('f.nii.gz', None) are
            # different weights that would otherwise share a key.
            cache_key = (gradwarp, fieldwarp, ec_image)
            if cache_key not in cache:
                cache[cache_key] = multiply_maps(
                    factors,
                    fname_presuffix(
                        self.inputs.dwi_files[index],
                        suffix=f'_jacobian-{index:05d}',
                        newpath=runtime.cwd,
                        use_ext=True,
                    ),
                )
                check_weight_map(cache[cache_key], self.inputs.mask)
            weights.append(cache[cache_key])

        self._results['jacobian_weight_images'] = weights
        return runtime
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -v
docker run --rm -v /mnt/c/Users/tsalo/Documents/linc/qsiprep:/src \
  --entrypoint pytest pennlinc/qsiprep:test \
  /src/qsiprep/tests/test_interfaces_jacobian.py -v
```

Expected: PASS in both (pure tests locally, all tests in the image).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/jacobian.py qsiprep/tests/test_interfaces_jacobian.py
git commit -m "feat: add ComposeJacobianWeights interface

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Rename `ApplyScalingImages` and expose the resampled maps

**Files:**
- Modify: `qsiprep/interfaces/fmap.py:1017-1135`
- Modify: `qsiprep/workflows/dwi/resampling.py:18` (the import)
- Test: `qsiprep/tests/test_interfaces_fmap.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ApplyJacobianWeights` with input `jacobian_weight_images` (replacing `scaling_image_files`) and outputs `scaled_images` (unchanged name) plus `resampled_weight_images` (the unique output-grid maps, in first-appearance order).

- [ ] **Step 1: Write the failing test**

```python
# Append to qsiprep/tests/test_interfaces_fmap.py

import nibabel as nb
import numpy as np
from nipype.interfaces.base import isdefined


def _unit_image(path, value=1.0):
    nb.Nifti1Image(np.full((8, 8, 8), value, dtype='float32'), np.eye(4)).to_filename(
        str(path)
    )
    return str(path)


def test_apply_jacobian_weights_passes_through_without_weights(tmp_path):
    """No weights means the resampled DWIs are handed on untouched."""
    from qsiprep.interfaces.fmap import ApplyJacobianWeights

    dwis = [_unit_image(tmp_path / f'd{i}.nii.gz') for i in range(3)]
    result = ApplyJacobianWeights(
        dwi_files=dwis,
        reference_image=_unit_image(tmp_path / 'grid.nii.gz'),
    ).run()
    assert result.outputs.scaled_images == dwis
    assert not isdefined(result.outputs.resampled_weight_images)


def test_apply_jacobian_weights_rejects_a_count_mismatch(tmp_path):
    from qsiprep.interfaces.fmap import ApplyJacobianWeights

    with pytest.raises(Exception, match='Mismatch'):
        ApplyJacobianWeights(
            dwi_files=[_unit_image(tmp_path / f'd{i}.nii.gz') for i in range(3)],
            jacobian_weight_images=[_unit_image(tmp_path / 'w.nii.gz')],
            reference_image=_unit_image(tmp_path / 'grid.nii.gz'),
        ).run()


def test_apply_scaling_images_name_is_gone():
    """The old name must not linger as an alias -- it meant something else."""
    import qsiprep.interfaces.fmap as fmap

    assert not hasattr(fmap, 'ApplyScalingImages')
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_fmap.py -k "jacobian_weights or scaling_images_name" -v
```

Expected: `ImportError: cannot import name 'ApplyJacobianWeights'`.

- [ ] **Step 3: Rewrite the interface**

In `qsiprep/interfaces/fmap.py`, rename the three classes and change the input name and the resampled-map bookkeeping:

```python
class _ApplyJacobianWeightsInputSpec(ApplyTransformsInputSpec):
    input_image = traits.File(mandatory=False)
    jacobian_weight_images = InputMultiObject(
        File(exists=True),
        mandatory=False,
        desc='per-volume Jacobian weight maps, in undistorted b0ref space',
    )
    dwi_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='list of dwi files, already resampled into their output space',
    )
    reference_image = File(exists=True, mandatory=True, desc='output grid')
    # ... the transform inputs below are unchanged ...


class _ApplyJacobianWeightsOutputSpec(TraitedSpec):
    scaled_images = OutputMultiObject(File(exists=True), desc='Weighted dwi files')
    resampled_weight_images = OutputMultiObject(
        File(exists=True),
        desc='the unique weight maps, resampled to the output grid, in '
        'first-appearance order; indexed by the derivative sidecar',
    )
    weight_index = traits.List(
        traits.Int(),
        desc='for each DWI volume, its index into resampled_weight_images',
    )


class ApplyJacobianWeights(SimpleInterface):
    """Transport Jacobian weight maps to the output grid and multiply them in.

    The maps arrive in undistorted b=0-reference space. Resampling them through
    the intramodal and coregistration transforms is what evaluates the gradwarp
    and SDC determinants at the coordinates the full composite evaluates them
    at -- see the design spec's coordinate-safety section.
    """

    input_spec = _ApplyJacobianWeightsInputSpec
    output_spec = _ApplyJacobianWeightsOutputSpec
```

In `_run_interface`, apply this **exact** rename list — the snippet below uses
`weights_to_dwis`, which does not exist until you rename it, and a literal
implementation without these renames raises `NameError`:

| Current name (`qsiprep/interfaces/fmap.py:1062-1135`) | New name |
|---|---|
| `self.inputs.scaling_image_files` | `self.inputs.jacobian_weight_images` |
| `scaling_images_to_dwis` | `weights_to_dwis` |
| `dwi_files_to_scalings` | `dwi_files_to_weights` |
| `scaling_image` (loop variable) | `weight_image` |
| `resampled_scaling_image` | `resampled_weight_image` |

Keep the transform-stack construction (`fmap.py:1071-1096`) and the
`defaultdict` dedup exactly as they are — the transport chain is unchanged and
is what evaluates the determinants at the right coordinates. Then record the
resampled maps and the per-volume index:

```python
        # Apply the transform, link the resampled weight map to resampled dwis
        dwi_files_to_weights = {}
        resampled_unique = []
        for weight_image in weights_to_dwis:
            resampled_weight_image = fname_presuffix(
                weight_image, suffix='_resampled', newpath=runtime.cwd
            )
            xfm = ants.ApplyTransforms(
                input_image=weight_image,
                transforms=transform_stack,
                reference_image=self.inputs.reference_image,
                output_image=resampled_weight_image,
                interpolation='LanczosWindowedSinc',
                dimension=3,
            )
            xfm.terminal_output = 'allatonce'
            xfm.resource_monitor = False
            runtime = xfm.run().runtime
            LOGGER.info(runtime.cmdline)
            resampled_unique.append(resampled_weight_image)
            for dwi_file in weights_to_dwis[weight_image]:
                dwi_files_to_weights[dwi_file] = resampled_weight_image

        # LanczosWindowedSinc can undershoot below zero on a positive scalar
        # map. The pre-transport guard cannot see that, because it ran before
        # this resampling, so clamp with an explicit floor and say so rather
        # than letting a negative weight through.
        for resampled_weight_image in resampled_unique:
            img = nb.load(resampled_weight_image)
            data = np.asanyarray(img.dataobj)
            if data.min() <= 0:
                LOGGER.warning(
                    'Resampled weight map %s undershot to %.4f (Lanczos ringing '
                    'on a positive scalar map); clamping to %g.',
                    resampled_weight_image,
                    data.min(),
                    WEIGHT_FLOOR,
                )
                nb.Nifti1Image(
                    np.maximum(data, WEIGHT_FLOOR).astype('float32'),
                    img.affine,
                    img.header,
                ).to_filename(resampled_weight_image)

        self._results['resampled_weight_images'] = resampled_unique
        # The per-volume lookup, as indices into resampled_unique. Task 12
        # cannot build JacobianWeightIndex without this: resampled_unique is
        # deduplicated, so it alone does not say which volume used which map.
        self._results['weight_index'] = [
            resampled_unique.index(dwi_files_to_weights[dwi_file])
            for dwi_file in self.inputs.dwi_files
        ]
```

Then update the import in `qsiprep/workflows/dwi/resampling.py`:

```python
from ...interfaces.fmap import ApplyJacobianWeights
```

and the node construction:

```python
    scale_dwis = pe.Node(ApplyJacobianWeights(), name='scale_dwis')
```

**Close the broken-connection window in this same task.** Renaming the trait
invalidates the existing `('sdc_scaling_images', 'scaling_image_files')`
connection in `init_dwi_trans_wf`, and nothing would catch that until Task 7,
because an import test does not build the workflow — Nipype validates
connections at construction time. So in *this* task, delete that one entry from
the `(inputnode, scale_dwis, [...])` block, leaving `scale_dwis` with no weight
input. That is a valid intermediate state: the interface's no-weights branch
passes the DWIs through untouched, so the pipeline still runs, just without
weighting until Task 7 wires `compose_jacobian` in.

Keep the rest of that connect block — the intramodal and coregistration
transforms `scale_dwis` uses to transport the maps are unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_fmap.py -v
micromamba run -n linc311 python -c "
from qsiprep import config
config.workflow.output_resolution = 2.0
from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf
init_dwi_trans_wf(source_file='/data/sub-1_dwi.nii.gz', mem_gb=1)
print('workflow constructs')
"
```

Expected: PASS, and the workflow *constructs* — an import alone would not
exercise Nipype's connection validation, which is what a stale trait name
breaks. Also grep for stragglers:

```bash
grep -rn "ApplyScalingImages\|scaling_image_files" --include=*.py qsiprep/
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/fmap.py qsiprep/workflows/dwi/resampling.py \
  qsiprep/tests/test_interfaces_fmap.py
git commit -m "refactor: rename ApplyScalingImages to ApplyJacobianWeights

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The CLI flag and the effective-eddy-method helper

**Files:**
- Create: `qsiprep/utils/eddy_config.py`
- Modify: `qsiprep/cli/parser.py` (in the `g_moco` group, after `--eddy-config`)
- Modify: `qsiprep/config.py` (the `workflow` section, near `gradient_file` at line 598)
- Test: `qsiprep/tests/test_cli.py`, `qsiprep/tests/test_utils_misc.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `config.workflow.jacobian_weighting: bool` (default `True`)
  - `effective_eddy_resampling_method(eddy_args: dict) -> str` returning `'jac'` or `'lsr'`
  - `eddy_modulates_distortion(eddy_args: dict) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# qsiprep/tests/test_utils_misc.py -- append

def test_effective_eddy_method_defaults_to_jac():
    from qsiprep.utils.eddy_config import effective_eddy_resampling_method

    assert effective_eddy_resampling_method({}) == 'jac'


def test_effective_eddy_method_reads_the_config():
    from qsiprep.utils.eddy_config import effective_eddy_resampling_method

    assert effective_eddy_resampling_method({'method': 'lsr'}) == 'lsr'


def test_eddy_modulates_distortion_only_for_jac():
    from qsiprep.utils.eddy_config import eddy_modulates_distortion

    assert eddy_modulates_distortion({'method': 'jac'})
    assert not eddy_modulates_distortion({'method': 'lsr'})


def test_shipped_default_config_modulates():
    """The shipped eddy_params.json must keep jac, or the backend table lies."""
    import json

    from qsiprep.data import load as load_data
    from qsiprep.utils.eddy_config import eddy_modulates_distortion

    assert eddy_modulates_distortion(json.loads(load_data('eddy_params.json').read_text()))
```

```python
# qsiprep/tests/test_cli.py -- append

def test_jacobian_weighting_defaults_on():
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args(
        ['bids', 'out', 'participant', '--output-resolution', '2']
    )
    assert opts.jacobian_weighting is True


def test_no_jacobian_weighting_turns_it_off():
    from qsiprep.cli.parser import _build_parser

    opts = _build_parser().parse_args(
        ['bids', 'out', 'participant', '--output-resolution', '2',
         '--no-jacobian-weighting']
    )
    assert opts.jacobian_weighting is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_utils_misc.py -k eddy \
  qsiprep/tests/test_cli.py -k jacobian -v
```

Expected: `ModuleNotFoundError: qsiprep.utils.eddy_config` and `AttributeError: 'Namespace' object has no attribute 'jacobian_weighting'`.

- [ ] **Step 3: Write the implementation**

`qsiprep/utils/eddy_config.py`:

```python
"""Questions about the effective FSL ``eddy`` configuration.

``eddy``'s resampling method decides whether it applied its own Jacobian
modulation for eddy-current and TOPUP susceptibility distortions. QSIPrep ships
``jac``, but ``--eddy-config`` lets a user supply any JSON, so this is a default
and not an invariant. Two places need the answer -- ``init_fsl_hmc_wf`` for the
warning and the methods boilerplate, and ``init_dwi_trans_wf`` for the
weighting decision -- so it is resolved once here, from the already-loaded
dict, rather than parsed twice.
"""

#: What ``eddy`` does when ``method`` is absent from the config. Nipype's trait
#: maps ``method`` to ``--resamp``, whose own default is ``jac``.
DEFAULT_RESAMPLING_METHOD = 'jac'


def effective_eddy_resampling_method(eddy_args):
    """The ``--resamp`` value ``eddy`` will actually run with."""
    return eddy_args.get('method') or DEFAULT_RESAMPLING_METHOD


def eddy_modulates_distortion(eddy_args):
    """Whether ``eddy`` Jacobian-modulates eddy-current and susceptibility.

    True for ``--resamp=jac``. False for ``lsr``, which is a different
    resampling model; the claim is deliberately narrow -- it says only that the
    Jacobian modulation this feature is about did not happen, not anything
    broader about least-squares restoration's intensity semantics.
    """
    return effective_eddy_resampling_method(eddy_args) == 'jac'
```

In `qsiprep/workflows/dwi/fsl.py`, right after `eddy_args` is loaded
(`qsiprep/workflows/dwi/fsl.py:182`), emit the user-visible runtime warning
and record the gap for the derivative sidecar:

```python
    from ...utils.eddy_config import eddy_modulates_distortion

    if not eddy_modulates_distortion(eddy_args):
        config.loggers.workflow.warning(
            'eddy is configured with method=%s, not "jac", so eddy-current '
            'and susceptibility distortion corrections will NOT be '
            'Jacobian-modulated. QSIPrep cannot retrofit this: eddy has '
            'already baked its resampling in and exports no field for the '
            'eddy-current component.',
            eddy_args.get('method'),
        )
        config.workflow.jacobian_unmodulated_corrections += [
            'eddy-current', 'susceptibility'
        ]
        config.workflow.jacobian_unmodulated_reason = (
            f'FSL eddy ran with --resamp={eddy_args.get("method")} rather than jac'
        )
```

In `qsiprep/cli/parser.py`, immediately after the `--eddy-config` argument:

```python
    g_moco.add_argument(
        '--jacobian-weighting',
        action=BooleanOptionalAction,
        default=True,
        help='Apply Jacobian intensity modulation for gradient-nonlinearity, '
        'susceptibility and eddy-current distortion corrections (default: on). '
        'This option controls only the modulation QSIPrep itself applies. With '
        '--hmc-method eddy, FSL eddy applies its own modulation for '
        'eddy-current and TOPUP susceptibility distortions whenever its '
        'resampling method is "jac" (the default; see --eddy-config), and that '
        'is internal to eddy and unaffected by this option.',
    )
```

`parser.py` imports argparse names *function-locally* inside `_build_parser`
(`qsiprep/cli/parser.py:46-51`), not at module scope. Add
`BooleanOptionalAction` to that existing `from argparse import (...)` block —
do not add a module-level `import argparse`.

In `qsiprep/config.py`, in the `workflow` section next to `gradient_file`:

```python
    jacobian_weighting = True
    """Apply Jacobian intensity modulation for the spatial distortion corrections."""
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_utils_misc.py -k eddy \
  qsiprep/tests/test_cli.py -k jacobian -v
micromamba run -n linc311 python -m pytest qsiprep/tests/test_config.py -v
```

Expected: PASS, and `test_config.py` still passes (it checks the config round-trips).

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/eddy_config.py qsiprep/cli/parser.py qsiprep/config.py \
  qsiprep/tests/test_utils_misc.py qsiprep/tests/test_cli.py
git commit -m "feat: add --jacobian-weighting flag and effective eddy-method helper

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Wire `compose_jacobian` into the resampling workflow

This is where GRE, SyN, T2Wreg and DRBUDDI all start getting weighted, because their warps already arrive as `fieldwarps`. No per-backend code is needed.

**Files:**
- Modify: `qsiprep/workflows/dwi/resampling.py`
- Test: `qsiprep/tests/test_workflows_jacobian.py`

**Interfaces:**
- Consumes: `ComposeJacobianWeights` (Task 4), `ApplyJacobianWeights` (Task 5), `config.workflow.jacobian_weighting` and `eddy_modulates_distortion` (Task 6).
- Produces: `init_dwi_trans_wf` containing a `compose_jacobian` node connected to `scale_dwis.jacobian_weight_images`, plus a new `ec_jacobian_images` inputnode field.

- [ ] **Step 1: Write the failing tests**

```python
"""Construction tests for Jacobian weighting inside the resampling workflow.

Asserts the graph, not the numbers: that the weighting node exists, that it is
fed from the right sources, and that ``--no-jacobian-weighting`` removes it.
The numeric correctness of the weights lives in
``test_interfaces_jacobian.py`` and ``test_jacobian_conservation.py``.
"""

import pytest

from qsiprep import config


@pytest.fixture(autouse=True)
def _reset_config():
    saved = (config.workflow.jacobian_weighting, config.workflow.output_resolution)
    config.workflow.jacobian_weighting = True
    config.workflow.output_resolution = 2.0
    yield
    config.workflow.jacobian_weighting, config.workflow.output_resolution = saved


def _trans_wf():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    return init_dwi_trans_wf(source_file='/data/sub-1_dwi.nii.gz', mem_gb=1)


def _edges(workflow):
    return {
        (src.name, dst.name, tuple(sorted(data['connect'])))
        for src, dst, data in workflow._graph.edges(data=True)
    }


def test_compose_jacobian_node_exists():
    assert _trans_wf().get_node('compose_jacobian') is not None


def test_compose_jacobian_feeds_the_weighting_node():
    edges = _edges(_trans_wf())
    assert any(
        src == 'compose_jacobian'
        and dst == 'scale_dwis'
        and ('jacobian_weight_images', 'jacobian_weight_images') in connect
        for src, dst, connect in edges
    )


def test_compose_jacobian_consumes_gradwarp_and_fieldwarps():
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if src == 'inputnode' and dst == 'compose_jacobian'
        for pair in connect
    }
    sources = {source for source, _ in forwarded}
    assert 'fieldwarps' in sources
    assert 'gradwarp_field' in sources
    assert 'dwi_files' in sources
    assert 'b0_ref_image' in sources
    assert 'dwi_mask' in sources
    assert 'ec_jacobian_images' in sources


def test_no_jacobian_weighting_removes_the_node():
    config.workflow.jacobian_weighting = False
    assert _trans_wf().get_node('compose_jacobian') is None


def test_no_jacobian_weighting_leaves_weights_unconnected():
    """With weighting off, scale_dwis must get nothing and pass data through."""
    config.workflow.jacobian_weighting = False
    edges = _edges(_trans_wf())
    assert not any(
        ('jacobian_weight_images', 'jacobian_weight_images') in connect
        for _, _, connect in edges
    )


def test_hmc_xforms_never_reach_the_jacobian_node():
    """HMC is excluded by policy; a wire here would silently modulate by it."""
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if dst == 'compose_jacobian'
        for pair in connect
    }
    assert not any('hmc' in source for source, _ in forwarded)


def test_coreg_and_template_transforms_never_reach_the_jacobian_node():
    edges = _edges(_trans_wf())
    forwarded = {
        pair
        for src, dst, connect in edges
        if dst == 'compose_jacobian'
        for pair in connect
    }
    excluded = ('itk_b0_to_t1', 'intramodal', 't1_2_mni')
    for source, _ in forwarded:
        assert not any(name in source for name in excluded), source
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -v
```

Expected: FAIL — `assert None is not None` on the node-exists test.

- [ ] **Step 3: Write the implementation**

In `qsiprep/workflows/dwi/resampling.py`, add the import:

```python
from ...interfaces.jacobian import ComposeJacobianWeights
```

Add `'ec_jacobian_images'` to the `inputnode` field list (next to `'sdc_scaling_images'`, which Task 10 removes), and replace the `sdc_scaling_images` wiring into `scale_dwis` with the new node. The `compose_jacobian` node is built only when weighting is enabled:

```python
    if config.workflow.jacobian_weighting:
        compose_jacobian = pe.Node(ComposeJacobianWeights(), name='compose_jacobian')
        workflow.connect([
            (inputnode, compose_jacobian, [
                ('dwi_files', 'dwi_files'),
                ('b0_ref_image', 'b0_ref_image'),
                ('dwi_mask', 'mask'),
                (('gradwarp_field', _listify), 'gradwarp_field'),
                ('fieldwarps', 'fieldwarps'),
                ('ec_jacobian_images', 'ec_jacobian_images'),
            ]),
            (compose_jacobian, scale_dwis, [
                ('jacobian_weight_images', 'jacobian_weight_images'),
            ]),
        ])  # fmt:skip
```

Note what is deliberately *not* connected: `hmc_xforms`, `itk_b0_to_t1`, the
intramodal transforms and `t1_2_mni_forward_transform`. `scale_dwis` still
receives the intramodal and coregistration transforms, because transporting the
weight map through them is what evaluates the determinants at the right
coordinates — that wiring is unchanged from the `sdc_scaling_images` era.

Add a comment above the node recording why:

```python
    # The weight covers gradwarp and SDC only. HMC is excluded by policy and is
    # coordinate-safe to exclude because it is the outermost transform in the
    # pull-back (see the design spec); coregistration and the intramodal and
    # template warps are excluded because modulating by a spatial-normalization
    # warp is VBM-style volume modulation, wrong for DWI signal.
```


- [ ] **Step 3a: Add the per-backend wiring tests**

The tests above build the generic resampling workflow. They prove
`compose_jacobian` is wired, but not that each fieldmap backend actually
*delivers* a warp into `fieldwarps` — which is the whole reason GRE, SyN,
T2Wreg and DRBUDDI need no per-backend code. Assert that claim per branch, or
a backend could silently stop emitting a warp and weighting would quietly
become a no-op for it.

```python
# Append to qsiprep/tests/test_workflows_jacobian.py

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep.tests.preproc_factory import make_preproc_unit


@pytest.mark.parametrize(
    ('method', 'sources'),
    [
        (
            CorrectionMethod.PEPOLAR,
            ['/data/sub-1/dwi/sub-1_dwi.nii.gz', '/data/sub-1/fmap/sub-1_epi.nii.gz'],
        ),
        (
            CorrectionMethod.FIELDMAP,
            [
                '/data/sub-1/fmap/sub-1_phasediff.nii.gz',
                '/data/sub-1/fmap/sub-1_magnitude1.nii.gz',
            ],
        ),
        (CorrectionMethod.SYN, ['/data/sub-1/dwi/sub-1_dwi.nii.gz']),
    ],
)
def test_each_sdc_branch_emits_a_warp_for_weighting(method, sources):
    """Every SDC branch must reach ``to_dwi_ref_warps``.

    ``to_dwi_ref_warps`` becomes ``fieldwarps``, which is the single input
    ComposeJacobianWeights derives the SDC determinant from. A branch that
    stops populating it loses weighting silently rather than loudly.
    """
    from qsiprep.workflows.fieldmap import init_sdc_wf

    unit = make_preproc_unit(
        ['/data/sub-1/dwi/sub-1_dwi.nii.gz'],
        method=method,
        pe_dir='j',
        estimation_sources=sources,
    )
    workflow = init_sdc_wf(unit)
    outputnode = workflow.get_node('outputnode')
    assert 'out_warp' in outputnode.outputs.copyable_trait_names()

    connected = {
        pair
        for _, dst, data in workflow._graph.edges(data=True)
        if dst.name == 'outputnode'
        for pair in data['connect']
    }
    assert any(target == 'out_warp' for _, target in connected), (
        f'{method} produced no out_warp, so nothing would reach fieldwarps '
        'and this backend would be silently unweighted.'
    )
```

Note the one branch deliberately absent from the list: TOPUP-only. It
populates `to_dwi_ref_warps` with `GatherEddyInputs.forward_warps`, which is
empty by design (Task 9), because `eddy` already applied and modulated that
field.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -v
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_native.py qsiprep/tests/test_workflows_gradwarp.py -v
```

Expected: the new tests PASS and the existing workflow-construction tests still match the baseline from Task 0.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/workflows/dwi/resampling.py qsiprep/tests/test_workflows_jacobian.py
git commit -m "feat: apply Jacobian weighting in the DWI resampling workflow

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Delete the dead GRE Jacobian node

`init_sdc_unwarp_wf` computes a Jacobian at `qsiprep/workflows/fieldmap/unwarp.py:177` and exposes it as `out_jacobian` at `:236`, and nothing has ever consumed it. It is the determinant of the SDC warp *alone*, whereas the design needs the determinant of the gradwarp-composed-with-SDC warp, which Task 4 computes. Delete it rather than leaving two sources of truth.

**Files:**
- Modify: `qsiprep/workflows/fieldmap/unwarp.py`
- Test: `qsiprep/tests/test_workflows_jacobian.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Write the failing test**

```python
# Append to qsiprep/tests/test_workflows_jacobian.py

def test_sdc_unwarp_wf_has_no_dead_jacobian_node():
    """The SDC-warp-only Jacobian is not what the weighting needs.

    It was computed and discarded for years. ComposeJacobianWeights derives the
    determinant of the *composed* gradwarp-and-SDC warp instead, so leaving
    this node in place would be a second, subtly-wrong source of truth.
    """
    from qsiprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf

    workflow = init_sdc_unwarp_wf()
    assert workflow.get_node('jac_dfm') is None
    assert 'out_jacobian' not in workflow.get_node('outputnode').outputs.copyable_trait_names()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -k dead_jacobian -v
```

Expected: FAIL — the node and the output still exist.

- [ ] **Step 3: Remove the dead code**

In `qsiprep/workflows/fieldmap/unwarp.py`:
- delete `'out_jacobian'` from the `outputnode` field list (~line 113)
- delete the `jac_dfm` node definition (~lines 177-180)
- delete the two connections referencing it: `(vsm2dfm, jac_dfm, [('out_file', 'deformationField')])` and `(jac_dfm, outputnode, [('jacobian_image', 'out_jacobian')])` (~lines 226, 236)
- delete the `out_jacobian` entry from the docstring's Outputs section (~lines 76-77)
- drop `ants` from the imports if nothing else in the module uses it

- [ ] **Step 4: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -v
grep -rn "out_jacobian\|jac_dfm" --include=*.py qsiprep/
micromamba run -n linc311 python -c "import qsiprep.workflows.fieldmap.unwarp"
```

Expected: PASS, no grep output, clean import.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/workflows/fieldmap/unwarp.py qsiprep/tests/test_workflows_jacobian.py
git commit -m "refactor: remove the unused SDC-only Jacobian node

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The TOPUP no-double-counting regression invariant

`GatherEddyInputs` sets `forward_warps = []` unconditionally (`qsiprep/interfaces/eddy.py:206`), so on the TOPUP-only path `fieldwarps` is empty and `eddy`'s internal modulation cannot be applied twice. That was checked by inspection during design review; this task makes it a test so a future change to `GatherEddyInputs` cannot silently introduce double-counting.

**Files:**
- Test: `qsiprep/tests/test_interfaces_eddy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (a guard).

- [ ] **Step 1: Write the test**

```python
# Append to qsiprep/tests/test_interfaces_eddy.py

def test_gather_eddy_inputs_exports_no_warps(tmp_path):
    """eddy bakes TOPUP's field in, so it must export no SDC warp downstream.

    If ``forward_warps`` ever carried the TOPUP field, it would reach
    ``fieldwarps`` and ``ComposeJacobianWeights`` would derive a determinant
    for a distortion ``eddy`` has already Jacobian-modulated internally --
    applying it twice. The empty list is load-bearing, not incidental.

    Asserted on the interface's actual *output*, not on its source text: a
    source-text check passes even if later code overwrites the list.
    """
    from qsiprep.interfaces.eddy import GatherEddyInputs
    from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients

    dwi = write_dwi_with_gradients(tmp_path / 'sub-1_dwi.nii.gz', nvols=4)
    stem = str(dwi).split('.nii')[0]
    json_file = tmp_path / 'sub-1_dwi.json'
    json_file.write_text(
        '{"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}'
    )

    result = GatherEddyInputs(
        dwi_file=dwi,
        bval_file=stem + '.bval',
        bvec_file=stem + '.bvec',
        json_file=str(json_file),
        original_files=[dwi] * 4,
        topup_requested=True,
    ).run(cwd=str(tmp_path))

    assert result.outputs.forward_warps == []
    assert result.outputs.forward_transforms == []
```

- [ ] **Step 2: Run the test**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_eddy.py -k exports_no_warps -v
```

Expected: PASS immediately — it documents an existing invariant rather than driving new code. If it fails, `GatherEddyInputs` has changed since the design review and Task 7's wiring must be revisited before going further.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_interfaces_eddy.py
git commit -m "test: guard the TOPUP no-double-counting invariant

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Retire DRBUDDI's ratio images

**Prerequisite: Task 2's finding.** If `test_jac_over_corrected_matches_analytic_determinant` passes, proceed. If it fails, **stop and report** — take the spec's option (2) fallback (keep DRBUDDI on ratio images for this release, ship the analytic path for the other backends) rather than deleting the ratios on an unvalidated basis.

**Files:**
- Modify: `qsiprep/interfaces/tortoise.py:499-541`
- Modify: `qsiprep/workflows/fieldmap/drbuddi.py`
- Modify: `qsiprep/workflows/dwi/{fsl,diffprep,hmc_sdc,base,finalize}.py`
- Modify: `qsiprep/workflows/dwi/resampling.py`
- Test: `qsiprep/tests/test_workflows_jacobian.py`

**Interfaces:**
- Consumes: `ComposeJacobianWeights` (Task 4), the Task 2 finding.
- Produces: the `sdc_scaling_images` channel is gone; `DRBUDDIAggregateOutputs` no longer has an `sdc_scaling_images` output.

- [ ] **Step 1: Confirm the Task 2 gate is still green**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_drbuddi_jac_semantics.py -v -m drbuddi_rpe
```

The gate was already decided before Task 4 (see Task 2's execution-order
banner), so this is a re-confirmation, not the decision point. If it has gone
red since, something in Tasks 3–9 changed the determinant; investigate that
rather than proceeding with the deletion.

- [ ] **Step 2: Write the failing test**

```python
# Append to qsiprep/tests/test_workflows_jacobian.py

def test_sdc_scaling_images_channel_is_gone():
    """DRBUDDI's ratio images are replaced by the analytic determinant.

    The channel existed only for DRBUDDI, whose warps already arrive as
    ``fieldwarps`` -- so ComposeJacobianWeights derives its determinant the
    same way as every other backend's, and the bespoke plumbing is dead.
    """
    import subprocess

    hits = subprocess.run(
        ['grep', '-rn', 'sdc_scaling_images', '--include=*.py', 'qsiprep/'],
        capture_output=True,
        text=True,
    ).stdout
    assert hits == '', f'sdc_scaling_images still referenced:\n{hits}'


def test_drbuddi_aggregate_has_no_scaling_output():
    from qsiprep.interfaces.tortoise import DRBUDDIAggregateOutputs

    outputs = DRBUDDIAggregateOutputs().output_spec().copyable_trait_names()
    assert 'sdc_scaling_images' not in outputs
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -k "scaling" -v
```

Expected: FAIL, with grep output listing the ~17 current references.

- [ ] **Step 4: Remove the channel**

In `qsiprep/interfaces/tortoise.py`:
- delete `sdc_scaling_images` from `_DRBUDDIAggregateOutputsOutputSpec`
- delete the ratio construction in `_run_interface` (the two `nim.math_img('a/b', ...)` blocks and the `scaling_blip_*_file` paths, ~lines 524-537)
- delete the `self._results['sdc_scaling_images'] = [...]` assignment (~line 539)
- keep `blip_*_b0_corrected_jac` inputs — they remain the validation oracle

In `qsiprep/workflows/fieldmap/drbuddi.py`: delete `'sdc_scaling_images'` from the outputnode field list and the connection at ~line 261.

In `qsiprep/workflows/dwi/fsl.py`, `diffprep.py`, `hmc_sdc.py`, `base.py`, `finalize.py`: delete every `sdc_scaling_images` field declaration and connection (the grep from Step 3 lists each line).

In `qsiprep/workflows/dwi/resampling.py`: delete `'sdc_scaling_images'` from the inputnode field list, and the `(inputnode, scale_dwis, [('sdc_scaling_images', 'scaling_image_files'), ...])` entry — keeping the rest of that connect block, which forwards the intramodal and coregistration transforms that `scale_dwis` needs to transport the weight maps.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -v
micromamba run -n linc311 python -m pytest qsiprep/tests -m "not integration" -q 2>&1 | tail -20
```

Expected: PASS, and the full suite matches the Task 0 baseline.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/tortoise.py qsiprep/workflows/fieldmap/drbuddi.py \
  qsiprep/workflows/dwi/fsl.py qsiprep/workflows/dwi/diffprep.py \
  qsiprep/workflows/dwi/hmc_sdc.py qsiprep/workflows/dwi/base.py \
  qsiprep/workflows/dwi/finalize.py qsiprep/workflows/dwi/resampling.py \
  qsiprep/tests/test_workflows_jacobian.py
git commit -m "refactor: replace DRBUDDI ratio images with the analytic Jacobian

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: TORTOISE eddy-current Jacobian

**Prerequisite, and it is an investigation, not a coding step:** the Okan quadratic coordinate-map formula is **not in this repository**. `_read_okan_transformations` (`qsiprep/interfaces/tortoise.py:1161`) parses 24 scalars and its comments identify columns 0–5 as rigid motion and 6–23 as eddy-current polynomial and centre parameters, but nothing specifies the parameter ordering within 6–23, the polynomial basis, normalization, the rotation/eddy-centre convention, forward-versus-inverse direction, or LPS handling.

**Before writing any code**, source the formula from TORTOISE's own source or documentation and record it — the basis functions, the parameter order, and the direction — in the module docstring with a citation. If it cannot be sourced, **stop and report**: the fallback is to document the EC gap and ship Tasks 1–10 and 12–14 without this one. Nothing else depends on it.

**Files:**
- Modify: `qsiprep/interfaces/jacobian.py`
- Modify: `qsiprep/workflows/dwi/diffprep.py`
- Modify: `qsiprep/workflows/dwi/{base,finalize}.py` (the `ec_jacobian_images` channel)
- Test: `qsiprep/tests/test_interfaces_jacobian.py`, `qsiprep/tests/test_interfaces_diffprep.py`

**Interfaces:**
- Consumes: `ComposeJacobianWeights.ec_jacobian_images` (Task 4).
- Produces: `OkanQuadraticJacobian` with inputs `transformations_file`, `reference_image`, `correction_mode` and output `ec_jacobian_images` (N paths, or `Undefined` for unsupported modes).

- [ ] **Step 1: Record the sourced formula**

Add to `qsiprep/interfaces/jacobian.py` a module-level docstring section giving the formula, its source (file and line in TORTOISE, or the documentation URL and version), the parameter order, the basis, and the direction convention. A reviewer must be able to check the implementation against it without re-deriving it.

- [ ] **Step 2: Write the failing tests**

```python
# Append to qsiprep/tests/test_interfaces_jacobian.py

from qsiprep.interfaces.jacobian import OkanQuadraticJacobian


def _write_transformations(path, rows):
    """Write a DIFFPREP _moteddy_transformations.txt with 24 columns per row."""
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(' '.join(f'{value:.8f}' for value in row) + '\n')
    return str(path)


def test_okan_jacobian_of_identity_parameters_is_unity(tmp_path):
    """All-zero parameters mean no motion and no eddy current, so det = 1."""
    transformations = _write_transformations(tmp_path / 'x.txt', [[0.0] * 24] * 3)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='quadratic',
    ).run()

    maps = result.outputs.ec_jacobian_images
    assert len(maps) == 3
    for path in maps:
        interior = np.asanyarray(nb.load(path).dataobj)[2:-2, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 1.0, atol=1e-4)


def test_okan_jacobian_ignores_the_rigid_columns(tmp_path):
    """Columns 0-5 are rigid motion, excluded by the scope policy."""
    rigid_only = [[1.0, 2.0, 3.0, 0.05, 0.05, 0.05] + [0.0] * 18]
    transformations = _write_transformations(tmp_path / 'x.txt', rigid_only)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='quadratic',
    ).run()

    interior = np.asanyarray(
        nb.load(result.outputs.ec_jacobian_images[0]).dataobj
    )[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 1.0, atol=1e-4)


def test_okan_jacobian_is_undefined_for_motion_only(tmp_path):
    """--sloppy forces correction_mode=motion, where no EC component exists."""
    transformations = _write_transformations(tmp_path / 'x.txt', [[0.0] * 24] * 2)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='motion',
    ).run()
    assert not isdefined(result.outputs.ec_jacobian_images)


def test_okan_jacobian_is_undefined_for_cubic_and_does_not_raise(tmp_path):
    """Cubic is a valid existing mode, so it must degrade, not abort.

    Weighting is on by default, so raising here would newly break runs that
    work today and push users to --no-jacobian-weighting, losing gradwarp and
    SDC weighting as collateral. Silently applying the quadratic formula to
    cubic parameters is what is forbidden.
    """
    transformations = _write_transformations(tmp_path / 'x.txt', [[0.0] * 24] * 2)
    result = OkanQuadraticJacobian(
        transformations_file=transformations,
        reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
        correction_mode='cubic',
    ).run()
    assert not isdefined(result.outputs.ec_jacobian_images)


def test_okan_jacobian_rejects_a_short_row(tmp_path):
    transformations = _write_transformations(tmp_path / 'x.txt', [[0.0] * 20])
    with pytest.raises(ValueError, match='24'):
        OkanQuadraticJacobian(
            transformations_file=transformations,
            reference_image=_write_map(tmp_path / 'ref.nii.gz', 1.0),
            correction_mode='quadratic',
        ).run()
```

```python
# Append to qsiprep/tests/test_interfaces_diffprep.py

@pytest.mark.integration
@pytest.mark.diffprep
def test_reconstructed_transform_reproduces_moteddy(tmp_path, working_dir):
    """The ship gate: our reconstruction must match TORTOISE's own output.

    Reproducing ``_moteddy.nii`` validates the *combined* motion+EC map, which
    is what pins the parameter convention. Splitting the EC determinant out of
    it is licensed separately, by DIFFPREP's motion component being rigid
    (columns 0-5), unlike SHORELine's affine default. Both must hold.
    """
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    from qsiprep.interfaces.jacobian import resample_with_okan_transform

    work = Path(working_dir)
    transformations = next(work.rglob('*_moteddy_transformations.txt'))
    tortoise_output = next(work.rglob('*_moteddy.nii'))
    imported = next(work.rglob('*_proc.nii'))

    ours = resample_with_okan_transform(
        str(imported), str(transformations), str(tmp_path / 'ours.nii.gz')
    )
    mine = np.asanyarray(nb.load(ours).dataobj)
    theirs = np.asanyarray(nb.load(str(tortoise_output)).dataobj)

    inside = theirs > np.percentile(theirs, 60)
    r = np.corrcoef(mine[inside], theirs[inside])[0, 1]
    assert r > 0.99, (
        f'Reconstructed motion+eddy resampling vs TORTOISE _moteddy.nii: '
        f'r={r:.5f}. Below 0.99 means the 24-parameter convention is wrong, '
        'and the EC Jacobian must NOT ship -- fall back to documenting the gap.'
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -k okan -v
```

Expected: `ImportError: cannot import name 'OkanQuadraticJacobian'`.

- [ ] **Step 4: Write the implementation**

Implement in `qsiprep/interfaces/jacobian.py`, following the formula recorded in Step 1:

- `okan_quadratic_jacobian(parameters, shape, affine) -> np.ndarray` — analytic `det ∇φ` of the eddy-current component on the given grid, using columns 6–23 only and ignoring 0–5.
- `resample_with_okan_transform(image, transformations_file, out_path) -> str` — the full 24-parameter resampling used only by the ship-gate test.
- `OkanQuadraticJacobian(SimpleInterface)` — reads the transformations file (reusing `_read_okan_transformations` from `qsiprep.interfaces.tortoise`), validates 24 columns per row, returns `Undefined` for `correction_mode` in `('motion', 'cubic')` with a `LOGGER.warning` naming the gap, and otherwise writes one 3D map per volume.

Then wire it in `qsiprep/workflows/dwi/diffprep.py`:

- add an `ec_jacobian_images` field to the outputnode;
- build the node from `corrected_node.transformations_file`;
- pass **`effective_correction_mode`**, not `correction_mode`. The workflow
  already derives it at `qsiprep/workflows/dwi/diffprep.py:351`, dropping
  `quadratic` to `motion` under `--sloppy`. Passing the configured value would
  attempt quadratic recovery from motion-only transforms on every `--sloppy`
  run;
- wire `reference_image` from `extract_b0s.b0_average` — the EC Jacobian is
  evaluated on the DIFFPREP input grid, which is the grid `_moteddy.nii` is in
  (`qsiprep/interfaces/tortoise.py:1123`). Do not use the output grid.

Add the matching passthrough field to `qsiprep/workflows/dwi/base.py` and
`finalize.py` so it reaches `init_dwi_trans_wf` — the same four-file path the
deleted `sdc_scaling_images` channel used.

`OkanQuadraticJacobian` must also validate that the transformations file has
one row per DWI volume, and derive its identity row from the sourced
convention rather than assuming 24 zeros encode identity (record in Step 1
whether it does; if the convention uses non-zero defaults for the eddy centre,
24 zeros is *not* identity and the first two unit tests above need their
expected rows changed accordingly).

- [ ] **Step 5: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_jacobian.py -v
micromamba run -n linc311 python -m pytest qsiprep/tests -m "not integration" -q 2>&1 | tail -20
```

Then the ship gate, which needs TORTOISE:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_diffprep.py -k reproduces_moteddy -v -m diffprep
```

Expected: unit tests PASS; the gate PASSES with r > 0.99. **If the gate fails, revert this task's `diffprep.py`/`base.py`/`finalize.py` wiring, keep the interface behind its `Undefined` branches, and report** — the EC gap becomes documentation (Task 14) and `UnmodulatedCorrections` (Task 12).

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/jacobian.py qsiprep/workflows/dwi/diffprep.py \
  qsiprep/workflows/dwi/base.py qsiprep/workflows/dwi/finalize.py \
  qsiprep/tests/test_interfaces_jacobian.py qsiprep/tests/test_interfaces_diffprep.py
git commit -m "feat: add TORTOISE eddy-current Jacobian weighting

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Write the weight maps out as derivatives

**Note a deliberate refinement of the spec.** The spec named the derivative `..._space-ACPC_desc-jacobian_dwi.nii.gz`. Use `_dwimap` instead: `io_spec.json` supports `space`/`desc` on the `dwimap` suffix, the file is a map rather than a DWI series, and `_dwi.nii.gz` would be picked up by downstream tools (QSIRecon) globbing for DWI series. Update the spec's Derivatives section to match.

**Files:**
- Modify: `qsiprep/workflows/dwi/derivatives.py`
- Modify: `qsiprep/workflows/dwi/finalize.py` (route the maps to the sink)
- Modify: `qsiprep/tests/data/*_outputs.txt`
- Test: `qsiprep/tests/test_workflows_jacobian.py`, `qsiprep/tests/test_t2w_derivatives.py` pattern

**Interfaces:**
- Consumes: `ApplyJacobianWeights.resampled_weight_images` (Task 5).
- Produces: `_jacobian_sidecar(weight_index, applied, unmodulated, reason) -> dict`, and a `ds_jacobian` sink.

- [ ] **Step 1: Write the failing tests**

```python
# Append to qsiprep/tests/test_workflows_jacobian.py

import json

from bids.layout.writing import build_path

from qsiprep.data import load as load_data


def _patterns():
    return json.loads(load_data('io_spec.json').read_text())['default_path_patterns']


def test_jacobian_derivative_path_renders():
    """Assert the rendered path, not the datasink inputs.

    Entity-level checks are blind to a pattern that silently drops an entity or
    collides with another derivative's name.
    """
    # strict=True: with strict=False an entity the pattern does not support is
    # silently dropped, so the test would pass while the real filename lost it.
    out = build_path(
        dict(
            subject='01',
            datatype='dwi',
            space='ACPC',
            desc='jacobian',
            suffix='dwimap',
            extension='.nii.gz',
        ),
        _patterns(),
        strict=True,
    )
    assert out == 'sub-01/dwi/sub-01_space-ACPC_desc-jacobian_dwimap.nii.gz'


def test_jacobian_derivative_does_not_collide_with_preproc_dwi():
    preproc = build_path(
        dict(
            subject='01', datatype='dwi', space='ACPC', desc='preproc',
            suffix='dwi', extension='.nii.gz',
        ),
        _patterns(),
        strict=False,
    )
    jacobian = build_path(
        dict(
            subject='01', datatype='dwi', space='ACPC', desc='jacobian',
            suffix='dwimap', extension='.nii.gz',
        ),
        _patterns(),
        strict=False,
    )
    assert preproc != jacobian
    assert not jacobian.endswith('_dwi.nii.gz')


def test_jacobian_sidecar_index_is_zero_based_and_full_length():
    from qsiprep.workflows.dwi.derivatives import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0, 0, 1, 0], applied=['gradwarp', 'sdc'],
        unmodulated=[], reason=None,
    )
    assert sidecar['JacobianWeightIndex'] == [0, 0, 1, 0]
    assert sidecar['AppliedCorrections'] == ['gradwarp', 'sdc']
    assert sidecar['UnmodulatedCorrections'] == []
    assert 'UnmodulatedReason' not in sidecar


def test_jacobian_sidecar_records_a_gap():
    from qsiprep.workflows.dwi.derivatives import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0, 0], applied=['gradwarp', 'sdc'],
        unmodulated=['eddy-current'],
        reason='TORTOISE correction_mode=cubic is not supported',
    )
    assert sidecar['UnmodulatedCorrections'] == ['eddy-current']
    assert 'cubic' in sidecar['UnmodulatedReason']


def test_jacobian_sidecar_collapsed_case_is_written_in_full():
    """All-zeros rather than omitted, so consumers need no special case."""
    from qsiprep.workflows.dwi.derivatives import _jacobian_sidecar

    sidecar = _jacobian_sidecar(
        weight_index=[0] * 5, applied=['sdc'], unmodulated=[], reason=None
    )
    assert sidecar['JacobianWeightIndex'] == [0, 0, 0, 0, 0]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -k "jacobian_derivative or sidecar" -v
```

Expected: `ImportError: cannot import name '_jacobian_sidecar'`.

- [ ] **Step 3: Write the implementation**

In `qsiprep/workflows/dwi/derivatives.py`:

```python
def _jacobian_sidecar(weight_index, applied, unmodulated, reason):
    """Sidecar for the Jacobian weight derivative.

    ``weight_index`` is zero-based, one entry per volume of the preprocessed
    DWI series, indexing volumes of the 4D weight file. Repeated maps appear as
    repeated indices, which makes the dedup visible rather than implicit; the
    collapsed single-map case is all zeros, written in full so consumers need
    no special case.
    """
    sidecar = {
        'JacobianWeightIndex': list(weight_index),
        'AppliedCorrections': list(applied),
        'UnmodulatedCorrections': list(unmodulated),
        'Description': (
            'Multiplicative Jacobian intensity modulation applied to the '
            'preprocessed DWI series immediately after spatial resampling. '
            'Dividing by the indexed volume reverses that multiplication at '
            'that point in the pipeline; it does not recover an unmodulated '
            'series, because denoising and bias-field correction run after '
            'resampling and do not commute with it.'
        ),
    }
    if unmodulated and reason:
        sidecar['UnmodulatedReason'] = reason
    return sidecar
```

Add the sink alongside the existing ones, following the `ds_cnr_map_t1` shape:

```python
    ds_jacobian = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            desc='jacobian',
            suffix='dwimap',
            extension='.nii.gz',
            compress=True,
        ),
        name='ds_jacobian',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
```

**The derivative contract, stated so it is not ambiguous.** The file records
*the weights QSIPrep applied*, nothing else. Consequences:

- When `ComposeJacobianWeights` produced no weights (`jacobian_weight_images`
  is `Undefined` -- e.g. `--hmc-method eddy` with TOPUP and no gradwarp, where
  every applied modulation is internal to `eddy`), **no file is written**. Use
  `DerivativesMaybeDataSink` (`qsiprep/interfaces/bids.py:245`), which no-ops
  on an undefined input. Do not synthesize a map of ones: that would assert
  "we modulated by 1", which is false -- the modulation happened inside `eddy`
  and QSIPrep has no map for it.
- `--no-jacobian-weighting` likewise writes nothing.
- This narrows the spec's "always" to "whenever QSIPrep applied weights".
  Update the spec's Derivatives section to say so.

**Stack and route.** Add to `qsiprep/interfaces/jacobian.py`:

```python
class _StackJacobianWeightsInputSpec(BaseInterfaceInputSpec):
    weight_images = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='unique output-grid weight maps, first-appearance order',
    )
    weight_index = traits.List(
        traits.Int(), mandatory=True, desc='per-volume index into weight_images'
    )


class _StackJacobianWeightsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='3D if one unique map, else 4D')
    meta_dict = traits.Dict(desc='sidecar for the derivative')


class StackJacobianWeights(SimpleInterface):
    # Stack the unique weight maps and build the sidecar. 3D when every volume
    # shares one map, 4D otherwise. The index is written out in full either
    # way, so consumers need no special case.

    input_spec = _StackJacobianWeightsInputSpec
    output_spec = _StackJacobianWeightsOutputSpec

    def _run_interface(self, runtime):
        from ... import config
        from ..workflows.dwi.derivatives import _jacobian_sidecar

        images = [nb.load(path) for path in self.inputs.weight_images]
        out_file = os.path.join(runtime.cwd, 'jacobian_weights.nii.gz')
        if len(images) == 1:
            data = np.asanyarray(images[0].dataobj)
        else:
            data = np.stack([np.asanyarray(img.dataobj) for img in images], axis=-1)
        nb.Nifti1Image(
            data.astype('float32'), images[0].affine, images[0].header
        ).to_filename(out_file)

        self._results['out_file'] = out_file
        self._results['meta_dict'] = _jacobian_sidecar(
            weight_index=self.inputs.weight_index,
            applied=config.workflow.jacobian_applied_corrections,
            unmodulated=config.workflow.jacobian_unmodulated_corrections,
            reason=config.workflow.jacobian_unmodulated_reason,
        )
        return runtime
```

`AppliedCorrections` / `UnmodulatedCorrections` / the reason have no producing
interface, so they are recorded on the config object at workflow-build time --
which is when they are known, since they follow from the backend, the fieldmap
branch, the effective eddy method and the TORTOISE correction mode. Add three
fields to `qsiprep/config.py` beside `jacobian_weighting`:

```python
    jacobian_applied_corrections = []
    """Corrections whose determinants QSIPrep multiplied in."""

    jacobian_unmodulated_corrections = []
    """Corrections that ran but were not Jacobian-modulated."""

    jacobian_unmodulated_reason = None
    """Why, in prose, for the sidecar."""
```

`init_fsl_hmc_wf` and `init_diffprep_wf` append to these as they decide (the
`lsr` branch, the `cubic` branch, a failed Okan gate); `init_dwi_trans_wf`
appends `gradwarp` and `sdc` when it wires those inputs.

**Routing**, all four boundaries -- a missed one gives a silently absent
derivative:

1. `init_dwi_trans_wf`: add `jacobian_weights` and `jacobian_weight_index` to
   its outputnode, connected from `scale_dwis`.
2. `init_dwi_finalize_wf`: the same two fields on its outputnode, forwarded
   from `transform_dwis_t1`.
3. `init_dwi_derivatives_wf`: the same two fields on its inputnode.
4. Inside the derivatives workflow: `StackJacobianWeights` -> `ds_jacobian`,
   with `meta_dict` connected to the sink.

Gate the whole block on `config.workflow.jacobian_weighting`.

- [ ] **Step 4: Update the expected-output lists**

```bash
grep -rln "desc-preproc_dwi.nii.gz" qsiprep/tests/data/
```

For each list, add the new derivative line in sorted position. The paths are subject- and session-specific, so mirror the neighbouring `space-ACPC` entries in each file exactly.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_jacobian.py -v
micromamba run -n linc311 python -m pytest qsiprep/tests -m "not integration" -q 2>&1 | tail -20
```

Expected: PASS, matching the Task 0 baseline.

- [ ] **Step 6: Update the spec's naming**

Change the Derivatives section of `docs/superpowers/specs/2026-09-17-jacobian-weighting-design.md` from `_desc-jacobian_dwi.nii.gz` to `_desc-jacobian_dwimap.nii.gz`, with the reason. Do not stage it.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/workflows/dwi/derivatives.py qsiprep/workflows/dwi/finalize.py \
  qsiprep/tests/test_workflows_jacobian.py qsiprep/tests/data/
git commit -m "feat: write Jacobian weight maps as a derivative

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: The conservation oracle

The correctness test for the whole feature, and the one that catches a sign or direction error. Jacobian modulation redistributes signal without changing its total, so summed in-mask signal after weighting must match the unweighted raw series.

**Files:**
- Create: `qsiprep/tests/test_jacobian_conservation.py`

**Interfaces:**
- Consumes: `ComposeJacobianWeights` (Task 4), `jacobian_determinant` (Task 3).
- Produces: nothing (validation).

- [ ] **Step 1: Write the test**

```python
"""Total-signal conservation: the oracle for the modulation direction.

For a pull-back map phi, corrected intensity is
``I_corr(x) = I_dist(phi(x)) . |det grad phi(x)|``, so

    integral I_corr = integral I_dist(phi(x)) |det grad phi| dx
                    = integral I_dist(y) dy

Conservation is therefore exact up to interpolation error, and it fails if the
weight is divided rather than multiplied, or if the field direction is
inverted. This is what distinguishes a correct implementation from one that is
merely self-consistent.

Requires ANTs; skips locally, runs in CircleCI's ``unit_tests`` job.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces import ants

from qsiprep.interfaces.jacobian import jacobian_determinant


def _compressing_field(path, shape=(24, 24, 24), amplitude=3.0):
    """A field that compresses along x, so det is well away from 1."""
    coords = np.stack(
        np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in shape], indexing='ij'),
        axis=-1,
    )
    data = np.zeros(shape + (1, 3), dtype='float32')
    # Smooth, monotone displacement along the first axis.
    data[..., 0, 0] = amplitude * np.sin(np.pi * coords[..., 0])
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


def _blob(path, shape=(24, 24, 24)):
    coords = np.stack(
        np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing='ij'),
        axis=-1,
    )
    radius = np.linalg.norm(coords, axis=-1)
    data = (100.0 * np.exp(-3.0 * radius**2)).astype('float32')
    nb.Nifti1Image(data, np.eye(4)).to_filename(str(path))
    return str(path)


def test_multiplying_by_the_jacobian_conserves_total_signal(tmp_path):
    if shutil.which('antsApplyTransforms') is None:
        pytest.skip('antsApplyTransforms required for this test')
    if shutil.which('CreateJacobianDeterminantImage') is None:
        pytest.skip('CreateJacobianDeterminantImage required for this test')

    source = _blob(tmp_path / 'source.nii.gz')
    field = _compressing_field(tmp_path / 'field.nii.gz')

    warped = str(tmp_path / 'warped.nii.gz')
    xfm = ants.ApplyTransforms(
        input_image=source,
        reference_image=source,
        transforms=[field],
        output_image=warped,
        interpolation='LanczosWindowedSinc',
        dimension=3,
        float=True,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    xfm.run()

    determinant = jacobian_determinant(field, str(tmp_path / 'det.nii.gz'))

    raw_total = float(np.asanyarray(nb.load(source).dataobj).sum())
    warped_data = np.asanyarray(nb.load(warped).dataobj)
    weights = np.asanyarray(nb.load(determinant).dataobj)

    modulated_total = float((warped_data * weights).sum())
    unmodulated_total = float(warped_data.sum())
    divided_total = float((warped_data / weights).sum())

    multiply_error = abs(modulated_total - raw_total)
    divide_error = abs(divided_total - raw_total)
    skip_error = abs(unmodulated_total - raw_total)

    assert modulated_total == pytest.approx(raw_total, rel=0.02), (
        f'Modulated total {modulated_total:.1f} should match the raw total '
        f'{raw_total:.1f}. If it is off by roughly the square of the expected '
        'factor, the weight is being applied twice.'
    )

    # A tolerance alone does not discriminate: for a mild deformation both
    # multiply and divide can land inside it. Require multiplying to beat both
    # alternatives by a clear margin, which is what actually pins the
    # convention.
    assert multiply_error * 3 < divide_error, (
        f'Multiplying by the Jacobian left an error of {multiply_error:.1f}; '
        f'dividing left {divide_error:.1f}. Multiplying must be decisively '
        'better. If they are comparable, this deformation is too mild to '
        'discriminate -- raise the amplitude in _compressing_field.'
    )
    assert multiply_error * 3 < skip_error, (
        f'Multiplying left an error of {multiply_error:.1f}; not weighting at '
        f'all left {skip_error:.1f}. If those are comparable the test proves '
        'nothing -- raise the amplitude in _compressing_field.'
    )
```

- [ ] **Step 2: Run the test**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_jacobian_conservation.py -v
docker run --rm -v /mnt/c/Users/tsalo/Documents/linc/qsiprep:/src \
  --entrypoint pytest pennlinc/qsiprep:test \
  /src/qsiprep/tests/test_jacobian_conservation.py -v
```

Expected: SKIP locally, PASS in the image. **If it fails in the direction of over-correction, stop** — the multiply/divide decision in `Global Constraints` is wrong and the spec's direction derivation needs revisiting before anything ships.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_jacobian_conservation.py
git commit -m "test: add the total-signal conservation oracle

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: Documentation and methods boilerplate

**Files:**
- Modify: `docs/preprocessing.rst` (~line 709 gradwarp section, ~line 817 FSL interpolation text)
- Modify: `docs/usage.rst`
- Modify: `docs/api.rst`
- Modify: `qsiprep/data/boilerplate.bib`
- Modify: `qsiprep/workflows/dwi/gradwarp.py` (`gradwarp_boilerplate`)
- Modify: `qsiprep/interfaces/eddy.py:442` (`boilerplate_from_eddy_config` — it
  lives in the interfaces module, not in `fsl.py`, which only calls it)
- Test: `qsiprep/tests/test_workflows_gradwarp.py` (boilerplate assertions)

**Interfaces:**
- Consumes: `eddy_modulates_distortion` (Task 6), `config.workflow.jacobian_weighting`.
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing boilerplate tests**

```python
# Append to qsiprep/tests/test_workflows_gradwarp.py

def test_boilerplate_states_modulation_when_enabled():
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.jacobian_weighting = True
    text = gradwarp_boilerplate('3D', 'metadata')
    assert 'Jacobian' in text


def test_boilerplate_states_the_absence_when_disabled():
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.jacobian_weighting = False
    text = gradwarp_boilerplate('3D', 'metadata')
    assert 'without Jacobian' in text or 'no Jacobian' in text


def test_eddy_boilerplate_flags_lsr_as_unmodulated():
    from qsiprep.interfaces.eddy import boilerplate_from_eddy_config

    text = boilerplate_from_eddy_config(
        {'method': 'lsr', 'flm': 'quadratic', 'slm': 'linear'}, 'epi', pepolar_method='topup'
    )
    assert 'not Jacobian-modulated' in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_gradwarp.py -k boilerplate -v
```

Expected: FAIL — the strings are absent.

- [ ] **Step 3: Write the documentation and boilerplate**

Add to `docs/preprocessing.rst`, after the gradwarp transform-chain discussion:

```rst
.. _jacobian_weighting:

Jacobian intensity modulation
-----------------------------

Correcting a spatial distortion moves signal between voxels, so the corrected
image must also be rescaled by the local volume change -- the Jacobian
determinant of the correction -- or regions the acquisition compressed stay
artificially bright. *QSIPrep* applies this modulation for gradient
nonlinearity, susceptibility distortion and eddy-current correction. It is
**not** applied for head motion (a rigid or affine realignment is not a
measured volume change), nor for coregistration, the intramodal template, or
template-space normalization: modulating by a spatial-normalization warp is
VBM-style volume modulation, which would corrupt DWI signal intensities and
every model fitted to them.

Which component performs the modulation depends on the backend:

============================== ==========================================
Correction                     Modulated by
============================== ==========================================
Gradient nonlinearity          *QSIPrep*
Susceptibility (TOPUP)         ``eddy``, internally
Susceptibility (DRBUDDI)       *QSIPrep*
Susceptibility (GRE, SyN)      *QSIPrep*
Susceptibility (T2Wreg)        *QSIPrep*
Eddy current (``eddy``)        ``eddy``, internally
Eddy current (DIFFPREP)        *QSIPrep*
============================== ==========================================

``--no-jacobian-weighting`` disables the modulation *QSIPrep* applies. It
cannot disable ``eddy``'s: ``eddy``'s ``--resamp`` accepts only ``jac`` or
``lsr``, and ``lsr`` requires exactly two opposite-polarity acquisitions, so on
``--hmc-method eddy`` the eddy-current and TOPUP susceptibility modulation is
internal to ``eddy`` and unaffected by the flag. Conversely, if you supply
``--eddy-config`` with ``"method": "lsr"``, those two corrections are *not*
Jacobian-modulated and *QSIPrep* cannot retrofit it -- ``eddy`` has already
baked its resampling in. The run warns, and the methods boilerplate says so.

The weight maps are written out as
``*_space-ACPC_desc-jacobian_dwimap.nii.gz``, with a sidecar giving
``JacobianWeightIndex`` (one zero-based entry per DWI volume, indexing volumes
of the weight file), ``AppliedCorrections``, and ``UnmodulatedCorrections``
with a reason where coverage is partial. Dividing the preprocessed series by
the indexed weight volume recovers the unmodulated data.
```

Update the "two total interpolations" paragraph (~line 817) to note that the
FSL path's eddy-current and susceptibility modulation happens inside ``eddy``
at the first interpolation, while gradwarp's happens at the second.

Add the flag to `docs/usage.rst` and the two new interfaces to `docs/api.rst`.

Extend `gradwarp_boilerplate` and `boilerplate_from_eddy_config` to state
whether modulation was applied, reading `config.workflow.jacobian_weighting`
and `eddy_modulates_distortion` respectively. Add a citation for the modulation
rationale to `qsiprep/data/boilerplate.bib`.

- [ ] **Step 4: Run the tests and build the docs**

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_gradwarp.py -v
micromamba run -n linc311 python -m sphinx -b html docs docs/_build/html -q 2>&1 | tail -20
```

Expected: tests PASS; no new Sphinx warnings about the added references.

- [ ] **Step 5: Commit**

```bash
git add docs/preprocessing.rst docs/usage.rst docs/api.rst \
  qsiprep/data/boilerplate.bib qsiprep/workflows/dwi/gradwarp.py \
  qsiprep/interfaces/eddy.py qsiprep/tests/test_workflows_gradwarp.py
git commit -m "docs: document Jacobian intensity modulation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: Integration validation

**Files:**
- Modify: `qsiprep/tests/test_cli.py` (assertions in the existing marked tests)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Run the branch-covering integration markers**

Existing markers already cover every affected path; no new CI jobs are needed.

```bash
for marker in dsdti_topup dsdti_synfmap maternal_brain_project forrest_gump \
              drbuddi_rpe diffprep diffprep_drbuddi; do
  # A separate output directory per marker: with one shared /tmp/out, files
  # from an earlier backend make a later one look like it produced output.
  docker run --rm -v /mnt/c/Users/tsalo/Documents/linc/qsiprep:/src \
    -v "/tmp/out/$marker:/out" --entrypoint pytest pennlinc/qsiprep:test \
    /src/qsiprep/tests -m "$marker" -v --output_dir=/out 2>&1 | tail -30
done
```

Expected derivative counts per marker, so an empty result is a failure rather
than a silent pass:

| Marker | Jacobian derivative expected? |
|---|---|
| `dsdti_topup` | No — eddy+TOPUP, no gradwarp, so every modulation is internal to `eddy` |
| `dsdti_synfmap` | Yes — SyN warp |
| `maternal_brain_project` | Yes — GRE fieldmap |
| `forrest_gump` | Yes — GRE fieldmap |
| `drbuddi_rpe` | Yes — two unique maps (up/down) |
| `diffprep` | Yes if the Okan gate passed, else no |
| `diffprep_drbuddi` | Yes |

Per the CircleCI note: step output is truncated at ~400 KB. If a run's log is
cut, recover the failures by replaying `check_generated_files` against the
stored `/tmp/out` artifacts rather than re-running the whole marker.

- [ ] **Step 2: Assert the weight map exists and is sane**

For each marker's output tree, confirm the derivative was written and the
sidecar is well-formed:

```bash
micromamba run -n linc311 python - <<'PY'
import json
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

found = sorted(Path('/tmp/out').rglob('*_desc-jacobian_dwimap.nii.gz'))
if not found:
    sys.exit(
        'No Jacobian derivatives found. An empty glob is not a pass -- either '
        'the sink is not wired, or this marker legitimately produces none '
        '(eddy+TOPUP with no gradwarp), in which case assert that explicitly '
        'for this marker rather than iterating over nothing.'
    )
print(f'{len(found)} Jacobian derivative(s)')

for nii in found:
    sidecar = json.loads(nii.with_suffix('').with_suffix('.json').read_text())
    data = np.asanyarray(nb.load(str(nii)).dataobj)
    index = sidecar['JacobianWeightIndex']
    n_weight_volumes = data.shape[3] if data.ndim == 4 else 1
    assert min(index) == 0, nii
    assert max(index) < n_weight_volumes, nii
    assert np.isfinite(data).all(), nii
    print(nii.name, 'volumes:', n_weight_volumes, 'median:', float(np.median(data)))
    print('  applied:', sidecar['AppliedCorrections'],
          'unmodulated:', sidecar['UnmodulatedCorrections'])
PY
```

Expected: one entry per run, medians near 1, indices in range.

- [ ] **Step 3: Record the DRBUDDI numeric delta**

The DRBUDDI change alters output values. Diff the preprocessed series against a
pre-change run of `drbuddi_rpe` and record the in-mask median ratio and 95th
percentile absolute difference in the PR body. This is the evidence that the
values change was measured rather than discovered by a user.

- [ ] **Step 4: Commit any assertion changes**

```bash
git add qsiprep/tests/test_cli.py
git commit -m "test: assert Jacobian weight derivatives in integration runs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Walking the spec section by section: Problem table → Tasks 7/8/10/11 address every "No" row. Decision 1 (scope) → Task 7's exclusion tests. Decision 2 (default + flag) → Task 6. Decision 3 (DRBUDDI) → Task 10, gated by Task 2. Decision 4 (derivatives) → Task 12. Decision 5 (TORTOISE EC) → Task 11. External eddy constraint → Task 6's helper plus Task 14's boilerplate. Correction direction → Task 13. HMC coordinate safety → Task 1 (convention) plus Task 7 (wiring exclusions). `ComposeJacobianWeights` → Tasks 3–4. Coordinate domains and guard → Task 3's `validate_field_geometry` and Task 4's entry validation. Per-backend contract table → Task 7 (GRE/SyN/T2Wreg/DRBUDDI fall out of `fieldwarps`), Task 9 (TOPUP invariant), Task 11 (DIFFPREP EC). CLI → Task 6. Derivatives + sidecar schema → Task 12. Testing section → Tasks 1, 3, 4, 9, 12, 13, 15. Documentation → Task 14. No gaps.

**Type consistency.** `jacobian_weight_images` is the output of `ComposeJacobianWeights` (Task 4) and the input of `ApplyJacobianWeights` (Task 5) — same name both sides. `ec_jacobian_images` is the output of `OkanQuadraticJacobian` (Task 11), the inputnode field added in Task 7, and the `ComposeJacobianWeights` input in Task 4 — consistent. `resampled_weight_images` is produced in Task 5 and consumed in Task 12. `jacobian_determinant` is defined in Task 3 and used in Tasks 2, 4 and 13. `weight_key`/`multiply_maps`/`compose_fields`/`validate_field_geometry`/`check_weight_map` are all defined in Task 3 before Task 4 uses them. `effective_eddy_resampling_method`/`eddy_modulates_distortion` are defined in Task 6 and used in Task 14. `_jacobian_sidecar` is defined and used within Task 12.

**Ordering dependencies.** The document order is not the execution order in
one place: **Task 2 executes after Task 3 and before Task 4**, because it needs
`jacobian_determinant` and because it must gate the DRBUDDI decision *before*
Tasks 5 and 7 retire the legacy scaling-image path — otherwise its stop-and-
re-plan outcome would be unreachable. Task 2 carries that banner at its head.

Otherwise: Task 1 → Task 4 (convention). Task 3 → Tasks 2, 4, 13. Task 4 →
Task 7. Task 5 → Tasks 7, 12. Task 6 → Tasks 7, 14. Task 11's formula-sourcing
step precedes its own code. Task 15 is last.

**Execution order in full:** 0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.

**Four stop-and-report gates**, each with a defined fallback rather than a
workaround, and each positioned before the work it protects: Task 1 Step 3
(ordering convention), Task 2 Step 2 (DRBUDDI `_JAC`, run before any removal),
Task 11 Step 5 (Okan ship gate), Task 13 Step 2 (conservation, which blocks
everything).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-17-jacobian-weighting.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
