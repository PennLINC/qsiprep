# DWI Output Spaces (`anat`, `dwiref`, native resolutions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/superpowers/specs/2026-10-02-dwi-output-spaces-design.md`:
- `acpc`, `anat` and `dwiref` as DWI output spaces;
- `--output-spaces` optional, defaulting to `acpc:res-nativemin`;
- native resolutions determined per output, with an error when an output combines runs of
  different voxel sizes;
- anisotropic output;
- `res-native` written without `res-`;
- anatomical derivatives at the anatomical's own voxel size.

**Architecture:**
- **One spec type throughout.** The ACPC-only fan-out (`acpc_specs`) becomes a fan-out over
  every DWI-carrying spec (`dwi_specs`), keyed by (space, resolution).
- **Transform chains per spec.** Each spec selects its chain: `acpc` as today, `anat` with one
  trailing rigid, `dwiref` with coregistration dropped. Its grid is always axis-aligned LPS.
- **Unchanged:** estimation (HMC, SDC, coregistration, normalization), and the anatomical
  images *fed to* DWI processing. Only resampling and the derivatives fan out.

**Tech stack:** Python 3.11, nipype, niworkflows (`GenerateSamplingReference`), ANTs, AFNI,
pybids path patterns, pytest, ruff 0.15.21.

## Prerequisites (check before Task 0.1)

- [ ] `output-spaces-new` is merged into `main`, along with the bug-fix PR from
      `fix-transform-and-merge-qc-bugs`. Create this branch with
      `git switch --no-track -c dwi-output-spaces origin/main`. **`--no-track` is required**:
      without it GitHub Desktop pushes to `main`.
- [ ] QSIRecon selects ACPC outputs by `res-`, or will in the same release. `0dcc27e` already
      depends on this, and Task 1.2 makes `res-nativemin` the default label.
- [ ] The Codex adversarial review of the spec has run. It is blocked on workspace credits as of
      2026-10-02. Fix any valid findings in the spec before starting; tasks below cite the spec's
      decisions by number.

## Correction to the spec, applied here

**Phase 5 did not leave estimation unchanged as first written.** The ACPC anatomicals that
`init_anat_preproc_wf` exposes (`t1_preproc`, `t1_brain`, `t1_mask`, `t1_seg`, `t1_aseg`;
`anatomical/volume.py`, the `rigid_acpc_resample_*` nodes) are *inputs* to DWI processing:

- the b=0→anatomical coregistration;
- DIFFPREP's T2Wreg and DRBUDDI structurals (`dwi/diffprep.py`);
- eddy/TOPUP's SynB0 (`dwi/fsl.py`);
- the distortion-group merge and the reports.

Moving them to a different grid would change those registrations numerically. **Tasks 5.1–5.2
therefore keep the internal anatomicals on the anchor grid** and resample *only the written
derivatives* onto the anatomical-resolution grid. The spec's Phase 5 text was corrected in the
same commit as this plan.

## Global constraints

- Run everything through micromamba in WSL:
  `MSYS_NO_PATHCONV=1 wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 <command>"`.
- Lint touched files with `ruff check` and `ruff format` (ruff 0.15.21, numpydoc docstrings:
  imperative summary line; "Test that ..." for tests).
- **Never** `git stash`, `git add -A`, or create a git worktree or other sibling folder. Stage
  files by name. For before/after comparisons, copy files into the scratchpad and restore them,
  or use `git show <rev>:<path>`.
- **Never edit `docs/changes.md`.**
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Baseline:** in `linc311` with qsiplan 0.4.2, the suite has 4 failures and 5 errors, all from
  missing binaries (ANTs, FreeSurfer, MRtrix, trxscan) or the truncated `template_qc` fixture.
  Record it before Task 0.1:
  ```bash
  micromamba run -n linc311 python -m pytest qsiprep/tests -q -p no:cacheprovider -n 8 \
      | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort > /tmp/baseline_fail.txt
  ```
  After every task, rerun into `/tmp/task_fail.txt` and check that
  `comm -13 /tmp/baseline_fail.txt /tmp/task_fail.txt` is empty.
- **Prove each new test bites.** For every behaviour change, temporarily restore the previous
  source (copy it to the scratchpad, `git show HEAD:<path> > <path>`, run, copy back) and confirm
  the new test fails. Record which test failed in the commit message body.
- **Acpc-only runs must not change** until Task 1.2 changes the default, and then only as the
  spec says. `qsiprep/tests/test_output_spaces_naming.py` guards the filenames, and the
  integration manifests (`qsiprep/tests/data/*_outputs.txt`) guard the file sets.

## File map

| File | Changes in tasks |
|---|---|
| `qsiprep/utils/spaces.py` | 0.1, 1.1, 2.1, 2.2 |
| `qsiprep/cli/parser.py` | 1.1, 1.2, 4.4 |
| `qsiprep/config.py` | 1.2 |
| `qsiprep/workflows/base.py` | 0.2, 1.3, 1.4, 3.3, 4.3, 4.4 |
| `qsiprep/workflows/dwi/finalize.py` | 0.2, 0.4, 1.3, 3.3, 4.2 |
| `qsiprep/workflows/dwi/derivatives.py` | 0.2, 1.3 |
| `qsiprep/workflows/dwi/distortion_group_merge.py` | 0.2, 1.3 |
| `qsiprep/workflows/dwi/resampling.py` | 0.4, 2.2 |
| `qsiprep/workflows/dwi/util.py` (`add_synb0_outputs`) | 0.2, 1.4 |
| `qsiprep/workflows/anatomical/volume.py` | 0.1, 1.3, 2.1, 2.2, 3.1, 3.2, 5.1, 5.2 |
| `qsiprep/interfaces/gradients.py` (`ComposeTransforms`) | 0.4 |
| `qsiprep/interfaces/fmap.py` (`ApplyJacobianWeights`) | 0.4 |
| `qsiprep/interfaces/images.py` (`ChooseInterpolator`) | 2.2 |
| `qsiprep/interfaces/anatomical.py` (`VoxelSizeChooser`) | 1.3, 2.2 |
| `qsiprep/data/reports-spec.yml` | 0.3 |
| `docs/running.rst`, `docs/outputs.rst`, `docs/upgrading.rst` | 1.2, 2.2, 3.3, 4.4, 5.2 |
| Tests | every task |

---

## Phase 0 — Groundwork (no behaviour change)

### Task 0.1: Replace `SpaceSpec.standard` with `is_template` and `writes_dwi`

`standard` is `space != ACPC` (`utils/spaces.py:63-65`). `anat` and `dwiref` would be
misclassified as templates.

**Files:** `qsiprep/utils/spaces.py`. Call sites:
- `cli/parser.py:1203`;
- `utils/spaces.py:254`;
- `workflows/anatomical/volume.py:387, 1728, 1848`;
- `workflows/base.py:272-273`;
- `workflows/dwi/util.py:63`.

Tests go in `qsiprep/tests/test_utils_spaces.py`.

- [ ] **Step 1: Failing test**

```python
@pytest.mark.parametrize(
    ('token', 'is_template', 'writes_dwi'),
    [
        ('acpc:res-2mm', False, True),
        ('MNI152NLin2009cAsym', True, False),
        ('MNI152NLin6Asym:res-2', True, False),
    ],
)
def test_space_kind_properties(token, is_template, writes_dwi):
    """Test that each space reports whether it is a template and whether it carries DWI."""
    (spec,) = parse_space_token(token)
    assert spec.is_template is is_template
    assert spec.writes_dwi is writes_dwi
```

- [ ] **Step 2: Implement**

```python
NATIVE_FRAMES = ('anat', 'dwiref')  # parsed from Task 1.1 on
DWI_SPACES = (ACPC, *NATIVE_FRAMES)

@property
def is_template(self) -> bool:
    return self.space not in DWI_SPACES

@property
def writes_dwi(self) -> bool:
    return self.space in DWI_SPACES
```

Remove `standard` and switch every call site above. Use `is_template` where the code means "a
TemplateFlow space" (registration, anatomical derivatives, `templateflow_kwargs`), and
`writes_dwi` where it means "gets DWI" (`base.py:272`, `util.py:63`). Keep the local names
`acpc_specs` / `standard_specs` until Task 0.2.

- [ ] **Step 3:** Run the suite against the baseline, then lint.
- [ ] **Step 4: Commit**: `refactor: tell template spaces from DWI spaces by name, not by "not acpc"`.

### Task 0.2: Fan out over `dwi_specs` and name every DWI sink's space from its spec

**Files:**
- `workflows/base.py` (`acpc_specs` at `:272`, the merge construction at `:520-555`, the
  `init_dwi_finalize_wf` call at `:768`, SynB0 at `:793`);
- `workflows/dwi/finalize.py` (the `acpc_specs` parameter and the fan-out loop);
- `workflows/dwi/derivatives.py`;
- `workflows/dwi/distortion_group_merge.py`;
- `workflows/dwi/util.py`.

**Interfaces:**
- `init_dwi_finalize_wf(..., dwi_specs, ...)` replaces `acpc_specs`.
- `init_dwi_derivatives_wf(..., space='ACPC', ...)` and
  `init_distortion_group_merge_wf(..., space='ACPC', ...)` gain a `space` argument. Its default
  keeps the current filenames.
- A helper in `utils/spaces.py`:

```python
def bids_space(spec) -> str:
    """Return the ``space-`` entity value for a DWI spec (``ACPC`` for ``acpc``)."""
    return {'acpc': 'ACPC'}.get(spec.space, spec.space)
```

`dwiref` resolves to its level in Task 4.3. Until then only `acpc` reaches this helper.

- [ ] **Step 1: Failing tests** in `test_output_spaces_naming.py`:
  - `test_dwi_derivatives_take_their_space_from_the_caller`: build
    `init_dwi_derivatives_wf(..., space='anat', resolution=<2mm>)` and assert every sink that
    had `space='ACPC'` now has `space='anat'`.
  - The same for `_merge_wf(..., space='anat')`, adding a `space` parameter to the helper in
    `test_workflows_native.py`.
- [ ] **Step 2: Implement.**
  - Replace every hardcoded `space='ACPC'` on a DWI-grid sink with the passed `space`:
    11 in `derivatives.py`; `ds_series_qc`, `ds_merged_sidecar`, `ds_carpetplot` and
    `ds_grad_dev` in `finalize.py`; `ds_merged_sidecar` and `ds_series_qc` in the merge workflow.
  - Do **not** touch `ds_dwiref_acpc`, `ds_run_dwiref`, the coregistration transforms, or
    anatomical sinks.
  - Rename `acpc_specs` → `dwi_specs` in finalize, `base.py` and the merge construction.
  - Node-name suffixes become `f'_{bids_space(spec)}res{label}'` only when there is more than
    one DWI spec, and stay `f'_res{label}'` when all specs are `acpc`, so existing working
    directories still match. Put this in one helper, `_spec_suffix(spec, dwi_specs)`, in
    `finalize.py` and import it in `base.py`.
- [ ] **Step 3:** Update `test_single_subject_wf_builds_a_merge_workflow_per_resolution`
      (`test_workflows_native.py`) for the renamed loop variable.
- [ ] **Step 4:** Run the suite against the baseline. Integration manifests must not change.
- [ ] **Step 5: Commit**: `refactor: fan out over every DWI spec and name each sink's space from it`.

### Task 0.3: Keep per-space reportlets apart

Figure sinks carry `res-` (via `sink_entities`) but no `space`. Two DWI specs in different
spaces at the same resolution would collide. `reports-spec.yml` filters DWI figures by `desc`
only, so every space's figures land in one section.

**Files:** `workflows/dwi/finalize.py` (`sink_entities` built from `res_entities`),
`workflows/dwi/util.py` (`init_dwi_reference_wf` reportlet), `workflows/dwi/distortion_group_merge.py`,
`data/reports-spec.yml`, `qsiprep/tests/test_output_spaces_naming.py`.

- [ ] **Step 1: Failing test.** Build finalize with two specs of different spaces and one
      resolution, using a test-only `SpaceSpec(space='anat', ...)` constructed directly.
      Collect figure entities and assert `assert_no_collisions` on the rendered paths. Fails
      because the figures share names.
- [ ] **Step 2: Implement.** Add `space=bids_space(spec)` to `sink_entities` whenever there is a
      non-`acpc` DWI spec. With only `acpc` specs, leave figures unchanged so existing reports
      keep their names. In `reports-spec.yml`, add `space` to the DWI per-output figure filters
      with `regex_search: true` and `space: .*`, matching the anatomical normalization entry.
      Check every DWI figure entry: `qcwarnings`, `biascorrpost.*`, `b0ref`/`resampledb0ref`,
      `sdcwarp`, `sdcrefinement`, `samplingscheme`.
- [ ] **Step 3:** Run the report-spec tests (`test_reports.py`) and the suite.
- [ ] **Step 4: Commit**: `fix: give per-space reportlets their own names and report sections`.

### Task 0.4: Choose transform stages per spec, and remove the dead to-template hook

**Files:**
- `interfaces/gradients.py` (`ComposeTransforms`, `_TRANSFORM_STAGES` at `:398-405`, the
  shortcut at `:557`, the hook at `:534-539`);
- `interfaces/fmap.py` (`ApplyJacobianWeights`, the stack at `:1169-1172`);
- `workflows/dwi/resampling.py` (`init_dwi_trans_wf`);
- `workflows/dwi/finalize.py`;
- tests in `test_sdc_warp_derivative.py`, `test_workflows_jacobian.py`.

**Interfaces:**
- `ComposeTransforms` gains:
  - `drop_stages = traits.List(traits.Enum(*_TRANSFORM_STAGES))`, default empty;
  - `trailing_transforms = InputMultiObject(File(exists=True))`, applied after every stage.
- It loses `t1_2_mni_forward_transform`, which asserts a legacy `[affine, warp]` pair and is never
  connected (`resampling.py:185`).
- `ApplyJacobianWeights` gains the same two inputs and builds its transport stack from them,
  instead of the hardcoded `[dwiref_affine, dwiref_warp, coreg]`.
- `init_dwi_trans_wf(..., drop_stages=(), trailing_transforms=None)` passes them through to both.

- [ ] **Step 1: Failing tests.**
  - `ComposeTransforms._stage_names(included, drop)` (a new classmethod, like
    `_sdc_warp_stage_names`) returns the stages minus `drop`.
  - Construction tests (no ANTs needed; follow
    `test_compose_transforms_exposes_the_sdc_warp_subchain`):
    - only coregistration present and `drop_stages=['b=0 to T1w']` gives
      `out_warps == ['identity']`-equivalent output, i.e. **an identity path, not the coreg
      shortcut**;
    - a `trailing_transforms` file appears last in `transform_lists[0]`.
  - `ApplyJacobianWeights` with `drop_stages=['b=0 to T1w']` builds a stack without coreg.
    Test it via a new pure helper `_transport_stack(...)` extracted from `_run_interface`.
- [ ] **Step 2: Implement.**
  - Filter `image_transform_names` by `drop_stages` before composition.
  - Append `trailing_transforms` after the last stage, as one more ANTs-ordered entry.
  - When no stage remains, return `['identity']` lists of the right length, mirroring
    `sdc_warp_transforms`' fallback.
  - `compose_affines` must include the trailing rigid: it is a `.mat`, so the existing
    `'.nii' not in` filter keeps it.
  - `_sdc_warp_stage_names` and `_hmc_corrected_stage_names` (the latter from the bug-fix PR)
    must also honour `drop_stages`, or the SDC map and the CNR map would be conjugated through a
    dropped coregistration.
- [ ] **Step 3:** For `acpc` specs, finalize passes nothing new. Assert in
      `test_workflows_jacobian.py` that `transform_lists` for a default finalize build are
      unchanged.
- [ ] **Step 4:** Suite against the baseline, then lint.
- [ ] **Step 5: Commit**: `feat: select resampling stages per output space; drop the dead to-template hook`.

---

## Phase 1 — Defaults, grammar, native resolutions

### Task 1.1: Grammar for `anat`, `dwiref` and `native`; "≥1 DWI space"; DWI-space default resolution

**Files:**
- `utils/spaces.py`: `_NATIVE_RE`; `_parse_resolution` (`:130-164`); `parse_space_token`
  (`:197-245`); `parse_output_spaces` (`:265-287`);
- `cli/parser.py` (`_finalize_output_spaces`, `:1167`);
- `test_utils_spaces.py`.

- [ ] **Step 1: Failing tests:**
  - `parse_space_token('anat')` → one spec with `resolution.label == 'nativemin'`.
  - `'dwiref'` and `'acpc'` behave the same. `'acpc'` no longer raises "needs a resolution".
  - `'anat:res-1p5x1p5x3mm'` parses as an anisotropic `mm` resolution.
  - `'acpc:res-native'` parses as `kind='native', strategy=None`.
  - `'MNI152NLin6Asym:res-native'` and `'MNI152NLin6Asym:res-2mm'` are rejected, as today.
  - `'anat:cohort-1'` is rejected ("does not accept a cohort").
  - `parse_output_spaces(['anat'])` succeeds; `parse_output_spaces(['MNI152NLin6Asym'])` raises
    "at least one DWI space".
  - `_finalize_output_spaces` with `['anat']` logs a warning naming QSIRecon. Use the `caplog`
    pattern from `test_cli.py`.
- [ ] **Step 2: Implement.**
  - `_NATIVE_RE = re.compile(r'^native(?P<strategy>min|max)?$')`.
  - In `parse_space_token`, accept `space in DWI_SPACES or space in _templates()`. A DWI space
    without `res-` gets `res_values = ['nativemin']`.
  - Drop the `space == ACPC` anisotropy rejection: anisotropic `mm` is allowed on every DWI
    space.
  - **Temporary gates**, each with an `OutputSpacesError` naming the phase that implements it:
    - plain `native` → Task 2.1;
    - anisotropic `mm` → Task 2.2;
    - `anat` → Task 3.3;
    - `dwiref` → Task 4.4.

    Each later task deletes its gate.
  - `parse_output_spaces`: require `any(spec.writes_dwi ...)`.
  - `_finalize_output_spaces`: if no spec has `space == ACPC`, call
    `config.loggers.cli.warning(...)` to say QSIRecon reads only `space-ACPC` outputs.
- [ ] **Step 3:** Suite against the baseline. CLI tests still pass explicit `acpc:res-5mm`.
- [ ] **Step 4: Commit**: `feat: parse anat, dwiref and native resolutions; default DWI spaces to nativemin`.

### Task 1.2: `--output-spaces` optional, defaulting to `acpc:res-nativemin`

**Files:**
- `cli/parser.py`: the argument at `:330`, which moves from `g_required` (`:302`) to the
  output-spaces section of its own group; `_finalize_output_spaces`;
- `config.py` (`workflow.output_spaces` docstring);
- `docs/running.rst` ("A minimal command"; "Output spaces and resolutions");
- `docs/upgrading.rst`;
- `test_cli.py`, `test_utils_spaces.py`.

- [ ] **Step 1: Failing tests.** `test_nothing_given_at_all_is_an_error` becomes
      `test_nothing_given_defaults_to_acpc_nativemin`: `opts.output_spaces == ['acpc:res-nativemin']`.
      Add one asserting that `--help` no longer lists `--output-spaces` under "Required arguments".
- [ ] **Step 2: Implement.** In `_finalize_output_spaces`, set `given = ['acpc:res-nativemin']`
      when empty, instead of failing. Move the argument out of `g_required`, and update the
      group's description, which mentions "--output-spaces has no default".
- [ ] **Step 3: Docs.**
  - `running.rst`: the minimal command drops `--output-spaces`, and the text explains the
    default.
  - `upgrading.rst`: "`--output-spaces` is optional (default `acpc:res-nativemin`)", with one
    sentence on runs that combine voxel sizes (Task 1.3).
- [ ] **Step 4:** Suite against the baseline.
- [ ] **Step 5: Commit**: `feat: make --output-spaces optional, defaulting to acpc:res-nativemin`.

### Task 1.3: Native resolutions per output, with a mismatch error

The voxel sizes of each output's runs are known at **build time**, because `base.py` already has
each output's unit list after grouping. So `native*` resolves to concrete zooms when the workflow
is built, and needs no run-time `VoxelSizeChooser`.

**Files:**
- `workflows/base.py`: after `concatenation_scheme`, about `:491`;
- `workflows/anatomical/volume.py`: the grid loop at `:249-281`, which keeps physical sizes only;
- `workflows/dwi/finalize.py`: grids per spec become an input list built per output;
- `workflows/dwi/distortion_group_merge.py`: `dwi_sampling_grid` from the per-output grid;
- `utils/spaces.py`: the helper below;
- tests in a new `qsiprep/tests/test_native_resolution.py`, plus `test_multires_merge_integration.py`.

**Helper:**

```python
def resolve_native_zooms(dwi_files) -> tuple[float, float, float]:
    """Return the shared voxel size of ``dwi_files`` on world-aligned axes, or raise.

    Zooms are taken after ``nb.as_closest_canonical`` so an oblique or sagittal run is compared
    axis-for-axis, and rounded to 3 decimals as ``GenerateSamplingReference`` does.
    """
```

It raises `OutputSpacesError` listing every file and its zooms when they differ. `base.py` turns
that into the spec's message, which names the output and suggests `res-<N>mm`.

- [ ] **Step 1: Failing tests:**
  - `resolve_native_zooms` with two 2 mm runs → `(2.0, 2.0, 2.0)`.
  - With 1.5 mm and 2 mm runs → raises, and the message lists both files.
  - With a sagittal-acquisition affine (axes permuted), zooms map onto canonical axes.
  - Integration, extending `_write_two_pair_dataset` in `test_multires_merge_integration.py`:
    give acq-b a 2.5 mm affine and build with the default `acpc:res-nativemin` and
    `--distortion-group-merge concat`. `init_single_subject_wf` raises with "combines runs with
    different voxel sizes".
  - The same dataset with `acpc:res-2mm` builds.
- [ ] **Step 2: Implement.**
  - In `base.py`, for each final output and each `native*` spec, compute the zooms from the
    output's raw DWI files (`units_by_name[...]`, following `merged_to_subgroups`). `nativemin`
    and `nativemax` take `min(zooms)` / `max(zooms)` as an isotropic size.
  - Build one `init_output_grid_wf` per (final output, spec), using a fixed `voxel_size`. Pass
    that output's grid list to its finalize workflows and merge workflows.
  - Physical-size specs keep the subject-level `dwi_sampling_grids` from `init_anat_preproc_wf`,
    so nothing changes for `acpc:res-<N>mm`.
  - `init_anat_preproc_wf` stops building grids for `native*` specs and loses its `dwi_files`
    argument. Remove the `--anat-only` fallback comment at `volume.py:262-271`.
  - `VoxelSizeChooser` is then unused for `native*`; keep it only if a physical-size path still
    needs it, otherwise delete it with its tests.
- [ ] **Step 3:** `grid_metadata` (`finalize.py`) still writes the resolved `Resolution` into the
      sidecar for `native*`. Assert this in the existing
      `test_single_native_acpc_records_its_resolution`.
- [ ] **Step 4:** Suite against the baseline. The integration manifests are unchanged: every CLI
      test uses `res-5mm` / `res-2mm`.
- [ ] **Step 5: Commit**: `feat: resolve native resolutions per output and refuse mixed voxel sizes`.

### Task 1.4: `res-native` writes no `res-`; SynB0's side output without `acpc`

**Files:** `workflows/dwi/finalize.py` (`res_entities`), `workflows/dwi/derivatives.py`,
`workflows/dwi/distortion_group_merge.py`, `workflows/dwi/util.py` (`add_synb0_outputs`),
`workflows/base.py` (the `dwi_sampling_grid` passed to `dwi_preproc_wf` for SynB0, at `:793`),
`test_output_spaces_naming.py`.

- [ ] **Step 1: Failing tests:**
  - `res_entities_for(spec)` (new, in `utils/spaces.py`) returns `{}` for
    `Resolution(kind='native', strategy=None)` and `{'res': label}` otherwise. Use it in
    finalize, derivatives and the merge workflow.
  - With `output_spaces = ['anat:res-2mm']` (no `acpc`), `add_synb0_outputs` writes
    `space-distortiongroup_desc-synb0_dwiref` from `outputnode.synthetic_b0`, with no `res-`.
    That image is already on the unit's b=0 grid (`fieldmap/synb0.py:290-292`).
  - With an `acpc` spec it still writes `space-ACPC_res-<first acpc label>_desc-synb0_dwiref`
    from `synthetic_b0_acpc`.
- [ ] **Step 2: Implement.**
  - `add_synb0_outputs` chooses the field and entities from the parsed specs.
  - `base.py` passes the *output's* first ACPC grid to `dwi_preproc_wf.inputnode.dwi_sampling_grid`
    (Task 1.3 made grids per output), or nothing when there is no `acpc` spec.
- [ ] **Step 3:** Suite against the baseline.
- [ ] **Step 4: Commit**: `feat: leave res- off res-native outputs; keep SynB0's side output without acpc`.

---

## Phase 2 — `res-native` and anisotropy

### Task 2.1: `res-native` grids via `GenerateSamplingReference`, reoriented to LPS

**Files:**
- `workflows/anatomical/volume.py`: a new `init_native_grid_wf`, next to `init_output_grid_wf`
  (`:1659`);
- `workflows/base.py`: per-output grids from Task 1.3;
- `utils/spaces.py`: remove the Task 1.1 gate for plain `native`;
- tests in `test_native_resolution.py`.

```python
def init_native_grid_wf(name='native_grid_wf') -> Workflow:
    """Build a grid with ``fixed_image``'s FOV and ``moving_image``'s voxel size, in LPS.

    niworkflows' GenerateSamplingReference resamples onto ``np.diag(zooms)``, which is
    axis-aligned RAS; the AFNI ``RAI`` reorientation (DICOM LPS, as init_template_lps_wf uses)
    makes it LPS, which every QSIPrep gradient writer assumes.
    """
    # inputnode: fixed_image, moving_image, fov_mask (optional); outputnode: grid_image
    # GenerateSamplingReference(keep_native=True) -> afni.Resample(orientation='RAI')
```

- [ ] **Step 1: Failing tests:**
  - **Pure function:** run `niworkflows.interfaces.nibabel._gen_reference` on synthetic
    images, a 1 mm LPS fixed image and a 1.5×1.5×3 mm oblique moving image, then the
    reorientation function used by the node. Assert:
    - the result's zooms are `(1.5, 1.5, 3.0)`;
    - `nb.aff2axcodes(...) == ('L', 'P', 'S')`;
    - the affine's 3×3 part is diagonal (axis-aligned).
  - `acpc:res-native` now parses, and the gate test from 1.1 is removed.
- [ ] **Step 2: Implement** the workflow and wire it for `native`:
  - *fixed*: the autoboxed anchor template from `init_output_grid_wf`'s first two nodes; expose
    `outputnode.autoboxed` on that workflow;
  - *moving*: the output's first raw DWI run;
  - no `fov_mask`: the autoboxed template is already cropped.
- [ ] **Step 3:** Suite against the baseline.
- [ ] **Step 4: Commit**: `feat: build res-native grids with GenerateSamplingReference in LPS`.

### Task 2.2: Anisotropic physical sizes and orientation-aware interpolation

**Files:**
- `workflows/anatomical/volume.py` (`init_output_grid_wf`, `_tupleize` at `:1711`);
- `interfaces/anatomical.py` (`VoxelSizeChooser`, if kept);
- `interfaces/images.py` (`ChooseInterpolator`, `:566-598`);
- `workflows/dwi/resampling.py` (the methods boilerplate at `:166-169`);
- `utils/spaces.py` (remove the anisotropy gate);
- `docs/running.rst`.

- [ ] **Step 1: Failing tests:**
  - `init_output_grid_wf(Resolution(kind='mm', label='1p5x1p5x3mm', zooms=(1.5, 1.5, 3.0)))`
    passes `(1.5, 1.5, 3.0)` to `afni.Resample`.
  - `ChooseInterpolator` with a sagittal input (axes permuted) and an upsampling grid on the
    *matching world axis* picks `Linear`. With index pairing it picked Lanczos.
  - The boilerplate says "1.5×1.5×3 mm voxels", not "isotropic".
- [ ] **Step 2: Implement.**
  - `init_output_grid_wf` passes `resolution.zooms` as a tuple; drop `_tupleize` for `mm`.
  - In `ChooseInterpolator`, compare `nb.as_closest_canonical(img).header.get_zooms()` for both
    the input and the grid.
  - Remove the anisotropy gate.
- [ ] **Step 3: Gradient-writer fixtures, as checks.** On a 1×1×2 mm LPS grid, write a known bvec
      set through `MRTrixGradientTable`, `DSIStudioBTable` and `GradientRotation` (identity
      transform), and assert the outputs equal the isotropic-grid outputs. No code change is
      expected, because the grid stays axis-aligned. If one differs, stop and report.
- [ ] **Step 4:** `docs/running.rst`: document anisotropic sizes and that QSIRecon rejects them.
- [ ] **Step 5:** Suite against the baseline.
- [ ] **Step 6: Commit**: `feat: allow anisotropic DWI output sizes`.

---

## Phase 3 — `anat`

### Task 3.1: Expose native anatomical-space derivatives and write them with no `space-` entity

**Files:**
- `workflows/anatomical/volume.py`:
  - the `init_anat_preproc_wf` outputnode (`:204-227`), adding `anat_preproc`, `anat_brain`,
    `anat_mask` and `anat_dseg`;
  - the sources: `anat_reference_wf.outputnode.bias_corrected`, the `synthstrip_anat_wf`
    outputs, and `synthseg_anat_wf.outputnode.aparc_image` through a second
    `mrtrix3.LabelConvert`, like `acpc_aseg_to_dseg`;
  - `init_anat_derivatives_wf` (`:1836`), adding `write_native_anat: bool`;
- `workflows/base.py`: set `write_native_anat` when any spec has `space in ('anat', 'dwiref')`;
- `test_output_spaces_naming.py`.

- [ ] **Step 1: Failing tests:**
  - With `write_native_anat=True`, the derivatives workflow has sinks rendering to
    `sub-01/anat/sub-01_desc-preproc_T1w.nii.gz`, `sub-01_desc-brain_mask.nii.gz` and
    `sub-01_dseg.nii.gz`: **no `space-`, no `res-`**.
  - These do not collide with any existing anat sink. Run `assert_no_collisions` over
    `render_datasink_paths(..., include_figures=False)`, extended to include sinks without
    `space` for this test.
  - With `write_native_anat=False` (all current runs), none exist.
- [ ] **Step 2: Implement.** Use `DerivativesDataSink` with `space` left undefined. Check
      `io_spec.json`'s anat patterns (`:27`, `:30-31`): they render without `space` because
      `[_space-{space}]` is optional.
- [ ] **Step 3:** Suite against the baseline. Manifests are unchanged, because no CLI test
      requests `anat` or `dwiref`.
- [ ] **Step 4: Commit**: `feat: write native anatomical-space derivatives for anat and dwiref outputs`.

### Task 3.2: `anat` grids

**Files:** `workflows/anatomical/volume.py` (grids built from the anat-space brain),
`workflows/base.py` (per-output grids for `anat` specs, from Task 1.3's structure).

- [ ] **Step 1: Failing test.** For an `anat:res-2mm` spec, the grid workflow's *fixed* input
      comes from `anat_brain`, not `anchor_lps_wf`. `anat:res-native` uses
      `init_native_grid_wf`, with the anat-space anatomical as *fixed* and `anat_mask` as
      `fov_mask`.
- [ ] **Step 2: Implement.** For physical, `nativemin` and `nativemax` sizes, reuse
      `init_output_grid_wf` on the anat-space brain. Its autobox and deoblique already produce
      an axis-aligned grid in the anat world frame. For `native`, use `init_native_grid_wf`.
- [ ] **Step 3:** Suite against the baseline.
- [ ] **Step 4: Commit**: `feat: build output grids in the anatomical frame`.

### Task 3.3: `anat` DWI outputs

**Files:**
- `workflows/base.py`: wire `acpc_inv_transform` into `dwi_finalize_wf` (it currently reaches
  only `dwi_preproc_wf`, at `:792`);
- `workflows/dwi/finalize.py`: per `anat` spec, `trailing_transforms=[acpc_inv_transform]` and
  `space='anat'`;
- `utils/spaces.py`: remove the `anat` gate;
- `docs/outputs.rst`, `docs/running.rst`;
- tests.

- [ ] **Step 1: Failing tests:**
  - For an `anat:res-2mm` spec, `dwi_trans_wf`'s `ComposeTransforms` receives
    `trailing_transforms` from `inputnode.acpc_inv_transform`, and the DWI sinks render
    `space-anat_res-2mm_desc-preproc_dwi`.
  - The Jacobian transport receives the same trailing transform.
  - Merging: `test_multires_merge_integration.py` with `--output-spaces acpc:res-2mm anat:res-2mm`
    builds one merge workflow per spec, each with its own `space`.
  - **Dice (D4):** the series QC for an `anat` spec takes `t1_dice_score` from the ACPC
    computation. When no `acpc` spec exists, finalize resamples only the DWI brain mask onto the
    anchor grid for Dice. That mask is not written.
  - Integration: build with `--output-spaces anat` only. The parser warns, and the workflow
    builds.
- [ ] **Step 2: Implement.** Gradient rotation needs no change: `compose_affines` folds the
      rigid into each volume's affine (Task 0.4).
- [ ] **Step 3: Docs.** `outputs.rst` gains the `space-anat` DWI files and the space-less
      anatomicals. `running.rst` documents `anat`.
- [ ] **Step 4:** Suite against the baseline.
- [ ] **Step 5: Commit**: `feat: write preprocessed DWI in the anatomical frame (anat)`.

---

## Phase 4 — `dwiref`

### Task 4.1: Pin which frame each backend's corrected data sits in (investigation + tests)

**This is the plan's main technical risk.** Do it before writing any `dwiref` output. With
coregistration dropped, the chain ends in whatever frame the backend's `b0_ref_image` defines. It
must be the frame the HMC/SDC stages map *into*:

| Backend | `hmc` stage | What must be confirmed |
|---|---|---|
| eddy | none (eddy writes corrected volumes; `interfaces/eddy.py`) | Eddy's output is in its reference frame (the first b=0, LAS-conformed). `b0_ref_image` is computed from eddy's output, not from pre-eddy data. |
| DIFFPREP | identity (`interfaces/tortoise.py`) | The corrected volumes and `b0_ref_image` share DIFFPREP's frame. |
| SHORELine | per-volume affines (`dwi/hmc_sdc.py`) | The affines map into `b0_ref_image`'s frame (the b=0 template). |

- [ ] **Step 1:** Trace, for each backend, the node that produces
      `dwi_preproc_wf.outputnode.b0_ref_image` and the node that produces the volumes and
      transforms finalize resamples. Write the result into the spec under `dwiref`, with
      `path:line` citations.
- [ ] **Step 2:** Pin each link with a construction test in `test_workflows_native.py`, asserting
      the edges the trace found: `b0_ref_image`'s source node and the input `dwi_files` come
      from the same backend output.
- [ ] **Step 3:** **If any backend's frames differ, stop and report.** Do not patch around it;
      the spec needs a per-backend reference frame, which is a design change.
- [ ] **Step 4: Commit**: `test: pin the frame each correction backend's outputs live in`.

### Task 4.2: `dwiref` chains

**Files:** `workflows/dwi/finalize.py`, tests in `test_sdc_warp_derivative.py` and
`test_workflows_jacobian.py`.

- [ ] **Step 1: Failing tests:**
  - At the distortion-group level, `drop_stages == ['to b=0 affine', 'to b=0 warp', 'b=0 to T1w']`.
  - At the subject level, `drop_stages == ['b=0 to T1w']`.
  - SDC map and CNR conjugation drop the same stages (Task 0.4).
  - For eddy + TOPUP at the distortion-group level, the chain is identity: the eddy identity
    path.
- [ ] **Step 2: Implement.** Select `drop_stages` from the resolved `--dwiref-definition`
      (`base.py`'s `make_dwiref` / fallback).
- [ ] **Step 3:** Suite against the baseline.
- [ ] **Step 4: Commit**: `feat: resample DWI in its dwiref frame without coregistration`.

### Task 4.3: `dwiref` grids and naming

**Files:** `workflows/base.py` (grids per unit at the distortion-group level, per subject at the
subject level), `utils/spaces.py` (`bids_space` for `dwiref` takes the resolved level), tests.

- [ ] **Step 1: Failing tests:**
  - The `dwiref` grid's *fixed* image is the unit's `b0_ref_image` (distortion group) or
    `dwiref_wf.outputnode.dwiref` (subject), always through `init_native_grid_wf`.
    - `native`: *moving* is the raw run, with the reference's brain mask as `fov_mask`.
    - Physical sizes: *moving* is a synthetic header with the requested zooms. Add a small
      `_zooms_header(zooms)` function node.
  - Filenames are `space-distortiongroup_res-...` or `space-subject_res-...`, next to the
    existing `space-<level>_dwiref`.
  - A single-group subject under `--dwiref-definition subject` resolves to `distortiongroup`
    (today's fallback; `base.py`).
- [ ] **Step 2: Implement.**
- [ ] **Step 3:** Suite against the baseline.
- [ ] **Step 4: Commit**: `feat: build dwiref grids from the level's reference image`.

### Task 4.4: The merge error (D11), the gate, and docs

**Files:** `workflows/base.py` (after the concatenation scheme), `utils/spaces.py` (remove the
`dwiref` gate), `docs/running.rst`, `docs/outputs.rst`, tests.

- [ ] **Step 1: Failing tests:**
  - `--dwiref-definition distortion-group --distortion-group-merge concat --output-spaces dwiref`
    on `_write_two_pair_dataset` raises at build time. The message names `sub-01` and suggests
    `--dwiref-definition subject` or `--distortion-group-merge none`.
  - The same with `--dwiref-definition subject` builds, with one merge workflow for the `dwiref`
    spec.
  - A single-unit output under distortion-group merging builds.
  - `dwiref` triggers the native anatomical-space anatomicals (Task 3.1).
- [ ] **Step 2: Implement** the check where `merged_to_subgroups` is built, and remove the gate.
- [ ] **Step 3: Docs.** Document `dwiref` and its levels, the merge error, and that QSIRecon does
      not read it.
- [ ] **Step 4:** Suite against the baseline.
- [ ] **Step 5: Commit**: `feat: write preprocessed DWI in its dwiref frame (dwiref)`.

---

## Phase 5 — Anatomical derivatives at the anatomical's own resolution (D9)

### Task 5.1: Written ACPC anatomicals, and the ACPC dwiref template, at the anatomical's voxel size

**Keep the internal anatomicals on the anchor grid.** Only the *written* copies move (see
"Correction to the spec").

**Files:**
- `workflows/anatomical/volume.py`: a new `acpc_native_grid` from `init_native_grid_wf`, with
  *fixed* = `anchor_lps_wf.outputnode.template_lps`, *moving* =
  `anat_reference_wf.outputnode.bias_corrected`, and no `fov_mask`;
- in `init_anat_derivatives_wf`, the ACPC sinks (`ds_t1_preproc`, `ds_t1_mask`, `ds_t1_seg`,
  `ds_t1_aseg`, `ds_t2_preproc`, `ds_t2w_unfatsat`) are fed from new rigid resamples of the
  anat-space images onto that grid, using `acpc_transform`, instead of from the internal
  `t1_*`;
- `workflows/base.py`: `ds_dwiref_acpc` is resampled onto the same grid;
- the spec's Phase 5 text;
- tests.

- [ ] **Step 1:** Re-read the spec's Phase 5 text. It was corrected with this plan to keep the
      internal grid and move only the written copies.
- [ ] **Step 2: Failing tests:**
  - The written `ds_t1_preproc`'s input comes from a node whose `reference_image` is the native
    grid. `outputnode.t1_preproc` / `t1_brain` (what DWI processing consumes) still come from
    `rigid_acpc_resample_*` onto `template_lps`. **Assert both**: the second assertion is what
    keeps estimation unchanged.
  - Pure check: `_gen_reference` with a 0.8 mm anatomical yields 0.8 mm zooms on the anchor
    FOV, in LPS after reorientation.
  - Filenames are unchanged (`test_acpc_anatomicals_write_no_res_entity`,
    `test_single_acpc_anat_paths_are_the_historical_ones`).
- [ ] **Step 3: Implement.** Use LanczosWindowedSinc for the image, NearestNeighbor or MultiLabel
      for the mask and segmentations, matching the existing `rigid_acpc_resample_*` choices.
- [ ] **Step 4:** Suite against the baseline. Manifests are unchanged (names only).
- [ ] **Step 5: Commit**: `feat: write ACPC anatomicals at the anatomical's own voxel size`.

### Task 5.2: Bare-template anatomicals at the anatomical's voxel size

**Files:** `workflows/anatomical/volume.py` (`init_anat_derivatives_wf`, the `resample_{label}_*`
nodes, which currently use `select_{label}_template_lps` as `reference_image`), tests in
`test_workflows_native.py` (`test_derivatives_reuse_the_preproc_template_chain`,
`test_transform_and_grid_lists_stay_index_aligned`), `docs/outputs.rst`.

- [ ] **Step 1: Failing tests.**
  - For a bare template, the resample's `reference_image` comes from `init_native_grid_wf`
    (*fixed* = that spec's `std_lps_wf` template, *moving* = the anat-space anatomical).
  - For `:res-2`, it still comes from the TemplateFlow grid.
  - Adjust `test_derivatives_reuse_the_preproc_template_chain`: the grid is *derived from* the
    chain's template, still with no second `get_template` node.
- [ ] **Step 2: Implement.** Registration keeps the TemplateFlow image as its reference, so the
      warps are unchanged.
- [ ] **Step 3: Docs.** `outputs.rst`: bare templates are written at the anatomical's voxel size;
      `:res-<label>` uses TemplateFlow's grid.
- [ ] **Step 4:** Suite against the baseline.
- [ ] **Step 5: Commit**: `feat: write bare-template anatomicals at the anatomical's own voxel size`.

---

## Final verification

- [ ] The full suite against the baseline: no new failures.
- [ ] `ruff check qsiprep` and `ruff format --check qsiprep`.
- [ ] `test_output_spaces_naming.py -v`: every naming pin passes.
- [ ] **One real run** on a CI dataset (e.g. DSDTI, `--sloppy`) with
      `--output-spaces acpc:res-5mm anat:res-5mm dwiref:res-native`. Confirm:
  - each space's DWI files exist, with the expected voxel sizes and orientations
    (`nib.aff2axcodes == ('L','P','S')`, diagonal affines);
  - gradient tables load in MRtrix (`mrinfo -dwgrad`) and DSI Studio;
  - the space-less anatomicals exist;
  - the ACPC outputs match a run from before this branch (same `.b`; DWI correlation > 0.999),
    which proves estimation is unchanged.
- [ ] Add the run's file list as a new integration manifest, plus a CLI test that requests
      `anat` and `dwiref`.
- [ ] A Codex adversarial review of the implementation (`codex review` on the branch diff), then
      an ordinary Codex review pass, per the user's instructions.

## Risks

- **Backend frames (Task 4.1).** If eddy's corrected volumes and `b0_ref_image` are not in one
  frame, `dwiref` at the distortion-group level needs a per-backend reference, which is a spec
  change.
- **Per-output grids (Task 1.3)** change where grids are built: from the anatomical workflow to
  `base.py`. That is the largest refactor in the plan. If it grows beyond one task, split it
  into "build-time zoom check" and "move grid construction".
- **Reports.** Task 0.3 changes figure filters. Check the HTML report from the final real run
  shows one section per space.
- **QSIRecon** must handle `res-nativemin`, symbolic `res-` labels and anisotropy errors before
  the release that contains Task 1.2.
- **Task count.** 18 tasks across Phases 0–5, in line with the spec's estimate of 17–19.
