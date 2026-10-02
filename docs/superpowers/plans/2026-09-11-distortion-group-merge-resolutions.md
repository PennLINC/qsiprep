# Multi-Resolution `--distortion-group-merge` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `--output-spaces` request several `acpc` resolutions together with
`--distortion-group-merge`, writing one merged output per requested resolution instead of
rejecting the combination.

**Architecture:** One merge workflow per ACPC resolution, not one merge workflow that fans out
internally. Merging is independent per grid — images from different correction units are already
resampled onto the same grid before they meet — so the existing single-resolution
`init_distortion_group_merge_wf` is very nearly right as it stands; what it lacks is a
resolution to label its outputs with and a way to avoid colliding with its siblings. The
enabling change is upstream: `init_dwi_finalize_wf`'s outputnode currently exposes only the
first resolution, so it must carry one entry per ACPC spec before `base.py` can wire slot *i* of
each unit into merge workflow *i*. Those two edits land in a single commit (Task 2) — splitting
them would break single-resolution merging, which is already supported.

**Tech Stack:** Python 3.11, nipype workflows, niworkflows `DerivativesDataSink`, pybids path
patterns, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-output-spaces-design.md` (the `--output-spaces`
feature) and `docs/superpowers/plans/2026-09-10-output-spaces-review-fixes.md` Task 9, which
rejected this combination and named supporting it as Option B. This plan is Option B.

## Revision history

- **2026-09-11.** Drafted, then put through a Codex adversarial review that confirmed its two
  load-bearing claims (the merge path is the only consumer of the finalize outputs it widens; the
  `merging_group_workflows` membership tests keep their meaning) and fixed three defects: the task
  order broke single-resolution merging, the path-collision guard skipped the two sinks most
  likely to collide, and "confirm a real invocation builds" only parsed arguments.
- **2026-10-02.** Updated after `b2077a2` merged `main` into the branch and `5409b2a` removed the
  deprecated output-space flags. What changed:
  - Every file/line reference re-derived from the current code. `finalize.py` was rebuilt in the
    merge, so its outputnode, truncation guard and `index == 0` block all moved.
  - `main` added `ds_report_qc_warnings` to the merge workflow (`distortion_group_merge.py:232`).
    It was missing from Task 1's sink inventory and would collide across resolutions.
  - `main` added Jacobian, SDC-displacement and `fieldmap_hz` fields to the finalize outputnode.
    They are wired at `index == 0` only and **nothing outside `finalize.py` reads them** (each
    resolution's derivatives read its own `dwi_trans_wf` directly), so Task 2 leaves them alone.
  - The parse-time rejection now lives in `_finalize_output_spaces`, not
    `_apply_output_space_deprecations`, and the docs note moved from the deleted `usage.rst` into
    `running.rst`.
  - Task 4's precedent was wrong: `test_cli_run.py` has no `_bids_layout` helper, only
    `generate_bids_skeleton` calls.
  - Task 5's CLI check claimed the positional paths need not exist. `bids_dir` is
    `type=PathExists`, so they must.
  - The baseline test state was re-measured (see Global Constraints).

## Background: what the rejection stands in for

`111cdb5` made `--output-spaces acpc:res-2mm acpc:res-1p5mm --distortion-group-merge concat` a
parse-time error (now `parser.py:1213-1224`), because the extra resolutions were resampled and
post-processed in full and then silently dropped:

- `init_dwi_finalize_wf` builds one `dwi_trans_wf` and one `final_denoise_wf` per ACPC spec, but
  only `index == 0` reaches its outputnode (`finalize.py:521-555`). When `write_derivatives` is
  false — i.e. the unit is merged later — it truncates `acpc_specs` to the first spec
  (`finalize.py:405-413`) so the extra subtrees are not built at all.
- `init_distortion_group_merge_wf` calls `init_dwi_derivatives_wf(source_file=source_file)` with
  no `resolution` (`distortion_group_merge.py:267`), so it writes one set of derivatives with no
  `res-` entity.

`main`'s move of denoising to the raw series (#1146, #1169) is what keeps this tractable. Every
step that runs per resolution is now resampling, N4, a b=0 reference or QC — all valid on any grid.
Nothing denoises resampled or concatenated data.

### What runs at each resolution once this plan lands

| Scope | Steps |
|---|---|
| Once per series/unit, shared by every resolution | Conform, denoise, unring (`init_dwi_series_denoise_wf`); HMC/eddy; SDC estimation; coregistration; dwiref; the N4 *decision* (`biascorr_by_output`) |
| Per unit **and** per resolution (`init_dwi_finalize_wf`, already built today) | `dwi_trans_wf`: single-shot resampling, Jacobian weighting, gradient rotation, SDC-map composition, field resampling. `final_denoise_wf`: N4 (`DWIBiasCorrect`), resampled b=0 reference + mask + reportlet, model-free QC, confound merge |
| Per merged group **and** per resolution (`init_distortion_group_merge_wf`, one copy per spec after this plan) | `MergeDWIs` concatenation with b=0 harmonization; merged b=0 reference + mask; mask Dice; processed-data QC; gradient tables; series QC; QC-warnings reportlet; `init_dwi_derivatives_wf`; provenance sidecar with the grid `Resolution` |
| Resolution-independent but repeated per merge copy | Raw concatenation inside `MergeDWIs` and `raw_qc_wf` (DSI Studio on native-space data); merged confounds. See Risks — sharing these is a follow-up, not part of this plan |

`dwi_finalize_wf.outputnode`'s five merge fields have exactly one consumer — the
`(dwi_finalize_wf, final_merge_wf, ...)` block at `workflows/base.py:936-943` — so widening them
touches only this path. Units that write their own derivatives (`write_derivatives=True`) never
read them.

## Global Constraints

- Run everything through micromamba: `micromamba run -n linc311 <command>` (from Git Bash on this
  machine: `MSYS_NO_PATHCONV=1 wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/qsiprep &&
  /home/tsalo/.local/bin/micromamba run -n linc311 <command>"`).
- Lint every touched file: `micromamba run -n linc311 ruff check <files>` and
  `ruff format <files>`. The repo pins ruff 0.15.21 (`pyproject.toml`), which `linc311` has, and
  enforces numpydoc docstrings: summary on the first line, imperative mood, a blank line before
  any description.
- **Never edit `docs/changes.md`.** It is assembled from PR titles at release time. Describe
  user-facing changes in the PR description instead.
- **Never run `git stash`** in this repo. Stage files **by name**, never `git add -A`.
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- The single-ACPC filenames QSIRecon reads must not change, on either the direct or the merged
  path. `qsiprep/tests/test_output_spaces_naming.py` guards this and must pass after every task.
- **Baseline test state is environmental, so compare, don't count.** `linc311` currently has
  qsiplan 0.1.1 where `pyproject.toml` requires `>=0.4.2, <0.5`, and no ANTs or trxscan on
  `PATH`. As of 2026-10-02 that gives **64 failures and 5 errors** on both `origin/main` and this
  branch, the same set on each. Before Task 1, either update qsiplan in `linc311` (ask first — it
  is a shared environment) or record the failing set:
  ```bash
  micromamba run -n linc311 python -m pytest qsiprep/tests -q -p no:cacheprovider -n 8 \
      | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort > /tmp/baseline_fail.txt
  ```
  After each task, rerun the same command into `/tmp/task_fail.txt` and check that
  `comm -13 /tmp/baseline_fail.txt /tmp/task_fail.txt` prints nothing. Any new line is a
  regression.
- **Ordering is load-bearing, in two places.**
  - **Task 2 must land as one commit.** Widening the finalize outputnode to lists and teaching
    `base.py` to index it are one change. Single-resolution distortion-group merging is *already
    supported* — the rejection blocks only *multiple* ACPC resolutions — so a commit that does the
    first without the second feeds a one-element list where `MergeDWIs` expects a path, breaking
    an invocation that works today. No construction-only test would catch it.
  - **Task 5 removes the parse-time rejection and must be last.** Until it lands, the combination
    this plan enables is still refused at the CLI, so every earlier task is exercised through the
    workflow builders directly.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `qsiprep/workflows/dwi/distortion_group_merge.py` | Take a resolution; label and de-collide its own outputs | 1 |
| `qsiprep/workflows/dwi/finalize.py` | Expose every ACPC resolution on the outputnode | 2 |
| `qsiprep/workflows/base.py` | Build one merge workflow per ACPC spec and wire slot *i* | 2 |
| `qsiprep/tests/test_output_spaces_naming.py` | Path-level collision guard | 2, 3 |
| `qsiprep/tests/test_workflows_native.py` | Merge-workflow construction tests | 1, 2 |
| `qsiprep/tests/` (integrated) | Build the real subject workflow once | 4 |
| `qsiprep/cli/parser.py` | Drop the rejection | 5 |
| `docs/running.rst` | Document the supported combination | 5 |

---

## Task 1: Teach the merge workflow which resolution it is

`init_distortion_group_merge_wf` owns these sinks:

| Sink | Where | Varies by grid? |
|---|---|---|
| `ds_series_qc` | `distortion_group_merge.py:219` | yes |
| `ds_report_qc_warnings` (a `DerivativesMaybeDataSink`, added on `main`) | `:232` | yes — QC runs on the resampled merged series |
| `ds_report_gradients` (+ `gradient_plot`) | `:248-265` | no |
| `ds_merged_sidecar` (+ `grid_metadata`), only when `assembly` is given | `:296` | yes |
| everything inside `init_dwi_derivatives_wf` | `:267` | yes, except hmcOptimization |
| `ds_report_b0_mask` inside the `merged_b0_ref` sub-workflow | `:191`, `workflows/dwi/util.py:208` | yes |

`init_mask_overlap_wf` and `init_modelfree_qc_wf` contribute no sinks —
`qsiprep/workflows/dwi/qc.py` contains no `DerivativesDataSink` — so the table is the complete
inventory. With one merge workflow per resolution, every "yes" row collides across siblings
exactly as the finalize reportlets did before `2802826`. This task gives the workflow a
`resolution` and a `write_shared_outputs` flag.

`gradient_plot`/`ds_report_gradients` is built only for the first resolution: bvecs do not change
with the output grid, so the plot would be identical. `write_hmc_optimization` on the derivatives
workflow follows the same rule, for the reason its own docstring gives.

**This task is first because it is the only one that changes nothing about how the merge workflow
is fed.** It is safe to land on its own.

**Files:**
- Modify: `qsiprep/workflows/dwi/distortion_group_merge.py:35-43` (signature), `:191-195`,
  `:219-267`, `:296-307`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_distortion_group_merge_wf(..., resolution=None, write_shared_outputs=True)`.
  `resolution` is a `qsiprep.utils.spaces.Resolution` or `None`; when set, a `res-<label>` entity
  goes on every sink this workflow owns and is forwarded to `init_dwi_derivatives_wf`.
  `write_shared_outputs=False` suppresses the sampling-scheme reportlet and the hmcOptimization
  sidecar, which do not vary by resolution.

- [x] **Step 1: Write the failing tests**

Add to `qsiprep/tests/test_workflows_native.py`, beside
`test_merged_native_resolution_reaches_the_sidecar` (which already builds a merge workflow with
`_cfg`, `_StubLayout` and `_write_dwi`; reuse those):

```python
def _merge_wf(tmp_path, resolution=None, write_shared_outputs=True, name='merge_wf'):
    """Build a distortion-group merge workflow with a real assembly, for the sink tests."""
    from qsiplan.plan import OutputAssembly

    from qsiprep.workflows.dwi.distortion_group_merge import init_distortion_group_merge_wf

    cfg = _cfg(layout=_StubLayout())
    cfg.execution.output_dir = str(tmp_path / 'out')
    a_file = _write_dwi(tmp_path / 'sub-01_acq-hi_dwi.nii.gz')
    b_file = _write_dwi(tmp_path / 'sub-01_acq-lo_dwi.nii.gz')
    unit_a = make_preproc_unit([a_file])
    unit_b = make_preproc_unit([b_file])
    assembly = OutputAssembly(
        output_group='sub-01',
        input_runs=(unit_a.output_name, unit_b.output_name),
        strategy='concat',
        output_name='sub-01',
    )
    return init_distortion_group_merge_wf(
        merging_strategy='concat',
        inputs_list=[unit_a.output_name, unit_b.output_name],
        source_file='sub-01_dwi.nii.gz',
        output_prefix='sub-01',
        name=name,
        assembly=assembly,
        units=[unit_a, unit_b],
        resolution=resolution,
        write_shared_outputs=write_shared_outputs,
    )


MERGE_RES_SINKS = ('ds_series_qc', 'ds_report_qc_warnings', 'ds_merged_sidecar')


def test_merge_wf_labels_its_sinks_with_the_resolution(tmp_path):
    """Test that each merge workflow names its grid on every sink it owns.

    Two merge workflows write to one directory, so an unlabelled sink collides.
    """
    from qsiprep.utils.spaces import parse_output_spaces

    (spec,) = parse_output_spaces(['acpc:res-1p5mm'])
    wf = _merge_wf(tmp_path, resolution=spec.resolution)
    for sink_name in MERGE_RES_SINKS:
        sink = wf.get_node(sink_name)
        assert sink is not None, f'{sink_name} is missing'
        assert sink.inputs.res == '1p5mm', f'{sink_name} carries no res- entity'
    # The nested b=0 reportlet is the one an entity test forgets; it lives inside
    # merged_b0_ref, not at the top level.
    report_sink = wf.get_node('merged_b0_ref').get_node('ds_report_b0_mask')
    assert report_sink is not None
    assert report_sink.inputs.res == '1p5mm'


def test_merge_wf_without_a_resolution_writes_the_historical_names(tmp_path):
    """Test that a single-resolution merged run keeps the names QSIRecon reads."""
    from nipype.interfaces.base import isdefined

    wf = _merge_wf(tmp_path)
    for sink_name in MERGE_RES_SINKS:
        assert not isdefined(wf.get_node(sink_name).inputs.res)
    assert not isdefined(wf.get_node('merged_b0_ref').get_node('ds_report_b0_mask').inputs.res)


def test_merge_wf_writes_shared_outputs_only_once(tmp_path):
    """Test that only one merge workflow per output plots the sampling scheme.

    bvecs do not change with the output grid, so the figure would be identical.
    """
    with_shared = _merge_wf(tmp_path / 'a', write_shared_outputs=True)
    without = _merge_wf(tmp_path / 'b', write_shared_outputs=False, name='merge_wf_res2')
    assert with_shared.get_node('ds_report_gradients') is not None
    assert without.get_node('ds_report_gradients') is None
    assert without.get_node('gradient_plot') is None
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "merge_wf_labels or merge_wf_without or merge_wf_writes_shared"`
Expected: FAIL with `TypeError: init_distortion_group_merge_wf() got an unexpected keyword
argument 'resolution'`.

- [x] **Step 3: Take the two new arguments**

```python
def init_distortion_group_merge_wf(
    merging_strategy,
    inputs_list,
    source_file,
    output_prefix,
    name,
    assembly=None,
    units=(),
    resolution=None,
    write_shared_outputs=True,
) -> Workflow:
```

Add to the docstring's Parameters section (numpydoc, matching the existing entries):

```
    resolution : :class:`~qsiprep.utils.spaces.Resolution` or None, optional
        Set when more than one ACPC resolution was requested. Adds a ``res-<label>``
        entity to every sink this workflow owns, so sibling merge workflows for the
        other resolutions do not write the same filenames.
    write_shared_outputs : bool, optional
        The sampling-scheme reportlet and the hmcOptimization sidecar do not vary by
        output resolution, so exactly one merge workflow per output writes them.
        Pass ``True`` for the first resolution and ``False`` for the rest.
```

Just after `workflow = Workflow(name=name)`, derive the entities once:

```python
    res_entities = {'res': resolution.label} if resolution is not None else {}
```

- [x] **Step 4: Apply the entities to every sink this workflow owns**

Add `**res_entities` to the sinks of `ds_series_qc`, `ds_report_qc_warnings` and
`ds_merged_sidecar`, and pass the resolution through to the derivatives workflow:

```python
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        source_file=source_file,
        resolution=resolution,
        write_hmc_optimization=write_shared_outputs,
    )
```

The b=0 reference sub-workflow (`:191`) is named `merged_b0_ref` and takes no `desc`, so it
defaults to `'initial'` and its reportlet is `desc-b0ref` (not `resampledb0ref`). Add the
entities and change nothing else about the call:

```python
    b0_ref_wf = init_dwi_reference_wf(
        name='merged_b0_ref',
        gen_report=True,
        source_file=source_file,
        sink_entities=res_entities,
    )
```

Guard `gradient_plot`, `ds_report_gradients`, and the four `workflow.connect` entries that mention
them (`(outputnode, gradient_plot, ...)`, `(distortion_merger, gradient_plot, ...)`,
`(gradient_plot, ds_report_gradients, ...)` in the big connect block, plus the
`gradient_plot.inputs.source_pe_dirs` assignment) with `if write_shared_outputs:`. Pull those
three tuples out of the shared `workflow.connect([...])` list into their own guarded call.

- [x] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider`
Expected: PASS, apart from failures already in `/tmp/baseline_fail.txt`.

- [x] **Step 6: Commit**

```bash
git add qsiprep/workflows/dwi/distortion_group_merge.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: let a merge workflow say which ACPC resolution it wrote

One merge workflow per resolution means its series QC, QC warnings, sidecar,
derivatives and b=0 reportlet all land on one filename unless each names its grid.
The sampling-scheme figure and the hmcOptimization sidecar do not vary by grid, so
one merge workflow per output writes them.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 2: Expose every resolution and index it, in one commit

Two changes that **must land together**. `init_dwi_finalize_wf`'s outputnode exposes only
`index == 0` (`finalize.py:521-555`); widening the merge fields to lists is what lets `base.py`
hand slot *i* to merge workflow *i*. Splitting them would leave an intermediate commit where the
outputs are one-element lists but `base.py` still passes them straight through — and
single-resolution distortion-group merging **is already supported**, so that intermediate state
feeds a list where `MergeDWIs` expects a path. Nothing in the construction-only test suite would
catch it; it breaks at run time on an invocation that works today.

The five fields the merge path consumes — `dwi_t1`, `bvals_t1`, `bvecs_t1`, `t1_b0_ref` and
`cnr_map_t1` — become lists in `acpc_specs` order. Everything else on the outputnode
(`outputnode` at `finalize.py:306-331`) stays single-valued and keeps taking index 0:

- `local_bvecs_t1`, `dwi_mask_t1`, `confounds`, `hmc_optimization_data`, `fieldmap_hz_t1`,
  `jacobian_weights`, `jacobian_weight_index`, `jacobian_method`, `sdc_warp_to_template`,
  `sdc_refinement_to_template`: no consumer outside `finalize.py`. On the direct path each
  resolution's own `dwi_derivatives_wf` and SDC reportlets read its `dwi_trans_wf` /
  `final_denoise_wf` directly. Verify with
  `grep -n "outputnode\.\(jacobian\|sdc_\|fieldmap_hz_t1\|local_bvecs\|dwi_mask_t1\)" qsiprep/workflows/base.py`
  — it must print nothing that reads from `dwi_finalize_wf`.
- `gradient_table_t1`, `btable_t1`: wired from the derivatives section's own `if index == 0:`
  block (`finalize.py:903`). Leave that block as it is.

**Files:**
- Modify: `qsiprep/workflows/dwi/finalize.py:405-413` (truncation), `:445` (before the loop),
  `:521-555` (`index == 0` block)
- Modify: `qsiprep/workflows/base.py:81` (import), `:511-543` (merge construction),
  `:920-951` (feeding the merge workflows)
- Test: `qsiprep/tests/test_output_spaces_naming.py`, `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_dwi_finalize_wf`'s outputnode fields `dwi_t1`, `bvals_t1`, `bvecs_t1`,
  `t1_b0_ref` and `cnr_map_t1` become **lists**, one entry per `acpc_specs` entry, in that order.
  A single-resolution run yields a one-element list, so every consumer indexes rather than
  assuming a path.
- Produces: `merging_group_workflows: dict[str, list[Workflow]]`. Every membership test
  (`... in merging_group_workflows` at `base.py:734` and `:751`) still asks about the group name
  and is unchanged in meaning.

- [x] **Step 1: Write the failing tests**

Add to `qsiprep/tests/test_output_spaces_naming.py` (its `_build_finalize(tmp_path, output_spaces,
write_derivatives=True)` returns `(wf, acpc_specs)`):

```python
def test_finalize_outputnode_carries_every_acpc_resolution(tmp_path):
    """Test that the merge fields carry one slot per ACPC resolution.

    The merge path reads this outputnode, and it used to expose only index 0.
    """
    wf, acpc_specs = _build_finalize(
        tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'], write_derivatives=False
    )
    outputnode = wf.get_node('outputnode')
    for field in ('dwi_t1', 'bvals_t1', 'bvecs_t1', 't1_b0_ref', 'cnr_map_t1'):
        merge = wf.get_node(f'merge_out_{field}')
        assert merge is not None, f'{field} is not merged across resolutions'
        assert merge.interface._numinputs == len(acpc_specs)
        filled = {
            name
            for _, _, data in wf._graph.in_edges(merge, data=True)
            for _, name in data['connect']
        }
        assert filled == {f'in{i}' for i in range(1, len(acpc_specs) + 1)}, (
            f'merge_out_{field} slots not all connected: {sorted(filled)}'
        )
        edge = wf._graph.get_edge_data(merge, outputnode)
        assert edge is not None and ('out', field) in edge['connect']


def test_single_acpc_finalize_outputnode_is_a_one_element_list(tmp_path):
    """Test that a single resolution takes the same shape, so consumers never branch."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'], write_derivatives=False)
    merge = wf.get_node('merge_out_dwi_t1')
    assert merge is not None
    assert merge.interface._numinputs == 1
```

And to `qsiprep/tests/test_workflows_native.py`, for the `base.py` side. No test in this repo
builds `init_single_subject_wf`; follow the `inspect.getsource` precedent of
`test_workflows_gradwarp.py` here, and let Task 4 build the real graph.

```python
def test_single_subject_wf_builds_a_merge_workflow_per_resolution():
    """Test that each ACPC spec gets its own merge workflow, fed slot i.

    Textual: the integrated graph is built for real in
    test_two_resolutions_build_two_merge_workflows.
    """
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base.init_single_subject_wf)
    assert 'for index, spec in enumerate(acpc_specs)' in src
    assert 'write_shared_outputs=(index == 0)' in src
    assert "(('outputnode.dwi_t1', _select_grid, index), image_name)" in src
    assert "(('outputnode.bvals_t1', _select_grid, index), bval_name)" in src
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "outputnode_carries or one_element_list or builds_a_merge_workflow_per"`
Expected: FAIL — `merge_out_dwi_t1` does not exist, and none of the four source strings is present.

- [x] **Step 3: Drop the truncation and merge the outputs**

Delete this block from `init_dwi_finalize_wf` (`finalize.py:405-413`). Keep the gradwarp warning
directly above it (`:390-403`); it is about a different gap (see Risks).

```python
    if not write_derivatives:
        # This unit is concatenated later by init_distortion_group_merge_wf, which
        # writes one resolution through this workflow's single-valued outputnode.
        # nipype prunes nothing, so building the rest would resample and denoise
        # them in full and then discard the result. Truncating here, before
        # multi_acpc is computed, also names the survivor as the single resolution
        # it now is. The parser rejects this combination; the guard keeps it cheap
        # if that check is ever relaxed.
        acpc_specs = acpc_specs[:1]
```

Immediately before `for index, spec in enumerate(acpc_specs):` (`:445`), build one `Merge` per
field:

```python
    # The merge path (final_merge_wf in base.py) consumes these per resolution, so
    # every ACPC spec reaches the outputnode as one slot of a list in acpc_specs
    # order. A single resolution gives a one-element list, so consumers index rather
    # than special-case.
    merged_outputs = {
        field: pe.Node(niu.Merge(len(acpc_specs)), name=f'merge_out_{field}')
        for field in ('dwi_t1', 'bvals_t1', 'bvecs_t1', 't1_b0_ref', 'cnr_map_t1')
    }
    workflow.connect(
        [(merge, outputnode, [('out', field)]) for field, merge in merged_outputs.items()]
    )
```

Replace the first `workflow.connect` inside `if index == 0:` (`:521-541`) with an unconditional
per-resolution connect followed by a reduced `index == 0` connect. Leave the `doing_topup`,
`jacobian` and `sdc_fields` sub-blocks under `if index == 0:` exactly as they are.

```python
        workflow.connect([
            (dwi_trans_wf, merged_outputs['bvals_t1'], [
                ('outputnode.bvals', f'in{index + 1}'),
            ]),
            (dwi_trans_wf, merged_outputs['bvecs_t1'], [
                ('outputnode.rotated_bvecs', f'in{index + 1}'),
            ]),
            (dwi_trans_wf, merged_outputs['cnr_map_t1'], [
                ('outputnode.cnr_map_resampled', f'in{index + 1}'),
            ]),
            (final_denoise_wf, merged_outputs['dwi_t1'], [
                ('outputnode.dwi_t1', f'in{index + 1}'),
            ]),
            (final_denoise_wf, merged_outputs['t1_b0_ref'], [
                ('outputnode.t1_b0_ref', f'in{index + 1}'),
            ]),
        ])  # fmt:skip

        if index == 0:
            # Nothing outside this workflow reads these per resolution: the direct
            # path's derivatives read each resolution's own nodes, and the merge path
            # reads only the five list-valued fields above.
            workflow.connect([
                (dwi_trans_wf, outputnode, [
                    ('outputnode.local_bvecs', 'local_bvecs_t1'),
                ]),
                (final_denoise_wf, outputnode, [
                    ('outputnode.confounds', 'confounds'),
                    ('outputnode.dwi_mask_t1', 'dwi_mask_t1'),
                ]),
                (inputnode, outputnode, [('hmc_optimization_data', 'hmc_optimization_data')]),
            ])  # fmt:skip
            # ... existing doing_topup / jacobian / sdc_fields blocks, unchanged ...
```

Update the outputnode field comment at `:320-321` ("forwarded from the first dwi_trans_wf") only
if it stops being true; it should not.

- [x] **Step 4: Build one merge workflow per spec**

In `init_single_subject_wf`, replace the single construction and its `workflow.connect`
(`base.py:523-543`):

```python
            merging_group_workflows[merged_group] = [
                init_distortion_group_merge_wf(
                    merging_strategy=config.workflow.distortion_group_merge,
                    source_file=merged_group + '_dwi.nii.gz',
                    inputs_list=merged_to_subgroups[merged_group],
                    output_prefix=merged_group,
                    name=(
                        merged_group.replace('-', '_')
                        + '_final_merge_wf'
                        + (f'_res{spec.resolution.label}' if len(acpc_specs) > 1 else '')
                    ),
                    assembly=assembly_by_name[merged_group],
                    units=[units_by_name[key] for key in merged_to_subgroups[merged_group]],
                    # The res- entity only appears once more than one resolution was
                    # requested: a single-resolution merged run must keep producing
                    # exactly the filenames QSIRecon already reads.
                    resolution=spec.resolution if len(acpc_specs) > 1 else None,
                    write_shared_outputs=(index == 0),
                )
                for index, spec in enumerate(acpc_specs)
            ]

            for index, merge_wf in enumerate(merging_group_workflows[merged_group]):
                workflow.connect([
                    (anat_preproc_wf, merge_wf, [
                        ('outputnode.t1_brain', 'inputnode.t1_brain'),
                        ('outputnode.t1_seg', 'inputnode.t1_seg'),
                        ('outputnode.t1_mask', 'inputnode.t1_mask'),
                        (('outputnode.dwi_sampling_grids', _select_grid, index),
                         'inputnode.dwi_sampling_grid'),
                    ]),
                ])  # fmt:skip
```

Change the import at `base.py:81` to
`from .dwi.finalize import _select_grid, init_dwi_finalize_wf`. Keep `_first_sampling_grid`
(`:127`): the HMC/SDC reference grid still uses it at `:783`. Pass the index as a **bare**
argument, not `[index]`: `Workflow.connect` stores everything after the function as the argument
tuple, so a list-wrapped index arrives as a list (the bug fixed in `eca9a53`).

- [x] **Step 5: Hand each merge workflow its own slot**

Replace the `final_merge_wf` block at the end of the per-output loop (`base.py:920-951`):

```python
        final_merge_wfs = (
            merging_group_workflows.get(concatenation_scheme[output_fname], [])
            if merging_distortion_groups
            else []
        )
        for index, final_merge_wf in enumerate(final_merge_wfs):
            image_name = f'inputnode.{output_wfname}_image'
            bval_name = f'inputnode.{output_wfname}_bval'
            bvec_name = f'inputnode.{output_wfname}_bvec'
            original_bvec_name = f'inputnode.{output_wfname}_original_bvec'
            original_bids_name = f'inputnode.{output_wfname}_original_image'
            raw_concatenated_image_name = f'inputnode.{output_wfname}_raw_concatenated_image'
            confounds_name = f'inputnode.{output_wfname}_confounds'
            b0_ref_name = f'inputnode.{output_wfname}_b0_ref'
            cnr_name = f'inputnode.{output_wfname}_cnr'
            carpetplot_name = f'inputnode.{output_wfname}_carpetplot_data'
            workflow.connect([
                # Slot index of each list-valued output: the merge workflow for
                # resolution i only ever sees images already on grid i.
                (dwi_finalize_wf, final_merge_wf, [
                    (('outputnode.bvals_t1', _select_grid, index), bval_name),
                    (('outputnode.bvecs_t1', _select_grid, index), bvec_name),
                    (('outputnode.dwi_t1', _select_grid, index), image_name),
                    (('outputnode.t1_b0_ref', _select_grid, index), b0_ref_name),
                    (('outputnode.cnr_map_t1', _select_grid, index), cnr_name),
                ]),
                (dwi_preproc_wf, final_merge_wf, [
                    ('outputnode.raw_concatenated', raw_concatenated_image_name),
                    ('outputnode.original_bvecs', original_bvec_name),
                    ('outputnode.original_files', original_bids_name),
                    ('outputnode.carpetplot_data', carpetplot_name),
                    ('outputnode.confounds', confounds_name),
                ]),
            ])  # fmt:skip
```

`_select_grid(grids, index)` is a plain `grids[index]`, so it reads any of these lists. If its
name reads oddly here, rename it to `_select_index` in `finalize.py` and update its call sites
there (`:495`, `:759`), the `base.py` import, and the two source-string assertions in Step 1 — in
this same commit, not later.

- [x] **Step 6: Update the test that asserted the truncation**

`test_merged_groups_build_only_the_first_resolution` (`test_output_spaces_naming.py:611`)
asserted the behaviour Step 3 removes. Rename it to
`test_merged_groups_build_every_resolution` and assert
`prefixes == {'dwi_trans_wf_res2mm', 'dwi_trans_wf_res1p5mm'}` with `write_derivatives=False`,
with a docstring saying each resolution now feeds its own merge workflow.

- [x] **Step 7: Run the whole suite**

Run the baseline comparison from Global Constraints. Expected: no new failures.

- [x] **Step 8: Commit**

```bash
git add qsiprep/workflows/dwi/finalize.py qsiprep/workflows/base.py \
        qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: merge each ACPC resolution into its own output

The finalize outputnode carried only index 0, which is why extra resolutions had
nowhere to go and were truncated away. The five fields the merge path consumes are
now lists in acpc_specs order, and base.py builds one merge workflow per spec and
hands it slot i.

These land together deliberately. Single-resolution distortion-group merging is
already supported, so widening the outputs without teaching base.py to index them
would feed a one-element list where MergeDWIs expects a path -- a run-time break on
an invocation that works today, invisible to a construction-only test suite.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 3: Guard the merged filenames at path level

Entities are not filenames. `qsiprep/data/io_spec.json` decides which entities survive into a
path, so two merge workflows can carry different `res` values and still land on one file.

Two traps this test must avoid, both of which would let it pass while the bug it guards is
present:

- `render_datasink_paths` (`test_output_spaces_naming.py:410`) skips any sink without a `space`
  entity unless `include_figures=True`. The merge workflow's `ds_series_qc` has **no** `space`
  (unlike finalize's), and the nested b=0 reportlet and the QC-warnings reportlet are
  `datatype='figures'` with no `space` — so the default call silently covers none of the three.
- Filtering by what happens to be collected hides a sink that stops being emitted. Assert the
  expected sink set explicitly. (`collect_datasink_entities` matches by `isinstance`, so
  `DerivativesMaybeDataSink` — a subclass — is collected.)

**Files:**
- Test only: `qsiprep/tests/test_output_spaces_naming.py`

- [x] **Step 1: Write the test**

```python
MERGED_DWI_BASE = {'subject': '01', 'datatype': 'dwi', 'suffix': 'dwi'}


def test_merged_resolutions_write_distinct_paths(tmp_path):
    """Test that two merged outputs in one directory do not share a filename."""
    from qsiprep.tests.test_workflows_native import _merge_wf
    from qsiprep.utils.spaces import parse_output_spaces

    specs = parse_output_spaces(['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = {}
    for index, spec in enumerate(specs):
        wf = _merge_wf(
            tmp_path / f'r{index}',
            resolution=spec.resolution,
            write_shared_outputs=(index == 0),
            name=f'merge_wf_res{spec.resolution.label}',
        )
        collected = {
            **collect_datasink_entities(wf, full_names=True),
            **collect_figure_entities(wf),
        }
        for node_name, entities in collected.items():
            found[f'{spec.resolution.label}.{node_name}'] = entities

    # A sink that stops being emitted must fail here, not vanish quietly.
    short_names = {key.split('.')[-1] for key in found}
    expected = {'ds_series_qc', 'ds_report_qc_warnings', 'ds_merged_sidecar', 'ds_report_b0_mask'}
    assert expected <= short_names, f'sink inventory shrank: {sorted(short_names)}'

    # include_figures also lets through the space-less ds_series_qc.
    paths = render_datasink_paths(found, MERGED_DWI_BASE, include_figures=True)
    assert_no_collisions(paths)


def test_single_merged_resolution_keeps_the_historical_paths(tmp_path):
    """Test that a one-resolution merged run writes no res- entity.

    QSIRecon reads these paths.
    """
    from qsiprep.tests.test_workflows_native import _merge_wf

    wf = _merge_wf(tmp_path)
    found = {
        **collect_datasink_entities(wf, full_names=True),
        **collect_figure_entities(wf),
    }
    paths = render_datasink_paths(found, MERGED_DWI_BASE, include_figures=True)
    assert paths
    assert all('_res-' not in path for path in paths.values()), paths
```

`collect_datasink_entities` (`:82`), `collect_figure_entities` (`:584`),
`render_datasink_paths` and `assert_no_collisions` (`:445`) all already exist in this module.
`DWI_BASE` is not reused because it carries `'session': '1'` and the merged source file
`sub-01_dwi.nii.gz` has none.

- [x] **Step 2: Run the tests**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "merged_resolutions or single_merged"`
Expected: PASS if Tasks 1 and 2 are right. A collision here means an entity is being dropped by
the path pattern — fix `io_spec.json`, not the test.

- [x] **Step 3: Prove the guard bites**

Temporarily delete `sink_entities=res_entities` from the `merged_b0_ref` call in
`distortion_group_merge.py`, re-run the two tests, and confirm
`test_merged_resolutions_write_distinct_paths` **fails** on the two `desc-b0ref` reportlets.
Restore it, then repeat with `**res_entities` removed from `ds_report_qc_warnings`. A path guard
that cannot fail is not a guard; the first draft of this test silently excluded exactly these
sinks.

- [x] **Step 4: Commit**

```bash
git add qsiprep/tests/test_output_spaces_naming.py
git commit -m "test: guard merged output filenames at path level

Entities are not filenames: io_spec.json decides which survive, so two merge
workflows carrying different res values can still collide. Rendered with
include_figures, because the merge workflow's ds_series_qc has no space entity and
the b=0 and QC-warnings reportlets are figures -- the default call covers none of
them, so the obvious version of this test passes with the bug present.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 4: Build the integrated graph once, for real

Every test so far is either a construction test of one sub-workflow or a source-string assertion.
The path most likely to fail — `init_single_subject_wf` actually wiring N finalize outputs into N
merge workflows — is never constructed. This task builds it once.

**No test in this repo constructs `init_single_subject_wf` today**
(`grep -rn "init_single_subject_wf(" qsiprep/tests` is empty). It reads a BIDS dataset through
`collect_data`, groups it with qsiplan, and compiles a plan, so a fixture has to satisfy all
three. `test_cli_run.py` writes datasets with `niworkflows.utils.testing.generate_bids_skeleton`
(e.g. `:189`, `:240`); that is the precedent for the dataset, but nothing yet shows a plan
compiling from a skeleton alone.

**Files:**
- Test: a new `qsiprep/tests/test_multires_merge_integration.py`

- [x] **Step 1: Write the failing test**

```python
def test_two_resolutions_build_two_merge_workflows(tmp_path):
    """Test that base.py hands each merge workflow its own slot.

    The one test that constructs the integrated graph. Everything else is a
    sub-workflow construction test or a source-string assertion; this is what
    catches base.py handing the wrong slot to the wrong merge workflow.
    """
    from qsiprep.workflows.base import init_single_subject_wf

    # Two DWI runs in opposing phase-encoding directions so they merge into one
    # output group, and two ACPC resolutions so each gets its own merge workflow.
    bids_dir = _write_rpe_skeleton(tmp_path)
    _configure(bids_dir, tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'], merge='concat')

    wf = init_single_subject_wf('01', [])
    merge_wfs = sorted(
        {name.split('.')[0] for name in wf.list_node_names() if 'final_merge_wf' in name}
    )
    assert len(merge_wfs) == 2, merge_wfs
    assert any(name.endswith('_res2mm') for name in merge_wfs), merge_wfs
    assert any(name.endswith('_res1p5mm') for name in merge_wfs), merge_wfs

    # Slot i of the finalize outputs reaches merge workflow i, and no two merge
    # workflows read the same slot.
    indices = {}
    for merge_name in merge_wfs:
        merge_wf = wf.get_node(merge_name)
        for _, _, data in wf._graph.in_edges(merge_wf, data=True):
            for source, dest in data['connect']:
                if isinstance(source, tuple) and dest.endswith('_image'):
                    indices[merge_name] = source[2]
    assert sorted(indices.values()) == [0, 1], indices
```

Write `_write_rpe_skeleton` with `generate_bids_skeleton`: one T1w and two DWI runs with
opposite `PhaseEncodingDirection` and matching `TotalReadoutTime`, in a single session, so
qsiplan groups them into one distortion group. Write `_configure` from what
`init_single_subject_wf` actually reads — `grep -n "config\.\(execution\|workflow\)\." qsiprep/workflows/base.py`
— rather than guessing; it needs at least `execution.bids_dir`, `execution.layout`,
`execution.output_dir`, `workflow.output_spaces` and `workflow.distortion_group_merge`, plus
whatever `method_selection_from_config()` and `policy_from_namespace(config.workflow)` read.

If a plan cannot be compiled from a skeleton alone, **stop and report** rather than stubbing the
plan — a stub would make this test assert the fixture, not the wiring. Note that this test needs
a qsiplan that satisfies `pyproject.toml` (`>=0.4.2`); with the 0.1.1 in `linc311` it will fail on
the `PreprocUnit` attributes `main` started using, before reaching the code under test.

- [x] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest -q -p no:cacheprovider qsiprep/tests/test_multires_merge_integration.py`
Expected: FAIL before Task 2's `base.py` change is in place; PASS after. Run it against the
commit before Task 2 first if you want to see it fail for the right reason.

- [x] **Step 3: Commit**

```bash
git add qsiprep/tests/test_multires_merge_integration.py
git commit -m "test: construct the integrated multi-resolution merge graph

Every other test here builds one sub-workflow or greps source text. This one builds
init_single_subject_wf and asserts two merge workflows exist and read different
slots of the finalize outputs -- the wiring most likely to be wrong and least
likely to be noticed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 5: Allow the combination

Only now is the parse-time rejection wrong. Removing it earlier would let a user request an
output the workflows could not yet produce.

**Files:**
- Modify: `qsiprep/cli/parser.py:1213-1224` (the `acpc_count > 1` check in
  `_finalize_output_spaces`)
- Modify: `docs/running.rst:630-635` (the paragraph under "Multiple ``acpc`` resolutions")
- Test: `qsiprep/tests/test_utils_spaces.py`

- [x] **Step 1: Turn the rejection test around**

In `qsiprep/tests/test_utils_spaces.py`, replace
`test_multi_acpc_with_distortion_group_merge_is_rejected` (`:418`) with:

```python
def test_multi_acpc_with_distortion_group_merge_is_allowed(tmp_path):
    """Test that each requested resolution gets its own merged output."""
    from qsiprep.cli.parser import _finalize_output_spaces

    opts = _parse(
        tmp_path,
        '--output-spaces',
        'acpc:res-2mm',
        'acpc:res-1p5mm',
        '--distortion-group-merge',
        'concat',
    )
    _finalize_output_spaces(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'acpc:res-1p5mm']
```

Keep `test_single_acpc_with_distortion_group_merge_is_allowed` (`:437`) as it is.

- [x] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_utils_spaces.py -q -p no:cacheprovider -k "distortion_group_merge"`
Expected: FAIL with `SystemExit`.

- [x] **Step 3: Delete the rejection**

Remove this block from `_finalize_output_spaces`:

```python
    # init_distortion_group_merge_wf writes one set of derivatives with no res-
    # entity, so extra ACPC resolutions would be resampled, denoised and then
    # dropped without a trace in the output. A single resolution merges fine.
    acpc_count = sum(1 for spec in specs if not spec.standard)
    merging = getattr(opts, 'distortion_group_merge', 'none')
    if acpc_count > 1 and merging not in (None, 'none'):
        fail(
            f'--distortion-group-merge {merging} writes a single ACPC resolution, but '
            f'--output-spaces requested {acpc_count}. Request one "acpc" space, or use '
            '--distortion-group-merge none.'
        )
```

- [x] **Step 4: Rewrite the docs paragraph**

In `docs/running.rst`, under "Multiple ``acpc`` resolutions", replace the paragraph that begins
"More than one ``acpc`` resolution requires ``--distortion-group-merge none``." with:

```rst
Multiple ``acpc`` resolutions work with ``--distortion-group-merge``. Each
resolution is merged separately and written with its own ``res-`` entity, so
*N* resolutions cost *N* merges, each with its own QC, on top of the *N*
resampling passes per correction unit.
```

The preceding paragraph already says a single resolution is written without a `res-` entity;
leave it.

- [x] **Step 5: Run the whole suite**

Run the baseline comparison from Global Constraints. Expected: no new failures.

- [x] **Step 6: Confirm the CLI accepts it**

This checks the parser only — Task 4 is what proves the graph builds. `bids_dir` is
`type=PathExists`, so the positional paths must exist:

```bash
micromamba run -n linc311 python -c "
import tempfile, pathlib
from qsiprep.cli.parser import _build_parser, _finalize_output_spaces
root = pathlib.Path(tempfile.mkdtemp())
(root / 'bids').mkdir()
parser = _build_parser()
opts = parser.parse_args([str(root / 'bids'), str(root / 'out'), 'participant',
    '--output-spaces', 'acpc:res-nativemin', 'acpc:res-1p5mm',
    '--distortion-group-merge', 'concat'])
_finalize_output_spaces(opts, parser)
print(opts.output_spaces)
"
```

Expected: `['acpc:res-nativemin', 'acpc:res-1p5mm']`, no exception.

- [x] **Step 7: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_utils_spaces.py docs/running.rst
git commit -m "feat: support multiple ACPC resolutions with --distortion-group-merge

111cdb5 rejected the combination because the merge workflow wrote one set of
derivatives with no res- entity and the extra resolutions were silently dropped.
Each resolution now gets its own merge workflow and its own res- entity, so the
rejection has nothing left to protect.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Final verification

- [x] **Run the whole non-integration suite**

Run the baseline comparison from Global Constraints. Expected: no failure outside
`/tmp/baseline_fail.txt`.

- [x] **Confirm the single-resolution filenames are unchanged**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -v -p no:cacheprovider`
Expected: PASS. Any changed ACPC filename on either path breaks QSIRecon — stop and fix.

- [x] **Lint**

Run: `micromamba run -n linc311 ruff check qsiprep/ && micromamba run -n linc311 ruff format --check qsiprep/`
Expected: clean.

---

## Risks and open questions

- **Cost is multiplicative and unbudgeted.** N resolutions already cost N resampling, N4 and QC
  passes per unit; this adds N merges, N merged-derivative sets and N processed-data QC passes
  per merged output. Nothing warns the user. If that matters, a one-line
  `config.loggers.workflow.info` at build time naming the multiplier is the cheapest honest fix.
- **Resolution-independent work repeats per merge copy (follow-up, not in this plan).**
  `MergeDWIs` also concatenates the raw series (`merged_raw_dwi`), and `raw_qc_wf` runs DSI Studio
  QC on it; neither depends on the output grid, but each merge copy redoes both. Sharing them means
  splitting the raw concatenation out of `MergeDWIs` (or adding a "skip raw" flag) and feeding
  `series_qc.pre_qc` from one shared `raw_qc_wf`. Worth doing if multi-resolution merged runs
  become common; until then it costs time, not correctness.
- **The merge path drops main's new per-unit derivatives (pre-existing, any resolution count).**
  `init_distortion_group_merge_wf` writes no Jacobian weight map, no SDC displacement maps
  (`desc-sdc`, `desc-sdcrefinement`) and no `graddev`. Only the graddev loss is announced (the
  warning at `finalize.py:390-403`); the other two disappear silently for merged outputs. This
  plan multiplies the loss by N but does not cause it. Decide separately whether to carry them
  through the merge or at least warn — ideally before this lands, so the PR description can say
  which derivatives a merged multi-resolution run produces.
- **b=0 harmonization differs slightly per resolution.** `MergeDWIs` scales each unit to a common
  mean b=0 intensity computed on the resampled data, so the factors differ by interpolation noise
  between resolutions. Expected, not a bug, but worth one sentence in the PR description.
- **`_select_grid` is reused for non-grid lists.** Task 2 reads bvals, bvecs, images and b=0
  references through a function named for grids. Renaming it to `_select_index` is the tidier
  option and is called out in Task 2 Step 5; either is defensible, but do not leave a
  half-renamed pair.
- **No execution coverage.** Task 4 constructs the integrated graph, but nothing in this repo's
  non-integration suite *runs* a merge, so "two merged outputs actually appear on disk with
  different voxel sizes" is still verified only by a real run. Do one before merging the PR.
- **Task 4 may not be buildable as written.** See its preamble: no existing test builds
  `init_single_subject_wf`, and whether qsiplan compiles a plan from a skeleton alone is
  unverified. Its step says to stop and report rather than stub.
- **`average` is untested with multiple resolutions.** `--distortion-group-merge average` pairs
  q-space coordinates across units; that logic is resolution-independent, but only `concat` is
  exercised by the tests above. If `average` matters, add one `_merge_wf` case for it in Task 2.

---

## Execution notes (2026-10-02)

Implemented as `4c97ad2` (Task 1), `83aca18` (Task 2), `a738ec1` (Task 3), `9e510fb` (Task 4)
and `a734d2b` (Task 5). Where the code differs from the steps above:

- **Environment.** `linc311` was moved to qsiplan 0.4.2 before Task 4, replacing an editable link
  to an old local QSIPlan checkout. The baseline then dropped to 4 failures and 5 errors, all from
  missing binaries (ANTs, FreeSurfer, MRtrix, trxscan) or the truncated `template_qc` fixture.
- **Task 1 test helper.** `_merge_wf` creates `tmp_path` first: the shared-outputs test and Task 3
  pass subdirectories (`tmp_path / 'a'`, `tmp_path / f'r{index}'`) that did not exist.
- **Task 2 assertion style.** The `edge is not None and ...` assertion was split in two to satisfy
  ruff's PT018.
- **Task 3 guard check.** Done as written for both figure sinks: removing the `res-` entity from
  `merged_b0_ref` or from `ds_report_qc_warnings` each makes
  `test_merged_resolutions_write_distinct_paths` fail.
- **Task 4 dataset.** The plan's "two opposing-PE runs" builds a single correction unit, so no
  merge workflow exists. Under eddy, TOPUP pools both directions; under TORTOISE, one complete
  blip pair stays one DRBUDDI unit. The test instead writes two AP/PA pairs with different
  `TotalReadoutTime` (two blip groups) and a shared `MultipartID`, with `--hmc-method tortoise`.
  qsiplan then splits them into two units merged into one output. It uses
  `qsiprep.tests.utils.build_test_dataset` with a 20-volume b=1000 shell, not
  `generate_bids_skeleton` alone, because the series workflows read the image headers.
- **Task 4 slot assertion.** nipype stores a connection function's extra arguments as a tuple, so
  the slot read back is `(0,)`/`(1,)`, not `0`/`1`. The test asserts the exact slot per unit and
  per merge workflow, and fails against the Task 1 versions of `base.py`/`finalize.py`.
- **Task 5.** The integration test's config workaround was removed in the same commit; it now
  requests both resolutions through the CLI.
