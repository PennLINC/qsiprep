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

## Background: what the rejection stands in for

`111cdb5` made `--output-spaces acpc:res-2mm acpc:res-1p5mm --distortion-group-merge concat` a
parse-time error, because before it the extra resolutions were resampled and denoised in full
and then silently dropped:

- `init_dwi_finalize_wf` builds one `dwi_trans_wf` and one `final_denoise_wf` per ACPC spec, but
  only `index == 0` reaches its outputnode (`finalize.py:417`).
- `init_distortion_group_merge_wf` calls `init_dwi_derivatives_wf(source_file=source_file)` with
  no `resolution`, so it writes one set of derivatives with no `res-` entity
  (`distortion_group_merge.py:244`).

`dwi_finalize_wf.outputnode` has exactly one consumer — the `(dwi_finalize_wf, final_merge_wf,
...)` block at `workflows/base.py:771` — so widening it touches only this path. Units that write
their own derivatives (`write_derivatives=True`) never read it.

## Review history

Drafted 2026-09-11, then put through a Codex adversarial review of the plan itself. It confirmed
the plan's two load-bearing claims by repository-wide tracing — the five finalize outputs have no
other consumer, and the `merging_group_workflows` membership tests keep their meaning — and found
three defects, fixed here before any task was executed:

- **The task order broke a supported configuration.** Widening the finalize outputnode (then Task
  1) landed before `base.py` learned to index it (then Task 3), and single-resolution
  distortion-group merging is already allowed, so the intermediate commits fed a one-element list
  where a path was expected. The two edits are now one task.
- **The path-collision guard excluded the two sinks most likely to collide.** It called
  `render_datasink_paths` without `include_figures=True`, which skips every sink lacking a
  `space` entity — and the merge workflow's `ds_series_qc` has none (unlike finalize's), while
  the nested b=0 reportlet is a figure. Omitting `sink_entities` would have left colliding
  reports with every proposed test passing. Task 3 now renders with `include_figures=True`,
  asserts the expected sink inventory, and has a step that deliberately breaks the code to prove
  the guard bites.
- **"Confirm a real invocation builds" only parsed arguments.** It never constructed a workflow,
  so it would have passed with the whole plan absent. That step is now labelled for what it is,
  and Task 4 builds the integrated graph.

A fourth correction came from checking the review's claims against the source: the plan's Task 1
edited a `b0_ref_wf = init_dwi_reference_wf(gen_report=True, desc='resampled', name='b0_ref_wf', ...)`
call that does not exist. The real one is `name='merged_b0_ref'` with no `desc`, so its reportlet
is `desc-b0ref`, not `desc-resampledb0ref`.

## Global Constraints

- Run everything through micromamba: `micromamba run -n linc311 <command>`. Not pixi — the pixi
  env is Python 3.10 and cannot import this repo.
- Lint every touched file: `micromamba run -n linc311 ruff check <files>`. Format **only files
  this plan touches**: `qsiprep/workflows/anatomical/volume.py` and four test modules are
  already non-conformant at `HEAD`, so `ruff format qsiprep/` would bury the change in unrelated
  churn.
- **Never edit `docs/changes.md`.** It is assembled from PR titles at release time. Describe
  user-facing changes in the PR description instead.
- **Never run `git stash`** in this repo. Stage files **by name**, never `git add -A`.
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R
  ```
- The single-ACPC filenames QSIRecon reads must not change, on either the direct or the merged
  path. `qsiprep/tests/test_output_spaces_naming.py` guards this and must pass after every task.
- Baseline test state: **8 failures**, all reproducing on `origin/main` —
  `test_plan_cli_spec.py` (4, a `qsiplan` version skew), `test_trxscan_kit.py` (2, missing
  binaries), `test_template_qc.py` (1, truncated fixture), and
  `test_workflows_native.py::test_fsl_hmc_synb0_feeds_topup` (1). Plus 4 errors from missing
  FreeSurfer/MRtrix binaries, and `test_interfaces_dipy.py::test_patch2self` (needs
  `--data_dir`). Any other failure is a regression.
- **Ordering is load-bearing, in two places.**
  - **Task 2 must land as one commit.** Widening the finalize outputnode to lists and teaching
    `base.py` to index it are one change. Single-resolution distortion-group merging is *already
    supported* — the rejection in `111cdb5` blocks only *multiple* ACPC resolutions — so a commit
    that does the first without the second feeds a one-element list where `MergeDWIs` expects a
    path, breaking an invocation that works today. No construction-only test would catch it.
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
| `qsiprep/tests/test_output_spaces_naming.py` | Path-level collision guard | 3 |
| `qsiprep/tests/` (integrated) | Build the real subject workflow once | 4 |
| `qsiprep/cli/parser.py` | Drop the rejection | 5 |
| `docs/usage.rst` | Document the supported combination | 5 |

---

## Task 1: Teach the merge workflow which resolution it is

`init_distortion_group_merge_wf` writes `ds_series_qc`, `ds_report_gradients`,
`ds_merged_sidecar`, everything inside `init_dwi_derivatives_wf`, and the b=0 reference reportlet
inside its `merged_b0_ref` sub-workflow. With one merge workflow per resolution, all of those
collide across siblings exactly as the finalize reportlets did before `2802826`. This task gives
the workflow a `resolution` and a `write_shared_outputs` flag.

`init_mask_overlap_wf` and `init_modelfree_qc_wf` contribute no sinks — `qsiprep/workflows/dwi/qc.py`
contains no `DerivativesDataSink` — so the list above is the complete inventory.

`gradient_plot`/`ds_report_gradients` is built only for the first resolution: bvecs do not change
with the output grid, so the plot would be identical. `write_hmc_optimization` on the derivatives
workflow follows the same rule, for the reason its own docstring already gives.

**This task is first because it is the only one that changes nothing about how the merge workflow
is fed.** It is safe to land on its own.

**Files:**
- Modify: `qsiprep/workflows/dwi/distortion_group_merge.py:33-41` (signature), `:180-290`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_distortion_group_merge_wf(..., resolution=None, write_shared_outputs=True)`.
  `resolution` is a `qsiprep.utils.spaces.Resolution` or `None`; when set, a `res-<label>` entity
  goes on every sink this workflow owns and is forwarded to `init_dwi_derivatives_wf`.
  `write_shared_outputs=False` suppresses the sampling-scheme reportlet and the hmcOptimization
  sidecar, which do not vary by resolution.

- [ ] **Step 1: Write the failing tests**

Add to `qsiprep/tests/test_workflows_native.py`, beside the existing merge tests:

```python
def _merge_wf(tmp_path, resolution=None, write_shared_outputs=True, name='merge_wf'):
    """A distortion-group merge workflow with a real assembly, for the sink tests."""
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


def test_merge_wf_labels_its_sinks_with_the_resolution(tmp_path):
    """Two merge workflows write to one directory, so each must name its grid."""
    from qsiprep.utils.spaces import parse_output_spaces

    (spec,) = parse_output_spaces(['acpc:res-1p5mm'])
    wf = _merge_wf(tmp_path, resolution=spec.resolution)
    for sink_name in ('ds_series_qc', 'ds_merged_sidecar'):
        sink = wf.get_node(sink_name)
        assert sink is not None, f'{sink_name} is missing'
        assert sink.inputs.res == '1p5mm', f'{sink_name} carries no res- entity'
    # The nested b=0 reportlet is the one an entity test forgets; it lives inside
    # merged_b0_ref, not at the top level.
    report_sink = wf.get_node('merged_b0_ref').get_node('ds_report_b0_mask')
    assert report_sink is not None
    assert report_sink.inputs.res == '1p5mm'


def test_merge_wf_without_a_resolution_writes_the_historical_names(tmp_path):
    """A single-resolution merged run is what QSIRecon already reads."""
    from nipype.interfaces.base import isdefined

    wf = _merge_wf(tmp_path)
    for sink_name in ('ds_series_qc', 'ds_merged_sidecar'):
        assert not isdefined(wf.get_node(sink_name).inputs.res)
    assert not isdefined(wf.get_node('merged_b0_ref').get_node('ds_report_b0_mask').inputs.res)


def test_merge_wf_writes_shared_outputs_only_once(tmp_path):
    """bvecs do not change with the output grid, so one sampling-scheme figure."""
    with_shared = _merge_wf(tmp_path / 'a', write_shared_outputs=True)
    without = _merge_wf(tmp_path / 'b', write_shared_outputs=False, name='merge_wf_res2')
    assert with_shared.get_node('ds_report_gradients') is not None
    assert without.get_node('ds_report_gradients') is None
    assert without.get_node('gradient_plot') is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "merge_wf_labels or merge_wf_without or merge_wf_writes_shared"`
Expected: FAIL with `TypeError: init_distortion_group_merge_wf() got an unexpected keyword
argument 'resolution'`.

- [ ] **Step 3: Take the two new arguments**

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

Add to the docstring's Parameters section:

```
    resolution: :class:`~qsiprep.utils.spaces.Resolution` or None
        Set when more than one ACPC resolution was requested. Adds a ``res-<label>``
        entity to every sink this workflow owns, so sibling merge workflows for the
        other resolutions do not write the same filenames.
    write_shared_outputs: bool
        The sampling-scheme reportlet and the hmcOptimization sidecar do not vary by
        output resolution, so exactly one merge workflow per output writes them.
        Pass ``True`` for the first resolution and ``False`` for the rest.
```

Just after `workflow = Workflow(name=name)`, derive the entities once:

```python
    res_entities = {'res': resolution.label} if resolution is not None else {}
```

- [ ] **Step 4: Apply the entities to every sink this workflow owns**

Add `**res_entities` to the `DerivativesDataSink` of `ds_series_qc` and `ds_merged_sidecar`, and
pass the resolution through to the derivatives workflow:

```python
    dwi_derivatives_wf = init_dwi_derivatives_wf(
        source_file=source_file,
        resolution=resolution,
        write_hmc_optimization=write_shared_outputs,
    )
```

The b=0 reference sub-workflow is at `distortion_group_merge.py:180` and does **not** look like
finalize's. It is named `merged_b0_ref` and takes no `desc`, so it defaults to `'initial'` and its
reportlet is `desc-b0ref` (not `resampledb0ref`). Add the entities it gained in `2802826`, and
change nothing else about the call:

```python
    b0_ref_wf = init_dwi_reference_wf(
        name='merged_b0_ref',
        gen_report=True,
        source_file=source_file,
        sink_entities=res_entities,
    )
```

Guard the sampling-scheme plot and its sink — both the nodes and every `workflow.connect` that
mentions them — with `if write_shared_outputs:`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider`
Expected: PASS, apart from the baseline `test_fsl_hmc_synb0_feeds_topup`.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/dwi/distortion_group_merge.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: let a merge workflow say which ACPC resolution it wrote

One merge workflow per resolution means its series QC, sidecar, derivatives and b=0
reportlet all land on one filename unless each names its grid. The sampling-scheme
figure and the hmcOptimization sidecar do not vary by grid, so one merge workflow
per output writes them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 2: Expose every resolution and index it, in one commit

Two changes that **must land together**. `init_dwi_finalize_wf`'s outputnode currently exposes
only `index == 0` (`finalize.py:417`); widening it to lists is what lets `base.py` hand slot *i*
to merge workflow *i*. Splitting them would leave an intermediate commit where the outputs are
one-element lists but `base.py` still passes them straight through — and single-resolution
distortion-group merging **is already supported** (the rejection added in `111cdb5` only blocks
*multiple* ACPC resolutions), so that intermediate state feeds a list where `MergeDWIs` expects a
path. Nothing in the construction-only test suite would catch it; it breaks at run time on an
invocation that works today.

The five fields the merge path consumes become lists in `acpc_specs` order. The rest
(`local_bvecs_t1`, `dwi_mask_t1`, `confounds`, `gradient_table_t1`, `btable_t1`,
`hmc_optimization_data`, `fieldmap_hz_t1`) stay single-valued and keep taking index 0: nothing
downstream reads them on the merge path, and on the direct path each resolution's own
`dwi_derivatives_wf` writes them.

**Files:**
- Modify: `qsiprep/workflows/dwi/finalize.py:236-255` (outputnode), `:417-440`, and the
  `acpc_specs` truncation guard above the fan-out comment
- Modify: `qsiprep/workflows/base.py:442`, `:458-475`, `:754-785`
- Test: `qsiprep/tests/test_output_spaces_naming.py`, `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_dwi_finalize_wf`'s outputnode fields `dwi_t1`, `bvals_t1`, `bvecs_t1`,
  `t1_b0_ref` and `cnr_map_t1` become **lists**, one entry per `acpc_specs` entry, in that order.
  A single-resolution run yields a one-element list, so every consumer indexes rather than
  assuming a path.
- Produces: `merging_group_workflows: dict[str, list[Workflow]]`. Every membership test
  (`... in merging_group_workflows` at `:633` and `:652`) still asks about the group name and is
  unchanged in meaning.

- [ ] **Step 1: Write the failing tests**

Add to `qsiprep/tests/test_output_spaces_naming.py`:

```python
def test_finalize_outputnode_carries_every_acpc_resolution(tmp_path):
    """The merge path reads this outputnode, and it used to expose only index 0."""
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
    """A single resolution takes the same shape, so consumers never branch."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'], write_derivatives=False)
    merge = wf.get_node('merge_out_dwi_t1')
    assert merge is not None
    assert merge.interface._numinputs == 1
```

And to `qsiprep/tests/test_workflows_native.py`, for the `base.py` side. `init_single_subject_wf`
is too heavy for a unit test — no test in this repo builds one, and
`test_intramodal_transforms.py:113` and `test_workflows_gradwarp.py:800` both assert on
`inspect.getsource` instead and say why. Follow that precedent here; Task 4 builds the real graph.

```python
def test_single_subject_wf_builds_a_merge_workflow_per_resolution():
    """One merge workflow per ACPC spec, each fed slot i of the finalize outputs.

    Textual, per the precedent in test_intramodal_transforms.py: the integrated
    graph is built for real in test_two_resolutions_build_two_merge_workflows.
    """
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base.init_single_subject_wf)
    assert 'for index, spec in enumerate(acpc_specs)' in src
    assert 'write_shared_outputs=(index == 0)' in src
    assert "(('outputnode.dwi_t1', _select_grid, index), image_name)" in src
    assert "(('outputnode.bvals_t1', _select_grid, index), bval_name)" in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "outputnode_carries or one_element_list or builds_a_merge_workflow_per"`
Expected: FAIL — `merge_out_dwi_t1` does not exist, and none of the four source strings is present.

- [ ] **Step 3: Drop the truncation and merge the outputs**

Delete this block from `init_dwi_finalize_wf` (just above the `# Fan out the resampling` comment):

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

Immediately before `for index, spec in enumerate(acpc_specs):`, build one `Merge` per field:

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

Replace the `if index == 0:` block's first two connect groups:

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
            # These are the same for every requested resolution, or are written by
            # each resolution's own derivatives workflow on the direct path, so the
            # first spec is the one that reaches the single-valued fields.
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
            if doing_topup:
                workflow.connect([
                    (dwi_trans_wf, outputnode, [
                        ('outputnode.fieldmap_hz_resampled', 'fieldmap_hz_t1'),
                    ]),
                ])  # fmt:skip
```

Keep the existing `if index == 0:` block wiring `gtab_t1`/`btab_t1` in the derivatives section
exactly as it is.

- [ ] **Step 4: Build one merge workflow per spec**

In `init_single_subject_wf`, replace the single construction:

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

Import `_select_grid` from `qsiprep/workflows/dwi/finalize.py` in `base.py`, beside
`_first_sampling_grid` — which stays, because the HMC/SDC reference grid still uses it
(`base.py:667`). Pass the index as a **bare** argument, not `[index]`: `Workflow.connect` stores
everything after the function as the argument tuple, so a list-wrapped index arrives as a list
(the bug fixed in `eca9a53`).

- [ ] **Step 5: Hand each merge workflow its own slot**

Replace the `final_merge_wf` block at the end of the per-output loop:

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

`_select_grid(values, index)` is a plain `values[index]`, so it reads any of these lists. If its
name reads oddly here, rename it to `_select_index` in `finalize.py` and update its three call
sites there plus the two source-string assertions in Step 1 — in this same commit, not later.

- [ ] **Step 6: Run the whole suite**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/ -q -p no:cacheprovider --deselect qsiprep/tests/test_interfaces_dipy.py::test_patch2self`
Expected: the 8 baseline failures and nothing else.
`test_merged_groups_build_only_the_first_resolution` asserted the truncation this step removes —
update it to assert both `dwi_trans_wf_res2mm` and `dwi_trans_wf_res1p5mm` are built when
`write_derivatives=False`, which is now the point.

- [ ] **Step 7: Commit**

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

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 3: Guard the merged filenames at path level

Entities are not filenames. `qsiprep/data/io_spec.json` decides which entities survive into a
path, so two merge workflows can carry different `res` values and still land on one file.

Two traps this test must avoid, both of which would let it pass while the bug it guards is
present:

- `render_datasink_paths` skips any sink without a `space` entity unless `include_figures=True`.
  The merge workflow's `ds_series_qc` has **no** `space` (unlike finalize's), and the nested b=0
  reportlet is `datatype='figures'` with no `space` either — so the default call silently covers
  neither.
- Filtering by what happens to be collected hides a sink that stops being emitted. Assert the
  expected sink set explicitly.

**Files:**
- Test only: `qsiprep/tests/test_output_spaces_naming.py`

- [ ] **Step 1: Write the test**

```python
MERGED_DWI_BASE = {'subject': '01', 'datatype': 'dwi', 'suffix': 'dwi'}


def test_merged_resolutions_write_distinct_paths(tmp_path):
    """Two merged outputs in one directory must not share a filename."""
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
    assert {'ds_series_qc', 'ds_merged_sidecar', 'ds_report_b0_mask'} <= short_names, (
        f'sink inventory shrank: {sorted(short_names)}'
    )

    # include_figures also lets through the space-less ds_series_qc.
    paths = render_datasink_paths(found, MERGED_DWI_BASE, include_figures=True)
    assert_no_collisions(paths)


def test_single_merged_resolution_keeps_the_historical_paths(tmp_path):
    """QSIRecon reads these; a one-resolution merged run must not change them."""
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

`collect_datasink_entities`, `collect_figure_entities`, `render_datasink_paths` and
`assert_no_collisions` all already exist in this module. `DWI_BASE` is not reused because it
carries `'session': '1'` and the merged source file `sub-01_dwi.nii.gz` has none.

- [ ] **Step 2: Run the tests**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "merged_resolutions or single_merged"`
Expected: PASS if Tasks 1 and 2 are right. A collision here means an entity is being dropped by
the path pattern — fix `io_spec.json`, not the test.

- [ ] **Step 3: Prove the guard bites**

Temporarily delete `sink_entities=res_entities` from the `merged_b0_ref` call in
`distortion_group_merge.py`, re-run the two tests, and confirm
`test_merged_resolutions_write_distinct_paths` **fails** on the `resampledb0ref`/`b0ref` pair.
Restore it. A path guard that cannot fail is not a guard; the first draft of this test silently
excluded exactly this sink.

- [ ] **Step 4: Commit**

```bash
git add qsiprep/tests/test_output_spaces_naming.py
git commit -m "test: guard merged output filenames at path level

Entities are not filenames: io_spec.json decides which survive, so two merge
workflows carrying different res values can still collide. Rendered with
include_figures, because the merge workflow's ds_series_qc has no space entity and
the nested b=0 reportlet is a figure -- the default call covers neither, so the
obvious version of this test passes with the bug present.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 4: Build the integrated graph once, for real

Every test so far is either a construction test of one sub-workflow or a source-string assertion.
The path most likely to fail — `init_single_subject_wf` actually wiring N finalize outputs into N
merge workflows — is never constructed. This task builds it once.

**Files:**
- Test: `qsiprep/tests/test_workflows_native.py` (or a new
  `qsiprep/tests/test_multires_merge_integration.py` if the fixtures make that tidier)

- [ ] **Step 1: Write the failing test**

`qsiprep/tests/test_cli_run.py` already builds a real `BIDSLayout` over a
`generate_bids_skeleton` fixture — see its `_bids_layout` helper and the `long` skeleton dict.
Copy that pattern, then compile a plan and build the subject workflow:

```python
def test_two_resolutions_build_two_merge_workflows(tmp_path):
    """The one test that constructs the integrated graph.

    Everything else here is a sub-workflow construction test or a source-string
    assertion; this is what catches base.py handing the wrong slot to the wrong
    merge workflow.
    """
    from qsiprep.workflows.base import init_single_subject_wf

    # Two DWI runs in opposing phase-encoding directions so they merge into one
    # output group, and two ACPC resolutions so each gets its own merge workflow.
    bids_dir = _write_rpe_skeleton(tmp_path)
    config.execution.layout = _bids_layout(bids_dir)
    config.execution.output_dir = str(tmp_path / 'out')
    config.workflow.output_spaces = ['acpc:res-2mm', 'acpc:res-1p5mm']
    config.workflow.distortion_group_merge = 'concat'

    wf = init_single_subject_wf('01')
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
        for src, _, data in wf._graph.in_edges(merge_wf, data=True):
            for source, dest in data['connect']:
                if isinstance(source, tuple) and dest.endswith('_image'):
                    indices[merge_name] = source[2]
    assert sorted(indices.values()) == [0, 1], indices
```

Write `_write_rpe_skeleton` from the `generate_bids_skeleton` pattern in
`qsiprep/tests/utils.py`, with one T1w and two opposing-PE DWI runs in a single session.
`init_single_subject_wf`'s exact signature and the config keys it needs are best read from the
function itself: `grep -n "def init_single_subject_wf" -A 30 qsiprep/workflows/base.py`. If it
requires a compiled plan that cannot be produced from a skeleton alone, stop and say so rather
than stubbing the plan — a stub would make this test assert the fixture, not the wiring.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest -q -p no:cacheprovider -k "two_resolutions_build_two_merge_workflows"`
Expected: FAIL before Task 2's `base.py` change is in place; PASS after. Run it against
`HEAD~1` first if you want to see it fail for the right reason.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/
git commit -m "test: construct the integrated multi-resolution merge graph

Every other test here builds one sub-workflow or greps source text. This one builds
init_single_subject_wf and asserts two merge workflows exist and read different
slots of the finalize outputs -- the wiring most likely to be wrong and least
likely to be noticed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 5: Allow the combination

Only now is the parse-time rejection wrong. Removing it earlier would let a user request an
output the workflows could not yet produce.

**Files:**
- Modify: `qsiprep/cli/parser.py` (the `acpc_count > 1` check added in `111cdb5`)
- Modify: `docs/usage.rst` (the note added in `111cdb5`)
- Test: `qsiprep/tests/test_utils_spaces.py`

- [ ] **Step 1: Turn the rejection test around**

In `qsiprep/tests/test_utils_spaces.py`, replace
`test_multi_acpc_with_distortion_group_merge_is_rejected` with:

```python
def test_multi_acpc_with_distortion_group_merge_is_allowed(tmp_path):
    """Each requested resolution gets its own merged output."""
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(
        tmp_path,
        '--output-spaces',
        'acpc:res-2mm',
        'acpc:res-1p5mm',
        '--distortion-group-merge',
        'concat',
    )
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'acpc:res-1p5mm']
```

Keep `test_single_acpc_with_distortion_group_merge_is_allowed` as it is.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_utils_spaces.py -q -p no:cacheprovider -k "distortion_group_merge"`
Expected: FAIL with `SystemExit`.

- [ ] **Step 3: Delete the rejection**

Remove this block from `_apply_output_space_deprecations`:

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

- [ ] **Step 4: Rewrite the docs note**

In `docs/usage.rst`, replace the note added by `111cdb5` under "Multiple ``acpc`` entries":

```rst
.. note::

   Multiple ``acpc`` resolutions work with ``--distortion-group-merge``. Each requested
   resolution is merged separately and written with its own ``res-`` entity, so N
   resolutions cost N merges on top of N resampling passes. A single resolution is
   written without a ``res-`` entity, exactly as before.
```

- [ ] **Step 5: Run the whole suite**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/ -q -p no:cacheprovider --deselect qsiprep/tests/test_interfaces_dipy.py::test_patch2self`
Expected: the 8 baseline failures and nothing else.

- [ ] **Step 6: Confirm the CLI accepts it**

This checks the parser only — Task 4 is what proves the graph builds.

```bash
micromamba run -n linc311 python -c "
from qsiprep.cli.parser import _build_parser, _apply_output_space_deprecations
parser = _build_parser()
opts = parser.parse_args(['bids', 'out', 'participant',
    '--output-spaces', 'acpc:res-nativemin', 'acpc:res-1p5mm',
    '--distortion-group-merge', 'concat'])
_apply_output_space_deprecations(opts)
print(opts.output_spaces, opts.acpc_anchor)
"
```

Expected: `['acpc:res-nativemin', 'acpc:res-1p5mm', 'MNI152NLin2009cAsym'] MNI152NLin2009cAsym`,
no exception. (The positional `bids`/`out` paths need not exist for parsing.)

- [ ] **Step 7: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_utils_spaces.py docs/usage.rst
git commit -m "feat: support multiple ACPC resolutions with --distortion-group-merge

111cdb5 rejected the combination because the merge workflow wrote one set of
derivatives with no res- entity and the extra resolutions were silently dropped.
Each resolution now gets its own merge workflow and its own res- entity, so the
rejection has nothing left to protect.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Final verification

- [ ] **Run the whole non-integration suite**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/ -q -p no:cacheprovider --deselect qsiprep/tests/test_interfaces_dipy.py::test_patch2self`
Expected: the 8 baseline failures, nothing else.

- [ ] **Confirm the single-resolution filenames are unchanged**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -v -p no:cacheprovider`
Expected: PASS. Any changed ACPC filename on either path breaks QSIRecon — stop and fix.

- [ ] **Confirm a legacy invocation still builds**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli_run.py -q -p no:cacheprovider`
Expected: PASS, with deprecation warnings on stderr.

- [ ] **Lint**

Run: `micromamba run -n linc311 ruff check qsiprep/`
Expected: clean. Format only the files this plan touched.

---

## Risks and open questions

- **Cost is multiplicative and unbudgeted.** N resolutions already cost N resampling and
  denoising passes per unit; this adds N merges, N merged-derivative sets and N QC passes per
  merged output. Nothing warns the user. If that turns out to matter, a one-line
  `config.loggers.workflow.info` at build time naming the multiplier is the cheapest honest fix.
- **`_select_grid` is reused for non-grid lists.** Task 3 reads bvals, bvecs, images and b=0
  references through a function named for grids. Renaming it to `_select_index` is the tidier
  option and is called out in Task 3 Step 4; either is defensible, but do not leave a
  half-renamed pair.
- **No execution coverage.** Task 4 constructs the integrated graph, but nothing in this repo's
  non-integration suite *runs* a merge, so "two merged outputs actually appear on disk with
  different voxel sizes" is still verified only by a real run. Do one before merging the PR.
- **Task 4 may not be buildable as written.** No existing test constructs
  `init_single_subject_wf`; it needs a BIDS layout and a compiled plan. `test_cli_run.py` gets as
  far as a real `BIDSLayout` over a `generate_bids_skeleton` fixture, which is the precedent to
  copy, but whether a plan can be compiled from a skeleton alone is unverified. If it cannot,
  Task 4's step says to stop and report rather than stub the plan — a stubbed plan would make the
  test assert the fixture instead of the wiring.
- **`average` is untested with multiple resolutions.** `--distortion-group-merge average` pairs
  q-space coordinates across units; that logic is resolution-independent, but only `concat` is
  exercised by the tests above. If `average` matters, add one `_merge_wf` case for it in Task 2.
