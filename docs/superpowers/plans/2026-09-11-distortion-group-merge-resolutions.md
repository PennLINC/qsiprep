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
each unit into merge workflow *i*.

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
- **Task 5 removes the parse-time rejection and must be last.** Until it lands, the combination
  this plan enables is still refused at the CLI, so every earlier task is exercised through the
  workflow builders directly.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `qsiprep/workflows/dwi/finalize.py` | Expose every ACPC resolution on the outputnode | 1 |
| `qsiprep/workflows/dwi/distortion_group_merge.py` | Take a resolution; label and de-collide its own outputs | 2 |
| `qsiprep/workflows/base.py` | Build one merge workflow per ACPC spec and wire slot *i* | 3 |
| `qsiprep/cli/parser.py` | Drop the rejection | 5 |
| `docs/usage.rst` | Document the supported combination | 5 |

---

## Task 1: Expose every ACPC resolution on the finalize outputnode

Today `finalize.py:417` wires only `index == 0` into `outputnode`, with a comment saying the
merge path "has no notion of multiple output resolutions". This task gives it one: the five
fields the merge path consumes become lists in `acpc_specs` order, built by one `niu.Merge` per
field.

The remaining outputnode fields (`local_bvecs_t1`, `dwi_mask_t1`, `confounds`,
`gradient_table_t1`, `btable_t1`, `hmc_optimization_data`, `fieldmap_hz_t1`) stay single-valued
and keep taking index 0: nothing downstream reads them on the merge path, and on the direct path
each resolution's own `dwi_derivatives_wf` writes them.

**Files:**
- Modify: `qsiprep/workflows/dwi/finalize.py:236-255` (outputnode), `:417-440` (the `index == 0`
  block)
- Test: `qsiprep/tests/test_output_spaces_naming.py`

**Interfaces:**
- Produces: `init_dwi_finalize_wf`'s outputnode fields `dwi_t1`, `bvals_t1`, `bvecs_t1`,
  `t1_b0_ref` and `cnr_map_t1` become **lists**, one entry per entry of `acpc_specs`, in that
  order. A single-resolution run yields a one-element list, so every consumer must index rather
  than assume a path.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_output_spaces_naming.py`:

```python
def test_finalize_outputnode_carries_every_acpc_resolution(tmp_path):
    """The merge path reads this outputnode, and it used to expose only index 0.

    One Merge per field, filled in acpc_specs order, so base.py can hand slot i to
    the merge workflow built for resolution i.
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
    """A single resolution takes the same shape, so consumers never branch."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'], write_derivatives=False)
    merge = wf.get_node('merge_out_dwi_t1')
    assert merge is not None
    assert merge.interface._numinputs == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "outputnode_carries or one_element_list"`
Expected: FAIL — `merge_out_dwi_t1` does not exist.

Note: `_build_finalize` truncates `acpc_specs` to one entry when `write_derivatives=False`
(`finalize.py`, the guard added in `111cdb5`). Remove that truncation as part of Step 3 — it
exists precisely because the extra resolutions had nowhere to go, which is what this plan
changes. The first test will not even see two specs until it is gone.

- [ ] **Step 3: Drop the truncation and merge the outputs**

Delete this block from `init_dwi_finalize_wf` (it sits just above the `# Fan out the resampling`
comment):

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

Immediately before the `for index, spec in enumerate(acpc_specs):` loop, build one `Merge` per
exposed field:

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

Then replace the `if index == 0:` block's first two connect groups. The per-resolution values go
to the merges; the fields that stay single-valued keep their `index == 0` guard:

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

Keep the existing `if index == 0:` block that wires `gtab_t1`/`btab_t1` into
`gradient_table_t1`/`btable_t1` inside the derivatives section exactly as it is.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_workflows_gradwarp.py -q -p no:cacheprovider`
Expected: PASS. `test_merged_groups_build_only_the_first_resolution` asserted the truncation just
removed — update it to assert both `dwi_trans_wf_res2mm` and `dwi_trans_wf_res1p5mm` are built
when `write_derivatives=False`, which is now the point.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_output_spaces_naming.py
git commit -m "feat: expose every ACPC resolution on the finalize outputnode

The merge path read this outputnode and it carried only index 0, which is why
extra resolutions had nowhere to go and were truncated away. The five fields the
merge path consumes are now lists in acpc_specs order; a single resolution gives a
one-element list, so consumers index instead of special-casing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 2: Teach the merge workflow which resolution it is

`init_distortion_group_merge_wf` writes `ds_series_qc`, `ds_report_gradients`,
`ds_merged_sidecar`, everything inside `init_dwi_derivatives_wf`, and the `resampledb0ref`
reportlet inside its `b0_ref_wf`. With one merge workflow per resolution, all of those collide
across siblings exactly as the finalize reportlets did before `2802826`. This task gives the
workflow a `resolution` and a `write_shared_outputs` flag.

`gradient_plot`/`ds_report_gradients` is built only for the first resolution: bvecs do not change
with the output grid, and the figures path pattern carries `res-` but the plot would be
identical. `write_hmc_optimization` on the derivatives workflow follows the same rule, for the
reason its own docstring already gives.

**Files:**
- Modify: `qsiprep/workflows/dwi/distortion_group_merge.py:33-41` (signature), `:180-290`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_distortion_group_merge_wf(..., resolution=None, write_shared_outputs=True)`.
  `resolution` is a `qsiprep.utils.spaces.Resolution` or `None`; when set, a `res-<label>` entity
  goes on every sink this workflow owns and is forwarded to `init_dwi_derivatives_wf`.
  `write_shared_outputs=False` suppresses the sampling-scheme reportlet and the
  hmcOptimization sidecar, which do not vary by resolution.

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


def test_merge_wf_without_a_resolution_writes_the_historical_names(tmp_path):
    """A single-resolution merged run is what QSIRecon already reads."""
    from nipype.interfaces.base import isdefined

    wf = _merge_wf(tmp_path)
    for sink_name in ('ds_series_qc', 'ds_merged_sidecar'):
        assert not isdefined(wf.get_node(sink_name).inputs.res)


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

Forward the entities into the b=0 reference reportlet, which gained `sink_entities` in `2802826`:

```python
    b0_ref_wf = init_dwi_reference_wf(
        gen_report=True,
        desc='resampled',
        name='b0_ref_wf',
        source_file=source_file,
        sink_entities=res_entities,
    )
```

Check the existing call's arguments before editing:
`grep -n "b0_ref_wf = init_dwi_reference_wf" -A 6 qsiprep/workflows/dwi/distortion_group_merge.py`

Guard the sampling-scheme plot and its sink — both the nodes and every `workflow.connect` that
mentions them — with `if write_shared_outputs:`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider`
Expected: PASS, apart from the baseline `test_fsl_hmc_synb0_feeds_topup`.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/dwi/distortion_group_merge.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: let a merge workflow say which ACPC resolution it wrote

One merge workflow per resolution means its series QC, sidecar, derivatives and
resampledb0ref reportlet all land on one filename unless each names its grid. The
sampling-scheme figure and the hmcOptimization sidecar do not vary by grid, so one
merge workflow per output writes them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 3: Build one merge workflow per ACPC resolution

`merging_group_workflows` maps a merged-group name to a single workflow. It becomes a map to a
*list* of workflows, one per ACPC spec, and each unit's finalize outputs are handed to the merge
workflow for the matching index.

**Files:**
- Modify: `qsiprep/workflows/base.py:442`, `:458-475`, `:754-785`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Consumes: `init_dwi_finalize_wf`'s list-valued outputnode fields from Task 1, and
  `init_distortion_group_merge_wf`'s `resolution`/`write_shared_outputs` from Task 2.
- Produces: `merging_group_workflows: dict[str, list[Workflow]]`. Every membership test
  (`... in merging_group_workflows` at `:633` and `:652`) still asks about the group name and is
  unchanged.

- [ ] **Step 1: Write the failing test**

`init_single_subject_wf` needs a BIDS layout and a compiled plan, and **no test in this repo
builds one** — `test_intramodal_transforms.py:113` and `test_workflows_gradwarp.py:800` both
assert on `inspect.getsource(base.init_single_subject_wf)` instead, and say why. Follow that
precedent; the behaviour itself is covered by Tasks 2 and 4, which build real workflows.

Add to `qsiprep/tests/test_workflows_native.py`:

```python
def test_single_subject_wf_builds_a_merge_workflow_per_resolution():
    """One merge workflow per ACPC spec, each fed slot i of the finalize outputs.

    init_single_subject_wf is too heavy to build in a unit test (BIDS layout,
    anatomical workflow, a compiled plan), so this checks the wiring is textually
    present -- the precedent set in test_intramodal_transforms.py. What the merge
    workflows then write is covered by test_merge_wf_labels_its_sinks_with_the_resolution
    and test_merged_resolutions_write_distinct_paths.
    """
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base.init_single_subject_wf)
    # built per spec, and each handed its own resolution
    assert 'for index, spec in enumerate(acpc_specs)' in src
    assert 'write_shared_outputs=(index == 0)' in src
    # fed slot index of each list-valued finalize output
    assert "(('outputnode.dwi_t1', _select_grid, index), image_name)" in src
    assert "(('outputnode.bvals_t1', _select_grid, index), bval_name)" in src
```

If Task 3 Step 4 renames `_select_grid` to `_select_index`, update these two assertions to
match.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "builds_a_merge_workflow_per_resolution"`
Expected: FAIL — none of those four strings is in the source yet.

- [ ] **Step 3: Build the list**

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

`_select_grid` lives in `qsiprep/workflows/dwi/finalize.py`; import it in `base.py` beside
`_first_sampling_grid`. Pass the index as a **bare** argument, not `[index]` — `Workflow.connect`
stores everything after the function as the argument tuple, so a list-wrapped index arrives as a
list (this is the bug fixed in `eca9a53`).

- [ ] **Step 4: Hand each merge workflow its own slot**

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

`_select_grid(values, index)` is a plain `values[index]`, so it reads any of these lists; if its
name reads oddly here, rename it to `_select_index` in `finalize.py` and update the three call
sites there in the same commit.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/ -q -p no:cacheprovider --deselect qsiprep/tests/test_interfaces_dipy.py::test_patch2self`
Expected: the 8 baseline failures and nothing else.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/base.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: build one distortion-group merge workflow per ACPC resolution

Merging is independent per grid -- images meet only after they are resampled onto
the same one -- so each requested resolution gets its own merge workflow fed slot i
of the finalize outputs, rather than one workflow fanning out internally.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 4: Guard the merged filenames at path level

Entities are not filenames. `qsiprep/data/io_spec.json` decides which entities survive into a
path, so two merge workflows can carry different `res` values and still land on one file. This
task renders them the way `test_output_spaces_naming.py` already renders the direct path's
sinks.

**Files:**
- Test only: `qsiprep/tests/test_output_spaces_naming.py`

- [ ] **Step 1: Write the test**

```python
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
        for node_name, entities in collect_datasink_entities(wf).items():
            found[f'{spec.resolution.label}.{node_name}'] = entities

    paths = render_datasink_paths(found, MERGED_DWI_BASE)
    assert paths, 'expected ACPC-space sinks in the merge workflows'
    assert_no_collisions(paths)


def test_single_merged_resolution_keeps_the_historical_paths(tmp_path):
    """QSIRecon reads these; a one-resolution merged run must not change them."""
    from qsiprep.tests.test_workflows_native import _merge_wf

    wf = _merge_wf(tmp_path)
    paths = render_datasink_paths(collect_datasink_entities(wf), MERGED_DWI_BASE)
    assert all('_res-' not in path for path in paths.values()), paths
```

`collect_datasink_entities` already exists in this module. `DWI_BASE` does too, but it carries
`'session': '1'` and the merged source file `sub-01_dwi.nii.gz` has no session, so define a
session-less base beside these two tests rather than reusing it:

```python
MERGED_DWI_BASE = {'subject': '01', 'datatype': 'dwi', 'suffix': 'dwi'}
```

and pass `MERGED_DWI_BASE` to both `render_datasink_paths` calls above.

- [ ] **Step 2: Run the tests**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "merged_resolutions or single_merged"`
Expected: PASS if Tasks 2 and 3 are right. A collision here means an entity is being dropped by
the path pattern — fix `io_spec.json`, not the test.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/tests/test_output_spaces_naming.py
git commit -m "test: guard merged output filenames at path level

Entities are not filenames: io_spec.json decides which survive, so two merge
workflows carrying different res values can still collide. Render them.

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

- [ ] **Step 6: Confirm a real invocation builds**

Run:

```bash
micromamba run -n linc311 python -c "
from qsiprep.cli.parser import _build_parser, _apply_output_space_deprecations
import sys
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
- **No end-to-end coverage.** Every test here is a workflow-construction test. Nothing in this
  repo's non-integration suite runs a merge, so "two merged outputs actually appear on disk with
  different voxel sizes" is verified only by a real run. Worth one before merging the PR.
- **`average` is untested with multiple resolutions.** `--distortion-group-merge average` pairs
  q-space coordinates across units; that logic is resolution-independent, but only `concat` is
  exercised by the tests above. If `average` matters, add one `_merge_wf` case for it in Task 2.
