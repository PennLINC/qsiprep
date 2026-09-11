# `--output-spaces` Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ten findings raised by the 2026-09-10 review of `output-spaces-new`, so that
every requested output space is written in the right frame, on the right grid, to a filename
nothing else overwrites.

**Architecture:** Four independent groups. (A) Two output-collision fixes local to the DWI
derivatives path. (C) Three parse-time/anchor-selection fixes, testable at the CLI without
building a workflow. (B) One restructuring of the standard-space normalization plumbing —
register every non-anchor space *from ACPC* rather than from its own rigid frame, register once
per template rather than once per resolution label, and hand the derivatives workflow the
template chain the preproc workflow already built. (D) One decision about
`--distortion-group-merge` with more than one ACPC resolution.

**Tech Stack:** Python 3.11, nipype workflows, niworkflows `DerivativesDataSink`, pybids path
patterns (`qsiprep/data/io_spec.json`), ANTs, AFNI, TemplateFlow, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-output-spaces-design.md` (the feature this branch
implements) and `~/Downloads/2026-09-10-output-spaces-review.md` (the review being addressed;
not tracked in the repo — each finding is restated in full in the task that fixes it, so this
plan stands alone).

## Global Constraints

- Run everything through micromamba: `micromamba run -n linc311 <command>`. Not pixi — the
  pixi env is Python 3.10 and cannot import this repo.
- Lint and format every touched file: `micromamba run -n linc311 ruff check <files>` and
  `micromamba run -n linc311 ruff format <files>`.
- **Never edit `docs/changes.md`.** It is assembled from PR titles at release time. Describe
  user-facing changes in the PR description instead.
- **Never run `git stash`** in this repo.
- The working tree is CRLF-dirty by default: stage files **by name**, never `git add -A`.
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R
  ```
- The single-ACPC filenames QSIRecon reads must not change. `qsiprep/tests/test_output_spaces_naming.py`
  guards this; it must pass after every task.
- Baseline test state on this branch is **8 failures**, all of which reproduce on `origin/main`:
  `test_plan_cli_spec.py` (4, a `qsiplan` version skew), `test_trxscan_kit.py` (2, missing
  binaries), `test_template_qc.py` (1, truncated fixture), `test_workflows_native.py::test_fsl_hmc_synb0_feeds_topup`
  (1). Plus 4 errors from missing FreeSurfer/MRtrix binaries and `test_interfaces_dipy.py::test_patch2self`
  (needs `--data_dir`). Any *other* failure is a regression introduced by this plan.
- Finding 1 of the review (`_select_grid` indexed with a list) is **already fixed** in commit
  `eca9a53` and has no task here.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `qsiprep/interfaces/bids.py` | `DerivativesSidecar` gains a runtime-merged `extra_data` input | 1 |
| `qsiprep/workflows/dwi/derivatives.py` | Drop the `resolution_meta` sink plumbing | 1 |
| `qsiprep/workflows/dwi/finalize.py` | Route the resolved grid size to the unit sidecar; per-resolution reportlet entities; stop building discarded subtrees | 1, 2, 9 |
| `qsiprep/workflows/dwi/resampling.py` | Thread `res` into the nested b0-ref reportlet | 2 |
| `qsiprep/workflows/dwi/util.py` | `init_dwi_reference_wf` accepts extra sink entities | 2 |
| `qsiprep/data/io_spec.json` | Add `[_res-{res}]` to the figures pattern | 2 |
| `qsiprep/utils/spaces.py` | Anchor drops the res label; infant-anchor selection | 3, 4 |
| `qsiprep/cli/parser.py` | `--infant` shim; reject mm-on-standard; reject multi-ACPC + merge | 4, 5, 9 |
| `qsiprep/workflows/anatomical/volume.py` | Normalization frame, registration reuse, shared template chain | 6, 7, 8 |
| `docs/api.rst` | Drop the deleted `init_spaces` member | 10 |

---

## Task 1: Write the resolved voxel size into the unit sidecar, not the data sinks

Review finding 2 (Blocker). When `resolution_meta=True` — any `res-native*` request, or more
than one ACPC resolution — `{'Resolution': [...]}` is wired into `meta_dict` of `ds_dwi_t1`,
`ds_bvals_t1`, `ds_bvecs_t1`, `ds_t1_b0_ref`, `ds_dwi_mask_t1`, `ds_gradient_table_t1` and
`ds_btable_t1`. niworkflows' `DerivativesDataSink._run_interface` writes
`<name before first '.'>.json` next to its output whenever metadata is non-empty
(`niworkflows/interfaces/bids.py:759`, `unlink` then `write_text`). Five of those sinks share
the stem `..._space-ACPC[_res-X]_desc-preproc_dwi` (`.nii.gz`, `.bval`, `.bvec`, `.b`,
`.b_table.txt`), so all five write `..._desc-preproc_dwi.json` containing only `Resolution`.
That is the exact path `ds_merged_sidecar` writes the full unit sidecar to. All six are
`run_without_submitting` with no ordering between them, so the last writer wins and the sidecar
nondeterministically loses either the acquisition metadata or the resolution.

**Files:**
- Modify: `qsiprep/interfaces/bids.py:254-274`
- Modify: `qsiprep/workflows/dwi/derivatives.py:65-68`, `:113-116`, `:311-322`
- Modify: `qsiprep/workflows/dwi/finalize.py:506-523`, `:542-552`
- Test: `qsiprep/tests/test_output_spaces_naming.py`

**Interfaces:**
- Consumes: `_grid_metadata(grid_file) -> {'Resolution': [x, y, z]}` (`finalize.py:46`), unchanged.
- Produces: `DerivativesSidecar` gains `extra_data = traits.Dict()`, merged over `sidecar_data`
  at run time. `init_dwi_derivatives_wf` loses its `resolution_meta` parameter and its
  `inputnode.resolution_meta` field; callers must stop passing them.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_output_spaces_naming.py`:

```python
def test_resolution_does_not_collide_with_the_unit_sidecar(tmp_path):
    """The resolved voxel size and the unit sidecar must not race for one path.

    niworkflows writes ``<stem before first '.'>.json`` beside any sink carrying
    metadata, and the preproc dwi/bval/bvec/b/b_table sinks all share one stem --
    the stem ds_merged_sidecar writes the full unit sidecar to.
    """
    wf, _ = _build_finalize(tmp_path, ['acpc:res-nativemin'])
    derivatives_wf = wf.get_node('dwi_derivatives_wf')
    assert derivatives_wf is not None
    # These five share the stem ..._desc-preproc_dwi, which is the path
    # ds_merged_sidecar writes the unit sidecar to.
    for sink_name in ('ds_dwi_t1', 'ds_bvals_t1', 'ds_bvecs_t1',
                      'ds_gradient_table_t1', 'ds_btable_t1'):
        sink = derivatives_wf.get_node(sink_name)
        assert sink is not None, f'{sink_name} is missing'
        assert not isdefined(sink.inputs.meta_dict), (
            f'{sink_name} writes a sidecar that collides with ds_merged_sidecar'
        )


def test_resolved_voxel_size_reaches_the_unit_sidecar(tmp_path):
    """res-native* is only reported in the sidecar, so it must still be written."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-nativemin'])
    grid_metadata = wf.get_node('grid_metadata')
    merged_sidecar = wf.get_node('merged_sidecar')
    assert grid_metadata is not None
    edge = wf._graph.get_edge_data(grid_metadata, merged_sidecar)
    assert edge is not None
    assert ('meta_dict', 'extra_data') in edge['connect']
```

Add `from nipype.interfaces.base import isdefined` to the imports at the top of the file if it
is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "sidecar or voxel_size"`
Expected: FAIL — the first with "writes a sidecar that collides", the second with
`assert None is not None` (no such edge).

- [ ] **Step 3: Give `DerivativesSidecar` a runtime-merged `extra_data` input**

In `qsiprep/interfaces/bids.py`, replace the input spec and `_run_interface`:

```python
class _DerivativesSidecarInputSpec(BaseInterfaceInputSpec):
    sidecar_data = traits.Dict()
    # Anything only a running node can know -- the voxel size a res-native* grid
    # turned out to be, say. Merged over sidecar_data rather than attached to a
    # DerivativesDataSink, because niworkflows would then write it to the same
    # <stem>.json this node writes and one would silently overwrite the other.
    extra_data = traits.Dict()
    source_file = File()
```

```python
    def _run_interface(self, runtime):
        json_fname = fname_presuffix(
            self.inputs.source_file, use_ext=False, suffix='.json', newpath=runtime.cwd
        )
        sidecar_data = dict(self.inputs.sidecar_data)
        if isdefined(self.inputs.extra_data):
            sidecar_data.update(self.inputs.extra_data)
        with open(json_fname, 'w') as jsonf:
            dump(sidecar_data, jsonf, sort_keys=True, indent=4)
        self._results['derivatives_json'] = json_fname
        return runtime
```

Confirm `isdefined` is imported in that module; add it to the existing
`from nipype.interfaces.base import ...` if not.

- [ ] **Step 4: Route the grid metadata to the sidecar in `finalize.py`**

Replace the `if write_resolution_meta:` block (currently `finalize.py:506-523`) so it connects
to `merged_sidecar` instead of `dwi_derivatives_wf`. `merged_sidecar` is defined below it in
the loop body, so move this block to sit immediately *after* `merged_sidecar` is created:

```python
        if write_resolution_meta:
            # res-native* is resolved from the DWI headers at run time, so the
            # sidecar is the only place a run reports what the grid turned out to
            # be. It goes here rather than on the data sinks: niworkflows derives a
            # sidecar path from the sink's own filename, and the preproc dwi, bval,
            # bvec, b and b_table sinks all share this node's stem.
            grid_metadata = pe.Node(
                niu.Function(
                    input_names=['grid_file'],
                    output_names=['meta_dict'],
                    function=_grid_metadata,
                ),
                name=f'grid_metadata{suffix}',
                run_without_submitting=True,
            )
            workflow.connect([
                (inputnode, grid_metadata, [
                    (('dwi_sampling_grids', _select_grid, index), 'grid_file'),
                ]),
                (grid_metadata, merged_sidecar, [('meta_dict', 'extra_data')]),
            ])  # fmt:skip
```

- [ ] **Step 5: Delete the `resolution_meta` plumbing from the derivatives workflow**

In `qsiprep/workflows/dwi/derivatives.py`: remove the `resolution_meta=False` parameter from
`init_dwi_derivatives_wf`, remove its paragraph from the docstring, remove `'resolution_meta'`
from the `inputnode` field list, and delete the whole `if resolution_meta:` block at `:311-322`.
Leave `ds_cnr_map_t1` and `ds_tsnr` alone — they carry their own descriptive `meta_dict` and
their stems (`_model-MAPMRI_dwimap`, `_dwimap`) collide with nothing.

In `qsiprep/workflows/dwi/finalize.py`, drop the argument from the call:

```python
        dwi_derivatives_wf = init_dwi_derivatives_wf(
            source_file=source_file,
            resolution=resolution_for_derivatives,
            # hmcOptimization is produced before resampling and is the same for
            # every ACPC resolution; writing it from every dwi_derivatives_wf
            # instance would be a same-path collision, so only the first spec
            # writes it.
            write_hmc_optimization=(index == 0),
            name=f'dwi_derivatives_wf{suffix}',
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_workflows_gradwarp.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/interfaces/bids.py qsiprep/workflows/dwi/derivatives.py \
        qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_output_spaces_naming.py
git commit -m "fix: write the resolved voxel size to the unit sidecar, not the data sinks

niworkflows writes <stem before first '.'>.json beside any sink carrying metadata.
The preproc dwi, bval, bvec, b and b_table sinks share one stem -- the stem
ds_merged_sidecar writes the full unit sidecar to -- so attaching Resolution to
them left six unordered nodes racing for one path, and the sidecar lost either the
acquisition metadata or the resolution depending on which ran last.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 2: Give per-resolution reportlets their own filenames

Review finding 4 (Blocker). The per-resolution loop builds one `init_finalize_denoising_wf` and
one `init_dwi_trans_wf` per ACPC spec, but their nested figure sinks never receive the `res`
entity: `ds_report_<name>_biascorr` (`finalize.py:843`, `desc='biascorrpost'`, only under
`--b1-biascorrect-stage final`) and the `resampledb0ref` reportlet built by
`init_dwi_reference_wf` from both `finalize.py:952` and `resampling.py:296`. With two
resolutions, four nodes write `figures/sub-X_desc-resampledb0ref_dwi.svg` and two write
`..._desc-biascorrpost_dwi.svg`; last writer wins and the report shows the wrong resolution's
figure. The figures path pattern in `io_spec.json` also has no `[_res-{res}]` slot, so passing
the entity through is not sufficient on its own.

The `trans_wf`/`denoise_wf` double write of `resampledb0ref` predates this branch. This task
also closes it, by turning off the `init_dwi_trans_wf` copy: the denoising workflow's reference
is computed from the final image, so it is the one worth showing.

**Files:**
- Modify: `qsiprep/data/io_spec.json` (figures pattern)
- Modify: `qsiprep/workflows/dwi/util.py:93` (signature), `:206-214` (sink)
- Modify: `qsiprep/workflows/dwi/finalize.py:~795` (denoise signature), `:843-851`, `:952-957`
- Modify: `qsiprep/workflows/dwi/resampling.py:296-301`
- Test: `qsiprep/tests/test_output_spaces_naming.py`

**Interfaces:**
- Produces: `init_dwi_reference_wf(..., sink_entities=None)` — a dict merged into
  `ds_report_b0_mask`'s `DerivativesDataSink`. `init_finalize_denoising_wf(..., sink_entities=None)`
  — same, forwarded to its own figure sinks and to its `final_b0_ref`.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_output_spaces_naming.py`:

```python
FIGURE_BASE = {'subject': '01', 'datatype': 'figures', 'suffix': 'dwi'}


def collect_figure_entities(wf):
    """Every figures-datatype sink in the workflow, by fully qualified node name."""
    found = {}
    for node_name in wf.list_node_names():
        node = wf.get_node(node_name)
        interface = getattr(node, 'interface', None)
        inputs = getattr(interface, 'inputs', None)
        if inputs is None or getattr(inputs, 'datatype', None) != 'figures':
            continue
        entities = {'datatype': 'figures', 'desc': inputs.desc, 'extension': '.svg'}
        if isdefined(getattr(inputs, 'res', Undefined)):
            entities['res'] = inputs.res
        found[node_name] = entities
    return found


def test_two_acpc_resolutions_write_distinct_figure_paths(tmp_path):
    """Two resolutions must not overwrite each other's reportlets."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = collect_figure_entities(wf)
    assert found, 'expected figure sinks in the finalize workflow'
    paths = render_datasink_paths(found, FIGURE_BASE)
    assert_no_collisions(paths)
```

`render_datasink_paths` skips entries with no `space` key; figure sinks have none, so relax that
guard by changing its `if 'space' not in entities: continue` to
`if 'space' not in entities and entities.get('datatype') != 'figures': continue`
(`collect_figure_entities` already sets `datatype`). Import `Undefined` from
`nipype.interfaces.base` at the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py::test_two_acpc_resolutions_write_distinct_figure_paths -q -p no:cacheprovider`
Expected: FAIL — "sinks collapsed onto N paths".

- [ ] **Step 3: Add `res` to the figures path pattern**

In `qsiprep/data/io_spec.json`, insert `[_res-{res}]` after `[_cohort-{cohort}]` in the figures
pattern, so it reads:

```
"sub-{subject}[/ses-{session}]/{datatype<figures>}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_dir-{direction}][_run-{run}][_part-{part}][_chunk-{chunk}][_space-{space}][_cohort-{cohort}][_res-{res}][_seg-{segmentation}][_desc-{desc}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio|dseg|mask|dwi|epiref|fieldmap>}{extension<.html|.svg|.png|.json|.png|.gif>}",
```

Verify the file still parses:
`micromamba run -n linc311 python -c "import json; json.load(open('qsiprep/data/io_spec.json'))"`

- [ ] **Step 4: Thread the entities through the nested workflows**

`qsiprep/workflows/dwi/util.py` — add a parameter to `init_dwi_reference_wf` and apply it:

```python
def init_dwi_reference_wf(
    ...,
    sink_entities=None,
):
```

```python
        ds_report_b0_mask = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc=report_desc,
                suffix='dwi',
                source_file=source_file,
                **(sink_entities or {}),
            ),
            name='ds_report_b0_mask',
            mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True,
```

`qsiprep/workflows/dwi/finalize.py` — add `sink_entities=None` to `init_finalize_denoising_wf`,
apply it to `ds_report_biascorr` (and the `ds_report_..._biascorr%d` variant at `:908`) and
forward it to `final_b0_ref`:

```python
            ds_report_biascorr = pe.Node(
                DerivativesDataSink(
                    datatype='figures',
                    desc='biascorrpost',
                    source_file=source_file,
                    **(sink_entities or {}),
                ),
```

```python
    final_b0_ref = init_dwi_reference_wf(
        gen_report=True,
        desc='resampled',
        name='final_b0_ref',
        source_file=source_file,
        sink_entities=sink_entities,
    )
```

Then pass `res_entities` at the call site inside the per-resolution loop:

```python
        final_denoise_wf = init_finalize_denoising_wf(
            source_file=source_file,
            do_biascorr=config.workflow.b1_biascorrect_stage == 'final',
            num_dwi_acquisitions=len(all_dwis),
            sink_entities=res_entities,
            name=f'final_denoise_wf{suffix}',
        )
```

- [ ] **Step 5: Stop `init_dwi_trans_wf` writing a second copy of the same reportlet**

In `qsiprep/workflows/dwi/resampling.py`, the `final_b0_ref` reference exists to produce the
resampled mask; its reportlet duplicates the one `init_finalize_denoising_wf` writes from the
final image. Turn it off:

```python
    final_b0_ref = init_dwi_reference_wf(
        # The denoising workflow writes this reportlet from the final image; a
        # second copy here has always raced it for the same filename.
        gen_report=False,
        desc='resampled',
        name='final_b0_ref',
        source_file=source_file,
    )
```

Check nothing consumed that node's `validation_report`:
`grep -n "final_b0_ref" qsiprep/workflows/dwi/resampling.py`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_reports.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 7: Check the report still finds its figures**

`qsiprep/data/reports-spec.yml:287` matches `desc: biascorrpost.*` with `regex_search: True`
and `:293` matches `desc: [b0ref, resampledb0ref]`. Adding a `res-` entity changes the
*filename*, not the `desc` value, so both still match. Confirm no other spec entry pins a full
filename: `grep -n "desc-" qsiprep/data/reports-spec.yml | head -40`

- [ ] **Step 8: Commit**

```bash
git add qsiprep/data/io_spec.json qsiprep/workflows/dwi/util.py \
        qsiprep/workflows/dwi/finalize.py qsiprep/workflows/dwi/resampling.py \
        qsiprep/tests/test_output_spaces_naming.py
git commit -m "fix: give each ACPC resolution's reportlets their own filenames

The nested biascorrpost and resampledb0ref sinks never saw the res- entity, and
the figures path pattern had nowhere to put one, so every resolution's reportlets
landed on one filename and the report showed whichever ran last. Also drops the
duplicate resampledb0ref that init_dwi_trans_wf has always written alongside the
denoising workflow's copy.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 3: The ACPC anchor drops the user's `res-` label

Review finding 5 (High). `select_acpc_anchor` returns the first infant `SpaceSpec` verbatim,
including any `res-` label, and `templateflow_kwargs` forwards a label-kind resolution to
`GetTemplate`, overriding its `'1'` default. So `--output-spaces acpc:res-1mm MNIInfant:cohort-auto:res-2`
fetches a 2 mm anchor; `anchor_lps_wf.outputnode.template_lps` is then the `reference_image` for
`rigid_acpc_resample_brain/mask/head/aseg`, whose outputs go straight to `ds_t1_preproc` with no
further resampling. The ACPC `desc-preproc_T1w`, brain mask, dseg and aseg are silently written
at 2 mm, and the output-grid autobox padding of 4 voxels doubles in mm, changing the DWI FoV.
It also makes the anchor list-order dependent for `MNIInfant:cohort-auto:res-1:res-2`,
contradicting the function's own docstring. The spec says the anchor carries template + cohort
only.

**Files:**
- Modify: `qsiprep/utils/spaces.py:298-320`
- Test: `qsiprep/tests/test_utils_spaces.py`

**Interfaces:**
- Produces: `select_acpc_anchor(specs, explicit=None) -> SpaceSpec` with `resolution=None`
  always. Callers comparing anchors already use `.fullname`, which ignores resolution.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_utils_spaces.py`:

```python
def test_anchor_drops_the_resolution_label():
    """The anchor sets the ACPC grid; a res- label on it would silently move it."""
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor

    specs = parse_output_spaces(['acpc:res-1mm', 'MNIInfant:cohort-auto:res-2'])
    anchor = select_acpc_anchor(specs)
    assert anchor.space == 'MNIInfant'
    assert anchor.cohort == 'auto'
    assert anchor.resolution is None


def test_anchor_is_independent_of_token_order():
    """Two resolutions of one infant template must not make the anchor order-dependent."""
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor

    forward = select_acpc_anchor(parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-auto:res-1:res-2']))
    reverse = select_acpc_anchor(parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-auto:res-2:res-1']))
    assert forward == reverse
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_utils_spaces.py -q -p no:cacheprovider -k "anchor_drops or anchor_is_independent"`
Expected: FAIL — `anchor.resolution` is `Resolution(kind='label', label='2', ...)`, and the two
orders differ.

- [ ] **Step 3: Strip the resolution in `select_acpc_anchor`**

In `qsiprep/utils/spaces.py`, add `replace` to the existing dataclasses import if it is not
already there (`from dataclasses import dataclass, replace`), and change the loop:

```python
    if explicit is not None:
        return explicit
    for name in INFANT_ANCHORS:
        for spec in specs:
            if spec.space == name:
                # The anchor fixes the ACPC grid every anatomical and the DWI FoV are
                # written on, so it takes the template's default resolution. Honouring
                # a res- label here would silently move that grid, and would make the
                # anchor depend on which of two labels the user typed first.
                return replace(spec, resolution=None)
    return SpaceSpec(space=DEFAULT_ANCHOR)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_utils_spaces.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/utils/spaces.py qsiprep/tests/test_utils_spaces.py
git commit -m "fix: keep the ACPC anchor at the template's default resolution

A res- label on the infant template reached GetTemplate and became the reference
grid for every ACPC anatomical and for the DWI FoV, so MNIInfant:cohort-auto:res-2
silently wrote 2mm anatomicals for a res-1mm request. It also made the anchor
depend on token order, which the docstring promised it did not.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 4: `--infant` ensures an infant anchor instead of hardcoding MNIInfant

Review finding 7 (Medium). `parser.py:1231` reads:

```python
if opts.infant and not any(s.split(':')[0] == 'MNIInfant' for s in given):
    given.append('MNIInfant:cohort-auto')
```

Two consequences. `--infant --output-spaces acpc:res-2mm UNCInfant:cohort-auto` appends
MNIInfant anyway and anchors to it, even though `INFANT_ANCHORS` already knows UNCInfant is an
infant template — the subject pays a second SyN and gets unrequested `space-MNIInfant`
derivatives. And on the legacy path, `main` set `anatomical_template = 'MNIInfant'`
unconditionally under `--infant`, discarding `--anatomical-template`; the shim keeps the legacy
adult template *and* appends MNIInfant, so `--infant --output-resolution 2 --anatomical-template MNI152NLin2009cAsym`
now runs an extra adult SyN and writes adult-space anatomicals it never used to.

**Files:**
- Modify: `qsiprep/cli/parser.py:1218-1232`
- Test: `qsiprep/tests/test_cli.py`

**Interfaces:**
- Consumes: `INFANT_ANCHORS` from `qsiprep.utils.spaces` (already `('MNIInfant', 'UNCInfant')`).

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_cli.py`, next to the other deprecation tests:

```python
def test_infant_accepts_an_explicit_infant_template(tmp_path):
    """UNCInfant is an infant anchor, so --infant must not append MNIInfant too."""
    from qsiprep.cli.parser import parse_args

    bids, out = _minimal_bids(tmp_path)
    parse_args([
        str(bids), str(out), 'participant',
        '--infant',
        '--output-spaces', 'acpc:res-2mm', 'UNCInfant:cohort-auto',
    ])
    assert 'MNIInfant:cohort-auto' not in config.workflow.output_spaces
    assert config.workflow.acpc_anchor.startswith('UNCInfant')


def test_legacy_infant_replaces_rather_than_augments_the_template(tmp_path):
    """main discarded --anatomical-template under --infant; the shim must too."""
    from qsiprep.cli.parser import parse_args

    bids, out = _minimal_bids(tmp_path)
    parse_args([
        str(bids), str(out), 'participant',
        '--infant', '--output-resolution', '2',
        '--anatomical-template', 'MNI152NLin2009cAsym',
    ])
    assert not any(
        s.startswith('MNI152NLin2009cAsym') for s in config.workflow.output_spaces
    )
```

Reuse whatever minimal-BIDS helper `test_cli.py` already has for the other `parse_args` tests
rather than adding `_minimal_bids`; check with `grep -n "def _.*bids\|parse_args(" qsiprep/tests/test_cli.py | head`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py -q -p no:cacheprovider -k "infant"`
Expected: FAIL — MNIInfant appended in both cases.

- [ ] **Step 3: Replace the shim**

In `qsiprep/cli/parser.py`:

```python
    # The infant template stands in for MNI152NLin2009cAsym, not alongside it.
    default_template = 'MNIInfant:cohort-auto' if opts.infant else 'MNI152NLin2009cAsym'

    if not given:
        if legacy_resolution is None:
            fail(
                '--output-spaces is required and must include at least one "acpc" space, '
                'for example: --output-spaces acpc:res-2mm MNI152NLin2009cAsym'
            )
        given = [f'acpc:res-{_format_mm(legacy_resolution)}']
        # --infant replaced the template outright on the legacy path, so an explicit
        # --anatomical-template does not survive it. Keeping both would add an adult
        # SyN and adult-space anatomicals to a run that never had them.
        given.append(default_template if opts.infant else (legacy_template or default_template))

    # Any infant template already anchors AC-PC (see INFANT_ANCHORS); only add the
    # default one when the request names none.
    if opts.infant and not any(s.split(':')[0] in INFANT_ANCHORS for s in given):
        given.append('MNIInfant:cohort-auto')
```

Add `INFANT_ANCHORS` to the existing `from ..utils.spaces import ...` in that module.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py qsiprep/tests/test_cli_run.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Update the migration table**

`docs/usage.rst` documents `--output-resolution 2 --infant` as
`--output-spaces acpc:res-2mm MNIInfant:cohort-auto`. That is now what the code does in all
cases; confirm the table says nothing about `--anatomical-template` surviving `--infant`:
`grep -n "infant" docs/usage.rst`

- [ ] **Step 6: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_cli.py docs/usage.rst
git commit -m "fix: --infant ensures an infant anchor rather than forcing MNIInfant

UNCInfant already anchors AC-PC, so appending MNIInfant beside it bought a second
SyN and unrequested derivatives. On the legacy path --infant replaced the template
outright, so keeping an explicit --anatomical-template alongside it changed what a
deprecated invocation produces.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 5: Reject physical sizes on standard spaces

Review finding 8 (Medium). A standard space with an `mm`-kind resolution gets no `res` entity
(only `label` kind does), yet `parse_output_spaces` dedups on `str(spec)` and keeps it distinct
from the bare template. The CLI only warns. Building `init_anat_derivatives_wf` with
`['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin2009cAsym:res-2mm']` yields 10 sinks
resolving to 7 paths: both preproc sinks write
`sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz`, likewise mask and dseg. Only the
xfm sinks are guarded by `transforms_written`. Since mm sizes on standard spaces are documented
as unimplemented, reject them rather than warn.

**Files:**
- Modify: `qsiprep/cli/parser.py:1238-1250` (the `unimplemented` warning)
- Test: `qsiprep/tests/test_cli.py`, `qsiprep/tests/test_utils_spaces.py`

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_cli.py`:

```python
def test_mm_resolution_on_a_standard_space_is_rejected(tmp_path, capsys):
    """It writes no res- entity, so it would collide with the bare template."""
    from qsiprep.cli.parser import parse_args

    bids, out = _minimal_bids(tmp_path)
    with pytest.raises(SystemExit):
        parse_args([
            str(bids), str(out), 'participant',
            '--output-spaces', 'acpc:res-2mm', 'MNI152NLin2009cAsym:res-2mm',
        ])
    assert 'MNI152NLin2009cAsym:res-2mm' in capsys.readouterr().err
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py -q -p no:cacheprovider -k "mm_resolution_on_a_standard"`
Expected: FAIL — no `SystemExit`; the parser warns and proceeds.

- [ ] **Step 3: Turn the warning into a failure**

In `qsiprep/cli/parser.py`, replace the warning branch:

```python
    # A physical size on a standard space parses but does nothing: nothing resamples
    # to it and no res- entity is written, so it lands on the same filenames as the
    # bare template. Reject it rather than silently overwriting.
    unimplemented = [
        str(spec)
        for spec in specs
        if spec.standard and spec.resolution is not None and spec.resolution.kind == 'mm'
    ]
    if unimplemented:
        fail(
            'Physical sizes on standard spaces are not implemented: '
            f'{", ".join(unimplemented)}. QSIPrep writes standard-space anatomicals on '
            "the template's own grid, so use a TemplateFlow res- label (for example "
            'MNI152NLin2009cAsym:res-2) or drop the resolution.'
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py qsiprep/tests/test_utils_spaces.py -q -p no:cacheprovider`
Expected: PASS. If any existing test asserts the old warning text, update it to expect the
failure.

- [ ] **Step 5: Update the docs**

`docs/usage.rst` describes mm-on-standard as accepted-but-unimplemented. Change that sentence to
say it is rejected, and point at the `res-` label form.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_cli.py docs/usage.rst
git commit -m "fix: reject physical sizes on standard spaces instead of warning

They write no res- entity, so a bare template and its res-<n>mm variant resolved to
one set of filenames and the second silently overwrote the first.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 6: Register non-anchor spaces from ACPC

Review finding 3 (Blocker) — the most consequential finding, and the reason Tasks 6-8 are
sequenced together.

Inside `init_anat_normalization_wf`, the SyN moving image is `rigid_acpc_resample_anat`'s
output, which applies the rigid transform this workflow extracted from *its own* `acpc_reg`
against *its own* template (`volume.py:1163-1249`). For the anchor that is correct: its rigid
transform *is* the ACPC transform. But each additional standard space gets its own instance fed
the raw `bias_corrected` image and its own template, so its `to_template_nonlinear_transform`
maps from "raw rigidly aligned to template B" to B — not from ACPC.

`t1_preproc`, `t1_mask` and `t1_seg` are produced with the *anchor's* rigid transform.
`init_anat_derivatives_wf` applies space B's composite to those anchor-frame images
(`resample_std_preproc/_mask/_dseg`, `volume.py:1975-1993`) and writes the same composite as
`from-ACPC_to-<B>_mode-image_xfm.h5`. Unless `rigid_A == rigid_B` exactly, every non-anchor
space's `desc-preproc`/mask/dseg and its xfm are off by `rigid_B⁻¹ ∘ rigid_A` — several mm for
an infant anchor with an adult standard space.

**Deliberate tradeoff:** feeding non-anchor normalizations the ACPC image makes them depend on
the anchor's normalization, so they no longer run in parallel with it. Correctness over
wall-clock; say so in the commit message.

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:1072-1074` (signature), `:1140-1260` (body),
  `:390-427` (fan-out call site)
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `init_anat_normalization_wf(spec, has_rois=False, nonlinear=True, moving_is_acpc=False, name=...)`.
  When `moving_is_acpc=True` the workflow skips `acpc_reg`, `disassemble_transform`,
  `extract_rigid_transform` and the `rigid_acpc_resample_*` nodes entirely, feeds
  `inputnode.anatomical_reference` straight to `anat_nlin_normalization.moving_image` and
  `inputnode.brain_mask` to `moving_mask`, and leaves `outputnode.to_template_rigid_transform`,
  `from_template_rigid_transform` and `to_template_affine_transform` undefined — nothing
  consumes them for a non-anchor space.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_workflows_native.py`:

```python
def test_non_anchor_normalization_starts_from_acpc(tmp_path):
    """A from-ACPC transform must actually start at ACPC.

    Each normalization used to estimate its own rigid alignment to its own
    template, so a non-anchor space's composite mapped from that space's rigid
    frame -- while the images it gets applied to are in the anchor's.
    """
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
    norm_wf = wf.get_node('anat_normalization_MNI152NLin6Asym_wf')
    assert norm_wf is not None
    assert norm_wf.get_node('acpc_reg') is None, (
        'a non-anchor space must not estimate its own ACPC frame'
    )
    nlin = norm_wf.get_node('anat_nlin_normalization')
    inputnode = norm_wf.get_node('inputnode')
    edge = norm_wf._graph.get_edge_data(inputnode, nlin)
    assert edge is not None
    assert ('anatomical_reference', 'moving_image') in edge['connect']


def test_non_anchor_normalization_is_fed_the_acpc_anatomical(tmp_path):
    """The moving image must be the ACPC-resampled head, not the raw reference."""
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
    norm_wf = wf.get_node('anat_normalization_MNI152NLin6Asym_wf')
    sources = {
        src.name
        for src, dst, data in wf._graph.in_edges(norm_wf, data=True)
        for _, field in data['connect']
        if field == 'inputnode.anatomical_reference'
    }
    assert sources == {'rigid_acpc_resample_head'}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "non_anchor_normalization"`
Expected: FAIL — `acpc_reg` exists, and the source is `anat_reference_wf`.

- [ ] **Step 3: Add the `moving_is_acpc` branch to the normalization workflow**

In `qsiprep/workflows/anatomical/volume.py`, change the signature:

```python
def init_anat_normalization_wf(
    spec, has_rois=False, nonlinear=True, moving_is_acpc=False, name='anat_normalization_wf'
) -> Workflow:
```

Guard the whole ACPC-estimation block — `acpc_reg`, `disassemble_transform`,
`extract_rigid_transform` and their `workflow.connect([...])` — with `if not moving_is_acpc:`,
and add the alternative wiring. The rigid resample nodes exist only to put the moving image in
the template's frame, which is already true when the input is ACPC:

```python
    if moving_is_acpc:
        # The caller hands us the anatomical already in the anchor's ACPC frame, so
        # the composite this workflow produces genuinely maps from-ACPC. Estimating
        # a second rigid alignment here would put the transform in this template's
        # own frame while the images it is applied to stay in the anchor's -- off by
        # rigid_B**-1 . rigid_A.
        workflow.connect([
            (inputnode, anat_nlin_normalization, [
                ('anatomical_reference', 'moving_image'),
                ('brain_mask', 'moving_mask'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, rigid_acpc_resample_mask, [
                ('template_image', 'reference_image'),
                ('brain_mask', 'input_image'),
            ]),
            (inputnode, rigid_acpc_resample_anat, [
                ('template_image', 'reference_image'),
                ('anatomical_reference', 'input_image'),
            ]),
            (extract_rigid_transform, rigid_acpc_resample_anat, [('rigid_transform', 'transforms')]),
            (extract_rigid_transform, rigid_acpc_resample_mask, [('rigid_transform', 'transforms')]),
            (rigid_acpc_resample_anat, anat_nlin_normalization, [('output_image', 'moving_image')]),
            (rigid_acpc_resample_mask, anat_nlin_normalization, [('output_image', 'moving_mask')]),
        ])  # fmt:skip
```

Create `rigid_acpc_resample_anat` and `rigid_acpc_resample_mask` only in the `else` branch, and
apply the same `if moving_is_acpc:` guard to the `if has_rois:` block's
`rigid_acpc_resample_roi` (pass `inputnode.roi` straight through as `lesion_mask` when the
moving image is already ACPC). Raise early on the unsupported combination:

```python
    if moving_is_acpc and not nonlinear:
        raise ValueError(
            'moving_is_acpc=True has nothing to do without the nonlinear stage: the '
            'rigid ACPC transform it would return is the caller\'s own input.'
        )
```

- [ ] **Step 4: Feed the ACPC anatomical at the fan-out call site**

In the `for spec in standard_specs:` loop, build the workflow with the new flag and change the
moving-image connection from `anat_reference_wf` to `rigid_acpc_resample_head` (the ACPC-frame
head, the same image `ds_t1_preproc` writes):

```python
        norm_wf = init_anat_normalization_wf(
            spec,
            has_rois=has_rois,
            moving_is_acpc=True,
            name=f'anat_normalization_{label}_wf',
        )
        standard_transform_wfs.append(norm_wf)
        workflow.connect([
            (get_std_template, std_lps_wf, [
                ('template_file', 'inputnode.template_file'),
                ('mask_file', 'inputnode.mask_file'),
            ]),
            (std_lps_wf, norm_wf, [
                ('outputnode.template_lps', 'inputnode.template_image'),
                ('outputnode.mask_lps', 'inputnode.template_mask'),
            ]),
            (inputnode, norm_wf, [('roi', 'inputnode.roi')]),
            (rigid_acpc_resample_mask, norm_wf, [('output_image', 'inputnode.brain_mask')]),
            (rigid_acpc_resample_head, norm_wf, [('output_image', 'inputnode.anatomical_reference')]),
        ])  # fmt:skip
```

Confirm the node names: `grep -n "rigid_acpc_resample_head\|rigid_acpc_resample_mask = " qsiprep/workflows/anatomical/volume.py`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_t2w_derivatives.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_workflows_native.py
git commit -m "fix: register non-anchor output spaces from ACPC, not their own rigid frame

Each normalization estimated its own rigid alignment to its own template, so a
non-anchor space's composite mapped from that template's rigid frame -- while
t1_preproc, t1_mask and t1_seg, and the xfm labelled from-ACPC, are all in the
anchor's. Everything written for a non-anchor space was off by rigid_B**-1 .
rigid_A: small between adult templates, several mm for an infant anchor.

Non-anchor normalizations now wait on the anchor's ACPC resample instead of running
beside it. That serialization is the price of a transform that means what its
filename says.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 7: Reuse one registration per template and cohort

Review finding 9 (Efficiency). `reuses_anchor` keys on the res label:

```python
reuses_anchor = (
    spec.fullname == acpc_anchor.fullname
    and _label_resolution(spec) == anchor_label_resolution
)
```

A standard space differing from the anchor only by a TemplateFlow `res-` label fails this test
and gets its own full affine + SyN `antsRegistration` to the same template, though the reports
code states these differ only in the grid the template was fetched on.
`acpc:res-2mm MNI152NLin2009cAsym:res-2` runs the registration twice per subject; the CLI help's
own example `MNI152NLin2009cAsym:res-1:res-2` runs it three times. Even
`MNI152NLin2009cAsym:res-1` fails to reuse, because the bare anchor has `resolution=None` while
`GetTemplate` defaults to `'1'`.

Task 3 already removed the anchor's resolution, so after this task the comparison is purely on
`fullname`.

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:383-397`
- Test: `qsiprep/tests/test_workflows_native.py`

- [ ] **Step 1: Write the failing test**

```python
def test_res_labels_of_one_template_share_a_registration(tmp_path):
    """res- labels differ only in the grid the template was fetched on."""
    wf = _build_anat_preproc_wf(
        tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym:res-1', 'MNI152NLin2009cAsym:res-2']
    )
    registrations = {
        name.split('.')[0]
        for name in wf.list_node_names()
        if 'anat_normalization' in name
    }
    assert registrations == {'anat_normalization_wf'}, (
        f'expected one registration to MNI152NLin2009cAsym, got {sorted(registrations)}'
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "share_a_registration"`
Expected: FAIL — three registration workflows.

- [ ] **Step 3: Key reuse on the template alone**

Replace `_label_resolution`, `anchor_label_resolution` and the `reuses_anchor` expression with a
per-template cache. A template already registered — the anchor, or an earlier spec — supplies
the transforms; only the `reference_image` for derivative resampling varies by `res-` label, and
that comes from the derivatives workflow (Task 8):

```python
    # One nonlinear normalization per template+cohort, not per requested resolution:
    # a res- label changes only the grid the template was fetched on, not where the
    # registration lands. The anchor's own normalization is always built, so it seeds
    # the cache.
    registrations = {acpc_anchor.fullname: anat_normalization_wf}

    standard_transform_wfs = []
    for spec in standard_specs:
        existing = registrations.get(spec.fullname)
        if existing is not None:
            standard_transform_wfs.append(existing)
            continue

        label = _spec_node_label(spec)
        ...
        standard_transform_wfs.append(norm_wf)
        registrations[spec.fullname] = norm_wf
```

Keep the rest of the loop body from Task 6 unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider`
Expected: PASS. `test_anchor_normalization_is_not_duplicated` must still pass.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_workflows_native.py
git commit -m "perf: register once per template, not once per requested res- label

Two res- labels of one template differ only in the grid the template was fetched
on, but each got its own affine+SyN antsRegistration. The help's own example,
MNI152NLin2009cAsym:res-1:res-2, ran it three times per subject.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 8: Share the template LPS chain with the derivatives workflow

Review finding 10 (Reuse). `init_anat_derivatives_wf` builds a second `GetTemplate`
(`get_template_{label}_deriv`) and a second `init_template_lps_wf` (AFNI `Calc` plus two
`Resample`) for every standard space, duplicating the chain `init_anat_preproc_wf` already built
for that spec. Even a default run pays two TemplateFlow lookups and six AFNI subprocess launches
per subject for byte-identical outputs. More importantly the grid the anatomicals are resampled
onto is derived independently from the grid they were registered to, so any divergence between
the two call sites silently puts `space-<tpl>` images on a different grid than the
`from-ACPC_to-<tpl>` transform.

This duplication is also what made the concurrent TemplateFlow downloads that crashed a run and
prompted commit `976bc6e`. **Keep that lock** — it still guards concurrent *subjects*.

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:~218` (outputnode fields), `:~429-460`
  (merges), `:659-665` (call site), `:1665-1680` (derivatives inputnode), `:1935-1975`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Produces: `anat_preproc_wf.outputnode` gains `standard_template_lps` — a list in
  `standard_specs` order, parallel to the existing `standard_forward_transforms`. The mask
  is not exposed: the derivatives workflow only ever used `template_lps`, as a
  `reference_image`. `init_anat_derivatives_wf` consumes them as
  `inputnode.t1_std_template_lps` and selects per index with `niu.Select`, exactly as it already
  does for `t1_std_forward_transforms`.

- [ ] **Step 1: Write the failing test**

```python
def test_derivatives_reuse_the_preproc_template_chain(tmp_path):
    """The grid images are resampled onto must be the grid they were registered to."""
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
    duplicated = [
        name for name in wf.list_node_names()
        if 'get_template' in name and name.endswith('_deriv')
    ]
    assert not duplicated, f'derivatives refetch templates: {duplicated}'
    duplicated_lps = [name for name in wf.list_node_names() if '_deriv_wf' in name]
    assert not duplicated_lps, f'derivatives rebuild the LPS chain: {duplicated_lps}'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py -q -p no:cacheprovider -k "reuse_the_preproc_template_chain"`
Expected: FAIL — both lists are non-empty.

- [ ] **Step 3: Expose the template chain on the preproc outputnode**

Add `'standard_template_lps'` to the `outputnode` field list in `init_anat_preproc_wf`. Beside
`merge_std_forward_transforms`, add:

```python
        merge_std_template_lps = pe.Node(
            niu.Merge(len(standard_specs)), name='merge_std_template_lps'
        )
```

In the per-spec loop, connect each spec's LPS output into slot `index + 1` — the anchor-reusing
branch feeds `anchor_lps_wf`, the rest feed their own `std_lps_wf`. Because Task 7 made several
specs share one registration while each still has its own `res-`-fetched template, the LPS
node is per *spec*, not per registration: build `get_std_template`/`std_lps_wf` for every spec
whose `templateflow_kwargs` differ from one already built, keyed on `str(spec)`.

Then: `(merge_std_template_lps, outputnode, [('out', 'standard_template_lps')])`.

- [ ] **Step 4: Consume it in the derivatives workflow**

Add `'t1_std_template_lps'` to `init_anat_derivatives_wf`'s inputnode fields. In the per-spec
loop, delete `get_std_template` and `std_lps_wf` and replace them with a `Select`:

```python
            select_template_lps = pe.Node(
                niu.Select(index=index), name=f'select_{label}_template_lps'
            )
```

Replace every `(std_lps_wf, resample_std_*, [('outputnode.template_lps', 'reference_image')])`
with `(select_template_lps, resample_std_*, [('out', 'reference_image')])`, and add
`(inputnode, select_template_lps, [('t1_std_template_lps', 'inlist')])`.

Wire the new field at the call site in `init_anat_preproc_wf`:

```python
            ('standard_template_lps', 'inputnode.t1_std_template_lps'),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_native.py qsiprep/tests/test_output_spaces_naming.py qsiprep/tests/test_t2w_derivatives.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_workflows_native.py
git commit -m "refactor: resample standard-space anatomicals onto the grid they were registered to

init_anat_derivatives_wf refetched every template and rebuilt the mask+LPS chain,
so the grid the anatomicals land on was derived independently of the grid the
transform was estimated against -- and every subject paid two TemplateFlow lookups
and six AFNI launches per space for byte-identical output. This duplication is also
what produced the concurrent downloads fixed in 976bc6e; that lock stays, since
subjects still fetch in parallel.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 9: `--distortion-group-merge` with several ACPC resolutions

Review finding 6 (High). **This task needs a decision before it is implemented.**

With `write_derivatives=False`, the loop in `finalize.py` still instantiates and connects
`dwi_trans_wf` and `final_denoise_wf` for every ACPC spec before checking, wires only `index == 0`
to `outputnode`, and hits `continue` for the rest. nipype has no dead-subgraph pruning, so the
extra resolutions' resampling and denoising run in full and are discarded. Downstream,
`init_distortion_group_merge_wf` calls `init_dwi_derivatives_wf(source_file=...)` with no
`resolution`, so only the first resolution is written, without a `res-` entity, and no
`Resolution` sidecar is produced for `res-native*` merged outputs. No parse-time check rejects
the combination and nothing warns.

**Option A (planned below): reject the combination at parse time, and stop building the
discarded subtrees.** Honest and small. It turns a currently-accepted invocation into an error —
including one in active local use — but that invocation silently produces one resolution today.

**Option B (not planned here): support it.** Pass `acpc_specs` into
`init_distortion_group_merge_wf` and fan out its derivatives the way `finalize.py` does. That is
a feature, not a fix, and deserves its own plan.

Confirm the choice with the user before starting. The steps below implement Option A.

**Files:**
- Modify: `qsiprep/cli/parser.py` (after `specs` is parsed)
- Modify: `qsiprep/workflows/dwi/finalize.py:~330` (loop entry)
- Test: `qsiprep/tests/test_cli.py`, `qsiprep/tests/test_output_spaces_naming.py`

- [ ] **Step 1: Write the failing tests**

In `qsiprep/tests/test_cli.py`:

```python
def test_multi_acpc_with_distortion_group_merge_is_rejected(tmp_path, capsys):
    """The merge workflow writes one resolution, so asking for two is a silent loss."""
    from qsiprep.cli.parser import parse_args

    bids, out = _minimal_bids(tmp_path)
    with pytest.raises(SystemExit):
        parse_args([
            str(bids), str(out), 'participant',
            '--output-spaces', 'acpc:res-2mm', 'acpc:res-1p5mm',
            '--distortion-group-merge', 'concat',
        ])
    assert '--distortion-group-merge' in capsys.readouterr().err
```

In `qsiprep/tests/test_output_spaces_naming.py`:

```python
def test_merged_groups_build_only_the_first_resolution(tmp_path):
    """Nipype prunes nothing, so a subtree nothing consumes still runs in full."""
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'], write_derivatives=False)
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    # Truncating before multi_acpc is computed means the survivor is named as the
    # single resolution it now is.
    assert prefixes == {'dwi_trans_wf'}
```

Give `_build_finalize` a `write_derivatives=True` parameter and forward it to
`init_dwi_finalize_wf`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider -k "distortion_group_merge or merged_groups"`
Expected: FAIL — no `SystemExit`; both trans workflows built.

- [ ] **Step 3: Reject the combination at parse time**

In `qsiprep/cli/parser.py`, after `specs` is parsed and validated:

```python
    # init_distortion_group_merge_wf writes one set of derivatives with no res-
    # entity, so extra ACPC resolutions would be resampled, denoised and then
    # dropped without a trace in the output.
    acpc_count = sum(1 for spec in specs if not spec.standard)
    if acpc_count > 1 and opts.distortion_group_merge not in (None, 'none'):
        fail(
            f'--distortion-group-merge {opts.distortion_group_merge} writes a single '
            'ACPC resolution, but --output-spaces requested '
            f'{acpc_count}. Request one "acpc" space, or drop --distortion-group-merge.'
        )
```

Check the sentinel for "no merging": `grep -n "distortion_group_merge" qsiprep/cli/parser.py qsiprep/config.py | head`

- [ ] **Step 4: Stop building the discarded subtrees**

In `qsiprep/workflows/dwi/finalize.py`, narrow the list when nothing downstream can consume the
extra resolutions. This must go **above** the `# Fan out the resampling` comment block, before
`multi_acpc = len(acpc_specs) > 1` is computed — otherwise the single surviving resolution would
still be named and entitled as though it were one of several:

```python
    if not write_derivatives:
        # This unit is concatenated later by init_distortion_group_merge_wf, which
        # writes one resolution through this workflow's single-valued outputnode.
        # nipype prunes nothing, so building the rest would resample and denoise
        # them in full and then discard the result. The parser rejects this
        # combination; the guard keeps it cheap if that check is ever relaxed.
        acpc_specs = acpc_specs[:1]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli.py qsiprep/tests/test_output_spaces_naming.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Document the restriction**

Add a sentence to the `--output-spaces` section of `docs/usage.rst` saying that multiple `acpc`
resolutions cannot be combined with `--distortion-group-merge`, and why.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/workflows/dwi/finalize.py \
        qsiprep/tests/test_cli.py qsiprep/tests/test_output_spaces_naming.py docs/usage.rst
git commit -m "fix: reject multiple ACPC resolutions with --distortion-group-merge

The merge workflow writes one set of derivatives with no res- entity, so the extra
resolutions were resampled and denoised in full and then silently dropped. Fail at
parse time instead, and stop building the subtrees nothing consumes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Task 10: Final verification

- [ ] **Step 1: Drop the deleted `init_spaces` from the API docs**

`docs/api.rst:12` still lists `init_spaces` in the `qsiprep.config` automodule members; the
function was deleted in the `--output-spaces` work. Remove `, init_spaces` from that line. (The
plan's original dead-reference grep was `--include=*.py`, which is why this survived.)

- [ ] **Step 2: Run the whole suite**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/ -q -p no:cacheprovider --deselect qsiprep/tests/test_interfaces_dipy.py::test_patch2self`
Expected: the 8 baseline failures listed in Global Constraints and nothing else.

- [ ] **Step 3: Confirm the single-ACPC filenames are unchanged**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_output_spaces_naming.py -v -p no:cacheprovider`
Expected: PASS. Any changed ACPC filename breaks QSIRecon — stop and fix.

- [ ] **Step 4: Confirm a legacy invocation still builds**

Run: `micromamba run -n linc311 python -m pytest qsiprep/tests/test_cli_run.py -q -p no:cacheprovider`
Expected: PASS, with deprecation warnings on stderr.

- [ ] **Step 5: Lint**

Run: `micromamba run -n linc311 ruff check qsiprep/ && micromamba run -n linc311 ruff format --check qsiprep/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add docs/api.rst
git commit -m "docs: drop the deleted init_spaces from the config automodule

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GyEjXrw7jjBW8AA1Lgpi7R"
```

---

## Not addressed by this plan

From the review's own "pre-existing" and "lower-priority" sections — each is real, none is
introduced by this branch, and none is required to make `--output-spaces` correct:

- **Fieldmap-less SyN SDC under `--infant`** warps the bundled adult-MNI fieldmap atlas through
  an MNIInfant reverse transform. Exists on `main`; worth its own investigation.
- **`init_diffprep_hmc_wf` re-derives the anchor from config with `cohort-auto` unresolved**, so
  it passes `MNIInfant` where the SHORELine/eddy path passes `MNIInfant+3`. Latent because
  `syn.py` never reads its `template` input.
- **TemplateFlow layout queries in `parse_space_token` are uncached** and re-run per subject and
  per DIFFPREP unit at build time.
- **The HMC/SDC reference grid is `grids[0]`**, so that reference depends on the order ACPC
  tokens were typed. Defensible, but undocumented.
- **`SubjectSummaryInputSpec.output_spaces`** is a dead trait beside the new `templates` trait.
- **`_apply_output_space_deprecations` re-parses tokens** `OutputSpacesAction` already parsed.
