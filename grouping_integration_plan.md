# Integrating `qsiprep/grouping` into the workflow — implementation handoff

**Status (2026-08-14):** the standalone grouping package is complete, reviewed,
and green (111 unit tests + 38 golden reports; `pytest
qsiprep/tests/test_grouping_*.py`). Nothing in `workflows/`, `cli/`, or
`utils/grouping.py` has been touched. This document is the plan for wiring it
in. Design history: `~/.claude/plans/i-d-like-you-to-zazzy-sphinx.md` and the
`grouping-redesign` memory file; neither is required reading — this document
is self-contained.

## What exists

```
qsiprep/grouping/
    __init__.py   build_dwi_grouping(layout, subject_data, separate_all_dwis,
                  ignore_fieldmaps, ignore_shims, ignore_fov, force_t2wreg,
                  use_synb0, strict) -> DWIGrouping
                  (docstring holds the "adding an estimation method" checklist)
    models.py     Frozen value objects: FileRecord (sidecar + bval + NIfTI-grid
                  facts), DistortionSignature, GridInfo, FieldmapEstimation,
                  DistortionGroup, ConcatenationGroup, DWIGrouping;
                  derive_output_name (the single output-naming implementation)
    metadata.py   The ONLY module that reads input files (sidecars via pybids
                  inheritance, own IntendedFor/BIDS-URI resolution, .bval
                  shelled/max-b classification, NIfTI grids). Unreadable data
                  degrades to "undetermined" and checks skip.
    inference.py  Estimation resolution (curated B0FieldIdentifier >
                  IntendedFor translation > per-session/axis heuristic),
                  application (B0FieldSource) resolution, the fieldmap-less
                  ladder (force_t2wreg > use_synb0 > inferred T2Wreg),
                  distortion/concatenation partitioning
    validation.py GroupingIssue/GroupingError; grouping-time data checks
                  (maxb-mismatch, fov-shifted/oblique/grid); check_backend()
                  per-backend feasibility; shared decision helpers:
                  structural_target(), dwi_bidirectional_axes()
    report.py     report_text() + describe_processing(grouping, backend) -
                  the user-facing previews; renders decisions computed by the
                  validation helpers, never decides anything itself
    __main__.py   python -m qsiprep.grouping <bids_dir> [flags] - pre-run
                  preview CLI, works today on any BIDS dataset

qsiprep/tests/
    grouping_scenarios.py            fixture loader (SCENARIO_BVALS /
                                     SCENARIO_NIFTIS side-tables)
    test_grouping_{models,inference,report}.py
    data/skeleton_grouping_*.yml     26 scenario fixtures
    data/grouping_reports/*.txt      38 golden reports = the behavioral
                                     contract AND the documentation.
                                     Regenerate + review diffs with:
                                     QSIPREP_REGEN_GROUPING_REPORTS=1 pytest
                                     qsiprep/tests/test_grouping_report.py
```

## Invariants — do not violate during integration

1. **The model describes data, never tool arrangements.** No `rpe_series`-style
   synthetic vocabulary enters `qsiprep/grouping/`. Tool-specific shapes are
   produced by adapters that consume the model.
2. **Curated > translated > forced > inferred**, and every decision carries its
   `Provenance`. A real fieldmap always beats fieldmap-less methods.
3. **Estimation membership ≠ concatenation membership** (borrowing). An
   estimation may use series from outputs it doesn't belong to.
4. **All input reading stays in `metadata.py`**; downstream code never re-reads
   sidecars, bvals, or headers (this kills the legacy layout re-reads).
5. **`check_backend` is the single feasibility authority.** Workflow builders
   must not keep their own compatibility `raise`s (e.g. today's
   `'topup' in pepolar_method` checks in `hmc_sdc.py:182` / `diffprep.py:536`);
   they assume a validated grouping.
6. **Preview prose and pipeline behavior derive from the same helpers**
   (`structural_target`, `dwi_bidirectional_axes`, signature counts). Adapters
   import those helpers; they never re-derive the logic.
7. Issues are structured (`code`, `severity`, `scope`); consuming code reads
   codes, never message text.

## Decisions already made (do not re-litigate)

- Breaking release; PR #967 will be closed unmerged.
- Inferred concatenation merges session-wide shim-compatible corrected runs
  (loud `inferred-concat-merge` warning; MultipartID / `--separate-all-dwis`
  are the escape hatches).
- eddy **requires shelled** data (`eddy-requires-shelled` error on fsl/mixed);
  TORTOISE takes either, warns on shelled/non-shelled mixtures in one output.
- Max-b spread > 100 in one output → warning (scanners change acquisition
  parameters with max b).
- FoV tiers: translation offset → warning with ShimSetting evidence;
  orientation difference → error, downgradable via `ignore_fov`; grid-size
  difference → hard error, never downgradable.
- SyNb0 is a **target provider**: TOPUP's missing blip, T2Wreg's registration
  target, and DRBUDDI's `--structural` image — preferred **over a real T2w**
  when requested (`validation.structural_target`).
- TOPUP+DRBUDDI two-stage is only worthwhile with reverse-PE dMRI *series*
  (`dwi_bidirectional_axes`); otherwise warn `drbuddi-refinement-not-useful`
  (soft-block) and the second stage is TOPUP+eddy → T2Wreg against the
  structural target, else single-stage.
- SHORELine and SyN-SDC are deprecated; nothing new is built for them. The
  adapter may emit shapes SHORELine still understands, but no design bends
  for it. SyN-SDC should still be usable, though.

## Phase 1 — legacy adapter (this PR, no behavior change yet)

New `qsiprep/grouping/adapters.py`:

```python
def backend_for_config(hmc_model: str, pepolar_method: str) -> str:
    # 'eddy' + TOPUP -> 'fsl'; 'eddy' + *DRBUDDI* -> 'mixed';
    # 'tortoise' -> 'tortoise'  (shoreline models: treat as 'tortoise'-like
    # DRBUDDI semantics until removal; do not add a fourth backend)

def to_legacy_scan_groups(grouping: DWIGrouping) -> tuple[list[dict], dict]:
    # one legacy dict per ConcatenationGroup; returns (scan_groups,
    # identity concatenation_scheme) matching group_dwi_scans' contract
```

Per-output mapping (method of the applied estimation → legacy
`fieldmap_info`):

| Model state | Legacy emission |
|---|---|
| PEPOLAR, both polarities of dMRI in the output | `suffix='rpe_series'`: plus-polarity files in `dwi_series`, minus in `fieldmap_info['rpe_series']`, epi fmap sources in `['epi']`. `dwi_series_pedir` from the plus signature. |
| PEPOLAR, epi fmap only | `suffix='epi'`, `epi=[sorted fmap sources]` |
| PEPOLAR with **borrowed** dwi series (sources outside the output) | borrowed series' paths go into `fieldmap_info['epi']` — `get_best_b0_topup_inputs_from` already treats `epi_fmaps` as extra b=0 candidates keyed by their own sidecars, so borrowing rides existing plumbing |
| PHASEDIFF / PHASES / DIRECT | reconstruct `{'suffix': 'phasediff'\|'phase1'\|'fieldmap', <suffix>: path, 'magnitude*': paths}` from sources by suffix; fill `fieldmap_info['metadata']` from `FileRecord.metadata` (removes the layout re-read at `workflows/dwi/base.py:173`) |
| ANAT_CONTRAST (T2Wreg) | `{'suffix': None}` — the tortoise path already expresses T2Wreg as fieldmap-less + `t2w_sdc=True`; the adapter also exposes a per-grouping `structural_target()` result for base.py to drive `t2w_sdc`/`additional_t2ws` |
| SYNB0 | `NotImplementedError` with a clear message — the synthesis workflow lands with the separate SyNb0 plan; do not invent a fake suffix |
| None | `{'suffix': None}` |

Rules: every dict freshly built (the legacy in-place mutation bugs at
`utils/bids.py:684`, `finalize.py:150`, `dwi/util.py:210` must not find shared
state to corrupt); `concatenated_bids_name` = `ConcatenationGroup.output_name`.

Also in this PR: promote the signature-count helper if the adapter needs it,
and **add the adapter as touchpoint 6 in the `__init__.py` extension
checklist**.

Tests: differential comparison against `utils.grouping.group_dwi_scans` on
`skeleton_simple_multiped.yml` and `skeleton_complex_relpaths.yml`; every
intentional divergence asserted explicitly (not discovered). Known intended
divergences: same-session runs with independent fieldmaps now merge (rule-b),
curation/linkage outranks the reverse-PE heuristic, duplicate output names
error instead of silently collapsing.

## Phase 2 — switch-over (the high-risk PR)

In `workflows/base.py::init_single_subject_wf`:

1. Replace the `group_dwi_scans` call (line ~363) with:
   `grouping = build_dwi_grouping(..., strict=False)`;
   `backend = backend_for_config(config.workflow.hmc_model, config.workflow.pepolar_method)`;
   collect `grouping.errors + [i for i in check_backend(grouping, backend) if i.severity == 'error']`
   and raise `GroupingError` listing all of them at once (constructION-time,
   before any node exists — this is where "non-shelled data never reaches
   eddy" is enforced). Log warnings; log `report_text(grouping)` and
   `describe_processing(grouping, backend)` at workflow level.
2. `scan_groups, concatenation_scheme = to_legacy_scan_groups(grouping)`;
   everything downstream unchanged for now.
3. Delete the `--force-syn` override block (base.py:400-402) — it silently
   drops reverse-PE data.
4. `summary.inputs.dwi_groupings = grouping.to_dict()`; rewrite
   `utils/bids.py::scan_groups_to_sidecar` as
   `grouping_to_sidecar(grouping, multipart_id)` (records estimation id,
   sources, provenance; fixes its in-place mutation).
5. Remove the builder-local feasibility raises superseded by `check_backend`
   (`hmc_sdc.py:182`, `diffprep.py:536-540`).
6. Delete `qsiprep/utils/grouping.py`, `tests/test_utils_grouping.py` (its
   live scenarios are covered by the grouping scenario tests; port anything
   found missing first), the duplicate `_get_concatenated_bids_name` in
   `workflows/dwi/util.py` (use `models.derive_output_name`), and the dead
   `'rpe_series' in scan_groups` check at `finalize.py:151`.

Gate: full CircleCI dataset matrix. Output names shift for some layouts
(rule-b merges) — produce a changelog table of before/after names per CI
dataset. This PR should contain **no** behavior changes beyond the enumerated
divergences from Phase 1.

## Phase 3 — CLI surface

- Add `shims` and `fov` to the `--ignore` choices → `ignore_shims`,
  `ignore_fov` (spelling settled: reuse the `--ignore` idiom; a policy enum
  can supersede later if resampling ever becomes a feature).
- Remove `--use-syn-sdc` (dead) and `--force-syn` (data-dropping) with parser
  errors carrying migration hints.
- Redefine `--distortion-group-merge`: within-concatenation-group strategy,
  default `concat` (matches today's actual merged outputs); `average` =
  post-correction opposing-PE averaging validated at adapter time; `none` =
  per-distortion-group outputs. Rewire `distortion_group_merge.py` to consume
  concatenation groups (its input has been a degenerate identity map since
  PR #966).
- `--separate-all-dwis` keeps its name; document that SDC is now preserved
  for singletons.
- `--force-t2wreg` and the SyNb0 flag (`--use-synb0-sdc` per the SyNb0 plan)
  map to `force_t2wreg`/`use_synb0`. Open (minor): whether synb0-as-target
  and synb0-as-gap-filler ever need separate flags; currently one flag does
  both, split is a one-liner later.

## Phase 4 — native backend plans (kills tool-shaped data handling)

- `to_eddy_plan(grouping, multipart_id)`: acqp rows + per-volume index derived
  from `DistortionSignature`s instead of `interfaces/epi_fmap.py` re-reading
  sidecars (keep `read_nifti_sidecar` only as a detached-node fallback). Pass
  `FileRecord.shelled` through to eddy's existing `is_shelled` input so
  `--data_is_shelled` is set deliberately. Assert acqp/index bit-compatibility
  against current outputs on the CI datasets.
- `to_drbuddi_plan(grouping, multipart_id)`: explicit up/down file lists and
  per-volume blip assignments from the polarity distortion groups (replaces
  the assert-exactly-2 in `split_into_up_and_down_niis`); `structural_target`
  drives the `--structural`/T2Wreg image (synthetic b=0 when requested,
  overriding T2w). Lets `diffprep.py` run DIFFPREP per distortion group
  without the concat-then-resplit round trip.

## Phase 5+ (separate efforts, anticipated)

`pre_hmc.py` consuming per-distortion-group lists directly (the `rpe_series`
shape then dies entirely); SyNb0 synthesis workflow (its own plan:
`~/.claude/plans/although-this-branch-is-staged-sedgewick.md` — model already
in the container image, do not vendor); SHORELine + SyN-SDC removal
(`init_b0_hmc_wf` and `CalculateCNR` are shared and must be extracted first);
future `--grouping-json` curation injection.

## Verification per phase

1. Adapter: differential tests green; grouping suite green;
   `test_utils_grouping.py` untouched and green.
2. Switch-over: full CI matrix; before/after output-name table reviewed; the
   grouping report visible in the workflow log for every CI dataset.
3. CLI: parser tests incl. migration-hint errors.
4. Native plans: bit-compatibility assertions for acqp/index; DRBUDDI CI jobs
   (`diffprep_*`, `drbuddi_*`).
Throughout: golden reports are the contract — any diff to
`qsiprep/tests/data/grouping_reports/` must be intentional and reviewed.

## Cautions

- The tortoise path currently hard-fails under default `--pepolar-method
  TOPUP`; once `backend_for_config`+`check_backend` own feasibility, give
  this a proper error message (or default pepolar_method per hmc_model).
- `DWIGrouping.application_candidates` may reference estimations kept solely
  for report lookup ("(not used)" entries) — adapters should iterate applied
  estimations (via `DistortionGroup.b0field_source`), never assume every
  estimation is applied.
- Metadata inheritance: `index_subject` relies on `layout.get_metadata`;
  workflow nodes that run detached from a layout must receive values from the
  grouping, not re-read (see invariant 4).
- `local env`: `test_cli.py` fails collection for lack of `nireports` —
  pre-existing, unrelated.
