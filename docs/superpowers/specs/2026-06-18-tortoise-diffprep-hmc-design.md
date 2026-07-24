# TORTOISE DIFFPREP as a first-class qsiprep HMC backend

**Date:** 2026-06-18
**Status:** Approved design, ready for implementation planning
**Target repo:** `qsiprep` (PennLINC/qsiprep checkout)

## Context

`abcd_vs_csdsi/PLAN.md` describes a TORTOISE DIFFPREP HMC backend that the
`csdsi_preproc` wrapper depends on (it imports
`qsiprep.workflows.dwi.diffprep.init_diffprep_hmc_wf` and
`qsiprep.interfaces.tortoise.DIFFPREP`). That backend **does not exist in this
qsiprep checkout** — this is stock PennLINC qsiprep:

- `interfaces/tortoise.py` has `DRBUDDI`, `Gibbs`, `TORTOISEConvert`,
  `GatherDRBUDDIInputs`, but **no** `TORTOISEProcess`, `DIFFPREP`,
  `okan_decompose`, or `DIFFPREPMotionParams`.
- `cli/parser.py` `--hmc-model` choices are `['none', '3dSHORE', 'eddy', 'tensor']`.
- There is no `workflows/dwi/diffprep.py`.

This spec covers building that backend into qsiprep as a first-class,
upstream-quality `--hmc-model` option set, using `PLAN.md` as the blueprint.

## Why DIFFPREP

SHORELine handles motion on non-shelled compressed-sensing DSI (via 3dSHORE
prediction targets) but does **not** correct eddy currents. FSL eddy corrects
eddy currents but requires shells. TORTOISE v4 DIFFPREP fits a SHORE/MAPMRI
signal model under the 24-parameter Okan quadratic transform — it works on
arbitrary q-space **and** includes eddy correction. For CS-DSI cohorts this is
the missing correction.

## Decisions (locked)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Integration depth | **Upstream-quality, full pipeline** | First-class `--hmc-model` option, mergeable as a PennLINC PR; also satisfies the `csdsi_preproc` wrapper import. |
| CLI surface | **Three `--hmc-model` choices** (`diffprep_motion`, `diffprep_quadratic`, `diffprep_cubic`) | Matches PLAN.md + the wrapper's existing strings; `base.py` branches on `hmc_model.startswith('diffprep_')`. |
| Transform model | **Bake-in (like eddy)** | Use TORTOISE's resampled DWI + TORTOISE-rotated bvecs; emit identity per-volume affines. Lowest risk. Accepts one extra interpolation if SDC runs afterward. |
| SDC scope | **HMC-only, reuse existing SDC** | `init_diffprep_hmc_wf` does motion+eddy only; susceptibility correction stays with qsiprep's existing fieldmap/DRBUDDI machinery. |
| `--diffprep-config` | **In scope** | Mirrors `--eddy-config`; advanced TORTOISE knobs overridable via JSON. |

**Alternative set aside:** decompose-to-affine + voxel-shift-map (port
`OkanQuadraticTransform` → per-volume 4×4 affine + 1D shift map so HMC composes
with SDC/template warps in a single interpolation, the "qsiprep way" like
SHORELine). More faithful but far higher risk; needs the C++ port and a
correctness test. Deferred — a one-line seam comment will mark where it slots in.

## Components

### 1. CLI / config plumbing

- **`cli/parser.py`**: extend `--hmc-model` choices to
  `['none', '3dSHORE', 'eddy', 'tensor', 'diffprep_motion', 'diffprep_quadratic', 'diffprep_cubic']`;
  add `--diffprep-config` (JSON path, mirroring `--eddy-config`); update help text.
- **`config.py`**: add `workflow.diffprep_config` plus a resolved execution-section
  path, following the existing `eddy_config` pattern.
- **`data/diffprep_params.json`** (NEW): default TORTOISE knobs
  (`b0_id=-1`, `is_human_brain=true`, `rot_eddy_center=isocenter`, …).
- **`utils/misc.py`**: optional `validate_diffprep_config` parallel to
  `validate_eddy_config`.

### 2. Interface layer — `interfaces/tortoise.py` (extend existing file)

- **`TORTOISEProcess`** — CommandLine interface wrapping the `TORTOISEProcess`
  binary in DIFFPREP mode. Inputs: corrected-format `.nii` + `.bmtxt` + mask
  (+ optional structural), config JSON, transformation order
  (rigid/quadratic/cubic), b0 id. Outputs: corrected DWI, TORTOISE-rotated
  `.bmtxt`/bvecs, per-volume transform text.
- **`DIFFPREPMotionParams`** (SimpleInterface) — parse the per-volume transform
  text → translations (mm) / rotations (rad) in qsiprep's confounds/QC format
  (reads the rigid 6 of the 24 Okan parameters).
- **Reuse existing `TORTOISEConvert`** for `.nii.gz` + bval/bvec + mask →
  `.nii` + `.bmtxt`.
- **`generate_diffprep_boilerplate(...)`** for methods text (sibling to
  `generate_drbuddi_boilerplate`).
- **Not** building `okan_decompose` (bake-in needs only TORTOISE's own outputs);
  leave a one-line comment marking the seam for a future decompose path.

### 3. Workflow — `workflows/dwi/diffprep.py` (NEW)

`init_diffprep_hmc_wf(scan_groups, source_file, t2w_sdc, dwi_metadata, name='diffprep_hmc_wf')`
must expose the **identical inputnode/outputnode contract** as
`init_fsl_hmc_wf` / `init_qsiprep_hmcsdc_wf`.

Output contract base.py consumes (verified): `dwi_files_to_transform`,
`bvec_files_to_transform`, `to_dwi_ref_affines`, `to_dwi_ref_warps`,
`b0_template`, `b0_template_mask`, `bval_files`, `b0_indices`, `sdc_method`,
`motion_params`, `cnr_map`, `hmc_optimization_data`, `slice_quality`,
`pre_sdc_template`, `sdc_scaling_images`, and the SDC report fields (left
undefined / pass-through in HMC-only mode).

Internals (all LPS+, matching the non-eddy `pre_hmc` orientation so b0 templates
align with SHORELine for the downstream intramodal/template stages):

```
inputnode.dwi_file / bval / bvec / mask
  → TORTOISEConvert                       → (.nii, .bmtxt, mask.nii)
  → TORTOISEProcess(--transform <rigid|quadratic|cubic>,
                    --config diffprep_params.json)
                                          → corrected .nii + rotated .bmtxt
                                            + per-volume transforms
  → convert back to LPS+ .nii.gz, split into per-volume files
  → b0 extraction                         → b0_template + b0_template_mask
  → DIFFPREPMotionParams                  → motion_params
  → outputnode:
        dwi_files_to_transform   = corrected split volumes (baked in)
        bvec_files_to_transform  = TORTOISE-rotated bvecs (no re-rotation)
        to_dwi_ref_affines       = identity per volume
        sdc_method               = 'None'
        + remainder of the contract
```

### 4. base.py branching

In `init_dwi_preproc_wf`, add:

```python
elif config.workflow.hmc_model.startswith('diffprep_'):
    hmc_wf = init_diffprep_hmc_wf(
        scan_groups=scan_groups,
        source_file=source_file,
        dwi_metadata=dwi_metadata,
        t2w_sdc=t2w_sdc,
        name='hmc_sdc_wf',
    )
```

deriving the transform order from the `diffprep_*` suffix. The existing
`pre_hmc` orientation line already routes non-eddy models to `'LPS'`, and the
`shoreline_iters` guard already excludes diffprep, so neither needs changing.
`DiffusionSummary` already renders the `hmc_model` string.

### 5. Tests — `tests/test_interfaces_diffprep.py` (NEW)

- `DIFFPREPMotionParams` parsing against a synthetic TORTOISE transform file
  (known input → known mm/rad output).
- Workflow-construction test: `init_diffprep_hmc_wf` builds and exposes the full
  outputnode contract.
- CLI-parse test: the three new `--hmc-model` choices parse and thread into
  config, and `--diffprep-config` is honored.

### 6. Container & docs

- **Container:** TORTOISE binaries (incl. `TORTOISEProcess`) are already in the
  qsiprep image (`Dockerfile.base` copies them from `pennlinc/qsiprep-drbuddi`),
  so **no Dockerfile change**. Add a PATH-existence guard that fails with a clear
  message if `TORTOISEProcess` is missing.
- **Docs:** update `usage.rst` / workflow docs to describe the three options and
  when to use them (non-shelled / CS schemes; eddy-current correction without
  shells).

## Error handling

- Validate `TORTOISEProcess` on `PATH` at workflow build; clear failure message
  otherwise.
- Validate `--diffprep-config` JSON if supplied.
- Bail clearly if a brain mask is unavailable (TORTOISE requires one).
- No shell-count enforcement — DIFFPREP is valid on any q-space.

## Data flow (one DWI series)

```
pre_hmc_wf (denoise + reorient to LPS+)
  → init_diffprep_hmc_wf (TORTOISEConvert → TORTOISEProcess → back-to-LPS + split
                          → b0 template + motion params → contract)
  → b0_coreg_wf / confounds_wf / resampling_wf   (unchanged)
```

## Open items to verify during implementation

1. Exact `TORTOISEProcess` flag names for transformation order and config input
   (e.g. `--transformation_type` / `--correction_mode`) — confirm against the
   binary's `--help` inside the qsiprep image.
2. The precise format of TORTOISE's per-volume transform output that
   `DIFFPREPMotionParams` parses (the rigid 6 of the 24 Okan parameters).

## Out of scope

- DRBUDDI-integrated DIFFPREP (single-pass motion+eddy+susceptibility).
- `okan_decompose` / decompose-to-affine resampling.
- Changes to the `csdsi_preproc` wrapper (its import contract is satisfied by the
  new `init_diffprep_hmc_wf` + interfaces).
