# DWI output in the dwiref and anatomical spaces, and resolution handling

Design spec and feasibility evaluation — 2026-10-02 (revised after review)
Branch: follow-up to `output-spaces-new`, as a separate PR
Related:
- `2026-08-26-output-spaces-design.md`: the `--output-spaces` feature this extends.
- `../plans/2026-09-11-distortion-group-merge-resolutions.md`: per-resolution merging, executed
  as `4c97ad2`..`a734d2b`. Phase 0 here builds on it.

Status: **proposal**; nothing here is implemented.

## Summary

- Add two DWI output spaces besides ACPC:
  - **`dwiref`**: the DWI reference frame chosen by `--dwiref-definition`.
  - **`anat`**: the anatomical reference frame before AC-PC alignment.
- Make `--output-spaces` **optional**, defaulting to **`acpc:res-nativemin`**.
- Determine native resolutions **per output**, and refuse to combine runs of different native
  resolutions.
- Allow **anisotropic** output.
- Write a **`res-` entity on every output except `res-native`**, as fMRIPrep does.

Template spaces keep today's meaning, transforms plus anatomical derivatives, and **never**
receive DWI.

```
(no --output-spaces)                  # = acpc:res-nativemin
--output-spaces acpc                  # = acpc:res-nativemin
--output-spaces acpc:res-native       # input voxel size, may be anisotropic; QSIRecon rejects it
--output-spaces dwiref acpc:res-1p5mm # dwiref:res-nativemin plus 1.5 mm ACPC
--output-spaces anat                  # anat:res-nativemin; warns that QSIRecon needs acpc
--output-spaces acpc MNI152NLin6Asym  # ACPC DWI; transforms (+ anatomicals) for MNI
```

**Verdict.** Feasible and contained:
- Estimation is untouched, and neither new space needs a new estimate. `dwiref` *drops* stages
  from the transform chain, and `anat` appends a rigid transform that already exists.
- Every grid is built axis-aligned in LPS (see Grids). That avoids an oblique-frame gradient-table
  path even for `dwiref`, whose reference image can be oblique.
- The two costly parts are not resampling:
  - the filename contract: `res-` on every non-native output, which needs QSIRecon to select
    outputs by `res-`;
  - `dwiref` under `--dwiref-definition distortion-group`, where every correction unit has its
    own frame, so those outputs cannot be merged (D11).

## Decisions

| # | Decision |
|---|---|
| DWI spaces | `acpc`, `anat`, `dwiref`. No `session`, `subject`/`individual` or `distortion-group` *tokens*: `dwiref` covers the dwiref frames via `--dwiref-definition`. |
| Default | `acpc:res-nativemin`, both when `--output-spaces` is omitted and when a DWI space has no `res-`. |
| Native resolutions | `native`, `nativemin` and `nativemax` are computed from the runs that make up each output, not pooled across the subject. If an output combines runs with different native voxel sizes, either concatenated within a correction unit or merged across units, QSIPrep raises an error and asks for a physical size (e.g. `res-1p5mm`). |
| No `acpc` requested | Allowed, with a warning that QSIRecon reads only `space-ACPC`. |
| `anat` derivatives | Preprocessed anatomical, brain mask and segmentation in `anat` space, as for templates. |
| `res-` in filenames | On every output except `res-native`, as fMRIPrep does. `acpc:res-nativemin` writes `space-ACPC_res-nativemin_...`. |
| Templates | Transforms + anatomical derivatives only; no DWI; no `:transform-only` keyword. |
| `--dwiref-definition` | Keeps `distortion-group` / `subject`, matching fMRIPrep's `--bold-coreg-level`. Its edge-case differences from fMRIPrep are tracked in a separate issue. |
| Delivery | A follow-up PR after `output-spaces-new` merges. |
| SynB0 side output with no `acpc` | Written in the distortion-group frame (see Phase 1). |

## Current state (verified 2026-10-02)

- **DWI is written only in ACPC.**
  - `init_dwi_finalize_wf` loops over `acpc_specs` (`workflows/dwi/finalize.py`).
  - Every DWI sink hardcodes `space='ACPC'`: 11 in `dwi/derivatives.py`, 4 in `finalize.py`, and
    the merge workflow's sidecar.
  - Since `83aca18`, the merge path fans out per ACPC spec.
- **`--output-spaces` is required and must include `acpc`** (`cli/parser.py`
  `_finalize_output_spaces`; `utils/spaces.py` `parse_output_spaces`). An `acpc` entry must
  carry an explicit `res-`.
- **`res-` is written only when there is more than one ACPC spec** (`finalize.py` `multi_acpc`;
  `distortion_group_merge.py`, `resolution=None` for a single spec).
  `test_output_spaces_naming.py` pins this (`test_single_acpc_dwi_paths_are_the_historical_ones`,
  `test_single_merged_resolution_keeps_the_historical_paths`).
- **Native resolutions are pooled across the subject.** `VoxelSizeChooser`
  (`interfaces/anatomical.py`) takes every zoom of every raw DWI run (`dwi_files=subject_data['dwi']`)
  and returns one scalar. `init_anat_preproc_wf` builds one subject-level grid per ACPC spec
  (`dwi_sampling_grids`).
- **Grids are isotropic and axis-aligned in LPS.** `init_output_grid_wf` (`anatomical/volume.py`)
  autoboxes the anchor template, deobliques it, and resamples to `_tupleize(voxel_size)`. The
  gradient writers rely on this:
  - `MRTrixGradientTable`'s flip (`interfaces/mrtrix.py`);
  - `btable_from_bvals_bvecs` (`interfaces/dsi_studio.py`);
  - the processed QC's `DIPY` convention;
  - `GradientRotation`'s non-oblique check (`interfaces/gradients.py`).
- **The resampling chain always ends at ACPC.** `ComposeTransforms._TRANSFORM_STAGES`
  (`interfaces/gradients.py`) composes `hmc`, `gradwarp`, `fieldwarp`, the subject-dwiref stages
  (`to b=0 affine` / `to b=0 warp`, wired only with `--dwiref-definition subject`; `base.py`
  `make_dwiref`), and `b=0 to T1w`. `ApplyJacobianWeights` (`interfaces/fmap.py`) transports
  weights through a hardcoded `[dwiref_affine, dwiref_warp, coreg]` stack.
- **The dwiref frames already exist on disk.**
  - With `distortion-group`, each unit's `b0_ref_image` is written as
    `space-distortiongroup_dwiref`, with `from-distortiongroup_to-ACPC`.
  - With `subject`, the template is written as `space-subject_dwiref`, with
    `from-subject_to-ACPC` (`base.py`).
  - A single-group subject under `subject` falls back to `distortion-group` today.
- **The DWI series are conformed, not deobliqued** (`LAS` for eddy, `LPS` otherwise; `base.py`),
  so a dwiref reference image can be oblique.
- **The anat-space mask and segmentation exist only inside `init_anat_preproc_wf`**
  (`synthstrip_anat_wf`, `synthseg_anat_wf`). `acpc_inv_transform` reaches `dwi_preproc_wf` but
  not `dwi_finalize_wf`.
- **The output grid does not feed estimation.** Its only consumer outside resampling is SynB0's
  `space-ACPC_desc-synb0_dwiref` side output (`fieldmap/synb0.py`), which takes the first ACPC
  grid.

## Grammar

```
token   ::= space (":" key)*
space   ::= "acpc" | "anat" | "dwiref" | <TemplateFlow template>
key     ::= "res-" value | "cohort-" value
```

### Spaces

| Token | Frame | Filename `space-` | fMRIPrep equivalent |
|---|---|---|---|
| `acpc` | AC-PC-aligned anatomical (today) | `ACPC` | none (no AC-PC frame) |
| `anat` | Anatomical reference before AC-PC alignment | `anat` | `anat` / `T1w` |
| `dwiref` | The `--dwiref-definition` frame: each distortion group's b=0 reference, or the subject template | `distortiongroup` or `subject`, the resolved level | `boldref`/`run`, which is always run-level in fMRIPrep |
| template | TemplateFlow space; no DWI | `<template>` | template (BOLD *is* written there) |

- **`dwiref` is named by its resolved level, never `space-dwiref`.** It follows
  `--dwiref-definition`, including today's single-group fallback from `subject` to
  `distortion-group`. The DWI then lands beside its reference image:
  `space-subject_res-nativemin_desc-preproc_dwi` next to `space-subject_dwiref`. The hyphen in
  `distortion-group` is dropped in the entity (`distortiongroup`), as `main` already does.
- **Unlike fMRIPrep's `boldref`**, which is always the run-level frame regardless of
  `--bold-coreg-level`, QSIPrep's `dwiref` follows the coregistration level. That is the
  frame the user chose to treat as the DWI reference.

### Resolutions

| Value | Meaning | Allowed on | Filename |
|---|---|---|---|
| `nativemin` (**default**) / `nativemax` | Smallest/largest zoom of the output's runs, isotropic | DWI spaces | `res-nativemin` / `res-nativemax` |
| `native` | The output's runs' own voxel size per axis; may be anisotropic | DWI spaces | no `res-` |
| `<N>mm`, `<X>x<Y>x<Z>mm` | Physical size; anisotropic allowed | DWI spaces | `res-<label>` (e.g. `res-1p5mm`) |
| `<label>` | TemplateFlow `res` entity | templates | `res-<label>` (unchanged) |

- **The label is the spec as written**, matching how `Resolution.label` is used today. So
  `res-nativemin` stays symbolic in the filename, and the resolved voxel size goes in the JSON
  sidecar's `Resolution` key. The branch already writes that key for `native*`.
- **Templates keep today's rules:** TemplateFlow labels only.
- **Validation:** replace "≥1 `acpc`" with "≥1 DWI space". Warn when no `acpc` is requested. Drop
  the anisotropy rejection on `acpc`. Replace `SpaceSpec.standard` (`space != ACPC`, which would
  misclassify `anat` and `dwiref`) with `is_template` / `writes_dwi`.

### Native resolution per output

For each final output (after `--distortion-group-merge`, if any), collect the zooms of every raw
run it contains:

- **`native`, `nativemin`, `nativemax`:** all of the output's runs must share voxel sizes. If they
  don't, fail at workflow build time, after grouping, naming the output and listing each run with
  its voxel size:

  > `sub-01` combines runs with different voxel sizes (`acq-hi`: 1.5×1.5×1.5 mm, `acq-lo`:
  > 2×2×2 mm), so `acpc:res-nativemin` is ambiguous. Request a physical size instead, for example
  > `acpc:res-1p5mm`.

- **Physical sizes** never need the check.
- **What the zooms are measured on:** the raw runs, mapped onto world axes by orientation (see
  Grids), so an oblique or sagittal run is compared axis-for-axis correctly.

Grids therefore become **per output** rather than per subject. The subject-level
`dwi_sampling_grids` remains only for physical sizes in `acpc`, where every output of the subject
shares the grid.

### Grids

Every DWI grid is **axis-aligned and LPS-oriented**, so the gradient writers listed under Current
state stay correct without an oblique-frame code path:

| Space | Field of view from | Physical size, `nativemin`, `nativemax` | `native` |
|---|---|---|---|
| `acpc` | Autoboxed anchor template | Today's `init_output_grid_wf` | `GenerateSamplingReference(keep_native=True)` |
| `anat` | Anat-space anatomical, cropped to its brain mask | `init_output_grid_wf` on the anat-space brain | `GenerateSamplingReference` |
| `dwiref` | The level's reference image (unit `b0_ref_image` or subject template), cropped to its b=0 mask | `GenerateSamplingReference` with the spacing forced | `GenerateSamplingReference` |

- **niworkflows' `GenerateSamplingReference`** (`interfaces/nibabel.py`, `_gen_reference`) takes
  the voxel sizes from the *moving* image after `nb.as_closest_canonical`, so a sagittal
  acquisition's slice spacing lands on the right world axis. It resamples the *fixed* image onto
  `np.diag(zooms)` and crops to the `fov_mask` bounding box plus two voxels. The result is
  axis-aligned, which is how fMRIPrep builds its T1w-space grids.
- **It is RAS, not LPS.** Reorient its output to LPS before use: a fixed axis flip, as
  `init_template_lps_wf` already does for templates.
- **For a physical size**, set the zooms directly instead of taking them from the moving image: a
  small wrapper, or `_gen_reference` called with a synthetic moving header.
- **A space is a coordinate frame, not a voxel grid.** An axis-aligned grid in the same world
  coordinates is still `anat` or `dwiref` space, as in fMRIPrep. So even `dwiref:res-native` is an
  axis-aligned resampling of an oblique reference, not the acquisition grid itself. Every output
  is interpolated once anyway, so this costs no extra interpolation, and it keeps every gradient
  table in a world-aligned frame.

## What each new space needs

### `dwiref`

- **Chain:**
  - Distortion-group level: `hmc`, `gradwarp`, `fieldwarp` only.
  - Subject level: those three plus `to b=0 affine` / `to b=0 warp`.
  - Never `b=0 to T1w`.
- **Work:**
  - Gate the coregistration stage per spec in `ComposeTransforms`.
  - Add a real identity path. Under eddy + TOPUP at the distortion-group level, no stage remains,
    and the current shortcut assumes coregistration is present.
  - Make `ApplyJacobianWeights`' transport stack an input instead of the hardcoded
    `[dwiref_affine, dwiref_warp, coreg]`.
  - `sdc_warp_transforms` becomes identity at the distortion-group level on its own.
- **Verify early: is `b0_ref_image` the frame each backend's chain ends in once coregistration is
  dropped?** This must hold for eddy (LAS-conformed, with no `hmc` stage), DIFFPREP (identity
  `hmc`) and SHORELine. If one backend's corrected data sits on a different reference, that
  backend needs its own frame. This is the main technical risk in the design. Settle it with a
  construction test per backend before writing derivatives.
- **Merging:**
  - At the subject level all units share one frame, so merged outputs work as for ACPC: one merge
    workflow per spec.
  - At the distortion-group level the units being merged live in *different* frames, so their
    `dwiref` outputs cannot be concatenated (D11).
- **Single-group subjects** under `subject`: follow today's fallback, so outputs are named
  `distortiongroup`. If the separate `--dwiref-definition` issue changes the fallback, `dwiref`
  follows it automatically.

### `anat`

- **Chain:** the full ACPC chain, then the ACPC→anat rigid. Wire `acpc_inv_transform` into
  `dwi_finalize_wf` and append it as a new trailing stage in `ComposeTransforms`.
- **Jacobian transport:** the same configurable stack, with the rigid appended. It is transport
  only: the determinant is 1, so it does not modulate.
- **Gradient rotation:** the rigid's rotation is folded into each volume's affine by the existing
  `compose_affines` path.
- **Anatomical derivatives:** expose the anat-space bias-corrected anatomical, brain mask and
  segmentation on `init_anat_preproc_wf`'s outputnode. Write them through the same sinks
  `init_anat_derivatives_wf` uses for templates, as `space-anat`, resampled onto each `anat`
  spec's grid with the matching `res-` entity.
- **Merging:** a single subject-level frame, so merged outputs work as for ACPC.

## Effect on processing

- **Estimation is unchanged.** HMC, eddy, DIFFPREP, SHORELine, every SDC method, coregistration,
  dwiref construction and normalization run before the output grid. Neither new space adds an
  estimate.
- **Per DWI spec, these repeat:** single-shot resampling, Jacobian transport, gradient rotation,
  N4 (one decision per output, one fit per grid), the b=0 reference and mask, processed QC, the
  derivative sinks, and a merge per spec when merging.
- **The default run changes resolution choice.** A run without `--output-spaces` gets
  `acpc:res-nativemin` per output.
  - For subjects whose runs share a voxel size, that is the acquired isotropic resolution, as
    before.
  - For subjects that combine runs of different sizes, the default now **fails** and asks for a
    physical size.

## Behaviour changes

- **`res-` on every non-`native` output.** This reverses the branch's current rule that a single
  ACPC spec writes no `res-`. Every existing ACPC filename gains an entity: the default output
  becomes `space-ACPC_res-nativemin_desc-preproc_dwi`. Integration manifests,
  `test_output_spaces_naming.py`'s historical-path pins, and `docs/outputs.rst` change.
- **`--output-spaces` becomes optional.** The option moves out of "Required arguments", and the
  parser help, `docs/running.rst` and `docs/upgrading.rst` change.
- **Combining runs of different voxel sizes with a native resolution is an error**, including
  under the default.
- **Bare templates keep their meaning.**

## Contract with QSIRecon

Agreed:

- QSIRecon consumes **only `space-ACPC`** DWI, so `dwiref` and `anat` outputs are for other tools.
- It **rejects anisotropic input** for now.
- When several ACPC resolutions exist (e.g. `res-nativemin`, `res-nativemax`, `res-1p5mm`), it
  picks one with its own `--output-spaces`. Until that is implemented, it picks with
  `--output-resolution`.

This creates a **release-ordering dependency**. Once this lands, every ACPC output carries `res-`,
including the default. A QSIRecon that matches the old `res`-less filenames finds nothing.
QSIRecon has to select by `res-` before, or in the same release as, this change ships.

## Phased implementation

A "task" is one reviewed commit with tests. All phases go in one follow-up PR after
`output-spaces-new` merges.

| Phase | Scope | Size |
|---|---|---|
| 0 | **Groundwork, no new behaviour.** `is_template`/`writes_dwi` replace `SpaceSpec.standard`. `acpc_specs` → `dwi_specs` across `finalize`, `base.py` and the merge path, keyed by (space, resolution). `space=` parameterized on every DWI sink and reportlet. `space` added to DWI figure filters in `reports-spec.yml`. Per-spec stage selection and a configurable Jacobian transport stack in `ComposeTransforms`/`ApplyJacobianWeights`, ACPC behaviour unchanged. The unused, broken to-template hook in `ComposeTransforms` removed. | 4 tasks |
| 1 | **Defaults, naming and native resolutions.** Optional `--output-spaces` defaulting to `acpc:res-nativemin`; `nativemin` as the default `res-`; the "≥1 DWI space" rule and no-`acpc` warning; `res-` on every non-`native` output (manifests, naming tests, docs); per-output `native*` with the mismatch error and per-output grids; SynB0's side output written as `space-distortiongroup_desc-synb0_dwiref` when no `acpc` is requested (the synthetic b=0 is already on the native b=0 grid, `fieldmap/synb0.py`). | 4 tasks |
| 2 | **`res-native` and anisotropy.** `GenerateSamplingReference` + LPS reorientation, plus the physical-size wrapper; drop the isotropy rejection; `ChooseInterpolator` pairs axes by orientation, not index; anisotropic fixture tests for every gradient writer, as a check, since grids stay axis-aligned. | 2–3 tasks |
| 3 | **`dwiref`.** First, the per-backend frame check (eddy, DIFFPREP, SHORELine). Then the coregistration-free chains and identity path, dwiref grids from the level's reference image, the D11 merge policy, and Dice (D4). | 3–4 tasks |
| 4 | **`anat`.** Expose anat-space anatomical/mask/segmentation; ACPC→anat stage; anat grids; `space-anat` DWI and anatomical derivatives; merging; Dice (D4). | 3 tasks |

Total: about **16–18 tasks**. Phases 3 and 4 are independent of each other.

## Still open

- **D4 — Mask Dice outside ACPC.** Series QC compares the anatomical mask with the DWI mask on a
  shared grid. For `anat` the anat-space mask will exist. For `dwiref` the anatomical mask would
  have to be pulled back through the inverse coregistration. *Recommend computing it once in ACPC*
  and reporting it for every space: it measures coregistration quality, which does not depend on
  the output space, and it avoids the extra transform for `dwiref`.
- **D9 — `res-` on template anatomical derivatives.** A bare template writes anatomical derivatives
  on TemplateFlow's default grid (`res-1` for most templates) with no `res-` today. Should those
  become `res-1`, or does the rule apply only to DWI-carrying spaces? *Recommend DWI spaces only*:
  template derivatives already follow TemplateFlow's convention, where an omitted `res` means the
  template's default.
- **D10 — When the `res-` rule ships.** It changes ACPC filenames. If `output-spaces-new` reaches a
  release first with `res`-less single-ACPC names, ACPC filenames change twice. *Recommend moving
  just the naming rule into `output-spaces-new` before it merges*, if QSIRecon can be updated for
  the same release.
- **D11 — `dwiref` with `--distortion-group-merge` at the distortion-group level.** The units being
  merged each have their own frame, so their `dwiref` outputs cannot be concatenated. Options:
  - error at build time when an output that merges ≥2 units requests `dwiref`, suggesting
    `--dwiref-definition subject` (one shared frame) or `--distortion-group-merge none`;
  - write those outputs per unit, unmerged, with a warning.

  *Recommend the error*: per-unit files from a run that asked for merging would be surprising. At
  the subject level, and for any output with a single unit, no conflict arises.
- **D12 — Anatomical derivatives in `dwiref` space.** `anat` and templates get the preprocessed
  anatomical, brain mask and segmentation. Should `dwiref` too? At the subject level that is one
  set. At the distortion-group level it is one set per unit, each needing the inverse
  coregistration. *Recommend no*, at least initially. The dwiref frames already have their b=0
  reference and DWI brain mask, and the anatomical images are easy to bring in with the written
  `from-ACPC_to-<level>` transforms.

## Relation to fMRIPrep

| | fMRIPrep | This design |
|---|---|---|
| Default | `MNI152NLin2009cAsym:res-native` (BOLD in template space) | `acpc:res-nativemin` |
| Bare template | Data resampled there at `res-native` | Transforms + anatomical derivatives only |
| Non-template output frames | Run boldref (`func`/`run`/`bold`/`boldref`/`sbref`), `anat`/`T1w` | `acpc`, `anat`, `dwiref` (at the `--dwiref-definition` level) |
| `res-` entity | On every output except `res-native` | Same |
| Native-resolution grid | `GenerateSamplingReference`, axis-aligned RAS | The same, reoriented to LPS, for every DWI space |
| Isotropy | Never forced | Default `nativemin` is isotropic; `native` and anisotropic `mm` are opt-in |
| Physical sizes | Not supported (niworkflows#997) | `<N>mm`, `<X>x<Y>x<Z>mm` |
| Fit vs apply | Estimation once, one resampling per space | Same structure (Phase 0) |
