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
    own frame, so merging those outputs is a build-time error (D11).

## Decisions

| # | Decision |
|---|---|
| DWI spaces | `acpc`, `anat`, `dwiref`. No `session`, `subject`/`individual` or `distortion-group` *tokens*: `dwiref` covers the dwiref frames via `--dwiref-definition`. |
| Default | `acpc:res-nativemin`, both when `--output-spaces` is omitted and when a DWI space has no `res-`. |
| Native resolutions | `native`, `nativemin` and `nativemax` are computed from the runs that make up each output, not pooled across the subject. If an output combines runs with different native voxel sizes, either concatenated within a correction unit or merged across units, QSIPrep raises an error and asks for a physical size (e.g. `res-1p5mm`). |
| No `acpc` requested | Allowed, with a warning that QSIRecon reads only `space-ACPC`. |
| Anatomical derivatives for native spaces | Whenever `anat` or `dwiref` is requested, the preprocessed anatomical, brain mask and segmentation are written **once**, in native anatomical space at the anatomical's own resolution, with **no `space-` entity** and no `res-` (e.g. `sub-01_desc-preproc_T1w.nii.gz`), matching fMRIPrep. They are not resampled onto each DWI grid, and `dwiref` gets no anatomical derivatives of its own (D12). DWI resampled to `anat` keeps `space-anat`, as fMRIPrep's BOLD in T1w space keeps `space-T1w`. |
| `dwiref` with merging | A build-time error when an output merges ≥2 units at the distortion-group level (D11). |
| `res-` in filenames | On every output on a DWI grid except `res-native`, as fMRIPrep does. `acpc:res-nativemin` writes `space-ACPC_res-nativemin_...`. Already implemented on `output-spaces-new` (`0dcc27e`, D10). |
| Mask Dice | Computed once, in ACPC, and reported for every space (D4). |
| Anatomical resolution | Anatomical derivatives without a TemplateFlow `res-` label are at the **anatomical's own voxel size**, with no `res-`. This covers ACPC anatomicals, the dwiref template resampled into ACPC, and bare-template anatomicals (D9). An explicit TemplateFlow label keeps TemplateFlow's grid and its `res-`. |
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
- **Every ACPC output on a DWI grid carries `res-`**, even with one ACPC spec (`0dcc27e`, D10).
  ACPC anatomicals and the dwiref template resampled into ACPC carry none.
- **ACPC anatomicals are on the anchor template's grid, not the anatomical's own.**
  `init_anat_preproc_wf` resamples the preprocessed anatomical, brain, mask and aseg onto
  `anchor_lps_wf.outputnode.template_lps` (`anatomical/volume.py`, the `rigid_acpc_resample_*`
  nodes). That is the AC-PC anchor at TemplateFlow's default resolution: the anchor spec drops any
  `res-` label, so `GetTemplate` uses `resolution='1'`, i.e. 1 mm for MNI152NLin2009cAsym. The
  dwiref template resampled into ACPC uses the same grid, because it is resampled onto `t1_brain`.
- **Bare-template anatomicals are on TemplateFlow's default grid** (`res-1` for most templates)
  with no `res-`. An explicit label (`MNI152NLin6Asym:res-2`) uses that label's grid and writes
  `res-2`.
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
    `dwiref` outputs cannot be concatenated. A build-time error names the output and suggests
    `--dwiref-definition subject` (one shared frame) or `--distortion-group-merge none` (D11).
    Outputs with a single unit are unaffected.
- **Anatomical derivatives:** none in the dwiref frame (D12). Requesting `dwiref` writes the
  native anatomical-space anatomicals described under `anat`. Users bring them into a dwiref frame
  with the written `from-anat_to-ACPC` and `from-ACPC_to-<level>` transforms.
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
  segmentation on `init_anat_preproc_wf`'s outputnode. Write them once, whenever `anat` or `dwiref`
  is requested, at the anatomical's own resolution with **no `space-` entity** and no `res-`:
  `<source_entities>_desc-preproc_T1w.nii.gz`, `_desc-brain_mask.nii.gz`, `_dseg.nii.gz`. That is
  how fMRIPrep writes its native anatomical-space derivatives. They are not resampled onto each
  `anat` DWI grid; one anatomical-resolution set is what consumers need. No existing QSIPrep output
  in `anat/` lacks a `space-` entity apart from transforms, so these names are free. The
  integration manifests list only `space-ACPC`, `space-<template>` and `from-`/`to-` files there.
- **DWI naming:** DWI resampled into this frame keeps `space-anat`, as fMRIPrep's BOLD resampled
  to T1w keeps `space-T1w`. Only the anatomicals, which *are* the native anatomical space, go
  without the entity.

### Anatomical derivatives at the anatomical's own resolution (D9)

Every anatomical derivative without a TemplateFlow `res-` label moves to the anatomical's own
voxel size, so "no `res-`" always means "native", as in fMRIPrep:

| Output | Today | After |
|---|---|---|
| `space-ACPC_desc-preproc_<T1w\|T2w>`, `_desc-brain_mask`, `_dseg`, `_desc-aseg_dseg`, `_desc-unfatsat_T2w` | Anchor template grid, TemplateFlow default resolution | Anchor template's field of view at the anatomical's voxel size |
| `space-ACPC_desc-<level>_dwiref` (dwiref template in ACPC) | Same grid, via `t1_brain` | Follows the ACPC anatomicals |
| `space-<tpl>[_cohort-]_desc-preproc_*`, `_desc-brain_mask`, `_dseg` for a bare template | TemplateFlow default grid | Template's field of view at the anatomical's voxel size |
| The same with `:res-<label>` | TemplateFlow `<label>` grid, `res-<label>` | Unchanged |
| Native anatomical-space anatomicals (`anat`/`dwiref` requested) | Not written | The anatomical reference's own grid |

- **Grid:** niworkflows' `GenerateSamplingReference(keep_native=True)`, then the LPS reorientation:
  - *fixed*: the template in LPS (`anchor_lps_wf` / the spec's `std_lps_wf`);
  - *moving*: the anatomical reference before AC-PC alignment (`anat_reference_wf`), whose
    voxel sizes are reoriented to the template's axes;
  - no `fov_mask`: these images include the head, so the template's whole field of view is kept.
- **Only the written copies move; the internal anatomicals stay on the anchor grid.** The ACPC
  anatomicals `init_anat_preproc_wf` exposes (`t1_preproc`, `t1_brain`, `t1_mask`, `t1_seg`,
  `t1_aseg`) are inputs to DWI processing: b=0→anatomical coregistration, DIFFPREP's T2Wreg and
  DRBUDDI structurals, SynB0, the merge workflow, and reports. Moving them to another grid would
  change those registrations numerically. So the derivatives are written from separate rigid
  resamples of the anat-space images onto the anatomical-resolution grid, and the internal ones
  are untouched.
- **With that, estimation does not change.** The rigid and nonlinear registrations still use the
  TemplateFlow image as their reference, and the warp is defined in world coordinates, so only the
  grid the written outputs are resampled onto changes. The anchor grid also stays what
  `init_output_grid_wf` autoboxes for the DWI grids, so `acpc` DWI is unaffected.
- **Effect:** filenames are unchanged. For a 1 mm isotropic anatomical, so is the content. Other
  anatomicals change voxel size, possibly to anisotropic ones such as 1×1×1.2 mm. QSIRecon reads
  the ACPC anatomicals, so this belongs in the same release note as the `res-` change.
- **Tests:** `test_derivatives_reuse_the_preproc_template_chain` asserts template-space
  derivatives are resampled onto the exact grid they were registered against. It changes to
  "derived from that chain's template", still with no second TemplateFlow fetch.
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

- **`res-` on every non-`native` output on a DWI grid.** Already on `output-spaces-new`
  (`0dcc27e`): a single ACPC spec now writes `res-` too, so `--output-spaces acpc:res-2mm` writes
  `space-ACPC_res-2mm_desc-preproc_dwi`. Under this design the default becomes
  `space-ACPC_res-nativemin_desc-preproc_dwi`.
- **Anatomicals move to the anatomical's own voxel size** (D9): ACPC anatomicals, the dwiref
  template in ACPC, and bare-template anatomicals. Filenames are unchanged, and so is the content
  for 1 mm isotropic anatomicals.
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
| 1 | **Defaults and native resolutions.** Optional `--output-spaces` defaulting to `acpc:res-nativemin`; `nativemin` as the default `res-`; the "≥1 DWI space" rule and no-`acpc` warning; `res-native` writing no `res-` (the always-`res-` rule itself landed in `0dcc27e`); per-output `native*` with the mismatch error and per-output grids; SynB0's side output written as `space-distortiongroup_desc-synb0_dwiref` when no `acpc` is requested (the synthetic b=0 is already on the native b=0 grid, `fieldmap/synb0.py`). | 4 tasks |
| 2 | **`res-native` and anisotropy.** `GenerateSamplingReference` + LPS reorientation, plus the physical-size wrapper; drop the isotropy rejection; `ChooseInterpolator` pairs axes by orientation, not index; anisotropic fixture tests for every gradient writer, as a check, since grids stay axis-aligned. | 2–3 tasks |
| 3 | **`anat`.** Expose anat-space anatomical/mask/segmentation and write them with no `space-` entity (also triggered by `dwiref`); ACPC→anat stage; anat grids; `space-anat` DWI; merging; ACPC Dice reported for the space (D4). | 3 tasks |
| 4 | **`dwiref`.** First, the per-backend frame check (eddy, DIFFPREP, SHORELine). Then the coregistration-free chains and identity path, dwiref grids from the level's reference image, the D11 build-time error, ACPC Dice reported for the space (D4), and the native anatomical-space anatomicals from Phase 3. | 3–4 tasks |
| 5 | **Anatomical resolution (D9).** `GenerateSamplingReference` + LPS grids for ACPC anatomicals, the ACPC dwiref template and bare-template anatomicals; adjust the template-chain reuse test; manifests unchanged (names only); docs. Independent of Phases 1–4. | 2 tasks |

Total: about **17–19 tasks**: the `res-` rule already landed, and Phase 5 adds two.
`anat` comes first, because `dwiref` writes its native anatomical-space anatomicals.

## Settled in review

- **D11 — `dwiref` with `--distortion-group-merge` at the distortion-group level:** build-time
  error.
- **D12 — Anatomical derivatives for `dwiref`:** none in the dwiref frame. Native
  anatomical-space anatomicals, with no `space-` entity, are written instead.
- **D4 — Mask Dice outside ACPC:** computed once in ACPC and reported for every space.
- **D10 — When the `res-` rule ships:** moved into `output-spaces-new` before it merges (`0dcc27e`).
  QSIRecon must select ACPC outputs by `res-` before that release.
- **D9 — Resolution of anatomical derivatives without a TemplateFlow label:** the anatomical's own
  voxel size, with no `res-`. This applies to ACPC anatomicals, the dwiref template in ACPC and
  bare templates (Phase 5, in the follow-up PR).

No decisions remain open.


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
