# DWI output in the anatomical space, and resolution handling

Design spec and feasibility evaluation — 2026-10-02 (revised after review)
Branch: follow-up to `output-spaces-new`, as a separate PR
Related:
- `2026-08-26-output-spaces-design.md`: the `--output-spaces` feature this extends.
- `../plans/2026-09-11-distortion-group-merge-resolutions.md`: per-resolution merging, executed
  as `4c97ad2`..`a734d2b`. Phase 0 here builds on it.

Status: **proposal**; nothing here is implemented.

## Summary

- Add one DWI output space besides ACPC: **`anat`**, the anatomical reference frame before AC-PC
  alignment. It gets the same files as other spaces: preprocessed DWI, plus the preprocessed
  anatomical, brain mask and segmentation.
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
--output-spaces anat acpc:res-1p5mm   # anat:res-nativemin plus 1.5 mm ACPC
--output-spaces acpc MNI152NLin6Asym  # ACPC DWI; transforms (+ anatomicals) for MNI
```

**Verdict.** Feasible and contained:
- Estimation is untouched, and `anat` needs no new estimates. Its only new transform, ACPC→anat,
  already exists.
- Every grid is built axis-aligned in LPS (see Grids). That removes the oblique gradient-table
  problem the earlier drafts carried.
- The largest change is not code but the filename contract: `res-` on every non-native output.
  That needs QSIRecon to select outputs by `res-` (see "Contract with QSIRecon").

## Decisions

| # | Decision |
|---|---|
| Default | `acpc:res-nativemin`, both when `--output-spaces` is omitted and when a DWI space has no `res-`. |
| Native resolutions | `native`, `nativemin` and `nativemax` are computed from the runs that make up each output, not pooled across the subject. If an output combines runs with different native voxel sizes, either concatenated within a correction unit or merged across units, QSIPrep raises an error and asks for a physical size (e.g. `res-1p5mm`). |
| No `acpc` requested | Allowed, with a warning that QSIRecon reads only `space-ACPC`. |
| Native frames | `anat` only. No `distortion-group`, `session`, `subject`/`individual` or `dwiref` output spaces. fMRIPrep's only non-template BOLD frames are the run boldref and `anat`/`T1w`, and QSIPrep's distortion-group frame is not offered. |
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
- **`res-` is written only when there is more than one ACPC spec** (`finalize.py`
  `multi_acpc`; `distortion_group_merge.py`, `resolution=None` for a single spec). This keeps a
  single ACPC resolution on the pre-26.1 filenames. `test_output_spaces_naming.py` pins it
  (`test_single_acpc_dwi_paths_are_the_historical_ones`,
  `test_single_merged_resolution_keeps_the_historical_paths`).
- **Native resolutions are pooled across the subject.** `VoxelSizeChooser`
  (`interfaces/anatomical.py`) takes every zoom of every raw DWI run (`dwi_files=subject_data['dwi']`)
  and returns one scalar. `init_anat_preproc_wf` then builds one subject-level grid per ACPC spec
  (`dwi_sampling_grids`).
- **Grids are isotropic and axis-aligned in LPS.** `init_output_grid_wf` (`anatomical/volume.py`)
  autoboxes the anchor template, deobliques it, and resamples to `_tupleize(voxel_size)`. The
  gradient writers rely on this:
  - `MRTrixGradientTable`'s flip (`interfaces/mrtrix.py`);
  - `btable_from_bvals_bvecs` (`interfaces/dsi_studio.py`);
  - the processed QC's `DIPY` convention;
  - `GradientRotation`'s non-oblique check (`interfaces/gradients.py`).
- **The anat-space mask and segmentation exist only inside `init_anat_preproc_wf`**
  (`synthstrip_anat_wf`, `synthseg_anat_wf`). Only the `from-anat_to-ACPC`/`from-ACPC_to-anat`
  transforms are written. `acpc_inv_transform` reaches `dwi_preproc_wf` but not
  `dwi_finalize_wf`.
- **The output grid does not feed estimation.** Its only consumer outside resampling is SynB0's
  `space-ACPC_desc-synb0_dwiref` side output (`fieldmap/synb0.py`), which takes the first ACPC
  grid.

## Grammar

```
token   ::= space (":" key)*
space   ::= "acpc" | "anat" | <TemplateFlow template>
key     ::= "res-" value | "cohort-" value
```

| Value | Meaning | Allowed on | Filename |
|---|---|---|---|
| `nativemin` (**default**) / `nativemax` | Smallest/largest zoom of the output's runs, isotropic | `acpc`, `anat` | `res-nativemin` / `res-nativemax` |
| `native` | The output's runs' own voxel size per axis; may be anisotropic | `acpc`, `anat` | no `res-` |
| `<N>mm`, `<X>x<Y>x<Z>mm` | Physical size; anisotropic allowed | `acpc`, `anat` | `res-<label>` (e.g. `res-1p5mm`) |
| `<label>` | TemplateFlow `res` entity | templates | `res-<label>` (unchanged) |

- **The label is the spec as written**, matching how `Resolution.label` is used today. So
  `res-nativemin` stays symbolic in the filename, and the resolved voxel size goes in the JSON
  sidecar's `Resolution` key. The branch already writes that key for `native*`.
- **Templates keep today's rules:** TemplateFlow labels only. Physical and native sizes are
  rejected, because no DWI is resampled there.
- **Validation:** replace "≥1 `acpc`" with "≥1 DWI space" (`acpc` or `anat`). Warn when no `acpc`
  is requested. Drop the anisotropy rejection on `acpc`. Replace `SpaceSpec.standard`
  (`space != ACPC`, which would misclassify `anat`) with `is_template` / `writes_dwi`.

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
- **When it runs:** this is a build-time error, not a parse-time one. Which runs are combined is
  only known after qsiplan groups them.
- **What the zooms are measured on:** the raw runs, mapped onto world axes by orientation (see
  Grids), so an oblique or sagittal run is compared axis-for-axis correctly.

Grids therefore become **per output** rather than per subject. The subject-level
`dwi_sampling_grids` remains only for physical sizes, where every output of the subject shares
the grid.

### Grids

Every DWI grid is **axis-aligned and LPS-oriented**, so the gradient writers listed under Current
state stay correct without an oblique-frame code path:

| Space | Physical size, `nativemin`, `nativemax` | `native` |
|---|---|---|
| `acpc` | Today's `init_output_grid_wf`: autobox the anchor template, deoblique, resample | `GenerateSamplingReference(keep_native=True)` with the autoboxed anchor template as *fixed* |
| `anat` | `init_output_grid_wf` run on the anat-space brain instead of the anchor template | `GenerateSamplingReference` with the anat-space anatomical as *fixed* and its brain mask as `fov_mask` |

- **niworkflows' `GenerateSamplingReference`** (`interfaces/nibabel.py`, `_gen_reference`) takes
  the voxel sizes from the *moving* image after `nb.as_closest_canonical`, so a sagittal
  acquisition's slice spacing lands on the right world axis. It resamples the *fixed* image onto
  `np.diag(zooms)` and crops to the `fov_mask` bounding box plus two voxels. The result is
  axis-aligned, which is how fMRIPrep builds its T1w-space grids.
- **It is RAS, not LPS.** Reorient its output to LPS before use: a fixed axis flip, as
  `init_template_lps_wf` already does for templates.
- **`anat` means the anatomical *coordinate frame*,** not the anatomical image's voxel grid. An
  axis-aligned grid in the same world coordinates is still `anat` space, as in fMRIPrep. This
  settles the former D5.

## What `anat` needs

- **Chain:** the full ACPC chain, then the ACPC→anat rigid. Wire `acpc_inv_transform` into
  `dwi_finalize_wf` and append it to `ComposeTransforms` as a new stage after `b=0 to T1w`.
- **Jacobian transport:** `ApplyJacobianWeights` resamples weights through a hardcoded
  `[dwiref_affine, dwiref_warp, coreg]` stack (`interfaces/fmap.py`). It needs an extra trailing
  transform input. The rigid itself does not modulate (determinant 1), so it is transport only.
- **Gradient rotation:** the rigid's rotation is folded into each volume's affine by the existing
  `compose_affines` path.
- **Anatomical derivatives:** expose the anat-space bias-corrected anatomical, brain mask and
  segmentation on `init_anat_preproc_wf`'s outputnode. Write them through the same sinks
  `init_anat_derivatives_wf` uses for templates, as `space-anat`, resampled onto each `anat`
  spec's grid with the matching `res-` entity.
- **Merging:** `anat` is a single subject-level frame, like ACPC, so merged outputs work exactly
  as for ACPC: one merge workflow per spec.
- **Mask Dice:** with the anat-space brain mask exposed, Dice can be computed in each space against
  that space's own mask (D4).

## Effect on processing

- **Estimation is unchanged.** HMC, eddy, DIFFPREP, SHORELine, every SDC method, coregistration,
  dwiref construction and normalization run before the output grid. `anat` adds no estimate.
- **Per DWI spec, these repeat:** single-shot resampling, Jacobian transport, gradient rotation,
  N4 (one decision per output, one fit per grid), the b=0 reference and mask, processed QC, the
  derivative sinks, and a merge per spec when merging.
- **The default run changes resolution choice.** A run without `--output-spaces` gets
  `acpc:res-nativemin` per output.
  - For subjects whose runs share a voxel size, that is the acquired isotropic resolution, as
    before.
  - For subjects that combine runs of different sizes, the default now **fails** and asks for a
    physical size. Previously one size was pooled across all runs.

## Behaviour changes

- **`res-` on every non-`native` output.** This reverses the branch's current rule that a single
  ACPC spec writes no `res-`. Every existing filename gains an entity: the default output becomes
  `space-ACPC_res-nativemin_desc-preproc_dwi`, and `--output-spaces acpc:res-2mm` now writes
  `res-2mm`. Integration manifests, `test_output_spaces_naming.py`'s historical-path pins, and
  `docs/outputs.rst` change.
- **`--output-spaces` becomes optional.** The option moves out of "Required arguments", and the
  parser help, `docs/running.rst` and `docs/upgrading.rst` change.
- **Combining runs of different voxel sizes with a native resolution is an error**, including
  under the default.
- **Bare templates keep their meaning.**

## Contract with QSIRecon

Agreed:

- QSIRecon consumes **only `space-ACPC`** DWI.
- It **rejects anisotropic input** for now.
- When several ACPC resolutions exist (e.g. `res-nativemin`, `res-nativemax`, `res-1p5mm`), it
  picks one with its own `--output-spaces`. Until that is implemented, it picks with
  `--output-resolution`.

This creates a **release-ordering dependency**. Once this lands, *every* ACPC output carries
`res-`, including the default. A QSIRecon that matches the old `res`-less filenames will find
nothing. QSIRecon has to select by `res-` before, or in the same release as, this change ships.

## Phased implementation

A "task" is one reviewed commit with tests. All phases go in one follow-up PR after
`output-spaces-new` merges.

| Phase | Scope | Size |
|---|---|---|
| 0 | **Groundwork, no new behaviour.** `is_template`/`writes_dwi` replace `SpaceSpec.standard`. `acpc_specs` → `dwi_specs` across `finalize`, `base.py` and the merge path, keyed by (space, resolution). `space=` parameterized on every DWI sink and reportlet. `space` added to DWI figure filters in `reports-spec.yml`. The unused, broken to-template hook in `ComposeTransforms` removed. | 3 tasks |
| 1 | **Defaults, naming and native resolutions.** Optional `--output-spaces` defaulting to `acpc:res-nativemin`; `nativemin` as the default `res-`; the "≥1 DWI space" rule and no-`acpc` warning; `res-` on every non-`native` output (manifests, naming tests, docs); per-output `native*` with the mismatch error and per-output grids; SynB0's side output written as `space-distortiongroup_desc-synb0_dwiref` when no `acpc` is requested (the synthetic b=0 is already on the native b=0 grid, `fieldmap/synb0.py`). | 4 tasks |
| 2 | **`res-native` and anisotropy.** `GenerateSamplingReference` + LPS reorientation for `native` grids; drop the isotropy rejection; `ChooseInterpolator` pairs axes by orientation, not index; anisotropic fixture tests for every gradient writer, as a check, since grids stay axis-aligned. | 2–3 tasks |
| 3 | **`anat`.** Expose anat-space anatomical/mask/segmentation; ACPC→anat stage in `ComposeTransforms`; trailing transform in `ApplyJacobianWeights`; anat grids; `space-anat` DWI and anatomical derivatives; merging; Dice (D4). | 3 tasks |

Total: about **12–13 tasks**.

## Still open

- **D4 — Mask Dice outside ACPC.** With the anat-space mask exposed (Phase 3), Dice can be computed
  per space against that space's own anatomical mask, or once in ACPC and reported for every
  space. *Recommend per space*: it is now as cheap, and keeps every space's QC self-contained.
- **D9 — `res-` on template anatomical derivatives.** A bare template (`MNI152NLin6Asym`) writes
  anatomical derivatives on TemplateFlow's default grid (`res-1` for most templates) with no
  `res-` today. Under "`res-` on everything but `native`", should those become `res-1`, or does
  the rule apply only to DWI-carrying spaces? *Recommend DWI spaces only.* Template derivatives
  already follow TemplateFlow's convention, where an omitted `res` means the template's default.
- **D10 — When the `res-` rule ships.** It changes ACPC filenames. If `output-spaces-new` reaches a
  release first with `res`-less single-ACPC names, ACPC filenames change twice: in 26.1 and again
  in this follow-up. Moving just the naming rule into `output-spaces-new` before it merges would
  change them once. *Recommend that,* if QSIRecon can be updated for the same release.

## Relation to fMRIPrep

| | fMRIPrep | This design |
|---|---|---|
| Default | `MNI152NLin2009cAsym:res-native` (BOLD in template space) | `acpc:res-nativemin` |
| Bare template | Data resampled there at `res-native` | Transforms + anatomical derivatives only |
| Non-template output frames | Run boldref (`func`/`run`/`bold`/`boldref`/`sbref`), `anat`/`T1w` | `acpc`, `anat` |
| `res-` entity | On every output except `res-native` | Same |
| Native-resolution grid | `GenerateSamplingReference`, axis-aligned RAS | The same, reoriented to LPS |
| Isotropy | Never forced | Default `nativemin` is isotropic; `native` and anisotropic `mm` are opt-in |
| Physical sizes | Not supported (niworkflows#997) | `<N>mm`, `<X>x<Y>x<Z>mm` |
| Fit vs apply | Estimation once, one resampling per space | Same structure (Phase 0) |
