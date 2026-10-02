# DWI output in native and anatomical spaces

Design spec and feasibility evaluation — 2026-10-02 (revised twice the same day)
Branch: `output-spaces-new`
Related:
- `2026-08-26-output-spaces-design.md`: the `--output-spaces` feature this extends.
- `../plans/2026-09-11-distortion-group-merge-resolutions.md`: per-resolution merging, executed
  as `4c97ad2`..`a734d2b`. Phase 0 here builds on it.

Status: **proposal for evaluation**, in parallel with the current output-spaces work. Nothing
here is implemented.

## Summary

Let `--output-spaces` write preprocessed DWI in two frames besides ACPC:
- each distortion group's own b=0 reference frame;
- the anatomical frame before AC-PC alignment.

Also allow anisotropic resolutions. Template spaces keep today's meaning, transforms plus
anatomical derivatives, and **never** receive DWI.

```
(no --output-spaces)                        # = acpc:res-nativemin
--output-spaces acpc                        # = acpc:res-nativemin
--output-spaces distortion-group:res-native # each distortion group's b=0 frame, input voxel size
--output-spaces anat acpc:res-1p5mm         # pre-ACPC anatomical frame, plus 1.5 mm ACPC
--output-spaces acpc MNI152NLin6Asym        # ACPC DWI; transforms (+ anatomicals) for MNI
```

**Decisions taken so far:**

1. **Default is `acpc:res-nativemin`.** It applies both when `--output-spaces` is omitted and
   when a DWI-carrying space is given without `res-`.
2. **Only the native frames fMRIPrep supports.** fMRIPrep resamples BOLD into two non-template
   frames: the run-level boldref frame (`func`/`run`/`bold`/`boldref`/`sbref`) and the anatomical
   frame (`anat`/`T1w`). QSIPrep's equivalents are `distortion-group` and `anat`.
   - Session- and subject-level templates are coregistration intermediates in fMRIPrep, never
     output spaces, so QSIPrep does not offer them as output spaces either.
   - `--dwiref-definition` keeps its `distortion-group`/`subject` values, matching fMRIPrep's
     `--bold-coreg-level`.
3. **No `:transform-only`, and no template-space DWI.** A template in `--output-spaces` writes the
   ACPC↔template transforms and the anatomical derivatives, exactly as the branch does today.
4. **Grids reuse niworkflows' `GenerateSamplingReference`** for `res-native`, as fMRIPrep does.

**Verdict.** Feasible without touching estimation:
- `distortion-group` is cheap: its reference image and transforms already exist.
- `anat` and `res-native` are moderate. Most of their cost is writing gradient tables correctly
  in a frame that may be oblique.

## Current state (verified 2026-10-02)

- **DWI is written only in ACPC.**
  - `init_dwi_finalize_wf` loops over `acpc_specs` (`workflows/dwi/finalize.py`).
  - Every DWI sink hardcodes `space='ACPC'`: 11 in `dwi/derivatives.py`, 4 in `finalize.py`, and
    the merge workflow's sidecar.
  - Since `83aca18`, the merge path follows the same per-resolution fan-out: one
    `init_distortion_group_merge_wf` per ACPC spec, fed slot *i* of each unit's finalize outputs.
- **`--output-spaces` is required and must include an `acpc` entry.** The parser enforces it
  (`cli/parser.py` `_finalize_output_spaces`), as does `qsiprep/utils/spaces.py`
  (`parse_output_spaces`). An `acpc` entry must carry an explicit `res-`.
- **The resampling chain always ends at ACPC.** `ComposeTransforms._TRANSFORM_STAGES`
  (`interfaces/gradients.py`) composes `hmc`, `gradwarp`, `fieldwarp`, the subject-dwiref stages
  (`to b=0 affine` / `to b=0 warp`, only with `--dwiref-definition subject`), and
  `b=0 to T1w`.
- **Distortion-group references already exist on disk.** Each unit's `b0_ref_image` is written as
  `space-distortiongroup_dwiref`, with `from-distortiongroup_to-ACPC` transforms (`base.py`).
- **Grids are isotropic by construction.**
  - `init_output_grid_wf` (`anatomical/volume.py`) autoboxes the anchor template, deobliques it,
    and resamples to `_tupleize(voxel_size)`.
  - `VoxelSizeChooser` (`interfaces/anatomical.py`) pools every zoom of every raw DWI run of the
    subject into one scalar.
- **The output grid does not feed estimation.**
  - `_first_sampling_grid` reaches `hmc_wf` but is never connected in `hmc_sdc.py`.
  - Its only consumer is SynB0's `space-ACPC_desc-synb0_dwiref` side output.
  - The b=0 reference and mask use SynthStrip on the b=0 itself; their declared anatomical inputs
    are never connected (`dwi/util.py` `init_dwi_reference_wf`).
- **The DWI series are conformed, not deobliqued.** They are reoriented to `LAS` (eddy) or `LPS`
  (`base.py`), so a distortion group's frame can be oblique.

## Grammar

```
token   ::= space (":" key)*
space   ::= "acpc" | "anat" | "distortion-group" | <TemplateFlow template>
key     ::= "res-" value | "cohort-" value
```

### Spaces

| Token | Frame | Filename `space-` | DWI written? | fMRIPrep equivalent |
|---|---|---|---|---|
| `acpc` | AC-PC-aligned anatomical (today) | `ACPC` | yes | none (no AC-PC frame) |
| `anat` | Anatomical reference before AC-PC alignment | `anat` | yes | `anat` / `T1w` |
| `distortion-group` | Each distortion group's own b=0 reference | `distortiongroup` | yes | `run` / `boldref` |
| template | TemplateFlow space | `<template>` | **no**: transforms and anatomical derivatives only | template (BOLD *is* written there) |

- **`distortion-group` instead of fMRIPrep's `run`/`boldref`.** A QSIPrep distortion group can
  pool several BIDS runs, so `run` would mislead, and `dwiref` would be ambiguous with the
  `--dwiref-definition subject` template. `distortion-group` is the name `--dwiref-definition`
  and the existing `space-distortiongroup_dwiref` files already use.
- **The filename entity drops the hyphen** (`distortiongroup`), because BIDS entity values are
  alphanumeric, as `main` already does. QSIPrep parses tokens itself (`utils/spaces.py`), so the
  hyphenated token needs no niworkflows support.
- **`anat`** is fMRIPrep's name, and the one the existing `from-anat_to-ACPC` transforms use.

### Resolutions

| Value | Meaning | Allowed on |
|---|---|---|
| `nativemin` (**default**) / `nativemax` | Smallest/largest input zoom, isotropic | every DWI space |
| `native` (**new**) | The input DWI's own voxel size per axis, not forced isotropic | every DWI space |
| `<N>mm`, `<X>x<Y>x<Z>mm` | Physical size; anisotropic now allowed | every DWI space |
| `<label>` | TemplateFlow `res` entity | templates (unchanged) |

- **The default is `nativemin`.** It applies when `--output-spaces` is omitted (giving
  `acpc:res-nativemin`) and when a DWI space is given without `res-`. That keeps outputs isotropic
  by default, which is what reconstruction expects, and makes anisotropic output an explicit
  choice. Explicit `acpc:res-<N>mm` commands keep their meaning.
- **Templates keep today's rules.** TemplateFlow labels only, and physical or native sizes are
  rejected, because no DWI is resampled there.
- **`native` and `nativemin`/`nativemax` read the zooms of the output's own runs**, not the whole
  subject's as `VoxelSizeChooser` does today. For a merged output that is the union of its units'
  runs. `native` requires those runs to share zooms; a mismatch is a build-time error that
  suggests `nativemin`/`nativemax`.

### Grids (reuse niworkflows)

| Space | `res-native` | `nativemin` / `nativemax` / `mm` |
|---|---|---|
| `distortion-group` | The unit's b=0 reference itself: its affine, FOV and obliquity | `ReferenceGridAtSpacing` (`interfaces/images.py`): that image's FOV and orientation at the new spacing; already used by `dwi/registration.py` |
| `acpc`, `anat` | `GenerateSamplingReference(keep_native=True)` | Today's `init_output_grid_wf` path |

**Why `GenerateSamplingReference` fits.** niworkflows' `GenerateSamplingReference`
(`niworkflows/interfaces/nibabel.py`, `_gen_reference`) takes:
- the field of view from a *fixed* image;
- the voxel sizes from a *moving* image, after reorienting it to the fixed image's orientation;
- an optional `fov_mask`, which crops to the brain's bounding box plus two voxels.

It writes `np.diag(zooms)`, so anisotropy survives. This is how fMRIPrep builds `res-native`
grids, and it solves the axis-mapping problem a hand-rolled chooser would have: a sagittal
acquisition's slice spacing lands on the right world axis. QSIPrep passes:
- *fixed*: the autoboxed anchor template (`acpc`) or the anatomical reference (`anat`);
- *moving*: the output's b=0 reference;
- *fov_mask*: the matching brain mask.

## What each space needs

### `distortion-group` — low

- **Chain:** `hmc`, `gradwarp`, `fieldwarp`. No dwiref stages and no coregistration.
- **Grid:** the unit's `b0_ref_image`.
- **Work:**
  - Gate the `b=0 to T1w` stage, and the subject-dwiref stages when
    `--dwiref-definition subject`: neither applies inside the group's own frame.
  - Give eddy + TOPUP a real identity path. There only coregistration survives today, and the
    `ComposeTransforms` shortcut assumes it is present.
  - Make `ApplyJacobianWeights`' hardcoded `[dwiref_affine, dwiref_warp, coreg]` transport stack
    an input (`interfaces/fmap.py`).
  - `sdc_warp_transforms` becomes identity on its own.
- **Merging:** distortion groups are by definition the units being merged, and they live in
  different frames. So `distortion-group` outputs are always written per unit, never merged.
  With `--distortion-group-merge` on, `distortion-group` is still allowed. It writes per-unit
  files, and the parser says so.

### `anat` — low to moderate

- **Chain:** the full ACPC chain, then the ACPC→anat rigid (`acpc_inv_transform`). That transform
  reaches `dwi_preproc_wf` but not `dwi_finalize_wf` today.
- **Grid:** `GenerateSamplingReference` on the anatomical reference for `native`; otherwise the
  autobox path, run on the anat-space brain.
- **Work:** expose the anat-space brain mask, which today exists only inside
  `init_anat_preproc_wf` (`synthstrip_anat_wf`).

### Anisotropic and oblique grids — moderate, cuts across both

- **Per-output grids.** `res-native` and per-output `nativemin` make grids depend on the output,
  not the subject. Build them in `base.py` or `finalize` from each output's units, replacing the
  subject-level `dwi_sampling_grids` for DWI-native resolutions.
- **Gradient tables in the output frame.** This is the real cost.
  - Input bvecs are deobliqued into world LPS+ (`interfaces/images.py`).
  - The writers assume world axes equal voxel axes: `MRTrixGradientTable`'s flip
    (`interfaces/mrtrix.py`), `btable_from_bvals_bvecs` (`interfaces/dsi_studio.py`), the `DIPY`
    convention of the processed QC, and `GradientRotation`'s non-oblique check
    (`interfaces/gradients.py`).
  - Every grid QSIPrep builds today is deobliqued, so these assumptions hold. The
    `distortion-group` frame at `res-native`, and `anat` if D5 keeps its own grid, can be oblique.
  - So: write `.bvec` in the output's voxel frame, and `.b` / `b_table` in its world frame. One
    tested helper, used by every gradient-table writer.
- **Interpolation.** `ChooseInterpolator` already compares zooms per axis
  (`interfaces/images.py`). It pairs axes by index, so reorient before comparing.

## Template spaces

Unchanged from the branch: transforms (`from-ACPC_to-<tpl>` and reverse) plus the preprocessed
anatomical, brain mask and dseg on the template's TemplateFlow grid (`init_anat_derivatives_wf`).
No template ever receives DWI, so:

- the to-template hook in `ComposeTransforms` stays unused. It is broken (it asserts a legacy
  `[affine, warp]` pair, while the anatomical workflow writes one `.h5` composite), so removing it
  in Phase 0 is reasonable;
- nonlinear gradient reorientation, Jacobian/SDC-map conjugation through a SyN warp, and
  graddev orientation in a nonlinear target are all out of scope;
- QSIPrep's documented position stays true: DWI is normalized after model fitting, in QSIRecon.

## Effect on processing

- **Estimation is unchanged.** HMC, eddy, DIFFPREP, SHORELine, every SDC method, coregistration,
  dwiref construction and normalization are all upstream of the output grid. Neither new space
  needs any new estimate: `distortion-group` drops stages from the chain, and `anat` appends a
  rigid transform that already exists.
- **Per DWI spec, these repeat:** single-shot resampling, Jacobian transport, gradient rotation,
  N4 (one decision per output, one fit per grid), the b=0 reference and mask, processed QC,
  derivative sinks, and a merge per spec when merging (except `distortion-group`, which is never
  merged).
- **Default runs change only in resolution choice.** A run without `--output-spaces` gets
  `acpc:res-nativemin`. That is today's behaviour for anyone who passed the acquired isotropic
  size, but for anisotropic acquisitions it differs from passing e.g. 2 mm.
- **Disk.** `distortion-group:res-native` is the smallest DWI output possible: no upsampling, no
  padding to an ACPC bounding box.

## Behaviour changes

- **`--output-spaces` becomes optional** (default `acpc:res-nativemin`). That reverses `main`'s
  "`--output-resolution` has no default and must be given explicitly" (the Required-arguments
  group in `cli/parser.py`). The parser help, `docs/running.rst` and `docs/upgrading.rst` must say
  so, and the option moves out of "Required arguments".
- **The `acpc` requirement goes.** A run can write only `distortion-group` or `anat` DWI. Such a
  run produces nothing QSIRecon reads today, so the parser should warn when no `acpc` space is
  requested.
- **Bare templates keep their meaning.** Nothing on the branch changes meaning, so no test,
  manifest or doc example needs updating for templates.

## Contract with QSIRecon

QSIRecon is not in this repo, so its input query could not be checked. Before landing any phase,
confirm:

1. Whether it filters on `space-ACPC`, or would also pick up `space-distortiongroup` and
   `space-anat` files, which would make the new outputs ambiguous to it.
2. Whether it can use anisotropic (`res-native`) input.
3. Whether a `res-` entity on the default `acpc:res-nativemin` output would be expected. It should
   not appear: a single ACPC spec writes no `res-`.

## Phased implementation

Each phase is shippable. An explicit `acpc:res-<N>mm` run is unchanged by every phase. A "task"
is one reviewed commit with tests.

| Phase | Scope | Size | Depends on |
|---|---|---|---|
| 0 | **Groundwork, no new behaviour.** Replace `SpaceSpec.standard` (`space != ACPC`) with `is_template`/`writes_dwi`. Rename `acpc_specs` → `dwi_specs` across `finalize`, `base.py` and the merge path, keyed by (space, resolution). Parameterize `space=` on every DWI sink and reportlet. Add `space` to DWI figure filters in `reports-spec.yml`. Remove the unused to-template hook in `ComposeTransforms`. | 3 tasks | merge-resolutions plan (done) |
| 1 | **Defaults and grammar.** `--output-spaces` optional with default `acpc:res-nativemin`; `nativemin` as the default `res-`; the `distortion-group` and `anat` tokens (parsed, rejected as "not yet implemented" until their phase lands); replace "≥1 `acpc`" with "≥1 DWI space" plus a warning without `acpc`. Docs: `running.rst`, `upgrading.rst`, parser help. | 2 tasks | 0 |
| 2 | **`distortion-group`.** Gating of the coregistration and subject-dwiref stages, the eddy identity path, a configurable Jacobian transport stack, the reference-image grid, per-unit writing under merging, Dice policy (D4). | 3 tasks | 1 |
| 3 | **`res-native` and anisotropy, then `anat`.** Per-output grids via `GenerateSamplingReference` and `ReferenceGridAtSpacing`; output-frame gradient tables for every writer; oblique-grid tests; `ChooseInterpolator` axis pairing; anat-space mask and grid, plus the ACPC→anat stage. | 4–5 tasks | 2 |

Total: about **12–13 tasks**. Phase 2 delivers `distortion-group` at isotropic resolutions;
Phase 3 delivers `anat` and non-isotropic output.

### Latent bugs found while scoping

All three were present on `main` and are fixed there separately (PR from branch
`fix-transform-and-merge-qc-bugs`):

- `dwiref_to_t1_warp` was silently dropped;
- SHORELine's CNR map was resampled through volume 0's head motion;
- the merged series QC lacked `space-ACPC`.

Merge `main` into this branch once that lands. The merge-workflow QC sink will conflict trivially
with `4c97ad2` (`space='ACPC'` there, `**res_entities` here; keep both).

## Decisions

Settled:

- **Template-space DWI:** not supported. Templates write transforms and anatomical derivatives
  only. No `:transform-only` keyword.
- **Default resolution and space:** `acpc:res-nativemin`.
- **Native output spaces:** `distortion-group` and `anat` only, mirroring the non-template frames
  fMRIPrep writes BOLD to. No `session`, `subject`/`individual` or `dwiref` output spaces.
- **`--dwiref-definition`:** keeps `distortion-group` and `subject`, matching fMRIPrep's
  `--bold-coreg-level`.
- **Grid construction:** reuse `GenerateSamplingReference` and `ReferenceGridAtSpacing`.

Still open:

- **D4 — Mask Dice outside ACPC.** Series QC's Dice compares the ACPC anatomical mask with the DWI
  mask on a shared grid. `init_mask_overlap_wf` resamples without a transform, and `DiceOverlap`
  fails on a shape mismatch. *Recommend computing it once in ACPC* and reporting it for every
  space: it measures coregistration quality, which does not depend on the output space.
- **D5 — `anat` grid.** The anatomical image's own, possibly oblique, grid, or world-aligned
  around it? *Recommend its own grid*, for symmetry with `distortion-group`.

## Related differences in `--dwiref-definition` (outside this design)

fMRIPrep's `--bold-coreg-level` (on `master`, unreleased as of 25.2.5) differs from QSIPrep's
`--dwiref-definition subject` in two ways. These affect coregistration, not output spaces, so
they are listed here for a separate decision:

- **Single group.** fMRIPrep builds the template anyway. niworkflows' `StructuralReference` copies
  the lone boldref and writes an identity transform, so `space-subject_boldref` and
  `from-subject_to-T1w` always exist. QSIPrep instead falls back to `distortion-group` with a
  warning, and writes no `space-subject_dwiref`.
- **Sessionwise.** fMRIPrep rejects `--bold-coreg-level subject` with
  `--subject-anatomical-reference sessionwise` at parse time, pointing to `session`. QSIPrep
  accepts the combination and quietly builds a per-session template under the `subject` name.

## Relation to fMRIPrep

| | fMRIPrep | This design |
|---|---|---|
| Default | `MNI152NLin2009cAsym:res-native` (BOLD in template space) | `acpc:res-nativemin` |
| Bare template | Data resampled there at `res-native` (niworkflows appends `:res-native`) | Transforms + anatomical derivatives only |
| Native output frames | Run boldref (`func`/`run`/`bold`/`boldref`/`sbref`), `anat`/`T1w` | `distortion-group`, `anat`, plus `acpc` |
| Session/subject templates | Coregistration intermediates (`space-session_boldref`, `space-subject_boldref`), not output spaces | Same: `space-subject_dwiref` is a reference image, not an output space |
| `res-native` grid | `GenerateSamplingReference` | The same interface for `acpc`/`anat`; the reference image itself for `distortion-group` |
| Isotropy | Never forced | Default `nativemin` is isotropic; `native` and anisotropic `mm` are opt-in |
| Physical sizes | Not supported (niworkflows#997) | `<N>mm`, `<X>x<Y>x<Z>mm` |
| Fit vs apply | Estimation once, one resampling per space | Same structure (Phase 0) |

The main deliberate departures are the template-space policy and the default. Both follow from
diffusion-specific constraints: gradient reorientation under nonlinear warps, and reconstruction's
preference for isotropic voxels in the anatomical frame.
