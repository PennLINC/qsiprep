# DWI output in native and anatomical spaces

Design spec and feasibility evaluation — 2026-10-02 (revised the same day)
Branch: `output-spaces-new`
Related:
- `2026-08-26-output-spaces-design.md`: the `--output-spaces` feature this extends.
- `../plans/2026-09-11-distortion-group-merge-resolutions.md`: per-resolution merging, executed
  as `4c97ad2`..`a734d2b`. Phase 0 here builds on it.

Status: **proposal for evaluation**, in parallel with the current output-spaces work. Nothing
here is implemented.

## Summary

Let `--output-spaces` write preprocessed DWI in spaces other than ACPC: the DWI's own reference
frames and the anatomical frame before AC-PC alignment. Also allow anisotropic resolutions.
Template spaces keep today's meaning, transforms plus anatomical derivatives, and **never**
receive DWI.

```
(no --output-spaces)                        # = acpc:res-nativemin
--output-spaces acpc                        # = acpc:res-nativemin
--output-spaces distortion-group:res-native # each distortion group's b=0 frame, input voxel size
--output-spaces individual acpc:res-1p5mm   # subject dwiref frame, plus 1.5 mm ACPC
--output-spaces dwiref                      # whichever level --dwiref-definition selects
--output-spaces acpc MNI152NLin6Asym        # ACPC DWI; transforms (+ anatomicals) for MNI
```

**Decisions taken in this revision:**

1. **Default is `acpc:res-nativemin`.** It applies both when `--output-spaces` is omitted and
   when a DWI-carrying space is given without `res-`.
2. **Native-space names follow fMRIPrep's (niworkflows') vocabulary** where one exists:
   `distortion-group`, `session`, `individual`, plus a `dwiref` alias. All four are feasible; see
   "Native DWI spaces" for the cost of each.
3. **No `:transform-only`, and no template-space DWI.** A template in `--output-spaces` writes the
   ACPC↔template transforms and the anatomical derivatives, exactly as the branch does today.
4. **Grids reuse niworkflows' `GenerateSamplingReference`** for `res-native`, rather than new
   grid code.

**Verdict.** Feasible without touching estimation:
- `distortion-group`, `dwiref` and `res-native` are cheap.
- `individual` is moderate: the subject template has to be built on request, independently of
  coregistration.
- `session` is the most work (a new per-session template), but it reuses the same machinery.

Dropping template-space DWI removes the only scientifically contested part: nonlinear gradient
reorientation.

## Current state (verified 2026-10-02)

- **DWI is written only in ACPC.**
  - `init_dwi_finalize_wf` loops over `acpc_specs` (`workflows/dwi/finalize.py`).
  - Every DWI sink hardcodes `space='ACPC'`: 11 in `dwi/derivatives.py`, 4 in `finalize.py`, and
    the merge workflow's sidecar.
  - Since `83aca18`, the merge path follows the same per-resolution fan-out: one
    `init_distortion_group_merge_wf` per ACPC spec, fed slot *i* of each unit's finalize outputs.
- **`--output-spaces` is required and must include an `acpc` entry.** The parser enforces it
  (`cli/parser.py` `_finalize_output_spaces`), as does `qsiprep/utils/spaces.py`
  (`parse_output_spaces`, "at least one `acpc`"). An `acpc` entry must carry an explicit `res-`.
- **The resampling chain always ends at ACPC.** `ComposeTransforms._TRANSFORM_STAGES`
  (`interfaces/gradients.py`) composes:
  - `hmc`
  - `gradwarp`
  - `fieldwarp`
  - `to b=0 affine` / `to b=0 warp`, the subject dwiref, only when it is used for coregistration
  - `b=0 to T1w`
- **The subject dwiref template is coupled to coregistration.**
  - `init_dwiref_wf` (`workflows/dwi/dwiref.py`) builds the template with mvtc2 and also
    coregisters it to the anatomical (`b0_coreg_wf`).
  - `base.py` builds it only when `--dwiref-definition subject` and the subject has ≥2 groups
    (`make_dwiref`). Its transforms then replace the per-group coregistration in finalize.
  - The written files are `space-subject_dwiref` and `from-subject_to-ACPC`.
  - There is no session level. `main`'s comment on `dwiref_space` anticipates adding one "as a
    data change".
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
  (`base.py`), so a native DWI frame can be oblique.

## Grammar

```
token   ::= space (":" key)*
space   ::= "acpc" | "anat" | "distortion-group" | "session" | "individual" | "dwiref"
          | <TemplateFlow template>
key     ::= "res-" value | "cohort-" value
```

### Spaces

| Token | Frame | Filename `space-` | DWI written? |
|---|---|---|---|
| `acpc` | AC-PC-aligned anatomical (today) | `ACPC` | yes |
| `anat` | Anatomical reference before AC-PC alignment | `anat` | yes |
| `distortion-group` | Each distortion group's own b=0 reference | `distortiongroup` | yes |
| `session` | A per-session dwiref template (**new**) | `session` | yes |
| `individual` | The subject dwiref template | `individual` | yes |
| `dwiref` | Alias: the level `--dwiref-definition` resolves to | that level's | yes |
| template | TemplateFlow space | `<template>` | **no**: transforms and anatomical derivatives only |

**How the names line up with fMRIPrep.** niworkflows' `NONSTANDARD_REFERENCES`
(`niworkflows/utils/spaces.py`, v1.10.0) lists `T1w`, `T2w`, `anat`, `fsnative`, `func`, `run`,
`sbref`, `session`, `individual`, `dwi` and `asl`. This design reuses:

- **`anat`, `session` and `individual` unchanged.**
- **`distortion-group` instead of `run`.** A QSIPrep distortion group can pool several BIDS runs,
  so `run` would mislead. It is the name `--dwiref-definition` and the
  `space-distortiongroup_dwiref` files already use.
- **`dwiref` as QSIPrep's own alias.** niworkflows' generic `dwi` would not say which level is
  meant.

The filename entity drops the hyphen, because BIDS entity values are alphanumeric
(`distortiongroup`), as `main` already does. QSIPrep parses tokens itself (`utils/spaces.py`), so
the hyphenated token is not a problem, and niworkflows' `Reference` is not used.

**`dwiref` resolves at build time.** It becomes `distortion-group`, `session` or `individual`
according to the resolved `--dwiref-definition`, including the single-group fallback. The
outputs carry that level's name, never `space-dwiref`. A request listing both `dwiref` and the
level it resolves to is de-duplicated after resolution.

**`--dwiref-definition` should gain the same names** (D6). Today it accepts `distortion-group`
and `subject`, and writes `space-subject_dwiref`. With `individual` as an output space, keeping
`subject` there would put `space-subject_dwiref` beside `space-individual_dwi` for the same frame.
Proposal:

- `--dwiref-definition {distortion-group,session,individual}`;
- the reference image written as `space-individual_dwiref`;
- `subject` rejected with a pointer to `individual`.

26.1 is still a prerelease, so the earlier this happens the better.

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

### Grids (point 2: reuse niworkflows)

| Space | `res-native` | `nativemin` / `nativemax` / `mm` |
|---|---|---|
| `distortion-group`, `session`, `individual` | The reference image itself: its affine, FOV and obliquity | `ReferenceGridAtSpacing` (`interfaces/images.py`): that image's FOV and orientation at the new spacing; already used by `dwi/registration.py` |
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

## Native DWI spaces: what each needs

Each entry gives the transform chain `dwi_trans_wf` composes, the grid, and the new work. All
four reuse Phase 0's per-spec fan-out.

### `distortion-group` — low

- **Chain:** `hmc`, `gradwarp`, `fieldwarp`. No dwiref stages and no coregistration.
- **Grid:** the unit's `b0_ref_image`.
- **Work:**
  - Gate the `b=0 to T1w` stage.
  - Give eddy + TOPUP a real identity path. There only coregistration survives today, and the
    `ComposeTransforms` shortcut assumes it is present.
  - Make `ApplyJacobianWeights`' hardcoded `[dwiref_affine, dwiref_warp, coreg]` transport stack
    an input (`interfaces/fmap.py`).
  - `sdc_warp_transforms` becomes identity on its own.
- **Merging:** distortion groups are by definition the units being merged, and they live in
  different frames. So `distortion-group` outputs are always written per unit, never merged.
  With `--distortion-group-merge` on, `distortion-group` is still allowed. It writes per-unit
  files, and the parser says so.

### `individual` — moderate

- **Chain:** `hmc`, `gradwarp`, `fieldwarp`, `to b=0 affine`, `to b=0 warp`. No coregistration.
- **Grid:** `dwiref_wf.outputnode.dwiref`.
- **Work: decouple building the template from using it.** Build `init_dwiref_wf` whenever
  `individual` is requested or `--dwiref-definition` is `individual`. Feed its
  `dwiref_to_t1_affine` into finalize's coregistration *only* in the latter case. Its
  `b0_to_dwiref_transforms` reach finalize in both cases. The individual→ACPC transforms the
  workflow already writes stay useful either way.
- **Cost:** when `individual` is requested under `--dwiref-definition distortion-group`, an extra
  template build (mvtc2, `BSplineSyN` by default) plus one more b=0→anatomical registration.
  This is the one place in this design where requesting an output space adds *estimation* work.
  The new estimate only feeds the output; it never changes the ACPC results.
- **Single-group subjects:** mvtc2 needs ≥2 inputs. See D7.

### `session` — moderate to high

- **Chain:** as `individual`, against the session's template.
- **Grid:** the session template.
- **Work:**
  - Partition the subject's units by their `ses-` entity.
  - Build one `init_dwiref_wf` per session with ≥2 units.
  - Key each unit's `b0_to_dwiref_transforms` by its session.
  - Write `space-session` outputs per session, with the session in the filename like every other
    session-level derivative.
  - Add `session` to `--dwiref-definition`. With the decoupling above done, that is mostly a
    grouping change.
- **Overlap with `--subject-anatomical-reference sessionwise`:** each session is already its own
  subject workflow, so `session` and `individual` describe the same frame. Resolve `individual`
  to `session` there (D8).

### `dwiref` — trivial

An alias resolved at build time, as described under Grammar.

### `anat` — low to moderate

- **Chain:** the full ACPC chain, then the ACPC→anat rigid (`acpc_inv_transform`). That transform
  reaches `dwi_preproc_wf` but not `dwi_finalize_wf` today.
- **Grid:** `GenerateSamplingReference` on the anatomical reference for `native`; otherwise the
  autobox path, run on the anat-space brain.
- **Work:** expose the anat-space brain mask, which today exists only inside
  `init_anat_preproc_wf` (`synthstrip_anat_wf`).

### Anisotropic and oblique grids — moderate, cuts across all of the above

- **Per-output grids.** `res-native` and per-output `nativemin` make grids depend on the output,
  not the subject. Build them in `base.py` or `finalize` from each output's units, replacing the
  subject-level `dwi_sampling_grids` for DWI-native resolutions.
- **Gradient tables in the output frame.** This is the real cost.
  - Input bvecs are deobliqued into world LPS+ (`interfaces/images.py`).
  - The writers assume world axes equal voxel axes: `MRTrixGradientTable`'s flip
    (`interfaces/mrtrix.py`), `btable_from_bvals_bvecs` (`interfaces/dsi_studio.py`), the `DIPY`
    convention of the processed QC, and `GradientRotation`'s non-oblique check
    (`interfaces/gradients.py`).
  - Every grid QSIPrep builds today is deobliqued, so these assumptions hold. The native frames
    (`distortion-group`, `session`, `individual` at `res-native`) can be oblique.
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

- **Estimation is unchanged**, with one opt-in exception. HMC, eddy, DIFFPREP, SHORELine, every
  SDC method, coregistration and normalization are all upstream of the output grid. The exception
  is `individual` or `session` requested while `--dwiref-definition` is a different level. That
  builds a template and its b=0→anatomical registration only to provide the output frame. Those
  extra estimates never feed the ACPC outputs.
- **Per DWI spec, these repeat:** single-shot resampling, Jacobian transport, gradient rotation,
  N4 (one decision per output, one fit per grid), the b=0 reference and mask, processed QC,
  derivative sinks, and a merge per spec when merging.
- **Default runs change only in resolution choice.** A run without `--output-spaces` gets
  `acpc:res-nativemin`. That is today's behaviour for anyone who passed the acquired isotropic
  size, but for anisotropic acquisitions it differs from passing e.g. 2 mm.
- **Disk.** Native frames at `res-native` are the smallest DWI outputs possible: no upsampling,
  no padding to an ACPC bounding box.

## Behaviour changes

- **`--output-spaces` becomes optional** (default `acpc:res-nativemin`). That reverses `main`'s
  "`--output-resolution` has no default and must be given explicitly" (the Required-arguments
  group in `cli/parser.py`). The parser help, `docs/running.rst` and `docs/upgrading.rst` must say
  so, and the option moves out of "Required arguments".
- **The `acpc` requirement goes.** A run can write only native-frame DWI. Such a run produces
  nothing QSIRecon reads today, so the parser should warn when no `acpc` space is requested.
- **Bare templates keep their meaning.** Unlike the first draft of this spec, nothing on the
  branch changes meaning: no test, manifest or doc example needs `:transform-only`.
- **`--dwiref-definition subject` → `individual`** (if D6 is accepted). This renames
  `space-subject_dwiref` / `from-subject_to-ACPC` to `space-individual_*`.

## Contract with QSIRecon

QSIRecon is not in this repo, so its input query could not be checked. Before landing any phase,
confirm:

1. Whether it filters on `space-ACPC`, or would also pick up `space-distortiongroup`,
   `space-individual` and similar files, which would make the new outputs ambiguous to it.
2. Whether it can use anisotropic (`res-native`) input.
3. Whether a `res-` entity on the default `acpc:res-nativemin` output would be expected. It should
   not appear: a single ACPC spec writes no `res-`.

## Phased implementation

Each phase is shippable. An explicit `acpc:res-<N>mm` run is unchanged by every phase. A "task"
is one reviewed commit with tests.

| Phase | Scope | Size | Depends on |
|---|---|---|---|
| 0 | **Groundwork, no new behaviour.** Replace `SpaceSpec.standard` (`space != ACPC`) with `is_template`/`writes_dwi`. Rename `acpc_specs` → `dwi_specs` across `finalize`, `base.py` and the merge path, keyed by (space, resolution). Parameterize `space=` on every DWI sink and reportlet. Add `space` to DWI figure filters in `reports-spec.yml`. Remove the unused to-template hook in `ComposeTransforms`. | 3 tasks | merge-resolutions plan (done) |
| 1 | **Defaults and grammar.** `--output-spaces` optional with default `acpc:res-nativemin`; `nativemin` as the default `res-`; the native-space tokens (parsed, rejected as "not yet implemented" until their phase lands); `dwiref` resolution and de-duplication; replace "≥1 `acpc`" with "≥1 DWI space" plus a warning without `acpc`. Docs: `running.rst`, `upgrading.rst`, parser help. | 2 tasks | 0 |
| 2 | **`distortion-group` and `dwiref`.** Coregistration gating, the eddy identity path, a configurable Jacobian transport stack, the reference-image grid, per-unit writing under merging, Dice policy (D4). | 3 tasks | 1 |
| 3 | **`individual`.** Decouple the template build from coregistration in `base.py`; the single-group policy (D7); D6 rename if accepted. | 2 tasks | 2 |
| 4 | **`res-native` and anisotropy, then `anat`.** Per-output grids via `GenerateSamplingReference` and `ReferenceGridAtSpacing`; output-frame gradient tables for every writer; oblique-grid tests; `ChooseInterpolator` axis pairing; anat-space mask and grid, plus the ACPC→anat stage. | 4–5 tasks | 2 |
| 5 | **`session`.** Per-session templates, `--dwiref-definition session`, session-keyed transforms, the sessionwise overlap (D8). | 3 tasks | 3 |

Total: about **17–18 tasks**. Phases 2 and 3 alone deliver the requested native-space names except
`session`; Phase 4 delivers non-isotropic output. Phases 3–5 can be dropped or reordered
independently.

### Latent bugs found while scoping

All three were present on `main` and are fixed there separately (PR from branch
`fix-transform-and-merge-qc-bugs`):

- `dwiref_to_t1_warp` was silently dropped;
- SHORELine's CNR map was resampled through volume 0's head motion;
- the merged series QC lacked `space-ACPC`.

Merge `main` into this branch once that lands. The merge-workflow QC sink will conflict trivially
with `4c97ad2` (`space='ACPC'` there, `**res_entities` here; keep both).

## Decisions

Settled in this revision:

- **Template-space DWI:** not supported. Templates write transforms and anatomical derivatives
  only. No `:transform-only` keyword.
- **Default resolution and space:** `acpc:res-nativemin`.
- **Native-space vocabulary:** `distortion-group`, `session`, `individual`, `dwiref`, aligned with
  niworkflows where it has a name.
- **Grid construction:** reuse `GenerateSamplingReference` and `ReferenceGridAtSpacing`.

Still open:

- **D4 — Mask Dice outside ACPC.** Series QC's Dice compares the ACPC anatomical mask with the DWI
  mask on a shared grid. `init_mask_overlap_wf` resamples without a transform, and `DiceOverlap`
  fails on a shape mismatch. *Recommend computing it once in ACPC* and reporting it for every
  space: it measures coregistration quality, which does not depend on the output space.
- **D5 — `anat` grid.** The anatomical image's own, possibly oblique, grid, or world-aligned
  around it? *Recommend its own grid*, for symmetry with the native DWI frames.
- **D6 — Rename `--dwiref-definition subject` to `individual`.** *Recommend yes*, while 26.1 is a
  prerelease, so the reference image and the output space share one name.
- **D7 — `individual`/`session` with a single distortion group.** mvtc2 needs ≥2 inputs. Options:
  - write the outputs on that group's b=0 frame under the requested name, with an identity
    b0→template transform;
  - skip them with a warning, as `--dwiref-definition` falls back today;
  - error.

  *Recommend the first*, so a requested filename always exists and cohort scripts don't need
  special cases. The sidecar should record that the template is a single group.
- **D8 — `individual` under `--subject-anatomical-reference sessionwise`.** Each session is its own
  subject workflow. *Recommend resolving `individual` to `session` there*, with a note, rather
  than rejecting it.

## Relation to fMRIPrep

| | fMRIPrep | This design |
|---|---|---|
| Default | `MNI152NLin2009cAsym:res-native` (BOLD in template space) | `acpc:res-nativemin` |
| Bare template | Data resampled there at `res-native` (niworkflows appends `:res-native`) | Transforms + anatomical derivatives only |
| Native names | `run`, `session`, `individual`, `func`, `sbref`, `anat`, `T1w` | `distortion-group`, `session`, `individual`, `anat`, plus `dwiref` and `acpc` |
| `res-native` grid | `GenerateSamplingReference` | The same interface for `acpc`/`anat`; the reference image itself for native frames |
| Isotropy | Never forced | Default `nativemin` is isotropic; `native` and anisotropic `mm` are opt-in |
| Physical sizes | Not supported (niworkflows#997) | `<N>mm`, `<X>x<Y>x<Z>mm` |
| Fit vs apply | Estimation once, one resampling per space | Same structure (Phase 0) |

The main deliberate departures are the template-space policy and the default. Both follow from
diffusion-specific constraints: gradient reorientation under nonlinear warps, and reconstruction's
preference for isotropic voxels in the anatomical frame.
