# DWI output in native, anatomical and standard spaces

Design spec and feasibility evaluation — 2026-10-02
Branch: `output-spaces-new` (at `0672e2b`)
Related: `2026-08-26-output-spaces-design.md` (the `--output-spaces` feature this extends),
`../plans/2026-09-11-distortion-group-merge-resolutions.md` (multi-resolution merging, which this
depends on — see "Relation to the current plan").

Status: **proposal for evaluation**, in parallel with the current output-spaces plan. Nothing
here is implemented.

## Summary

Let every `--output-spaces` entry decide, per space, whether *preprocessed DWI* is written there
or only the transforms are. Add two non-template spaces — `dwiref` (the DWI's own reference
frame, following `--dwiref-definition`) and `anat` (the anatomical reference before AC-PC
alignment) — and allow anisotropic resolutions, including a `res-native` that keeps the input
DWI's own voxel size.

```
--output-spaces dwiref:res-native               # dwiref frame, input voxel size, not isotropic
--output-spaces MNI152NLin6Asym:transform-only  # today's behaviour: transforms + anatomicals
--output-spaces MNI152NLin6Asym                 # DWI resampled into MNI152NLin6Asym, native res
```

**Verdict.** The non-template spaces are cheap and safe. `dwiref` and `anat` are truncations or
extensions of the transform chain QSIPrep already composes, and nothing upstream of resampling
changes. Anisotropic grids are moderate work, mostly around gradient-table conventions on
oblique grids. **DWI in standard space is the expensive and scientifically contentious part.**
The code path for it exists only as a dead, broken hook. A nonlinear warp also needs voxelwise
gradient reorientation, which QSIPrep does not do. And it reverses the project's documented
position that DWI is normalized after model fitting. Ship it last, behind an explicit decision
(D1 below).

Estimation is untouched in every case: head motion, eddy current and susceptibility correction,
coregistration and normalization run exactly as today. Only resampling, the steps after it, and
the derivatives fan out per space.

## Current state (verified 2026-10-02)

- **DWI is written only in ACPC.** `init_dwi_finalize_wf` loops over `acpc_specs`
  (`finalize.py:445`), and every DWI sink hardcodes `space='ACPC'`: 11 in `dwi/derivatives.py`,
  4 in `finalize.py` (`:691`, `:730`, `:767`, `:826`), and the merge sidecar
  (`distortion_group_merge.py:298`).
- **Standard spaces carry transforms and anatomicals only.** `init_anat_derivatives_wf` resamples
  the preprocessed anatomical, brain mask and dseg onto each template's TemplateFlow grid
  (`anatomical/volume.py:2039-2177`). Nothing resamples DWI there. `standard_forward_transforms`
  (`volume.py:583`) has no DWI consumer.
- **The resampling chain always ends at ACPC.** `ComposeTransforms._TRANSFORM_STAGES`
  (`interfaces/gradients.py:398-405`) composes, in this order:
  - `hmc`
  - `gradwarp`
  - `fieldwarp` (SDC)
  - `to b=0 affine` and `to b=0 warp` (dwiref)
  - `b=0 to T1w` (coregistration)

  A to-template hook exists (`:533-539`), but `init_dwi_trans_wf` never connects
  `t1_2_mni_forward_transform` (`resampling.py:185`, `:271-283`). The hook also asserts a legacy
  `[affine, warp]` pair, while the anatomical workflow produces one composite `.h5` per template.
- **Grids are isotropic by construction.**
  - `init_output_grid_wf` (`volume.py:1659-1708`) autoboxes the anchor template brain (8 mm
    padding, 4 for infants), deobliques it, and resamples to `_tupleize(voxel_size)`.
  - `VoxelSizeChooser` (`interfaces/anatomical.py:57-96`) pools every zoom of every raw DWI run
    into one scalar.
  - The grid's field of view is the template's brain, never the DWI's.
- **Gradients are rotated by one affine per volume.**
  - `GradientRotation` (`gradients.py:747-769`) applies the linear part of each volume's affine
    chain: rotation, scale and shear, not a polar-decomposed rotation. Warps are ignored.
  - `LocalGradientRotation` (`:783-803`) exists, but its node is commented out
    (`resampling.py:498-504`). `local_bvecs` is forwarded but never populated, and its derivative
    sink is commented out (`derivatives.py:420-434`).
- **The b=0 reference and brain mask are already independent of space.**
  `init_dwi_reference_wf` (`dwi/util.py:94-231`) masks with SynthStrip on the b=0. It declares
  `t1_brain`, `t1_mask` and `t1_seg` but never connects them; the one use is commented out
  (`:187-190`). The N4 weights (`finalize.py:1024-1031`) and SeriesQC's CNR and b=0 statistics
  also use this DWI-derived mask.
- **The output grid does not feed estimation.** `_first_sampling_grid` (`base.py:127-135`) reaches
  `hmc_wf` but is never connected in `hmc_sdc.py`. Its only consumer is SynB0's side output,
  `space-ACPC_desc-synb0_dwiref` (`fieldmap/synb0.py:298-302`).
- **The DWI series are conformed, not deobliqued.** Each series is reoriented to `LAS` (eddy) or
  `LPS` (everything else) (`base.py:718-720`). Obliquity is preserved, so the dwiref frame can be
  oblique.

## Grammar

```
token   ::= space (":" key)*
space   ::= "acpc" | "dwiref" | "anat" | <TemplateFlow template>
key     ::= "res-" value | "cohort-" value | "transform-only"
```

### Spaces

| Space | Frame | Filename `space-` | DWI written? |
|---|---|---|---|
| `acpc` | AC-PC-aligned anatomical (today) | `ACPC` | yes |
| `dwiref` | The resolved `--dwiref-definition` level | `distortiongroup` or `subject` | yes |
| `anat` | Anatomical reference before AC-PC alignment | `anat` | yes |
| template | TemplateFlow space via the ACPC→template composite | `<template>` | yes, unless `transform-only` |

- **`dwiref` names the *resolved* level.** `--dwiref-definition subject` falls back to
  `distortion-group` for a subject with one DWI group (`base.py` "Falling back to
  --dwiref-definition distortion-group"). The output follows the same fallback. The filename
  entity matches the reference images `main` already writes, so a `space-distortiongroup_dwi`
  sits beside its `space-distortiongroup_dwiref`.
- **`anat`** uses the name the existing `from-anat_to-ACPC` transforms already use.
- **`transform-only` is legal only on template spaces.** ACPC, dwiref and anat transforms are
  always written, so the flag would be meaningless there.

### Resolutions

| Value | Meaning | Allowed on |
|---|---|---|
| `native` (**new**) | The input DWI's own voxel size, per axis, *not* forced isotropic | every DWI space |
| `nativemin` / `nativemax` | Smallest/largest input zoom, isotropic (today) | every DWI space |
| `<N>mm`, `<X>x<Y>x<Z>mm` | Physical size; anisotropic now allowed | every DWI space |
| `<label>` | TemplateFlow `res` entity | template spaces |

The default when no `res-` is given is **`native`** for every DWI-carrying space. That matches the
user-facing example (`MNI152NLin6Asym` means native resolution). It also replaces today's rule
that `acpc` must carry an explicit `res-`. See D3.

What `native` means depends on the frame:

- **`dwiref`:** the reference image's own grid: its affine, field of view and obliquity, exactly.
  This is the only space where "native" is literal.
- **`acpc`, `anat`, template:** a world-aligned grid whose spacing is the input zooms mapped onto
  the output axes. The mapping uses the input affine's orientation (`nib.io_orientation`), so a
  sagittal acquisition's slice spacing lands on x, not z. The field of view is the space's own
  bounding box. The input and output frames differ by a rotation, so this is "native spacing,
  re-gridded", and the docs must say so.

Runs feeding one output must agree on zooms for `native`. Otherwise the parser cannot know which
run wins, so a mismatch is a build-time error that suggests `nativemin`/`nativemax`. Today
`VoxelSizeChooser` silently pools every run of the subject.

### Validation changes

- Drop "at least one `acpc`" (`spaces.py:276-280`). Require instead **at least one space that
  writes DWI**: any non-`transform-only` entry. ACPC alignment is still always computed, because
  it is the anatomical frame everything is defined in; it just stops being the only output frame.
- Drop the anisotropy rejection on `acpc` (`spaces.py:143-148`) and the rejection of `native*` on
  non-acpc spaces (`:133-137`).
- `SpaceSpec.standard` is `space != ACPC` today (`:63-65`), which would misclassify `dwiref` and
  `anat`. Replace it with explicit `is_template` and `writes_dwi` properties.
- `SpaceSpec.__str__` is the dedup key and the stored canonical form, so it must include
  `transform-only`.
- `res-<label>` on a template with DWI selects the TemplateFlow grid for both the DWI and the
  anatomical derivatives. `native`/`mm` on a template builds a new grid: the template's brain
  bounding box at that spacing. See D2 for what the anatomical derivatives then use.
- `--distortion-group-merge` combined with more than one DWI spec needs one merge workflow per
  spec. That is exactly what the current merge-resolutions plan builds (Task 2 there), generalized
  from "per ACPC resolution" to "per DWI spec".

## What changes per target space

Each row is the transform chain `dwi_trans_wf` would compose, the grid it resamples onto, and
what must exist first. Spaces are ordered by difficulty.

### 1. `dwiref` — low difficulty

| | |
|---|---|
| Chain | Distortion-group: `hmc`, `gradwarp`, `fieldwarp` and nothing else. Subject: the same, plus `to b=0 affine` and `to b=0 warp`. Coregistration is dropped in both. |
| Grid | `res-native`: the distortion group's `b0_ref_image` or the subject `dwiref` image, as is. Other resolutions: that image's field of view and orientation at a new spacing. `ReferenceGridAtSpacing` (`interfaces/images.py:601-670`, used by `dwi/registration.py:159`) already does this and keeps the direction cosines. |
| Inputs | All present at `dwi_trans_wf` today (`base.py:874-889`). |
| Work | Gate the `b=0 to T1w` stage. The shortcut at `gradients.py:557` assumes coregistration exists, so eddy + TOPUP, where only coregistration survives, needs a real identity path. `ApplyJacobianWeights` transports weights through a hardcoded `[dwiref_affine, dwiref_warp, coreg]` stack (`interfaces/fmap.py:1150-1170`); make the stack an input. `sdc_warp_transforms` becomes identity automatically. |

### 2. `anat` — low to moderate

| | |
|---|---|
| Chain | The full ACPC chain, then the ACPC→anat rigid (`acpc_inv_transform`). |
| Grid | None exists. Build one like ACPC's: autobox the anatomical brain *in anat space*, at the requested spacing. Deobliquing it changes the frame, so decide whether `anat` means "the anatomical image's own grid" (oblique allowed, like dwiref) or "world-aligned around it". |
| Inputs | `acpc_inv_transform` reaches `dwi_preproc_wf` (`base.py:782`) but not `dwi_finalize_wf`; wire it. The anat-space brain mask (`synthstrip_anat_wf`) exists only inside `init_anat_preproc_wf` (`volume.py:689-696`); expose it on the outputnode. |

### 3. Anisotropic and oblique grids — moderate, cuts across 1–4

- **Grid construction.** `VoxelSizeChooser` outputs a scalar and `_tupleize` makes it isotropic.
  Make the chooser return a 3-tuple and map axes by orientation. Make "which runs" the runs of the
  output, not the whole subject. Grids then become per output (built in `finalize` or `base.py`
  from that output's units) instead of the subject-level `dwi_sampling_grids`.
- **Gradient tables on an oblique grid.** This is the real cost, not the voxel size.
  - Input bvecs are deobliqued into world LPS+ (`images.py:835-862`).
  - The writers assume world axes equal voxel axes: `MRTrixGradientTable`'s `[-1,-1,1,1]` flip
    (`interfaces/mrtrix.py:69-73`), the LPS+ assumption in `btable_from_bvals_bvecs`
    (`dsi_studio.py:518-523`), and the `bvec_convention='DIPY'` used for processed QC.
  - `GradientRotation` requires a non-oblique frame (`gradients.py` around `:733`).
  - That holds for every grid QSIPrep builds today, because they are deobliqued. It does not hold
    for `dwiref:res-native`.
  - Each output therefore needs its bvecs re-expressed in its own voxel frame (FSL `.bvec`) and
    its world frame (MRtrix `.b`, DSI Studio `b_table`).
  - This is one well-tested helper, but it touches every gradient-table writer.
- **Everything else is already per axis.** `ChooseInterpolator` compares zooms per axis
  (`images.py:589-596`). DSI Studio QC already runs on anisotropic raw data. SynthStrip regrids to
  1 mm internally. N4's spline distance is in mm. Erosion uses `max(zooms)`.
  - `ChooseInterpolator` pairs axes by index regardless of orientation. Fix it alongside the
    axis-mapping helper.

### 4. Standard spaces with DWI — high difficulty, contested

| | |
|---|---|
| Chain | The full ACPC chain, then the ACPC→template composite (`standard_forward_transforms[i]`, `.h5`). |
| Grid | `res-<label>`: the TemplateFlow grid (`standard_template_lps`). That is huge: a 1 mm whole-template FOV for a 4D series. `native`/`mm`: a new grid, the template brain's bounding box at that spacing, made with `init_output_grid_wf` on `std_lps_wf.outputnode.template_lps`. |
| Inputs | The composites exist whenever any standard space is requested (`volume.py:398`, `:551-586`). They reach finalize's inputnode but not `dwi_trans_wf`. |

What breaks, in order of severity:

1. **Gradient reorientation under a nonlinear warp.**
   - Global bvec rotation (`GradientRotation`) only sees affines. `_compose_tfms` sends every
     non-`.nii` transform to `compose_affines` (`gradients.py:1074-1092`), so an `.h5` composite
     is handed to `antsApplyTransforms -o Linear[...]`. That fails, or at best linearizes.
   - Correct reorientation needs a per-voxel rotation from the warp's Jacobian (finite strain or
     PPD), written as voxelwise bvecs or a per-voxel b-matrix. `LocalGradientRotation` is a
     starting point, but it is dead code that was never wired or tested. QSIRecon would also need
     to *read* voxelwise gradients, which, as far as this repo shows, it does not.
2. **The hook itself.** Replace `t1_2_mni_forward_transform`'s `[affine, warp]` assert with a
   per-spec composite input. Split it, either with `CompositeTransformUtil --disassemble` or with
   the existing `DisassembleTransform` used in `anat_normalization_wf`, so the affine part can join
   `compose_affines` and the warp part rides only the image transform.
3. **SDC map conjugation and Jacobian transport.** `ComposeSDCWarp` inverts non-`.nii` stages on
   the fly (`gradients.py:690-700`), which is wrong for a SyN composite. For template spaces, the
   SDC displacement and Jacobian derivatives should stay in ACPC (they are maps of the correction)
   or be dropped, not conjugated through the warp. The design rule in `jacobian.py:9-17` already
   says template warps must never modulate DWI signal; keep it.
4. **graddev.** It is oriented by TORTOISE's own rigid registration (`finalize.py:786-866`), which
   is meaningless in a nonlinear target. Do not write it for template spaces.
5. **Scientific position.** The docs say DWI is "never written in template space; spatial
   normalization is done after models are fit, in QSIRecon" (`docs/running.rst:571-576`;
   `troubleshooting.rst:91-95`). Interpolating diffusion-weighted signal through a T1-driven SyN
   warp mixes tissue orientations and distorts the q-space signal. Some uses accept that: group
   tractography templates, or quick-look QC. Most model fits do not. Supporting it means
   documenting exactly what was and was not reoriented.

## Effect on processing

- **Estimation is unchanged.** HMC, eddy, DIFFPREP, SHORELine, every SDC method, coregistration,
  dwiref construction and anatomical normalization are all upstream of the output grid. The
  exception is SynB0's `space-ACPC_desc-synb0_dwiref` side output (`fieldmap/synb0.py:298-302`),
  which picks the first ACPC grid. Under this design it picks the first ACPC spec if there is one,
  and otherwise is written in dwiref space.
- **Per DWI spec, these repeat:**
  - single-shot resampling;
  - Jacobian weight transport;
  - gradient rotation;
  - N4 (still one *decision* per output, but one fit per grid);
  - the resampled b=0 reference and mask;
  - processed QC;
  - the derivative sinks;
  - with `--distortion-group-merge`, a merge per spec.

  That is roughly the per-resolution cost the current plan already accepts, applied to spaces
  instead of resolutions.
- **Standard-space DWI forces the nonlinear registration.** It already runs whenever any template
  is listed, so this changes nothing for users who list one today.
- **Disk.** `dwiref:res-native` is the smallest DWI output possible: no upsampling, no padding to a
  template FOV. Template-space DWI on a TemplateFlow `res-1` grid is the largest. It covers the
  whole template at 1 mm, against ACPC's brain bounding box at the requested spacing, so it is
  roughly an order of magnitude more voxels than a 2 mm ACPC output (estimate from grid sizes,
  not measured).
- **Nothing changes for an `acpc`-only run**, provided the defaults hold:
  - single ACPC spec ⇒ no `res-` entity;
  - an explicit `acpc:res-<N>mm` keeps today's isotropic grid.

  The one exception is a bare `acpc`, which becomes legal and means `native`, i.e. anisotropic.
  Today's parser rejects that token, so no existing command changes meaning.

## Behaviour changes for the branch's users

The `--output-spaces` grammar has not been released, so these change only the branch:

- **A bare template now writes DWI.** `--output-spaces acpc:res-2mm MNI152NLin2009cAsym` today
  means "transforms + anatomicals". Under this design it also resamples the DWI into the template.
  Every test, manifest and doc example that lists a bare template must add `:transform-only` to
  keep its meaning. That includes the CLI examples in `docs/running.rst`, the integration commands
  in `qsiprep/tests/test_cli.py`, and `qsiprep/data/tests/config.toml`. See D1 for the alternative.
- **`acpc` is no longer required.** A run can write only `dwiref`. QSIRecon, as documented, reads
  `space-ACPC`, so such a run would produce nothing QSIRecon consumes. The parser should warn
  when no `acpc` space is requested.

## Contract with QSIRecon

QSIRecon is not in this repo, so its input query could not be checked. The docs here state that:

- ACPC is the only DWI space;
- reconstruction requires isotropic DWI (`docs/running.rst:595-601`, `spaces.py:145-147`);
- a single ACPC spec keeps "the filenames QSIRecon expects" (`running.rst:626-628`).

Before landing any phase, confirm with QSIRecon:

1. Whether it filters on `space-ACPC` or would pick up `space-distortiongroup` / `space-anat`
   files too, which would make the new outputs ambiguous for it.
2. Whether it can use anisotropic input, or whether `res-native` outputs are for other consumers
   only.
3. Whether it could consume template-space DWI at all, given (1) above on gradients.

## Phased implementation

Each phase is shippable and leaves `acpc`-only runs bit-identical. Sizes are in the units the
current plans use (a "task" ≈ one reviewed commit with tests).

| Phase | Scope | Size | Depends on |
|---|---|---|---|
| 0 | **Groundwork, no new behaviour.** Replace `SpaceSpec.standard` with `is_template`/`writes_dwi`. Turn `acpc_specs` into `dwi_specs` throughout `finalize`, `base.py` and the merge path. Parameterize `space=` on every DWI sink and reportlet. Add `space` to DWI figure filters in `reports-spec.yml`. Fix the three latent bugs below. | 3–4 tasks | merge-resolutions plan Tasks 1–2 (per-spec merging) |
| 1 | **`transform-only` and the template default.** Grammar plus validation. Bare templates are still `transform-only` until Phase 4 (D1). | 1–2 tasks | 0 |
| 2 | **`dwiref`.** Chain gating, the identity path, a configurable Jacobian transport stack, the reference-image grid, a per-space Dice (or skip it, D4). | 3 tasks | 0 |
| 3 | **Anisotropic and `res-native`.** Per-output 3-tuple chooser, orientation mapping, output-frame gradient tables for every writer, oblique-grid tests, `ChooseInterpolator` axis pairing. Then `anat`: expose the anat-space mask, the grid, and the ACPC→anat stage. | 4–5 tasks | 2 |
| 4 | **Template-space DWI.** Composite disassembly into `ComposeTransforms`, the template grid, gradient reorientation (D1), SDC/Jacobian/graddev policy, a docs reversal. | 5+ tasks, plus a scientific review | 3, D1 |

Phases 0–3 total about **11–14 tasks**: roughly the size of the original `--output-spaces`
feature. Phase 4 is a separate project. Its gradient-reorientation part should be reviewed by
someone who owns the reconstruction side.

### Latent bugs found while scoping (fix in Phase 0 regardless)

- **`dwiref_to_t1_warp` is ignored.** `gradients.py:513-515` assigns the warp to
  `dwiref_to_t1_affine`, a local that `by_name` never reads. `init_dwiref_wf` also never sets
  that output (`dwiref.py:107`; only the affine is connected). This is harmless today because the
  output is always undefined, but a trap for anyone adding a dwiref→T1 warp.
- **SHORELine's CNR and fieldmap images get the wrong transform.** `cnr_tfm` and
  `fieldmap_hz_tfm` use volume 0's full composite (`resampling.py:284-285`, `:372`; there is a
  TODO at `:284`). For SHORELine that includes volume 0's head-motion transform.
- **The merged series QC drops `space-`.** The merge workflow writes `desc-image_qc.tsv` with no
  `space` (`distortion_group_merge.py:219-230`). The direct path writes
  `space-ACPC_desc-image_qc.tsv`. No integration manifest covers the merged name.

## Decisions needed

- **D1 — Template-space DWI: support it, and how?**
  - (a) Phases 0–3 only. Bare templates stay `transform-only` by default, and a template DWI
    output is a parse error until Phase 4.
  - (b) Phase 4 with global-affine gradient rotation only, documented loudly as approximate.
  - (c) Phase 4 with voxelwise reorientation.

  *Recommend (a) now.* It keeps the grammar forward-compatible: the user-facing example
  `MNI152NLin6Asym` errors, with a message pointing at `:transform-only`, until (c) is designed.
  (b) would put silently mis-oriented gradients into reconstructions.
- **D2 — Anatomical derivatives on a template that also carries DWI.** Write them on the DWI grid
  (one grid per spec, simplest) or keep them on the TemplateFlow grid? *Recommend one grid per
  spec,* so a template's anatomical and DWI outputs always overlay voxel for voxel.
- **D3 — Default resolution.** `native` everywhere when `res-` is omitted, or keep requiring
  `res-` on DWI spaces? *Recommend `native`,* to match the requested semantics. Today's explicit
  `acpc:res-<N>mm` commands are unaffected.
- **D4 — Mask Dice in non-ACPC spaces.** The Dice in series QC compares the ACPC anatomical mask
  with the DWI mask and assumes they share a grid (`qc.py:121-135`); `DiceOverlap` hard-fails on a
  shape mismatch. Options:
  - resample the anatomical mask into each space (dwiref needs the inverse coregistration);
  - compute Dice in ACPC only and report it for every space;
  - write n/a outside ACPC.

  *Recommend computing it once in ACPC.* It measures coregistration quality, which does not depend
  on the output space.
- **D5 — `anat` grid.** The anatomical image's own (possibly oblique) grid, or world-aligned
  around it? *Recommend its own grid,* for symmetry with `dwiref`. That makes `anat` depend on
  Phase 3's oblique gradient tables, which is why it sits in Phase 3.

## Relation to the current plan

The merge-resolutions plan is a **prerequisite**, not a competitor:

- Its Task 2 turns the finalize outputnode into per-spec lists and builds one merge workflow per
  spec. Phase 0 generalizes the key from "ACPC resolution" to "DWI spec" and needs the same
  wiring. Landing that plan first and renaming `acpc_specs` → `dwi_specs` afterwards is cheaper
  than doing both at once.
- Its gap of no Jacobian, SDC or graddev derivatives on merged outputs carries over unchanged.
  Phase 0 should not widen it.
- If the two are evaluated as alternatives anyway: that plan is five tasks of well-understood
  work. This one is about 11–14 tasks through Phase 3, and Phase 4 is open-ended.
