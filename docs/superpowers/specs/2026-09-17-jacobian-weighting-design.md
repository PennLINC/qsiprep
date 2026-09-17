# Jacobian weighting for all spatial transforms

Design spec. Addresses [PennLINC/qsiprep#1133](https://github.com/PennLINC/qsiprep/issues/1133).

Date: 2026-09-17
Branch: `jacobian-weighting`

## Problem

QSIPrep applies Jacobian intensity modulation inconsistently. Two of the six
correction paths modulate; the rest silently do not, so preprocessed signal
intensities are not comparable across `--sdc-method` or `--hmc-method` choices,
and no documentation says which is which.

Verified current state:

| Correction | Modulated today | Evidence |
|---|---|---|
| FSL `eddy` (HMC + EC + TOPUP field) | Yes | `method: "jac"` in `qsiprep/data/eddy_params.json:11` -> `eddy --resamp=jac` |
| DRBUDDI | Yes, empirically | `DRBUDDIAggregateOutputs` builds `undistorted_reference / blip_*_b0_corrected` ratio images, `qsiprep/interfaces/tortoise.py:528-537` |
| GRE fieldmap | No | `init_sdc_unwarp_wf` computes `out_jacobian` (`workflows/fieldmap/unwarp.py:177,236`) but no consumer exists; `init_sdc_wf` forwards only `out_warp` and `out_reference` |
| SyN fieldmapless | No | `workflows/fieldmap/syn.py` emits only `out_warp` |
| TORTOISE T2Wreg | No | `sdc_warp` forwarded bare, `workflows/dwi/diffprep.py:717` |
| Gradient nonlinearity | No | `gradwarp_field` goes straight into `ComposeTransforms`, `workflows/dwi/resampling.py:220` |
| TORTOISE DIFFPREP eddy current | No | DIFFPREP bakes motion+eddy into `_moteddy.nii` and emits identity affines, `workflows/dwi/diffprep.py:225` |

## Decisions

These were settled with the maintainer before design and are not open:

1. **Scope.** Only native-space distortion corrections contribute to the weight:
   gradient nonlinearity, susceptibility distortion, eddy current. Head motion,
   the coregistration/ACPC affine, the intramodal-template affine and warp, and
   the template (MNI) nonlinear warp are all excluded. Modulating by a
   spatial-normalization warp is VBM-style volume modulation and would corrupt
   DWI signal intensities and downstream model fits.

   The exclusions are a deliberate policy choice, **not** a claim that the
   excluded transforms have unit determinant. Several of them do not:

   - SHORELine head motion defaults to `Affine`, not `Rigid`
     (`qsiprep/data/shoreline_params.json`, documented at
     `qsiprep/cli/parser.py:813`). An affine HMC has a spatially constant but
     generally non-unit determinant, and it can absorb EC-like scaling and
     shear.
   - `--b0-to-anat-transform` accepts `Affine` as well as `Rigid`
     (`qsiprep/cli/parser.py:739-742`; `Rigid` is the default, and
     `--b0-to-t1w-transform` is its deprecated alias, removed in 27.0.0). An
     `Affine` coregistration has a non-unit constant determinant.
   - `--intramodal-template-transform` defaults to `BSplineSyN` -- nonlinear,
     with a spatially varying determinant -- and also accepts `Rigid`,
     `Affine` and `SyN` (`qsiprep/cli/parser.py:771-773`).

   The intramodal case deserves emphasis because it is the one excluded
   transform that is nonlinear *by default*. Excluding its determinant is
   correct under the scope decision (it is a within-subject alignment, not a
   measured distortion), and it is not a coordinate-safety problem either: it
   sits between the weight map's domain and the output grid, so
   `ApplyScalingImages` already applies it when transporting the map, which is
   exactly what puts the gradwarp and SDC determinants at the right
   coordinates. Its own determinant is simply never multiplied in.

   The policy is therefore stated positively: the weight includes the
   determinant of the gradwarp and SDC warps and the TORTOISE EC transform, and
   of nothing else, regardless of what the excluded transforms' determinants
   happen to be. A per-volume global rescaling from an affine HMC fit is an
   artifact of the fit, not a measured volume change, and modulating by it would
   rescale whole volumes.
2. **Default.** On by default, with `--no-jacobian-weighting` to opt out.
3. **DRBUDDI.** Its empirical ratio images are replaced by the analytic Jacobian
   of its own warps, so every path uses one mechanism.
4. **Derivatives.** The weight maps are written out, in output space, always
   (unless weighting is disabled).
5. **TORTOISE eddy current.** In scope, gated behind a convention-validation
   test with a defined fallback.

## External constraint: `eddy`'s own modulation

`eddy`'s resampling method is `--resamp`, which accepts only `jac` or `lsr`
(`lsr` requires exactly two opposite-polarity acquisitions). QSIPrep's shipped
default is `jac` (`qsiprep/data/eddy_params.json:11`), which reaches the command
line as `--resamp=jac` via Nipype's `method` trait, so by default EC and TOPUP
susceptibility modulation are applied internally by `eddy`, upstream of
`ComposeTransforms`.

**This is a default, not an invariant.** `--eddy-config` lets a user supply
their own JSON, which is loaded and splatted into `ExtendedEddy`
(`qsiprep/workflows/dwi/fsl.py:182` and `:224`), so `method` can be set to
`lsr`. The design must therefore branch on the *effective* eddy configuration
rather than assuming `jac`:

- `method == 'jac'` (default): `eddy` already modulated EC and TOPUP
  susceptibility. QSIPrep contributes only the gradwarp determinant on that
  path. Re-deriving the TOPUP field's determinant here would double-count.
- `method == 'lsr'`: `eddy` did **not** Jacobian-modulate. QSIPrep cannot
  retrofit it either, because `eddy` has already baked its resampling in and
  exports no field for the EC component. This combination warns that EC and
  susceptibility corrections are **not Jacobian-modulated**, and the methods
  boilerplate says the same. The claim is deliberately narrow: least-squares
  restoration is a different resampling model whose full intensity semantics
  are not established by anything in this repository, so the spec does not
  assert that `lsr` output is un-normalized in some broader sense -- only that
  the Jacobian modulation this feature is about did not happen.

  **Where this is detected.** `eddy_args` is read in `init_fsl_hmc_wf`
  (`qsiprep/workflows/dwi/fsl.py:182`), but the weighting node is built later in
  `init_dwi_trans_wf`. Rather than re-reading the config in two places, the
  effective method is resolved once by a small helper in
  `qsiprep/utils/`, called from `fsl.py` for the warning and boilerplate and
  from `init_dwi_trans_wf` for the weighting decision. The helper takes the
  already-loaded `eddy_args` dict so there is one parse and one source of
  truth; it is not a new outputnode field, because that would make a
  build-time decision look like a runtime one.

No double-counting exists on the TOPUP-only path regardless: `GatherEddyInputs`
sets `forward_warps = []` unconditionally, with the comment "these have already
had HMC, SDC applied" (`qsiprep/interfaces/eddy.py:206`), so the
`forward_warps -> to_dwi_ref_warps` connection at
`qsiprep/workflows/dwi/fsl.py:516` transmits an empty list. The TOPUP field
reaches `eddy` only through its `field` input (`fsl.py:488`). This was the
design's highest-rated risk and is resolved by inspection; the regression test
in the testing section keeps it resolved.

## Why omitting HMC is coordinate-safe

Excluding a transform from the composed field is only valid if it does not move
the coordinates at which the remaining determinants are evaluated. It does not,
because HMC sits at the far end of the chain from the output grid.

### Disambiguating "applied first"

ANTs and Nipype both document the transform list as "applied in reverse order;
the last specified transform will be applied first"
(`nipype/interfaces/ants/resampling.py:361`), and QSIPrep repeats the rule at
`qsiprep/workflows/dwi/diffprep.py:599`. **That sentence describes image-warp
order, not point-map order, and the two are opposite.** Conflating them inverts
the whole analysis, so state it once, precisely:

For `-t T_1 -t T_2 ... -t T_n`, the composite point map from an output
(reference) point to an input (moving) point is

    phi = T_n . T_(n-1) . ... . T_1

so the **first-listed** transform is the **innermost** -- applied first to the
point -- and the last-listed is outermost. Equivalently, and this is what the
docs mean: the last-listed transform is the first *image* operation applied to
the moving image. Both readings describe the same composite; they run in
opposite directions because image warping and point lookup are inverses.

Three independent confirmations:

1. The canonical ANTs invocation is
   `antsApplyTransforms -t out1Warp.nii.gz -t out0GenericAffine.mat`. The
   affine is the coarse alignment and acts first as an image operation, which
   is exactly why it is listed *last*. On a point the composite is
   `T_affine . T_warp`: first-listed innermost.
2. `qsiprep/workflows/dwi/diffprep.py:599` says the resampling happens "in
   chain order (gradwarp, then SDC)" -- the image-operation order -- and that
   the ANTs list is therefore `[SDC, gradwarp]`. With image ops
   `O_1 = gradwarp`, `O_2 = SDC`, the resulting image is
   `J(x) = native(T_gradwarp(T_SDC(x)))`, so on a point the SDC warp (first
   listed) acts first. Consistent.
3. Physical type-checking of the full chain, below.

### The chain

`ComposeTransforms._TRANSFORM_STAGES` is ordered native-to-target and is
**reversed** before use (`qsiprep/interfaces/gradients.py:533`), so the ANTs
transform list is `[coreg, intramodal warp, intramodal affine, fieldwarp,
gradwarp, hmc]`. Applying the rule above, the pull-back from an output point
`x` is:

    phi_full(x) = hmc(gradwarp(fieldwarp(intramodal(coreg(x)))))

giving

    det grad phi_full(x) = det[grad hmc] . det[grad gradwarp](u)
                           . det[grad fieldwarp](v) . det[grad intramodal](.)
                           . det[grad coreg]

with `v = intramodal(coreg(x))` and `u = fieldwarp(v)`.

HMC is the **outermost** function, so its determinant is a constant factor and
cannot affect `u` or `v`. The evaluation coordinates of the gradwarp and
fieldwarp determinants are fixed by `coreg` and `intramodal` alone -- and those
are exactly the transforms `ApplyScalingImages` applies to transport the weight
map (`qsiprep/interfaces/fmap.py:1091` builds
`[intramodal_affine, intramodal_warp, coreg][::-1]`, the same reversal). So the
transported weight equals the gradwarp and fieldwarp factors of the true
determinant, evaluated where the true determinant evaluates them.

The ordering is also the only one that type-checks physically. `x` is a point
in ACPC output space. Under the composite above, `coreg` acts on it first
(ACPC -> b=0-reference space), `intramodal` and `fieldwarp` and `gradwarp`
follow, and `hmc` acts last, landing in the individual volume's moved frame --
which is where the raw per-volume data actually lives, and is what
`dwi_transform` samples. The opposite ordering would feed an ACPC-space point
into an HMC affine defined in native DWI space, and would terminate in
b=0-reference space rather than the raw volume frame.

Note what this argument does *not* say. It is specific to HMC's position at the
native end of the chain. It would not hold for a transform sitting *between*
the weight map's domain and the output grid; such a transform must either be
included in the composition or applied when transporting the map. The
intramodal template warp is exactly such a transform, and it is handled -- by
`ApplyScalingImages` applying it during transport, not by inclusion in the
composite.

### This argument must be tested, not trusted

The ordering convention above is the single assumption the whole
`ComposeJacobianWeights` algorithm rests on, it is documented in language that
reads the opposite way on first encounter, and it survived one adversarial
review round only by re-deriving it. It therefore gets an executable test
rather than a prose argument:

Construct two deliberately **non-commuting** synthetic transforms -- e.g. an
anisotropic scaling and an off-centre rotation or shear -- as ITK transform
files. Resample a test image through `ApplyTransforms` with
`transforms=[A, B]`, and independently compute both candidate composites
(`A . B` and `B . A`) with direct coordinate arithmetic. Assert which one the
ANTs output matches. Because the transforms do not commute, exactly one can
match, so the test pins the convention unambiguously and will fail loudly if a
future ANTs or Nipype release changes it.

A name-matching test against `_TRANSFORM_STAGES` is not a substitute: it
verifies that our list order tracks QSIPrep's declared chain order, not that
our understanding of what ANTs does with that list is correct.

## Direction of the correction

`antsApplyTransforms --print-out-composite-warp-file` emits a pull-back field
phi mapping output coordinates to input coordinates. Where EPI distortion
compressed anatomy 2:1, correction stretches 5 distorted voxels back to 10
corrected voxels, so `|det grad phi| = 0.5` there, and the resampled signal --
still carrying the piled-up distorted intensity -- must be **multiplied** by it.

This makes total-signal conservation exact:

    integral I_corr(x) dx = integral I_dist(phi(x)) |det grad phi(x)| dx
                          = integral I_dist(y) dy

so conservation is a genuine correctness oracle for the sign, not a heuristic.
It also confirms DRBUDDI's existing ratio is a multiplicative Jacobian
surrogate -- the same *kind* of quantity, multiplicative and near unity, which
is what makes decision 3 coherent at all. It is **not** a like-for-like swap:
the ratio is an image ratio and carries non-geometric content a determinant
does not, so the change alters output values and has to be validated. See the
DRBUDDI equivalence test.

## Architecture

The single-interpolation contract is unchanged. One new node in
`init_dwi_trans_wf` (`qsiprep/workflows/dwi/resampling.py`) sits beside the
existing `compose_transforms` and produces the weight maps. The existing
`ApplyScalingImages` node consumes them exactly as it consumes DRBUDDI's ratio
images today.

```
                 +- compose_transforms --> dwi_transform (ApplyTransforms) -+
 gradwarp_field -+                                                          +-> ApplyJacobianWeights -> merge
 fieldwarps -----+- compose_jacobian ---------------------------------------+        (multiply)
 ec_jacobians ------------------------+
```

The decisive simplification is that **DRBUDDI needs no special case in
production weighting** (its validation does -- see the `_JAC` oracle below). Its
per-volume warps already arrive as `fieldwarps` (via `to_dwi_ref_warps`), so its
analytic Jacobian falls out of the same composition as every other path's. That
is why decision 3 is a net reduction in code: the `sdc_scaling_images` channel
through `fsl.py`, `diffprep.py`, `hmc_sdc.py`, `dwi/base.py` and `finalize.py`
is deleted, and the only backend-specific channel remaining is TORTOISE's
eddy-current Jacobian.

### `ComposeJacobianWeights`

New interface in `qsiprep/interfaces/gradients.py`, beside `ComposeTransforms`.

Inputs: `dwi_files` (N, native grid; supplies the volume count and the
composition reference), `gradwarp_field` (0 or 1), `fieldwarps` (0, 1 or N),
`ec_jacobian_images` (0 or N), `mask`.

`mask` is fed from `init_dwi_trans_wf`'s existing `dwi_mask` inputnode field
(`qsiprep/workflows/dwi/resampling.py:158`), which is the native-space brain
mask for the run. It exists for the positivity guard, whose in-mask criteria
are otherwise not implementable. Grid contract: `dwi_mask` must be on the same
lattice as `b0_ref_image`; if it is not, it is resampled to it with
nearest-neighbour interpolation rather than being used as-is, since a
mislatticed mask would silently move which voxels the guard inspects.

Output: `jacobian_weight_images` -- N paths, with repeats, satisfying
`ApplyScalingImages`' existing length-equality contract.

Algorithm, keyed over the *unique* `(gradwarp, fieldwarp_i)` pairs so that one
gradwarp field plus one SDC warp costs one ANTs call, DRBUDDI rpe_series costs
two, and gradwarp-only costs zero:

1. Both present: compose with
   `antsApplyTransforms --print-out-composite-warp-file -r b0_ref_image`, then
   `CreateJacobianDeterminantImage(imageDimension=3, doLogJacobian=0, useGeometric=0)`.
   The reference is a **native** lattice, deliberately not `output_grid`: the
   composite `gradwarp . fieldwarp` is a map whose domain is the undistorted
   b=0-reference space, and materializing it on the output lattice would
   evaluate a native-domain function at output coordinates. See the
   coordinate-domain table below for the domain contract this reference has to
   satisfy.
2. Exactly one present: `CreateJacobianDeterminantImage` directly on that field.
   No composition and no reference image are needed -- the tool emits on the
   deformation field's own grid.
3. Neither present: unity weight for that volume.

Then multiply in `ec_jacobian_images[i]` when present. If every factor is
absent, return `Undefined` so `ApplyScalingImages` takes its existing no-op
branch.

Composition order matches the `gradwarp` then `fieldwarp` sub-sequence of
`ComposeTransforms._TRANSFORM_STAGES`, reversed for ANTs. It is a contiguous
two-stage subset of that chain, not the whole of it; the preceding section
establishes why dropping the surrounding stages is coordinate-safe. A test
asserts the sub-sequence stays in lockstep with `_TRANSFORM_STAGES`: a
divergence here yields a plausible-looking but wrong weight map.

### Coordinate domains

`ApplyScalingImages` documents its weight maps as living in "undistorted b0ref
space" (`qsiprep/interfaces/fmap.py:1019`) and transports them through
intramodal and coregistration transforms only. The composed
`gradwarp . fieldwarp` map's domain must therefore be that space. This is not
something to assume per backend -- it is verified per backend, and the
implementation asserts it:

| Source | Domain lattice | Evidence | Status |
|---|---|---|---|
| Gradwarp from coefficients | field generated on the supplied reference grid, which is volume zero of the pre-HMC series | `qsiprep/interfaces/gradunwarp.py:88`, `qsiprep/workflows/dwi/base.py:320` | native DWI lattice, verified |
| Gradwarp from `--gradient-file` displacement field | user-supplied; **no grid or world-frame validation exists** | `qsiprep/workflows/dwi/gradwarp.py:379` | **needs a guard** |
| GRE fieldmap warp | VSM built with the b=0 reference as both input and reference | `qsiprep/workflows/fieldmap/unwarp.py:221` | b=0-reference lattice, verified |
| SyN warp | transform applied with `bold_ref` as input and reference | `qsiprep/workflows/fieldmap/syn.py:218` | b=0-reference lattice, verified |
| DRBUDDI down composite | explicitly materialized on `undistorted_reference` | `qsiprep/interfaces/tortoise.py:508` | verified |
| DRBUDDI `deformation_finv` | returned directly from TORTOISE; header not checked against `undistorted_reference` | `qsiprep/interfaces/tortoise.py:433,475` | **needs a guard** |
| TORTOISE T2Wreg `sdc_warp` | "in the DWI world frame" per its own docstring, not otherwise validated | `qsiprep/interfaces/tortoise.py` output spec | **needs a guard** |

The composition reference is `b0_ref_image`, which `init_dwi_trans_wf` already
exposes on its inputnode alongside `dwi_files`
(`qsiprep/workflows/dwi/resampling.py:154`). It is named for exactly the domain
the weight map must live in, so it is preferred over `dwi_files[0]`: the latter
would work only via the indirect argument that the SDC warp is a within-lattice
displacement on the native DWI grid, and would silently depend on every first
DWI volume sharing that intended domain. The three rows
marked "needs a guard" are cases where nothing in the current source proves it,
so `ComposeJacobianWeights` validates on entry.

**What the guard checks**, concretely, so a plan-writer need not invent it:

- `field.shape[:3] == reference.shape[:3]`
- `np.allclose(field.affine, reference.affine, rtol=1e-5, atol=1e-4)` -- loose
  enough for float32 header round-trips, tight enough that a different lattice
  or orientation fails
- the displacement axis has 3 components, and the field is shaped as ANTs
  writes ITK vector images
- `mask` agrees with `reference` on shape and affine, after the
  nearest-neighbour resampling described above

**What the guard cannot check, and what covers it instead.** Header geometry
proves only that a field is sampled on the expected lattice. It cannot prove
the field encodes the intended coordinate *domain*, and it cannot prove
*direction* -- an inverted field has identical headers. Three rows above rest
on naming and docstrings for direction, which is not proof. Direction errors
are caught behaviourally rather than structurally:

- the conservation oracle fails for an inverted field, because the modulation
  then compounds the distortion instead of undoing it;
- the positivity/median guard flags a systematically inverted determinant
  (median near `1/k` rather than `k`);
- the per-backend integration assertions compare against a known-good run.

The entry guard is therefore a cheap structural screen, and direction
correctness is established by the tests. Neither alone suffices, which is why
both exist.

**Strict lattice equality is a conservative restriction, not a mathematical
necessity.** ANTs composes transforms in physical coordinates, so a field on a
different but world-compatible lattice is not automatically wrong. The first
implementation rejects such inputs anyway, because telling "world-compatible
but differently sampled" apart from "wrong domain" needs case work that no
current QSIPrep input exercises. If a real dataset needs it, the guard grows an
explicit resample-to-reference branch rather than loosening the check.

### TORTOISE eddy-current Jacobian

DIFFPREP bakes motion+eddy into `_moteddy.nii` and emits identity affines, so
the EC Jacobian is recoverable only from the 24-parameter Okan-quadratic
`transformations_file`, already parsed by `_read_okan_transformations`
(`qsiprep/interfaces/tortoise.py:1161`).

New interface `OkanQuadraticJacobian`: reconstruct the per-volume quadratic
coordinate map, evaluate `det grad phi` analytically on the DWI grid, write one
3D map per volume. Carried to `resampling.py` on a new narrow
`ec_jacobian_images` channel (`diffprep.py` -> `dwi/base.py` -> `finalize.py` ->
`resampling.py`), replacing the deleted `sdc_scaling_images` plumbing of the
same shape.

**The formula is not in this repository.** `_read_okan_transformations` parses
24 scalars per volume and its comments identify columns 0-5 as rigid motion and
6-23 as eddy-current polynomial and centre parameters, but nothing here
specifies the parameter ordering within columns 6-23, the polynomial basis,
normalization, the rotation/eddy-centre convention, forward-versus-inverse
direction, or LPS handling beyond the six exported motion parameters. The
implementation must source the coordinate-map formula from TORTOISE's own
source or documentation as its first step. Parsing 24 numbers is not evidence
that the transform can be reconstructed.

**Mode gating.** `correction_mode` accepts `motion`, `quadratic` and `cubic`
(`qsiprep/interfaces/tortoise.py:967`), and `--sloppy` forcibly drops it to
`motion` (`qsiprep/workflows/dwi/diffprep.py:345-350`). Behaviour per mode:

- `motion`: no EC component exists. No EC Jacobian is produced, and none is
  needed -- the rigid motion determinant is excluded by the scope policy. This
  is also the `--sloppy` path, so the whole feature must be inert there rather
  than erroring.
- `quadratic` (the default): the 24-parameter reconstruction described above.
- `cubic`: a different, higher-order basis. Out of scope for the first
  implementation. It must **not** abort the run: `cubic` is an existing,
  accepted `correction_mode` (`qsiprep/interfaces/tortoise.py:967`) and
  weighting is on by default, so erroring would newly break runs that work
  today and would force users into `--no-jacobian-weighting`, losing gradwarp
  and SDC weighting as collateral. Instead the EC contribution alone is
  skipped, gradwarp and SDC weighting proceed, and a warning plus the methods
  boilerplate record the EC gap. This matches the quadratic ship-gate fallback.
  What is forbidden is silently applying the quadratic formula to cubic
  parameters.

**Ship gate.** The quadratic path ships only if a validation test passes:
resampling a volume with the reconstructed **full 24-parameter** transform must
reproduce TORTOISE's own `_moteddy.nii` to within interpolation error, starting
from the same imported image representation, with matching interpolation and
boundary handling, compared before any later SDC stage. `_moteddy.nii` is
exactly the motion+eddy output in the input grid
(`qsiprep/interfaces/tortoise.py:1123`), so it is the right target.

Note the gate's limit: reproducing `_moteddy.nii` validates the *combined*
motion+EC map, not the EC component in isolation. Splitting the EC determinant
out of it is only sound if the motion component is genuinely rigid, which for
DIFFPREP it is (`correction_mode` documents columns 0-5 as rigid motion, unlike
SHORELine's affine default). The test therefore validates the convention, and
the rigid-motion property is what licenses the split. Both must hold. If either
fails, the EC Jacobian becomes a documented gap rather than shipping
unvalidated. Nothing else in this spec depends on it.

## Per-backend contract

| Backend / fieldmap | Gradwarp J | SDC J | EC J | Change |
|---|---|---|---|---|
| `eddy` + TOPUP, `method=jac` | new | already in `eddy` | already in `eddy` | gradwarp J added |
| `eddy` + TOPUP, `method=lsr` | new | **none** (warn) | **none** (warn) | gradwarp J added |
| `eddy`, no fieldmap, `method=jac` | new | n/a | already in `eddy` | gradwarp J added |
| `eddy`, no fieldmap, `method=lsr` | new | n/a | **none** (warn) | gradwarp J added |
| `eddy`/`tortoise` + DRBUDDI | new | analytic, replaces ratio | backend | ratio deleted |
| GRE fieldmap | new | new | per HMC backend | new |
| SyN fieldmapless | new | new | per HMC backend | new |
| TORTOISE T2Wreg | new | new (from `sdc_warp`) | new | new |
| `tortoise` DIFFPREP | new | per fieldmap | new, from `transformations_file` | new |
| SHORELine / `3dSHORE` | new | per fieldmap | no separate EC stage (see note) | new |

Two notes on the table.

`init_sdc_unwarp_wf`'s existing `out_jacobian` is **not** wired up. It is the
Jacobian of the SDC warp alone, but the design needs the Jacobian of the
gradwarp-composed-with-SDC warp, and the two are not related by a simple
product. The `jac_dfm` node and the `out_jacobian` output are deleted as dead
code instead.

`eddy`'s contribution is **not** recomputed, and cannot double-count. On the
TOPUP-only path `fieldwarps` carries `gather_inputs.forward_warps`, which
`GatherEddyInputs` sets to `[]` unconditionally
(`qsiprep/interfaces/eddy.py:206`). This was checked, not assumed; it is now a
**regression invariant** with a test, not an open question.

## CLI and config

`--jacobian-weighting` / `--no-jacobian-weighting` in `qsiprep/cli/parser.py`,
default on, backed by `config.workflow.jacobian_weighting`. When off,
`compose_jacobian` is not built and `ApplyScalingImages` is fed nothing, taking
its existing pass-through branch.

Help text, stating the constraint rather than implying a clean switch:

> Apply Jacobian intensity modulation for gradient-nonlinearity, susceptibility
> and eddy-current distortion corrections (default: on). This option controls
> only the modulation QSIPrep itself applies. With `--hmc-method eddy`, FSL
> `eddy` applies its own modulation for eddy-current and TOPUP susceptibility
> distortions whenever its resampling method is `jac` (the default; see
> `--eddy-config`), and that is internal to `eddy` and unaffected by this
> option.

The earlier draft of this help text claimed the option suppresses "only the
gradient-nonlinearity modulation" under `--hmc-method eddy`. That is wrong and
must not be restored: on `eddy` + DRBUDDI, `eddy` + GRE and `eddy` + SyN, the
new node also supplies the SDC determinant, so disabling it suppresses that
too. The text above scopes the statement to what `eddy` does internally rather
than enumerating what the flag suppresses.

## Derivatives

`ApplyScalingImages` is renamed `ApplyJacobianWeights` and gains an output
carrying the unique output-grid weight maps it already computes internally and
currently discards. A new sink in `qsiprep/workflows/dwi/derivatives.py` writes:

    sub-X[_ses-Y]..._space-ACPC_desc-jacobian_dwi.nii.gz

a 4D file of the unique maps, collapsing to 3D when all volumes share one, plus
a sidecar. The sidecar schema, specified rather than left to the implementer:

```json
{
  "JacobianWeightIndex": [0, 0, 1, 0],
  "AppliedCorrections": ["gradwarp", "sdc"],
  "UnmodulatedCorrections": ["eddy-current"],
  "UnmodulatedReason": "TORTOISE correction_mode=cubic is not supported"
}
```

- `JacobianWeightIndex` is **zero-based**, has one entry per volume of the
  *preprocessed DWI series*, and indexes volumes of this 4D weight file.
  Repeated maps are represented by repeated indices, which makes the dedup
  visible rather than implicit. In the collapsed 3D case it is all zeros,
  written out in full rather than omitted, so consumers need no special case.
- `AppliedCorrections` lists corrections whose determinants are *in* these
  maps, drawn from `gradwarp`, `sdc`, `eddy-current`.
- `UnmodulatedCorrections` lists corrections that happened but were **not**
  modulated, with `UnmodulatedReason` giving the cause in prose. This is how
  partial-gap cases surface in the derivatives rather than only in logs:
  `eddy` with `method: lsr`, TORTOISE `cubic`, and a failed Okan ship gate all
  populate it. An empty list means full coverage.
- Corrections that did not occur at all appear in neither list.

Size, stated plainly: with TORTOISE EC active every volume is unique, so this is
a full-length 4D series. Jacobian maps are smooth and `float32`, so gzip does
far better on them than on noisy DWI data, but this is the one place the design
spends real disk. Every `qsiprep/tests/data/*_outputs.txt` expected-file list
needs the new entry.

## Testing

**Analytic unit tests** on `ComposeJacobianWeights` against synthetic fields
with known determinants: pure translation -> 1.0; uniform 2x scaling along one
axis -> 2.0; a 1D linear ramp shear -> its closed-form value. These catch a
sign/reciprocal error directly.

**Conservation test**, the primary oracle. On a small synthetic series with a
known nontrivial warp, summed in-mask signal after weighting matches the
unweighted raw series within interpolation tolerance. The weighting-disabled
output must *fail* this check, or the test proves nothing.

**Double-counting test.** On the TOPUP-only path, assert that the `fieldwarps`
reaching `compose_jacobian` contain no TOPUP field. A wiring assertion in the
style of `qsiprep/tests/test_intramodal_transforms.py`, plus an integration
check that `dsdti_topup` mean b=0 intensity does not shift by the square of the
expected factor.

**DRBUDDI equivalence.** Correlation against the ratio image is too weak an
acceptance criterion on its own: the ratio
(`undistorted_reference / blip_*_b0_corrected`,
`qsiprep/interfaces/tortoise.py:527`) is an image ratio, so it also carries
interpolation differences, b=0 noise, contrast differences and whatever
intensity processing DRBUDDI's reference construction applied. A geometric
determinant carries only local volume change, and no identity makes the two
equal. Discarding the non-geometric content is the *point* of the change -- it
does not belong in a modulation weight -- but that has to be demonstrated, not
asserted.

The primary oracle is therefore TORTOISE's own Jacobian products, not the
ratio. DRBUDDI already emits `blip_up_b0_corrected_JAC` and
`blip_down_b0_corrected_JAC` (`qsiprep/interfaces/tortoise.py:400-442`), wired
into the aggregator at `qsiprep/workflows/fieldmap/drbuddi.py:241` and
currently unused downstream. The implementation's first step on this path is to
establish what those files contain; if they are the jacobian-modulated
corrected b=0s, then `_JAC / b0_corrected` is TORTOISE's own determinant and
our analytic Jacobian must match it to tight tolerance. That is an equivalence
test against the tool's own arithmetic rather than a correlation against a
noisy surrogate.

**If `_JAC` turns out to mean something else**, the decision tree does not
dead-end. The fallback order is: (1) validate the analytic determinant against
the synthetic analytic tests and the conservation oracle, which do not depend
on TORTOISE at all, and accept the replacement on that basis with the ratio
comparison as a regression check; (2) if the analytic determinant and the ratio
disagree beyond the agreed threshold and no third oracle explains why, keep
DRBUDDI on its existing ratio images for this release and ship the analytic
path for the other backends only. Option (2) means retaining the
`sdc_scaling_images` channel for DRBUDDI alone, which costs the code deletion
this design otherwise gets -- so it is a real cost, deliberately accepted
rather than discovered late.

The ratio comparison is retained as a secondary, looser check (in-mask Pearson
r plus median agreement, thresholds set from a first measured run rather than
asserted blind), to catch a gross regression on the most-used SDC path.

**Okan convention test**, integration-marked (needs TORTOISE). See the ship gate
above.

**Wiring tests** in the style of `qsiprep/tests/test_workflows_gradwarp.py`: for
each backend x fieldmap branch, `compose_jacobian` exists and is connected to
`ApplyJacobianWeights`; under `--no-jacobian-weighting` it is absent everywhere.

**Positivity guard** inside the interface, with two distinct severities so a
real fold fails loudly while a merely unusual field does not block a run:

- Hard failure: any non-finite voxel, or any non-positive voxel in-mask. A
  folded warp gives a non-positive determinant, and silently multiplying DWI
  data by a negative number is worse than failing.
- Logged warning: in-mask median outside `[0.9, 1.1]`. A correct distortion
  field redistributes signal without changing its total, so the median should
  sit near unity; a median far from it means the field or the composition order
  is wrong, but the bound is a smell test and not a correctness claim.

**Integration.** Existing markers already cover every branch -- `dsdti_topup`,
`maternal_brain_project` and `forrest_gump` (GRE), `dsdti_synfmap` (SyN),
`drbuddi_*`, `diffprep*`. No new CI jobs; extend assertions and the
expected-output lists.

## Documentation

- `docs/preprocessing.rst`: new subsection under the transform-chain discussion
  (~line 709) stating which corrections are Jacobian-modulated, by which
  component, and the `eddy` asymmetry. The "two total interpolations" text
  (~line 817) gains the modulation story.
- `docs/usage.rst`: the new flag, with the `eddy` caveat.
- `docs/api.rst`: the new interfaces.
- Methods boilerplate: extend `boilerplate_from_eddy_config` and
  `gradwarp_boilerplate` so the generated methods section states whether
  modulation was applied. Citation added to `qsiprep/data/boilerplate.bib`.
- `docs/changes.md` is **not** edited; it is generated from PR titles at release
  time. The change description goes in the PR body.

## Risks

1. **Okan formula is not in this repository.** Highest remaining risk. It must
   be sourced from TORTOISE before the EC work starts, and it is fenced by the
   ship gate and mode gating above.
2. **DRBUDDI numeric change.** A values change to the most-used SDC path.
   Mitigated by the `_JAC` equivalence test, which is a real oracle rather than
   a correlation, but the values do change.
3. **Unvalidated input domains** for user-supplied gradwarp displacement
   fields, DRBUDDI's `deformation_finv`, and TORTOISE's T2Wreg `sdc_warp`. The
   entry guard in `ComposeJacobianWeights` converts a silent wrong-space bug
   into a loud failure, but the guard's tolerance needs setting from real data.
4. **`--eddy-config` with `method: lsr`** leaves EC and susceptibility
   unmodulated and unrecoverable. Handled by warning rather than by correction.
5. **Derivative size** when TORTOISE EC is active.

Resolved by the adversarial review, recorded so they are not re-litigated:

- **TOPUP double-counting** does not occur; `forward_warps` is unconditionally
  empty (`qsiprep/interfaces/eddy.py:206`). A regression test keeps it that way.
- **Omitting HMC from the composition is coordinate-safe**, because HMC is the
  outermost function in the pull-back. See the dedicated section above for the
  derivation. This holds despite SHORELine's HMC being affine.

## Review record

An adversarial design review was run against this spec before implementation
(Codex, 2026-09-17). Its findings were adjudicated individually against the
source:

**Accepted and folded in:** `method: "jac"` is a user-overridable default, not
an invariant; SHORELine's default HMC is `Affine` so the "rigid" rationale was
wrong; the CLI help contradicted the backend table; TORTOISE mode gating was
unspecified; the positivity guard referenced a mask the interface did not take;
DRBUDDI's unused `_JAC` outputs are a better oracle than the ratio; the
coordinate-domain claims needed a per-backend table and an entry guard rather
than a "world coordinates" hand-wave; the Okan coordinate formula is absent
from this repository and the `_moteddy.nii` gate validates the combined
motion+EC map rather than EC alone.

**Rejected:** that omitting HMC misplaces the nonlinear determinants. The
review applied the chain rule as `det D(F . R)(x) = det DF(Rx) . det DR`, which
holds when `R` is applied first to the output point. HMC is applied last --
`_TRANSFORM_STAGES` is native-to-target and is reversed for ANTs
(`qsiprep/interfaces/gradients.py:533`) -- so its determinant is a constant
factor that cannot move `u` or `v`. Verified independently against the DRBUDDI
composite at `qsiprep/interfaces/tortoise.py:508`, where the physical ordering
is unambiguous.

**Rejected:** that `output_grid` should be the composition reference because
that is what `ComposeTransforms` uses. The composite `gradwarp . fieldwarp` has
a native domain; materializing it on the output lattice would evaluate a
native-domain function at output coordinates. The review's underlying point --
that the native lattice choice was asserted rather than shown -- is accepted
and answered by the coordinate-domain table. **Conceded in round two**, and the
reference changed from `dwi_files[0]` to the explicit `b0_ref_image`.

### Round two

A second pass was run on the revised spec (same Codex thread). It conceded the
`output_grid` point and held its ground on the HMC ordering, citing Nipype's
`transforms` docstring ("the last specified transform will be applied first",
`nipype/interfaces/ants/resampling.py:361`) and QSIPrep's own comment at
`qsiprep/workflows/dwi/diffprep.py:599`.

**Still rejected, and the reason is now written into the spec.** Both quoted
sources describe *image-warp* order; the derivation needs *point-map* order,
and the two are opposite. The `diffprep.py` comment says so itself: it
describes the resampling as happening "in chain order (gradwarp, then SDC)" --
the image-operation order -- and derives the reversed ANTs list `[SDC,
gradwarp]` from it. Parsed that way, the comment supports this spec's ordering
rather than contradicting it. See "Disambiguating 'applied first'" above, which
exists because this ambiguity cost two review rounds and would have cost an
implementer more.

The round-two suggestion to settle it with two non-commuting synthetic
transforms was adopted, and is a better answer than any amount of prose: see
"This argument must be tested, not trusted".

Accepted from round two and folded in: the explicit `b0_ref_image` reference;
conditioning the backend table on `jac` versus `lsr`; naming where the
effective eddy method is resolved; narrowing the `lsr` claim to "not
Jacobian-modulated"; not aborting `cubic` TORTOISE runs but skipping only the
EC contribution; wiring `mask` from the existing `dwi_mask` with a stated grid
contract; a defined fallback if DRBUDDI's `_JAC` files mean something other
than assumed; concrete guard checks plus an explicit statement of what header
geometry cannot prove and which tests cover direction instead; framing strict
lattice equality as a conservative restriction; the sidecar schema; removing
the "like-for-like" contradiction; qualifying "DRBUDDI needs no special case";
and deleting the stale TOPUP risk note.

## Out of scope

- Head motion correction. Excluded by policy, not because its determinant
  is unity -- SHORELine's affine default has a non-unit constant determinant.
- Coregistration, ACPC, intramodal-template and template-space warps.
- `t1_2_mni_forward_transform`, which `resampling.py` accepts on its inputnode
  but never connects, so output space is ACPC in practice.
- The `graddev` voxelwise gradient deviation map, which concerns diffusion
  encoding rather than voxel position.
