# Gradient nonlinearity distortion correction

Date: 2026-08-25
Status: approved, ready for implementation planning

## Problem

Gradient coils deviate from their nominal linear field. The deviation causes two
distinct errors, and QSIPrep currently corrects neither:

1. **Spatial.** Voxels are displaced, increasingly so away from isocentre.
2. **Diffusion encoding.** The gradient actually applied at a voxel is `L @ g`
   rather than the nominal `g`. Because `L` is a general 3x3 matrix — scaling
   and shear, not a rotation — both the direction and the magnitude of the
   encoding are wrong, so the b-vector *and* the b-value deviate per voxel.

The two are separable, and scanners already correct some of the first. Siemens
tags reconstructed images in the DICOM `ImageType` field: `DIS2D` means only
in-plane gradwarp was applied at the scanner, leaving through-plane distortion;
`DIS3D` means the full spatial correction was applied.

Neither tag implies any correction to the diffusion encoding. That correction
cannot be expressed in the bval/bvec table at all: those files hold one value
per volume and have nowhere to put spatially varying information. It is why
grad_dev exists as a separate voxelwise output, and why a run tagged `DIS3D`
still needs one.

This spec adds both corrections to QSIPrep, driven by a gradient nonlinearity
coefficient file, using TORTOISE V4.

## What TORTOISE does

Established by reading TORTOISEV4 at `main`. Cited here because several
behaviours are load-bearing and none are documented upstream.

| Location | Behaviour |
| --- | --- |
| `DRBUDDI_parserBase.cxx:130-138` | `--grad_nonlin file[is_GE,warp_dim]`; file is `.grad` (Siemens), `.dat` (GE), `.gc` (binary), or a `.nii` ITK displacement field |
| `DRBUDDI_parserBase.cxx:442-457` | Unrecognised extension prints a warning and **silently disables** correction |
| `DRBUDDI_parserBase.cxx:141-148` | `--NO_gradwarp` disables the spatial warp but still emits a voxelwise b-matrix, for data the scanner already corrected |
| `TORTOISE.cxx:1943-2023` | At the Import step, builds the gradwarp displacement field once per run |
| `TORTOISE.cxx:1994-2012` | `warp_dim` masks field components: `2D` zeroes component 2; `1D` zeroes components 0 and 1 |
| `TORTOISE.cxx:1977-1992` | `is_GE` additionally shifts the field origin in z |
| `DIFFPREP.cxx` | Motion/eddy estimation **never** sees the gradwarp field |
| `DRBUDDI.cxx:133-183`, `EPIREG.cxx:68-88` | SDC estimation resamples its b0/FA inputs through the gradwarp field first |
| `FINALDATA.cxx:930-976` | Composite transform order: motion/eddy, then gradwarp, then EPI/SDC — one resample |
| `FINALDATA.cxx:548` | Loads the file named `..._gradnonlin_field_inv.nii` into the `gradwarp_field` used for composition |
| `FINALDATA.cxx:3404-3446` | grad_dev is computed using only a rigid b0-to-structural transform; the nonlinear SDC chain is not inverted |
| `CMakeLists.txt:251,434` | `CreateNonlinearityDisplacementMap` and `CreateGradientNonlinearityBMatrix` are built as standalone binaries |

**Two files, one name.** `CMakeLists.txt:251` builds
`CreateNonlinearityDisplacementMap` from `src/tools/gradnonlin/mk_displacementMaps.cxx`.
A second, **unbuilt** copy sits at `src/tools/CreateNonlinearityDisplacementMap/`
and differs in ways that matter. Every statement below describes the *built*
file; read that one, not the one whose directory matches the binary's name.

Three findings are silent-wrong-answer hazards and are called out again where
they bite:

- **Argument order.** The built `main` calls `mk_displacement(argv[1], img, is_GE)`
  — coefficient file first, base NIfTI second. The unbuilt duplicate has them
  reversed.
- **The reference is read as 3D.** The built `main` does
  `readImageD<ImageType3D>(argv[2])`, not `read_3D_volume_from_4D`. Handing it a
  4D DWI series throws in ITK, so a single 3D volume must be extracted first.
  The gradwarp field depends only on the sampling grid, so any volume on that
  grid serves.
- **No inversion.** `mk_displacement` returns what TORTOISE names
  `gradwarp_field_inv`, and that is the file `FINALDATA.cxx:548` composes and
  `DRBUDDI.cxx:141` resamples with. The binary's direct output is what we want;
  the separate `InvertDisplacementField` step feeds a variable we do not use.

**Corrected 2026-08-25.** An earlier revision of this spec claimed `is_GE` was
read as `(bool)(argv[4])` — a pointer cast making `"0"` evaluate to true — and
built a wrapper hazard around it. That is the *unbuilt* duplicate. The built
tool uses `(bool)atoi(argv[4])`, so `"0"` correctly means false. The
implementation is unaffected: it omits the argument when not GE, which leaves
`is_GE` false under either reading. `CreateGradientNonlinearityBMatrix` uses
`atoi` as well (`getIsGE()`), so `--isGE 0` is likewise safe there — the two
tools are *not* asymmetric in this respect, as an earlier revision claimed.

## Decisions

| Decision | Choice |
| --- | --- |
| Outputs | Spatial unwarp and grad_dev, independently toggled |
| Backends | All (`eddy`, `tortoise`, `3dSHORE`/`tensor`, `none`) |
| Field generation | TORTOISE's standalone binaries, already in the container |
| `--gradient-file` scope | One path, dataset-wide |
| Auto-detection | No DIS tag: 3D. `DIS2D`: through-plane only. `DIS3D`: grad_dev only |
| `--ignore gradients` | Both outputs off |
| `--force gradients` | Full 3D, regardless of `ImageType` |
| SDC fidelity | Gradwarp-correct SDC estimation inputs where the field is applied downstream of gradwarp |
| Overrides | None; `is_GE` and `warp_dim` are derived from metadata |
| Dead local-bvec code | Left alone; out of scope |

`DIS2D` maps to a through-plane-only warp rather than to "off". The scanner
corrected in-plane distortion; the residual is exactly the component
`warp_dim=1D` retains. Treating `DIS2D` as "off" would leave a real, correctable
distortion in the data.

## Architecture

```
raw DWI (native grid)
   |
   +-- merge / denoise / Gibbs          unchanged
   |
   +-- HMC estimation                   unchanged; never sees gradwarp
   |
   +-- gradwarp_wf --------------------> gradwarp field (native space, ITK)
   |
   +-- SDC estimation                   gets gradwarp-corrected b0s
   |                                    (except eddy+TOPUP; see below)
   |
   +-- ComposeTransforms
         hmc -> GRADWARP -> fieldwarp -> intramodal -> coreg -> (mni)
```

The single-resample guarantee holds: the gradwarp field only ever enters the
composed transform chain. The DWI series is never resampled by it separately.
The only added resampling is of the b0/FA images used for SDC estimation, which
is what TORTOISE does.

### Where SDC estimation sees a corrected image

The backends differ in where SDC lands. `fsl.py:255-273` shows eddy's
`out_corrected` feeding `dwi_files_to_transform` directly: eddy resamples once
itself, and with TOPUP it applies the susceptibility field internally via
`--field`. That SDC is baked in *upstream* of `ComposeTransforms`. In the
DRBUDDI, GRE, SyN and TORTOISE paths, SDC remains a separate warp in
`to_dwi_ref_warps`, applied *downstream* of where gradwarp sits.

The rule:

> Gradwarp-correct the SDC estimation inputs exactly when that SDC field is
> applied downstream of gradwarp in the composed chain.

- **eddy + TOPUP**: TOPUP keeps raw b0s. Eddy applies the field to raw data, so
  estimating it on raw data is internally consistent. Gradwarp composes after
  eddy's output, which is on the native grid.
- **eddy + DRBUDDI / GRE / SyN, TORTOISE + anything, SHORELine**: SDC estimation
  receives gradwarp-corrected inputs, matching `DRBUDDI::Step0_CreateImages` and
  `EPIREG`.

### grad_dev fidelity

`CreateGradientNonlinearityBMatrix` rigidly registers its `-i` (initial) image to
its `-f` (final) image and computes the L matrix through that rigid transform.
It does not invert the nonlinear SDC chain. This matches TORTOISE's own internal
path at `FINALDATA.cxx:3404-3446`, which uses only `b0_t0_str_trans`. Using the
standalone binary is therefore exactly as faithful as TORTOISE is to itself, and
no more approximate.

## CLI and configuration

In `qsiprep/cli/parser.py`, group `g_conf`:

```python
# existing --ignore, choices gain 'gradients'
choices=['fieldmaps', 't2w', 'phase', 'sdc', 'gradients']

# new
'--force', nargs='+', default=[], choices=['gradients']
'--gradient-file', action='store', type=IsFile   # IsFile at parser.py:275
```

`--force` is new to QSIPrep; only `--force-syn` exists today. It is defined with
a single choice, extensible later. Folding `--force-syn` into `--force syn` is a
separate deprecation-bearing change and is **not** in scope.

`qsiprep/config.py`, `workflow` section: two new keys, `force` and
`gradient_file`, alongside the existing `ignore`.

Validation in `parse_args`, following the `--eddy-config` precedent at
`parser.py:967`:

| Condition | Result |
| --- | --- |
| `gradients` in both `--force` and `--ignore` | `parser.error` |
| `--force gradients` without `--gradient-file` | `parser.error` |
| `--gradient-file` extension not in `.grad`, `.dat`, `.gc`, `.nii`, `.nii.gz` | `parser.error` |
| `--gradient-file` given with `--ignore gradients` | warn that the file is unused; continue |
| `--gradient-file` absent, no flags | feature off; `ImageType` never consulted |

The extension check is a hard error, departing from TORTOISE, which warns and
silently disables correction. Silently producing uncorrected output is the wrong
default for a batch pipeline.

## Per-unit plan resolution

New `qsiprep/workflows/dwi/gradwarp.py` holds the decision and returns a frozen
`GradwarpPlan(warp_dim, is_ge, coeff_file)` or `None`.

`warp_dim` comes from `ImageType`, already parsed into `FileRecord.metadata`
(`grouping/models.py:206`), so no new BIDS reads are needed:

| Input | `warp_dim` | grad_dev |
| --- | --- | --- |
| `--ignore gradients` | no plan | not written |
| `--force gradients` | `3D` | written |
| `ImageType` contains `DIS3D` | `None` | written |
| `ImageType` contains `DIS2D` | `1D` | written |
| neither tag, or no `ImageType` | `3D` | written |

A `CorrectionUnit` spans several `dwi_files` that are concatenated *before* HMC
and share one field, so they must agree. When they disagree, take the
**minimum** warp under `none < 1D < 3D` and log a warning naming the disagreeing
files. Applying a 3D field to a run the scanner already corrected in-plane would
double-correct; under-correcting is the recoverable error.

`is_ge` comes from the BIDS `Manufacturer` field. It drives the z-origin shift at
`TORTOISE.cxx:1977-1992` and the GE coefficient reader at `gradcal.cxx:185`. The
value is free text from DICOM (`SIEMENS`, `GE MEDICAL SYSTEMS`, `Philips Medical
Systems`, or absent), so the matcher must handle variants, not one spelling.

## Interfaces

New `qsiprep/interfaces/gradunwarp.py`, following the `TORTOISECommandLine`
pattern at `interfaces/tortoise.py:73`.

**`CreateNonlinearityDisplacementMap`** — `CommandLine`. Positional arguments
`(coeff_file, ref_image, out_field)`, plus a conditional trailing `is_ge`
argument. Outputs the gradwarp field in native space.

Two constraints, both from the findings above: the coefficient file is the
*first* positional argument, and the `is_ge` argument is **appended only when
GE** — never formatted as `%d`, because `0` is read as true.

**`MaskWarpDimensions`** — `SimpleInterface`, pure nibabel/numpy. Zeroes field
components per `warp_dim`, reproducing `TORTOISE.cxx:1994-2012`: `3D`
passthrough, `2D` zeroes component 2, `1D` zeroes components 0 and 1.

**`CreateGradientNonlinearityBMatrix`** — `CommandLine` with `-f`, `-i`, `-g`,
and a conditional `--isGE`. Its outputs are named by derivation from `-f`'s stem
(`_graddev_c.nii` for a coefficient input, `_graddev_f.nii` for a field input),
so `_list_outputs` must build both candidate names and select the one present on
disk.

**`init_gradwarp_wf(unit)`** in `qsiprep/workflows/dwi/gradwarp.py` chains the
first two and exposes `outputnode.gradwarp_field`, carrying the resolved `.plan`
for the report and the methods boilerplate. It builds only nodes whose outputs
are consumed:

- When the plan resolves to grad_dev-only (`DIS3D`), it builds **no nodes at
  all**. Nothing resamples through a field for such a unit, and the grad_dev
  node in `finalize` is fed the *coefficient* file rather than a field
  (`CreateGradientNonlinearityBMatrix` accepts either), so generating one here
  would invoke an external binary per unit and discard both of its outputs. The
  workflow object still exists — and is still added to the parent graph — so the
  plan and the `DIS3D` boilerplate survive.
- When `--gradient-file` is a `.nii`/`.nii.gz` ITK displacement field rather
  than a coefficient file, `CreateNonlinearityDisplacementMap` is skipped and
  the file feeds `MaskWarpDimensions` directly. That binary *is* the coefficient
  expander and does no extension dispatch of its own — TORTOISE branches on the
  extension in `TORTOISE.cxx:1943-2023`, before ever calling it.

`init_gradwarp_wf` sets `.needs_reference` to say whether
`inputnode.ref_image` is consumed, so `base.py` knows whether to build the 3D
extraction node that feeds it.

## Wiring

| File | Change |
| --- | --- |
| `interfaces/gradients.py:445` | `gradwarp` entry in `transform_order`, between `hmc` and `fieldwarp` |
| `interfaces/gradients.py:497` | add `gradwarp` to the `ifargs.pop` key list so it is not passed to `ApplyTransforms` |
| `workflows/dwi/base.py` | instantiate `gradwarp_wf` once per unit; pass the field to hmc_sdc and finalize |
| `workflows/dwi/fsl.py` | gradwarp-resample SDC estimation inputs in the DRBUDDI, GRE and SyN branches; TOPUP branch untouched |
| `workflows/dwi/diffprep.py`, `workflows/dwi/hmc_sdc.py` | same, for their SDC branches |
| `workflows/dwi/resampling.py` | new `gradwarp_field` input on `inputnode`, forwarded to `compose_transforms` |
| `workflows/dwi/finalize.py` | grad_dev node and derivative sink |

## Outputs

### grad_dev derivative

A new `graddev` suffix, added to `default_path_patterns` in
`qsiprep/data/io_spec.json`:

```
sub-{subject}[/ses-{session}]/{datatype<dwi>|dwi}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_rec-{reconstruction}][_dir-{direction}][_run-{run}][_part-{part}][_chunk-{chunk}][_space-{space}][_desc-{desc}]_{suffix<graddev>|graddev}.{extension<nii|nii.gz|json>|nii.gz}
```

producing `sub-X_..._space-ACPC_graddev.nii.gz`: a 9-volume float NIfTI in ACPC
space holding the row-major 3x3 L matrix per voxel, in the component order
`CreateGradientNonlinearityBMatrix` writes at its `Limg[0..8]` assignments.

The name matches what HCP, FSL and DSI Studio already use.
`DSIStudioCreateSrc.grad_dev` at `interfaces/dsi_studio.py:82` already accepts
exactly this file.

grad_dev is **not** a spatial transform and must not be filed under `xfm`: it
maps gradient vectors, not coordinates, and never displaces a voxel. It is also
not a `dwimap`, which in QSIPrep denotes a derived map of the tissue
(`model-eddy_stat-cnr_dwimap`); grad_dev describes the scanner.

It is written whenever `--gradient-file` is given and `--ignore gradients` is
absent, including the `DIS3D` case — which is the point of separating the two
outputs.

Its JSON sidecar records:

- `GradientCoefficientFile` — basename only, never the full host path
- `GradientWarpDimensions` — `3D`, `1D`, or `none`
- `GradientCoefficientIsGE` — boolean. Not a `Manufacturer` string: all that is
  resolved is whether TORTOISE's GE code path was taken.
- how the decision was reached: `metadata` or `--force gradients`

so a reader can distinguish an auto-detected run from a forced one without the
logs.

### Main output sidecar

`GradientWarpDimensions` is also written into `desc-preproc_dwi.json` through the
existing `DerivativesSidecar` path. Whether the images were spatially unwarped
changes how they should be interpreted.

### Report

`DiffusionSummaryInputSpec` (`interfaces/reports.py:203`) gains a
`gradient_correction` string, rendered as one more `<li>` beside the existing
`HMC Model` and `Impute slice threshold` lines. Values read as `3D (from
ImageType)`, `through-plane only (ImageType: DIS2D)`, `b-matrix only (ImageType:
DIS3D)`, `forced 3D`, or `none`.

No new figure. A gradwarp field has no natural before/after b0 panel that is not
already dominated by the SDC reportlet.

### Boilerplate

`init_gradwarp_wf` sets `workflow.__desc__`, following the
`generate_diffprep_boilerplate` precedent in `interfaces/tortoise.py`, citing
TORTOISE V4 and stating the warp dimensionality actually used and where in the
chain it was applied. The text must vary with the resolved plan: boilerplate
claiming 3D correction on `DIS2D` data would be a methods-section error.

### Logging

One `config.loggers.workflow.info` per correction unit, stating the resolved plan
and its basis, plus the mixed-`ImageType` warning. `resolve_gradwarp_plan` is
called three times for one unit (base, the selected HMC/SDC backend, finalize),
so it suppresses exact repeats of a rendered line: every message names
`unit.output_name`, and one plan printed three times reads like three units.

## Testing

The TORTOISE binaries are not present in the local `lincapps` environment.
Everything decidable in Python gets a real unit test; everything needing a binary
gets an interface-construction test plus a container-only integration check.

**Pure Python** — new `qsiprep/tests/test_gradwarp.py`, in the style of the
`test_grouping_*.py` modules, using `grouping_scenarios.py` fixtures:

- Plan resolution across the full `ImageType` x `--force` x `--ignore` x
  `--gradient-file`-present matrix, including `ImageType` absent, `ImageType`
  present with neither tag, and both tags on one run.
- Mixed-`ImageType` units resolve to the minimum warp and emit the warning.
- `is_ge` detection across realistic `Manufacturer` strings.
- `MaskWarpDimensions` on a synthetic 3-component field: `3D` is identity, `2D`
  zeroes only component 2, `1D` zeroes components 0 and 1 and preserves 2.

**CLI** — added to `test_cli.py`: every `parser.error` case above, and that
`ImageType` is not consulted when `--gradient-file` is absent.

**Interface construction** — new `test_interfaces_gradunwarp.py`, asserting
`cmdline` without executing. Dedicated tests for the two hazards, which fail
silently rather than crashing:

- `is_ge=False` produces a command line with **no** fourth argument.
- The coefficient file is the first positional argument, the base NIfTI second.
- `_list_outputs` resolves `_graddev_c.nii` against `_graddev_f.nii` from disk.

**Workflow wiring** — `test_workflows_native.py` style, building workflows
without running them:

- `gradwarp` appears between `hmc` and `fieldwarp` in the composed chain, for
  every `--hmc-model`.
- The placement rule holds per backend: SDC estimation inputs are
  gradwarp-resampled in the DRBUDDI, GRE and SyN branches, and not in the
  eddy+TOPUP branch.
- `DIS3D` builds the grad_dev node but wires no field into `ComposeTransforms`.
- `--ignore gradients` produces a workflow identical to today's. This is the
  regression guard that matters most: the default path must be untouched when the
  feature is off.

**Integration** — one container-only test on the existing `drbuddi_rpe` fixture
with a synthetic Siemens `.grad` file, asserting the `graddev` derivative exists
with 9 volumes and that the preproc DWI is still produced. Marked to skip outside
the container.

## Out of scope

- Migrating `--force-syn` to `--force syn`.
- The vestigial local-bvec machinery (`LocalGradientRotation`,
  `local_bvec_rotation`, `write_local_bvecs`, the commented-out sinks in
  `derivatives.py`). Left untouched.
- Per-site or per-subject coefficient files. One dataset-wide file; multi-site
  users run QSIPrep once per site.
- Inverting the nonlinear SDC chain when computing grad_dev. TORTOISE does not
  do this either.
- `vbmat` output (full per-volume voxelwise b-matrix). At 6 components per
  volume it is impractical as a derivative, and `grad_dev` is what downstream
  tools consume.

### Known limitations, found during implementation

Both are second-order placement errors of the same kind as the eddy+TOPUP
carve-out this design already accepts. Each deserves its own follow-up spec.

- **DIFFPREP's `T2Wreg` path is not gradwarp-corrected.** Its EPI field is
  estimated *inside* the `TORTOISEProcess` binary, so there is no hand-off for
  qsiprep to interpose gradwarp-corrected inputs on — yet the resulting
  `sdc_warp` is applied downstream of gradwarp like every other SDC warp.
  Correcting it means plumbing `--grad_nonlin` through the DIFFPREP interface,
  which reverses this design's decision to use the standalone tools. Affects
  only `--hmc-model tortoise` with fieldmap-less T2Wreg SDC. Recorded in code
  at the branch itself.

- **The b0→T1w coregistration reference is not gradwarp-consistent across
  backends.** `ComposeTransforms` applies the coregistration affine *after*
  gradwarp, but `b0_template` — the image `b0_coreg_wf` estimates that affine
  from — is gradwarp-corrected only on the DRBUDDI and GRE/SyN branches, as a
  side effect of correcting their SDC estimation inputs. It stays raw on the
  TOPUP-only branch, the fsl/diffprep no-fieldmap branches, and the T2Wreg
  branch. The five branches are therefore mutually inconsistent. Making them
  consistent means touching the TOPUP branch this design deliberately carves
  out and the T2Wreg branch that cannot be reached at all, so it is not a
  local fix.
