# `--output-spaces` for QSIPrep

Design spec — 2026-08-26
Issue: [PennLINC/qsiprep#681](https://github.com/PennLINC/qsiprep/issues/681)
Related: [nipreps/niworkflows#997](https://github.com/nipreps/niworkflows/issues/997)

## Summary

Replace `--output-resolution` and `--anatomical-template` with a single
`--output-spaces` argument modeled on fMRIPrep's. QSIPrep gains the ability to write
preprocessed DWI at several isotropic resolutions in ACPC space, to resolve output
resolution from the input data (`res-nativemin` / `res-nativemax`), to produce
transforms and anatomical derivatives for several standard spaces in one run, and to
select a template cohort from the participant's age (`cohort-auto`) for any template
with a cohort entity.

DWI data is still written **only** in ACPC space. Standard spaces produce transforms
and anatomical derivatives. QSIRecon consumes those transforms; resampling DWI into
standard space remains out of scope.

## Current state

`--anatomical-template` is declared with `choices=['MNI152NLin2009cAsym']`, so it has
exactly one legal value today. It does three jobs at once:

1. The ACPC alignment reference — `GetTemplate` (`interfaces/anatomical.py:209`)
   fetches the template and mask that the anatomical is rigidly aligned to.
2. The base for the DWI output grid — `init_output_grid_wf`
   (`workflows/anatomical/volume.py:1242`) autoboxes that template image, deobliques
   it, and resamples it to `--output-resolution`.
3. The nonlinear normalization target, which names the `from-`/`to-` transform
   derivatives.

`--infant` swaps it to `MNIInfant`, resolves a cohort from the participant's age via
`cohort_by_months` (`utils/bids.py:973`), and requires sessionwise processing.

Two findings from reading the code shaped this design:

- **`config.workflow.spaces` and `init_spaces()` are vestigial.** `init_spaces()`
  (`config.py:803`) builds a niworkflows `SpatialReferences` and assigns it to
  `workflow.spaces`, which no workflow reads. It is leftover fMRIPrep scaffolding.
  There is no interoperability to preserve, so it is deleted rather than adapted.
- **`init_output_grid_wf`'s `inputnode.input_image` is never connected.**
  `VoxelSizeChooser` (`interfaces/anatomical.py:69`) already implements `min`/`max`/
  `mean` anisotropic strategies over an input image's zooms, but that path is
  unreachable because `--output-resolution` is required. `res-nativemin` and
  `res-nativemax` map onto machinery that already exists; it needs a DWI image wired
  in and the ability to consider all runs.

niworkflows' own parser gets partway there: `Reference.from_string` passes `res-3mm`
through untouched and expands `MNI152NLin6Asym:res-01:res-3mm` into two references.
But it rejects `acpc` as a space identifier and rejects `cohort-auto`, because it
validates cohorts against `TF_LAYOUT.get_cohorts()`.

## Decisions

| Question | Decision |
|---|---|
| What anchors ACPC alignment? | Implicit, no flag. `MNIInfant`+resolved cohort if an infant template is requested (or `--infant` given), else `MNI152NLin2009cAsym`. |
| Is `--output-spaces` required? | Yes, and it must include at least one `acpc:res-*`. |
| What do standard spaces produce? | Transforms **and** resampled anatomicals (T1w/T2w preproc, brain mask, dseg). |
| Bare `res-2` on `acpc`? | Hard error. The `mm` suffix is required everywhere; one rule, no exceptions. |
| Old flags | Warn and forward for one release, then remove. |
| Multiple `acpc` resolutions | Allowed. Each produces its own grid and its own resampled DWI. |
| `res-` entity on ACPC DWI | Only when more than one ACPC resolution was requested. |
| The bundled SyN-SDC fieldmap atlas | Stays bundled (no TemplateFlow equivalent); add a NOTICE entry and track contributing it upstream separately. |

## Approach

A QSIPrep-owned `qsiprep/utils/spaces.py`, roughly 150 lines: a `SpaceSpec` dataclass
plus a parser we control end to end. Template names and cohorts validate against
TemplateFlow directly, which QSIPrep already imports, so `TF_LAYOUT.get_cohorts()`
gives the "any template with a cohort entity" rule with nothing hardcoded to
MNIInfant.

Two alternatives were considered and rejected. Extending niworkflows' `Reference`
would require appending `'acpc'` to the module-global `NONSTANDARD_REFERENCES` and
subclassing to relax cohort validation — mutating a library global that changes
validation for anything else importing niworkflows in the same process, in exchange
for an interoperability benefit that the grep shows does not exist. Vendoring
niworkflows' spaces module means maintaining roughly 400 lines of borrowed code for
the 40 actually needed.

## 1. Grammar

```
token   ::= space (":" key "-" value)*
space   ::= "acpc" | <TemplateFlow template name>
key     ::= "res" | "cohort"
```

`res-` values come in three families:

| Form | Meaning | Example |
|---|---|---|
| `nativemin` / `nativemax` | min/max zoom across the subject's DWI runs, made isotropic | `acpc:res-nativemax` |
| `<size>mm` | physical size; `p` is the decimal point, `x` separates axes (niworkflows#997) | `res-2mm`, `res-1p5mm`, `res-6x6x3mm` |
| bare label | TemplateFlow `res` entity, as in fMRIPrep | `res-01`, `res-2` |

`cohort-` values are a TemplateFlow cohort label or `auto`.

### Validation rules

1. At least one `acpc` token is required. `acpc` may appear more than once; each
   occurrence must carry exactly one `res-`.
2. On `acpc`, only `nativemin`, `nativemax`, or an **isotropic** `<N>mm` are legal.
   - `acpc:res-2` errors, naming `acpc:res-2mm`.
   - `acpc:res-2x2x3mm` errors. Reconstruction requires isotropic voxels — that is
     the entire reason `nativemin`/`nativemax` exist.
3. `res-native*` is rejected on standard spaces. "Native" means the input DWI's grid,
   which has no meaning in template space.
4. A template that TemplateFlow reports cohorts for must carry a `cohort-` spec.
   Generic, driven by `TF_LAYOUT.get_cohorts()`.
5. `cohort-auto` is legal only for templates present in `cohort_by_months`' table —
   currently `MNIInfant` and `UNCInfant`. For any other cohort template, parse-time
   error listing the valid cohort labels.
6. Repeated `res-` on a standard space expands to multiple outputs, per
   niworkflows#997: `MNI152NLin6Asym:res-01:res-2mm` yields two.
7. Duplicate tokens are de-duplicated.

### Storage

`config.workflow.output_spaces` holds a list of canonical token strings, keeping the
TOML round-trip through `config.to_filename`/`config.load` trivial and letting the
`isinstance(v, SpatialReferences)` special case at `config.py:273` go away. A cached
`parse_output_spaces()` returns `SpaceSpec` objects. `init_spaces()` and
`workflow.spaces` are deleted.

Two specs stay **symbolic** through parsing and config, because neither is knowable at
CLI time: `cohort-auto` (needs the participant's age) and `res-native*` (needs the DWI
headers).

## 2. CLI surface

`--output-spaces` takes `nargs='+'` with a custom argparse action that parses each
token immediately and reports failures through `parser.error`.

It cannot be `required=True` in argparse, because `--output-resolution 2` alone must
still work during the deprecation window. argparse marks it optional; `parse_args`
enforces "at least one `acpc:res-*`" *after* forwarding, so one error message covers
both spellings.

### Deprecation

Reusing the `DeprecatedStoreAction` / `DeprecatedForwardAction` machinery already in
`cli/parser.py` for `--longitudinal` and `--b0-to-t1w-transform`:

| Old | Forwards to |
|---|---|
| `--output-resolution 2` | `acpc:res-2mm` + the anchor template token |
| `--output-resolution 1.5` | `acpc:res-1p5mm` + the anchor template token |
| `--anatomical-template MNI152NLin2009cAsym` | appends that token |
| `--skip-anat-based-spatial-normalization` | drops standard spaces from the list |

Legacy forwarding must reproduce today's behavior exactly, and today every run
normalizes to the anatomical template. So `--output-resolution 2` alone forwards to
`acpc:res-2mm MNI152NLin2009cAsym` — or `acpc:res-2mm MNIInfant:cohort-auto` under
`--infant` — so existing invocations keep producing the transforms QSIRecon expects.

This requires `--anatomical-template` to switch from `default='MNI152NLin2009cAsym'`
to `default=SUPPRESS`, so "was it given?" is just `hasattr`. `cli/parser.py` already
uses that trick for `--b0-to-anat-transform` and documents why.

Passing any of these alongside `--output-spaces` is an error rather than a merge.
Silently combining two ways of saying the same thing is how users end up with a
resolution they did not ask for.

`--skip-anat-based-spatial-normalization` is fully subsumed: requesting no standard
space *is* skipping normalization. ACPC alignment is unaffected — it is a rigid
alignment to the anchor template and runs regardless.

### `--infant`

Not deprecated; it controls more than template choice (padding 4, T2w forcing, the
sessionwise requirement). It additionally appends `MNIInfant:cohort-auto` unless an
`MNIInfant` token is already present.

### The ACPC anchor

> anchor = `MNIInfant` + resolved cohort if an infant template is among the requested
> spaces (or `--infant` was given), else `MNI152NLin2009cAsym`.

Derived, with no flag behind it, and independent of list order. This reconciles
"the anchor is not user-selectable" with the requirement that `MNIInfant:cohort-auto`
take over `--infant`'s template-specification role.

## 3. Deferred resolution

### `cohort-auto` — workflow-build time, per subject

Resolved in `init_single_subject_wf`, extending what is already at
`workflows/base.py:207-226`. `parse_bids_for_age_months` runs **once** per subject and
feeds `cohort_by_months` for each `cohort-auto` space, so
`MNIInfant:cohort-auto UNCInfant:cohort-auto` does not re-parse anything. The result
is a concrete per-subject spec list, passed down the way `anatomical_template` is
today. A missing age stays a hard `RuntimeError`, with a message naming the space that
needed it. The existing single-session restriction is unchanged.

### `res-native*` — node run time

- `VoxelSizeChooser` gains an `InputMultiObject` of images and takes min/max across
  **all** of the subject's DWI runs, not just the first. The `min`/`max`/`mean` logic
  at `interfaces/anatomical.py:73-91` is otherwise unchanged.
- `subject_data['dwi']` is a plain file list available at build time, so it is set as
  a static input on `reference_grid_wf.inputs.inputnode.input_image` rather than
  connected.
- Under `--anat-only` there are no DWI files, but `dwi_sampling_grid` also has no
  consumer (it flows anat → `base.py` → `finalize.py:317` → `init_dwi_trans_wf` and is
  never datasunk). `input_image` falls back to the anatomical reference. No error is
  raised for a spec that cannot affect any output.

### Downstream consumers losing the config float

- `ChooseInterpolator` (`workflows/dwi/resampling.py:192`) currently reads
  `config.workflow.output_resolution`. It reads zooms off the `output_grid` image
  instead — already an inputnode field in that workflow. This removes a config
  dependency and makes the interpolator correct per-resolution automatically in the
  multi-grid case.
- The boilerplate at `resampling.py:135-138` interpolates `{vox}mm` at build time. For
  `native*` there is no number yet, so it renders from the spec label instead.

## 4. Workflow changes

**Output grid fan-out.** `init_output_grid_wf(spec)` is built once per `acpc` token,
named from its label (`output_grid_res2mm_wf`). The anat `outputnode.dwi_sampling_grid`
becomes a list in spec order. `finalize.py:317` selects by index via a
function-connection, since the specs are known at build time. The autobox of the anchor
template is identical across grids; only the final `afni.Resample` differs.

**DWI resampling fan-out.** One `init_dwi_trans_wf` and its derivative sinks per ACPC
spec. This is the expensive axis: N resolutions is roughly N× the per-volume
`ApplyTransforms` cost.

**Normalization fan-out.** `init_anat_normalization_wf` runs once per requested
standard space. N standard spaces means N nonlinear registrations — documented,
because `--output-spaces` makes it easy to type a list that triples runtime. ACPC
alignment is unaffected and stays a single rigid alignment to the anchor template,
whether or not the anchor's template appears in the requested list.

## 5. Derivatives naming

| Output | Pattern |
|---|---|
| DWI, single ACPC res | `space-ACPC_desc-preproc_dwi.nii.gz` — unchanged from today |
| DWI, multiple ACPC res | `space-ACPC_res-2mm_desc-preproc_dwi.nii.gz` |
| Anat in standard space | `space-MNIInfant_cohort-3[_res-1]_desc-preproc_T1w.nii.gz` |
| Transforms | `from-ACPC_to-MNIInfant+3_mode-image_xfm.h5` and inverse |

The `res-` entity on anatomical outputs appears **only when a `res-` spec was
given**. A bare `MNI152NLin6Asym` resamples onto the template's own `res-1` grid and
writes `space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz` with no `res-`, matching
fMRIPrep.

The `res-` label is the spec **as written** (`res-2mm`, `res-1p5mm`, `res-nativemax`),
not the resolved number. `DerivativesDataSink` needs the entity at workflow-build time,
and `nativemax` is not resolved until `VoxelSizeChooser` runs; a resolved-numeric label
would require injecting entities at runtime.

The `space-` / `cohort-` split follows the convention already in the repo:
`data/io_spec.json:9` defines a `cohort` entity and `_template_to_report_entities`
(`volume.py:1378`) already splits `MNIInfant+3` into `space=MNIInfant, cohort=3` for
reportlets. `from-`/`to-` keep `+cohort` inline because a transform label has nowhere
else to put it — matching nibabies.

**Sidecars.** Each ACPC DWI sidecar records the resolved voxel size. This is the only
place a `res-nativemax` run reports what `nativemax` actually turned out to be.

**Reports.** `SubjectSummary(template=...)` takes the list;
`_template_to_report_entities` generalizes to handle it. The summary reportlet names
every requested space.

## 6. TemplateFlow sourcing and orientation

QSIPrep's preferred orientation is **LPS+**. `interfaces/niworkflows.py:47` states that
"qsiprep forces LPS", and the existing reorientation nodes use
`afni.Resample(orientation='RAI')`, which is AFNI's spelling of LPS+.

### Templates are already TemplateFlow-sourced

`GetTemplate` (`interfaces/anatomical.py:214`) already calls `templateflow.api.get` for
both the template image and its brain mask. No anatomical template is bundled. The only
NIfTI in `qsiprep/data` is `mni_lps_fmap_atlas.nii.gz`, which is not a template:

- 193x229x193, 1 mm, LPS+ (the MNI152NLin2009cAsym res-01 grid)
- value range -2.8 to 10.65 -- negative values, so a fieldmap (the Treiber et al.
  average field behind fMRIPrep's fieldmap-less approach), not an intensity image
- used only by `syn_sdc_wf`, where `ThreshAndBin` thresholds it at 2 and binarizes it
  into a registration mask

It has **no TemplateFlow equivalent**. The nearest candidate,
`tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref`, correlates 0.66 with it after
resampling and ranges 0-1352 with no negatives -- an average EPI reference, a different
image. `git log --follow` shows the file landed in 2018 with no NOTICE entry.

**Decision:** keep it bundled, add the missing provenance and attribution entry to
`qsiprep/data/NOTICE`, and open a separate issue to contribute it to TemplateFlow.
It is a fieldmap-less SDC registration mask and is unrelated to output spaces, so it
does not block this work.

### Two changes the multi-space design does need

**`GetTemplate` is res- and cohort-blind.** It hardcodes `resolution='1'`. It takes the
resolved `SpaceSpec` instead, so a `res-` spec on a standard space selects the
corresponding TemplateFlow grid, and the cohort comes from the spec rather than from
splitting a `template+cohort` string.

**LPS reorientation is hardcoded to the anchor.** `reorient_tpl_brain_to_lps` and
`reorient_tpl_mask_to_lps` (`workflows/anatomical/volume.py:200-206`) exist only for the
ACPC anchor pair. They become a small reusable sub-workflow applied to every requested
standard space, so every anatomical output QSIPrep writes is LPS+, matching the rest of
its derivatives. Standard-space outputs are therefore written in LPS+ rather than
TemplateFlow's native RAS+.

## 7. Errors

All of the following fail at CLI parse time via `parser.error`, so nothing dies an hour
into a run:

- unknown space name
- `acpc:res-2` — points at `acpc:res-2mm`
- anisotropic or bare-label `res-` on `acpc`
- `res-native*` on a standard space
- a cohort template with no `cohort-` spec
- `cohort-auto` on a template with no age table — lists the valid cohorts
- a deprecated flag combined with `--output-spaces`
- no `acpc` token at all

Only two errors cannot be caught there, both raised in `init_single_subject_wf`: age
not found for a `cohort-auto` space, and the existing multi-session restriction. Both
name the space that triggered them.

## 8. Testing

- `qsiprep/tests/test_utils_spaces.py` — a parametrized accept/reject table over the
  grammar. Pure functions; no BIDS tree, no TemplateFlow downloads beyond the cohort
  query.
- Cohort resolution — parametrized over ages against the `cohort_by_months` table,
  including boundary months and the "age exceeds all cohorts" case.
- Deprecation forwarding — `--output-resolution 1.5` produces `acpc:res-1p5mm`;
  combining old and new errors; `--skip-anat-based-spatial-normalization` drops
  standard spaces.
- **Regression guard** — a single-`acpc` run builds exactly the derivative filenames it
  builds today. This is what protects QSIRecon, and it should land *before* any of the
  fan-out work.
- Workflow-build tests — N grids and N `dwi_trans_wf`s for N ACPC specs; N
  normalization workflows for N standard spaces.

## 9. Migration surface

27 files reference `output_resolution` or `--output-resolution`:

- **12 `.circleci/*.sh`** integration scripts. Forwarding keeps them green, so most
  convert to `--output-spaces`, with one or two deliberately left on the old flags as
  live deprecation coverage.
- **5 docs files** — `quickstart.rst` (6 occurrences), `installation.rst` (4),
  `preprocessing.rst` (3), `usage.rst`, `changes.md` — plus a new migration table in
  `usage.rst`.
- `qsiprep/data/tests/config.toml`.
- `qsiprep/tests/test_cli.py` (21 occurrences), `test_cli_run.py`,
  `test_gpu_cpu_ratio.py`, `test_utils_misc.py`.
- Source: `cli/parser.py`, `config.py`, `interfaces/images.py`,
  `workflows/anatomical/volume.py`, `workflows/dwi/finalize.py`,
  `workflows/dwi/resampling.py`.

Additionally, outside the `output_resolution` grep: `interfaces/anatomical.py`
(`GetTemplate`, `VoxelSizeChooser`) and `qsiprep/data/NOTICE`.

## Out of scope

- Resampling DWI into standard spaces. QSIPrep writes preprocessed DWI in ACPC space
  only.
- QSIRecon's consumption of the new transforms and multi-resolution ACPC outputs.
- Surface spaces (`fsaverage`, `fsLR`). QSIPrep has no surface outputs.
- Contributing `mni_lps_fmap_atlas.nii.gz` to TemplateFlow. Tracked separately; the
  file stays bundled, with a NOTICE entry added here.
