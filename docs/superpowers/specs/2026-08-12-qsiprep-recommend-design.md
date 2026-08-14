# qsiprep-recommend design

Date: 2026-08-12
Revised: 2026-08-14, for PRs #1071, #1077, #1080, #1083, #1085, #1087, #1088
Issue: https://github.com/PennLINC/qsiprep/issues/1081
Branch: `recommender-cli`

## Problem

TORTOISE support added many new ways to configure a QSIPrep run. Users cannot
easily tell which flags their data call for, even though the documentation
already states the answers (for example, `--hmc-model tortoise` for
non-shelled CS-DSI data). `qsiprep-recommend` reads a BIDS dataset and prints
the flags the documentation recommends for that data, with the reasoning.

## Scope

**In scope (v1):** recommendations derivable from JSON sidecars, `.bval`/`.bvec`
files, NIfTI headers, and `participants.tsv`/`sessions.tsv`.

**Out of scope (v1):** image-based probes, including running N4 on anatomicals
and b=0 images to decide on B1 bias correction (requested in the issue
comments). The architecture keeps `probe.py` as the single ingress module so an
opt-in image-probing step can be added later without restructuring.

**Out of scope entirely:** validating a user-supplied command line, and any
recommendation the documentation does not support.

## Architecture

New package `qsiprep/recommend/`, alongside `interfaces/`, `workflows/`, and
`utils/`, with a thin CLI entry point in `qsiprep/cli/recommend.py`.

```
qsiprep/recommend/
  probe.py      BIDS  -> SubjectFacts             (only pybids-aware module)
  profiles.py   facts -> AcquisitionProfile[]     (dedupe subjects by signature)
  rules.py      facts -> Recommendation[]         (pure functions)
  report.py     profiles + recs -> text           (rendering only)
qsiprep/cli/recommend.py            argparse + orchestration
docs/sphinxext/recommend_rules.py   directive rendering rules.py into the docs
```

Data flow is a straight pipeline with no back-edges.

### Ingress (`probe.py`)

Reuses QSIPrep's own ingress rather than reimplementing it. One `BIDSLayout` is
built for the dataset, then per subject: `qsiprep.utils.bids.collect_data()`
followed by `qsiprep.utils.grouping.group_dwi_scans()`. Both couple to
`qsiprep.config` only for logging, and `collect_data` accepts a pre-built
`BIDSLayout`, so neither needs workflow configuration state.

Reuse is the point: when the report says three runs will be concatenated, that
is the grouping QSIPrep will actually compute, not a lookalike. The cost is
pybids indexing time on large datasets, bounded by `--participant-label`.

Sidecar metadata and gradient tables are read through the layout-free helpers
`qsiprep.utils.bids.load_sidecar`, `find_bval`, and `find_bvec`, added in
\#1080, rather than through `layout.get_metadata` / `layout.get_bval`. Those
helpers implement the BIDS inheritance principle directly, so a
`sub-01_dwi.bval` that applies to both `part-mag` and `part-phase` images
resolves correctly — the arrangement `docs/usage.rst` used to tell users to
avoid, and no longer does.

The per-subject result is reduced to a `SubjectFacts` dataclass holding:

- sampling scheme classification and shell structure
- volume count, unique b-value count, voxel size (per DWI series)
- phase-encoding directions present; reverse-PE DWI series
- fieldmap types present (`epi`, `phasediff`/GRE, none)
- `ImageType`, `PartialFourier`, `MultipartID`, part-phase availability
- anatomical modalities present (T1w, T2w)
- age in months, when available
- the scan groups and concatenation grouping from `group_dwi_scans`

Everything downstream reads `SubjectFacts` and never touches pybids or the
filesystem.

### Profiles (`profiles.py`)

Each `SubjectFacts` is reduced to a signature tuple (scheme class, sorted PE
directions, rounded voxel size, fieldmap types, anatomical modalities, number
of DWI runs per session, number of sessions, partial-Fourier flag, NORM flag,
infant flag). Subjects sharing a
signature form one `AcquisitionProfile`. A homogeneous 500-subject dataset
renders one block listing 500 subjects; a mixed dataset renders one block per
acquisition.

### Rules (`rules.py`)

An ordered registry of pure functions from facts to
`Recommendation(flag, value, rationale, docs_anchor, severity)`, where
`severity` is one of `recommended`, `consider`, `note`, `warning`, or
`undetermined`. A `note` carries no flag; it explains behavior the user cannot
change or a choice the data cannot settle. Order is
explicit and declared, because some rules read earlier decisions: the
`--pepolar-method DRBUDDI` recommendation follows from the `--hmc-model
tortoise` decision, not directly from the data. Rules receive a `decisions`
mapping of what earlier rules concluded. No dependency-graph machinery.

Two ordering constraints matter: `pepolar_method` and `sdc_fallback` run after
`hmc_model`, and `anat_modality` runs after `infant`, because `--infant` forces
a T2w anatomical reference and the two flags must not both be emitted. (Note
that `docs/preprocessing.rst` advises `--anat-modality none` alongside
`--infant` to sidestep infant skull-stripping, while `--anat-modality`'s own
help text says T2w is forced under `--infant`. The parser wins: the recommender
stays silent about `--anat-modality` whenever it recommends `--infant`.)

Rules abstain when the metadata they need is absent; they never guess.

Because every rule is a pure function over a dataclass, the whole knowledge
base is unit-testable with no BIDS dataset on disk.

## Rule set

### Core

| Rule | Condition | Emits |
|---|---|---|
| `hmc_model` | after dropping b < `b0_threshold` (100 s/mm^2), remaining b-values cluster within a 100 s/mm^2 tolerance into <=4 clusters of >=6 directions each | nothing; the `eddy` default is correct |
| | otherwise, non-shelled (DSI / CS-DSI) | `--hmc-model tortoise`, noting `3dSHORE` as the motion-only alternative |
| `pepolar_method` | reverse-PE data present and a `tortoise` decision was made | `--pepolar-method DRBUDDI` (TOPUP is unsupported with that backend) |
| `sdc_fallback` | no fieldmap, no reverse-PE, `tortoise`, T2w present | note: TORTOISE `--epi T2Wreg` applies automatically |
| | no fieldmap, no reverse-PE, otherwise | `--use-syn-sdc warn` (consider) |
| `output_resolution` | always | largest input voxel dimension, rounded to 2 decimal places; warns on anisotropic voxels or mixed voxel sizes within a profile |
| `distortion_group_merge` | a complete series duplicated in opposing PE directions | `--distortion-group-merge average` (consider; `concat` is the alternative) |

\#1085 collapsed the three `diffprep_*` values into a single `--hmc-model
tortoise`; the correction mode moved into the `--diffprep-config` JSON as
`"correction_mode"` (`motion`, `quadratic` — the default — or `cubic`). The
`hmc_model` rule therefore emits one value and mentions `--diffprep-config` in
its rationale rather than choosing between three flag values. `hmc_model` is
also the only rule affected by the rename, because `decisions['--hmc-model']`
is now compared for equality with `'tortoise'` instead of tested with
`startswith('diffprep')`.

### Denoising

# TODO: Recommend dwidenoise or dwidenoise2 if phase data are available.

| Rule | Condition | Emits |
|---|---|---|
| `unringing_method` | `PartialFourier` < 1 | `--unringing-method rpg` |
| | `PartialFourier` == 1 | `--unringing-method mrdegibbs` (consider) |
| `b1_biascorrect_stage` | DWI `ImageType` contains `NORM` | `--b1-biascorrect-stage none` |
| `denoise_volume_count` | fewer than 30 volumes in the denoising unit | warning that MP-PCA has little data to work with |

`--denoise-method` is no longer an argparse `choices` list: since #1071 it is a
free-form spec string parsed by `qsiprep.utils.misc.parse_denoise_method`
(`dwidenoise`, `dwidenoise2`, `patch2self`, `none`, with
`dwidenoise2;name:value;...` parameters). No rule emits it, but it is the
reason the value-level invariant test below skips any action whose `choices`
is `None` rather than assuming every flag has an enumerated set of values.

### Anatomical and cohort

| Rule | Condition | Emits |
|---|---|---|
| `anat_modality` | no T1w, T2w present, and `--infant` was not recommended | `--anat-modality T2w` |
| | neither present | `--anat-modality none` |
| `infant` | age <= 60 months, read via `parse_bids_for_age_months` from `participants.tsv` or `sessions.tsv` (an `MNIInfant` cohort exists up to 60 months) | `--infant`, naming the cohort `cohort_by_months` would select |
| `anatomical_reference` | more than one session per subject | note: `sessionwise` vs `first-lex` is a study-design decision, and it also determines where reports land unless `--report-output-level` says otherwise |

### Grouping

| Rule | Condition | Emits |
|---|---|---|
| `concatenation_preview` | any group holds more than one run | note showing the groups `group_dwi_scans` computed, so `--separate-all-dwis` is an informed choice |
| `denoise_after_combining` | same | consider, with the documentation's caveat that evidence is thin |
| `multipart_id_missing` | multiple runs and no `MultipartID` | warning to verify the grouping (`usage.rst`) |

### Deliberate omissions

- **No phase/complex to `dwidenoise2` rule.** #1071 shipped `dwidenoise2` and
  its parameter spec (`dwidenoise2;demodulate:linear;...`), but no
  documentation recommends it for any data condition. Inventing a rule would
  violate the premise that recommendations match the documentation. This is the
  most likely rule to add once the docs take a position, and `SubjectFacts`
  already carries `has_phase_data` for it.
- **No `--anat-biascorrect auto` on NORM anatomicals.** The parser's help states
  that scanner-side normalization "does not remove the need for it", so
  recommending `auto` would contradict QSIPrep itself.
- **No `--report-output-level` rule.** #1087 added the flag, but its `auto`
  default already follows `--subject-anatomical-reference`, so there is nothing
  the data can add. The multi-session note mentions it instead.
- **Deprecated flags are never emitted.** #1083 turned the parser's
  `deprecations` map into `{option string: (removal version, what happens
  instead)}` and gave every deprecated option a `Deprecated*Action`, so the
  registry test can detect them by action class. This covers `--dwi-only`
  (the rule emits `--anat-modality none` instead), `--longitudinal`,
  `--dwi-no-biascorr`, `--prefer-dedicated-fmaps`, `--b0-motion-corr-to`, and
  `--b0-to-t1w-transform` (renamed to `--b0-to-anat-transform`).

## Documentation synchronization

`rules.py` is the single source of truth. Each rule carries its condition
description, recommended flags, and rationale text. A Sphinx directive,
`qsiprep-recommendation-rules`, implemented in
`docs/sphinxext/recommend_rules.py`, imports the registry and renders it as a
list-table, so the documentation and the CLI cannot disagree. Surrounding prose
stays hand-written.

A new page `docs/recommend.rst` joins the toctree, containing a short
introduction, the CLI reference auto-rendered by `sphinxarg.ext`, and the rules
table. `quickstart.rst` and `preprocessing.rst` each gain a one-line pointer.

`docs/preprocessing.rst:200` currently tells users to pass `--combine-all-dwis`,
which is no longer a flag; the current flag is `--separate-all-dwis`, off by
default. This fix rides along with the change.

## CLI

```
qsiprep-recommend BIDS_DIR [--output-dir OUT] [--participant-label ...]
                  [--session-id ...] [--bids-filter-file F]
                  [--skip-bids-validation] [--bids-database-dir D]
                  [-v] [--version]
```

Only `BIDS_DIR` is required. `--output-dir` exists so the printed command is
directly runnable; without it, the command carries a literal
`/path/to/outputs` placeholder. `--bids-database-dir` points at a pybids
database to reuse or create, so repeated runs on a large dataset skip
re-indexing. The filtering flags mirror QSIPrep's own spellings.

## Report format

Plain ASCII wrapped at 79 columns, no color and no box-drawing characters, so
it survives being pasted into an issue or a cluster log. One block per profile.

```
qsiprep-recommend 1.1.0 -- /data/HBN
Indexed 43 subjects; 2 acquisition profiles.

=== Profile 1 of 2 -- 41 subjects =============================
Subjects: sub-01, sub-02, sub-03, sub-04, sub-05 (+36 more; -v for all)

Detected
  Sampling scheme   non-shelled, 55 volumes, 52 unique b-values
  Voxel size        2.0 x 2.0 x 2.0 mm
  DWI runs          1 run, 1 session
  Fieldmaps         reverse-PE epi (dir-PA)
  Anatomicals       T1w, T2w
  Partial Fourier   0.75

Recommended
  --hmc-model tortoise
      Non-shelled scheme; FSL eddy requires shells, while TORTOISE DIFFPREP
      fits a signal model over arbitrary q-space.  [quickstart: hmc model]
  --pepolar-method DRBUDDI
      TOPUP is not supported with the tortoise backend.  [quickstart]

Consider
  --unringing-method rpg
      PartialFourier is 0.75; rpg is suggested for partial Fourier data.

Warnings
  3 runs will be concatenated but no MultipartID is set; check that the
  grouping matches your intent.  [usage: MultipartID]

Not determined
  --b1-biascorrect-stage: no ImageType field in the DWI sidecars, so
      prescan normalization could not be detected.

qsiprep /data/HBN /path/to/outputs participant \
  --output-resolution 2.0 \
  --hmc-model tortoise \
  --pepolar-method DRBUDDI \
  --unringing-method rpg \
  --participant-label sub-01 sub-02 sub-03 ...
```

Five buckets, in this order:

1. **Recommended** — the documentation says to do this.
2. **Consider** — a real choice the data makes relevant, which the
   documentation does not settle.
3. **Notes** — behavior the user cannot change, or a decision the data cannot
   settle. Carries no flag.
4. **Warnings** — something looks wrong.
5. **Not determined** — a rule abstained, and why.

The last bucket is what stops silence from reading as endorsement, and it
doubles as a nudge to fix sidecar metadata. Empty buckets are omitted.

A final `Skipped` section, outside the per-profile blocks, lists any subjects
that were passed over and the reason (for example, no DWI data).

Both **Recommended** and **Consider** flags go into the printed command.
`--participant-label` is appended only when the dataset has more than one
profile, so the per-profile commands can be run side by side. Subject lists are
truncated to five names plus a count unless `-v` is given.

## Error handling

- A missing or unindexable `BIDS_DIR`, or a `--participant-label` matching no
  subjects, exits 1 with a plain message.
- A subject with no DWI data is skipped and listed in a `Skipped` section
  rather than crashing the run.
- Individual rule exceptions are caught per rule and converted into
  `Not determined` entries carrying the exception message, so one failing rule
  degrades the report instead of destroying it.
- pybids validation is on by default, disabled with `--skip-bids-validation`.
  This is pure-Python validation with no node binary, so it works outside the
  container.
- Successful analysis always exits 0. The tool is advisory; it does not fail a
  dataset.

## Testing

Three layers, cheapest first.

**`test_recommend_rules.py`** exercises every rule as a pure function over
hand-built `SubjectFacts`: a positive case, a negative case, and an abstain
case each. No filesystem, no pybids.

**`test_recommend_probe.py`** covers ingress against real trees. The existing
convention is `generate_bids_skeleton` with `skeleton_*.yml` files under
`qsiprep/tests/data/`, but skeletons alone are insufficient here: the probe
reads voxel sizes from NIfTI headers and shells from `.bval` files, and
skeleton files have no usable image content. This needs a fixture factory that
generates the skeleton, then overwrites each `_dwi.nii.gz` with a 4x4x4xN
nibabel image at the intended zooms and writes matching `.bval`/`.bvec` files.
That helper is the one piece of test infrastructure this feature adds, and it
is reusable beyond it.

Fixtures to cover: single-shell with a PA epi fieldmap; CS-DSI non-shelled;
HCP-style dual-PE multi-shell; no fieldmap; no T1w; multi-run with and without
`MultipartID`; a heterogeneous dataset that must split into two profiles; and,
since #1080, complex-valued DWI whose `.bval`/`.bvec` are inherited from a
`part`-less sibling rather than duplicated per `part`, which `usage.rst` now
presents as the correct layout.

**`test_recommend_cli.py`** runs the entry point end to end on a fixture and
asserts exit status 0 and — the key invariant — that the printed command parses
cleanly through QSIPrep's own `_build_parser()`.

A registry-wide check asserts that no rule can emit a flag absent from
`_build_parser()` or registered with one of the parser's deprecation actions
(`DeprecatedAction`, `DeprecatedForwardAction`, `DeprecatedStoreAction`), so a
flag rename or a new deprecation breaks CI rather than quietly producing bad
advice. That check is what would have caught the `diffprep_* -> tortoise`
rename in #1085 had the recommender already existed, and it is why
`--hmc-model`'s *value* deserves a companion assertion against the action's
`choices`.

## Packaging

One line in `pyproject.toml`:

```toml
[project.scripts]
qsiprep-recommend = "qsiprep.cli.recommend:main"
```

No new dependencies: pybids, nibabel, and numpy are already required.
