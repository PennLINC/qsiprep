# Scan grouping & field map selection refactor — design

**Date:** 2026-06-10
**Status:** Approved (design); implementation plan pending
**Scope:** `qsiprep/utils/grouping.py`, `qsiprep/workflows/base.py` (adapter), new `qsiprep/utils/fieldmaps.py`, tests
**Branch context:** builds on `group-dwi-scans` (PR #967), which introduced the current four-group `group_dwi_scans` interface.

## Problem

`qsiprep/utils/grouping.py` is a single ~1185-line module that determines, per
subject/session:

1. **Distortion groups** — which DWI files share distortions and are merged
   before denoising.
2. **Field map estimation groups** — which files are combined to estimate each
   field map.
3. **Field map application groups** — which distortion groups each field map
   corrects.
4. **Concatenation groups** — which preprocessed distortion groups are merged in
   the outputs.

The code is hard to follow and modify for several concrete reasons:

- **Per-suffix field map knowledge is scattered** across `FMAP_PRIORITY`,
  `classify_fmap_files`, `_find_sibling_fmap_file`, `get_highest_priority_fieldmap`,
  and a suffix dispatch in `base._build_outputs_to_files`. A recent bug (GRE
  phasediff field maps mislabeled as pepolar `epi`, crashing TOPUP) was a direct
  consequence of this duplication.
- **A post-hoc "refine/remap" dance** (`refine_distortion_groups`,
  `_remap_groups_after_refinement`, `_remap_group_members_with_new_dg_ids`)
  exists only because distortion groups are built *before* estimators and must
  then be split and re-keyed when an estimator disagrees.
- **Stringly-typed groups.** Everything is `dict[str, list[str]]` whose members
  are "either a distortion-group id or a file path," disambiguated at use sites
  with `m in distortion_groups`.
- **Repetitive metadata access** via `_get_metadata_field`, `_dg_pe_direction`,
  `_get_distortion_group_signature`, `_get_member_session`, etc., each
  re-querying the BIDS layout.

## Goals

- Make the grouping and field map selection code simpler and more interpretable.
- **Preserve behavior exactly** — the four-dict output of `group_dwi_scans` and
  the legacy `fieldmap_info` consumed by downstream workflows must not change.
- Draw inspiration from `sdcflows` (`sdcflows/fieldmaps.py`,
  `sdcflows/utils/wrangler.py`), mirroring its names and structure so a future
  offload of distortion correction to SDCFlows becomes feasible.

## Non-goals

- Actually offloading distortion correction to SDCFlows. SDCFlows compatibility
  is a *consideration*, not a requirement. (Recorded as a future end-state.)
- Changing any user-facing behavior, CLI parameters, or workflow outputs.
- Refactoring the SDC *workflows* (`qsiprep/workflows/fieldmap/`,
  `qsiprep/workflows/dwi/`). This work stops at producing `fieldmap_info`.

## Behavior contract

`group_dwi_scans` has exactly one caller (`workflows/base.py`), and its four
dicts feed only `_build_outputs_to_files`, which serializes them to the legacy
`fieldmap_info` shape (`{'suffix': ..., <type-specific keys>}`) that downstream
workflows depend on. The contract to preserve is therefore:

1. The four-dict return of `group_dwi_scans` for any given input, including the
   exact `auto_xxxxx` key scheme, group ordering, and generalized BIDS names.
2. The `fieldmap_info` produced per group by `_build_outputs_to_files`.

The baseline is **current `group-dwi-scans` HEAD**, i.e. the intended new
behavior from PR #967 — *not* `main`, whose grouping semantics #967 deliberately
changed.

## Design decisions (settled during brainstorming)

- **Value-object redesign**, with the four-dict / `fieldmap_info` outputs as thin
  serializers over the value objects.
- **Golden/characterization tests first**, frozen against current HEAD.
- **Mirror SDCFlows' names and structure** (`FieldmapFile`, `FieldmapEstimation`,
  `EstimatorType`, a `find_estimators`-shaped discovery), adapted for QSIPrep's
  distortion/concatenation-group concepts (which have no SDCFlows analog).
- **Two-layer module split** (chosen over enrich-in-place or direct dependency).

## Architecture

### New file: `qsiprep/utils/fieldmaps.py`

Mirrors `sdcflows/fieldmaps.py` in names and structure. SDCFlows uses
`@attr.s(slots=True)`, but `attrs` is only a *transitive* dependency in QSIPrep,
not a declared one. To avoid taking on a new direct dependency, the value objects
use stdlib `@dataclass(slots=True)` (Python 3.10+, which QSIPrep already
requires). This is an implementation detail that does not change the mirrored
names or structure; switching to `attrs` later (e.g. for closer SDCFlows parity)
is a localized change.

```
class EstimatorType(Enum):
    UNKNOWN, PEPOLAR, PHASEDIFF, MAPPED, ANAT

MODALITIES: dict[str, EstimatorType | None]
    # dwi, epi -> PEPOLAR; phasediff, phase1, phase2 -> PHASEDIFF;
    # fieldmap -> MAPPED; T1w, T2w -> ANAT; magnitude* -> None

@dataclass(slots=True)
class FieldmapFile:
    """One file plus metadata, read once."""
    path, entities, suffix, metadata, bids_root
    # validates the metadata each suffix requires
    # discovers sibling magnitude / phase2 files by filename

@dataclass(slots=True)
class FieldmapEstimation:
    """A set of FieldmapFiles that defines one field map."""
    sources, method, bids_id, sanitized_id
    # infers EstimatorType from sources
    # owns the B0FieldIdentifier, or an auto_xxxxx fallback
    def to_fieldmap_info(self) -> dict:
        # -> legacy {'suffix': ..., 'epi'/'phasediff'/'fieldmap'/...: ...}

class EstimatorRegistry:
    """Estimator + intent lookup. Instance-scoped (see divergence note)."""
```

This single layer replaces `classify_fmap_files`, `_find_sibling_fmap_file`,
`get_highest_priority_fieldmap`, and `FMAP_PRIORITY`.

**Intentional divergence from SDCFlows:** SDCFlows keeps `_estimators` and
`_intents` as *module-global* registries requiring `clear_registry()` between
subjects — a known footgun. We make the registry an **instance** created per
`group_dwi_scans` call. Same structure and names, no global mutable state.

### Slimmed `qsiprep/utils/grouping.py`

Keeps only the QSIPrep-specific concepts (SDCFlows never merges multiple DWI
runs):

```
@dataclass(slots=True)
class DwiSeries:
    """One DWI file plus distortion-relevant metadata, read once."""
    path, session, pe_dir, shim, trt, b0_identifier, b0_source, multipart_id

def build_distortion_groups(series, ...) -> ...
def build_concatenation_groups(series, ...) -> ...
def find_estimators(series, fmap_files, ...) -> list[FieldmapEstimation]
    # priority: B0FieldIdentifier -> IntendedFor -> heuristic
    # honors estimate_per_axis (DRBUDDI per-axis pairing)
def group_dwi_scans(...) -> (distortion, estimation, application, concatenation)
    # thin orchestrator; serializes value objects to the four dicts
```

This replaces `_get_metadata_field`, `_dg_pe_direction`,
`_get_distortion_group_signature`, `_get_member_session`, and the `_build_*`
helper sprawl.

### Adapter: `workflows/base.py`

`_build_outputs_to_files` shrinks to read `FieldmapEstimation.to_fieldmap_info()`
instead of re-deriving suffix typing. `group_dwi_scans` keeps its signature and
four-dict return.

## Discovery flow (eliminates refine/remap)

New ordering:

1. Build `DwiSeries` for every DWI file.
2. Build **preliminary** distortion groups by *physical* signature
   `(session, pe_dir, shim, trt, b0_identifier)` — identical to today's
   distortion groups.
3. `find_estimators(...)` assigns each series to a `FieldmapEstimation`,
   recording the estimator id per series. Priority and `estimate_per_axis`
   behavior unchanged. The heuristic pairs preliminary groups by PE direction.
4. **Final** distortion groups = preliminary signature **+ estimator id**. This
   is the single place a group ever splits; final keys are generated once via
   `_get_unique_concatenated_bids_names`, so nothing is remapped afterward.
5. Concatenation groups from final distortion groups (MultipartID / session /
   separate-all-dwis).
6. Validate cross-group consistency.
7. Serialize to the four dicts and per-group `fieldmap_info`.

Today steps 2–4 are spread across build-distortion → build-estimation →
`refine_distortion_groups` → `_remap_groups_after_refinement`. Refine/remap
exists only because a group built in step 2 can later be found to span two
estimators (when `IntendedFor` / `B0FieldSource` splits files sharing a physical
signature). Folding the estimator id into the final signature handles that by
construction, so **`refine_distortion_groups`,
`_remap_groups_after_refinement`, and `_remap_group_members_with_new_dg_ids` are
deleted.**

## Serialization

`group_dwi_scans` returns today's four dicts, byte-identical:

- `distortion_groups`: `name -> [files]`
- `fmap_estimation_groups`: `bids_id -> [dg_ids and/or fmap paths]` — a
  `FieldmapEstimation` maps its DWI sources back to dg-ids and keeps fmap paths
  as paths.
- `fmap_application_groups`: `bids_id -> [dg_ids]`
- `concatenation_groups`: `name -> [dg_ids]`
- per-group `fieldmap_info` via `FieldmapEstimation.to_fieldmap_info()`.

**Auto-id & ordering:** the `auto_xxxxx` numbering, group ordering, and
generalized BIDS names must come out identical. The serializer reproduces today's
exact scheme; golden tests are the arbiter. If the value objects' natural order
differs, we adjust the serializer — never the snapshots.

## Error conditions (preserved, relocated)

- shim / `TotalReadoutTime` incompatibility within a metadata-defined group →
  `FieldmapEstimation` validation.
- `B0FieldIdentifier` spanning multiple PE axes under `estimate_per_axis` →
  `FieldmapEstimation` validation.
- estimation group spanning multiple concatenation groups, and a distortion group
  in multiple concatenation groups → grouping-level `validate_group_consistency`
  (kept).

## Behavior-preservation strategy

A new `test_grouping_golden.py`, parametrized over all 23 existing `dset_*`
skeleton fixtures plus a few additions for gaps (multi-session,
`B0FieldSource`-only, mixed EPI+GRE in one `fmap/`). For each fixture × relevant
flag combination (`combine_scans`, `ignore_fieldmaps`, `estimate_per_axis`) it
snapshots:

- the full four-dict return of `group_dwi_scans`, and
- the `fieldmap_info` for every group from `_build_outputs_to_files`.

Snapshots are captured from current HEAD and frozen as the contract. Expected
values are serialized human-readably (sorted, path-normalized), not opaque
pickles, so a regression diff is legible.

## Phasing

Each phase ends green; nothing half-migrated is shippable.

1. **Golden tests** — add against HEAD, all passing. No production changes.
2. **`fieldmaps.py`** — introduce `EstimatorType`, `FieldmapFile`,
   `FieldmapEstimation`, registry, with unit tests. Not yet wired in.
3. **`DwiSeries` + discovery** — add to `grouping.py`, unit-tested, not yet wired
   into `group_dwi_scans`.
4. **Swap `group_dwi_scans` internals** to the new ordering; delete refine/remap
   and superseded helpers. Golden tests stay green.
5. **Shrink the `base.py` adapter** to use `FieldmapEstimation.to_fieldmap_info()`.
   Golden tests green.
6. **Port helper-level tests** — rewrite by-name internal-helper tests onto the
   new API; delete tests for deleted helpers (coverage now in golden +
   value-object tests).

## Delivery

A **separate PR** stacked on `group-dwi-scans` (or on `main` after #967 merges),
not folded into #967. #967 is in review as a focused TOPUP-grouping bugfix; a
~1000-line refactor would bury that review. The golden tests are the bridge —
they encode #967's behavior as the contract the refactor preserves.

## Future SDCFlows offload (recorded, out of scope)

After this lands, `qsiprep/utils/fieldmaps.py` mirrors `sdcflows.fieldmaps`
closely enough that a future offload becomes "add the dependency, delete our
copy, delegate" rather than a rewrite. The instance-scoped registry is the one
intentional divergence to reconcile at that time.
