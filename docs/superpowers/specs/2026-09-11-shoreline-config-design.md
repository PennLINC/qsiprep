# `--shoreline-config` design

Issue: [PennLINC/qsiprep#1132](https://github.com/PennLINC/qsiprep/issues/1132)
Branch: `shoreline-config`

## Goal

Replace `--shoreline-model`, `--shoreline-iters` and `--hmc-transform` with a single
`--shoreline-config` JSON file, following the pattern of `--eddy-config` and
`--diffprep-config`.

## Decisions

| Question | Decision |
|---|---|
| Retiring the old flags | Remove all three outright. `--shoreline-model` was never released. `--shoreline-iters` and `--hmc-transform` shipped in 26.0.0 but are removed without a deprecation cycle. SHORELine itself is proposed for removal in 27.0.0 (#1086). |
| JSON schema | Short keys `model`, `iters`, `transform`, all optional. Validation is strict: unknown keys and bad values are errors. |
| Non-SHORELine runs | `--shoreline-config` without `--hmc-method shoreline` is a parse error. SHORELine settings are resolved only for SHORELine runs. The HTML summary drops the "HMC Transform" line for eddy and TORTOISE. |
| QSIPlan coupling | This PR changes qsiprep only: `--shoreline-model` is exempted in the plan-CLI conformance tests and a QSIPlan issue is opened. **Release gate:** the QSIPlan fix must be released, and qsiprep's `qsiplan` pin bumped, before the next qsiprep release (section 5). |

## Background

- `hmc_transform` only matters for SHORELine. `init_b0_hmc_wf` falls back to it only when
  called without `transform=`. The anatomical merge (`workflows/anatomical/volume.py`) and
  the intramodal template (`workflows/dwi/intramodal_template.py`) both pass their own
  transform. Its other use is the diffusion summary, which currently prints
  "HMC Transform: Affine" even for eddy and TORTOISE runs.
- The SHORELine model must be known at parse time. `DeprecationForwardingParser.parse_known_args`
  derives the legacy `hmc_model` from it, and `utils/plan.py` and QSIPlan's
  `selection_from_namespace` read the `shoreline_model` dest.
- `config.load` skips keys a config class doesn't have, so old `config.toml` files stay
  loadable either way.
- `parse_args` loads `--config-file` after parsing, then overlays the parsed namespace with
  `config.from_dict(vars(opts))` (`cli/parser.py` around lines 1147 and 1175).
  `_Config.load` skips `None` values (`config.py:226`). Values the parser resolves already
  win over a loaded file: `hmc_method` always resolves to a non-`None` value, so a config
  file never restores it.
- **Existing bug:** `--shoreline-iters` never reaches the workflow.
  `workflows/dwi/hmc_sdc.py` calls `init_dwi_hmc_wf(source_file=source_file)`, so
  `num_model_iterations` is always its default of 2. This was already true in 26.0.0.
  The value is read only by a validity check in `base.py` and by `derivatives.py`, which
  uses it to decide whether to write `hmcOptimization`.

## Approach

The parser resolves the config once. It loads and validates the JSON, then writes three
resolved internal values onto the namespace: `shoreline_model`, `shoreline_iters` and
`hmc_transform`. These are derived config fields with no flags, like `hmc_model` today.
Consumers keep reading the same names.

Rejected alternatives:

- **Path only, like `--diffprep-config`:** the parser would still have to load the file to
  derive the model, so the JSON would be read in two places and seven consumers would change.
- **Nested dict in config:** every consumer would change, and it adds a nested TOML table
  that no other config field uses.

## 1. CLI and resolution

### Flags

- **Add** `--shoreline-config PATH` to the motion-correction group, with `type=IsFile`.
  Its help text lists the keys and links to the shipped default:
  https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/shoreline_params.json
- **Remove** `--shoreline-model`, `--shoreline-iters` and `--hmc-transform`. A command line
  that uses them fails with argparse's "unrecognized arguments" error.
- **Update help text:**
  - `--hmc-method` points to `--shoreline-config` instead of the removed flags;
  - the `--hmc-model` help and its `deprecations` entry say "with a `--shoreline-config`
    `model`" instead of "with `--shoreline-model`".

### Shipped default

New file `qsiprep/data/shoreline_params.json`:

```json
{
  "model": "3dshore",
  "iters": 2,
  "transform": "Affine"
}
```

### Loader

New `load_shoreline_config(path, model=None)` in `qsiprep/utils/misc.py`, next to
`validate_diffprep_config`.

- **Input:** `path` is a config file, or `None`, which uses the defaults. `model` is an
  optional override from the deprecated `--hmc-model` alias (`3dshore`, `tensor` or `none`).
- **Returns:** a dict with exactly `model`, `iters` and `transform`: the user file merged
  over the shipped defaults, then `model` applied if given.
- **Raises `ValueError`** with a message naming the file and the offending key or value if:
  - the file does not exist, or is not valid JSON;
  - the top level is not a JSON object;
  - there is a key other than `model`, `iters` or `transform`;
  - `model` is not `3dshore`, `tensor` or `none`;
  - `transform` is not `Affine` or `Rigid`;
  - `iters` is not an `int`, or is a `bool`;
  - `iters < 1`, unless the resolved `model` is `none`, which ignores `iters`;
  - `model` is given and the user file also sets `"model"`. The message is:
    `--hmc-model conflicts with "model" in --shoreline-config <path>`.

### Where resolution happens

In `DeprecationForwardingParser.parse_known_args`, where the method axes are already
normalized:

1. Resolve `hmc_method` as today, including the deprecated `--hmc-model` alias.
2. If `shoreline_config` is set and `hmc_method != 'shoreline'`, call
   `self.error('--shoreline-config requires --hmc-method shoreline')`.
3. If `hmc_method == 'shoreline'`, call
   `load_shoreline_config(namespace.shoreline_config, model=legacy_model)` and turn any
   `ValueError` into `self.error(str(exc))`. `legacy_model` is the lowercased SHORELine
   model from `--hmc-model 3dSHORE|tensor|none` (through the existing
   `hmc_model_to_shoreline_model` map), or `None` when `--hmc-method` was used. The loader
   handles the legacy/`"model"` conflict, so `iters` and `transform` can still come from
   the file alongside a legacy model.
   - Set `namespace.shoreline_model`, `namespace.shoreline_iters` and
     `namespace.hmc_transform` from the resolved dict.
4. Otherwise set all three to `None`.
5. Derive `hmc_model` from `shoreline_model` as today.

The existing SHORELine removal notice printed to stderr is unchanged.

### Config storage

- `config.workflow.shoreline_config = None`, documented as "Configuration JSON for SHORELine."
- Add `shoreline_config` to **`workflow._paths`**, making it `('gradient_file', 'shoreline_config')`.
  `IsFile` returns an absolute `Path`, and `config.py` warns that an unlisted `Path` reaches
  the workflow-building subprocess as the literal string `"PosixPath('...')"`. The
  existing `eddy_config` and `diffprep_config` entries in `execution._paths` do take
  effect: `_Config.load` checks `_paths` before `hasattr`, so loading them creates
  path-valued attributes on `execution` that duplicate the `workflow` fields. That
  duplication is not needed for `shoreline_config`, and this design does not touch the
  existing entries.
- Rewrite the docstrings of `shoreline_iters`, `shoreline_model` and `hmc_transform` to say
  they are derived from `--shoreline-config` and are `None` unless `hmc_method` is `shoreline`.

### Interaction with `--config-file`

Relying on `config.from_dict(vars(opts))` alone would leave the SHORELine block
inconsistent after a `--config-file` reload:

- a loaded file's `shoreline_config` would survive when the command line gives none,
  because `None` is skipped, while the parser's resolved defaults overwrite the file's
  `shoreline_model`, `shoreline_iters` and `hmc_transform`;
- an old eddy `config.toml` would keep its stale `hmc_transform = "Affine"` and
  `shoreline_iters = 2`.

So the command line is authoritative for the SHORELine block, as it already is for
`hmc_method`. Right after `config.from_dict(vars(opts), ...)`, `parse_args` assigns
`config.workflow.shoreline_config`, `shoreline_model`, `shoreline_iters` and
`hmc_transform` directly from `opts`, including `None` values. A reloaded run that wants
custom SHORELine settings passes `--shoreline-config` again, just as it must pass
`--hmc-method shoreline` again.

Resolving after the merge instead would mean moving the method-axis normalization and the
`hmc_model` derivation out of `parse_known_args`. The parser tests and QSIPlan's
`selection_from_namespace` rely on that normalization happening during parsing. It would
also give the SHORELine settings different reload behavior from `hmc_method`.

## 2. Consumers and the summary

| Location | Change |
|---|---|
| `workflows/dwi/hmc_sdc.py:140` | Pass `num_model_iterations=config.workflow.shoreline_iters` to `init_dwi_hmc_wf`. This is the bug fix; see "Effects of the iteration fix" below. |
| `workflows/dwi/base.py:257-262` | Delete the runtime check that `iters` is at least 1; the loader enforces it. |
| `workflows/dwi/base.py:~491` | Pass `hmc_transform` to `DiffusionSummary` only when the resolved HMC method is SHORELine: `hmc_tool == 'shoreline'`, where `hmc_tool = unit.run.hmc_stage.tool` is set at `base.py:256` in the same function. This is the resolved form of `hmc_method`, and it also covers configs that carry only the legacy `hmc_model`. Gating on the method rather than on the value means a stale value could never reach an eddy or TORTOISE summary. |
| `interfaces/reports.py` | Make `DiffusionSummaryInputSpec.hmc_transform` optional (no longer `mandatory=True`). Render the "HMC Transform" `<li>` only when it is defined. The other lines are unchanged. |
| `workflows/dwi/hmc.py:~389` (`init_b0_hmc_wf`, iterative branch) | The methods text says "iterations of `{config.workflow.hmc_model}` registrations"; change it to the local `transform` (Affine or Rigid), which is what actually selects the ANTs settings (`hmc.py:~271`). The `first` branch already does this. It also corrects the anatomical merge and intramodal template text, which pass their own `transform`. |
| `workflows/dwi/hmc.py:~717-722` (`init_dwi_model_hmc_wf`) | The methods text always says "reconstructing the others using 3dSHORE [@merlet3dshore]". Name the model from `config.workflow.hmc_model`, the value `SignalPrediction` actually reads (`hmc.py:~585`): "3dSHORE [@merlet3dshore]" for `3dSHORE` and "a tensor model" for `tensor`. `none` never reaches this function, because `init_dwi_hmc_wf` returns early (`hmc.py:~162`). |
| `workflows/dwi/hmc.py` (lines 357, 583), `workflows/dwi/derivatives.py:89`, `utils/plan.py` | No change. They read the resolved fields. |

### Effects of the iteration fix

This is an observable processing change for anyone whose `iters` isn't 2:

- **`iters: 1`:** one model iteration instead of two. `summarize_iterations`, the
  `desc-shorelineiters` figure and `optimization_data` are no longer produced
  (`hmc.py:~823`). `hmcOptimization.csv` was already suppressed for `iters <= 1` by
  `derivatives.py:89`.
- **`iters: 3` or more:** extra registration passes, so different transforms and corrected
  images, a different optimization CSV, and a different iteration count in the methods text.

The PR body states this.
| `qsiprep/data/tests/config.toml` | Remove `hmc_transform` and `shoreline_iters`. That fixture is an eddy run. |

Tests that assign `config.workflow.hmc_transform` or `shoreline_iters` directly keep working.

## 3. Documentation

- **`docs/usage.rst`:** generated by sphinx-argparse; no edit.
- **`docs/quickstart.rst`**, "Head motion correction method":
  - replace the `--shoreline-model` wording with `--shoreline-config` and a short JSON example;
  - update the parenthetical on `--hmc-model`;
  - rewrite the `--shoreline-model none` paragraph as `"model": "none"`.
- **`docs/preprocessing.rst`:**
  - add a "Configuring SHORELine" subsection (`.. _configure_shoreline:`) modeled on
    "Configuring `eddy`", with a table of the three keys (allowed values and defaults)
    and a link to `shoreline_params.json`. It sits in the SHORELine section and is
    referenced from the quickstart;
  - reword "If `"none"` is specified as the hmc_model" to refer to the `"model"` key.
- **`docs/notebooks/grouping_tutorial.md`:** update the `--hmc-method` bullet.
- **Out of scope:** the `.. workflow::` directives in `preprocessing.rst` that already pass
  kwargs the current functions don't accept. `docs/changes.md` is not edited; the PR body
  describes the breaking change.

## 4. Tests

### Unit tests (run locally: `micromamba run -n linc311 pytest ...`)

**`load_shoreline_config`**, in `qsiprep/tests/test_cli.py` next to the DIFFPREP validator tests:

- the shipped `shoreline_params.json` is valid and equals the `None` result;
- a partial file (`{"model": "tensor"}`) is merged with the defaults;
- errors are raised for:
  - a missing file;
  - a JSON list at the top level;
  - an unknown key (`{"iter": 3}`);
  - a bad `model`;
  - a bad `transform`;
  - `iters` set to `0`, `true` or `"2"`;
- `{"model": "none", "iters": 0}` is accepted.

**Parser**, in `qsiprep/tests/test_cli_run.py`:

- **Defaults:** with default eddy, `shoreline_model`, `shoreline_iters` and `hmc_transform`
  are all `None`.
- **Shipped defaults:** `--hmc-method shoreline` gives `3dshore`, `2` and `Affine`, with
  `hmc_model == '3dSHORE'`.
- **Custom config:** `--hmc-method shoreline --shoreline-config <tensor, 3, Rigid>` resolves
  to those values, with `hmc_model == 'tensor'`.
- **Wrong method:** `--shoreline-config` with eddy or tortoise gives `SystemExit` and
  "requires --hmc-method shoreline".
- **Bad file:** an invalid `--shoreline-config` gives `SystemExit` with the loader's message.
- **Legacy merge:** `--hmc-model tensor --shoreline-config {"iters": 3}` resolves to tensor
  with 3 iterations and warns about the deprecation.
- **Legacy conflict:** `--hmc-model tensor --shoreline-config {"model": "3dshore"}` gives
  `SystemExit`.
- **Removed flags:** each of `--shoreline-model`, `--shoreline-iters` and `--hmc-transform`
  gives `SystemExit` with "unrecognized arguments".
- Replace `test_shoreline_model_requires_shoreline_method`. Update `test_method_axis_defaults`
  and `test_hmc_model_alias_maps_and_warns` so they don't depend on the removed flags.

**Config round-trip:** after parsing with `--shoreline-config`, `config.from_dict` followed by
`config.to_filename` and `config.load` leaves `config.workflow.shoreline_config` as a real
path, and the TOML contains no `PosixPath(`.

**`--config-file` reloads**, through `qsiprep.cli.parser.parse_args` so the real
load-then-overlay order is exercised:

- an old eddy TOML with `hmc_transform = "Affine"` and `shoreline_iters = 2`, reloaded
  with the default eddy method, leaves all four SHORELine fields `None`;
- a TOML with `shoreline_config` set and `shoreline_model = "tensor"`, reloaded with
  `--hmc-method shoreline` and no `--shoreline-config`, gives `shoreline_config is None` and
  the shipped defaults (`3dshore`, `2`, `Affine`);
- the same TOML reloaded with `--shoreline-config <rigid.json>` gives that path and its
  values.

**QSIPlan conformance** (`qsiprep/tests/test_plan_cli_spec.py`):

- both conformance tests exempt `--shoreline-model` through a named constant, commented
  with the QSIPlan follow-up issue;
- a new test, parametrized over `3dshore`, `tensor` and `none`, parses
  `--hmc-method shoreline --shoreline-config {"model": <model>}`.
  `cli_spec.selection_from_namespace` must return a SHORELine selection with that
  `shoreline_model`. With `--hmc-method eddy`, the namespace has `shoreline_model is None`
  and the selection resolves to eddy.

**Iteration wiring**, in `qsiprep/tests/test_workflows_gradwarp.py`, reusing
`_cfg_for_shoreline`, `_shoreline_wf` and `_rpe_unit`. These already build
`init_qsiprep_hmcsdc_wf`, the function that had the bug. Build the workflow with
`config.workflow.shoreline_iters` set to `3`, then `1`, and check the leaf names from
`wf.list_node_names()` under `dwi_hmc_wf.dwi_model_hmc_wf.`:

- with `3`: names under `shoreline_iteration002.` exist, `summarize_iterations` exists,
  and the `dwi_model_hmc_wf` methods text says "A total of 3 iterations";
- with `1`: nothing under `shoreline_iteration001.`, no `summarize_iterations`, and the
  text says "A total of 1 iterations".

This is the regression test for the bug fix. On the current code both builds contain
`shoreline_iteration001` and `summarize_iterations`, and say 2 iterations.

**Methods text:**

- building `init_dwi_model_hmc_wf` with `hmc_model = 'tensor'` and
  `hmc_transform = 'Rigid'` gives a `__desc__` that mentions a tensor model and a Rigid
  transform, and does not contain "3dSHORE";
- with `3dshore`, it contains "3dSHORE [@merlet3dshore]";
- `init_b0_hmc_wf(align_to='iterative', transform='Rigid')` describes "Rigid registrations".

**Summary:**

- `DiffusionSummary` rendered without `hmc_transform` has no "HMC Transform" line; with
  `hmc_transform='Rigid'` it shows "HMC Transform: Rigid";
- at the workflow level, the `summary` node built for an eddy run leaves `hmc_transform`
  undefined even when `config.workflow.hmc_transform` has been set to a stale `'Affine'`.

### Integration tests (`qsiprep/tests/test_cli.py`; Docker/CI only, not run locally)

New files under `qsiprep/tests/data/`, loaded with `get_test_data_path()` the way
`eddy_config.json` is:

| File | Contents | Replaces |
|---|---|---|
| `shoreline_none_config.json` | `{"model": "none", "iters": 1}` | `--shoreline-model=none --shoreline-iters=1` (drbuddi_shoreline_epi); also `--shoreline-model=none` in the second SHORELine run (around line 645) |
| `shoreline_tensor_config.json` | `{"model": "tensor", "iters": 1}` | `--shoreline-model=tensor --shoreline-iters=1` (drbuddi_tensorline_epi) |
| `shoreline_rigid_config.json` | `{"transform": "Rigid", "iters": 1}` | `--hmc-transform=Rigid --shoreline-iters=1` (dscsdsi) |

With the iteration fix, these `iters: 1` runs now do one model iteration instead of two. No
expected-output list should change: none of their `*_outputs.txt` files lists a
`desc-shorelineiters` figure or `hmcOptimization.csv`. The only list containing
`hmcOptimization.csv`, `maternal_brain_project_outputs.txt`, comes from a run that uses the
default `iters` of 2. If a list does change, it is updated in the PR.

## 5. QSIPlan follow-up (outside this PR; gates the next qsiprep release)

### Why it matters

qsiprep shows QSIPlan's `MethodSelection.cli_phrase()` to users in two places:

- the workflow log, through `describe_processing` (`workflows/base.py:~379`, whose title
  QSIPlan builds in `qsiplan/report.py:~241`);
- the grouping report embedded at the top of every subject report
  (`workflows/base.py:~387-393`), whose plan panel renders it (`qsiplan/interactive.py:~765`).

For SHORELine runs with a `tensor` or `none` model, that phrase is
`--hmc-method shoreline --shoreline-model <model> --sdc-method drbuddi`, a flag this change
removes. `3dshore` runs are unaffected because `cli_phrase()` omits the default model.
Until QSIPlan is fixed, `main` shows that stale phrase for deprecated tensor and none runs.
No release may ship it.

### Issue

Draft an issue for PennLINC/QSIPlan and file it only after the user approves:

- remove `--shoreline-model` from `cli_spec.PLAN_OPTIONS`, or mark it as a flag qsiprep
  doesn't expose;
- make `MethodSelection.cli_phrase()` and the explorer suggest
  `--shoreline-config` with a `"model"` key instead of `--shoreline-model`;
- update QSIPlan's tests (`test_cli_spec.py`, `test_explorer.py`, `test_serve.py`).

### Release gate

Before the next qsiprep release:

1. the QSIPlan fix is released;
2. qsiprep's `qsiplan` pin in `pyproject.toml` is bumped to that release;
3. the `--shoreline-model` conformance exemption is removed.

The PR body records this gate and links the QSIPlan issue, and the exemption constant's
comment points to that issue.

## Out of scope

- Removing SHORELine (#1086).
- Fixing the stale `.. workflow::` directives in `docs/preprocessing.rst`.
- Cleaning up the duplicate `eddy_config` and `diffprep_config` entries in `execution._paths`.

## Review log

A Codex adversarial review of the first draft (task `task-mtx37zjc-sadktc`) raised these points.

**Accepted:**

- **`--config-file` ordering (high):** fixed by making the command line authoritative for
  the SHORELine block (section 1, "Interaction with `--config-file`") and by gating the
  summary on the resolved HMC method. Tests added.
- **QSIPlan's `cli_phrase()` appears in qsiprep's own workflow log and embedded grouping
  report (high):** handled in section 5.
- **The iteration fix has observable effects beyond node count (medium):** documented in
  "Effects of the iteration fix", and the tests check `summarize_iterations` and the
  methods text.
- **The SHORELine methods text misreports the model and transform (medium):** both
  `hmc.py` strings are fixed and tested.
- **The `execution._paths` rationale was wrong (low):** corrected.

**Declined:**

- **Moving all resolution after the `--config-file` merge:** it would restructure the
  method-axis normalization for no benefit over the command-line-authoritative rule, and it
  would diverge from how `hmc_method` reloads.
- **Numerical integration expectations, tests of every model and transform combination,
  and a `--sloppy` matrix:** the resolved values feed existing code paths unchanged, apart
  from the fixes above, which have targeted tests.
