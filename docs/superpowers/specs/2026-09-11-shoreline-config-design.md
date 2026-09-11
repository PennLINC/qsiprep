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
| QSIPlan coupling | Change qsiprep only. Exempt `--shoreline-model` in the plan-CLI conformance tests and open a follow-up issue in QSIPlan. |

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
  `eddy_config` and `diffprep_config` entries in `execution._paths` have no effect, but
  those options store plain strings. This design does not touch them.
- Rewrite the docstrings of `shoreline_iters`, `shoreline_model` and `hmc_transform` to say
  they are derived from `--shoreline-config` and are `None` unless `hmc_method` is `shoreline`.

## 2. Consumers and the summary

| Location | Change |
|---|---|
| `workflows/dwi/hmc_sdc.py:140` | Pass `num_model_iterations=config.workflow.shoreline_iters` to `init_dwi_hmc_wf`. This is the bug fix. |
| `workflows/dwi/base.py:257-262` | Delete the runtime check that `iters` is at least 1; the loader enforces it. |
| `workflows/dwi/base.py:~491` | Pass `hmc_transform` to `DiffusionSummary` only when `config.workflow.hmc_transform` is not `None`. |
| `interfaces/reports.py` | Make `DiffusionSummaryInputSpec.hmc_transform` optional (no longer `mandatory=True`). Render the "HMC Transform" `<li>` only when it is defined. The other lines are unchanged. |
| `workflows/dwi/hmc.py` (lines 357, 583, 722), `workflows/dwi/derivatives.py:89`, `utils/plan.py` | No change. They read the resolved fields. |
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

**QSIPlan conformance** (`qsiprep/tests/test_plan_cli_spec.py`):

- both conformance tests exempt `--shoreline-model` through a named constant, commented
  with the QSIPlan follow-up issue;
- a new test parses `--hmc-method shoreline --shoreline-config {"model": "tensor"}`.
  `cli_spec.selection_from_namespace` must return a SHORELine selection with
  `shoreline_model == 'tensor'`.

**Iteration wiring**, in `qsiprep/tests/test_workflows_gradwarp.py`, reusing
`_cfg_for_shoreline`, `_shoreline_wf` and `_rpe_unit`. These already build
`init_qsiprep_hmcsdc_wf`, the function that had the bug. Build the workflow with
`config.workflow.shoreline_iters` set to `3`, then `2`, and check the leaf names from
`wf.list_node_names()`:

- with `3`, some names start with `dwi_hmc_wf.dwi_model_hmc_wf.shoreline_iteration002.`;
- with `2`, names start with `...shoreline_iteration001.` but none with
  `...shoreline_iteration002.`.

This is the regression test for the bug fix: on the current code both builds contain only
`shoreline_iteration001`.

**Summary:** `DiffusionSummary` rendered without `hmc_transform` has no "HMC Transform"
line; with `hmc_transform='Rigid'` it shows "HMC Transform: Rigid".

### Integration tests (`qsiprep/tests/test_cli.py`; Docker/CI only, not run locally)

New files under `qsiprep/tests/data/`, loaded with `get_test_data_path()` the way
`eddy_config.json` is:

| File | Contents | Replaces |
|---|---|---|
| `shoreline_none_config.json` | `{"model": "none", "iters": 1}` | `--shoreline-model=none --shoreline-iters=1` (drbuddi_shoreline_epi); also `--shoreline-model=none` in the second SHORELine run (around line 645) |
| `shoreline_tensor_config.json` | `{"model": "tensor", "iters": 1}` | `--shoreline-model=tensor --shoreline-iters=1` (drbuddi_tensorline_epi) |
| `shoreline_rigid_config.json` | `{"transform": "Rigid", "iters": 1}` | `--hmc-transform=Rigid --shoreline-iters=1` (dscsdsi) |

With the iteration fix, `iters: 1` now really does make these runs cheaper, so expected
outputs stay the same apart from anything that depends on the iteration count. The
`hmcOptimization` file is written only for `3dSHORE` with `iters > 1`, which these
configurations don't trigger. If a run's `*_outputs.txt` changes, it is updated in the PR.

## 5. QSIPlan follow-up (outside this PR)

Draft an issue for PennLINC/QSIPlan and file it only after the user approves:

- remove `--shoreline-model` from `cli_spec.PLAN_OPTIONS`, or mark it as a flag qsiprep
  doesn't expose;
- make `MethodSelection.cli_phrase()` and the explorer suggest
  `--shoreline-config` with a `"model"` key instead of `--shoreline-model`;
- update QSIPlan's tests (`test_cli_spec.py`, `test_explorer.py`, `test_serve.py`).

Once that is released and qsiprep's `qsiplan` pin is bumped, remove the conformance
exemption.

## Out of scope

- Removing SHORELine (#1086).
- Fixing the stale `.. workflow::` directives in `docs/preprocessing.rst`.
- Cleaning up the ineffective `eddy_config` and `diffprep_config` entries in `execution._paths`.
