# `--shoreline-config` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `--shoreline-model`, `--shoreline-iters` and `--hmc-transform` with a single `--shoreline-config` JSON file (issue #1132). Along the way, fix the long-standing bug where the SHORELine iteration count never reached the workflow, and the SHORELine methods text that misreports the model and transform.

**Architecture:** The parser loads and validates the JSON once, in `DeprecationForwardingParser.parse_known_args`, through a new `load_shoreline_config()` in `qsiprep/utils/misc.py`. It writes the resolved values onto the existing internal names `shoreline_model`, `shoreline_iters` and `hmc_transform`, which workflow builders, `utils/plan.py` and QSIPlan's `selection_from_namespace` already read. `parse_args` assigns the SHORELine block to `config.workflow` directly, so the command line stays authoritative on `--config-file` reloads.

**Tech Stack:** Python 3.11, argparse, Nipype workflows, pytest, ruff, Sphinx (reStructuredText).

**Spec:** `docs/superpowers/specs/2026-09-11-shoreline-config-design.md`. Read it before starting; this plan follows it.

## Global Constraints

- Repo root: `/mnt/c/Users/tsalo/Documents/linc/qsiprep`. Branch: `shoreline-config`. The package directory is `qsiprep/` under the root.
- Python environment: `micromamba run -n linc311 ...`. Never use `lincapps`, `pip install` into any environment, conda, or venv.
- **QSIPlan version for tests:** `linc311` has an editable development checkout of QSIPlan installed, which makes four tests in `test_plan_cli_spec.py` fail. qsiprep pins `qsiplan >= 0.4.0, < 0.5`, so every test command in this plan runs against released qsiplan 0.4.0 on `PYTHONPATH`, through the helper script from Task 0. `S` below is the scratchpad: `/tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad`. Every test command is written out in full.
- **Git:**
  - never run `git stash`;
  - never `git add -A` or `git add .`; stage files by name;
  - never edit `docs/changes.md`;
  - end every commit message with:
    ```
    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
    ```
- Style: ruff line length 99, single quotes. Match the surrounding comment density.
- JSON keys: exactly `model` (`3dshore` | `tensor` | `none`), `iters` (int ≥ 1, any int when `model` is `none`, bools rejected), `transform` (`Affine` | `Rigid`). Shipped defaults: `{"model": "3dshore", "iters": 2, "transform": "Affine"}`.
- `--shoreline-model`, `--shoreline-iters` and `--hmc-transform` are removed outright, with no deprecation shims.
- Do **not** file the QSIPlan issue. Its text is drafted for the user in Task 6.
- Integration tests (`-m integration`) need Docker data and are **not** run locally. Only collection is checked.

## File Map

| File | Responsibility | Task |
|---|---|---|
| `qsiprep/data/shoreline_params.json` (create) | Shipped SHORELine defaults | 1 |
| `qsiprep/utils/misc.py` | `SHORELINE_MODELS`, `SHORELINE_TRANSFORMS`, `load_shoreline_config()` | 1 |
| `qsiprep/tests/test_cli.py` | Loader unit tests (Task 1); integration flags (Task 4) | 1, 4 |
| `qsiprep/cli/parser.py` | Flags, resolution, command-line-authoritative config assignment | 2 |
| `qsiprep/config.py` | `workflow.shoreline_config`, `_paths`, docstrings | 2 |
| `qsiprep/data/tests/config.toml` | Drop stale SHORELine keys from the eddy fixture | 2 |
| `qsiprep/tests/test_cli_run.py` | Parser, reload and round-trip tests | 2 |
| `qsiprep/tests/test_plan_cli_spec.py` | Conformance exemption, selection tests | 2 |
| `qsiprep/workflows/dwi/hmc_sdc.py` | Pass `shoreline_iters` through (bug fix) | 3 |
| `qsiprep/workflows/dwi/hmc.py` | Methods-text fixes | 3 |
| `qsiprep/workflows/dwi/base.py` | Drop the dead check; gate the summary transform | 3 |
| `qsiprep/interfaces/reports.py` | Optional "HMC Transform" line | 3 |
| `qsiprep/tests/test_workflows_gradwarp.py`, `test_reports.py`, `test_template_registration_settings.py` | Consumer tests | 3 |
| `qsiprep/tests/data/shoreline_{none,tensor,rigid}_config.json` (create) | Integration configs | 4 |
| `docs/quickstart.rst`, `docs/preprocessing.rst`, `docs/notebooks/grouping_tutorial.md` | User docs | 5 |

---

### Task 0: Environment and baseline

**Files:** none in the repo. Creates `S/qsiplan040/` and `S/pytest040.sh`.

- [ ] **Step 1: Confirm the branch and a clean tree**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git branch --show-current && git status --short`
Expected: `shoreline-config` and no output from `git status --short`.

- [ ] **Step 2: Install qsiplan 0.4.0 into the scratchpad (not into the environment)**

Run:
```bash
micromamba run -n linc311 python -m pip install -q --no-deps --target /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/qsiplan040 "qsiplan==0.4.0"
```
Expected: exit 0. Skip this step if `S/qsiplan040/qsiplan/__init__.py` already exists.

- [ ] **Step 3: Write the test helper script**

Create `S/pytest040.sh` with this content:
```bash
#!/usr/bin/env bash
# Run qsiprep's pytest in linc311 against released qsiplan 0.4.0.
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep || exit 1
PYTHONPATH=/tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/qsiplan040 \
    exec micromamba run -n linc311 pytest -p no:cacheprovider "$@"
```

- [ ] **Step 4: Verify the helper resolves qsiplan 0.4.0**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_plan_cli_spec.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Record the unit-suite baseline**

Run:
```bash
bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests -q -n 4 2>&1 | grep -E "^FAILED|^ERROR|passed|failed" > /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/baseline_unit.txt
```
Expected: the file ends with a `N passed` summary line. Any `FAILED` or `ERROR` lines are pre-existing failures that Task 6 compares against. This takes several minutes; use a generous timeout. Skip it if `baseline_unit.txt` already exists and was produced on this branch's base commit.

---

### Task 1: Shipped defaults and `load_shoreline_config`

**Files:**
- Create: `qsiprep/data/shoreline_params.json`
- Modify: `qsiprep/utils/misc.py`: insert after `validate_diffprep_config` (it ends with `    return` just before `def validate_gradient_flags`)
- Test: `qsiprep/tests/test_cli.py`: insert after `test_validate_diffprep_config_accepts_each_correction_mode`

**Interfaces:**
- Produces: `qsiprep.utils.misc.load_shoreline_config(path=None, model=None) -> dict` returns exactly `{'model': str, 'iters': int, 'transform': str}` and raises `ValueError` on any invalid input. `path` is `str | os.PathLike | None`. `model` is `'3dshore' | 'tensor' | 'none' | None`: a legacy override that conflicts with a `"model"` key in the file. Also `SHORELINE_MODELS = ('3dshore', 'tensor', 'none')` and `SHORELINE_TRANSFORMS = ('Affine', 'Rigid')`.

- [ ] **Step 1: Write the failing tests**

Insert into `qsiprep/tests/test_cli.py` after `test_validate_diffprep_config_accepts_each_correction_mode`:

```python
_SHORELINE_DEFAULTS = {'model': '3dshore', 'iters': 2, 'transform': 'Affine'}


def test_load_shoreline_config_defaults_match_the_shipped_file():
    import json

    from qsiprep.data import load as load_data
    from qsiprep.utils.misc import load_shoreline_config

    shipped = load_data('shoreline_params.json')
    assert json.loads(shipped.read_text()) == _SHORELINE_DEFAULTS
    assert load_shoreline_config(None) == _SHORELINE_DEFAULTS
    assert load_shoreline_config(str(shipped)) == _SHORELINE_DEFAULTS


def test_load_shoreline_config_merges_a_partial_file(tmp_path):
    import json

    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'tensor.json'
    cfg.write_text(json.dumps({'model': 'tensor'}))
    assert load_shoreline_config(str(cfg)) == {**_SHORELINE_DEFAULTS, 'model': 'tensor'}


def test_load_shoreline_config_missing(tmp_path):
    from qsiprep.utils.misc import load_shoreline_config

    with pytest.raises(ValueError, match='does not exist'):
        load_shoreline_config(str(tmp_path / 'nope.json'))


@pytest.mark.parametrize(
    ('contents', 'match'),
    [
        ('[1, 2]', 'must contain a JSON object'),
        ('{"model": ', 'not valid JSON'),
        # A typo must fail loudly rather than silently fall back to a default.
        ('{"iter": 3}', 'unknown key'),
        # Values are case-sensitive: the legacy --hmc-model spelling is not accepted.
        ('{"model": "3dSHORE"}', 'model='),
        ('{"transform": "affine"}', 'transform='),
        ('{"iters": 0}', 'iters='),
        ('{"iters": true}', 'iters='),
        ('{"iters": "2"}', 'iters='),
        ('{"iters": 1.5}', 'iters='),
    ],
)
def test_load_shoreline_config_rejects_bad_files(tmp_path, contents, match):
    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'bad.json'
    cfg.write_text(contents)
    with pytest.raises(ValueError, match=match):
        load_shoreline_config(str(cfg))


def test_load_shoreline_config_model_none_ignores_iters(tmp_path):
    import json

    from qsiprep.utils.misc import load_shoreline_config

    cfg = tmp_path / 'none.json'
    cfg.write_text(json.dumps({'model': 'none', 'iters': 0}))
    assert load_shoreline_config(str(cfg)) == {**_SHORELINE_DEFAULTS, 'model': 'none', 'iters': 0}


def test_load_shoreline_config_legacy_model_override(tmp_path):
    """The deprecated --hmc-model alias supplies the model; the file may still set the rest."""
    import json

    from qsiprep.utils.misc import load_shoreline_config

    assert load_shoreline_config(None, model='tensor') == {
        **_SHORELINE_DEFAULTS,
        'model': 'tensor',
    }

    iters_only = tmp_path / 'iters.json'
    iters_only.write_text(json.dumps({'iters': 3}))
    assert load_shoreline_config(str(iters_only), model='none') == {
        **_SHORELINE_DEFAULTS,
        'model': 'none',
        'iters': 3,
    }

    with_model = tmp_path / 'with_model.json'
    with_model.write_text(json.dumps({'model': '3dshore'}))
    with pytest.raises(ValueError, match='conflicts'):
        load_shoreline_config(str(with_model), model='tensor')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_cli.py -q -k load_shoreline_config`
Expected: every selected test FAILS, with `ImportError: cannot import name 'load_shoreline_config'` (or `FileNotFoundError`/`KeyError` for the missing `shoreline_params.json`).

- [ ] **Step 3: Create the shipped defaults**

Create `qsiprep/data/shoreline_params.json`:
```json
{
  "model": "3dshore",
  "iters": 2,
  "transform": "Affine"
}
```

- [ ] **Step 4: Implement the loader**

In `qsiprep/utils/misc.py`, insert this after `validate_diffprep_config` (after its final `    return` and the two blank lines), before `def validate_gradient_flags`:

```python
SHORELINE_MODELS = ('3dshore', 'tensor', 'none')
"""Signal models SHORELine can use to predict motion-correction targets."""

SHORELINE_TRANSFORMS = ('Affine', 'Rigid')
"""Transformations SHORELine can optimize during head motion correction."""


def load_shoreline_config(path=None, model=None):
    """Load a ``--shoreline-config`` JSON file, merged over the shipped defaults.

    Parameters
    ----------
    path : str, os.PathLike or None
        The SHORELine configuration JSON file. ``None`` uses the defaults in
        ``qsiprep/data/shoreline_params.json``.
    model : str or None
        A SHORELine model from the deprecated ``--hmc-model`` alias (``3dshore``,
        ``tensor`` or ``none``). It replaces the default model, and conflicts with
        a ``model`` key in the file.

    Returns
    -------
    dict
        Exactly the keys ``model``, ``iters`` and ``transform``.

    Raises
    ------
    ValueError
        If the file does not exist, is not a JSON object, has an unknown key or an
        invalid value, or sets ``model`` while ``model`` is also given.
    """
    import json
    import os

    from ..data import load as load_data

    cfg = json.loads(load_data('shoreline_params.json').read_text())
    source = 'The default SHORELine configuration'
    if path is not None:
        source = f'SHORELine configuration file {path}'
        if not os.path.exists(path):
            raise ValueError(f'{source} does not exist.')
        with open(path) as f:
            try:
                user_cfg = json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f'{source} is not valid JSON: {err}') from err
        if not isinstance(user_cfg, dict):
            raise ValueError(f'{source} must contain a JSON object.')
        # Unknown keys are errors so a typo cannot silently fall back to a default.
        unknown = sorted(set(user_cfg) - set(cfg))
        if unknown:
            raise ValueError(
                f'{source} has unknown key(s) {", ".join(unknown)}; '
                f'valid keys are {", ".join(sorted(cfg))}.'
            )
        if model is not None and 'model' in user_cfg:
            raise ValueError(f'--hmc-model conflicts with "model" in --shoreline-config {path}.')
        cfg.update(user_cfg)
    if model is not None:
        cfg['model'] = model

    if cfg['model'] not in SHORELINE_MODELS:
        raise ValueError(
            f'{source} sets model={cfg["model"]!r}; must be one of '
            f'{", ".join(SHORELINE_MODELS)}.'
        )
    if cfg['transform'] not in SHORELINE_TRANSFORMS:
        raise ValueError(
            f'{source} sets transform={cfg["transform"]!r}; must be one of '
            f'{", ".join(SHORELINE_TRANSFORMS)}.'
        )
    iters = cfg['iters']
    # bool is a subclass of int, so "iters": true has to be rejected explicitly.
    if (
        isinstance(iters, bool)
        or not isinstance(iters, int)
        or (iters < 1 and cfg['model'] != 'none')
    ):
        raise ValueError(
            f'{source} sets iters={iters!r}; must be an integer >= 1 '
            '(any integer when model is "none").'
        )
    return cfg
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_cli.py -q -k "load_shoreline_config or validate_diffprep"`
Expected: all pass (15 loader cases plus the 4 existing DIFFPREP validator tests).

- [ ] **Step 6: Lint**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 ruff check qsiprep/utils/misc.py qsiprep/tests/test_cli.py && micromamba run -n linc311 ruff format --check qsiprep/utils/misc.py qsiprep/tests/test_cli.py`
Expected: `All checks passed!` and no files that would be reformatted. If the format check fails, run `ruff format` on those two files and re-run.

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git add qsiprep/data/shoreline_params.json qsiprep/utils/misc.py qsiprep/tests/test_cli.py && git commit -q -F - <<'EOF'
Add load_shoreline_config and shipped SHORELine defaults

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
EOF
```

---

### Task 2: `--shoreline-config` in the parser and config

**Files:**
- Modify: `qsiprep/cli/parser.py`, in these places:
  - line 31 (import);
  - lines 74-78 (the `--hmc-model` entry in `deprecations`);
  - lines 193-208 (the SHORELine branch of `parse_known_args`);
  - lines 865-871 (`--hmc-transform`);
  - lines 879-885 and 899-902 (help texts);
  - lines 904-913 (`--shoreline-model`);
  - lines 951-957 (`--shoreline-iters`);
  - line ~1175 (`config.from_dict` in `parse_args`).
- Modify: `qsiprep/config.py`, in `workflow`: the `hmc_transform`, `shoreline_iters` and `shoreline_model` docstrings; a new `shoreline_config`; and `_paths` at line 708.
- Modify: `qsiprep/data/tests/config.toml`: remove `hmc_transform = "Affine"` and `shoreline_iters = 2`.
- Test: `qsiprep/tests/test_cli_run.py`, `qsiprep/tests/test_plan_cli_spec.py`

**Interfaces:**
- Consumes: `load_shoreline_config(path=None, model=None)` from Task 1.
- Produces:
  - the parsed namespace always has `shoreline_config` (absolute `pathlib.Path` or `None`), `shoreline_model` (`'3dshore' | 'tensor' | 'none' | None`), `shoreline_iters` (`int | None`) and `hmc_transform` (`'Affine' | 'Rigid' | None`), with non-`None` values only when `hmc_method == 'shoreline'`;
  - `hmc_model` is still derived as before;
  - `config.workflow.shoreline_config` exists and is listed in `workflow._paths`;
  - after `parse_args`, the four `config.workflow` SHORELine fields equal the namespace values exactly, `None` included.

- [ ] **Step 1: Write the failing parser tests**

In `qsiprep/tests/test_cli_run.py`:

(a) In `test_method_axis_defaults`, after `assert opts.shoreline_model is None`, add:
```python
    assert opts.shoreline_config is None
    assert opts.shoreline_iters is None
    assert opts.hmc_transform is None
```

(b) In `test_hmc_method_shoreline_gets_model_and_drbuddi`, after `assert opts.shoreline_model == '3dshore'`, add:
```python
    assert opts.shoreline_iters == 2
    assert opts.hmc_transform == 'Affine'
```

(c) Delete `test_shoreline_model_requires_shoreline_method` entirely and put this in its place:

```python
def _shoreline_json(tmp_path, name='shoreline.json', **settings):
    import json

    path = tmp_path / name
    path.write_text(json.dumps(settings))
    return str(path)


def test_shoreline_config_values_reach_the_namespace(minimal_args, tmp_path):
    from pathlib import Path

    cfg = _shoreline_json(tmp_path, model='tensor', iters=3, transform='Rigid')
    opts = _parse(minimal_args, '--hmc-method', 'shoreline', '--shoreline-config', cfg)
    assert opts.shoreline_config == Path(cfg).absolute()
    assert opts.shoreline_model == 'tensor'
    assert opts.shoreline_iters == 3
    assert opts.hmc_transform == 'Rigid'
    assert opts.hmc_model == 'tensor'


@pytest.mark.parametrize('method_args', [[], ['--hmc-method', 'eddy'], ['--hmc-method', 'tortoise']])
def test_shoreline_config_requires_shoreline_method(minimal_args, tmp_path, capsys, method_args):
    cfg = _shoreline_json(tmp_path, model='tensor')
    with pytest.raises(SystemExit):
        _parse(minimal_args, *method_args, '--shoreline-config', cfg)
    assert '--shoreline-config requires --hmc-method shoreline' in capsys.readouterr().err


def test_invalid_shoreline_config_is_a_parse_error(minimal_args, tmp_path, capsys):
    cfg = _shoreline_json(tmp_path, iter=3)
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-method', 'shoreline', '--shoreline-config', cfg)
    assert 'unknown key' in capsys.readouterr().err


def test_hmc_model_alias_merges_with_shoreline_config(minimal_args, tmp_path, capsys):
    cfg = _shoreline_json(tmp_path, iters=3)
    opts = _parse(minimal_args, '--hmc-model', 'tensor', '--shoreline-config', cfg)
    assert 'deprecated' in capsys.readouterr().err
    assert opts.hmc_method == 'shoreline'
    assert opts.shoreline_model == 'tensor'
    assert opts.shoreline_iters == 3
    assert opts.hmc_model == 'tensor'


def test_hmc_model_alias_conflicts_with_shoreline_config_model(minimal_args, tmp_path, capsys):
    cfg = _shoreline_json(tmp_path, model='3dshore')
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-model', 'tensor', '--shoreline-config', cfg)
    assert 'conflicts' in capsys.readouterr().err


@pytest.mark.parametrize(
    ('flag', 'value'),
    [('--shoreline-model', 'tensor'), ('--shoreline-iters', '3'), ('--hmc-transform', 'Rigid')],
)
def test_removed_shoreline_flags_are_rejected(minimal_args, capsys, flag, value):
    with pytest.raises(SystemExit):
        _parse(minimal_args, '--hmc-method', 'shoreline', flag, value)
    assert 'unrecognized arguments' in capsys.readouterr().err


_SHORELINE_KEYS = ('shoreline_config', 'shoreline_model', 'shoreline_iters', 'hmc_transform')


@pytest.fixture
def restore_shoreline_config():
    """Yield qsiprep.config, restoring the SHORELine block that parse_args writes."""
    from qsiprep import config

    saved = {key: getattr(config.workflow, key) for key in _SHORELINE_KEYS}
    yield config
    for key, value in saved.items():
        setattr(config.workflow, key, value)


def test_shoreline_config_survives_a_config_round_trip(
    minimal_args, tmp_path, restore_shoreline_config
):
    """A Path must be written as a path string, not the literal "PosixPath('...')"."""
    from pathlib import Path

    import toml

    config = restore_shoreline_config
    cfg = _shoreline_json(tmp_path, model='tensor')
    opts = _parse(minimal_args, '--hmc-method', 'shoreline', '--shoreline-config', cfg)
    config.workflow.load({'shoreline_config': opts.shoreline_config}, init=False)
    dumped = toml.dumps({'workflow': config.workflow.get()})
    assert 'PosixPath(' not in dumped
    config.workflow.shoreline_config = None
    config.workflow.load(toml.loads(dumped)['workflow'], init=False)
    assert config.workflow.shoreline_config == Path(cfg).absolute()


def _parse_with_config_file(tmp_path, toml_text, *extra):
    """Run the real parse_args, loading ``toml_text`` as a --config-file."""
    from qsiprep import config
    from qsiprep.cli.parser import parse_args

    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(str(bids_dir), long)
    work_dir = tmp_path / 'work'
    config_file = tmp_path / 'previous.toml'
    config_file.write_text(toml_text)
    config.from_dict({'bids_dir': str(bids_dir), 'work_dir': str(work_dir)}, init=True)
    parse_args(
        [
            str(bids_dir),
            str(tmp_path / 'out'),
            'participant',
            '--participant-label',
            '01',
            '--output-resolution',
            '2',
            '--work-dir',
            str(work_dir),
            '--skip-bids-validation',
            '--config-file',
            str(config_file),
            *extra,
        ]
    )
    return config


def test_config_file_reload_drops_stale_shoreline_settings(tmp_path, restore_shoreline_config):
    """An old eddy config.toml carried hmc_transform/shoreline_iters; they must not survive."""
    config = _parse_with_config_file(
        tmp_path,
        '[workflow]\nhmc_model = "eddy"\nhmc_transform = "Affine"\nshoreline_iters = 2\n',
    )
    for key in _SHORELINE_KEYS:
        assert getattr(config.workflow, key) is None, key


def _previous_shoreline_toml(tmp_path):
    old_json = _shoreline_json(tmp_path, name='old.json', model='tensor', iters=5)
    return (
        '[workflow]\n'
        f'shoreline_config = "{old_json}"\n'
        'shoreline_model = "tensor"\n'
        'shoreline_iters = 5\n'
        'hmc_transform = "Rigid"\n'
    )


def test_config_file_reload_uses_command_line_shoreline_defaults(
    tmp_path, restore_shoreline_config
):
    config = _parse_with_config_file(
        tmp_path, _previous_shoreline_toml(tmp_path), '--hmc-method', 'shoreline'
    )
    assert config.workflow.shoreline_config is None
    assert config.workflow.shoreline_model == '3dshore'
    assert config.workflow.shoreline_iters == 2
    assert config.workflow.hmc_transform == 'Affine'


def test_config_file_reload_uses_command_line_shoreline_config(
    tmp_path, restore_shoreline_config
):
    from pathlib import Path

    new_json = _shoreline_json(tmp_path, name='new.json', iters=3, transform='Rigid')
    config = _parse_with_config_file(
        tmp_path,
        _previous_shoreline_toml(tmp_path),
        '--hmc-method',
        'shoreline',
        '--shoreline-config',
        new_json,
    )
    assert config.workflow.shoreline_config == Path(new_json).absolute()
    assert config.workflow.shoreline_model == '3dshore'
    assert config.workflow.shoreline_iters == 3
    assert config.workflow.hmc_transform == 'Rigid'
```

- [ ] **Step 2: Write the failing QSIPlan conformance changes**

In `qsiprep/tests/test_plan_cli_spec.py`:

(a) Change the imports at the top to:
```python
import argparse
import json

import pytest
from qsiplan import cli_spec

from qsiprep.cli.parser import _build_parser
```

(b) After the imports, add:
```python
# QSIPlan still lists --shoreline-model as a qsiprep flag, but qsiprep takes the
# SHORELine model from --shoreline-config (a JSON "model" key) and fills the same
# ``shoreline_model`` dest that selection_from_namespace reads. Remove this, and
# bump the qsiplan pin, once QSIPlan stops listing the flag (QSIPlan issue:
# "Replace --shoreline-model with qsiprep's --shoreline-config").
_NOT_A_QSIPREP_FLAG = frozenset({'--shoreline-model'})
```

(c) Replace `test_parser_realizes_every_implemented_plan_option` and `test_planned_options_are_the_only_gaps` with:
```python
def test_parser_realizes_every_implemented_plan_option():
    actions = _actions()
    for option in cli_spec.PLAN_OPTIONS:
        if option.planned or option.flag in _NOT_A_QSIPREP_FLAG:
            continue
        action = actions.get(option.flag)
        assert action is not None, f'qsiprep parser is missing {option.flag}'
        missing = set(option.owned_choices()) - set(action.choices or ())
        assert not missing, f'{option.flag}: qsiprep is missing choices {sorted(missing)}'


def test_planned_options_are_the_only_gaps():
    actions = _actions()
    absent = sorted(
        o.flag
        for o in cli_spec.PLAN_OPTIONS
        if o.flag not in actions and o.flag not in _NOT_A_QSIPREP_FLAG
    )
    planned = sorted(o.flag for o in cli_spec.PLAN_OPTIONS if o.planned)
    assert absent == planned  # today: []
    # The exemption must not outlive the flag in QSIPlan's spec.
    assert _NOT_A_QSIPREP_FLAG <= {o.flag for o in cli_spec.PLAN_OPTIONS}
```

(d) Append at the end of the file:
```python
@pytest.mark.parametrize('model', ['3dshore', 'tensor', 'none'])
def test_shoreline_config_model_reaches_the_method_selection(tmp_path, model):
    cfg = tmp_path / 'shoreline.json'
    cfg.write_text(json.dumps({'model': model}))
    namespace = _parse_minimal(tmp_path, '--hmc-method', 'shoreline', '--shoreline-config', str(cfg))
    selection = cli_spec.selection_from_namespace(namespace)
    assert selection.hmc.value == 'shoreline'
    assert selection.shoreline_model == model


def test_eddy_namespace_has_no_shoreline_model(tmp_path):
    namespace = _parse_minimal(tmp_path, '--hmc-method', 'eddy')
    assert namespace.shoreline_model is None
    assert cli_spec.selection_from_namespace(namespace).hmc.value == 'eddy'
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_cli_run.py qsiprep/tests/test_plan_cli_spec.py -q`
Expected: the new tests FAIL, for example `unrecognized arguments: --shoreline-config` or `AttributeError: 'Namespace' object has no attribute 'shoreline_config'`. `test_removed_shoreline_flags_are_rejected` fails for all three flags because they still parse. The two rewritten conformance tests pass, since the flag still exists.

- [ ] **Step 4: Update `qsiprep/config.py`**

(a) Replace the `hmc_transform` field and docstring:
```python
    hmc_transform = None
    """Transformation SHORELine optimizes during head motion correction: Affine or
    Rigid. Derived from ``--shoreline-config``; None unless ``hmc_method`` is
    shoreline."""
```

(b) Replace the `shoreline_iters` and `shoreline_model` fields and docstrings (they sit between `separate_all_dwis` and `tortoise_gpu_cpu_ratio`) with:
```python
    shoreline_config = None
    """Configuration JSON for SHORELine (``--shoreline-config``)."""
    shoreline_iters = None
    """How many iterations to run SHORELine. Derived from ``--shoreline-config``;
    None unless ``hmc_method`` is shoreline."""
    shoreline_model = None
    """Signal model SHORELine uses to predict motion-correction targets:
    3dshore, tensor or none. Derived from ``--shoreline-config``; None unless
    ``hmc_method`` is shoreline."""
```

(c) At the end of `class workflow`, change `_paths = ('gradient_file',)` to:
```python
    _paths = ('gradient_file', 'shoreline_config')
```

- [ ] **Step 5: Update `qsiprep/cli/parser.py`**

(a) Line 31: replace `from ..utils.misc import parse_denoise_method` with:
```python
from ..utils.misc import load_shoreline_config, parse_denoise_method
```

(b) In `deprecations`, replace the `'--hmc-model'` entry with:
```python
        '--hmc-model': (
            '27.0.0',
            'Use `--hmc-method` instead (with a `--shoreline-config` "model" for the '
            'SHORELine signal model).',
        ),
```

(c) In `DeprecationForwardingParser.parse_known_args`, replace this block:
```python
            if legacy_hmc is not None:
                if namespace.shoreline_model is not None:
                    self.error(
                        '--shoreline-model requires --hmc-method shoreline '
                        '(not the deprecated --hmc-model)'
                    )
                namespace.hmc_method = hmc_model_to_method[legacy_hmc]
                namespace.shoreline_model = hmc_model_to_shoreline_model.get(legacy_hmc)
            if namespace.hmc_method is None:
                namespace.hmc_method = 'eddy'
            if namespace.shoreline_model is not None and namespace.hmc_method != 'shoreline':
                self.error('--shoreline-model requires --hmc-method shoreline')
            if namespace.hmc_method == 'shoreline':
                if namespace.shoreline_model is None:
                    namespace.shoreline_model = '3dshore'
                print(
```
with:
```python
            legacy_shoreline_model = None
            if legacy_hmc is not None:
                namespace.hmc_method = hmc_model_to_method[legacy_hmc]
                legacy_shoreline_model = hmc_model_to_shoreline_model.get(legacy_hmc)
            if namespace.hmc_method is None:
                namespace.hmc_method = 'eddy'
            if namespace.shoreline_config is not None and namespace.hmc_method != 'shoreline':
                self.error('--shoreline-config requires --hmc-method shoreline')
            # SHORELine settings come from --shoreline-config (or the shipped
            # defaults). Config and the workflow builders read only these resolved
            # values, which stay None for the other methods.
            namespace.shoreline_model = None
            namespace.shoreline_iters = None
            namespace.hmc_transform = None
            if namespace.hmc_method == 'shoreline':
                try:
                    shoreline = load_shoreline_config(
                        namespace.shoreline_config, model=legacy_shoreline_model
                    )
                except ValueError as err:
                    self.error(str(err))
                namespace.shoreline_model = shoreline['model']
                namespace.shoreline_iters = shoreline['iters']
                namespace.hmc_transform = shoreline['transform']
                print(
```
The `print(...)` removal notice and everything after it stay as they are. That includes `namespace.hmc_model = shoreline_model_to_hmc_model[namespace.shoreline_model]`.

(d) Delete the whole `--hmc-transform` argument:
```python
    g_moco.add_argument(
        '--hmc-transform',
        action='store',
        default='Affine',
        choices=['Affine', 'Rigid'],
        help='transformation to be optimized during head motion correction (default: affine)',
    )
```

(e) In the `--hmc-method` help, replace:
```python
        'sampling; see --shoreline-model and --shoreline-iters; scheduled '
```
with:
```python
        'sampling; configured with --shoreline-config; scheduled '
```

(f) Replace the `--hmc-model` help with:
```python
        help='DEPRECATED: use --hmc-method (and --shoreline-config) instead. '
        '"eddy" means `--hmc-method eddy`; "tortoise" means `--hmc-method '
        'tortoise`; "3dSHORE", "tensor" and "none" mean `--hmc-method '
        'shoreline` with the matching --shoreline-config "model".',
```

(g) Replace the whole `--shoreline-model` argument with:
```python
    g_moco.add_argument(
        '--shoreline-config',
        action='store',
        type=IsFile,
        default=None,
        help='path to a JSON file with settings for SHORELine (only valid with '
        '--hmc-method shoreline). Every key is optional: "model" is the signal model '
        'used to predict motion-correction targets ("3dshore", the default; "tensor"; '
        'or "none", which warps each non-b=0 image with the transform of its nearest '
        'b=0 image), "iters" is the number of SHORELine iterations (default: 2), and '
        '"transform" is the transformation optimized during head motion correction '
        '("Affine", the default, or "Rigid"). Unknown keys are an error. The current '
        'default can be found here: '
        'https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/shoreline_params.json',
    )
```

(h) Delete the whole `--shoreline-iters` argument:
```python
    g_moco.add_argument(
        '--shoreline-iters',
        action='store',
        type=int,
        default=2,
        help='number of SHORELine iterations. (default: 2)',
    )
```

(i) In `parse_args`, replace:
```python
    config.from_dict(vars(opts), init=['nipype'])
```
with:
```python
    config.from_dict(vars(opts), init=['nipype'])
    # The command line is authoritative for SHORELine settings, as it already is
    # for hmc_method. from_dict skips None values, so without this a --config-file
    # could leave a stale shoreline_config, hmc_transform or shoreline_iters behind.
    for key in ('shoreline_config', 'shoreline_model', 'shoreline_iters', 'hmc_transform'):
        setattr(config.workflow, key, getattr(opts, key))
```

- [ ] **Step 6: Update the eddy config fixture**

In `qsiprep/data/tests/config.toml`, delete these two lines from `[workflow]`:
```
hmc_transform = "Affine"
```
```
shoreline_iters = 2
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_cli_run.py qsiprep/tests/test_plan_cli_spec.py qsiprep/tests/test_cli.py qsiprep/tests/test_method_selection.py -q`
Expected: all pass. `test_cli.py` runs only non-integration tests by default.

- [ ] **Step 8: Lint**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 ruff check qsiprep/cli/parser.py qsiprep/config.py qsiprep/tests/test_cli_run.py qsiprep/tests/test_plan_cli_spec.py && micromamba run -n linc311 ruff format --check qsiprep/cli/parser.py qsiprep/config.py qsiprep/tests/test_cli_run.py qsiprep/tests/test_plan_cli_spec.py`
Expected: clean. Run `ruff format` on any file it flags, then re-run.

- [ ] **Step 9: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git add qsiprep/cli/parser.py qsiprep/config.py qsiprep/data/tests/config.toml qsiprep/tests/test_cli_run.py qsiprep/tests/test_plan_cli_spec.py && git commit -q -F - <<'EOF'
Replace SHORELine flags with --shoreline-config

Remove --shoreline-model, --shoreline-iters and --hmc-transform. The
parser resolves --shoreline-config (or the shipped defaults) into the
existing shoreline_model/shoreline_iters/hmc_transform values, and the
command line stays authoritative for them on --config-file reloads.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
EOF
```

---

### Task 3: Workflow consumers, summary and methods text

**Files:**
- Modify: `qsiprep/workflows/dwi/hmc_sdc.py:140`
- Modify: `qsiprep/workflows/dwi/hmc.py`: add a module constant after `DEFAULT_MEMORY_MIN_GB`; the iterative desc at ~386-389; the model-HMC `__desc__` at ~717-723
- Modify: `qsiprep/workflows/dwi/base.py:257-262` and the `summary` node at ~485-495
- Modify: `qsiprep/interfaces/reports.py`: `DIFFUSION_TEMPLATE`, `DiffusionSummaryInputSpec.hmc_transform`, `DiffusionSummary._generate_segment`
- Test: `qsiprep/tests/test_workflows_gradwarp.py`, `qsiprep/tests/test_reports.py`, `qsiprep/tests/test_template_registration_settings.py`

**Interfaces:**
- Consumes: the resolved `config.workflow.shoreline_iters` and `hmc_transform` from Task 2, and `config.workflow.hmc_model`, which the parser still derives (`'3dSHORE' | 'tensor' | 'none' | 'eddy' | 'tortoise'`).
- Produces: `DiffusionSummaryInputSpec.hmc_transform` is optional, and the "HMC Transform" line renders only when it is defined.

- [ ] **Step 1: Write the failing workflow tests**

Append to `qsiprep/tests/test_workflows_gradwarp.py`, after `test_shoreline_dis3d_does_not_gradwarp_sdc_inputs`:

```python
def test_shoreline_iters_sets_the_model_iteration_count(tmp_path):
    """Regression: --shoreline-iters never reached init_dwi_hmc_wf (always 2)."""
    _cfg_for_shoreline(tmp_path)
    config.workflow.shoreline_iters = 3
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path))
    model_wf = wf.get_node('dwi_hmc_wf.dwi_model_hmc_wf')
    assert model_wf.get_node('shoreline_iteration002') is not None
    assert model_wf.get_node('shoreline_iteration003') is None
    assert model_wf.get_node('summarize_iterations') is not None
    assert 'A total of 3 iterations' in model_wf.__desc__


def test_single_shoreline_iteration_skips_the_iteration_summary(tmp_path):
    _cfg_for_shoreline(tmp_path)
    config.workflow.shoreline_iters = 1
    wf = _shoreline_wf(tmp_path, _rpe_unit(tmp_path))
    model_wf = wf.get_node('dwi_hmc_wf.dwi_model_hmc_wf')
    assert model_wf.get_node('initial_model_iteration') is not None
    assert model_wf.get_node('shoreline_iteration001') is None
    assert model_wf.get_node('summarize_iterations') is None
    assert 'A total of 1 iterations' in model_wf.__desc__


@pytest.mark.parametrize(
    ('hmc_model', 'expected', 'unexpected'),
    [
        ('tensor', 'using a tensor model', '3dSHORE'),
        ('3dSHORE', 'using 3dSHORE [@merlet3dshore]', 'tensor model'),
    ],
)
def test_shoreline_methods_text_names_the_model_and_transform(
    tmp_path, hmc_model, expected, unexpected
):
    from qsiprep.workflows.dwi.hmc import init_dwi_model_hmc_wf

    _cfg_for_shoreline(tmp_path)
    config.workflow.hmc_model = hmc_model
    config.workflow.hmc_transform = 'Rigid'
    wf = init_dwi_model_hmc_wf(num_iters=2)
    assert expected in wf.__desc__
    assert unexpected not in wf.__desc__
    assert 'using a Rigid transform' in wf.__desc__


def test_eddy_summary_leaves_hmc_transform_undefined(tmp_path):
    """A stale hmc_transform (e.g. from a reloaded config) must not reach an eddy summary."""
    from nipype.interfaces.base import isdefined

    wf = _preproc_wf(tmp_path)
    # _dwi_preproc_cfg sets this, standing in for a stale value.
    assert config.workflow.hmc_transform == 'Affine'
    assert not isdefined(wf.get_node('summary').inputs.hmc_transform)
```

Append to `qsiprep/tests/test_reports.py`, after `test_diffusion_summary_renders_gradient_correction`:

```python
def _diffusion_summary(**overrides):
    from qsiprep.interfaces.reports import DiffusionSummary

    inputs = {
        'distortion_correction': 'TOPUP',
        'pe_direction': 'j',
        'hmc_model': 'eddy',
        'b0_to_anat_transform': 'Rigid',
        'denoise_method': 'dwidenoise',
        'dwi_denoise_window': 5,
    }
    inputs.update(overrides)
    return DiffusionSummary(**inputs)


def test_diffusion_summary_omits_hmc_transform_when_undefined():
    segment = _diffusion_summary()._generate_segment()
    assert 'HMC Transform' not in segment
    assert 'HMC Model: eddy' in segment


def test_diffusion_summary_shows_hmc_transform_when_given():
    segment = _diffusion_summary(hmc_model='3dSHORE', hmc_transform='Rigid')._generate_segment()
    assert '<li>HMC Transform: Rigid</li>' in segment
    assert 'HMC Model: 3dSHORE' in segment
```

Append to `qsiprep/tests/test_template_registration_settings.py`, after `test_within_scan_hmc_still_uses_shoreline`:

```python
def test_iterative_b0_description_names_the_transform(tmp_path):
    """The methods text named hmc_model ("tortoise registrations") instead of the transform."""
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid')
    assert 'iterations of Rigid registrations' in wf.__desc__
    assert 'tortoise' not in wf.__desc__
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_workflows_gradwarp.py qsiprep/tests/test_reports.py qsiprep/tests/test_template_registration_settings.py -q -k "shoreline_iter or methods_text or eddy_summary or hmc_transform or iterative_b0_description"`
Expected FAILs:
- `test_shoreline_iters_sets_the_model_iteration_count`: `shoreline_iteration002` is `None`.
- `test_single_shoreline_iteration_skips_the_iteration_summary`: `shoreline_iteration001` is not `None`.
- The tensor case of `test_shoreline_methods_text_names_the_model_and_transform`: "3dSHORE" is present.
- `test_eddy_summary_leaves_hmc_transform_undefined`: it is `'Affine'`.
- `test_diffusion_summary_omits_hmc_transform_when_undefined`: renders `HMC Transform: <undefined>`.
- `test_iterative_b0_description_names_the_transform`: says "tortoise registrations".

`test_diffusion_summary_shows_hmc_transform_when_given` and the `3dSHORE` methods-text case may already pass.

- [ ] **Step 3: Pass the iteration count through (`qsiprep/workflows/dwi/hmc_sdc.py`)**

Replace:
```python
    dwi_hmc_wf = init_dwi_hmc_wf(source_file=source_file)
```
with:
```python
    dwi_hmc_wf = init_dwi_hmc_wf(
        source_file=source_file,
        num_model_iterations=config.workflow.shoreline_iters,
    )
```

- [ ] **Step 4: Fix the methods text (`qsiprep/workflows/dwi/hmc.py`)**

(a) After `DEFAULT_MEMORY_MIN_GB = 0.01`, add:
```python

# Methods-text names for the SHORELine signal models, keyed by the hmc_model value
# SignalPrediction reads. "none" never builds the model-based workflow.
_SHORELINE_MODEL_DESCRIPTIONS = {
    '3dSHORE': '3dSHORE [@merlet3dshore]',
    'tensor': 'a tensor model',
}
```

(b) In `init_b0_hmc_wf`, iterative branch, replace:
```python
            f'of {config.workflow.hmc_model} registrations. '
```
with:
```python
            f'of {transform} registrations. '
```

(c) In `init_dwi_model_hmc_wf`, replace:
```python
    workflow.__desc__ = (
        'The SHORELine method was used to estimate head motion in b>0 '
        'images. This entails leaving out each b>0 image and reconstructing '
        'the others using 3dSHORE [@merlet3dshore]. The signal for the left-'
        f'out image serves as the registration target. A total of {num_iters} '
        f'iterations were run using a {config.workflow.hmc_transform} transform. '
    )
```
with:
```python
    model_description = _SHORELINE_MODEL_DESCRIPTIONS.get(
        config.workflow.hmc_model, config.workflow.hmc_model
    )
    workflow.__desc__ = (
        'The SHORELine method was used to estimate head motion in b>0 '
        'images. This entails leaving out each b>0 image and reconstructing '
        f'the others using {model_description}. The signal for the left-'
        f'out image serves as the registration target. A total of {num_iters} '
        f'iterations were run using a {config.workflow.hmc_transform} transform. '
    )
```

- [ ] **Step 5: Update `qsiprep/workflows/dwi/base.py`**

(a) Replace:
```python
    if hmc_tool == 'shoreline':
        if config.workflow.shoreline_model != 'none' and config.workflow.shoreline_iters < 1:
            raise Exception(
                '--shoreline-iters must be > 0 when --shoreline-model is '
                f'{config.workflow.shoreline_model}'
            )
        hmc_wf = init_qsiprep_hmcsdc_wf(
```
with:
```python
    if hmc_tool == 'shoreline':
        hmc_wf = init_qsiprep_hmcsdc_wf(
```

(b) In the `summary = pe.Node(DiffusionSummary(...), ...)` call, delete the line:
```python
            hmc_transform=config.workflow.hmc_transform,
```
and immediately after the closing `)` of that `summary = pe.Node(...)` statement, add:
```python
    if hmc_tool == 'shoreline':
        # Only SHORELine optimizes a selectable transform. Gating on the resolved
        # method, not the value, keeps a stale hmc_transform out of eddy and
        # TORTOISE summaries.
        summary.inputs.hmc_transform = config.workflow.hmc_transform
```
`hmc_tool` is `unit.run.hmc_stage.tool`, set earlier in the same function (`base.py:256`).

- [ ] **Step 6: Make the summary line optional (`qsiprep/interfaces/reports.py`)**

(a) In `DIFFUSION_TEMPLATE`, replace:
```
\t\t\t<li>HMC Transform: {hmc_transform}</li>
\t\t\t<li>HMC Model: {hmc_model}</li>
```
with:
```
{hmc_transform_line}\t\t\t<li>HMC Model: {hmc_model}</li>
```

(b) In `DiffusionSummaryInputSpec`, replace:
```python
    hmc_transform = traits.Str(mandatory=True, desc='transform used during HMC')
```
with:
```python
    hmc_transform = traits.Str(desc='transform optimized during HMC (SHORELine runs only)')
```

(c) In `DiffusionSummary._generate_segment`, just before `return DIFFUSION_TEMPLATE.format(`, add:
```python
        hmc_transform_line = ''
        if isdefined(self.inputs.hmc_transform):
            hmc_transform_line = f'\t\t\t<li>HMC Transform: {self.inputs.hmc_transform}</li>\n'

```
and in the `.format(...)` call, replace `hmc_transform=self.inputs.hmc_transform,` with:
```python
            hmc_transform_line=hmc_transform_line,
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_workflows_gradwarp.py qsiprep/tests/test_reports.py qsiprep/tests/test_template_registration_settings.py qsiprep/tests/test_workflows_native.py qsiprep/tests/test_intramodal_template.py -q`
Expected: all pass. That includes the existing SHORELine gradwarp tests, which run at the fixture's `shoreline_iters = 2`.

- [ ] **Step 8: Lint**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 ruff check qsiprep/workflows/dwi/hmc_sdc.py qsiprep/workflows/dwi/hmc.py qsiprep/workflows/dwi/base.py qsiprep/interfaces/reports.py qsiprep/tests/test_workflows_gradwarp.py qsiprep/tests/test_reports.py qsiprep/tests/test_template_registration_settings.py && micromamba run -n linc311 ruff format --check qsiprep/workflows/dwi/hmc_sdc.py qsiprep/workflows/dwi/hmc.py qsiprep/workflows/dwi/base.py qsiprep/interfaces/reports.py qsiprep/tests/test_workflows_gradwarp.py qsiprep/tests/test_reports.py qsiprep/tests/test_template_registration_settings.py`
Expected: clean. Run `ruff format` on any file it flags, then re-run.

- [ ] **Step 9: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git add qsiprep/workflows/dwi/hmc_sdc.py qsiprep/workflows/dwi/hmc.py qsiprep/workflows/dwi/base.py qsiprep/interfaces/reports.py qsiprep/tests/test_workflows_gradwarp.py qsiprep/tests/test_reports.py qsiprep/tests/test_template_registration_settings.py && git commit -q -F - <<'EOF'
Honor SHORELine iterations and report its settings accurately

Pass shoreline_iters to init_dwi_hmc_wf, which always ran two model
iterations. Name the actual SHORELine model and transform in the methods
text, and show "HMC Transform" in the summary only for SHORELine runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
EOF
```

---

### Task 4: Integration test configurations

**Files:**
- Create: `qsiprep/tests/data/shoreline_none_config.json`, `qsiprep/tests/data/shoreline_tensor_config.json`, `qsiprep/tests/data/shoreline_rigid_config.json`
- Modify: `qsiprep/tests/test_cli.py`: the four integration tests that pass the removed flags (`drbuddi_shoreline_epi`, `test_drbuddi_tensorline_epi`, `test_dscsdsi`, `intramodal_template`)

**Interfaces:**
- Consumes: `--shoreline-config` from Task 2; `get_test_data_path()` is already imported in `test_cli.py`.

- [ ] **Step 1: Create the three JSON files**

`qsiprep/tests/data/shoreline_none_config.json`:
```json
{
  "model": "none",
  "iters": 1
}
```
`qsiprep/tests/data/shoreline_tensor_config.json`:
```json
{
  "model": "tensor",
  "iters": 1
}
```
`qsiprep/tests/data/shoreline_rigid_config.json`:
```json
{
  "transform": "Rigid",
  "iters": 1
}
```

- [ ] **Step 2: Update the `drbuddi_shoreline_epi` test (the one with `--shoreline-model=none` and `--sdc-method=drbuddi`)**

Just before its `parameters = [`, add:
```python
    shoreline_config = os.path.join(get_test_data_path(), 'shoreline_none_config.json')
```
In its `parameters`, replace:
```python
        '--hmc-method=shoreline',
        '--shoreline-model=none',
        '--sdc-method=drbuddi',
        '--output-resolution=2',
        '--shoreline-iters=1',
```
with:
```python
        '--hmc-method=shoreline',
        f'--shoreline-config={shoreline_config}',
        '--sdc-method=drbuddi',
        '--output-resolution=2',
```

- [ ] **Step 3: Update `test_drbuddi_tensorline_epi`**

Just before its `parameters = [`, add:
```python
    shoreline_config = os.path.join(get_test_data_path(), 'shoreline_tensor_config.json')
```
Replace:
```python
        '--hmc-method=shoreline',
        '--shoreline-model=tensor',
        '--sdc-method=drbuddi',
        '--output-resolution=5',
        '--shoreline-iters=1',
```
with:
```python
        '--hmc-method=shoreline',
        f'--shoreline-config={shoreline_config}',
        '--sdc-method=drbuddi',
        '--output-resolution=5',
```

- [ ] **Step 4: Update `test_dscsdsi`**

Just before its `parameters = [`, add:
```python
    shoreline_config = os.path.join(get_test_data_path(), 'shoreline_rigid_config.json')
```
Replace:
```python
        '--hmc-method=shoreline',
        '--hmc-transform=Rigid',
        '--output-resolution=5',
        '--shoreline-iters=1',
```
with:
```python
        '--hmc-method=shoreline',
        f'--shoreline-config={shoreline_config}',
        '--output-resolution=5',
```

- [ ] **Step 5: Update the `intramodal_template` test**

Just before its `parameters = [`, add:
```python
    shoreline_config = os.path.join(get_test_data_path(), 'shoreline_none_config.json')
```
Replace:
```python
        '--hmc-method=shoreline',
        '--shoreline-model=none',
        '--b0-motion-corr-to=first',
```
with:
```python
        '--hmc-method=shoreline',
        f'--shoreline-config={shoreline_config}',
        '--b0-motion-corr-to=first',
```

- [ ] **Step 6: Verify: no removed flags remain, the files validate, and collection works**

Run:
```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && ! grep -n -- "--shoreline-model\|--shoreline-iters\|--hmc-transform" qsiprep/tests/test_cli.py && micromamba run -n linc311 python -c "
import os, qsiprep.tests.test_cli as t
from qsiprep.utils.misc import load_shoreline_config
d = t.get_test_data_path()
for name in ('none', 'tensor', 'rigid'):
    print(name, load_shoreline_config(os.path.join(d, f'shoreline_{name}_config.json')))
"
```
Expected: the grep prints nothing. The loader prints `{'model': 'none', 'iters': 1, 'transform': 'Affine'}`, `{'model': 'tensor', 'iters': 1, 'transform': 'Affine'}` and `{'model': '3dshore', 'iters': 1, 'transform': 'Rigid'}`.

Then run: `bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests/test_cli.py -m integration --collect-only -q 2>&1 | tail -3`
Expected: collection succeeds with no errors. Integration tests are not run locally.

- [ ] **Step 7: Lint and commit**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 ruff check qsiprep/tests/test_cli.py && micromamba run -n linc311 ruff format --check qsiprep/tests/test_cli.py`
Expected: clean.

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git add qsiprep/tests/data/shoreline_none_config.json qsiprep/tests/data/shoreline_tensor_config.json qsiprep/tests/data/shoreline_rigid_config.json qsiprep/tests/test_cli.py && git commit -q -F - <<'EOF'
Use --shoreline-config in SHORELine integration tests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
EOF
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/quickstart.rst` ("Head motion correction method", about lines 136-155)
- Modify: `docs/preprocessing.rst` (the SHORELine section, about lines 939-1007)
- Modify: `docs/notebooks/grouping_tutorial.md` (about lines 430-432)

`docs/usage.rst` is generated by sphinx-argparse from the parser, so it needs no edit. Do not touch `docs/changes.md`, or the stale `.. workflow::` directives in `preprocessing.rst`.

- [ ] **Step 1: `docs/quickstart.rst`**

Replace:
```rst
Head motion correction is selected with ``--hmc-method``, which takes
``eddy``, ``shoreline`` or ``tortoise``. (The deprecated ``--hmc-model``
values map onto these: ``eddy`` is ``--hmc-method eddy``, ``tortoise`` is
``--hmc-method tortoise``, and ``3dSHORE``/``tensor``/``none`` are
``--hmc-method shoreline`` with the matching ``--shoreline-model``.)

Choosing ``eddy`` (the default) runs FSL's ``eddy`` for head motion correction
and eddy current correction. This will work for single-shell and multi-shell
sampling schemes. The ``shoreline`` option (SHORELine) works for multi-shell,
Cartesian grid sampling (DSI) and random q-space sampling (CS-DSI); its
signal model is chosen with ``--shoreline-model``, either ``3dshore`` (the
default) or ``tensor``.

``--shoreline-model none`` will register all the b=0 images to one another
and the b>0 images will have the transform from the nearest b=0 image
applied. This is not recommended. Between ``eddy`` and ``shoreline``, all
```
with:
```rst
Head motion correction is selected with ``--hmc-method``, which takes
``eddy``, ``shoreline`` or ``tortoise``. (The deprecated ``--hmc-model``
values map onto these: ``eddy`` is ``--hmc-method eddy``, ``tortoise`` is
``--hmc-method tortoise``, and ``3dSHORE``/``tensor``/``none`` are
``--hmc-method shoreline`` with the matching SHORELine ``"model"``.)

Choosing ``eddy`` (the default) runs FSL's ``eddy`` for head motion correction
and eddy current correction. This will work for single-shell and multi-shell
sampling schemes. The ``shoreline`` option (SHORELine) works for multi-shell,
Cartesian grid sampling (DSI) and random q-space sampling (CS-DSI). Its
settings are passed as a JSON file with ``--shoreline-config``, for example:

.. code-block:: json

   {
     "model": "tensor",
     "iters": 2,
     "transform": "Rigid"
   }

``"model"`` is the signal model, either ``"3dshore"`` (the default) or
``"tensor"``; ``"iters"`` is the number of SHORELine iterations (default 2);
and ``"transform"`` is the transformation optimized during head motion
correction, ``"Affine"`` (the default) or ``"Rigid"``. Every key is optional.
See :ref:`configure_shoreline` for details.

Setting ``"model": "none"`` will register all the b=0 images to one another
and the b>0 images will have the transform from the nearest b=0 image
applied. This is not recommended. Between ``eddy`` and ``shoreline``, all
```

- [ ] **Step 2: `docs/preprocessing.rst`, the "none" paragraph**

Replace:
```rst
If ``"none"`` is specified as the hmc_model, then only the b0 images are used
and the non-b0 images are transformed based on their nearest b0 image. This
is probably not a great idea.
```
with:
```rst
If ``"model": "none"`` is set in ``--shoreline-config``, then only the b0
images are used and the non-b0 images are transformed based on their nearest
b0 image. This is probably not a great idea.
```

- [ ] **Step 3: `docs/preprocessing.rst`, the new subsection**

At the end of the SHORELine section, after the `.. workflow::` block that ends with `        dwi_metadata={},` and `    )`, and before the blank lines and `.. _dwi_sdc:`, insert:
```rst

.. _configure_shoreline:

Configuring SHORELine
^^^^^^^^^^^^^^^^^^^^^

SHORELine's settings are passed to *QSIPrep* as a JSON file with the
``--shoreline-config`` option, which is only valid with
``--hmc-method shoreline``. Every key is optional: missing keys take the
defaults below, and unknown keys or invalid values are an error.

.. list-table::
   :header-rows: 1

   * - Key
     - Allowed values
     - Default
     - Meaning
   * - ``model``
     - ``"3dshore"``, ``"tensor"``, ``"none"``
     - ``"3dshore"``
     - Signal model used to predict each left-out image. ``"none"`` skips the
       model and gives each b>0 image the transform of its nearest b=0 image.
   * - ``iters``
     - An integer of at least 1 (ignored when ``model`` is ``"none"``)
     - ``2``
     - Number of SHORELine iterations.
   * - ``transform``
     - ``"Affine"``, ``"Rigid"``
     - ``"Affine"``
     - Transformation optimized during head motion correction.

The default configuration can be viewed or downloaded `here
<https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/shoreline_params.json>`__.
```

- [ ] **Step 4: `docs/notebooks/grouping_tutorial.md`**

Replace:
```markdown
  (FSL), `tortoise` (TORTOISE's DIFFPREP), or `shoreline` (with
  `--shoreline-model` naming its signal model).
```
with:
```markdown
  (FSL), `tortoise` (TORTOISE's DIFFPREP), or `shoreline` (with a
  `--shoreline-config` JSON whose `"model"` names its signal model).
```

- [ ] **Step 5: Verify no user-facing references to the removed flags remain**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && grep -rn -- "--shoreline-model\|--shoreline-iters\|--hmc-transform\|shoreline-model" docs qsiprep --include=*.rst --include=*.md --include=*.py | grep -v "docs/changes.md\|docs/superpowers/\|docs/_build/\|tests/pytests/\|tests/test_data/"`
Expected: only these hits:
- the `_NOT_A_QSIPREP_FLAG` constant and its comment in `qsiprep/tests/test_plan_cli_spec.py`;
- the `test_removed_shoreline_flags_are_rejected` parametrization in `qsiprep/tests/test_cli_run.py`.

Any other hit must be fixed.

- [ ] **Step 6: Check the reST syntax of the edited sections**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 python -c "
import docutils.core, docutils.utils
for path in ('docs/quickstart.rst',):
    docutils.core.publish_doctree(open(path).read(), settings_overrides={'report_level': 2, 'halt_level': 5})
print('ok')
" 2>&1 | grep -v 'Unknown interpreted text role\|No role entry\|Unknown directive type\|No directive entry\|Trying\|:ref:\|workflow::\|^$' | tail -15`
Expected: `ok`, with no warnings about the new code block or list. Sphinx-only roles and directives such as `:ref:` and `.. workflow::` are filtered out. Then inspect the `preprocessing.rst` diff by eye (`git diff docs/preprocessing.rst`). `list-table` rows need consistent `* -` / `  -` indentation.

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git add docs/quickstart.rst docs/preprocessing.rst docs/notebooks/grouping_tutorial.md && git commit -q -F - <<'EOF'
Document --shoreline-config

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Fw6LUSiixwJEwerwhYxH5R
EOF
```

---

### Task 6: Final verification and handoff

**Files:** none modified, unless verification finds a problem.

- [ ] **Step 1: Full unit suite against the baseline**

Run:
```bash
bash /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/pytest040.sh qsiprep/tests -q -n 4 2>&1 | grep -E "^FAILED|^ERROR|passed|failed" > /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/final_unit.txt; diff <(grep -E "^FAILED|^ERROR" /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/baseline_unit.txt | sort) <(grep -E "^FAILED|^ERROR" /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/final_unit.txt | sort); tail -1 /tmp/claude-1000/-mnt-c-Users-tsalo-Documents-linc-qsiprep-qsiprep/69542bb0-c883-4df4-a123-46e41d78a10d/scratchpad/final_unit.txt
```
Expected: no line starting with `>`, meaning no new failures. Lines starting with `<` are baseline failures that now pass, which is fine. The passed count is higher than the baseline's by the number of new tests. Investigate any new failure with superpowers:systematic-debugging before continuing.

- [ ] **Step 2: Lint the whole change**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && git diff --name-only main...HEAD -- '*.py' | xargs micromamba run -n linc311 ruff check && git diff --name-only main...HEAD -- '*.py' | xargs micromamba run -n linc311 ruff format --check`
Expected: clean.

- [ ] **Step 3: Confirm the parser help renders**

Run: `cd /mnt/c/Users/tsalo/Documents/linc/qsiprep && micromamba run -n linc311 python -c "from qsiprep.cli.parser import _build_parser; _build_parser().print_help()" | grep -A12 -- "--shoreline-config"`
Expected: the `--shoreline-config` help text. `--shoreline-model`, `--shoreline-iters` and `--hmc-transform` appear nowhere in `print_help()` output (check with `| grep -c -- "--shoreline-model\|--shoreline-iters\|--hmc-transform"`, which should print `0`).

- [ ] **Step 4: Draft the QSIPlan issue for the user (do not file it)**

Present this text to the user in chat and wait for approval before filing it with `gh issue create -R PennLINC/QSIPlan`:

```markdown
Title: Replace --shoreline-model with qsiprep's --shoreline-config

qsiprep (PennLINC/qsiprep#1132) is replacing `--shoreline-model`, `--shoreline-iters`
and `--hmc-transform` with a single `--shoreline-config` JSON file whose `"model"` key
(`3dshore`, `tensor` or `none`) selects the SHORELine signal model. qsiprep still fills
the `shoreline_model` namespace dest, so `selection_from_namespace` keeps working.

Needed in QSIPlan:
- Drop `--shoreline-model` from `cli_spec.PLAN_OPTIONS`, or mark it as not exposed by
  qsiprep.
- `MethodSelection.cli_phrase()` should stop suggesting `--shoreline-model <model>` and
  instead mention a `--shoreline-config` file with `"model": "<model>"`. qsiprep shows
  this phrase in its workflow log and in the grouping report embedded in every subject
  report.
- Update the explorer (`interactive.py`) and tests (`test_cli_spec.py`,
  `test_explorer.py`, `test_serve.py`).

qsiprep's plan-CLI conformance test exempts `--shoreline-model` until this is released
and qsiprep's `qsiplan` pin is bumped. That must happen before the next qsiprep release.
```

If the user approves and the issue is filed, add its URL to the `_NOT_A_QSIPREP_FLAG` comment in `qsiprep/tests/test_plan_cli_spec.py` and commit.

- [ ] **Step 5: PR body notes (for when the user asks for a PR)**

The PR body, not `docs/changes.md`, must state:
- **Breaking:** `--shoreline-model`, `--shoreline-iters` and `--hmc-transform` are removed; use `--shoreline-config` (show the example JSON). 26.0.0 command lines that use `--shoreline-iters` or `--hmc-transform` now fail with "unrecognized arguments".
- **Behavior change:** the iteration count is now honored. Before, SHORELine always ran 2 model iterations regardless of `--shoreline-iters`, including in 26.0.0.
- **Release gate:** the QSIPlan follow-up (issue link) must be released, qsiprep's pin bumped and the conformance exemption removed before the next qsiprep release.
- The summary shows "HMC Transform" only for SHORELine runs, and the SHORELine methods text now names the actual model and transform.
