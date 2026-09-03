# Selectable MRtrix3 Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--mrtrix-version {stable,dev}` option that selects between a released MRtrix3 and the development branch, with complex-valued `mrdegibbs` available only under `dev`.

**Architecture:** The QSIPrep container ships two MRtrix3 installations, declared to Python through `MRTRIX3_STABLE_HOME` and `MRTRIX3_DEV_HOME`. A new `config.workflow.init()` reorders `PATH` so the selected tree resolves first — the only mechanism that also controls the helper commands `dwibiascorrect` spawns for itself. The two version-dependent argument spellings are handled by an explicit interface trait rather than a `config` lookup, so the interface stays testable in isolation.

**Tech Stack:** Python 3.11, nipype, argparse, pytest, Docker/BuildKit, CMake + Ninja.

**Spec:** `docs/superpowers/specs/2026-09-03-mrtrix-version-toggle-design.md`

## Global Constraints

- Run every Python command through micromamba: `micromamba run -n linc311 <command>`. Do not install packages and do not create environments.
- Run everything inside WSL, never PowerShell or cmd.exe.
- **Never edit `docs/changes.md`.** It is generated from PR titles at release time. User-facing descriptions go in the PR body.
- In `/mnt/c/Users/tsalo/Documents/linc/qsiprep_build`, stage files **by name**. Never `git add -A` or `git add .` there: the working tree carries a pre-existing whole-repo CRLF flip, and a blanket add would commit a 4015-line diff including `Dockerfile_MRtrix3`, which QSIRecon builds from.
- Never `git stash` in either repository.
- Image tags follow NiPreps CalVer, `YY.MINOR.PATCH`. Base-image tags are dates, `YYYYMMDD`.
- Flag values are exactly `stable` and `dev`. The default is `stable`.
- The environment variables are exactly `MRTRIX3_STABLE_HOME` and `MRTRIX3_DEV_HOME`.
- Container install paths are exactly `/opt/mrtrix3-stable` and `/opt/mrtrix3-dev`.
- Line length is 99 characters (ruff E501). Run `micromamba run -n linc311 ruff check qsiprep/` and `ruff format --check` before each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `qsiprep/cli/parser.py` | Declares `--mrtrix-version` | 1 |
| `qsiprep/config.py` | Holds `workflow.mrtrix_version`, reorders `PATH` in `workflow.init()`, records `environment.mrtrix3_home` | 1 |
| `qsiprep/tests/test_config.py` | **New.** Tests for `workflow.init()` | 1 |
| `qsiprep/tests/test_cli.py` | Parser tests for the new option | 1 |
| `qsiprep/interfaces/mrtrix.py` | `DWIBiasCorrect.mrtrix_version` and version-dependent ANTs spelling | 2 |
| `qsiprep/tests/test_interfaces_mrtrix.py` | Interface argument tests | 2, 6 |
| `qsiprep/workflows/dwi/merge.py` | Gates the complex path on the version; passes the version to `DWIBiasCorrect`; boilerplate | 3 |
| `qsiprep/tests/test_workflows_merge.py` | Graph-shape tests over both versions; container tests | 3, 6 |
| `docs/preprocessing.rst` | Qualifies complex unringing as development-branch-only | 3 |
| `qsiprep/interfaces/reports.py` | Warning banner and provenance line | 4 |
| `qsiprep/workflows/base.py` | Feeds the version into the summary interfaces | 4 |
| `qsiprep/tests/test_reports.py` | Report tests | 4 |
| `qsiprep_build/Dockerfile_MRtrix3dev` | Self-contained development-branch install (RPATH) | 5 |
| `Dockerfile.base` | Copies both trees, declares the env vars, sets the default `PATH` | 5 |
| `Dockerfile` | Base image bump, version-explicit verification | 5 |

`docs/usage.rst` needs no edit: it renders the parser with `.. argparse::`, so the option's help text *is* the documentation.

---

## Task 1: CLI option, config field, and `PATH` ordering

**Files:**
- Modify: `qsiprep/cli/parser.py:640` (immediately before `--unringing-method`)
- Modify: `qsiprep/config.py:281-313` (`environment`), `qsiprep/config.py:549+` (`workflow`)
- Create: `qsiprep/tests/test_config.py`
- Test: `qsiprep/tests/test_cli.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `config.workflow.mrtrix_version` — `str`, either `'stable'` or `'dev'`, default `'stable'`.
  - `config.workflow.init()` — classmethod, no arguments, returns `None`, may raise `RuntimeError`.
  - `config.environment.mrtrix3_home` — `str` or `None`, the absolute path of the selected tree.
  - `opts.mrtrix_version` on the parsed argparse namespace.

- [ ] **Step 1: Write the failing config tests**

Create `qsiprep/tests/test_config.py`:

```python
"""Tests for qsiprep.config."""

import os
from pathlib import Path

import pytest

from qsiprep import config


@pytest.fixture
def mrtrix_trees(tmp_path, monkeypatch):
    """Declare two fake MRtrix3 installations and make PATH restorable.

    ``workflow.init()`` mutates ``os.environ['PATH']`` directly, which monkeypatch
    would not undo on its own. Setting PATH through monkeypatch first registers it
    for restoration at teardown.
    """
    stable = tmp_path / 'mrtrix3-stable'
    dev = tmp_path / 'mrtrix3-dev'
    (stable / 'bin').mkdir(parents=True)
    (dev / 'bin').mkdir(parents=True)
    monkeypatch.setenv('MRTRIX3_STABLE_HOME', str(stable))
    monkeypatch.setenv('MRTRIX3_DEV_HOME', str(dev))
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    return str(stable), str(dev)


@pytest.mark.parametrize('selected', ['stable', 'dev'])
def test_workflow_init_puts_the_selected_tree_first(monkeypatch, mrtrix_trees, selected):
    """Resolve commands from the requested MRtrix3, falling through to the other tree.

    The second entry is load-bearing: dwidenoise2 exists only in the development
    tree, so it must remain reachable when ``stable`` is selected.
    """
    stable, dev = mrtrix_trees
    monkeypatch.setattr(config.workflow, 'mrtrix_version', selected)

    config.workflow.init()

    entries = os.environ['PATH'].split(os.pathsep)
    expected_first = dev if selected == 'dev' else stable
    expected_second = stable if selected == 'dev' else dev
    assert entries[0] == str(Path(expected_first, 'bin'))
    assert entries[1] == str(Path(expected_second, 'bin'))
    assert config.environment.mrtrix3_home == expected_first


def test_workflow_init_is_a_noop_without_declared_trees(monkeypatch):
    """Leave PATH alone on a bare-metal install, which has one MRtrix3 already on it."""
    monkeypatch.delenv('MRTRIX3_STABLE_HOME', raising=False)
    monkeypatch.delenv('MRTRIX3_DEV_HOME', raising=False)
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'dev')

    config.workflow.init()

    assert os.environ['PATH'] == '/usr/bin:/bin'
    assert config.environment.mrtrix3_home is None


def test_workflow_init_raises_when_the_selected_tree_is_missing(monkeypatch, tmp_path):
    """Fail loudly rather than silently running the other version.

    Falling back would build the complex workflow path and then hand complex data
    to a released mrdegibbs that cannot read it.
    """
    dev = tmp_path / 'mrtrix3-dev'
    (dev / 'bin').mkdir(parents=True)
    monkeypatch.setenv('MRTRIX3_DEV_HOME', str(dev))
    monkeypatch.delenv('MRTRIX3_STABLE_HOME', raising=False)
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'stable')

    with pytest.raises(RuntimeError, match='stable'):
        config.workflow.init()


def test_workflow_init_does_not_accumulate_duplicates(monkeypatch, mrtrix_trees):
    """Keep PATH stable across reloads; the image already bakes both trees into it."""
    stable, dev = mrtrix_trees
    monkeypatch.setattr(config.workflow, 'mrtrix_version', 'stable')

    config.workflow.init()
    after_first = os.environ['PATH']
    config.workflow.init()

    assert os.environ['PATH'] == after_first
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_config.py -v`
Expected: FAIL — `AttributeError: type object 'workflow' has no attribute 'init'`.

- [ ] **Step 3: Add the config field, `init()`, and the environment record**

In `qsiprep/config.py`, add to `class environment` (after the `cpu_count` block, keeping alphabetical-ish grouping is not required — the existing order is not strictly alphabetical):

```python
    mrtrix3_home = None
    """Absolute path of the MRtrix3 installation selected by ``--mrtrix-version``,
    or ``None`` when the platform declares no MRtrix3 installations."""
```

Add to `class workflow`, immediately after the `intramodal_template_transform` entry:

```python
    mrtrix_version = 'stable'
    """Which MRtrix3 installation to use: "stable" (a released version) or "dev"
    (the development branch, which is required for complex-valued ``mrdegibbs``)."""
```

Add this classmethod at the end of `class workflow`, after the `use_syn_sdc` attribute:

```python
    @classmethod
    def init(cls):
        """Put the MRtrix3 installation selected by ``--mrtrix-version`` first on ``PATH``.

        ``dwibiascorrect`` is a Python script that resolves ``mrcalc``, ``dwiextract``,
        ``mrmath``, ``mrconvert`` and ``N4BiasFieldCorrection`` through its own ``PATH``
        lookup, so ordering ``PATH`` is the only way to keep a single node on a single
        MRtrix3 version.
        """
        roots = {
            'stable': os.getenv('MRTRIX3_STABLE_HOME'),
            'dev': os.getenv('MRTRIX3_DEV_HOME'),
        }
        if not any(roots.values()):
            # A bare-metal installation has a single MRtrix3 on PATH already. The
            # setting still drives argument spellings and the workflow's shape.
            environment.mrtrix3_home = None
            return

        selected = roots[cls.mrtrix_version]
        if not selected or not Path(selected, 'bin').is_dir():
            raise RuntimeError(
                f'--mrtrix-version {cls.mrtrix_version} was requested, but no MRtrix3 '
                'installation was found for it.'
            )

        other = roots['dev' if cls.mrtrix_version == 'stable' else 'stable']
        # The other tree stays reachable: dwidenoise2 exists only in the development
        # branch, so it must resolve there whichever version is selected.
        bins = [str(Path(selected, 'bin'))]
        if other:
            bins.append(str(Path(other, 'bin')))

        # Drop existing entries for either tree first. The image bakes both into PATH,
        # so prepending without this would accumulate duplicates on every config reload.
        rest = [
            entry
            for entry in os.environ.get('PATH', '').split(os.pathsep)
            if entry and entry not in bins
        ]
        os.environ['PATH'] = os.pathsep.join(bins + rest)
        environment.mrtrix3_home = selected
```

`os` and `Path` are already module-level names in `config.py` (`Path` comes from the `try/finally` block at line 105). `environment` is defined above `workflow`, so referencing it from this method is fine.

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_config.py -v`
Expected: 5 PASSED (the first test is parameterized over two values).

- [ ] **Step 5: Write the failing parser tests**

Append to `qsiprep/tests/test_cli.py`:

```python
def test_parser_defaults_to_stable_mrtrix(tmp_path):
    """Default to a released MRtrix3, so existing runs are unchanged."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args([str(bids), str(out), 'participant', '--output-resolution', '2'])
    assert opts.mrtrix_version == 'stable'


def test_parser_accepts_dev_mrtrix(tmp_path):
    """``dev`` selects the development branch, which is what complex mrdegibbs needs."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [
            str(bids),
            str(out),
            'participant',
            '--mrtrix-version',
            'dev',
            '--output-resolution',
            '2',
        ]
    )
    assert opts.mrtrix_version == 'dev'


def test_parser_rejects_unknown_mrtrix_version(tmp_path):
    """Reject version strings; the flag names installations, not releases."""
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(bids),
                str(out),
                'participant',
                '--mrtrix-version',
                '3.0.8',
                '--output-resolution',
                '2',
            ]
        )
```

- [ ] **Step 6: Run the parser tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_cli.py -k mrtrix -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'mrtrix_version'` for the first two, and the third fails because argparse accepts the unknown option's value into nothing.

- [ ] **Step 7: Add the CLI option**

In `qsiprep/cli/parser.py`, insert immediately before the `'--unringing-method'` block (currently line 640):

```python
    g_conf.add_argument(
        '--mrtrix-version',
        action='store',
        choices=['stable', 'dev'],
        default='stable',
        help=(
            'Which MRtrix3 installation to use.\n'
            ' - stable: a released MRtrix3 (default)\n'
            ' - dev: the MRtrix3 development branch, which is required for '
            'complex-valued unringing with --unringing-method mrdegibbs. '
            'The development branch has not been through a release cycle and may '
            'contain bugs.'
        ),
    )
```

- [ ] **Step 8: Run the parser tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_cli.py -k mrtrix -v`
Expected: 3 PASSED.

- [ ] **Step 9: Check lint and formatting**

Run:
```bash
micromamba run -n linc311 ruff check qsiprep/config.py qsiprep/cli/parser.py qsiprep/tests/test_config.py qsiprep/tests/test_cli.py
micromamba run -n linc311 ruff format --check qsiprep/config.py qsiprep/cli/parser.py qsiprep/tests/test_config.py qsiprep/tests/test_cli.py
```
Expected: no findings.

- [ ] **Step 10: Commit**

```bash
git add qsiprep/config.py qsiprep/cli/parser.py qsiprep/tests/test_config.py qsiprep/tests/test_cli.py
git commit -m "Add --mrtrix-version and order PATH from the selected installation"
```

---

## Task 2: Version-dependent `dwibiascorrect` ANTs options

**Files:**
- Modify: `qsiprep/interfaces/mrtrix.py:447-457` (`DWIBiasCorrectInputSpec`), `qsiprep/interfaces/mrtrix.py:481` (`DWIBiasCorrect`)
- Test: `qsiprep/tests/test_interfaces_mrtrix.py:114-129` (rewrite the existing test)

**Interfaces:**
- Consumes: the `'stable'`/`'dev'` vocabulary established in Task 1. This task does **not** read `config`.
- Produces: `DWIBiasCorrect(mrtrix_version=...)`, a `traits.Enum('stable', 'dev', usedefault=True)` input.

**Background:** MRtrix3 3.0.x spells these options `-ants.b`, `-ants.c`, `-ants.s`; the development branch spells them `-ants_b`, `-ants_c`, `-ants_s` and rejects the dot form. `ants_b` and `ants_c` are `usedefault=True`, so they are emitted on every run regardless of any other setting. This is the only difference between the two MRtrix3 versions that breaks unconditionally.

- [ ] **Step 1: Write the failing test**

Replace `test_dwibiascorrect_uses_underscore_ants_options` in `qsiprep/tests/test_interfaces_mrtrix.py` (currently lines 114-129) with:

```python
@pytest.mark.parametrize(
    ('mrtrix_version', 'separator', 'rejected'),
    [('stable', '.', '-ants_'), ('dev', '_', '-ants.')],
)
def test_dwibiascorrect_ants_option_spelling(tmp_path, mrtrix_version, separator, rejected):
    """Spell the N4 options the way the selected MRtrix3 expects.

    3.0.x uses -ants.b and rejects the underscore form; the development branch
    renamed them to -ants_b and rejects the dot form. These options are emitted on
    every run, so getting this wrong fails every bias-correction node at runtime.
    """
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    interface = mrtrix.DWIBiasCorrect(
        method='ants',
        in_file=in_file,
        ants_s='4',
        mrtrix_version=mrtrix_version,
    )
    cmdline = interface.cmdline

    assert f'-ants{separator}b [150,3]' in cmdline
    assert f'-ants{separator}c [200x200,1e-6]' in cmdline
    assert f'-ants{separator}s 4' in cmdline
    assert rejected not in cmdline
    # mrtrix_version selects a spelling; it is not itself an mrtrix option
    assert 'mrtrix_version' not in cmdline
    assert '--mrtrix' not in cmdline


def test_dwibiascorrect_defaults_to_stable_spelling(tmp_path):
    """A bare DWIBiasCorrect() matches the released MRtrix3, like the CLI default."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    cmdline = mrtrix.DWIBiasCorrect(method='ants', in_file=in_file).cmdline
    assert '-ants.b [150,3]' in cmdline
    assert '-ants_' not in cmdline
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_mrtrix.py -k dwibiascorrect -v`
Expected: FAIL — `TraitError` on the unknown `mrtrix_version` input for the parameterized test, and the default test fails on `-ants.b` not being present.

- [ ] **Step 3: Implement the version-dependent spelling**

In `qsiprep/interfaces/mrtrix.py`, replace the three ANTs trait definitions in `DWIBiasCorrectInputSpec`:

```python
    # The argstr is a bare placeholder: DWIBiasCorrect._format_arg builds the real
    # flag, because 3.0.x spells these -ants.b and the development branch -ants_b.
    # The placeholder is load-bearing — nipype formats only traits that declare an
    # argstr, so removing it would silently emit nothing at all.
    ants_b = traits.Str(default_value='[150,3]', argstr='%s', usedefault=True)
    ants_c = traits.Str(default_value='[200x200,1e-6]', argstr='%s', usedefault=True)
    ants_s = traits.Str(default_value='4', argstr='%s')
    mrtrix_version = traits.Enum(
        'stable',
        'dev',
        usedefault=True,
        desc='which MRtrix3 installation this node will run against',
    )
```

`mrtrix_version` deliberately carries no `argstr`, so nipype never emits it as a flag.

In the same file, note the version requirement on `MRDeGibbsInputSpec.dimensionality`
(currently line 524). Nothing in the workflow sets it, so there is no live
incompatibility and no guard is needed, but the option exists only on the development
branch. Extend its `desc`:

```python
        desc=(
            'dimensionality of the operation: 2 for the slice-wise method of Kellner et al., '
            '3 for the volume-wise extension of Bautista et al. Left unset, mrdegibbs '
            'defaults to 2. Requires --mrtrix-version dev; the released mrdegibbs does '
            'not accept this option.'
        ),
```

Add this method to `class DWIBiasCorrect`, immediately after the `_cmd`/`input_spec`/`output_spec` assignments and before `_get_plotting_images`:

```python
    def _format_arg(self, name, spec, value):
        if name in ('ants_b', 'ants_c', 'ants_s'):
            separator = '_' if self.inputs.mrtrix_version == 'dev' else '.'
            return f'-ants{separator}{name[-1]} {value}'
        return super()._format_arg(name, spec, value)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_mrtrix.py -v`
Expected: all PASSED, including the pre-existing tests in that file.

- [ ] **Step 5: Check lint and formatting**

Run:
```bash
micromamba run -n linc311 ruff check qsiprep/interfaces/mrtrix.py qsiprep/tests/test_interfaces_mrtrix.py
micromamba run -n linc311 ruff format --check qsiprep/interfaces/mrtrix.py qsiprep/tests/test_interfaces_mrtrix.py
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/mrtrix.py qsiprep/tests/test_interfaces_mrtrix.py
git commit -m "Spell the dwibiascorrect ANTs options per MRtrix3 version"
```

---

## Task 3: Gate the complex path on the selected version

**Files:**
- Modify: `qsiprep/workflows/dwi/merge.py:476-480` (the `unring_complex` flag), `:624-634` (the `mrdegibbs` boilerplate), `:688` (`DWIBiasCorrect` construction)
- Modify: `qsiprep/workflows/dwi/finalize.py:585`, `:638` (the other two `DWIBiasCorrect` nodes)
- Modify: `docs/preprocessing.rst:144-148`
- Test: `qsiprep/tests/test_workflows_merge.py:442-459` (`_build_denoising_wf`), `:200-207` (`_run_denoising_wf`), `:463-477`; `qsiprep/tests/test_n4_robustness.py`

**Interfaces:**
- Consumes: `config.workflow.mrtrix_version` (Task 1); `DWIBiasCorrect(mrtrix_version=...)` (Task 2).
- Produces: `_build_denoising_wf(..., mrtrix_version='dev')` and `_run_denoising_wf(..., mrtrix_version='dev')` test helpers, both keyword-only with default `'dev'`.

**Background:** `merge.py` reads `config.workflow.*` directly, so nothing threads through function signatures. `_ImageChain.to_magnitude()` already runs before the unringing block whenever `unring_complex` is false, so the `stable` fallback needs no new code path — it is `main`'s path.

The test helpers default to `'dev'` rather than `'stable'` because every existing complex test in that file was written against the development branch. Defaulting to `'stable'` would silently invert what those tests assert.

- [ ] **Step 1: Write the failing graph-shape test**

In `qsiprep/tests/test_workflows_merge.py`, change `_build_denoising_wf` (line 442) to accept and set the version:

```python
def _build_denoising_wf(
    monkeypatch,
    denoise_method,
    unringing_method,
    use_phase,
    do_biascorr=False,
    mrtrix_version='dev',
):
    """Build (without running) a denoising workflow with the given configuration.

    ``mrtrix_version`` defaults to ``'dev'`` because the complex-path tests in this
    module were written against the development branch, where mrdegibbs reads and
    writes complex data.
    """
    monkeypatch.setattr(config.workflow, 'denoise_method', denoise_method)
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', unringing_method)
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    return init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=use_phase,
        do_biascorr=do_biascorr,
    )
```

Then add these tests immediately after `test_complex_data_stay_complex_through_mrdegibbs` (line 477):

```python
@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_stable_mrtrix_splits_before_mrdegibbs(monkeypatch, denoise_method):
    """Reduce to magnitude before mrdegibbs when a released MRtrix3 is selected.

    3.0.x mrdegibbs cannot read complex data, so handing it complex input would fail
    at runtime. This is the behavior QSIPrep had before complex unringing existed.
    """
    workflow = _build_denoising_wf(
        monkeypatch, denoise_method, 'mrdegibbs', use_phase=True, mrtrix_version='stable'
    )
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'degibbser')] == {('out_file', 'in_file')}
    assert connections[('degibbser', 'outputnode')] == {('out_file', 'dwi_file')}
    # The split happens once, before unringing, not after it
    assert ('degibbser', 'split_complex') not in connections


def test_stable_mrdegibbs_says_what_dev_would_buy(monkeypatch, caplog):
    """Tell the user that complex unringing exists, but only where it is actionable.

    The message belongs at workflow-build time rather than parse time: use_phase is a
    per-scan property the parser cannot know.
    """
    with caplog.at_level('INFO', logger='nipype.workflow'):
        _build_denoising_wf(
            monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True, mrtrix_version='stable'
        )

    assert '--mrtrix-version dev' in caplog.text


@pytest.mark.parametrize('unringing_method', ['rpg', 'none'])
def test_no_advice_when_mrdegibbs_is_not_running(monkeypatch, caplog, unringing_method):
    """Stay quiet where the advice would not apply; rpg is magnitude-only anyway."""
    with caplog.at_level('INFO', logger='nipype.workflow'):
        _build_denoising_wf(
            monkeypatch, 'dwidenoise', unringing_method, use_phase=True, mrtrix_version='stable'
        )

    assert '--mrtrix-version dev' not in caplog.text


@pytest.mark.parametrize('mrtrix_version', ['stable', 'dev'])
def test_biascorr_gets_the_selected_mrtrix_version(monkeypatch, mrtrix_version):
    """Give dwibiascorrect the option spelling its own MRtrix3 accepts."""
    workflow = _build_denoising_wf(
        monkeypatch,
        'dwidenoise',
        'mrdegibbs',
        use_phase=True,
        do_biascorr=True,
        mrtrix_version=mrtrix_version,
    )
    biascorr = next(
        node for node in workflow._get_all_nodes() if node.name == 'biascorr'
    )

    assert biascorr.interface.inputs.mrtrix_version == mrtrix_version


@pytest.mark.parametrize('mrtrix_version', ['stable', 'dev'])
def test_biascorr_never_receives_complex_data(monkeypatch, mrtrix_version):
    """Keep dwibiascorrect on magnitude data under either MRtrix3 version.

    dwibiascorrect is magnitude-only in both, so the split must precede it however
    the complex data reached that point.
    """
    workflow = _build_denoising_wf(
        monkeypatch,
        'dwidenoise',
        'mrdegibbs',
        use_phase=True,
        do_biascorr=True,
        mrtrix_version=mrtrix_version,
    )
    connections = _connections(workflow)

    assert connections[('split_complex', 'biascorr')] == {('out_file', 'in_file')}
    assert connections[('split_complex', 'get_b0s')] == {('out_file', 'dwi_series')}
    assert connections[('biascorr', 'outputnode')] >= {('out_file', 'dwi_file')}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_merge.py -k "stable_mrtrix or selected_mrtrix or never_receives_complex or what_dev_would_buy or no_advice" -v`
Expected: FAIL. `test_stable_mrtrix_splits_before_mrdegibbs` fails with a `KeyError` on `('denoiser', 'split_complex')`, because the workflow still routes complex data into `mrdegibbs`. `test_biascorr_gets_the_selected_mrtrix_version` fails because `biascorr` has no `mrtrix_version` input set. `test_stable_mrdegibbs_says_what_dev_would_buy` fails because nothing logs the advice yet. `test_no_advice_when_mrdegibbs_is_not_running` and `test_biascorr_never_receives_complex_data` may pass already — they pin behavior that must survive, not behavior being added.

- [ ] **Step 3: Gate the flag and pass the version to `DWIBiasCorrect`**

In `qsiprep/workflows/dwi/merge.py`, replace the `unring_complex` assignment (line 480):

```python
    # mrdegibbs is built on the Fourier shift theorem and reads and writes complex data
    # on MRtrix3's development branch, so complex data stay complex through unringing.
    # The released mrdegibbs cannot, and TORTOISE's rpg is magnitude-only.
    unring_complex = (
        denoise_complex
        and unringing_method == 'mrdegibbs'
        and config.workflow.mrtrix_version == 'dev'
    )
    if denoise_complex and unringing_method == 'mrdegibbs' and not unring_complex:
        config.loggers.workflow.info(
            'Complex-valued Gibbs unringing is available with --mrtrix-version dev. '
            'The magnitude data will be unrung instead.'
        )
```

Replace the `biascorr` node construction (line 691):

```python
        biascorr = pe.Node(
            DWIBiasCorrect(method='ants', mrtrix_version=config.workflow.mrtrix_version),
            name='biascorr',
            n_procs=omp_nthreads,
        )
```

`merge.py` is not the only place that builds this node. `qsiprep/workflows/dwi/finalize.py`
constructs `DWIBiasCorrect` twice more, and `b1_biascorrect_stage` defaults to `final`, which
is the finalize path — so these are the *common* case, not an edge case. Missing them would
fail every `--mrtrix-version dev` run that does bias correction. Apply the same change at
`finalize.py:585`:

```python
            biascorr = pe.Node(
                DWIBiasCorrect(
                    method='ants',
                    bzero_max=config.workflow.b0_threshold,
                    mrtrix_version=config.workflow.mrtrix_version,
                ),
                name='biascorr',
                n_procs=omp_nthreads,
            )
```

and at `finalize.py:638`:

```python
                biascorrs.append(
                    pe.Node(
                        DWIBiasCorrect(
                            method='ants',
                            bzero_max=config.workflow.b0_threshold,
                            mrtrix_version=config.workflow.mrtrix_version,
                        ),
                        name='biascorr%d' % scan_num,
                        n_procs=omp_nthreads,
                    )
                )
```

- [ ] **Step 4: Cover the finalize-stage bias-correction nodes**

Add to `qsiprep/tests/test_n4_robustness.py`, which already builds this workflow (see
`test_biascorr_receives_the_conditioned_weights_not_the_raw_mask` at line 138 for the
`_config()` fixture pattern used in that module):

```python
@pytest.mark.parametrize('split_biascorr', [False, True])
@pytest.mark.parametrize('mrtrix_version', ['stable', 'dev'])
def test_finalize_biascorr_gets_the_selected_mrtrix_version(
    tmp_path, monkeypatch, split_biascorr, mrtrix_version
):
    """Give every dwibiascorrect node the option spelling its MRtrix3 accepts.

    b1_biascorrect_stage defaults to "final", so these nodes are on the common path.
    A node left on the default spelling fails at runtime under --mrtrix-version dev.
    """
    from qsiprep import config
    from qsiprep.workflows.dwi.finalize import init_finalize_denoising_wf

    _config().execution.output_dir = str(tmp_path)
    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)

    wf = init_finalize_denoising_wf(
        source_file='/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz',
        do_biascorr=True,
        num_dwi_acquisitions=1,
        split_biascorr=split_biascorr,
    )

    biascorrs = [
        node for node in wf._get_all_nodes() if node.name.startswith('biascorr')
    ]
    assert biascorrs, 'no bias-correction node was built'
    for node in biascorrs:
        assert node.interface.inputs.mrtrix_version == mrtrix_version, node.name
```

Check whether `pytest` and `_config` are already imported in that module before adding
the test; both are used by the surrounding tests.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_n4_robustness.py -v`
Expected: all PASSED — the new tests and the pre-existing complex-path tests, which now pin `mrtrix_version='dev'` through the helper's default.

- [ ] **Step 6: Add the version to the run-level test helper**

In `qsiprep/tests/test_workflows_merge.py`, change `_run_denoising_wf` (line 200) to accept `mrtrix_version='dev'` and both set it and apply it:

```python
def _run_denoising_wf(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    use_phase,
    dwi_denoise_window='auto',
    unringing_method='none',
    mrtrix_version='dev',
):
```

and, among the other `monkeypatch.setattr` calls in that function, add:

```python
    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    # config.workflow.init() mutates os.environ['PATH'] directly, which monkeypatch
    # would not undo. Setting PATH through monkeypatch first registers it for
    # restoration at teardown.
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    config.workflow.init()
```

`os` is already imported at the top of that module.

- [ ] **Step 7: Run the non-container tests to verify nothing regressed**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_merge.py -v`
Expected: same result as Step 4. The container-gated tests will skip or error on missing data, as they did before this task; note which ones so the outcome can be compared after Task 6.

- [ ] **Step 8: Update the boilerplate**

In `qsiprep/workflows/dwi/merge.py`, the `unring_complex` branch of the `mrdegibbs` boilerplate (line 624) already reads correctly for `dev`. Change the `else` branch so the released case names the version explicitly:

```python
            else:
                desc += (
                    f'{last_step}Gibbs ringing was removed from the magnitude data using '
                    'MRtrix3 [@mrtrix3; @mrdegibbs]. '
                )
```

- [ ] **Step 9: Write the failing boilerplate test**

Add to `qsiprep/tests/test_workflows_merge.py`, immediately after `test_boilerplate_describes_where_the_split_happens`:

```python
def test_boilerplate_says_magnitude_under_stable_mrtrix(monkeypatch):
    """Describe what actually ran: released mrdegibbs sees magnitude data only."""
    workflow = _build_denoising_wf(
        monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True, mrtrix_version='stable'
    )

    assert 'Gibbs ringing was removed from the magnitude data' in workflow.__desc__
    assert 'complex-valued data' not in workflow.__desc__
    assert workflow.__desc__.index('split back into magnitude') < workflow.__desc__.index(
        'Gibbs ringing'
    )
```

- [ ] **Step 10: Run the boilerplate test to verify it passes**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_merge.py -k boilerplate -v`
Expected: PASSED. If `'complex-valued data'` still appears, the `unring_complex` gate in Step 3 was not applied.

- [ ] **Step 11: Qualify the documentation**

In `docs/preprocessing.rst`, replace lines 144-148 with:

```rst
When phase data are available, the denoising method is ``dwidenoise`` or
``dwidenoise2``, and ``--mrtrix-version dev`` is in effect, the complex-valued data are
carried through unringing rather than being reduced to magnitude immediately after
denoising. ``mrdegibbs`` is based on the Fourier shift theorem and operates on
complex-valued data directly, but only on MRtrix3's development branch; the released
``mrdegibbs`` reads magnitude data only. The ``rpg`` unringing method works on magnitude
data under either version.

``--denoise-method dwidenoise2`` is unaffected by ``--mrtrix-version``. ``dwidenoise2``
exists only on the development branch, so it always runs from that installation, while
every other MRtrix3 command follows ``--mrtrix-version``.
```

- [ ] **Step 12: Check lint and formatting**

Run:
```bash
micromamba run -n linc311 ruff check qsiprep/workflows/dwi/merge.py qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_n4_robustness.py
micromamba run -n linc311 ruff format --check qsiprep/workflows/dwi/merge.py qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_n4_robustness.py
```
Expected: no findings. Watch for E501 at 99 characters; the parameterized test bodies are close to the limit.

- [ ] **Step 13: Commit**

```bash
git add qsiprep/workflows/dwi/merge.py qsiprep/workflows/dwi/finalize.py \
  qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_n4_robustness.py docs/preprocessing.rst
git commit -m "Route complex data through mrdegibbs only on the development branch"
```

---

## Task 4: Report the version, and warn when it is unreleased

**Files:**
- Modify: `qsiprep/interfaces/reports.py:36-54` (subject templates), `:73-79` (`ABOUT_TEMPLATE`), `:136-145` (`SubjectSummaryInputSpec`), `:190-201` (`SubjectSummary._generate_segment`), `:260-275` (`AboutSummary`)
- Modify: `qsiprep/workflows/base.py:262-272`
- Test: `qsiprep/tests/test_reports.py`

**Interfaces:**
- Consumes: `config.workflow.mrtrix_version` and `config.environment.mrtrix3_home` (Task 1).
- Produces: `SubjectSummary(mrtrix_version=...)` and `AboutSummary(mrtrix_version=..., mrtrix3_home=...)`.

**Background:** Bootstrap alert classes render in the assembled report — `qsiprep/data/reports-spec.yml:281` already uses `class="alert alert-info" role="alert"`. `SubjectSummary` is instantiated unconditionally at `qsiprep/workflows/base.py:262`, and its reportlet lands in the **Summary** section, second from the top of the report.

- [ ] **Step 1: Write the failing report tests**

Add to `qsiprep/tests/test_reports.py`:

```python
def test_subject_summary_warns_about_the_development_branch():
    """Put an unmissable warning in the report when unreleased MRtrix3 was used."""
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        subject_id='01',
        template='MNI152NLin2009cAsym',
        mrtrix_version='dev',
    )
    segment = summary._generate_segment()

    assert 'alert alert-warning' in segment
    assert 'role="alert"' in segment
    assert 'development branch' in segment.lower()
    assert '--mrtrix-version dev' in segment


def test_subject_summary_is_quiet_under_stable_mrtrix():
    """Show no warning banner for a released MRtrix3, which is the default."""
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        subject_id='01',
        template='MNI152NLin2009cAsym',
        mrtrix_version='stable',
    )
    segment = summary._generate_segment()

    assert 'alert' not in segment
    assert 'development branch' not in segment.lower()


def test_about_summary_records_the_mrtrix_installation():
    """State which MRtrix3 ran, warning or not."""
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='dev',
        mrtrix3_home='/opt/mrtrix3-dev',
    )._generate_segment()

    assert 'dev' in segment
    assert '/opt/mrtrix3-dev' in segment


def test_about_summary_omits_an_unknown_mrtrix_path():
    """Report the version alone when the platform declares no installation paths."""
    from qsiprep.interfaces.reports import AboutSummary

    segment = AboutSummary(
        version='1.2.3',
        command='qsiprep ...',
        mrtrix_version='stable',
    )._generate_segment()

    assert 'MRtrix3' in segment
    assert 'stable' in segment
    assert 'None' not in segment
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_reports.py -k "mrtrix or development_branch or stable_mrtrix" -v`
Expected: FAIL — `TraitError` on the unknown `mrtrix_version` input.

- [ ] **Step 3: Add the warning and the provenance line**

In `qsiprep/interfaces/reports.py`, add this module-level constant immediately after `GROUPING_TEMPLATE`:

```python
MRTRIX_DEV_WARNING = """\t<div class="alert alert-warning" role="alert">
\t\t<strong>Development-branch MRtrix3.</strong>
\t\tThis run used the MRtrix3 development branch rather than a released version, at
\t\tyour request via <code>--mrtrix-version dev</code>. Development-branch code has not
\t\tbeen through a release cycle and may contain bugs. Inspect these outputs before
\t\trelying on them.
\t</div>
"""
```

Add a `{mrtrix_warning}` slot as the first line of `SUBJECT_TEMPLATE` (line 36):

```python
SUBJECT_TEMPLATE = """{mrtrix_warning}\t<ul class="elem-desc">
```

Extend `ABOUT_TEMPLATE` with a version line:

```python
ABOUT_TEMPLATE = """\t<ul>
\t\t<li>qsiprep version: {version}</li>
\t\t<li>qsiprep command: <code>{command}</code></li>
\t\t<li>MRtrix3: {mrtrix3}</li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""
```

Add to `SubjectSummaryInputSpec`:

```python
    mrtrix_version = traits.Enum(
        'stable', 'dev', usedefault=True, desc='which MRtrix3 installation was used'
    )
```

In `SubjectSummary._generate_segment`, add `mrtrix_warning` to the `SUBJECT_TEMPLATE.format(...)` call:

```python
        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1s=len(self.inputs.t1w),
            t2w=t2w_seg,
            n_dwis=len(input_files),
            n_outputs=n_outputs,
            groupings=groupings,
            output_spaces=['ACPC', self.inputs.template],
            mrtrix_warning=(
                MRTRIX_DEV_WARNING if self.inputs.mrtrix_version == 'dev' else ''
            ),
        )
```

Add to `AboutSummaryInputSpec`:

```python
    mrtrix_version = traits.Enum(
        'stable', 'dev', usedefault=True, desc='which MRtrix3 installation was used'
    )
    mrtrix3_home = Str(desc='path of the MRtrix3 installation that was used')
```

and rewrite `AboutSummary._generate_segment`:

```python
    def _generate_segment(self):
        mrtrix3 = self.inputs.mrtrix_version
        if isdefined(self.inputs.mrtrix3_home) and self.inputs.mrtrix3_home:
            mrtrix3 = f'{mrtrix3} ({self.inputs.mrtrix3_home})'

        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            mrtrix3=mrtrix3,
            date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        )
```

`isdefined` and `Str` are already imported in that module.

Leave `SUBJECT_SESSION_ANAT_TEMPLATE` alone. It is defined at line 46 and formatted
nowhere in the repository — adding a `{mrtrix_warning}` slot to a template nobody fills
would plant a `KeyError` for whoever revives it. Only `SUBJECT_TEMPLATE` is live.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_reports.py -v`
Expected: all PASSED, including the pre-existing `test_subject_summary_counts_inputs_uniquely`, which exercises `SUBJECT_TEMPLATE.format` and would raise `KeyError: 'mrtrix_warning'` if the new slot were left unfilled.

- [ ] **Step 5: Feed the version in from the workflow**

In `qsiprep/workflows/base.py`, replace the `summary` and `about` node constructions (lines 262-272):

```python
    summary = pe.Node(
        SubjectSummary(
            template=anatomical_template,
            mrtrix_version=config.workflow.mrtrix_version,
        ),
        name='summary',
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(
            version=config.environment.version,
            command=' '.join(sys.argv),
            mrtrix_version=config.workflow.mrtrix_version,
            mrtrix3_home=config.environment.mrtrix3_home or '',
        ),
        name='about',
        run_without_submitting=True,
    )
```

- [ ] **Step 6: Run the report and workflow tests**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_reports.py qsiprep/tests/test_workflows_merge.py -v`
Expected: all PASSED.

- [ ] **Step 7: Check lint and formatting**

Run:
```bash
micromamba run -n linc311 ruff check qsiprep/interfaces/reports.py qsiprep/workflows/base.py qsiprep/tests/test_reports.py
micromamba run -n linc311 ruff format --check qsiprep/interfaces/reports.py qsiprep/workflows/base.py qsiprep/tests/test_reports.py
```
Expected: no findings.

- [ ] **Step 8: Commit**

```bash
git add qsiprep/interfaces/reports.py qsiprep/workflows/base.py qsiprep/tests/test_reports.py
git commit -m "Warn in the report when the MRtrix3 development branch was used"
```

---

## Task 5: Ship both MRtrix3 installations in the container

**Files:**
- Modify: `/mnt/c/Users/tsalo/Documents/linc/qsiprep_build/Dockerfile_MRtrix3dev`
- Modify: `Dockerfile.base:4`, `:15`, `:38-46`
- Modify: `Dockerfile:1`, `:38-46`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: an image declaring `MRTRIX3_STABLE_HOME=/opt/mrtrix3-stable` and `MRTRIX3_DEV_HOME=/opt/mrtrix3-dev`, which `config.workflow.init()` (Task 1) reads.

**Background:** `Dockerfile_MRtrix3` is deliberately untouched. QSIRecon builds from `pennlinc/qsiprep-mrtrix3`, so reusing `26.1.0` as-is keeps the cross-repo blast radius at zero. Its classic `./configure && ./build` install is already self-contained — `main` puts `/opt/mrtrix3-latest/bin` on `PATH` with no `LD_LIBRARY_PATH` entry at all. The development image is the one that currently needs a global `LD_LIBRARY_PATH`, which becomes a shadowing hazard once two trees are present.

`26.9.0` is already published on Docker Hub, built from the pre-RPATH source (its image
config still carries a global `LD_LIBRARY_PATH`, which the RPATH change deliberately
drops) — re-tagging it is not an option. The RPATH build ships as `26.9.1` instead.

**Note:** Docker was not available in this WSL distro when the plan was written (`docker` was not found on `PATH`). If it is still unavailable, stop and report rather than guessing — every step here needs a real build.

- [ ] **Step 1: Make the development install self-contained**

In `qsiprep_build/Dockerfile_MRtrix3dev`, add the RPATH options to the `cmake -B build` invocation:

```dockerfile
RUN cmake -B build -GNinja \
          -DMRTRIX_BUILD_GUI=OFF \
          -DMRTRIX_ENABLE_GPU=OFF \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
          -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
          --preset=release && \
    cmake --build build && \
    cmake --install build --prefix /opt/mrtrix3
```

Delete the `LD_LIBRARY_PATH` from its `ENV`, leaving:

```dockerfile
ENV PATH="/opt/mrtrix3/bin:$PATH"
```

Replace the final verification `RUN` with one that clears `LD_LIBRARY_PATH`, since that is what proves the RPATH took effect:

```dockerfile
# Run with LD_LIBRARY_PATH cleared: these binaries must find their own libraries
# through RPATH, because the QSIPrep image carries two MRtrix3 installations and a
# global library path would let one shadow the other.
RUN env -u LD_LIBRARY_PATH sh -c '\
        mrdegibbs -help | grep -q dimensionality && \
        dwidenoise -version && \
        dwidenoise2 -version && \
        dwibiascorrect ants -help > /dev/null && \
        mrcalc -version && \
        mrtransform -version && \
        transformconvert -version' && \
    test -d /opt/mrtrix3/share/mrtrix3/dwidenoise2 && \
    test -s /opt/mrtrix3/share/licenses/mrtrix3-LICENCE.txt && \
    test -s /opt/mrtrix3/share/licenses/dwidenoise2-LICENSE
```

- [ ] **Step 2: Build the development image and verify the RPATH holds**

Run:
```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
docker build -f Dockerfile_MRtrix3dev -t pennlinc/qsiprep-mrtrix3dev:26.9.1 .
```
Expected: the build succeeds, including the verification `RUN`.

If it fails with a missing-shared-object error, the RPATH approach did not hold. The fallback recorded in the spec is a thin wrapper script per tree that sets `LD_LIBRARY_PATH` before exec'ing the real binary. Report the failure before implementing the fallback — it changes the shape of Task 6's assertions.

- [ ] **Step 3: Commit the development image change**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
git add Dockerfile_MRtrix3dev
git commit -m "Make the development-branch MRtrix3 install self-contained"
```

Stage that one file **by name**. `git add -A` in this repository would commit a 4015-line CRLF diff across all 24 tracked files, including `Dockerfile_MRtrix3`, which QSIRecon builds from.

- [ ] **Step 4: Copy both trees into the base image**

In `Dockerfile.base`, restore the released image's build ARG and stage. Change line 4 from a single ARG to:

```dockerfile
ARG TAG_MRTRIX3=26.1.0
ARG TAG_MRTRIX3DEV=26.9.1
```

Change the single MRtrix stage (line 15) to two:

```dockerfile
FROM pennlinc/qsiprep-mrtrix3:${TAG_MRTRIX3} AS build_mrtrix3
FROM pennlinc/qsiprep-mrtrix3dev:${TAG_MRTRIX3DEV} AS build_mrtrix3dev
```

Replace the MRtrix3 copy and `ENV` block (lines 38-46) with:

```dockerfile
## MRtrix3 — a released version and the development branch, side by side.
## --mrtrix-version selects between them at runtime by reordering PATH.
COPY --from=build_mrtrix3    /opt/mrtrix3-latest /opt/mrtrix3-stable
COPY --from=build_mrtrix3dev /opt/mrtrix3        /opt/mrtrix3-dev
## MRtrix3-3Tissue
COPY --from=build_3tissue /opt/3Tissue /opt/3Tissue
# Neither tree contributes to LD_LIBRARY_PATH: each finds its own libraries, so one
# cannot shadow the other. /opt/3Tissue ships its own 3.0.x mrdegibbs and dwidenoise
# and must stay last. This order is the --mrtrix-version stable default.
ENV MRTRIX3_STABLE_HOME="/opt/mrtrix3-stable" \
    MRTRIX3_DEV_HOME="/opt/mrtrix3-dev" \
    PATH="$PATH:/opt/mrtrix3-stable/bin:/opt/mrtrix3-dev/bin:/opt/3Tissue/bin" \
    MRTRIX3_DEPS="bzip2 ca-certificates curl libpng16-16 libblas3 liblapack3"
```

- [ ] **Step 5: Build the base image**

Run:
```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
docker build -f Dockerfile.base -t pennlinc/qsiprep-base:20260903 .
```
Expected: build succeeds.

- [ ] **Step 6: Pin command resolution in the application image**

In `Dockerfile`, change line 1:

```dockerfile
ARG BASE_IMAGE=pennlinc/qsiprep-base:20260903
```

Replace the verification `RUN` (lines 40-46) with:

```dockerfile
# Pin which tree each command resolves to. The default PATH order matches
# --mrtrix-version stable; config.workflow.init() reorders it for --mrtrix-version dev.
# dwidenoise2 exists only in the development tree, so it must fall through to it.
RUN test "$(command -v mrdegibbs)"      = "/opt/mrtrix3-stable/bin/mrdegibbs" && \
    test "$(command -v dwidenoise)"     = "/opt/mrtrix3-stable/bin/dwidenoise" && \
    test "$(command -v dwibiascorrect)" = "/opt/mrtrix3-stable/bin/dwibiascorrect" && \
    test "$(command -v dwidenoise2)"    = "/opt/mrtrix3-dev/bin/dwidenoise2" && \
    /opt/mrtrix3-dev/bin/mrdegibbs -help | grep -q dimensionality && \
    /opt/mrtrix3-stable/bin/dwibiascorrect ants -help > /dev/null && \
    test -d /opt/mrtrix3-dev/share/mrtrix3/dwidenoise2
```

The last three lines carry the weight: the first proves the development tree runs with no global library path, the second proves the released tree's Python wrapper still imports from its own `../lib`, and the third proves the `dwidenoise2` schedules travelled.

- [ ] **Step 7: Build the application image**

Run:
```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
docker build --target test -t qsiprep:test .
```
Expected: build succeeds, verification `RUN` included.

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
git add Dockerfile.base Dockerfile
git commit -m "Ship a released and a development MRtrix3 side by side"
```

---

## Task 6: Prove both versions work inside the container

**Files:**
- Test: `qsiprep/tests/test_workflows_merge.py`, `qsiprep/tests/test_interfaces_mrtrix.py`

**Interfaces:**
- Consumes: `_run_denoising_wf(..., mrtrix_version=...)` (Task 3); the two-tree image (Task 5).
- Produces: nothing later tasks depend on.

**Background:** These are the only tests that exercise real MRtrix3 binaries. Everything before this checks wiring and argument strings. The `stable` bias-correction test is the sole guard against the `-ants.b`/`-ants_b` break, which no in-process test can catch.

These tests need the container and the `nibs` test data, so they run through the image, not in `linc311`.

- [ ] **Step 1: Write the failing container tests**

Add to `qsiprep/tests/test_workflows_merge.py`, immediately after `test_denoising_wf_complex_mrdegibbs`:

```python
def test_denoising_wf_stable_mrdegibbs(monkeypatch, tmp_path, nibs_dwi):
    """Unring magnitude data with the released mrdegibbs and keep the result real.

    The graph-shape tests check that the split precedes unringing; this one checks
    that the binary the reordered PATH selects actually accepts what it is given.
    """
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method='dwidenoise',
        use_phase=True,
        unringing_method='mrdegibbs',
        mrtrix_version='stable',
    )

    degibbser = nodes['degibbser']
    degibbs_in = nb.load(degibbser.inputs.in_file)
    assert not np.issubdtype(degibbs_in.header.get_data_dtype(), np.complexfloating)

    degibbs_out = nb.load(degibbser.result.outputs.out_file)
    assert not np.issubdtype(degibbs_out.header.get_data_dtype(), np.complexfloating)
    assert degibbs_out.shape == degibbs_in.shape

    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])
```

Add to `qsiprep/tests/test_interfaces_mrtrix.py`:

```python
@pytest.mark.parametrize('mrtrix_version', ['stable', 'dev'])
def test_dwibiascorrect_options_are_accepted_by_the_real_binary(
    monkeypatch, tmp_path, mrtrix_version
):
    """Ask the selected dwibiascorrect whether it knows the options QSIPrep passes.

    This is the only test that can catch the -ants.b/-ants_b break, because it needs
    a real MRtrix3. -help is enough: MRtrix3 rejects unknown options during parsing.
    """
    import os
    import subprocess

    from qsiprep import config

    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    config.workflow.init()

    separator = '_' if mrtrix_version == 'dev' else '.'
    result = subprocess.run(
        ['dwibiascorrect', 'ants', f'-ants{separator}b', '[150,3]', '-help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
docker run --rm -v "$PWD":/src -w /src qsiprep:test \
  python -m pytest qsiprep/tests/test_workflows_merge.py::test_denoising_wf_stable_mrdegibbs \
  qsiprep/tests/test_interfaces_mrtrix.py -k "stable_mrdegibbs or real_binary" -v
```
Expected: FAIL. Confirm the *reason* before proceeding — a failure from missing `--data_dir` for the `nibs_dwi` fixture is an environment problem, not the behavior under test. Consult `qsiprep/tests/conftest.py:49` for how that fixture is supplied and pass the same option the existing container tests use.

- [ ] **Step 3: Fix whatever the tests reveal**

There is no new production code in this task. If the tests fail on behavior rather than setup, the defect is in Task 1, 2, 3 or 5, and the fix belongs there. Record which task and fix it there.

- [ ] **Step 4: Run the tests to verify they pass**

Run the command from Step 2.
Expected: 3 PASSED — one workflow test and two parameterized interface tests.

- [ ] **Step 5: Run the complete container suite**

Run:
```bash
docker run --rm -v "$PWD":/src -w /src qsiprep:test \
  python -m pytest qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_interfaces_mrtrix.py -v
```
Expected: no failures beyond the pre-existing, environment-gated ones recorded in Task 3 Step 6. Compare against that list; any new failure is a regression.

- [ ] **Step 6: Run the full non-container suite**

Run: `micromamba run -n linc311 pytest qsiprep/ -q`
Expected: the same failing set as on the merge base. The known environment-gated failures come from a missing `--data_dir` and missing FreeSurfer binaries; they are not caused by this work. Capture the counts.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/tests/test_workflows_merge.py qsiprep/tests/test_interfaces_mrtrix.py
git commit -m "Exercise both MRtrix3 installations against real binaries"
```

---

## Verification

After all six tasks:

1. `micromamba run -n linc311 pytest qsiprep/ -q` — compare the failing set against the merge base, which must be identical.
2. `micromamba run -n linc311 ruff check qsiprep/ && micromamba run -n linc311 ruff format --check qsiprep/` — clean.
3. `git diff main...HEAD -- docs/changes.md | wc -l` — must print `0`.
4. `docker build --target test -t qsiprep:test .` — succeeds, verification `RUN` included.
5. A default run (no `--mrtrix-version`) resolves `mrdegibbs`, `dwidenoise` and `dwibiascorrect` to `/opt/mrtrix3-stable/bin`, and `dwidenoise2` to `/opt/mrtrix3-dev/bin`.

## Release order

`26.9.0` is already pushed to `qsiprep_build` and `pennlinc/qsiprep-mrtrix3dev:26.9.0` is
published on Docker Hub, built from the pre-RPATH source. The order matters because
`Dockerfile.base` pulls its stages from the registry, and QSIPrep's own CI publishes the
base image automatically rather than as a separate manual step:

1. Merge `mrtrix3-dev` into `main` in `qsiprep_build`.
2. Tag `26.9.1` on GitHub. CircleCI's `dpkg --compare-versions` gate compares the pinned
   `required_tag` against the git tag; only the `mrtrix3dev` job's pin is behind `26.9.1`,
   so only it runs, pushing `pennlinc/qsiprep-mrtrix3dev:26.9.1`. Every other image job
   halts (pins are `26.1.x`-`26.8.x`); there is no AFNI build job, since that image is
   third-party.
3. Open the QSIPrep PR. `.circleci/continue_config.yml` derives `BASE_IMAGE` from
   `Dockerfile`'s `ARG BASE_IMAGE=` line, finds `pennlinc/qsiprep-base:20260903` missing
   via `docker manifest inspect`, and builds it from `Dockerfile.base` with `--pull`
   (pulling `qsiprep-mrtrix3dev:26.9.1` and `qsiprep-mrtrix3:26.1.0`), pushes it, then
   builds the application image and runs the suite.

`pennlinc/qsiprep-mrtrix3:26.1.0` already exists and is unchanged. Step 3 is the first
genuine end-to-end build of `Dockerfile.base` — it could not be built locally because a
corporate firewall intercepts container TLS to `public.boxcloud.com`, breaking an
unrelated download in an untouched `RUN` block.

## Open items carried from the spec

Both resolved:

- RPATH: `-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'` works. `readelf` reports `RUNPATH
  [$ORIGIN:$ORIGIN/../lib]` and the binaries run with `LD_LIBRARY_PATH` cleared. No
  wrapper-script fallback is needed.
- Released MRtrix3 version: SHA `670e7b06` is MRtrix3 3.0.4, confirmed by `mrinfo
  -version` and `lib/mrtrix3/_version.py` inside `pennlinc/qsiprep-mrtrix3:26.1.0` (not
  3.0.8; the development branch reports `3.0.8-2071-gb98b54e9`). Both values are baked
  into `Dockerfile.base` as `MRTRIX3_STABLE_VERSION`/`MRTRIX3_DEV_VERSION` and asserted at
  build time against `mrinfo -version`.
