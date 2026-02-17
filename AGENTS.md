# AGENTS.md -- QSIPrep

This file provides instructions for AI coding agents and human maintainers working on **QSIPrep**, a BIDS App for preprocessing and reconstructing q-space (diffusion) MRI images.

---

## Shared Instructions (All PennLINC BIDS Apps)

The following conventions apply equally to **qsiprep**, **qsirecon**, **xcp_d**, and **aslprep**. All four are PennLINC BIDS Apps built on the NiPreps stack.

### Ecosystem Context

- These projects belong to the [NiPreps](https://www.nipreps.org/) ecosystem and follow its community guidelines.
- Core dependencies include **nipype** (workflow engine), **niworkflows** (reusable workflow components), **nireports** (visual reports), **pybids** (BIDS dataset querying), and **nibabel** (neuroimaging I/O).
- All four apps are containerized via Docker and distributed on Docker Hub under the `pennlinc/` namespace.
- Contributions follow the [NiPreps contributing guidelines](https://www.nipreps.org/community/CONTRIBUTING/).

### Architecture Overview

Every PennLINC BIDS App follows this execution flow:

```
CLI (parser.py / run.py)
  -> config singleton (config.py, serialized as ToML)
    -> workflow graph construction (workflows/*.py)
      -> Nipype interfaces (interfaces/*.py)
        -> BIDS-compliant derivative outputs
```

- **CLI** (`<pkg>/cli/`): `parser.py` defines argparse arguments; `run.py` is the entry point; `workflow.py` builds the execution graph; `version.py` handles `--version`.
- **Config** (`<pkg>/config.py`): A singleton module with class-based sections (`environment`, `execution`, `workflow`, `nipype`, `seeds`). Config is serialized to ToML and passed across processes via the filesystem. Access settings as `config.section.setting`.
- **Workflows** (`<pkg>/workflows/`): Built using `nipype.pipeline.engine` (`pe.Workflow`, `pe.Node`, `pe.MapNode`). Use `LiterateWorkflow` from `niworkflows.engine.workflows` for auto-documentation. Every workflow factory function must be named `init_<descriptive_name>_wf`.
- **Interfaces** (`<pkg>/interfaces/`): Custom Nipype interfaces wrapping external tools or Python logic. Follow standard Nipype patterns: define `_InputSpec` / `_OutputSpec` with `BaseInterfaceInputSpec` / `TraitedSpec`, implement `_run_interface()`.
- **Utilities** (`<pkg>/utils/`): Shared helper functions. BIDS-specific helpers live in `utils/bids.py`.
- **Reports** (`<pkg>/reports/`): HTML report generation using nireports.
- **Data** (`<pkg>/data/`): Static package data (config files, templates, atlases). Accessed via `importlib.resources` or the `acres` package.
- **Tests** (`<pkg>/tests/`): Pytest-based. Unit tests run without external data. Integration tests are gated behind pytest markers and are skipped by default.

### Workflow Authoring Rules

1. Every workflow factory function must be named `init_<name>_wf` and return a `Workflow` object.
2. Use `LiterateWorkflow` (from `niworkflows.engine.workflows`) to enable automatic workflow graph documentation.
3. Define `inputnode` and `outputnode` as `niu.IdentityInterface` nodes to declare the workflow's external API.
4. Connect nodes using `workflow.connect([(source, dest, [('out_field', 'in_field')])])` syntax.
5. Add `# fmt:skip` after multi-line `workflow.connect()` calls to prevent ruff from reformatting them.
6. Include a docstring with `Workflow Graph` and `.. workflow::` Sphinx directive for auto-generated documentation.
7. Use `config` module values (not function parameters) for global settings inside workflow builders.

### Interface Conventions

1. Input/output specs use Nipype traits (`File`, `traits.Bool`, `traits.Int`, etc.).
2. `mandatory = True` for required inputs; provide `desc=` for all traits.
3. Implement `_run_interface(self, runtime)` -- never `run()`.
4. Return `runtime` from `_run_interface`.
5. Set outputs via `self._results['field'] = value`.

### Config Module Usage

```python
from <pkg> import config

# Read a setting
work_dir = config.execution.work_dir

# Serialize to disk
config.to_filename(path)

# Load from disk (in a subprocess)
config.load(path)
```

The config module is the single source of truth for runtime parameters. Never pass global settings as function arguments when they are available via config.

### Testing Conventions

- **Unit tests**: Files named `test_*.py` in `<pkg>/tests/`. Must not require external neuroimaging data or network access.
- **Integration tests**: Decorated with `@pytest.mark.<marker_name>`. Excluded by default via `addopts` in `pyproject.toml`. Require Docker or pre-downloaded test datasets.
- **Fixtures**: Defined in `conftest.py`. Common fixtures include `data_dir`, `working_dir`, `output_dir`, and `datasets`.
- **Coverage**: Configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`.

### Documentation

- Built with Sphinx using `sphinx_rtd_theme`.
- Source files in `docs/`.
- Workflow graphs are auto-rendered via `.. workflow::` directives that call `init_*_wf` functions.
- API docs generated via `sphinxcontrib-apidoc`.
- Bibliography managed with `sphinxcontrib-bibtex` and `boilerplate.bib`.

### Docker

- Each app has a custom base image: `pennlinc/<pkg>_build:<version>`.
- The `Dockerfile` installs the app via `pip install` into the base image.
- Entrypoint is the CLI command (e.g., `/opt/conda/envs/<pkg>/bin/<pkg>`).
- Labels follow the `org.label-schema` convention.

### Release Process

- Versions are derived from git tags via `hatch-vcs` (VCS-based versioning).
- GitHub Releases use auto-generated changelogs configured in `.github/release.yml`.
- Release categories: Breaking Changes, New Features, Deprecations, Bug Fixes, Other.
- Docker images are built and pushed via CI on tagged releases.

### Code Style

- **Formatter**: `ruff format` (target: all four repos).
- **Linter**: `ruff check` with an extended rule set (F, E, W, I, UP, YTT, S, BLE, B, A, C4, DTZ, T10, EXE, FA, ISC, ICN, PT, Q).
- **Import sorting**: Handled by ruff's `I` rule (isort-compatible).
- **Pre-commit**: Uses `ruff-pre-commit` hooks for both linting and formatting.
- **Black is disabled**: `[tool.black] exclude = ".*"` in repos that have migrated to ruff.

### BIDS Compliance

- All outputs must conform to the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) specification.
- Use `pybids.BIDSLayout` for querying input datasets.
- Use `DerivativesDataSink` (from the project's interfaces or niworkflows) for writing BIDS-compliant output files.
- Entity names, suffixes, and extensions must match the BIDS specification.

---

## QSIPrep-Specific Instructions

### Project Overview

QSIPrep is a BIDS App for preprocessing diffusion MRI (dMRI/DWI) data. It handles:
- Anatomical preprocessing (brain extraction, segmentation, normalization)
- DWI preprocessing (denoising, motion correction, eddy current correction, susceptibility distortion correction)
- Fieldmap-based and fieldmap-less distortion correction
- Head motion and eddy current correction via FSL's `eddy` or SHORELine
- DRBUDDI and TOPUP-based distortion correction
- Intramodal template construction for multi-session data
- Confound estimation (framewise displacement, etc.)

### Repository Details

| Item | Value |
|------|-------|
| Package name | `qsiprep` |
| Default branch | `master` |
| Entry point | `qsiprep.cli.run:main` |
| Python requirement | `>=3.10` |
| Build backend | hatchling + hatch-vcs + cython + numpy |
| Linter | ruff ~= 0.4.3 |
| Pre-commit | Yes (ruff v0.6.2) |
| Tox | Yes |
| Docker base | `pennlinc/qsiprep_build:<ver>` |
| Dockerfile | Multi-stage wheel build |

### Key Directories

- `qsiprep/workflows/dwi/`: DWI preprocessing workflow modules (motion correction, distortion correction, resampling, confounds)
- `qsiprep/workflows/anatomical/`: Anatomical preprocessing
- `qsiprep/workflows/fieldmap/`: Fieldmap processing (PEPOLAR, phase-difference, SyN, DRBUDDI)
- `qsiprep/interfaces/`: Nipype interfaces wrapping FSL eddy, MRtrix3, DSI Studio, DIPY, ANTs, FreeSurfer, TORTOISE
- `qsiprep/utils/maths.pyx`: Cython extension for performance-critical math operations

### Linting Notes

QSIPrep currently uses ruff ~= 0.4.3 (older than xcp_d/aslprep). There are 13 suppressed lint rules marked with `# TODO: Fix these` in `pyproject.toml`:
- `S605`, `DTZ005`, `B904`, `A001`, `B006`, `S607`, `S108`, `S602`, `E402`, `UP028`, `UP031`, `BLE001`

These should be addressed incrementally. When fixing code that triggers these rules, remove the corresponding ignore entry.

---

## Cross-Project Development Roadmap

This roadmap covers harmonization work across all four PennLINC BIDS Apps (qsiprep, qsirecon, xcp_d, aslprep) to reduce maintenance burden.

### Phase 1: Bring qsirecon to parity

1. **Migrate qsirecon from flake8+black+isort to ruff** -- copy the `[tool.ruff]` config from xcp_d's `pyproject.toml` and remove `[tool.black]`, `[tool.isort]`, `[tool.flake8]` sections.
2. **Add `.pre-commit-config.yaml` to qsirecon** -- identical to the config used by qsiprep, xcp_d, and aslprep.
3. **Add `tox.ini` to qsirecon** -- copy from qsiprep or xcp_d (they are identical).
4. **Add `.github/dependabot.yml` to qsirecon**.
5. **Reformat qsirecon codebase** -- run `ruff format` to switch from double quotes to single quotes.

### Phase 2: Standardize across all four repos

6. **Rename qsiprep default branch** from `master` to `main` and update `.github/workflows/lint.yml`.
7. **Rename aslprep test extras** from `test` to `tests` for consistency with the other three repos.
8. **Converge on version management** -- recommend the simpler `_version.py` direct-import pattern (used by qsiprep/qsirecon). Migrate xcp_d and aslprep away from `__about__.py`.
9. **Pin the same ruff version** in all four repos' dev dependencies and `.pre-commit-config.yaml`.
10. **Harmonize ruff ignore lists** -- adopt xcp_d's minimal set (`S105`, `S311`, `S603`) as the target; fix suppressed rules in qsiprep and aslprep incrementally.

### Phase 3: Shared infrastructure

11. **Extract a reusable GitHub Actions workflow** for lint + codespell + build checks, hosted in a shared repo (e.g., `PennLINC/.github`).
12. **Standardize Dockerfile patterns** -- adopt multi-stage wheel builds (as qsiprep does) across all four repos.
13. **Create a shared `pennlinc-style` package or cookiecutter template** providing `pyproject.toml` lint/test config, `.pre-commit-config.yaml`, `tox.ini`, and CI workflows.
14. **Evaluate `nipreps-versions` calver** -- the `raw-options = { version_scheme = "nipreps-calver" }` line is commented out in all four repos. Decide whether to adopt it.

