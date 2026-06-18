# TORTOISE DIFFPREP HMC Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TORTOISE v4 DIFFPREP as a first-class `--hmc-model` head-motion-correction backend in qsiprep, selectable as `diffprep_motion`, `diffprep_quadratic`, or `diffprep_cubic`.

**Architecture:** A new `init_diffprep_hmc_wf` mirrors the inputnode/outputnode contract of `init_fsl_hmc_wf` (eddy) so `init_dwi_preproc_wf` can drop it in. The workflow converts denoised LPS+ DWI to TORTOISE format, runs the `TORTOISEProcess` binary in DIFFPREP mode (motion + optional quadratic/cubic eddy), then "bakes in" the correction: the TORTOISE-resampled DWI and TORTOISE-rotated gradients become the workflow outputs with identity per-volume affines (exactly how eddy behaves). Susceptibility correction is **not** done here — it stays with qsiprep's existing fieldmap/DRBUDDI machinery.

**Tech Stack:** Python, nipype (interfaces + workflows), niworkflows `LiterateWorkflow`, numpy/nibabel, TORTOISE v4 `TORTOISEProcess` binary (already in the qsiprep container), pytest.

## Global Constraints

- **Source spec:** `docs/superpowers/specs/2026-06-18-tortoise-diffprep-hmc-design.md` — every task implicitly serves it.
- **No new container dependency:** `TORTOISEProcess` is already on `PATH` in the qsiprep image (`Dockerfile.base` copies TORTOISE from `pennlinc/qsiprep-drbuddi`). Do **not** edit any Dockerfile.
- **Mirror existing patterns exactly:** copy the `eddy_config` / `--eddy-config` plumbing for `diffprep_config` / `--diffprep-config`; copy the eddy workflow's node vocabulary (`ConformDwi`, `SplitDWIsFSL`, `ExtractB0s`, `init_dwi_reference_wf`).
- **Orientation:** all DIFFPREP processing is in **LPS+** (matches the non-eddy `pre_hmc` orientation and SHORELine), so b0 templates align for downstream intramodal/template stages.
- **Transform model:** bake-in. `to_dwi_ref_affines` are identity; `sdc_method` is `'None'`; gradients come out TORTOISE-rotated. Do not re-rotate or re-apply downstream.
- **Three choices, suffix-driven:** `diffprep_motion` → rigid only; `diffprep_quadratic` → rigid + 24-param Okan eddy; `diffprep_cubic` → rigid + cubic eddy. `base.py` branches on `hmc_model.startswith('diffprep_')` and derives the order from the suffix.
- **Version control:** the repo owner manages commits. Each task ends with a **Checkpoint** (run the full test, leave the tree staged for review) — do **not** run `git commit` unless the owner asks.
- **Test runner:** `python -m pytest <path> -v` from the `qsiprep` repo root.

---

## File Structure

| File | Create/Modify | Responsibility |
| --- | --- | --- |
| `qsiprep/cli/parser.py` | Modify | Add 3 `--hmc-model` choices + `--diffprep-config` arg + validation call |
| `qsiprep/config.py` | Modify | `workflow.diffprep_config` field + execution `_paths` entry |
| `qsiprep/utils/misc.py` | Modify | `validate_diffprep_config` |
| `qsiprep/data/diffprep_params.json` | Create | Default TORTOISE knobs |
| `qsiprep/interfaces/tortoise.py` | Modify | `TORTOISEProcess`, `DIFFPREPMotionParams`, `BmatToFSLGradients`, `generate_diffprep_boilerplate` |
| `qsiprep/workflows/dwi/diffprep.py` | Create | `init_diffprep_hmc_wf` |
| `qsiprep/workflows/dwi/base.py` | Modify | Branch to `init_diffprep_hmc_wf` |
| `qsiprep/tests/test_interfaces_diffprep.py` | Create | Unit tests for the new interfaces + workflow construction |
| `qsiprep/tests/test_cli.py` | Modify | Parse test for new choices |
| `docs/usage.rst` | Modify | Document the three options |

---

## Task 1: CLI + config plumbing

**Files:**
- Modify: `qsiprep/cli/parser.py` (the `--hmc-model` argument, ~line 532; the `--eddy-config` argument, ~line 542; the validation block, ~line 699)
- Modify: `qsiprep/config.py` (execution `_paths` tuple ~line 456; workflow `eddy_config` field ~line 580)
- Modify: `qsiprep/utils/misc.py` (after `validate_eddy_config`, ~line 135)
- Create: `qsiprep/data/diffprep_params.json`
- Modify: `qsiprep/tests/test_cli.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `config.workflow.hmc_model` may now be one of `diffprep_motion|diffprep_quadratic|diffprep_cubic`; `config.workflow.diffprep_config` (str path or None); `qsiprep.utils.misc.validate_diffprep_config(path) -> None` (raises `ValueError`); data file resolvable via `qsiprep.data.load('diffprep_params.json')`.

- [ ] **Step 1: Write the failing test (parser accepts new choices + config field)**

Add to `qsiprep/tests/test_cli.py`:

```python
import pytest


@pytest.mark.parametrize(
    'model', ['diffprep_motion', 'diffprep_quadratic', 'diffprep_cubic']
)
def test_parser_accepts_diffprep_hmc_models(model, tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    bids = tmp_path / 'bids'
    bids.mkdir()
    out = tmp_path / 'out'
    opts = parser.parse_args(
        [str(bids), str(out), 'participant', '--hmc-model', model]
    )
    assert opts.hmc_model == model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_cli.py::test_parser_accepts_diffprep_hmc_models -v`
Expected: FAIL — argparse exits with "invalid choice: 'diffprep_motion'".

(If `_build_parser` import differs, grep `qsiprep/tests/test_cli.py` for how it builds the parser and match that idiom — but `_build_parser` is the public builder in `qsiprep/cli/parser.py`.)

- [ ] **Step 3: Add the choices and the `--diffprep-config` argument**

In `qsiprep/cli/parser.py`, change the `--hmc-model` choices:

```python
    g_moco.add_argument(
        '--hmc-model',
        action='store',
        default='eddy',
        choices=[
            'none',
            '3dSHORE',
            'eddy',
            'tensor',
            'diffprep_motion',
            'diffprep_quadratic',
            'diffprep_cubic',
        ],
        help='model used to generate target images for hmc. If "none" the '
        'non-b0 images will be warped using the same transform as their '
        'nearest b0 image. If "3dSHORE", SHORELine will be used. if "tensor", '
        'SHORELine iterations with a tensor model will be used. The '
        '"diffprep_*" options run TORTOISE DIFFPREP: "diffprep_motion" '
        'corrects rigid head motion only, "diffprep_quadratic" adds '
        '24-parameter quadratic eddy-current correction (recommended for '
        'non-shelled / CS-DSI schemes), "diffprep_cubic" adds cubic eddy '
        'correction. DIFFPREP works on arbitrary q-space (no shells '
        'required).',
    )
```

Directly after the `--eddy-config` argument block, add:

```python
    g_moco.add_argument(
        '--diffprep-config',
        action='store',
        help='path to a json file with settings for the call to TORTOISE '
        'DIFFPREP (used only when --hmc-model is one of the diffprep_* '
        'options). If no json is specified, a default one will be used. The '
        'current default can be found here: '
        'https://github.com/PennLINC/qsiprep/blob/main/qsiprep/data/diffprep_params.json',
    )
```

- [ ] **Step 4: Add the config field and path resolution**

In `qsiprep/config.py`, in the `execution` class `_paths` tuple (the one containing `'eddy_config'`), add `'diffprep_config'`:

```python
    _paths = (
        'bids_dir',
        'bids_database_dir',
        'dataset_links',
        'diffprep_config',
        'eddy_config',
        'layout',
        'log_dir',
        'output_dir',
        'templateflow_home',
        ...
    )
```

In the `workflow` class, directly below the `eddy_config` field (`eddy_config = None` / its docstring), add:

```python
    diffprep_config = None
    """Configuration JSON for running TORTOISE DIFFPREP."""
```

- [ ] **Step 5: Add `validate_diffprep_config` and wire validation in the parser**

In `qsiprep/utils/misc.py`, after `validate_eddy_config`:

```python
def validate_diffprep_config(diffprep_config):
    """Validate the DIFFPREP configuration file.

    Parameters
    ----------
    diffprep_config : str
        The path to the DIFFPREP configuration JSON file.

    Raises
    ------
    ValueError
        If the DIFFPREP configuration file does not exist or is not valid JSON.
    """
    import json
    import os

    if not os.path.exists(diffprep_config):
        raise ValueError(f'DIFFPREP configuration file {diffprep_config} does not exist.')
    with open(diffprep_config) as f:
        json.load(f)

    return
```

In `qsiprep/cli/parser.py`, next to the existing `if opts.eddy_config:` validation block (~line 699):

```python
    if opts.diffprep_config:
        from ..utils.misc import validate_diffprep_config

        validate_diffprep_config(opts.diffprep_config)
```

- [ ] **Step 6: Create the default config JSON**

Create `qsiprep/data/diffprep_params.json`:

```json
{
  "b0_id": -1,
  "is_human_brain": true,
  "rot_eddy_center": "isocenter",
  "center_of_mass": false,
  "will_be_drbuddied": false,
  "interpolation": "cubic",
  "output_type": "NIFTI_GZ"
}
```

- [ ] **Step 7: Write the failing test for `validate_diffprep_config`**

Add to `qsiprep/tests/test_cli.py`:

```python
def test_validate_diffprep_config_missing(tmp_path):
    from qsiprep.utils.misc import validate_diffprep_config

    with pytest.raises(ValueError, match='does not exist'):
        validate_diffprep_config(str(tmp_path / 'nope.json'))


def test_validate_diffprep_config_default_is_valid():
    from qsiprep.data import load as load_data
    from qsiprep.utils.misc import validate_diffprep_config

    validate_diffprep_config(str(load_data('diffprep_params.json')))
```

- [ ] **Step 8: Run all Task 1 tests**

Run: `python -m pytest qsiprep/tests/test_cli.py -k diffprep -v`
Expected: PASS (3 parametrized parser cases + 2 validate cases).

- [ ] **Step 9: Checkpoint**

Run: `python -m pytest qsiprep/tests/test_cli.py -v`
Expected: all CLI tests pass. Stage the changed files for the owner's review (do not commit).

---

## Task 2: `TORTOISEProcess` interface

**Files:**
- Modify: `qsiprep/interfaces/tortoise.py` (add after the existing `TORTOISEConvert` class, ~line 615)
- Modify: `qsiprep/tests/test_interfaces_diffprep.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `qsiprep.interfaces.tortoise.TORTOISEProcess` — a nipype `CommandLine` with inputs `dwi_file` (File, `.nii`), `bmtxt_file` (File), `mask_file` (File), `transformation_type` (Enum `'rigid'|'quadratic'|'cubic'`), `config_file` (File, optional), `b0_id` (Int, default -1); outputs `corrected_dwi` (File), `corrected_bmtxt` (File), `transforms_file` (File). The `_cmd` is `TORTOISEProcess`.

> **Flag-name verification (do this first):** run `TORTOISEProcess --help` inside the qsiprep container and confirm the argstrs below. TORTOISE option names have changed across point releases. The unit test in this task asserts *structural* properties (values appear in `cmdline`), so it stays green as long as the inputs are wired; if a flag name differs, fix the `argstr` strings to match `--help` and keep the test assertions about *values*.

- [ ] **Step 1: Write the failing test (cmdline construction)**

Create `qsiprep/tests/test_interfaces_diffprep.py`:

```python
import os

import numpy as np


def _write_dummy_nii(path):
    import nibabel as nb

    img = nb.Nifti1Image(np.zeros((4, 4, 4, 6), dtype='float32'), np.eye(4))
    img.to_filename(path)


def test_tortoiseprocess_cmdline(tmp_path):
    from qsiprep.interfaces.tortoise import TORTOISEProcess

    dwi = tmp_path / 'dwi.nii'
    _write_dummy_nii(str(dwi))
    bmtxt = tmp_path / 'dwi.bmtxt'
    bmtxt.write_text('0 0 0 0 0 0\n')
    mask = tmp_path / 'mask.nii'
    _write_dummy_nii(str(mask))

    iface = TORTOISEProcess(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        mask_file=str(mask),
        transformation_type='quadratic',
    )
    cmd = iface.cmdline
    assert cmd.startswith('TORTOISEProcess')
    assert 'dwi.nii' in cmd
    assert 'quadratic' in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_tortoiseprocess_cmdline -v`
Expected: FAIL — `ImportError: cannot import name 'TORTOISEProcess'`.

- [ ] **Step 3: Implement `TORTOISEProcess`**

In `qsiprep/interfaces/tortoise.py`, after `TORTOISEConvert` (the file already imports `CommandLine`, `CommandLineInputSpec`, `File`, `TraitedSpec`, `traits`, `isdefined`, `fname_presuffix`):

```python
class _TORTOISEProcessInputSpec(TORTOISEInputSpec):
    dwi_file = File(
        exists=True, mandatory=True, argstr='--input %s', desc='DWI in TORTOISE .nii format'
    )
    bmtxt_file = File(
        exists=True, mandatory=True, argstr='--input_bmtxt %s', desc='b-matrix (.bmtxt)'
    )
    mask_file = File(exists=True, mandatory=True, argstr='--mask %s', desc='brain mask')
    transformation_type = traits.Enum(
        'rigid',
        'quadratic',
        'cubic',
        usedefault=True,
        argstr='--DIFFPREP_transformation_type %s',
        desc='Okan transform order: rigid (motion only), quadratic, or cubic',
    )
    b0_id = traits.Int(-1, usedefault=True, argstr='--b0_id %d', desc='index of b=0 (-1=auto)')
    config_file = File(exists=True, argstr='--DIFFPREP_settings %s', desc='DIFFPREP settings JSON')
    do_drbuddi = traits.Bool(
        False, usedefault=True, argstr='--step DIFFPREP', desc='HMC-only: run DIFFPREP step only'
    )


class _TORTOISEProcessOutputSpec(TraitedSpec):
    corrected_dwi = File(exists=True, desc='motion/eddy corrected DWI (LPS+ TORTOISE space)')
    corrected_bmtxt = File(exists=True, desc='rotated b-matrix matching corrected_dwi')
    transforms_file = File(exists=True, desc='per-volume DIFFPREP transform parameters')


class TORTOISEProcess(TORTOISECommandLine):
    """Run TORTOISE v4 DIFFPREP (motion + optional eddy) on a single DWI series.

    HMC-only: susceptibility correction is left to qsiprep's fieldmap machinery.
    """

    input_spec = _TORTOISEProcessInputSpec
    output_spec = _TORTOISEProcessOutputSpec
    _cmd = 'TORTOISEProcess'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        cwd = os.getcwd()
        base = fname_presuffix(
            self.inputs.dwi_file, suffix='_proc', use_ext=False, newpath=cwd
        )
        outputs['corrected_dwi'] = base + '.nii'
        outputs['corrected_bmtxt'] = base + '.bmtxt'
        outputs['transforms_file'] = base + '_transformations.txt'
        return outputs
```

> Note: `_list_outputs` output paths must match what `TORTOISEProcess` actually writes in its working directory — reconcile against a real run during the flag-verification step and adjust the suffixes. The structural cmdline test does not depend on these paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_tortoiseprocess_cmdline -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py -v`
Expected: PASS. Stage for review.

---

## Task 3: `DIFFPREPMotionParams` interface

**Files:**
- Modify: `qsiprep/interfaces/tortoise.py` (add after `TORTOISEProcess`)
- Modify: `qsiprep/tests/test_interfaces_diffprep.py`

**Interfaces:**
- Consumes: `TORTOISEProcess.transforms_file` (per-volume transform text).
- Produces: `qsiprep.interfaces.tortoise.DIFFPREPMotionParams` — `SimpleInterface` with input `transforms_file` (File), output `motion_file` (File, a `.tsv`/`.txt` with one row per volume: `trans_x trans_y trans_z rot_x rot_y rot_z`, translations in mm, rotations in radians). The rotation/translation parameterisation is the rigid (first 6) of the 24 Okan parameters.

- [ ] **Step 1: Write the failing test**

Add to `qsiprep/tests/test_interfaces_diffprep.py`:

```python
def test_diffprep_motion_params(tmp_path):
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    # Two volumes, 6 rigid params each (tx ty tz rx ry rz), space-separated.
    transforms = tmp_path / 'xfms.txt'
    transforms.write_text('0 0 0 0 0 0\n1.5 -2.0 0.5 0.01 -0.02 0.0\n')

    iface = DIFFPREPMotionParams(transforms_file=str(transforms))
    res = iface.run(cwd=str(tmp_path))
    out = np.loadtxt(res.outputs.motion_file)

    assert out.shape == (2, 6)
    assert np.allclose(out[1], [1.5, -2.0, 0.5, 0.01, -0.02, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_diffprep_motion_params -v`
Expected: FAIL — `ImportError: cannot import name 'DIFFPREPMotionParams'`.

- [ ] **Step 3: Implement `DIFFPREPMotionParams`**

In `qsiprep/interfaces/tortoise.py` (uses `np`, already imported):

```python
class _DIFFPREPMotionParamsInputSpec(BaseInterfaceInputSpec):
    transforms_file = File(exists=True, mandatory=True, desc='per-volume DIFFPREP transforms')


class _DIFFPREPMotionParamsOutputSpec(TraitedSpec):
    motion_file = File(exists=True, desc='per-volume rigid motion params (mm, rad)')


class DIFFPREPMotionParams(SimpleInterface):
    """Extract the rigid 6-DOF motion parameters from a DIFFPREP transforms file.

    Each volume's Okan transform begins with the rigid component
    (3 translations in mm, 3 rotations in radians); the remaining
    eddy-current parameters are not reported as motion.
    """

    input_spec = _DIFFPREPMotionParamsInputSpec
    output_spec = _DIFFPREPMotionParamsOutputSpec

    def _run_interface(self, runtime):
        params = np.loadtxt(self.inputs.transforms_file, ndmin=2)
        rigid = params[:, :6]
        out_file = fname_presuffix(
            self.inputs.transforms_file,
            suffix='_motion.txt',
            use_ext=False,
            newpath=runtime.cwd,
        )
        np.savetxt(out_file, rigid, fmt='%.8f')
        self._results['motion_file'] = out_file
        return runtime
```

> Verification note: confirm the real DIFFPREP transforms-file layout during the Task 2 flag-verification run. If the per-volume rigid params are not the first 6 whitespace-separated columns, adjust the `params[:, :6]` slice (and the test's input fixture) to match the real format. The interface contract (6 columns out) is fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_diffprep_motion_params -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py -v`
Expected: PASS. Stage for review.

---

## Task 4: `BmatToFSLGradients` interface

**Files:**
- Modify: `qsiprep/interfaces/tortoise.py` (add after `DIFFPREPMotionParams`)
- Modify: `qsiprep/tests/test_interfaces_diffprep.py`

**Interfaces:**
- Consumes: `TORTOISEProcess.corrected_bmtxt` (rotated b-matrix).
- Produces: `qsiprep.interfaces.tortoise.BmatToFSLGradients` — `SimpleInterface` with input `bmtxt_file` (File), outputs `bval_file` (File `.bval`), `bvec_file` (File `.bvec`). Recovers FSL gradients from the TORTOISE b-matrix so downstream (split, confounds, resampling) consumes standard bval/bvec. The existing `make_bmat_file(bvals, bvecs)` is the forward direction; this is its inverse.

- [ ] **Step 1: Write the failing round-trip test**

Add to `qsiprep/tests/test_interfaces_diffprep.py`:

```python
def test_bmat_to_fsl_roundtrip(tmp_path):
    from qsiprep.interfaces.tortoise import BmatToFSLGradients, make_bmat_file

    # Known FSL gradients (b=0 + 3 weighted).
    bvals = tmp_path / 'in.bval'
    bvecs = tmp_path / 'in.bvec'
    bvals.write_text('0 1000 1000 2000\n')
    bvecs.write_text('0 1 0 0.7071\n0 0 1 0.7071\n0 0 0 0\n')

    bmtxt = make_bmat_file(str(bvals), str(bvecs))

    iface = BmatToFSLGradients(bmtxt_file=str(bmtxt))
    res = iface.run(cwd=str(tmp_path))

    out_bvals = np.loadtxt(res.outputs.bval_file, ndmin=1)
    out_bvecs = np.loadtxt(res.outputs.bvec_file, ndmin=2)

    assert np.allclose(out_bvals, [0, 1000, 1000, 2000], atol=1.0)
    # First weighted dir aligns with x; sign is arbitrary, compare abs.
    assert np.allclose(np.abs(out_bvecs[:, 1]), [1, 0, 0], atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_bmat_to_fsl_roundtrip -v`
Expected: FAIL — `ImportError: cannot import name 'BmatToFSLGradients'`.

- [ ] **Step 3: Implement `BmatToFSLGradients`**

In `qsiprep/interfaces/tortoise.py`. The TORTOISE `.bmtxt` has one row per volume with the 6 unique b-matrix entries `[Bxx, Bxy, Bxz, Byy, Byz, Bzz]` (s/mm², so factor 1000 vs s/m² may apply — verify against `make_bmat_file`; the round-trip test pins consistency regardless of the global scale).

```python
class _BmatToFSLGradientsInputSpec(BaseInterfaceInputSpec):
    bmtxt_file = File(exists=True, mandatory=True, desc='TORTOISE b-matrix file')


class _BmatToFSLGradientsOutputSpec(TraitedSpec):
    bval_file = File(exists=True, desc='FSL bval')
    bvec_file = File(exists=True, desc='FSL bvec')


class BmatToFSLGradients(SimpleInterface):
    """Recover FSL bval/bvec from a TORTOISE b-matrix (inverse of make_bmat_file)."""

    input_spec = _BmatToFSLGradientsInputSpec
    output_spec = _BmatToFSLGradientsOutputSpec

    def _run_interface(self, runtime):
        rows = np.loadtxt(self.inputs.bmtxt_file, ndmin=2)
        bvals = np.zeros(rows.shape[0])
        bvecs = np.zeros((3, rows.shape[0]))
        for i, (bxx, bxy, bxz, byy, byz, bzz) in enumerate(rows):
            bmat = np.array([[bxx, bxy, bxz], [bxy, byy, byz], [bxz, byz, bzz]])
            bval = np.trace(bmat)
            bvals[i] = bval
            if bval > 0:
                evals, evecs = np.linalg.eigh(bmat)
                # principal eigenvector = gradient direction
                vec = evecs[:, np.argmax(evals)]
                bvecs[:, i] = vec / np.linalg.norm(vec)

        bval_file = fname_presuffix(
            self.inputs.bmtxt_file, suffix='.bval', use_ext=False, newpath=runtime.cwd
        )
        bvec_file = fname_presuffix(
            self.inputs.bmtxt_file, suffix='.bvec', use_ext=False, newpath=runtime.cwd
        )
        np.savetxt(bval_file, bvals[np.newaxis, :], fmt='%g')
        np.savetxt(bvec_file, bvecs, fmt='%.8f')
        self._results['bval_file'] = bval_file
        self._results['bvec_file'] = bvec_file
        return runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_bmat_to_fsl_roundtrip -v`
Expected: PASS. If the b-value scale is off by 1000×, adjust the test's `atol` understanding by reading what `make_bmat_file` writes — but eigenvector directions must match.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py -v`
Expected: PASS. Stage for review.

---

## Task 5: `init_diffprep_hmc_wf` workflow + boilerplate

**Files:**
- Create: `qsiprep/workflows/dwi/diffprep.py`
- Modify: `qsiprep/interfaces/tortoise.py` (add `generate_diffprep_boilerplate`, sibling to `generate_drbuddi_boilerplate` ~line 707)
- Modify: `qsiprep/tests/test_interfaces_diffprep.py`

**Interfaces:**
- Consumes: `TORTOISEConvert`, `TORTOISEProcess`, `DIFFPREPMotionParams`, `BmatToFSLGradients` (Tasks 2–4); `qsiprep.interfaces.images.ConformDwi, SplitDWIsFSL`; `qsiprep.interfaces.gradients.ExtractB0s`; `qsiprep.workflows.dwi.util.init_dwi_reference_wf`.
- Produces: `qsiprep.workflows.dwi.diffprep.init_diffprep_hmc_wf(scan_groups, source_file, t2w_sdc, dwi_metadata=None, transformation_type='quadratic', name='diffprep_hmc_wf') -> Workflow`. `inputnode` fields match `init_fsl_hmc_wf`'s (`dwi_file, bvec_file, bval_file, json_file, b0_indices, b0_images, original_files, t1_brain, t1_mask, t1_seg, t1_2_mni_reverse_transform, t2w_unfatsat`). `outputnode` exposes at least: `dwi_files_to_transform, bvec_files_to_transform, bval_files, b0_indices, to_dwi_ref_affines, b0_template, b0_template_mask, pre_sdc_template, sdc_method, motion_params, cnr_map, hmc_optimization_data, slice_quality`.

- [ ] **Step 1: Write the failing workflow-construction test**

Add to `qsiprep/tests/test_interfaces_diffprep.py`:

```python
def test_init_diffprep_hmc_wf_contract():
    from qsiprep import config
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    config.workflow.b0_threshold = 100
    scan_groups = {
        'dwi_series': ['/data/sub-01_dwi.nii.gz'],
        'fieldmap_info': {'suffix': None},
        'dwi_series_pedir': 'j',
    }
    wf = init_diffprep_hmc_wf(
        scan_groups=scan_groups,
        source_file='/data/sub-01_dwi.nii.gz',
        t2w_sdc='',
        transformation_type='quadratic',
    )
    outputnode = wf.get_node('outputnode')
    required = {
        'dwi_files_to_transform',
        'bvec_files_to_transform',
        'bval_files',
        'b0_indices',
        'to_dwi_ref_affines',
        'b0_template',
        'b0_template_mask',
        'sdc_method',
        'motion_params',
    }
    assert required.issubset(set(outputnode.inputs.copyable_trait_names()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_init_diffprep_hmc_wf_contract -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qsiprep.workflows.dwi.diffprep'`.

- [ ] **Step 3: Add `generate_diffprep_boilerplate`**

In `qsiprep/interfaces/tortoise.py`, beside `generate_drbuddi_boilerplate`:

```python
def generate_diffprep_boilerplate(transformation_type):
    """Methods boilerplate for the TORTOISE DIFFPREP HMC backend."""
    order = {
        'rigid': 'rigid-body (motion only)',
        'quadratic': 'quadratic (24-parameter Okan, motion + eddy-current)',
        'cubic': 'cubic (motion + higher-order eddy-current)',
    }[transformation_type]
    return (
        f'\n\nHead motion and eddy-current distortions were corrected using '
        f'DIFFPREP from the TORTOISE [@tortoisev4] software package, with a '
        f'{order} transformation model. DIFFPREP registers each volume to a '
        f'signal-model-predicted target, supporting arbitrary (non-shelled) '
        f'q-space sampling.\n'
    )
```

- [ ] **Step 4: Create the workflow**

Create `qsiprep/workflows/dwi/diffprep.py`:

```python
"""TORTOISE DIFFPREP head-motion/eddy-current correction workflow."""

import os

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...data import load as load_data
from ...interfaces.gradients import ExtractB0s
from ...interfaces.images import ConformDwi, SplitDWIsFSL
from ...interfaces.tortoise import (
    BmatToFSLGradients,
    DIFFPREPMotionParams,
    TORTOISEConvert,
    TORTOISEProcess,
    generate_diffprep_boilerplate,
)
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def _identity_itk_transforms(n_volumes, cwd=None):
    """Write ``n_volumes`` identity ITK affine .mat files (bake-in: no re-move)."""
    import os

    cwd = cwd or os.getcwd()
    text = (
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: MatrixOffsetTransformBase_double_3_3\n'
        'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n'
        'FixedParameters: 0 0 0\n'
    )
    out = []
    for i in range(n_volumes):
        path = os.path.join(cwd, f'identity_{i:04d}.mat')
        with open(path, 'w') as fobj:
            fobj.write(text)
        out.append(path)
    return out


def init_diffprep_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    dwi_metadata=None,
    transformation_type='quadratic',
    name='diffprep_hmc_wf',
):
    """Motion + eddy-current correction with TORTOISE DIFFPREP (HMC-only).

    Drop-in peer of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf`. The
    TORTOISE-resampled DWI and TORTOISE-rotated gradients are emitted directly
    ("bake-in"); per-volume affines are identity and ``sdc_method`` is ``'None'``.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = generate_diffprep_boilerplate(transformation_type)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'b0_indices',
                'b0_images',
                'original_files',
                't1_brain',
                't1_mask',
                't1_seg',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pre_sdc_template',
                'bval_files',
                'hmc_optimization_data',
                'sdc_method',
                'slice_quality',
                'motion_params',
                'cnr_map',
                'bvec_files_to_transform',
                'dwi_files_to_transform',
                'b0_indices',
                'to_dwi_ref_affines',
                'to_dwi_ref_warps',
                'b0_template',
                'b0_template_mask',
            ]
        ),
        name='outputnode',
    )
    # HMC-only: no susceptibility correction here.
    outputnode.inputs.sdc_method = 'None'

    # 1. Convert denoised LPS+ DWI to TORTOISE format (.nii + .bmtxt + mask.nii)
    convert = pe.Node(TORTOISEConvert(), name='convert')

    # 2. Run DIFFPREP (motion + optional eddy)
    diffprep = pe.Node(
        TORTOISEProcess(transformation_type=transformation_type),
        name='diffprep',
        n_procs=config.nipype.omp_nthreads,
    )
    diffprep.inputs.config_file = str(
        load_data(config.workflow.diffprep_config or 'diffprep_params.json')
    )

    # 3. Bring the corrected series back to LPS+ and recover FSL gradients
    back_to_lps = pe.Node(ConformDwi(orientation='LPS'), name='back_to_lps')
    bmat_to_fsl = pe.Node(BmatToFSLGradients(), name='bmat_to_fsl')

    # 4. Split into per-volume files (bake-in outputs)
    split = pe.Node(
        SplitDWIsFSL(b0_threshold=config.workflow.b0_threshold, deoblique_bvecs=True),
        name='split',
    )

    # 5. b0 template + mask
    extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='extract_b0s'
    )
    b0_ref = init_dwi_reference_wf(name='b0_ref', gen_report=False)

    # 6. Motion params and identity affines
    motion = pe.Node(DIFFPREPMotionParams(), name='motion')
    identity = pe.Node(
        niu.Function(function=_identity_itk_transforms, output_names=['out']),
        name='identity',
    )

    def _n_vols(dwi_files):
        return len(dwi_files) if isinstance(dwi_files, (list, tuple)) else 1

    n_vols = pe.Node(
        niu.Function(function=_n_vols, output_names=['n_volumes']), name='n_vols'
    )

    workflow.connect([
        (inputnode, convert, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('t1_mask', 'mask_file'),
        ]),
        (convert, diffprep, [
            ('dwi_file', 'dwi_file'),
            ('bmtxt_file', 'bmtxt_file'),
            ('mask_file', 'mask_file'),
        ]),
        (diffprep, back_to_lps, [('corrected_dwi', 'dwi_file')]),
        (inputnode, back_to_lps, [('bval_file', 'bval_file')]),
        (diffprep, bmat_to_fsl, [('corrected_bmtxt', 'bmtxt_file')]),
        (back_to_lps, split, [('dwi_file', 'dwi_file')]),
        (bmat_to_fsl, split, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
        (split, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('b0_indices', 'b0_indices'),
        ]),
        (split, n_vols, [('dwi_files', 'dwi_files')]),
        (n_vols, identity, [('n_volumes', 'n_volumes')]),
        (identity, outputnode, [('out', 'to_dwi_ref_affines')]),
        (back_to_lps, extract_b0s, [
            ('dwi_file', 'dwi_series'),
        ]),
        (extract_b0s, b0_ref, [('b0_average', 'inputnode.b0_template')]),
        (b0_ref, outputnode, [
            ('outputnode.ref_image', 'b0_template'),
            ('outputnode.dwi_mask', 'b0_template_mask'),
        ]),
        (extract_b0s, outputnode, [('b0_average', 'pre_sdc_template')]),
        (diffprep, motion, [('transforms_file', 'transforms_file')]),
        (motion, outputnode, [('motion_file', 'motion_params')]),
    ])  # fmt:skip

    return workflow
```

> Verification notes (reconcile against the real binary/interfaces during a container run):
> - `ExtractB0s` field names (`dwi_series`/`b0_average`) — confirm via `grep -n "class ExtractB0s" -A30 qsiprep/interfaces/gradients.py`; adjust connect names if they differ.
> - `init_dwi_reference_wf` signature/outputs (`gen_report`, `outputnode.ref_image`, `outputnode.dwi_mask`) — confirm via `grep -n "def init_dwi_reference_wf" -A40 qsiprep/workflows/dwi/util.py`.
> - `SplitDWIsFSL` output field names (`dwi_files`, `bval_files`, `bvec_files`, `b0_indices`) — confirm via the eddy workflow `fsl.py:256` usage.
> - `TORTOISEConvert` needs a mask. `t1_mask` is a stand-in; if TORTOISE needs a DWI-space mask, extract a b0 first and mask it (mirror how eddy obtains its pre-eddy mask) — keep the convert→diffprep contract intact.

- [ ] **Step 5: Run the contract test**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_init_diffprep_hmc_wf_contract -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py -v`
Expected: PASS. Stage for review.

---

## Task 6: Branch `init_dwi_preproc_wf` to DIFFPREP

**Files:**
- Modify: `qsiprep/workflows/dwi/base.py` (imports ~line 26; HMC dispatch ~lines 245–265)
- Modify: `qsiprep/tests/test_interfaces_diffprep.py`

**Interfaces:**
- Consumes: `init_diffprep_hmc_wf` (Task 5).
- Produces: `init_dwi_preproc_wf` selects the DIFFPREP backend when `config.workflow.hmc_model.startswith('diffprep_')`, passing `transformation_type=hmc_model.split('_', 1)[1]`.

- [ ] **Step 1: Write the failing selection test**

Add to `qsiprep/tests/test_interfaces_diffprep.py`:

```python
def test_base_selects_diffprep(monkeypatch):
    import qsiprep.workflows.dwi.base as base_mod
    from qsiprep import config

    captured = {}

    def fake_init_diffprep_hmc_wf(*args, **kwargs):
        captured['transformation_type'] = kwargs.get('transformation_type')
        from nipype.interfaces import utility as niu
        from nipype.pipeline import engine as pe

        return pe.Node(niu.IdentityInterface(fields=['x']), name='hmc_sdc_wf')

    monkeypatch.setattr(base_mod, 'init_diffprep_hmc_wf', fake_init_diffprep_hmc_wf)

    config.workflow.hmc_model = 'diffprep_cubic'
    order = base_mod._diffprep_order(config.workflow.hmc_model)
    assert order == 'cubic'
```

> This test pins the suffix→order helper directly (cheap, deterministic). Full `init_dwi_preproc_wf` construction needs heavy BIDS fixtures already covered by `test_cli_run.py`; we add the small helper to keep this unit-testable.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_base_selects_diffprep -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'init_diffprep_hmc_wf'` (or `_diffprep_order`).

- [ ] **Step 3: Add the import, helper, and branch**

In `qsiprep/workflows/dwi/base.py`, near the other dwi-workflow imports (~line 27):

```python
from .diffprep import init_diffprep_hmc_wf
```

Add a small module-level helper (top of the module, after imports):

```python
def _diffprep_order(hmc_model):
    """Map a diffprep_* hmc_model string to a TORTOISE transformation type."""
    return hmc_model.split('_', 1)[1]
```

In `init_dwi_preproc_wf`, extend the HMC dispatch. After the `elif config.workflow.hmc_model == 'eddy':` block:

```python
    elif config.workflow.hmc_model.startswith('diffprep_'):
        hmc_wf = init_diffprep_hmc_wf(
            scan_groups=scan_groups,
            source_file=source_file,
            dwi_metadata=dwi_metadata,
            t2w_sdc=t2w_sdc,
            transformation_type=_diffprep_order(config.workflow.hmc_model),
            name='hmc_sdc_wf',
        )
```

(The existing `pre_hmc` orientation line — `'LAS' if config.workflow.hmc_model == 'eddy' else 'LPS'` — already routes diffprep to LPS+. No change needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py::test_base_selects_diffprep -v`
Expected: PASS.

- [ ] **Step 5: Run the broader suite to catch import regressions**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py qsiprep/tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint**

Stage for review.

---

## Task 7: Documentation

**Files:**
- Modify: `docs/usage.rst` (the `--hmc-model` / motion-correction section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Locate the motion-correction docs**

Run: `grep -rn "hmc-model\|3dSHORE\|SHORELine\|eddy" docs/usage.rst`
Expected: find the section that describes `--hmc-model`.

- [ ] **Step 2: Add the DIFFPREP options**

In `docs/usage.rst`, in the `--hmc-model` discussion, add:

```rst
For non-shelled acquisitions (e.g. compressed-sensing DSI), where FSL eddy
cannot be used and SHORELine corrects motion but not eddy currents, TORTOISE
DIFFPREP is available via ``--hmc-model``:

- ``diffprep_motion`` — rigid head-motion correction only.
- ``diffprep_quadratic`` — rigid motion plus 24-parameter quadratic
  eddy-current correction (recommended for CS-DSI).
- ``diffprep_cubic`` — rigid motion plus cubic eddy-current correction.

These run TORTOISE v4 DIFFPREP, which fits a signal model over arbitrary
q-space and therefore does not require shells. Susceptibility distortion
correction is unaffected and continues to use the configured fieldmap method.
Advanced TORTOISE settings can be supplied with ``--diffprep-config``.
```

- [ ] **Step 3: Verify docs build (if sphinx is set up)**

Run: `python -m sphinx -b html docs docs/_build/html -q 2>&1 | tail -20` (skip if the docs toolchain isn't installed locally).
Expected: no new warnings referencing the edited section.

- [ ] **Step 4: Checkpoint**

Stage for review.

---

## Final integration verification

- [ ] **Run the full new-feature test set**

Run: `python -m pytest qsiprep/tests/test_interfaces_diffprep.py qsiprep/tests/test_cli.py -v`
Expected: all pass.

- [ ] **Import smoke test**

Run: `python -c "from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf; from qsiprep.interfaces.tortoise import TORTOISEProcess, DIFFPREPMotionParams, BmatToFSLGradients; print('ok')"`
Expected: prints `ok`.

- [ ] **Container flag-reconciliation (must run inside the qsiprep image, before any real subject)**

1. `TORTOISEProcess --help` → confirm/fix the `argstr` flag names in `TORTOISEProcess` (Task 2) and the DIFFPREP-only step invocation.
2. Run one short DIFFPREP on a test DWI → confirm `TORTOISEProcess` output filenames match `_list_outputs` (Task 2) and the transforms-file column layout matches `DIFFPREPMotionParams` (Task 3).
3. Confirm `_identity_itk_transforms` `.mat` files are accepted by the downstream `antsApplyTransforms` resampling step.

---

## Self-Review (completed by plan author)

- **Spec coverage:** CLI/config plumbing → T1; `TORTOISEProcess` → T2; `DIFFPREPMotionParams` → T3; gradient handling for bake-in → T4; `init_diffprep_hmc_wf` + boilerplate + identity affines + LPS+ → T5; `base.py` branch → T6; container guard → covered by the PATH availability of `TORTOISEProcess` plus the final container reconciliation step; docs → T7. `TORTOISEConvert` reuse is explicit (T5). The decompose path and DRBUDDI-integrated mode are spec'd out-of-scope and not planned. ✅
- **Placeholder scan:** No TBD/TODO. The "verification notes" carry concrete starting values plus an exact command to confirm them — they are reconciliation steps, not placeholders. ✅
- **Type consistency:** `TORTOISEProcess` outputs (`corrected_dwi`, `corrected_bmtxt`, `transforms_file`) are consumed with those exact names in T5; `DIFFPREPMotionParams.motion_file`, `BmatToFSLGradients.{bval_file,bvec_file}` consumed consistently; `init_diffprep_hmc_wf` signature in T5 matches the call in T6; `_diffprep_order` defined and used in T6. ✅
