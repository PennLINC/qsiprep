# TORTOISE issue draft: EPIREG never applies the gradient nonlinearity field

Target: https://github.com/QMICodeBase/TORTOISEV4/issues (new issue)
Observed against `main` @ `d3301e5`.
Drafted 2026-08-27 while implementing gradient nonlinearity support in
[QSIPrep #1106](https://github.com/PennLINC/qsiprep/pull/1106).

---

## Title

`--epi T2Wreg` with `--grad_nonlin` aborts: EPIREG builds the gradwarp field
path from the CLI argument instead of the generated file

## Summary

`EPIREG::Step0_CreateImages` is meant to gradwarp-correct the b=0 image before
estimating the T2w-based susceptibility deformation, mirroring what
`DRBUDDI::Step0_CreateImages` does for its own inputs. It cannot: it derives
the field's filename from the raw `--grad_nonlin` argument rather than from the
file `TORTOISE::` actually generates in the temp-proc folder. The path it
constructs does not exist for any accepted input, so `readImageD` throws, and
nothing in `TORTOISEProcess` catches it.

The equivalent code in `DRBUDDI.cxx` builds the path correctly, and the buggy
form is preserved one line below it as a comment — which suggests this was
fixed there and the fix was not carried across to `EPIREG.cxx`.

## Details

### 1. The setting holds the user's argument, not the generated field

`TORTOISE::UpdateSettingsFromCommandLine` (`src/main/TORTOISE.cxx:449`) does:

```cpp
RegistrationSettings::get().setValue<std::string>("grad_nonlin", parser->getGradNonlinInput());
```

`grad_nonlin` is never reassigned. Meanwhile the Import step
(`src/main/TORTOISE.cxx:1962-2043`) writes the expanded field to:

```
<temp_proc_folder>/<basename>_proc_gradnonlin_field_inv.nii
```

### 2. EPIREG reconstructs a different path

`src/main/EPIREG.cxx:68-72`:

```cpp
std::string gradnonlin_field_name = RegistrationSettings::get().getValue<std::string>("grad_nonlin");
if (gradnonlin_field_name != "")
{
    std::string gradnonlin_name_inv = gradnonlin_field_name.substr(0, gradnonlin_field_name.rfind(".nii")) + "_inv.nii";
    DisplacementFieldType::Pointer field = readImageD<DisplacementFieldType>(gradnonlin_name_inv);
```

For a coefficient file — the only kind that reaches this code, see (3) —
`rfind(".nii")` returns `npos`, so `substr(0, npos)` returns the whole string
and the constructed path is:

```
/path/to/coeffs.grad_inv.nii
```

which never exists.

### 3. `.nii` fields cannot reach this code at all

`DRBUDDI_PARSERBASE::getGradNonlinInput`
(`src/tools/DRBUDDI/DRBUDDI_parserBase.cxx:464-480`) rejects any extension
other than `.grad`, `.dat`, `.gc`:

```cpp
std::string ext = nm.substr(nm.rfind("."));
if (ext != ".grad" && ext != ".dat" && ext != ".gc")
{
    std::cout << "WARNING! Gradient nonlinearity file format not recognized. ..." << std::endl;
    std::cout << "Disabling gradient nonlinearity based processing..." << std::endl;
    return "";
}
```

So the `.nii` branch at `TORTOISE.cxx:1976` is unreachable from
`TORTOISEProcess`, and the `--grad_nonlin` help text
(`DRBUDDI_parserBase.cxx:128`, "Can be in ITK displacement field format ...",
with usage example `field.nii[1,2D]`) advertises an input the parser refuses.
This looks like a second, separable bug.

### 4. The exception is not caught

`readImageD` (`src/main/defines.cxx:12-20`) calls `reader->Update()` with no
error handling, and there is no `try`/`catch` in `src/main/main.cxx` or
`src/main/TORTOISE.cxx`. The uncaught `itk::ExceptionObject` terminates the
process.

### 5. DRBUDDI already has this right

`src/main/DRBUDDI.cxx:146-149`:

```cpp
std::string gradnonlin_name_inv = this->proc_folder + std::string("/") + basename + std::string("_proc_gradnonlin_field_inv.nii");

//  std::string gradnonlin_name_inv = gradnonlin_field_name.substr(0,gradnonlin_field_name.rfind(".nii"))+ "_inv.nii";
DisplacementFieldType::Pointer field = readImageD<DisplacementFieldType>(gradnonlin_name_inv);
```

The commented-out line is exactly the live line in `EPIREG.cxx`.

## Reproduction

```
TORTOISEProcess \
    --up_data sub-01_dwi.nii \
    --structural sub-01_T2w.nii \
    --grad_nonlin coeffs.grad \
    --epi T2Wreg
```

Expected: EPIREG resamples `b0_up` through the gradwarp field before
`Step1_RigidRegistration`.
Actual: aborts with an ITK read error on `coeffs.grad_inv.nii`.

Without `--epi T2Wreg` (or without `--grad_nonlin`) the run completes, so the
failure is specific to the combination.

## Impact

Any `--epi T2Wreg` run that also passes `--grad_nonlin` fails. Because the
failure is a hard abort rather than a silent skip, no incorrect results should
have been produced by this path — but it does mean the intended behaviour
(EPIREG estimating its deformation in gradwarp-corrected geometry) has never
actually run, so downstream tools that assume it did are mistaken. This is what
prompted the report: QSIPrep was about to be changed to match the intended
behaviour.

## Suggested fix

Mirror `DRBUDDI.cxx:146` in `EPIREG.cxx:71`:

```cpp
std::string up_name = this->parser->getUpInputName();
std::string basename = fs::path(up_name).filename().string();
basename = basename.substr(0, basename.rfind(".nii"));
std::string gradnonlin_name_inv = this->proc_folder + "/" + basename + "_proc_gradnonlin_field_inv.nii";
```

Separately, either make `getGradNonlinInput` accept `.nii`/`.nii.gz` as the
help text promises, or correct the help text and usage examples.

Happy to open a PR for either or both if that is useful.

---

## Notes for us (not part of the issue text)

- Confirming the abort behaviour empirically requires a TORTOISE build with a
  T2w and a coefficient file; the analysis above is from source only. Worth
  running before posting, and worth softening the "aborts" claim to "throws an
  uncaught ITK exception on a path that does not exist" if we do not.
- A related but distinct discrepancy, not included above because it is a design
  question rather than a clear bug: the GE z-origin shift applied at
  `TORTOISE.cxx:1996-2011` after `mk_displacement` has no counterpart in the
  standalone `CreateNonlinearityDisplacementMap` binary, so the two produce
  fields at different z for the same GE input. QSIPrep currently refuses GE
  coefficient files for this reason. If we want to ask about it, it should be a
  separate issue or a discussion thread, framed as a question.
