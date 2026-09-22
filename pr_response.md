# PR #1144 review responses

Draft replies to the eight review comments from @mattcieslak on
https://github.com/PennLINC/qsiprep/pull/1144, at head `cbd856c`.

Nothing here has been posted or committed. Three items need a decision before they
can be answered; they are marked **Needs a decision**.

---

## 1. `docs/preprocessing.rst:382` — "is this the convention fmriprep uses too?"

**Needs a decision.** @tsalo's reply is correct for released fMRIPrep. I first checked
`master` and reported its patterns as "the fMRIPrep convention", which was wrong. The
naming depends on the version, and the version that matters here is not released yet.

Checked `nipreps/fmriprep` `docs/outputs.rst` at tags 23.2.3, 24.1.1, 25.0.0, 25.1.4 and
`master`. The boldref patterns are identical across every release through 25.1.4:

```
sub-<subject_label>_[specifiers]_desc-hmc_boldref.nii.gz
sub-<subject_label>_[specifiers]_desc-coreg_boldref.nii.gz
sub-<subject_label>_[specifiers]_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt
```

That is what ds006185 contains. Its `dataset_description.json` gives
`GeneratedBy.Version = "25.2.4"`.

`master` is different, and `CHANGES.rst` dates it: "26.0.0 (TBD) ... This release adds
alternative BOLD coregistration target spaces, selected with
``--bold-coreg-level {run,session,subject}``."

```
sub-<subject_label>_[specifiers]_space-orig_desc-hmc_boldref.nii.gz
sub-<subject_label>_[specifiers]_space-run_boldref.nii.gz
sub-<subject_label>_[ses-<session_label>_]space-session_boldref.nii.gz
sub-<subject_label>_space-subject_boldref.nii.gz
sub-<subject_label>_[specifiers]_from-orig_to-run_mode-image_desc-hmc_xfm.txt
sub-<subject_label>_[specifiers]_from-run_to-T1w_mode-image_desc-coreg_xfm.txt
sub-<subject_label>_from-run_to-subject_mode-image_desc-coreg_xfm.txt
sub-<subject_label>_from-subject_to-T1w_mode-image_desc-coreg_xfm.txt
```

So between 25.x and 26.0.0 fMRIPrep makes three changes. The generic `boldref` space is
replaced by a level-named space (`run`, `session`, `subject`). `desc-coreg` comes off the
coregistration-target image, because the space name now carries that information, and
stays on the transforms. `desc-hmc` stays on an image and gains `space-orig`.

`--dwiref-definition` is the direct analogue of `--bold-coreg-level`, and QSIPrep 26.x is
contemporaneous with fMRIPrep 26.0.0, so 26.0.0 is the scheme worth comparing against.
Four differences from this PR:

| | fMRIPrep 26.0.0 | this PR |
|---|---|---|
| run-level reference | `space-run_boldref`, always written | no `space` entity; `desc-coreg` added only when the resolved level is `distortion-group` |
| level-named reference | `space-subject_boldref`, no `desc` | `space-subject_desc-coreg_dwiref` |
| `desc-coreg` | on the transforms | on the images; transforms carry no `desc` |
| source space of the run-to-template transform | `from-run` | `from-orig` |

The last row is the one I would fix regardless of what is decided about the others.
fMRIPrep 26.0.0 uses `orig` for the pre-HMC space and `run` for the post-HMC per-run
reference, which it makes explicit in `from-orig_to-run_mode-image_desc-hmc_xfm`. QSIPrep
already draws the same distinction: `docs/preprocessing.rst` documents
`from-orig_to-dwiref_mode-image_desc-eddy_xfm.h5` for the HMC transform. This PR writes
`from-orig_to-subject`, whose source is the post-HMC, post-SDC b=0 reference, not the
original. By both projects' existing usage that space is not `orig`.

Options:

1. Adopt the 26.0.0 scheme: name the run-level reference `space-distortiongroup`, write
   it unconditionally, drop `desc-coreg` from the images, put `desc-coreg` on the
   transforms, and rename the transform source away from `orig`.
2. Keep `desc-coreg` on the images, matching released fMRIPrep and ds006185, but still
   write the run-level reference unconditionally under one name rather than switching
   between two depending on an unrelated flag, and still fix the `from-orig` source.
3. Keep the current scheme and document the divergence.

My recommendation is 1, with 2 as a reasonable fallback if matching the derivatives
people have on disk today is worth more than matching the release QSIPrep 26.x will ship
alongside. In either case the run-level reference should not change name depending on
`--dwiref-definition`, and `from-orig` should be corrected.

---

## 2. `qsiprep/cli/parser.py:49` — "would it make sense to leave this infrastructure in for future deprecations?"

**Already answered; no change needed.** The infrastructure is still present. Only the
two lookup tables are empty:

- `deprecations = {}` (line 49) and `forwarded_deprecations = {}` (line 53)
- `_warn_deprecated` (line 55), `DeprecatedForwardAction` (line 62) and
  `DeprecationForwardingParser` (line 78) are all retained

`DeprecationForwardingParser` cannot be removed regardless: it is the parser class
(line 259) and it still normalizes `--hmc-method` and validates `--shoreline-config`
after the whole command line is read (lines 665, 733).

One loose end: `DeprecatedStoreAction` was removed in this PR because
`--b0-to-t1w-transform` was its only user. If the intent is to keep the full set of
actions available, that one should come back. It is about ten lines.

---

## 3. `qsiprep/cli/parser.py:497` — suggestion: "does not necessarily remove the need for it"

**Accept.** The suggested wording is more accurate. Console normalization removes the
need for N4 in some cases, which is the premise of `auto`. Saying it "does not remove
the need" contradicts the option two lines below.

Apply the suggestion as written. The same sentence appears in the `--dwi-biascorrect`
help (parser.py, around line 602) and should be changed there too, or the two options
will describe the same physics differently.

---

## 4. `qsiprep/workflows/dwi/pre_hmc.py:29` — "isn't having this default to True the legacy behavior that this PR is getting rid of?"

**Not the legacy behavior, but the default should still go.** The legacy path was an
N4 node that ran before merging. This PR deletes that node, along with the
`bias_image`/`bias_images` output traits and merge nodes it fed.

`do_biascorr` in `init_dwi_pre_hmc_wf` no longer gates any processing. Its only use is
line 97:

```python
workflow.__postdesc__ = gen_denoising_boilerplate(do_biascorr)
```

`gen_denoising_boilerplate` uses it to decide whether to emit one sentence describing
the final-stage N4 (`merge.py:675`). So `do_biascorr=True` here does not run N4. It
produces methods text stating that N4 ran.

That is still a defect. Two direct callers omit the argument
(`test_workflows_native.py:92` and `:102`) and get boilerplate claiming bias correction
was applied. The default hides the omission instead of surfacing it.

Proposed change: make `do_biascorr` required in `init_dwi_pre_hmc_wf`,
`init_dwi_preproc_wf` and `init_dwi_finalize_wf`, and pass it explicitly in the two
tests. See item 8 for a larger change that removes the parameter from the pre-HMC
workflow entirely.

---

## 5. `qsiprep/workflows/base.py:524` — "this comment is totally opaque to me"

**Agreed.** The comment states a rule without stating the situation it covers.
Proposed replacement:

```python
    # --dwiref-definition subject needs at least two DWI groups to build a
    # template from. A subject with fewer falls back to distortion-group, below.
    # The derivative labels follow the level that was actually built, not the one
    # requested, so a fallback subject's references are labelled distortion-group.
```

---

## 6. `qsiprep/workflows/base.py:576` — "So we get a native-to-dwi template transform in the derivatives along with a dwi template to acpc transform?"

**Yes for the second, conditionally for the first.** With
`--dwiref-definition subject`:

- `sub-X_from-subject_to-ACPC_mode-image_xfm.mat` is written whenever the template is
  built (`base.py:622`, not gated).
- `sub-X_[entities]_from-orig_to-subject_mode-image_xfm.mat` is written per DWI group,
  but only when `--dwiref-construction-transform` is `Rigid` or `Affine`
  (`base.py:645`, `:826`).

The default is `BSplineSyN` (parser.py:832). So by default the second transform is not
written, and there is no way to map a group into `space-subject` from the outputs. The
gate is pre-existing: `antsMultivariateTemplateConstruction2` produces an
`[affine, warp]` pair per input, which does not fit a single-file `.mat` sink.

This is worth treating as a gap rather than an explanation. This PR starts shipping
`space-subject_desc-coreg_dwiref.nii.gz` as a derivative, so under the default settings
users get an image in a space they cannot map anything into or out of, other than to
ACPC. Either the nonlinear pair should be written (as two files, or `.h5`), or the
docs should state that the run-to-template transform is only available for linear
templates.

---

## 7. `qsiprep/workflows/base.py:717` — "do these two do_biascorrs respect the user-requested stage?"

**There is no longer a stage, and yes, both respect the user's choice.**
`--b1-biascorrect-stage {final,none,legacy}` is replaced by
`--dwi-biascorrect {n4,auto,none}`, which selects whether N4 runs, not when. Only one
stage remains, after resampling.

Both arguments are the same value. It is computed once at `base.py:703`:

```python
do_biascorr = biascorr_by_output[final_output_name]
```

`biascorr_by_output` is built at `base.py:~500` by calling
`dwi_biascorrect_enabled(...)` once per final output, over the union of the DWI files
of every constituent correction unit. That function reads
`config.workflow.dwi_biascorrect`, returns `True` for `n4`, `False` for `none`, and for
`auto` inspects the BIDS `ImageType` of each file.

The union matters under `--distortion-group-merge`. Several finalized units are
concatenated into one output, so a per-unit decision could apply N4 to one constituent
and not another within a single output file.

The two arguments do different things despite being the same value:

- `init_dwi_preproc_wf` (line 710) uses it only for reporting: the methods boilerplate
  (item 4) and the `dwi_biascorrect_applied` field of the DWI report
  (`dwi/base.py:495`).
- `init_dwi_finalize_wf` (line 717) uses it to gate the actual N4 node
  (`finalize.py:311` to `:712`).

---

## 8. `qsiprep/workflows/base.py:710` — "still not sure this is needed anymore"

**Needs a decision.** This anchors on the `do_biascorr` passed to
`init_dwi_preproc_wf`. It is needed as the code stands, but only for reporting, as
described in item 7. No processing in the preproc workflow depends on it.

The value is threaded through two APIs that have no other use for it:

```
init_single_subject_wf -> init_dwi_preproc_wf -> init_dwi_pre_hmc_wf -> gen_denoising_boilerplate
```

The cause is that the methods text for the final-stage N4 is generated in the pre-HMC
workflow. The sentence `gen_denoising_boilerplate` emits reads "after corrected images
were resampled", which describes work that happens in `init_dwi_finalize_wf`.

Two options:

1. Keep it. One extra argument on two workflows, and the boilerplate stays in one
   place with the rest of the denoising text.
2. Move the bias-correction sentence out of `gen_denoising_boilerplate` and into the
   finalize workflow, where the decision is already available. `do_biascorr` then comes
   off `init_dwi_pre_hmc_wf` and `init_dwi_preproc_wf` entirely, and the two test
   callers in item 4 need no change. The DWI report field at `dwi/base.py:495` still
   needs the value, so `init_dwi_preproc_wf` would keep the argument unless the report
   input is also moved.

Option 2 removes the parameter from the workflow that has no use for it, at the cost
of splitting the denoising boilerplate across two functions. I lean to 2 if item 4 is
being changed anyway, and to 1 otherwise.

---

## Not raised in review, found while preparing these answers

### A. Stale text left in the `.. important::` block

`docs/preprocessing.rst:384-390`. The old sentence "QSIPrep does not currently write
out the coregistration transform from dwiref space to ACPC space" was replaced in this
PR, but its continuation was not removed:

```
  When it does start writing this transform out, it will be organized like this::

    sub-<label>/
      ses-<label>/
        dwi/
          sub-<label>_ses-<label>_from-dwiref_to-ACPC_mode-image_xfm.h5
          sub-<label>_ses-<label>_from-ACPC_to-dwiref_mode-image_xfm.h5
```

"When it does start writing this transform out" now follows a paragraph saying the
transform is written. The example filenames do not match what the code produces:
`from-dwiref_to-ACPC` rather than `from-subject_to-ACPC`, `.h5` rather than `.mat`,
and at session level rather than subject level. The inverse transform
(`from-ACPC_to-dwiref`) is not written at all.

Delete lines 384-390. This is a defect introduced by this PR.

### B. The twelve integration manifests are unverified

`qsiprep/tests/data/*_outputs.txt`. The renamed and added entries were derived from the
datasink definitions, not from a run, because the integration jobs need downloaded
datasets. The names were checked against `io_spec.json` by rendering each one through
`build_path`, and the counts are internally consistent, but only CI can confirm they
match a real run. `check_generated_files` compares the two lists exactly and prints
both on failure.
