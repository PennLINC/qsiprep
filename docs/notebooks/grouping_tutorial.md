---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Understanding scan grouping in QSIPrep

Before QSIPrep processes a single voxel, it has to answer three questions about
your diffusion data:

1. **Which scans estimate each fieldmap?** (in BIDS terms: `B0FieldIdentifier`)
2. **Which fieldmap corrects which scan?** (`B0FieldSource`)
3. **Which scans are combined into one preprocessed output file?** (`MultipartID`)

If you set those three fields in your sidecar JSONs, QSIPrep uses your answers
verbatim. Wherever you did not, it infers an answer and *tells you it guessed* —
every decision carries a provenance tag: **curated** (you set it),
**cli-override** (a command-line flag), **inferred** (a heuristic), or
**IntendedFor** (translated from the deprecated fieldmap linkage — see the
appendix).

This notebook builds tiny fake datasets *in memory* — no files, no BIDS layout,
just filenames and sidecar values — and shows how each metadata change alters
the grouping. Nothing is processed; grouping only reads metadata.

```{code-cell} ipython3
import html as _html

from IPython.display import HTML, display

from qsiplan import (
    DistortionSignature,
    FileRecord,
    build_grouping,
    describe_processing,
    render_html,
    report_text,
)

SUBJECT = '/bids/sub-01'


def scan(
    name,
    folder='dwi',
    suffix=None,
    PhaseEncodingDirection='j-',
    TotalReadoutTime=0.05,
    ShimSetting=None,
    session=None,
    B0FieldIdentifier=(),
    B0FieldSource=(),
    MultipartID=None,
    IntendedFor=(),
):
    """One imaging file plus its sidecar metadata.

    The keyword arguments are named after the exact BIDS sidecar fields
    they simulate, so each demo reads like the JSON you would write.
    """
    suffix = suffix or ('dwi' if folder == 'dwi' else 'epi')
    sess = f'ses-{session}/' if session else ''
    as_tuple = lambda v: (v,) if isinstance(v, str) else tuple(v or ())  # noqa: E731
    return FileRecord(
        path=f'{SUBJECT}/{sess}{folder}/{name}',
        datatype='anat' if suffix in ('T1w', 'T2w') else folder,
        suffix=suffix,
        session=session,
        signature=DistortionSignature(
            pe_dir=PhaseEncodingDirection,
            readout_time=TotalReadoutTime,
            shim=tuple(ShimSetting) if ShimSetting else None,
        ),
        b0field_identifiers=as_tuple(B0FieldIdentifier),
        b0field_sources=as_tuple(B0FieldSource),
        multipart_id=as_tuple(MultipartID),
        intended_for=as_tuple(IntendedFor),
    )


def group(*records, **options):
    """Run the full grouping on in-memory records."""
    return build_grouping(list(records), subject_id='01', **options)


def show(grouping, height=560):
    """Display the explanatory HTML page for a grouping inline."""
    page = _html.escape(render_html(grouping))
    display(
        HTML(
            f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;'
            'border:1px solid #cbd5e1;border-radius:8px;background:#fff"></iframe>'
        )
    )
```

## 1. One scan, no metadata

The simplest possible subject: a single DWI series, no fieldmaps, no curation.
Even here grouping has decisions to report: the scan forms one *distortion
group* (files that share the same susceptibility distortion), it becomes one
output file, and — since there is nothing to correct it with — susceptibility
distortion correction (SDC) is skipped.

```{code-cell} ipython3
g = group(scan('sub-01_dir-AP_dwi.nii.gz'))
print(report_text(g))
```

```{code-cell} ipython3
show(g, height=620)
```

Notice the *why* lines. The output name carries `MultipartID auto+concat+0
[inferred]`: the `auto+` prefix is reserved for identifiers QSIPrep made up
(`+` cannot appear in curated BIDS values, so they can never collide with your
names).

## 2. A reverse phase-encoded partner appears

Now add a second series acquired with the opposite phase encoding (`j` instead
of `j-` — an AP/PA pair). Two things happen:

- The scans land in **different distortion groups**, because they are squished
  in opposite directions along the phase-encoding axis. A distortion group is
  defined by `PhaseEncodingDirection` + `TotalReadoutTime` + `ShimSetting`.
- QSIPrep notices the opposite polarities and **infers a PEPOLAR fieldmap
  estimation** from the pair: their b=0 images can be compared to measure the
  distortion field. Both groups are corrected by it.

This is the zero-curation "HCP-style" case: acquire blip-up and blip-down, get
susceptibility correction for free.

```{code-cell} ipython3
ap = scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-')
pa = scan('sub-01_dir-PA_dwi.nii.gz', PhaseEncodingDirection='j')
g = group(ap, pa)
show(g, height=780)
```

The estimation card is **amber (inferred)** and named `auto+pepolar+j` — again
the reserved `auto+` prefix. The blip diagram (&#9650; j &harr; &#9660; j&minus;)
shows the pairing, and each scan row carries the Ⓐ chip marking that its b=0
volumes feed estimation A.

**The pairing does not require the same axis.** Any two differing phase
encodings jointly determine the susceptibility field — opposite blips on one
axis are just the best-conditioned case. Series encoded `i-` and `j` pool
into one estimation too. Whether a given tool can consume that shape is a
*backend* question, answered in the processing previews: TOPUP takes any
mix of encodings, while DRBUDDI needs a same-axis opposing pair and says so:

```{code-cell} ipython3
lr = scan('sub-01_dir-LR_dwi.nii.gz', PhaseEncodingDirection='i-')
pa = scan('sub-01_dir-PA_dwi.nii.gz', PhaseEncodingDirection='j')
g = group(lr, pa)
print(report_text(g))
print(describe_processing(g, 'tortoise'))
```

## 3. Curating fieldmaps: `B0FieldIdentifier` / `B0FieldSource`

Guessing is fine until it isn't. The BIDS way to make fieldmap relationships
explicit is a pair of sidecar fields:

- **`B0FieldIdentifier`** names an estimation, and goes on every file whose
  data *feed* it.
- **`B0FieldSource`** goes on each DWI series and names the estimation that
  should *correct* it.

Here is the same AP/PA pair from section 2, curated. The data are physically
identical — only the sidecars changed. The estimation now has **your** name
(no `auto+`), everything renders green, and no heuristic was involved:

```{code-cell} ipython3
ap = scan(
    'sub-01_dir-AP_dwi.nii.gz',
    PhaseEncodingDirection='j-',
    B0FieldIdentifier='pepolar_fmap',
    B0FieldSource='pepolar_fmap',
)
pa = scan(
    'sub-01_dir-PA_dwi.nii.gz',
    PhaseEncodingDirection='j',
    B0FieldIdentifier='pepolar_fmap',
    B0FieldSource='pepolar_fmap',
)
g = group(ap, pa)
show(g, height=780)
```

A dedicated `fmap/` image works exactly the same way: put the
`B0FieldIdentifier` on the fieldmap file *and* on the DWI (its b=0 supplies
the opposite blip in the estimation), and point the DWI's `B0FieldSource` at
it:

```{code-cell} ipython3
dwi = scan(
    'sub-01_dir-AP_dwi.nii.gz',
    PhaseEncodingDirection='j-',
    B0FieldIdentifier='epi_fmap',
    B0FieldSource='epi_fmap',
)
fmap = scan(
    'sub-01_dir-PA_epi.nii.gz',
    folder='fmap',
    PhaseEncodingDirection='j',
    B0FieldIdentifier='epi_fmap',
)
g = group(dwi, fmap)
show(g, height=700)
```

The fieldmap file lives in `fmap/`, so it is listed on the estimation card
("from `fmap/`") but never appears inside an output box — fieldmaps measure
distortion; they are not part of your diffusion data.

(Older datasets link fieldmaps with the `IntendedFor` field instead. QSIPrep
still honors it, but it is deprecated — see the appendix at the end of this
notebook.)

## 4. Splitting outputs with `MultipartID` — and borrowing

By default, all series in a session with compatible distortion parameters are
concatenated into **one** output file (better head-motion correction, one file
to analyze). Four runs, one output:

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_run-1_dwi.nii.gz', PhaseEncodingDirection='j-'),
    scan('sub-01_dir-AP_run-2_dwi.nii.gz', PhaseEncodingDirection='j-'),
    scan('sub-01_dir-PA_run-1_dwi.nii.gz', PhaseEncodingDirection='j'),
    scan('sub-01_dir-PA_run-2_dwi.nii.gz', PhaseEncodingDirection='j'),
]
g = group(*runs)
print(report_text(g))
```

`MultipartID` is the sidecar field that controls this. Give run-1 and run-2
different values and you get two output files. But watch what happens to the
fieldmap estimation — **estimation membership and concatenation membership are
independent by design**. All four scans still feed the *same* PEPOLAR
estimation; each output then *borrows* the b=0 images of the scans that ended
up in the other output:

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_run-1_dwi.nii.gz', PhaseEncodingDirection='j-', MultipartID='part1'),
    scan('sub-01_dir-AP_run-2_dwi.nii.gz', PhaseEncodingDirection='j-', MultipartID='part2'),
    scan('sub-01_dir-PA_run-1_dwi.nii.gz', PhaseEncodingDirection='j', MultipartID='part1'),
    scan('sub-01_dir-PA_run-2_dwi.nii.gz', PhaseEncodingDirection='j', MultipartID='part2'),
]
g = group(*runs)
show(g, height=900)
```

The dashed "borrows b=0 volumes from…" sentences make this visible: borrowed
scans improve the *fieldmap*, but their diffusion volumes are **not** written
into that output file.

**Naming the outputs.** A MultipartID that begins with `acq-` does double
duty: it still groups the scans, and its label becomes the `acq-` entity of
the output filename. This is also the fix when two outputs would otherwise
derive the same name (QSIPrep refuses to guess a name for you — colliding
names are a hard error until you name the groups):

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-', MultipartID='acq-multishell'),
    scan('sub-01_dir-PA_dwi.nii.gz', PhaseEncodingDirection='j', MultipartID='acq-multishell'),
]
g = group(*runs)
(concat,) = g.concatenation_groups.values()
print(concat.multipart_id, '->', concat.output_name)
```

## 5. Partial curation: curating some scans and not others

What if you curate *some* scans and leave the rest alone? QSIPrep's rule is
simple: **it guesses only when you have told it nothing.** In an uncurated
session, absent metadata means "nobody looked", and inferring a PEPOLAR
pairing is a service. Once anything in the session is curated, absent
metadata means "somebody looked and chose not to link these" — so the
heuristic switches off for the remaining scans, and a warning tells you
exactly what happened.

Here run-1 is curated into `pepolar01` and run-2 carries nothing. Run-2 does
**not** get an automatic estimation — if you had wanted the run-2 scans to
correct each other, you would have given them a `B0FieldIdentifier` too:

```{code-cell} ipython3
runs = [
    scan(
        'sub-01_dir-AP_run-1_dwi.nii.gz',
        PhaseEncodingDirection='j-',
        B0FieldIdentifier='pepolar01',
        B0FieldSource='pepolar01',
    ),
    scan(
        'sub-01_dir-PA_run-1_dwi.nii.gz',
        PhaseEncodingDirection='j',
        B0FieldIdentifier='pepolar01',
        B0FieldSource='pepolar01',
    ),
    scan('sub-01_dir-AP_run-2_dwi.nii.gz', PhaseEncodingDirection='j-'),
    scan('sub-01_dir-PA_run-2_dwi.nii.gz', PhaseEncodingDirection='j'),
]
g = group(*runs)
show(g, height=900)
```

Note the outputs: a correction boundary is a **correction-unit boundary** —
each unit is concatenated and corrected in its own pipeline, with exactly one
susceptibility correction. Corrected and uncorrected volumes never share a
final file, so the corrected run-1 pair becomes one output while the
uncorrected run-2 scans stand alone.

The safety net for unlinked scans is the **anatomical SDC reference**,
which stays available because — unlike sidecar metadata — it is under your
control at run time (`--sdc-anat-reference`; under the default `none`
nothing anatomical happens). Add a T2w to the same subject and ask for
`auto`, and run-2 is rescued, visibly tagged *inferred*, in its own
correction unit — and because every unit is now corrected, their corrected
results are concatenated into a single final output:

```{code-cell} ipython3
t2w = scan('sub-01_T2w.nii.gz', folder='anat', suffix='T2w', PhaseEncodingDirection=None, TotalReadoutTime=None)
g = group(*runs, t2w, sdc_anat_reference='auto')
print(report_text(g))
```

**Partial `MultipartID`** works the same way: series carrying one are combined
as asked, and series without one are *not* packaged with the curated groups —
each of their correction units becomes its own output file:

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_run-1_dwi.nii.gz', PhaseEncodingDirection='j-', MultipartID='combined'),
    scan('sub-01_dir-PA_run-1_dwi.nii.gz', PhaseEncodingDirection='j', MultipartID='combined'),
    scan('sub-01_dir-AP_run-2_dwi.nii.gz', PhaseEncodingDirection='j-'),
    scan('sub-01_dir-PA_run-2_dwi.nii.gz', PhaseEncodingDirection='j'),
]
g = group(*runs)
for concat in g.concatenation_groups.values():
    print(f'{concat.output_name}: {len(concat.dwi_files)} scan(s)')
print()
for issue in g.warnings:
    print(issue.render())
```

Note that the *estimation* still pools all four scans (none of them carries
`B0Field*` curation, and estimation membership is independent of the
packaging) — so each output borrows the others' b=0 images for its fieldmap.

## 6. `ShimSetting`: when the scanner re-shims

The scanner's shim state is part of the distortion signature. If run-2 was
acquired after a re-shim (different `ShimSetting` values), its distortion no
longer matches run-1 — so the runs get separate estimation pools and separate
correction units, each corrected with its own field, with a warning
explaining why. Because both units end up corrected, their corrected results
are then concatenated into one final output — the shim boundary changes how
the data are *corrected*, not how they are *packaged*.

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_run-1_dwi.nii.gz', PhaseEncodingDirection='j-', ShimSetting=(1.0, 2.0, 3.0)),
    scan('sub-01_dir-PA_run-1_dwi.nii.gz', PhaseEncodingDirection='j', ShimSetting=(1.0, 2.0, 3.0)),
    scan('sub-01_dir-AP_run-2_dwi.nii.gz', PhaseEncodingDirection='j-', ShimSetting=(9.0, 9.0, 9.0)),
    scan('sub-01_dir-PA_run-2_dwi.nii.gz', PhaseEncodingDirection='j', ShimSetting=(9.0, 9.0, 9.0)),
]
g = group(*runs)
print(report_text(g))
```

If you know the re-shim was inconsequential (or you want cross-shim correction
anyway), `--ignore shims` treats all shim values as compatible:

```{code-cell} ipython3
g = group(*runs, ignore_shims=True)
print(report_text(g))
```

## 7. No fieldmap at all: the anatomical SDC reference

With no fieldmap and no reverse-PE partner, a series is left uncorrected by
default — anatomical (fieldmap-less) correction is opt-in. The
`--sdc-anat-reference` flag names which anatomical-derived image serves as
the correction reference, and `auto` resolves it per subject: a T1w selects
the SyNb0 synthetic b=0, otherwise a T2w selects a **T2w registration**
(T2Wreg) — the distorted b=0 is registered to the undistorted T2w:

```{code-cell} ipython3
dwi = scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-')
t1w = scan('sub-01_T1w.nii.gz', folder='anat', suffix='T1w', PhaseEncodingDirection=None, TotalReadoutTime=None)
t2w = scan('sub-01_T2w.nii.gz', folder='anat', suffix='T2w', PhaseEncodingDirection=None, TotalReadoutTime=None)
g = group(dwi, t2w, sdc_anat_reference='auto')
show(g, height=680)
```

Naming a reference explicitly gets provenance **cli-override** (purple):
`synb0` synthesizes an undistorted b=0 from the T1w, and `invt1w` registers
an inverted-contrast T1w to a fieldmap atlas (the niworkflows SyN-SDC). A
real fieldmap always outranks the fallback — the reference only applies to
series that would otherwise go uncorrected, unless you escalate it with
`--force sdc-anat-reference`:

```{code-cell} ipython3
g = group(dwi, t1w, t2w, sdc_anat_reference='synb0')
print(report_text(g))
```

## 8. Same grouping, different pipelines

The grouping describes your *data*. How a processing backend consumes it is a
separate decision — and the previews spell out the difference. The FSL path
pools all b=0 images into one TOPUP estimation and models everything jointly in
eddy; the TORTOISE path motion-corrects each distortion group separately and
feeds blip-up/blip-down to DRBUDDI; the two-stage path runs TOPUP+eddy first
and then refines with DRBUDDI.

```{code-cell} ipython3
runs = [
    scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-'),
    scan('sub-01_dir-PA_dwi.nii.gz', PhaseEncodingDirection='j'),
]
g = group(*runs)
for backend in ('fsl', 'tortoise', 'mixed'):
    print(describe_processing(g, backend))
```

The text preview above is where the per-backend steps live. The HTML grouping
page concentrates on the grouping itself — which scans combine, which fieldmap
corrects which group, and the resulting sampling scheme.

## Appendix: `IntendedFor` (deprecated)

Older datasets link fieldmaps to DWI series with the `IntendedFor` field on
the fieldmap's sidecar. QSIPrep still honors it — the linkage is translated
into an estimation with provenance **IntendedFor** (blue) — but it is a
legacy mechanism. **Do not use it for new curation**; use
`B0FieldIdentifier`/`B0FieldSource` (section 3) instead.

```{code-cell} ipython3
dwi = scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-')
fmap = scan(
    'sub-01_dir-PA_epi.nii.gz',
    folder='fmap',
    PhaseEncodingDirection='j',
    IntendedFor=dwi.path,
)
g = group(dwi, fmap)
show(g, height=700)
```

Three rules keep the legacy path predictable:

**1. `B0Field*` supersedes it.** If a fieldmap carries *both* `IntendedFor`
and `B0FieldIdentifier`, the `B0Field*` links are used exclusively and the
`IntendedFor` is ignored, with a warning:

```{code-cell} ipython3
dwi = scan(
    'sub-01_dir-AP_dwi.nii.gz',
    PhaseEncodingDirection='j-',
    B0FieldIdentifier='my_fmap',  # its b=0 volumes feed the estimation...
    B0FieldSource='my_fmap',  # ...and the estimation corrects it
)
fmap = scan(
    'sub-01_dir-PA_epi.nii.gz',
    folder='fmap',
    PhaseEncodingDirection='j',
    B0FieldIdentifier='my_fmap',
    IntendedFor=dwi.path,  # ignored: B0FieldIdentifier supersedes it
)
g = group(dwi, fmap)
for issue in g.issues:
    print(issue.render())
```

**2. It counts as curation** (section 5): once an `IntendedFor` links any
series in a session, QSIPrep stops inferring reverse phase-encoding pairings
for the rest of it. Here the fieldmap is intended only for the AP series, so
the unlinked PA series gets no estimation and goes uncorrected:

```{code-cell} ipython3
ap = scan('sub-01_dir-AP_dwi.nii.gz', PhaseEncodingDirection='j-')
pa = scan('sub-01_dir-PA_dwi.nii.gz', PhaseEncodingDirection='j')
fmap = scan(
    'sub-01_dir-PA_epi.nii.gz',
    folder='fmap',
    PhaseEncodingDirection='j',
    IntendedFor=ap.path,  # links AP only; PA is left unlinked
)
g = group(ap, pa, fmap)
print(report_text(g))
```

**3. A fieldmap with neither field is never used.** It draws an
`unlinked-fmap` warning and is ignored entirely.

## Try it on your own data

Everything above runs from a real BIDS directory too — without processing
anything:

```bash
qsiplan /path/to/bids --html grouping.html
```

prints the grouping report plus the method-selection previews, and writes the
explanatory page you saw throughout this notebook (one per subject). If a
grouping decision surprises you, the provenance chip tells you which sidecar
field to set — `B0FieldIdentifier`, `B0FieldSource`, or `MultipartID` — to make
your intent explicit. Curate it once, and QSIPrep (and every other BIDS app)
will stop guessing.

```{note}
This page is a [MyST markdown notebook](https://myst-nb.readthedocs.io/):
the outputs above were executed during the documentation build. To run it
interactively, convert it with [jupytext](https://jupytext.readthedocs.io/)
(`jupytext --to notebook grouping_tutorial.md`) or open the `.md` directly
in JupyterLab or VS Code with the jupytext extension. It needs only an
environment where `qsiplan` is importable (`pip install qsiplan`).
```
