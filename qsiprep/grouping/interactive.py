"""Render a :class:`~.models.DWIGrouping` as a self-contained explanatory page.

:func:`render_html` returns a single HTML document (no external assets) that
reads top to bottom as a story:

1. **Fieldmaps** - one card per estimation, explaining the method in plain
   words, showing the blip-up/blip-down pairing for PEPOLAR, and stating why
   the estimation exists (curated, translated from IntendedFor, a flag, or
   inferred) plus the sidecar field that would change it.
2. **Outputs** - one box per output file. Concatenation is drawn as
   containment, not arrows: an output box holds its distortion groups, which
   hold the scan rows. Every scan appears exactly once; membership in a
   fieldmap estimation is a letter chip on the row, so identity is carried by
   letters (A, B, ...) while color stays dedicated to provenance. Borrowed
   b=0 sources are called out in a sentence. Each output ends with the
   plain-language processing steps for the chosen backend, with the other
   backends collapsed underneath.
3. **Notes** - the grouping's warnings and errors.

The page works without JavaScript; a small inline script adds hover
highlighting (an estimation card or letter chip lights up everything that
estimation touches).
"""

from __future__ import annotations

import html

from .models import CorrectionMethod, DWIGrouping, Provenance
from .report import processing_steps
from .validation import BACKENDS

#: Provenance value -> (fill, stroke).
_PROVENANCE_COLORS = {
    'curated': ('#dcfce7', '#16a34a'),
    'intendedfor': ('#dbeafe', '#2563eb'),
    'cli-override': ('#ede9fe', '#7c3aed'),
    'inferred': ('#fef3c7', '#d97706'),
    None: ('#f1f5f9', '#94a3b8'),
}

#: CorrectionMethod -> (title, one-sentence explanation for novices).
_METHOD_EXPLANATIONS = {
    CorrectionMethod.PEPOLAR: (
        'Reverse phase-encoding (PEPOLAR)',
        'Two sets of b=0 images were acquired with opposite phase encoding, so '
        'they are squished in opposite directions. Comparing them reveals the '
        'distortion field.',
    ),
    CorrectionMethod.DIRECT: (
        'Precomputed fieldmap',
        'A fieldmap image in Hz was provided directly.',
    ),
    CorrectionMethod.PHASEDIFF: (
        'GRE phase-difference fieldmap',
        'A gradient-echo fieldmap directly measures the B0 field inhomogeneity.',
    ),
    CorrectionMethod.PHASES: (
        'GRE two-phase fieldmap',
        'Two phase images at different echo times measure the B0 field.',
    ),
    CorrectionMethod.SYNB0: (
        'SyNb0 synthetic b=0',
        'No fieldmap was acquired: a synthetic undistorted b=0 is generated '
        'from the T1w and used as the missing opposite-blip image.',
    ),
    CorrectionMethod.T2WREG: (
        'Registration to T2w (T2Wreg)',
        'No fieldmap was acquired: the b=0 is registered to the undistorted '
        'T2w image to estimate the distortion.',
    ),
    CorrectionMethod.NIPREPS_SYN: (
        'Fieldmap-less SyN',
        'No fieldmap was acquired: a constrained ANTs registration of the '
        'inverted T1w to a fieldmap atlas approximates the distortion.',
    ),
}

_WHY_ESTIMATION = {
    Provenance.CURATED: 'You set <code>B0FieldIdentifier</code> in these sidecars '
    '&mdash; used as-is.',
    Provenance.TRANSLATED: 'Built from the deprecated <code>IntendedFor</code> field in the '
    'fieldmap sidecar. Prefer <code>B0FieldIdentifier</code>/<code>B0FieldSource</code>, '
    'which take precedence.',
    Provenance.FORCED: 'Requested by a command-line flag.',
    Provenance.INFERRED: 'QSIPrep found scans with opposite phase encoding and paired '
    'them automatically. Set <code>B0FieldIdentifier</code>/<code>B0FieldSource</code> '
    'to control this yourself.',
}

_WHY_CONCAT = {
    Provenance.CURATED: 'You set <code>MultipartID</code> on these scans &mdash; combined as-is.',
    Provenance.INFERRED: 'Combined automatically: same session and compatible shim '
    'settings. Set <code>MultipartID</code> (or use <code>--separate-all-dwis</code>) '
    'to change this.',
    Provenance.FORCED: 'Kept separate by <code>--separate-all-dwis</code>: every scan '
    'is its own output.',
}

_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0 auto;max-width:960px;padding:28px 24px;background:#f8fafc;color:#0f172a;
  font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
h1{font-size:20px;margin:0 0 2px}
h2{font-size:15px;margin:26px 0 10px;color:#334155}
.tagline{color:#475569;font-size:13.5px;margin:2px 0 8px}
.legend{font-size:12px;color:#64748b;margin:0}
.chip{display:inline-block;border:1.5px solid;border-radius:99px;padding:1px 9px;
  font-size:11.5px}
.chip.small{padding:0 7px;font-size:10.5px}
.est-rail{display:flex;gap:14px;flex-wrap:wrap}
.est{flex:1 1 280px;max-width:440px;border:2px solid;border-radius:10px;background:#fff;
  overflow:hidden;transition:box-shadow .1s}
.est-head{padding:8px 12px;font-size:13.5px;display:flex;align-items:center;gap:8px}
.est-body{padding:8px 12px 10px}
.badge{display:inline-flex;align-items:center;justify-content:center;width:20px;
  height:20px;border:2px solid;border-radius:50%;font-weight:700;font-size:11px;
  background:#fff;flex:none}
.badge.inline{width:16px;height:16px;font-size:9.5px;vertical-align:-3px;margin:0 2px}
.method{font-weight:600;font-size:12.5px;margin:0 0 3px}
.explain{font-size:11.5px;color:#475569;margin:0 0 6px;line-height:1.45}
.blips{display:flex;gap:8px;align-items:center;font-size:11.5px;font-weight:600;
  background:#f1f5f9;border-radius:6px;padding:4px 10px;margin:0 0 6px;width:fit-content}
.blips .vs{color:#64748b}
.srcs{font-size:11px;color:#475569;margin:0 0 4px}
.why{font-size:11px;margin:6px 0 0;line-height:1.45}
.why code{font-size:10.5px;background:#f1f5f9;padding:0 3px;border-radius:3px;
  color:inherit}
.unused{font-weight:400;font-size:11px;color:#64748b}
.output{background:#fff;border:1.5px solid #cbd5e1;border-radius:12px;margin:0 0 18px;
  padding:0 0 6px;overflow:hidden}
.out-head{background:#0f172a;color:#fff;padding:9px 14px;display:flex;gap:9px;
  align-items:center}
.out-name{font-weight:650;font-size:13px;font-family:ui-monospace,Menlo,monospace}
.out-count{margin-left:auto;font-size:11.5px;color:#94a3b8}
.out-why{padding:7px 14px 2px;color:#475569}
.dgroup{margin:8px 12px;border:1px solid #e2e8f0;border-left:5px solid;border-radius:7px;
  padding:7px 10px;background:#fcfdfe}
.dg-head{font-size:12.5px;display:flex;gap:7px;align-items:baseline;flex-wrap:wrap}
.dg-sig{color:#64748b;font-size:11.5px}
.dg-corr{margin-left:auto;font-size:11.5px}
.prov-word{font-size:10.5px}
.losing{color:#94a3b8;font-size:10.5px}
.nocorr{color:#b91c1c;font-weight:600}
.pol{font-size:11px}
.scan{display:flex;align-items:center;gap:7px;font-size:11.5px;padding:2.5px 0 0 20px}
.scan code{background:none;color:#334155}
.shells{color:#0e7490;font-size:10.5px;background:#ecfeff;border-radius:4px;padding:0 5px}
.borrow{font-size:11.5px;color:#475569;margin:4px 14px;background:#f8fafc;
  border:1px dashed #cbd5e1;border-radius:7px;padding:6px 10px}
.preview{margin:8px 12px 6px;font-size:12px;background:#f1f5f9;border-radius:8px;
  padding:7px 12px}
.preview summary{cursor:pointer;color:#334155;font-size:12px}
.preview ol{margin:8px 0 4px;padding-left:22px;line-height:1.55;color:#334155}
.preview li.issue{color:#b91c1c;list-style:none;margin-left:-14px}
.alt{margin:4px 0 4px 4px}
.alt summary{font-size:11.5px;color:#64748b}
.note{font-size:12px;border-radius:8px;padding:8px 12px;margin:0 0 8px;line-height:1.5}
.note.warning{background:#fffbeb;border:1px solid #fcd34d}
.note.error{background:#fef2f2;border:1px solid #fca5a5}
.none{font-size:13px;color:#b91c1c}
.hl{box-shadow:0 0 0 2.5px #0ea5e9}
"""

#: Hover an estimation card or letter chip -> highlight everything that
#: estimation touches (its card, its chips, the groups it corrects).
_JS = """
document.querySelectorAll('[data-est]').forEach(el => {
  const eid = el.dataset.est;
  el.addEventListener('mouseenter', () => {
    document.querySelectorAll('[data-est]').forEach(other => {
      if (other.dataset.est === eid) other.classList.add('hl');
    });
  });
  el.addEventListener('mouseleave', () => {
    document.querySelectorAll('.hl').forEach(other => other.classList.remove('hl'));
  });
});
"""


def _esc(text) -> str:
    return html.escape(str(text))


def _basename(path: str) -> str:
    return path.rsplit('/', 1)[-1]


def _prov_value(provenance) -> str | None:
    return provenance.value if isinstance(provenance, Provenance) else provenance


def _polarity_glyph(pe_dir: str | None) -> str:
    if not pe_dir:
        return '?'
    return '&#9660;' if pe_dir.endswith('-') else '&#9650;'  # filled down/up triangle


def _pe_phrase(pe_dir: str | None) -> str:
    """Spelled-out phrasing of a PhaseEncodingDirection value."""
    if not pe_dir:
        return 'phase encoding unknown'
    sign = 'negative' if pe_dir.endswith('-') else 'positive'
    return f'phase encoding {pe_dir} ({sign} along the {pe_dir[0]} axis)'


def _shell_text(record) -> str:
    if record.shelled is True:
        return 'b=' + '/'.join(str(int(centre)) for centre in record.shells)
    if record.shelled is False:
        return 'non-shelled sampling'
    return ''


def _badge(letter: str, stroke: str, eid: str, inline: bool = True) -> str:
    cls = 'badge inline' if inline else 'badge'
    return (
        f'<span class="{cls}" data-est="{_esc(eid)}" '
        f'style="border-color:{stroke};color:{stroke}">{letter}</span>'
    )


def _header(grouping: DWIGrouping) -> list[str]:
    n_scans = len(grouping.dwi_files)
    n_out = len(grouping.concatenation_groups)
    n_est = len(grouping.estimations)
    legend = ' '.join(
        f'<span class="chip" style="background:{fill};border-color:{stroke}">{label}</span>'
        for label, (fill, stroke) in [
            ('you curated it', _PROVENANCE_COLORS['curated']),
            ('command-line flag', _PROVENANCE_COLORS['cli-override']),
            ('QSIPrep guessed', _PROVENANCE_COLORS['inferred']),
            ('from IntendedFor (deprecated)', _PROVENANCE_COLORS['intendedfor']),
        ]
    )
    return [
        '<header>'
        f'<h1>How QSIPrep will process sub-{_esc(grouping.subject_id)}&rsquo;s '
        'diffusion data</h1>',
        f'<p class="tagline">{n_scans} DWI scan{"s" if n_scans != 1 else ""} &rarr; '
        f'{n_out} preprocessed output file{"s" if n_out != 1 else ""}, using '
        f'{n_est} fieldmap estimation{"s" if n_est != 1 else ""}</p>',
        f'<p class="legend">Colors show where each decision came from:&nbsp; {legend}</p>'
        '</header>',
    ]


def _blip_diagram(grouping: DWIGrouping, estimation) -> list[str]:
    """The blip-up/blip-down pairing summary on a PEPOLAR card."""
    parts = []
    for axis in sorted(estimation.pe_axes):
        up, down = [], []
        for path in estimation.sources:
            record = grouping.files.get(path)
            if record is None or not record.is_epi_like:
                continue
            if record.signature.pe_axis != axis:
                continue
            (down if (record.signature.pe_dir or '').endswith('-') else up).append(path)
        if up and down:
            parts.append(
                f'<div class="blips"><span>&#9650; {_esc(axis)} '
                f'({len(up)} scan{"s" if len(up) != 1 else ""})</span>'
                '<span class="vs">&harr;</span>'
                f'<span>&#9660; {_esc(axis)}&minus; '
                f'({len(down)} scan{"s" if len(down) != 1 else ""})</span></div>'
            )
        else:
            parts.append(f'<div class="blips one-way">{_esc(axis)} axis: one direction only</div>')
    if len(estimation.pe_axes) > 1 and not estimation.bidirectional_axes:
        parts.append(
            '<div class="blips">&#8646; no same-axis pair: the differing '
            'encodings are compared across axes</div>'
        )
    return parts


def _estimation_cards(grouping: DWIGrouping, letters: dict[str, str]) -> list[str]:
    parts = ['<section><h2>Step 1 &mdash; Fieldmaps: how distortion will be measured</h2>']
    if not grouping.estimations:
        parts.append(
            '<p class="none">No fieldmap estimations: susceptibility distortion '
            'will NOT be corrected.</p></section>'
        )
        return parts

    parts.append('<div class="est-rail">')
    applied = {b0field_id for b0field_id in grouping.application.values() if b0field_id}
    for eid, estimation in sorted(grouping.estimations.items()):
        fill, stroke = _PROVENANCE_COLORS[_prov_value(estimation.provenance)]
        title, explain = _METHOD_EXPLANATIONS[estimation.method]
        unused = '' if eid in applied else ' <span class="unused">(not used)</span>'
        parts.append(
            f'<div class="est" data-est="{_esc(eid)}" style="border-color:{stroke}">'
            f'<div class="est-head" style="background:{fill}">'
            f'{_badge(letters[eid], stroke, eid, inline=False)}'
            f'<b>{_esc(eid)}</b>{unused}</div>'
            f'<div class="est-body"><p class="method">{title}</p>'
            f'<p class="explain">{explain}</p>'
        )
        if estimation.is_pepolar:
            parts.extend(_blip_diagram(grouping, estimation))
        by_datatype = {'fmap': [], 'anat': [], 'dwi': []}
        for path in estimation.sources:
            record = grouping.files.get(path)
            if record is not None and record.datatype in by_datatype:
                by_datatype[record.datatype].append(path)
        for datatype in ('fmap', 'anat'):
            if by_datatype[datatype]:
                names = ', '.join(_esc(_basename(path)) for path in by_datatype[datatype])
                parts.append(f'<p class="srcs">from <code>{datatype}/</code>: {names}</p>')
        if by_datatype['dwi']:
            n_dwi = len(by_datatype['dwi'])
            parts.append(
                f'<p class="srcs">uses b=0 volumes from {n_dwi} DWI '
                f'scan{"s" if n_dwi != 1 else ""} &mdash; marked '
                f'{_badge(letters[eid], stroke, eid)} below</p>'
            )
        parts.append(
            f'<p class="why" style="color:{stroke}">{_WHY_ESTIMATION[estimation.provenance]}</p>'
            '</div></div>'
        )
    parts.append('</div></section>')
    return parts


def _correction_phrase(grouping: DWIGrouping, dgroup, letters: dict[str, str]) -> tuple[str, str]:
    """(stroke color, HTML phrase) describing what corrects ``dgroup``."""
    source = dgroup.b0field_source
    if source is None:
        return (
            _PROVENANCE_COLORS[None][1],
            '<span class="nocorr">&#9888; no distortion correction</span>',
        )
    estimation = grouping.estimations[source]
    _, stroke = _PROVENANCE_COLORS[_prov_value(estimation.provenance)]
    app_provenance = grouping.application_provenance[dgroup.dwi_files[0]]
    phrase = (
        f'corrected by {_badge(letters[source], stroke, source)} {_esc(source)} '
        f'<span class="prov-word" style="color:{stroke}">[{app_provenance.value}]</span>'
    )
    losing = [
        candidate
        for candidate in grouping.application_candidates.get(dgroup.dwi_files[0], ())
        if candidate != source
    ]
    if losing:
        names = ', '.join(f'{letters.get(c, "?")} {_esc(c)}' for c in losing)
        phrase += f' <span class="losing">(also eligible: {names})</span>'
    return stroke, phrase


def _scan_row(grouping: DWIGrouping, path: str, letters: dict[str, str]) -> str:
    record = grouping.files[path]
    shells = _shell_text(record)
    chips = ''.join(
        _badge(letters[eid], _PROVENANCE_COLORS[_prov_value(est.provenance)][1], eid)
        for eid, est in sorted(grouping.estimations.items())
        if path in est.sources
    )
    shell_span = f'<span class="shells">{_esc(shells)}</span>' if shells else ''
    return f'<div class="scan"><code>{_esc(_basename(path))}</code>{shell_span}{chips}</div>'


def _preview_list(steps: list[str]) -> list[str]:
    parts = ['<ol>']
    for step in steps:
        if step.startswith('!!'):
            parts.append(f'<li class="issue">&#10060; {_esc(step.lstrip("! "))}</li>')
        else:
            parts.append(f'<li>{_esc(step)}</li>')
    parts.append('</ol>')
    return parts


def _output_boxes(grouping: DWIGrouping, letters: dict[str, str], backend: str) -> list[str]:
    previews = {b: processing_steps(grouping, b) for b in BACKENDS}
    parts = [
        '<section><h2>Step 2 &mdash; Outputs: which scans are combined, '
        'and how each is corrected</h2>'
    ]
    for multipart_id, concat in sorted(grouping.concatenation_groups.items()):
        cfill, cstroke = _PROVENANCE_COLORS[_prov_value(concat.provenance)]
        n_scans = len(concat.dwi_files)
        parts.append(
            '<div class="output">'
            f'<div class="out-head"><span class="out-icon">&#128190;</span>'
            f'<span class="out-name">{_esc(concat.output_name)}</span>'
            f'<span class="out-count">one output file &middot; {n_scans} '
            f'scan{"s" if n_scans != 1 else ""} combined</span></div>'
            f'<p class="why out-why"><span class="chip small" style="background:{cfill};'
            f'border-color:{cstroke}">{_esc(concat.provenance.value)}</span> '
            f'{_WHY_CONCAT.get(concat.provenance, _esc(concat.provenance.value))}</p>'
        )
        for dgroup in grouping.distortion_groups_in(multipart_id):
            stroke, correction = _correction_phrase(grouping, dgroup, letters)
            signature = dgroup.signature
            trt = (
                f'readout {signature.readout_time:g}&thinsp;s'
                if signature.readout_time is not None
                else ''
            )
            parts.append(
                f'<div class="dgroup" style="border-left-color:{stroke}">'
                f'<div class="dg-head"><span class="pol">{_polarity_glyph(signature.pe_dir)}'
                f'</span> <b>{_esc(dgroup.key)}</b>'
                f'<span class="dg-sig">{_pe_phrase(signature.pe_dir)}'
                f'{" &middot; " + trt if trt else ""}</span>'
                f'<span class="dg-corr">{correction}</span></div>'
            )
            parts.extend(_scan_row(grouping, path, letters) for path in dgroup.dwi_files)
            parts.append('</div>')

        for eid, paths in sorted(grouping.borrowed_sources(multipart_id).items()):
            _, stroke = _PROVENANCE_COLORS[_prov_value(grouping.estimations[eid].provenance)]
            names = ', '.join(f'<code>{_esc(_basename(path))}</code>' for path in paths)
            parts.append(
                f'<p class="borrow">&#8618; fieldmap {_badge(letters[eid], stroke, eid)} '
                f'also borrows b=0 volumes from {names} &mdash; those scans are '
                '<b>not</b> part of this output file.</p>'
            )

        steps = previews[backend].get(concat.output_name, [])
        if steps:
            parts.append(
                '<details class="preview" open><summary>What will happen to this data '
                f'(<b>{_esc(backend)}</b> workflow)</summary>'
            )
            parts.extend(_preview_list(steps))
            for other in BACKENDS:
                other_steps = previews[other].get(concat.output_name, [])
                if other == backend or not other_steps:
                    continue
                parts.append(
                    f'<details class="alt"><summary>if run with the {_esc(other)} '
                    'workflow instead&hellip;</summary>'
                )
                parts.extend(_preview_list(other_steps))
                parts.append('</details>')
            parts.append('</details>')
        parts.append('</div>')
    parts.append('</section>')
    return parts


def _issue_notes(grouping: DWIGrouping) -> list[str]:
    if not grouping.issues:
        return []
    parts = ['<section><h2>Things you may want to know</h2>']
    for issue in grouping.issues:
        icon = '&#10060;' if issue.severity == 'error' else '&#9888;&#65039;'
        parts.append(
            f'<div class="note {_esc(issue.severity)}">{icon} '
            f'<b>{_esc(issue.code)}</b>: {_esc(issue.message)}</div>'
        )
    parts.append('</section>')
    return parts


def render_html(grouping: DWIGrouping, backend: str = 'fsl') -> str:
    """Return a standalone explanatory HTML document for ``grouping``.

    ``backend`` selects which workflow's processing preview is expanded on
    each output; the other backends are collapsed underneath it.
    """
    letters = {
        eid: chr(ord('A') + index) for index, eid in enumerate(sorted(grouping.estimations))
    }
    parts = _header(grouping)
    parts.extend(_estimation_cards(grouping, letters))
    parts.extend(_output_boxes(grouping, letters, backend))
    parts.extend(_issue_notes(grouping))
    body = ''.join(parts)
    return (
        '<!doctype html>\n'
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>DWI grouping for sub-{_esc(grouping.subject_id)}</title>\n'
        f'<style>{_CSS}</style></head>\n'
        f'<body>{body}<script>{_JS}</script></body></html>'
    )
