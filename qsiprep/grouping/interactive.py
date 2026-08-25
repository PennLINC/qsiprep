"""Render a :class:`~.models.DWIGrouping` as a self-contained explanatory page.

The grouping widget is produced two ways from the same markup:
:func:`render_html` returns a full standalone HTML document (for
``qsiprep-group --html``, phrased as a plan); :func:`render_report_segment`
returns a scoped fragment inlined directly into the subject report (phrased in
past tense, for a run that already happened). It reads top to bottom as a story:

1. **Fieldmaps** - one card per estimation, explaining the method in plain
   words, showing the blip-up/blip-down pairing for PEPOLAR, and stating why
   the estimation exists (curated, translated from IntendedFor, a flag, or
   inferred) plus the sidecar field that would change it.
2. **Outputs** - one box per output file. Concatenation is drawn as
   containment, not arrows: an output box holds its distortion groups, which
   hold the scan rows. Every scan appears exactly once; membership in a
   fieldmap estimation is a letter chip on the row, so identity is carried by
   letters (A, B, ...) while color stays dedicated to provenance. Borrowed
   b=0 sources are called out in a sentence. Each output ends with an
   interactive view of its concatenated sampling scheme. (The step-by-step
   processing narrative lives in a separate workflow display, not here.)
3. **Notes** - the grouping's warnings and errors.

The page works without JavaScript; a small inline script adds hover
highlighting (an estimation card or letter chip lights up everything that
estimation touches).
"""

from __future__ import annotations

import html

from ..viz.pipeline import _embedded_json, pipeline_assets, pipeline_div, plan_payload
from ..viz.qspace import q_points, scheme_div, scheme_payload, viewer_assets
from .metadata import B0_THRESHOLD, read_bvals_bvecs, sibling_bval, sibling_bvec
from .methods import reachable_selections
from .models import CorrectionMethod, DWIGrouping, Provenance
from .plan import compile_plan
from .report import processing_steps, shell_label

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

#: The root class every rule is scoped under, so the page can be inlined
#: directly into the subject report (a Bootstrap document) without its global
#: ``body``/``h1``/``h2`` styles leaking out, and without Bootstrap's ``.badge``
#: / ``code`` / heading rules leaking in. Also the standalone page's container.
ROOT_CLASS = 'qsi-grouping'

_CSS = """
.qsi-grouping{color-scheme:light;box-sizing:border-box;margin:0 auto;max-width:960px;
  padding:28px 24px;background:#f8fafc;color:#0f172a;line-height:1.4;
  font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
.qsi-grouping *{box-sizing:border-box}
.qsi-grouping code{font-family:ui-monospace,Menlo,monospace;color:inherit}
.qsi-grouping h1{font-size:20px;font-weight:700;line-height:1.2;margin:0 0 2px}
.qsi-grouping h2{font-size:15px;font-weight:700;line-height:1.2;margin:26px 0 10px;
  color:#334155}
.qsi-grouping .tagline{color:#475569;font-size:13.5px;margin:2px 0 8px}
.qsi-grouping .legend{font-size:12px;color:#64748b;margin:0}
.qsi-grouping .chip{display:inline-block;border:1.5px solid;border-radius:99px;
  padding:1px 9px;font-size:11.5px}
.qsi-grouping .chip.small{padding:0 7px;font-size:10.5px}
.qsi-grouping .est-rail{display:flex;gap:14px;flex-wrap:wrap}
.qsi-grouping .est{flex:1 1 280px;max-width:440px;border:2px solid;border-radius:10px;
  background:#fff;overflow:hidden;transition:box-shadow .1s}
.qsi-grouping .est-head{padding:8px 12px;font-size:13.5px;display:flex;
  align-items:center;gap:8px}
.qsi-grouping .est-body{padding:8px 12px 10px}
.qsi-grouping .badge{display:inline-flex;align-items:center;justify-content:center;
  width:20px;height:20px;padding:0;line-height:1;border:2px solid;border-radius:50%;
  font-weight:700;font-size:11px;background:#fff;flex:none}
.qsi-grouping .badge.inline{width:16px;height:16px;font-size:9.5px;
  vertical-align:-3px;margin:0 2px}
.qsi-grouping .method{font-weight:600;font-size:12.5px;margin:0 0 3px}
.qsi-grouping .explain{font-size:11.5px;color:#475569;margin:0 0 6px;line-height:1.45}
.qsi-grouping .blips{display:flex;gap:8px;align-items:center;font-size:11.5px;
  font-weight:600;background:#f1f5f9;border-radius:6px;padding:4px 10px;margin:0 0 6px;
  width:fit-content}
.qsi-grouping .blips .vs{color:#64748b}
.qsi-grouping .srcs{font-size:11px;color:#475569;margin:0 0 4px}
.qsi-grouping .why{font-size:11px;margin:6px 0 0;line-height:1.45}
.qsi-grouping .why code{font-size:10.5px;background:#f1f5f9;padding:0 3px;
  border-radius:3px;color:inherit}
.qsi-grouping .unused{font-weight:400;font-size:11px;color:#64748b}
.qsi-grouping .output{background:#fff;border:1.5px solid #cbd5e1;border-radius:12px;
  margin:0 0 18px;padding:0 0 6px;overflow:hidden}
.qsi-grouping .out-head{background:#0f172a;color:#fff;padding:9px 14px;display:flex;
  gap:9px;align-items:center}
.qsi-grouping .out-name{font-weight:650;font-size:13px;
  font-family:ui-monospace,Menlo,monospace}
.qsi-grouping .out-count{margin-left:auto;font-size:11.5px;color:#94a3b8}
.qsi-grouping .out-why{padding:7px 14px 2px;color:#475569}
.qsi-grouping .dgroup{margin:8px 12px;border:1px solid #e2e8f0;border-left:5px solid;
  border-radius:7px;padding:7px 10px;background:#fcfdfe}
.qsi-grouping .dg-head{font-size:12.5px;display:flex;gap:7px;align-items:baseline;
  flex-wrap:wrap}
.qsi-grouping .dg-sig{color:#64748b;font-size:11.5px}
.qsi-grouping .dg-corr{margin-left:auto;font-size:11.5px}
.qsi-grouping .prov-word{font-size:10.5px}
.qsi-grouping .losing{color:#94a3b8;font-size:10.5px}
.qsi-grouping .nocorr{color:#b91c1c;font-weight:600}
.qsi-grouping .pol{font-size:11px}
.qsi-grouping .scan{display:flex;align-items:center;gap:7px;font-size:11.5px;
  padding:2.5px 0 0 20px}
.qsi-grouping .scan code{background:none;color:#334155}
.qsi-grouping .shells{color:#0e7490;font-size:10.5px;background:#ecfeff;
  border-radius:4px;padding:0 5px}
.qsi-grouping .borrow{font-size:11.5px;color:#475569;margin:4px 14px;background:#f8fafc;
  border:1px dashed #cbd5e1;border-radius:7px;padding:6px 10px}
.qsi-grouping .cunit{border:1.5px dashed #94a3b8;border-radius:9px;margin:8px 12px;
  padding:0 0 4px;background:#f8fafc}
.qsi-grouping .cunit .dgroup{margin:6px 10px;background:#fff}
.qsi-grouping .cu-head{font-size:11px;color:#475569;padding:7px 12px 0}
.qsi-grouping .cu-note{color:#94a3b8;margin-left:6px}
.qsi-grouping .final-concat{font-size:11.5px;font-weight:600;color:#334155;
  margin:6px 14px 8px}
.qsi-grouping .note{font-size:12px;border-radius:8px;padding:8px 12px;margin:0 0 8px;
  line-height:1.5}
.qsi-grouping .note.warning{background:#fffbeb;border:1px solid #fcd34d}
.qsi-grouping .note.error{background:#fef2f2;border:1px solid #fca5a5}
.qsi-grouping .none{font-size:13px;color:#b91c1c}
.qsi-grouping .hl{box-shadow:0 0 0 2.5px #0ea5e9}
.qsi-grouping .scheme-block{margin:10px 12px 0;padding:10px 2px 4px;
  border-top:1px solid #e2e8f0}
.qsi-grouping .scheme-label{font-size:11.5px;font-weight:600;color:#0369a1;margin:0 0 8px}
.qsi-grouping .scheme-missing{font-size:12px;color:#94a3b8;padding:6px 0}
.qsi-grouping .plan-controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;
  margin:0 0 10px}
.qsi-grouping .plan-controls label{font-size:11.5px;color:#334155;
  font-family:ui-monospace,Menlo,monospace;display:flex;gap:6px;align-items:center}
.qsi-grouping .plan-controls select{font:inherit;font-size:12px;color:#0f172a;
  border:1.5px solid #cbd5e1;border-radius:7px;background:#fff;padding:2px 6px}
.qsi-grouping .plan-panel{display:none}
.qsi-grouping .plan-panel.on{display:block}
.qsi-grouping .plan-cli{font-size:11.5px;color:#64748b;
  font-family:ui-monospace,Menlo,monospace;margin:0 0 10px}
.qsi-grouping .plan-prose{margin:2px 0 14px}
.qsi-grouping .plan-prose summary{font-size:11.5px;color:#0369a1;cursor:pointer}
.qsi-grouping .plan-out{font-size:11.5px;font-weight:650;color:#334155;
  font-family:ui-monospace,Menlo,monospace;margin:8px 0 2px}
.qsi-grouping .plan-steps{font-size:12px;color:#334155;margin:2px 0 6px;
  padding-left:22px;line-height:1.5}
.qsi-grouping .plan-steps li.issue{list-style:none;margin-left:-16px;color:#92400e}
.qsi-grouping .plan-steps li.issue-error{color:#b91c1c}
"""

#: Everything is scoped to each `.qsi-grouping` root so the script is inert
#: against the rest of the (Bootstrap) report page and stays correct if more
#: than one grouping widget is ever inlined into a single document.
_JS = """
document.querySelectorAll('.qsi-grouping').forEach(root => {
  // Hover an estimation card or letter chip -> highlight everything that
  // estimation touches (its card, its chips, the groups it corrects).
  root.querySelectorAll('[data-est]').forEach(el => {
    const eid = el.dataset.est;
    el.addEventListener('mouseenter', () => {
      root.querySelectorAll('[data-est]').forEach(other => {
        if (other.dataset.est === eid) other.classList.add('hl');
      });
    });
    el.addEventListener('mouseleave', () => {
      root.querySelectorAll('.hl').forEach(other => other.classList.remove('hl'));
    });
  });
  // Interactive processing plan: the method controls pick a flag
  // combination; the provider maps its canonical key to a plan payload and
  // the diagram re-renders. Today the provider reads the embedded index;
  // a live host swaps in one that fetches /plan?<key> instead.
  root.querySelectorAll('.plan-interactive').forEach(section => {
    const index = JSON.parse(
      section.querySelector('script.plan-payloads').textContent);
    const provider = key => Promise.resolve(index[key]);
    const hmc = section.querySelector('.ctl-hmc');
    const model = section.querySelector('.ctl-model');
    const modelWrap = section.querySelector('.ctl-model-wrap');
    const sdc = section.querySelector('.ctl-sdc');
    const host = section.querySelector('.plan-host');
    const cli = section.querySelector('.plan-cli');
    const SDC_OPTIONS = {
      eddy: ['topup', 'drbuddi', 'topup+drbuddi'],
      shoreline: ['drbuddi'],
      tortoise: ['drbuddi'],
    };
    const currentKey = () => {
      const parts = ['hmc-method=' + hmc.value];
      if (hmc.value === 'shoreline') parts.push('shoreline-model=' + model.value);
      parts.push('sdc-method=' + sdc.value);
      return parts.join('&');
    };
    const update = () => {
      const options = SDC_OPTIONS[hmc.value];
      if (!options.includes(sdc.value)) sdc.value = options[0];
      [...sdc.options].forEach(option => {
        option.hidden = !options.includes(option.value);
        option.disabled = option.hidden;
      });
      modelWrap.style.display = hmc.value === 'shoreline' ? '' : 'none';
      const key = currentKey();
      provider(key).then(payload => {
        if (!payload) return;
        window.QSIPrepPipeline.render(host, payload);
        cli.textContent = payload.selection.label + '  (' + payload.selection.cli + ')';
        section.querySelectorAll('.plan-prose').forEach(details => {
          details.style.display = details.dataset.planKey === key ? '' : 'none';
        });
      });
    };
    [hmc, model, sdc].forEach(control => control.addEventListener('change', update));
    update();
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
        return shell_label(record)
    if record.shelled is False:
        return 'non-shelled sampling'
    return ''


def _badge(letter: str, stroke: str, eid: str, inline: bool = True) -> str:
    cls = 'badge inline' if inline else 'badge'
    return (
        f'<span class="{cls}" data-est="{_esc(eid)}" '
        f'style="border-color:{stroke};color:{stroke}">{letter}</span>'
    )


def _header(grouping: DWIGrouping, past: bool = False) -> list[str]:
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
    processed = 'processed' if past else 'will process'
    return [
        '<header>'
        f'<h1>How QSIPrep {processed} sub-{_esc(grouping.subject_id)}&rsquo;s '
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


def _estimation_cards(
    grouping: DWIGrouping, letters: dict[str, str], past: bool = False
) -> list[str]:
    measured = 'was measured' if past else 'will be measured'
    parts = [f'<section><h2>Step 1 &mdash; Fieldmaps: how distortion {measured}</h2>']
    if not grouping.estimations:
        not_corrected = 'was NOT corrected' if past else 'will NOT be corrected'
        parts.append(
            f'<p class="none">No fieldmap estimations: susceptibility distortion '
            f'{not_corrected}.</p></section>'
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


def _load_gradients(path: str):
    """``(bvals, bvecs)`` from a DWI's sidecars, or ``None`` if unreadable.

    ``bvals`` is a list of floats and ``bvecs`` a list of ``(x, y, z)``
    lists. Returning ``None`` on a missing or malformed sidecar lets the
    report degrade to a short notice instead of failing.
    """
    try:
        bvals, bvecs = read_bvals_bvecs(sibling_bval(path), sibling_bvec(path))
    except (OSError, ValueError):
        return None
    return bvals.tolist(), bvecs.tolist()


def _scheme_data(grouping: DWIGrouping, concat) -> dict | None:
    """Viewer payload for one output's concatenated sampling scheme.

    Each volume becomes a q-space point ``sqrt(b) * bvec`` tagged with its
    source file and phase-encoding direction, so the viewer can color by
    either. Returns ``None`` when no member has readable gradients.
    """
    coords, meta, files, pes = [], [], [], []
    for path in concat.dwi_files:
        loaded = _load_gradients(path)
        if loaded is None:
            continue
        bvals, bvecs = loaded
        record = grouping.files.get(path)
        pe = (record.signature.pe_dir if record else None) or 'unknown'
        if pe not in pes:
            pes.append(pe)
        file_index = len(files)
        files.append(_basename(path))
        coords.extend(q_points(bvals, bvecs))
        meta.extend({'b': int(round(bval)), 'file': file_index, 'pe': pe} for bval in bvals)
    if not coords:
        return None
    panels = [{'title': concat.output_name, 'coords': coords}]
    return scheme_payload(panels, meta, files, pes, b0_threshold=B0_THRESHOLD)


def _scheme_view(grouping: DWIGrouping, concat) -> str:
    """The sampling-scheme body: the interactive viewer, or a notice."""
    data = _scheme_data(grouping, concat)
    if data is None:
        return (
            '<p class="scheme-missing">Sampling scheme unavailable '
            '(no readable <code>.bval</code>/<code>.bvec</code>).</p>'
        )
    return scheme_div(data)


def _output_boxes(grouping: DWIGrouping, letters: dict[str, str], past: bool = False) -> list[str]:
    combined = 'were combined, and how each was' if past else 'are combined, and how each is'
    parts = [f'<section><h2>Step 2 &mdash; Outputs: which scans {combined} corrected</h2>']
    for concat_key, concat in sorted(grouping.concatenation_groups.items()):
        cfill, cstroke = _PROVENANCE_COLORS[_prov_value(concat.provenance)]
        n_scans = len(concat.dwi_files)
        why = _WHY_CONCAT.get(concat.provenance, _esc(concat.provenance.value))
        if concat.provenance is Provenance.CURATED and concat.multipart_id.startswith('acq-'):
            why += ' The <code>acq-</code> prefix also names the output file.'
        parts.append(
            '<div class="output">'
            f'<div class="out-head"><span class="out-icon">&#128190;</span>'
            f'<span class="out-name">{_esc(concat.output_name)}</span>'
            f'<span class="out-count">one output file &middot; {n_scans} '
            f'scan{"s" if n_scans != 1 else ""} combined</span></div>'
            f'<p class="why out-why"><span class="chip small" style="background:{cfill};'
            f'border-color:{cstroke}">{_esc(concat.provenance.value)}</span> '
            f'{why}</p>'
        )
        units = grouping.correction_units_in(concat_key)
        multi_unit = len(units) > 1
        for unit in units:
            if multi_unit:
                parts.append(
                    f'<div class="cunit"><div class="cu-head">Correction unit '
                    f'<b>{_esc(unit.key)}</b> <span class="cu-note">one pipeline, '
                    'one susceptibility correction</span></div>'
                )
            for dgroup_key in unit.distortion_groups:
                dgroup = grouping.distortion_groups[dgroup_key]
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
            if multi_unit:
                parts.append('</div>')
        if multi_unit:
            resampled = 'were' if past else 'are'
            parts.append(
                f'<p class="final-concat">&#10515; The corrected results of these '
                f'{len(units)} units {resampled} resampled onto one grid and combined '
                'into this output file.</p>'
            )

        for eid, paths in sorted(grouping.borrowed_sources(concat_key).items()):
            _, stroke = _PROVENANCE_COLORS[_prov_value(grouping.estimations[eid].provenance)]
            names = ', '.join(f'<code>{_esc(_basename(path))}</code>' for path in paths)
            borrows = 'borrowed' if past else 'also borrows'
            parts.append(
                f'<p class="borrow">&#8618; fieldmap {_badge(letters[eid], stroke, eid)} '
                f'{borrows} b=0 volumes from {names} &mdash; those scans are '
                '<b>not</b> part of this output file.</p>'
            )

        # The concatenated sampling scheme for this output.
        parts.append(
            '<div class="scheme-block"><p class="scheme-label">&#9673; Sampling scheme</p>'
            f'{_scheme_view(grouping, concat)}</div>'
        )
    parts.append('</section>')
    return parts


def _prose_steps(grouping: DWIGrouping, selection) -> str:
    """The numbered step-by-step text for one selection, as list markup."""
    parts = []
    for output_name, steps in processing_steps(grouping, selection).items():
        parts.append(f'<p class="plan-out">{_esc(output_name)}</p><ol class="plan-steps">')
        for step in steps:
            if step.startswith('!!'):
                severity = 'issue-error' if 'ERROR' in step else ''
                parts.append(f'<li class="issue {severity}">{_esc(step.lstrip("! "))}</li>')
            else:
                parts.append(f'<li>{_esc(step)}</li>')
        parts.append('</ol>')
    return ''.join(parts)


def _plan_panel(grouping: DWIGrouping, selection) -> str:
    """A single selection's processing plan: the flow diagram plus prose steps."""
    plan = compile_plan(grouping, selection)
    return ''.join(
        [
            '<div class="plan-panel on">',
            f'<p class="plan-cli">{_esc(selection.cli_phrase())}</p>',
            pipeline_div(plan_payload(grouping, plan)),
            '<details class="plan-prose"><summary>Step-by-step description</summary>',
            _prose_steps(grouping, selection),
            '</details></div>',
        ]
    )


def _plan_controls(selections) -> str:
    """The --hmc-method/--shoreline-model/--sdc-method dropdown row."""
    hmc_values = list(dict.fromkeys(sel.hmc.value for sel in selections))
    model_values = list(
        dict.fromkeys(sel.shoreline_model for sel in selections if sel.shoreline_model)
    )
    sdc_values = list(
        dict.fromkeys(
            '+'.join(tool.value for tool in sel.pepolar_tools) for sel in selections
        )
    )

    def options(values):
        return ''.join(f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in values)

    return (
        '<div class="plan-controls">'
        f'<label>--hmc-method <select class="ctl-hmc">{options(hmc_values)}</select></label>'
        '<label class="ctl-model-wrap">--shoreline-model '
        f'<select class="ctl-model">{options(model_values)}</select></label>'
        f'<label>--sdc-method <select class="ctl-sdc">{options(sdc_values)}</select></label>'
        '</div>'
    )


def _plan_section(grouping: DWIGrouping, selections, past: bool = False) -> list[str]:
    """The processing-plan section.

    With one selection (the subject report's executed run) it is a single
    static panel. With several (the standalone page) it becomes interactive:
    dropdowns mirroring the CLI flags pick a combination, whose canonical
    key (:meth:`~.methods.MethodSelection.combination_key`) looks up a
    precompiled payload in the embedded index and re-renders the diagram -
    the same key a live ``/plan`` endpoint would take.
    """
    if not selections:
        return []
    heading = (
        'Step 3 &mdash; Processing: how the data was corrected'
        if past
        else 'Step 3 &mdash; Processing: what will happen'
    )
    parts = [f'<section><h2>{heading}</h2>']
    if len(selections) == 1:
        parts.append(_plan_panel(grouping, selections[0]))
    else:
        payloads = {}
        prose = []
        for selection in selections:
            key = selection.combination_key()
            payloads[key] = plan_payload(grouping, compile_plan(grouping, selection))
            prose.append(
                f'<details class="plan-prose" data-plan-key="{_esc(key)}" '
                'style="display:none">'
                '<summary>Step-by-step description</summary>'
                f'{_prose_steps(grouping, selection)}</details>'
            )
        parts.append('<div class="plan-interactive">')
        parts.append(_plan_controls(selections))
        parts.append('<p class="plan-cli"></p>')
        parts.append('<div class="pipeline-viewer plan-host"></div>')
        parts.append(
            '<script type="application/json" class="plan-payloads">'
            f'{_embedded_json(payloads)}</script>'
        )
        parts.extend(prose)
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


def _body(grouping: DWIGrouping, letters: dict[str, str], past: bool, selections) -> str:
    """The grouping page's inner markup (no container, styles, or scripts)."""
    parts = _header(grouping, past)
    parts.extend(_estimation_cards(grouping, letters, past))
    parts.extend(_output_boxes(grouping, letters, past))
    parts.extend(_plan_section(grouping, selections, past))
    parts.extend(_issue_notes(grouping))
    return ''.join(parts)


def _fragment(grouping: DWIGrouping, past: bool, selections=()) -> str:
    """The scoped, self-contained grouping widget: styles + markup + scripts.

    Everything is scoped under :data:`ROOT_CLASS` and carried inline (the shared
    q-space and pipeline viewers' assets included), so the fragment drops
    straight into a host page with no iframe: the host's styles do not reach in,
    and these do not leak out. ``past`` picks report tense (a run that happened)
    vs plan tense (a run that has not). ``selections`` are the method
    selections whose processing plans the page shows.
    """
    letters = {
        eid: chr(ord('A') + index) for index, eid in enumerate(sorted(grouping.estimations))
    }
    viewer_css, viewer_js = viewer_assets()
    plan_css, plan_js = pipeline_assets()
    return (
        f'<style>{_CSS}\n{viewer_css}\n{plan_css}</style>'
        f'<div class="{ROOT_CLASS}">{_body(grouping, letters, past, selections)}</div>'
        f'<script>{viewer_js}</script>'
        f'<script>{plan_js}</script>'
        f'<script>{_JS}</script>'
    )


def render_report_segment(grouping: DWIGrouping, selection=None) -> str:
    """The grouping widget for inlining into the subject report (no iframe).

    The report describes a run that already happened, so the wording is past
    tense; ``selection`` is the method selection that actually ran, whose plan
    is drawn as the single processing panel. See :func:`_fragment` for how the
    styles/scripts stay isolated.
    """
    selections = [selection] if selection is not None else []
    return _fragment(grouping, past=True, selections=selections)


def render_html(grouping: DWIGrouping, selections=None) -> str:
    """Return a standalone explanatory HTML document for ``grouping``.

    Wraps the grouping fragment in a minimal page shell. Used by
    ``qsiprep-group --html``, which previews a run that has not happened yet, so
    the wording is future tense and the processing-plan section is interactive:
    dropdown controls over the requested method selections (default: every
    reachable combination) re-render the flow diagram.
    """
    if selections is None:
        selections = reachable_selections()
    return (
        '<!doctype html>\n'
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>DWI grouping for sub-{_esc(grouping.subject_id)}</title>\n'
        '<style>body{margin:0}</style></head>\n'
        f'<body>{_fragment(grouping, past=False, selections=selections)}</body></html>'
    )
