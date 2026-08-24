"""Build the interactive q-space sampling-scheme viewer.

Shared by the group report (:mod:`qsiprep.grouping.interactive`) and the subject
report (:class:`qsiprep.interfaces.reports.GradientPlot`) so gradient schemes
are depicted identically in both. The self-contained JS/CSS assets live in
``qsiprep/data/qspace_viewer.{js,css}``.

Two embedding styles are offered:

- :func:`scheme_div` - a bare ``.qspace-viewer`` element for a host page that
  loads the assets once (via :func:`viewer_assets`) and may hold several viewers.
- :func:`scheme_iframe` - a self-contained ``<iframe srcdoc>`` that inlines the
  assets, for isolated contexts such as a nireports reportlet.
"""

from __future__ import annotations

import html
import json
import math

from ..data import load

#: Default b=0 threshold for display. Matches ``grouping.metadata.B0_THRESHOLD``
#: (qsiprep's convention); callers may pass their context's own threshold.
DEFAULT_B0_THRESHOLD = 100.0

_AXIS_LABELS = ['L', 'P', 'S']
_AXIS_LABELS_NEG = ['R', 'A', 'I']

#: Minimal reset for the isolated iframe document (the host page has its own).
_FRAME_CSS = (
    'html,body{margin:0;padding:0}'
    "body{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,"
    'sans-serif;color:#0f172a;background:#fff;padding:6px 8px}'
)


def q_points(bvals, bvecs):
    """q-space points ``sqrt(b) * bvec`` (b=0 lands at the origin).

    ``bvecs`` is any ``(N, 3)`` sequence (rows are gradient directions).
    Returns a list of ``[x, y, z]`` Python floats, ready for JSON.
    """
    points = []
    for bval, bvec in zip(bvals, bvecs, strict=True):
        radius = math.sqrt(bval) if bval > 0 else 0.0
        points.append([float(radius * bvec[0]), float(radius * bvec[1]), float(radius * bvec[2])])
    return points


def scheme_payload(panels, meta, files, pes, b0_threshold=DEFAULT_B0_THRESHOLD):
    """Assemble the viewer's JSON payload.

    ``panels`` is ``[{'title': str, 'coords': [[x, y, z], ...]}, ...]`` and
    ``meta`` is one ``{'b', 'file', 'pe'}`` dict per point, shared across panels.
    """
    return {
        'panels': panels,
        'meta': meta,
        'files': files,
        'pes': pes,
        'axisLabels': _AXIS_LABELS,
        'axisLabelsNeg': _AXIS_LABELS_NEG,
        'b0Threshold': b0_threshold,
    }


def _embedded_json(data):
    # Escape ``</`` so an unlucky string cannot close the surrounding <script>.
    return json.dumps(data).replace('</', '<\\/')


def scheme_div(data):
    """A bare ``.qspace-viewer`` element. The host page supplies the JS/CSS once."""
    return (
        '<div class="qspace-viewer">'
        f'<script type="application/json">{_embedded_json(data)}</script></div>'
    )


def viewer_assets():
    """The viewer ``(css, js)`` text, for a page that embeds the viewer inline."""
    return (
        load.readable('qspace_viewer.css').read_text(),
        load.readable('qspace_viewer.js').read_text(),
    )


def document_iframe(document, title='embedded document', height=440):
    """Wrap a complete HTML document in a self-contained ``<iframe srcdoc>``.

    The frame fully isolates ``document``'s CSS and scripts from the host page,
    so a full standalone page (its own ``body``/``h1`` styles and all) can be
    dropped into a report without leaking style. ``height`` is the frame's
    starting height in pixels; a same-origin document may grow itself past it
    (see the resize script in :mod:`qsiprep.grouping.interactive`).
    """
    # The browser decodes ``srcdoc`` using the *host* page's charset, so make
    # the value pure ASCII (non-ASCII as numeric entities) to stay correct
    # regardless of how the surrounding report is served.
    srcdoc = html.escape(document, quote=True).encode('ascii', 'xmlcharrefreplace').decode('ascii')
    return (
        f'<iframe title="{html.escape(title, quote=True)}" srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height}px;border:0" loading="lazy"></iframe>'
    )


def scheme_iframe(data, height=440):
    """A self-contained ``<iframe srcdoc>`` with the viewer and its assets inlined.

    For isolated contexts (a nireports reportlet) where each widget carries its
    own copy of the code and cannot rely on the host page's scripts.
    """
    css, js = viewer_assets()
    document = (
        '<!doctype html><meta charset="utf-8">'
        f'<style>{_FRAME_CSS}{css}</style>'
        f'{scheme_div(data)}'
        f'<script>{js}</script>'
    )
    return document_iframe(document, title='DWI sampling scheme', height=height)
