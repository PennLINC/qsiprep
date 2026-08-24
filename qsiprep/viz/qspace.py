"""Build the interactive q-space sampling-scheme viewer.

Shared by the group report (:mod:`qsiprep.grouping.interactive`) and the subject
report (:class:`qsiprep.interfaces.reports.GradientPlot`) so gradient schemes
are depicted identically in both. The self-contained JS/CSS assets live in
``qsiprep/data/qspace_viewer.{js,css}``.

Two embedding styles are offered:

- :func:`scheme_div` - a bare ``.qspace-viewer`` element for a host page that
  loads the assets once (via :func:`viewer_assets`) and may hold several viewers.
- :func:`scheme_fragment` - a self-contained inline fragment (the viewer plus
  its assets) that nireports drops straight into the report, so the widget flows
  in the page instead of scrolling inside a fixed-height iframe.
"""

from __future__ import annotations

import json
import math

from ..data import load

#: Default b=0 threshold for display. Matches ``grouping.metadata.B0_THRESHOLD``
#: (qsiprep's convention); callers may pass their context's own threshold.
DEFAULT_B0_THRESHOLD = 100.0

_AXIS_LABELS = ['L', 'P', 'S']
_AXIS_LABELS_NEG = ['R', 'A', 'I']


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


def scheme_fragment(data):
    """A self-contained inline fragment: the viewer, its assets, and one scheme.

    For a nireports reportlet that inlines directly into the host report (no
    iframe), so the widget flows in the page instead of scrolling inside a
    fixed-height frame. The viewer's ``.qspace-viewer``/``.qs-*`` CSS is
    host-agnostic (no bare-element rules that could leak), and its JS boots
    idempotently, so several fragments can coexist in one report.
    """
    css, js = viewer_assets()
    return f'<style>{css}</style>{scheme_div(data)}<script>{js}</script>'
