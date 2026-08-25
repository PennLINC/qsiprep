"""Build the interactive pipeline-plan viewer.

Renders a compiled :class:`~qsiprep.grouping.plan.ExecutionPlan` as a flow
diagram: per output, each processing run's ordered stages (estimation feeding
eddy, HMC before DRBUDDI, refinement, assembly) drawn as connected nodes with
hover details. Shared by the group report (:mod:`qsiprep.grouping.interactive`)
and the subject report, so the planned and the executed pipeline are depicted
identically. The self-contained JS/CSS assets live in
``qsiprep/data/pipeline_viewer.{js,css}``.

Embedding mirrors the q-space viewer: :func:`pipeline_div` for a host that
loads :func:`pipeline_assets` once, :func:`pipeline_fragment` for a
self-contained inline reportlet.
"""

from __future__ import annotations

import json
import os.path as op

from ..data import load

PAYLOAD_SCHEMA_VERSION = 1

#: Stage tool value -> display label (kept in sync with the capability
#: registries' labels; duplicated here so the payload has no import cycle).
_TOOL_LABELS = {
    'topup': 'TOPUP',
    'eddy': 'eddy',
    'shoreline': 'SHORELine',
    'diffprep': 'DIFFPREP',
    'drbuddi': 'DRBUDDI',
    'fieldmap': 'GRE fieldmap',
    't2wreg': 'T2Wreg',
    'syn': 'SyN',
}


def _basename(path: str) -> str:
    return op.basename(path)


def plan_payload(grouping, plan) -> dict:
    """The viewer's JSON payload for one compiled execution plan."""
    from ..grouping.report import plan_step_records

    letters = {
        eid: chr(ord('A') + index) for index, eid in enumerate(sorted(grouping.estimations))
    }
    stage_texts = {
        (record.run, record.stage): record.text
        for records in plan_step_records(grouping, plan).values()
        for record in records
        if record.run is not None and record.stage is not None
    }

    runs = []
    for run in plan.runs:
        stages = []
        for stage in run.stages:
            stages.append(
                {
                    'index': stage.index,
                    'role': stage.role.value,
                    'tool': stage.tool,
                    'label': _TOOL_LABELS.get(stage.tool, stage.tool),
                    'method': stage.method.value if stage.method else None,
                    'estimation': stage.estimation,
                    'letter': letters.get(stage.estimation),
                    'target': stage.structural_target,
                    'consumes': stage.consumes,
                    'sources': [_basename(path) for path in stage.fieldmap_sources],
                    'borrowed': [_basename(path) for path in stage.borrowed_b0],
                    'text': stage_texts.get((run.key, stage.index), ''),
                }
            )
        runs.append(
            {
                'key': run.key,
                'unit': run.logical_unit,
                'files': [_basename(path) for path in run.dwi_files],
                'stages': stages,
            }
        )

    return {
        'schemaVersion': PAYLOAD_SCHEMA_VERSION,
        'subject': grouping.subject_id,
        'selection': {
            'label': plan.selection.label(),
            'cli': plan.selection.cli_phrase(),
            'hmc': plan.selection.hmc.value,
        },
        'outputs': [
            {
                'name': assembly.output_name,
                'group': assembly.output_group,
                'strategy': assembly.strategy,
                'runs': list(assembly.input_runs),
            }
            for assembly in plan.outputs
        ],
        'runs': runs,
        'estimations': [
            {
                'id': eid,
                'letter': letters[eid],
                'method': estimation.method.value,
                'sources': [_basename(path) for path in estimation.sources],
            }
            for eid, estimation in sorted(grouping.estimations.items())
        ],
        'issues': [
            {
                'severity': issue.severity,
                'code': issue.code,
                'message': issue.message,
                'scope': issue.scope,
                'run': issue.run,
            }
            for issue in plan.issues
        ],
    }


def _embedded_json(data):
    # Escape ``</`` so an unlucky string cannot close the surrounding <script>.
    return json.dumps(data).replace('</', '<\\/')


def pipeline_div(data):
    """A bare ``.pipeline-viewer`` element. The host page supplies the JS/CSS once."""
    return (
        '<div class="pipeline-viewer">'
        f'<script type="application/json">{_embedded_json(data)}</script></div>'
    )


def pipeline_assets():
    """The viewer ``(css, js)`` text, for a page that embeds the viewer inline."""
    return (
        load.readable('pipeline_viewer.css').read_text(),
        load.readable('pipeline_viewer.js').read_text(),
    )


def pipeline_fragment(data):
    """A self-contained inline fragment: the viewer, its assets, and one plan.

    For a nireports reportlet that inlines directly into the host report (no
    iframe). The ``.pipeline-viewer``/``.pp-*`` CSS is host-agnostic and the
    JS boots idempotently, so several fragments can coexist in one report.
    """
    css, js = pipeline_assets()
    return f'<style>{css}</style>{pipeline_div(data)}<script>{js}</script>'
