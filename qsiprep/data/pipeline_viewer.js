/* Interactive pipeline-plan viewer.
 *
 * Renders the execution-plan payload embedded by qsiprep/viz/pipeline.py as
 * an SVG flow diagram: one lane per output file, each processing run's stages
 * drawn in plan order (so eddy+TOPUP shows the field estimated first and
 * consumed during motion correction), fieldmap estimations feeding their
 * stages, and the assembly step joining multi-run outputs. Hovering a node
 * shows the full step sentence and file lists.
 *
 * No dependencies. Boots idempotently on every .pipeline-viewer element, so
 * any number of viewers can share one page.
 */
(function () {
  'use strict';

  var NODE_W = 128;
  var NODE_H = 46;
  var GAP_X = 44;
  var ROW_H = 96;
  var PAD = 14;
  var CHIP_H = 24;

  var STYLE = {
    files: { fill: '#f1f5f9', stroke: '#94a3b8', text: '#334155' },
    hmc: { fill: '#dbeafe', stroke: '#2563eb', text: '#1e3a8a' },
    'hmc-with-field': { fill: '#dbeafe', stroke: '#2563eb', text: '#1e3a8a' },
    estimate: { fill: '#fef3c7', stroke: '#d97706', text: '#92400e' },
    'estimate+apply': { fill: '#dcfce7', stroke: '#16a34a', text: '#14532d' },
    refine: { fill: '#ede9fe', stroke: '#7c3aed', text: '#4c1d95' },
    assemble: { fill: '#e2e8f0', stroke: '#475569', text: '#1e293b' },
    output: { fill: '#0f172a', stroke: '#0f172a', text: '#ffffff' },
  };

  var ROLE_PHRASES = {
    estimate: 'estimates the field',
    hmc: 'motion correction',
    'hmc-with-field': 'motion + field applied',
    'estimate+apply': 'estimates & corrects',
    refine: 'refines the correction',
  };

  function el(name, attrs, text) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function truncate(text, max) {
    return text.length > max ? text.slice(0, max - 1) + '…' : text;
  }

  function plural(n, word) {
    // 'series' is invariant; everything else takes a plain s.
    if (word === 'series' || n === 1) return n + ' ' + word;
    return n + ' ' + word + 's';
  }

  /* ------------------------------------------------------------------ */

  function render(root, data) {
    // Re-entrant: an interactive host calls this again with a new payload
    // whenever the user changes the method controls.
    root.querySelectorAll('.pp-output, .pp-tip').forEach(function (el) {
      el.remove();
    });

    var runsByKey = {};
    data.runs.forEach(function (run) { runsByKey[run.key] = run; });

    var tip = document.createElement('div');
    tip.className = 'pp-tip';
    root.appendChild(tip);

    data.outputs.forEach(function (output) {
      root.appendChild(renderOutput(data, output, runsByKey, tip, root));
    });
  }

  function init(root) {
    if (root.dataset.ppReady) return;
    root.dataset.ppReady = '1';
    var script = root.querySelector('script[type="application/json"]');
    if (!script) return;
    var data;
    try {
      data = JSON.parse(script.textContent);
    } catch (err) {
      return;
    }
    render(root, data);
  }

  function issuesFor(data, output) {
    return data.issues.filter(function (issue) {
      return (
        issue.scope === output.group ||
        issue.scope === null ||
        output.runs.indexOf(issue.run) >= 0
      );
    });
  }

  function renderOutput(data, output, runsByKey, tip, root) {
    var box = document.createElement('div');
    box.className = 'pp-output';
    var head = document.createElement('div');
    head.className = 'pp-head';
    head.textContent = output.name;
    box.appendChild(head);

    var runs = output.runs.map(function (key) { return runsByKey[key]; });
    var multi = runs.length > 1;
    // Columns: files, then the longest stage chain, then assemble (if
    // multi-run), then the output node.
    var maxStages = 0;
    runs.forEach(function (run) {
      if (run.stages.length > maxStages) maxStages = run.stages.length;
    });
    var tailCols = multi ? 2 : 1;
    var nCols = 1 + maxStages + tailCols;
    var width = PAD * 2 + nCols * NODE_W + (nCols - 1) * GAP_X;
    var height = PAD * 2 + CHIP_H + runs.length * ROW_H;

    var svg = el('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      width: width,
      height: height,
      class: 'pp-svg',
      role: 'img',
    });

    function colX(col) {
      return PAD + col * (NODE_W + GAP_X);
    }
    function rowY(row) {
      return PAD + CHIP_H + row * ROW_H + (ROW_H - NODE_H) / 2;
    }

    runs.forEach(function (run, row) {
      var y = rowY(row);
      var filesNode = drawNode(svg, colX(0), y, 'files', truncate(run.key, 20),
        plural(run.files.length, 'series') + ' · denoise');
      hover(filesNode, tip, root,
        '<b>' + esc(run.key) + '</b><br>' +
        'Each series is denoised, then the unit is concatenated.<br>' +
        esc(run.files.join(', ')));

      var prev = { x: colX(0), y: y };
      run.stages.forEach(function (stage, idx) {
        var x = colX(1 + idx);
        var style = stage.role === 'estimate' ? 'estimate' : stage.role;
        var subtitle = ROLE_PHRASES[stage.role] || stage.role;
        if (stage.channels) {
          // DRBUDDI: name the registration channels (b=0 / +FA / +T2w).
          subtitle = stage.channels + ' registration';
        } else if (stage.target) {
          subtitle += ' · ' + stage.target;
        }
        var node = drawNode(svg, x, y, style, stage.label, subtitle);
        var edgeLabel = null;
        var next = run.stages[idx + 1];
        if (next && next.consumes === stage.index) edgeLabel = 'field';
        drawEdge(svg, prev.x + NODE_W, prev.y + NODE_H / 2, x, y + NODE_H / 2, null);
        if (edgeLabel === null && stage.consumes !== null && stage.consumes !== undefined) {
          // The edge INTO this node carries the estimated field.
          drawEdgeLabel(svg, x - GAP_X / 2, y + NODE_H / 2 - 6, 'field');
        }
        if (stage.estimation) drawFieldmapChip(svg, x, y, stage, tip, root);
        var detail = '<b>' + esc(stage.label) + '</b><br>' + esc(stage.text || subtitle);
        if (stage.sources.length) {
          detail += '<br>fieldmap sources: ' + esc(stage.sources.join(', '));
        }
        if (stage.borrowed.length) {
          detail += '<br>borrowed b=0: ' + esc(stage.borrowed.join(', '));
        }
        hover(node, tip, root, detail);
        prev = { x: x, y: y };
      });
      run._endX = prev.x + NODE_W;
      run._endY = prev.y + NODE_H / 2;
    });

    var outCol = nCols - 1;
    var midY = PAD + CHIP_H + (runs.length * ROW_H - NODE_H) / 2;
    if (multi) {
      var asmX = colX(nCols - 2);
      var verb = output.strategy === 'average' ? 'average' : 'concatenate';
      var asmNode = drawNode(svg, asmX, midY, 'assemble', verb,
        plural(runs.length, 'run') + ' combined');
      hover(asmNode, tip, root,
        'The corrected results of the ' + runs.length +
        ' runs are resampled onto one grid and ' +
        (output.strategy === 'average' ? 'averaged' : 'concatenated') + '.');
      runs.forEach(function (run) {
        drawElbow(svg, run._endX, run._endY, asmX, midY + NODE_H / 2);
      });
      drawEdge(svg, asmX + NODE_W, midY + NODE_H / 2, colX(outCol), midY + NODE_H / 2);
    } else {
      drawEdge(svg, runs[0]._endX, runs[0]._endY, colX(outCol), midY + NODE_H / 2);
    }
    var outNode = drawNode(svg, colX(outCol), midY, 'output', 'output file',
      truncate(output.name, 20));
    hover(outNode, tip, root, '<b>' + esc(output.name) + '_dwi.nii.gz</b>');

    var outputIssues = issuesFor(data, output);
    if (outputIssues.length) {
      var hasError = outputIssues.some(function (issue) { return issue.severity === 'error'; });
      var badge = el('circle', {
        cx: colX(outCol) + NODE_W - 4,
        cy: midY + 4,
        r: 8,
        fill: hasError ? '#dc2626' : '#f59e0b',
        stroke: '#fff',
        'stroke-width': 1.5,
        class: 'pp-node',
      });
      svg.appendChild(badge);
      hover(badge, tip, root, outputIssues.map(function (issue) {
        return '<b>' + esc(issue.severity.toUpperCase()) + '</b> [' +
          esc(issue.code) + '] ' + esc(issue.message);
      }).join('<br>'));
    }

    var scroller = document.createElement('div');
    scroller.className = 'pp-scroll';
    scroller.appendChild(svg);
    box.appendChild(scroller);
    return box;
  }

  /* -------------------------- drawing helpers ----------------------- */

  function drawNode(svg, x, y, styleName, title, subtitle) {
    var style = STYLE[styleName] || STYLE.files;
    var group = el('g', { class: 'pp-node' });
    group.appendChild(el('rect', {
      x: x, y: y, width: NODE_W, height: NODE_H, rx: 9,
      fill: style.fill, stroke: style.stroke, 'stroke-width': 1.6,
    }));
    group.appendChild(el('text', {
      x: x + NODE_W / 2, y: y + 19, 'text-anchor': 'middle',
      'font-size': 12.5, 'font-weight': 650, fill: style.text,
    }, truncate(title, 17)));
    if (subtitle) {
      group.appendChild(el('text', {
        x: x + NODE_W / 2, y: y + 35, 'text-anchor': 'middle',
        'font-size': 9.5, fill: style.text, opacity: 0.85,
      }, truncate(subtitle, 24)));
    }
    svg.appendChild(group);
    return group;
  }

  function drawEdge(svg, x1, y1, x2, y2) {
    svg.appendChild(el('line', {
      x1: x1 + 2, y1: y1, x2: x2 - 6, y2: y2,
      stroke: '#94a3b8', 'stroke-width': 1.6,
    }));
    svg.appendChild(el('path', {
      d: 'M' + (x2 - 7) + ',' + (y2 - 4) + ' L' + (x2 - 1) + ',' + y2 +
        ' L' + (x2 - 7) + ',' + (y2 + 4) + ' Z',
      fill: '#94a3b8',
    }));
  }

  function drawElbow(svg, x1, y1, x2, y2) {
    var mid = x1 + GAP_X / 2;
    svg.appendChild(el('path', {
      d: 'M' + (x1 + 2) + ',' + y1 + ' H' + mid + ' V' + y2 + ' H' + (x2 - 6),
      fill: 'none', stroke: '#94a3b8', 'stroke-width': 1.6,
    }));
    svg.appendChild(el('path', {
      d: 'M' + (x2 - 7) + ',' + (y2 - 4) + ' L' + (x2 - 1) + ',' + y2 +
        ' L' + (x2 - 7) + ',' + (y2 + 4) + ' Z',
      fill: '#94a3b8',
    }));
  }

  function drawEdgeLabel(svg, x, y, text) {
    svg.appendChild(el('text', {
      x: x, y: y, 'text-anchor': 'middle', 'font-size': 9,
      fill: '#b45309', 'font-weight': 650,
    }, text));
  }

  function drawFieldmapChip(svg, x, y, stage, tip, root) {
    var chipY = y - CHIP_H + 3;
    var group = el('g', { class: 'pp-node' });
    group.appendChild(el('line', {
      x1: x + NODE_W / 2, y1: chipY + 16, x2: x + NODE_W / 2, y2: y,
      stroke: '#d97706', 'stroke-width': 1.2, 'stroke-dasharray': '3,2',
    }));
    group.appendChild(el('rect', {
      x: x + 12, y: chipY, width: NODE_W - 24, height: 17, rx: 8,
      fill: '#fffbeb', stroke: '#d97706', 'stroke-width': 1.2,
    }));
    var label = (stage.letter ? stage.letter + ' · ' : '') + stage.estimation;
    group.appendChild(el('text', {
      x: x + NODE_W / 2, y: chipY + 12, 'text-anchor': 'middle',
      'font-size': 9, fill: '#92400e', 'font-weight': 650,
    }, truncate(label, 20)));
    svg.appendChild(group);
    var detail = '<b>fieldmap ' + esc(stage.estimation) + '</b>';
    if (stage.sources.length) detail += '<br>' + esc(stage.sources.join(', '));
    hover(group, tip, root, detail);
  }

  /* --------------------------- interaction -------------------------- */

  function esc(text) {
    var div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
  }

  function hover(node, tip, root, html) {
    node.addEventListener('mouseenter', function () {
      tip.innerHTML = html;
      tip.style.display = 'block';
    });
    node.addEventListener('mousemove', function (event) {
      var bounds = root.getBoundingClientRect();
      var x = event.clientX - bounds.left + 14;
      var y = event.clientY - bounds.top + 14;
      var maxX = root.clientWidth - tip.offsetWidth - 8;
      tip.style.left = Math.max(0, Math.min(x, maxX)) + 'px';
      tip.style.top = y + 'px';
    });
    node.addEventListener('mouseleave', function () {
      tip.style.display = 'none';
    });
  }

  function boot() {
    document.querySelectorAll('.pipeline-viewer').forEach(init);
  }

  // The host-page API: interactive hosts re-render a container from a new
  // payload; the embedded-script boot path stays for static pages.
  window.QSIPrepPipeline = { render: render };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
