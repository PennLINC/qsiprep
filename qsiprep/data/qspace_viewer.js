/* QSIPrep q-space sampling-scheme viewer.
 *
 * Dependency-free (no Plotly/Three/D3), self-contained, offline-ready. One
 * `<div class="qspace-viewer">` per scheme, carrying its data as an embedded
 * `<script type="application/json">`. The same widget serves both reports:
 *   - group report  : one panel, the concatenated *acquired* scheme.
 *   - subject report : two linked panels, before vs after preprocessing.
 *
 * Each point is q = sqrt(b) * bvec. Color is driven by a categorical key that
 * the caller supplies per point (input-file index and phase-encoding dir);
 * the "Color by" control just switches which key maps to the palette, so
 * adding more keys later is a data change, not a code change.
 */
(function () {
  "use strict";

  // Okabe-Ito colorblind-safe categorical palette.
  var PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
                 "#CC79A7", "#56B4E9", "#F0E442", "#000000"];
  var B0_COLOR = "#9aa7b8";      // b=0 volumes sit at the origin
  var AXIS_COLOR = "#c3ccd8";

  // b-values are displayed rounded to the nearest 100 so small acquisition
  // deviations (e.g. 1585, 4985) do not read as distinct shells.
  function roundB(b) { return Math.round(b / 100) * 100; }

  function elem(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function init(root) {
    var src = root.querySelector('script[type="application/json"]');
    if (!src) return;
    var data = JSON.parse(src.textContent);

    var panels = data.panels;              // [{title, coords:[[x,y,z],...]}]
    var meta = data.meta;                  // [{b,file,pe}] indexed like coords
    var files = data.files || [];          // file index -> label
    var pes = data.pes || [];              // pe order for the legend
    var axisLabels = data.axisLabels || ["x", "y", "z"];      // +ends
    var axisLabelsNeg = data.axisLabelsNeg || null;           // -ends (optional)
    // A volume counts as b=0 (direction meaningless) at or below this value;
    // the emitter passes qsiprep's real b0 threshold. Default: only exactly 0.
    var b0Threshold = data.b0Threshold != null ? data.b0Threshold : 0;
    function isB0(m) { return m.b <= b0Threshold; }
    // Unique diffusion-weighted shells, for the concentric reference rings.
    var shells = Array.from(new Set(meta.filter(function (m) { return !isB0(m); })
                                        .map(function (m) { return m.b; })))
                      .sort(function (a, b) { return a - b; });
    // maxR and per-shell ring radius are measured per panel (see viewGeometry),
    // so panels at different radial scales (e.g. sqrt(b) vs b) each auto-fit,
    // and the rings match whatever radial convention the caller used.

    // Which categorical key colors the points, and how to read it per point.
    var MODES = {
      file: { label: "input file", cats: files,
              of: function (m) { return m.file; } },
      pe:   { label: "phase encoding", cats: pes,
              of: function (m) { return pes.indexOf(m.pe); } }
    };

    var st = {
      mode: files.length ? "file" : "pe",
      az: 0.5, el: 0.35, scale: 1,
      hidden: { file: {}, pe: {} },        // per-mode toggled-off categories
      spin: false,
      antipodal: false,
      hover: null                          // {view, cl}
    };

    // Coincident samples: volumes sharing a q-coordinate (identical b and
    // direction) — e.g. the AP and PA copy of every direction in an appa
    // scheme. They fall on one pixel, so we cluster them and draw a split
    // marker; otherwise the last one drawn silently hides the rest.
    // Membership is direction/b identity, so it holds across panels (rotation
    // and sqrt/linear scaling preserve coincidence) — compute once.
    var clusters = (function () {
      var scale = 1e-6, c0 = panels[0].coords;
      c0.forEach(function (c) {
        var r = Math.hypot(c[0], c[1], c[2]); if (r > scale) scale = r;
      });
      scale *= 1e-3;                                   // 0.1% of the max radius
      var byKey = {};
      c0.forEach(function (c, i) {
        var k = Math.round(c[0] / scale) + "|" + Math.round(c[1] / scale) +
                "|" + Math.round(c[2] / scale);
        (byKey[k] || (byKey[k] = [])).push(i);
      });
      return Object.keys(byKey).map(function (k) { return { idx: byKey[k] }; });
    })();
    var hasCoincident = clusters.some(function (cl) { return cl.idx.length > 1; });

    // One-line scheme summary.
    var summaryText = (function () {
      var nB0 = meta.filter(isB0).length;
      var nCoin = clusters.filter(function (cl) { return cl.idx.length > 1; }).length;
      var parts = [meta.length + " volumes"];
      if (nB0) parts.push(nB0 + " × b=0");
      if (shells.length && shells.length <= 6) {
        var rounded = Array.from(new Set(shells.map(roundB)));
        parts.push("b=" + rounded.join("/"));
      } else if (shells.length) {
        parts.push(shells.length + " b-values (max " +
                   roundB(shells[shells.length - 1]) + ")");
      }
      if (files.length > 1) parts.push(files.length + " series");
      if (nCoin) parts.push(nCoin + " shared coordinate" + (nCoin > 1 ? "s" : ""));
      return parts.join("  ·  ");
    })();

    // Per-panel geometry: max radius (for auto-fit) and the measured radius of
    // each shell (for the reference rings). Computed once per panel.
    function viewGeometry(panel) {
      var mr = 1e-6, sum = {}, cnt = {};
      panel.coords.forEach(function (c, i) {
        var r = Math.hypot(c[0], c[1], c[2]);
        if (r > mr) mr = r;
        var b = meta[i].b;
        if (b > b0Threshold) { sum[b] = (sum[b] || 0) + r; cnt[b] = (cnt[b] || 0) + 1; }
      });
      var sr = {};
      shells.forEach(function (b) { if (cnt[b]) sr[b] = sum[b] / cnt[b]; });
      return { maxR: mr, shellRadius: sr };
    }

    // ---- DOM scaffold -----------------------------------------------------
    root.innerHTML = "";
    root.appendChild(elem("div", "qs-summary", summaryText));
    var controls = elem("div", "qs-controls");
    root.appendChild(controls);

    // Color-by segmented control (only when there's a real choice).
    if (files.length > 1 && pes.length > 1) {
      var seg = elem("div", "qs-seg");
      seg.appendChild(elem("span", "qs-seg-label", "Color by"));
      ["file", "pe"].forEach(function (m) {
        var b = elem("button", "qs-seg-btn", MODES[m].label);
        b.dataset.mode = m;
        b.onclick = function () { st.mode = m; syncSeg(); buildLegend(); render(); };
        seg.appendChild(b);
      });
      controls.appendChild(seg);
    }

    var spinWrap = elem("label", "qs-check");
    var spinBox = elem("input");
    spinBox.type = "checkbox";
    spinBox.onchange = function () { st.spin = spinBox.checked; if (st.spin) tick(); };
    spinWrap.appendChild(spinBox);
    spinWrap.appendChild(document.createTextNode(" spin"));
    controls.appendChild(spinWrap);

    var antiWrap = elem("label", "qs-check");
    var antiBox = elem("input");
    antiBox.type = "checkbox";
    antiBox.onchange = function () { st.antipodal = antiBox.checked; render(); };
    antiWrap.appendChild(antiBox);
    antiWrap.appendChild(document.createTextNode(" antipodal"));
    controls.appendChild(antiWrap);

    var resetBtn = elem("button", "qs-btn", "reset view");
    resetBtn.onclick = function () { st.az = 0.5; st.el = 0.35; st.scale = 1; render(); };
    controls.appendChild(resetBtn);

    var legend = elem("div", "qs-legend");
    controls.appendChild(legend);

    // One canvas per panel; all share rotation/zoom/visibility.
    var stage = elem("div", "qs-stage");
    root.appendChild(stage);
    var views = panels.map(function (p) {
      var cell = elem("div", "qs-cell");
      cell.appendChild(elem("div", "qs-title", p.title || ""));
      var canvas = elem("canvas", "qs-canvas");
      cell.appendChild(canvas);
      stage.appendChild(cell);
      var g = viewGeometry(p);
      return { panel: p, canvas: canvas, ctx: canvas.getContext("2d"), proj: [],
               maxR: g.maxR, shellRadius: g.shellRadius };
    });

    var tip = elem("div", "qs-tip");
    tip.style.display = "none";
    root.appendChild(tip);

    // Self-documenting note: only shown when the scheme actually has coincidence.
    if (hasCoincident) {
      root.appendChild(elem("div", "qs-note",
        "&#9680; a split marker is one q-space coordinate sampled by more than " +
        "one series / phase-encoding direction &mdash; hover to list them"));
    }

    // ---- controls state ---------------------------------------------------
    function syncSeg() {
      controls.querySelectorAll(".qs-seg-btn").forEach(function (b) {
        b.classList.toggle("on", b.dataset.mode === st.mode);
      });
    }

    function buildLegend() {
      legend.innerHTML = "";
      var mode = MODES[st.mode];
      mode.cats.forEach(function (label, i) {
        var off = st.hidden[st.mode][i];
        var item = elem("button", "qs-leg" + (off ? " off" : ""));
        var sw = elem("span", "qs-sw");
        sw.style.background = PALETTE[i % PALETTE.length];
        item.appendChild(sw);
        item.appendChild(document.createTextNode(label));
        item.onclick = function () {
          st.hidden[st.mode][i] = !st.hidden[st.mode][i];
          buildLegend();
          render();
        };
        legend.appendChild(item);
      });
    }

    // ---- projection + drawing --------------------------------------------
    function project(c, cx, cy, s) {
      // Rotate azimuth about vertical (y), then elevation about horizontal (x).
      var ca = Math.cos(st.az), sa = Math.sin(st.az);
      var x1 = c[0] * ca + c[2] * sa;
      var z1 = -c[0] * sa + c[2] * ca;
      var y1 = c[1];
      var ce = Math.cos(st.el), se = Math.sin(st.el);
      var y2 = y1 * ce - z1 * se;
      var z2 = y1 * se + z1 * ce;
      return { x: cx + s * x1, y: cy - s * y2, depth: z2 };
    }

    function render() {
      var mode = MODES[st.mode];
      views.forEach(function (v) {
        var dpr = window.devicePixelRatio || 1;
        var w = v.canvas.clientWidth, h = v.canvas.clientHeight;
        if (v.canvas.width !== w * dpr || v.canvas.height !== h * dpr) {
          v.canvas.width = w * dpr; v.canvas.height = h * dpr;
        }
        var ctx = v.ctx;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        var cx = w / 2, cy = h / 2;
        var half = Math.min(w, h) / 2 - 22;
        if (half <= 1) return;                 // canvas too small to draw
        var s = half / v.maxR * st.scale;

        drawShells(ctx, cx, cy, s, v.shellRadius);
        drawAxes(ctx, cx, cy, s, v.maxR);

        // Antipodal mirror: faint -g copy of each diffusion-weighted sample, so
        // a half-sphere acquisition reads as the full shell it is equivalent to.
        if (st.antipodal) {
          ctx.globalAlpha = 0.16;
          ctx.fillStyle = "#94a3b8";
          clusters.forEach(function (cl) {
            if (isB0(meta[cl.idx[0]])) return;
            var shown = cl.idx.some(function (i) {
              return !st.hidden[st.mode][mode.of(meta[i])];
            });
            if (!shown) return;
            var c = v.panel.coords[cl.idx[0]];
            var pr = project([-c[0], -c[1], -c[2]], cx, cy, s);
            ctx.beginPath(); ctx.arc(pr.x, pr.y, 2.4, 0, 6.2832); ctx.fill();
          });
          ctx.globalAlpha = 1;
        }

        // Build the visible, depth-sorted list of clusters. Coincident samples
        // share one screen position, so we draw per cluster, not per point.
        var drawList = [];
        clusters.forEach(function (cl) {
          var vis = cl.idx.filter(function (i) {
            var m = meta[i];
            return isB0(m) || !st.hidden[st.mode][mode.of(m)];
          });
          if (!vis.length) return;
          var pr = project(v.panel.coords[cl.idx[0]], cx, cy, s);
          drawList.push({ cl: cl, vis: vis, x: pr.x, y: pr.y, depth: pr.depth });
        });
        drawList.sort(function (a, b) { return a.depth - b.depth; });
        v.proj = drawList;

        var dmax = v.maxR * s || 1;
        drawList.forEach(function (d) {
          var t = (d.depth * s + dmax) / (2 * dmax);      // 0 far .. 1 near
          var b0 = isB0(meta[d.vis[0]]);
          var baseR = b0 ? 3.2 : 2.6 + 2.8 * t;
          ctx.globalAlpha = b0 ? 0.92 : 0.5 + 0.5 * t;

          if (b0) {
            // Neutral labeled marker; direction is undefined at b=0.
            ctx.fillStyle = B0_COLOR;
            ctx.beginPath(); ctx.arc(d.x, d.y, baseR, 0, 6.2832); ctx.fill();
            ctx.lineWidth = 1; ctx.strokeStyle = "#64748b"; ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.fillStyle = "#475569";
            ctx.font = "9px ui-sans-serif,system-ui,sans-serif";
            ctx.fillText("b=0" + (d.vis.length > 1 ? " ×" + d.vis.length : ""),
                         d.x + baseR + 2, d.y - baseR);
          } else {
            var cats = [];
            d.vis.forEach(function (i) {
              var c = mode.of(meta[i]);
              if (cats.indexOf(c) < 0) cats.push(c);
            });
            if (cats.length <= 1) {
              ctx.fillStyle = PALETTE[cats[0] % PALETTE.length];
              ctx.beginPath(); ctx.arc(d.x, d.y, baseR, 0, 6.2832); ctx.fill();
            } else {
              // Split marker; grow with wedge count so thin slices stay legible.
              var rr = baseR + 1.4 + 0.8 * Math.min(cats.length - 2, 4);
              var a = -Math.PI / 2, step = 2 * Math.PI / cats.length;
              cats.forEach(function (c) {
                ctx.fillStyle = PALETTE[c % PALETTE.length];
                ctx.beginPath(); ctx.moveTo(d.x, d.y);
                ctx.arc(d.x, d.y, rr, a, a + step); ctx.closePath(); ctx.fill();
                a += step;
              });
            }
          }
          if (st.hover && st.hover.view === v && st.hover.cl === d.cl) {
            ctx.globalAlpha = 1;
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#0f172a";
            ctx.beginPath();
            ctx.arc(d.x, d.y, baseR + 3, 0, 6.2832);
            ctx.stroke();
          }
        });
        ctx.globalAlpha = 1;
      });
    }

    var LOG10 = Math.log(10);
    function niceStep(x) {
      var p = Math.pow(10, Math.floor(Math.log(x) / LOG10));
      var f = x / p;
      return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
    }
    function interpRadius(bs, sr, b) {
      if (b <= bs[0]) return sr[bs[0]] * b / bs[0];
      for (var i = 1; i < bs.length; i++) {
        if (b <= bs[i]) {
          var t = (b - bs[i - 1]) / (bs[i] - bs[i - 1]);
          return sr[bs[i - 1]] + t * (sr[bs[i]] - sr[bs[i - 1]]);
        }
      }
      return sr[bs[bs.length - 1]] * b / bs[bs.length - 1];
    }
    // Which b-values get a labeled ring. Few shells -> one ring each. Many
    // distinct b-values (Cartesian/DSI grid) -> a handful of round-number
    // rings, radius interpolated from the points so the convention still holds.
    function ringTicks(shellRadius) {
      var bs = Object.keys(shellRadius).map(Number)
                     .sort(function (a, b) { return a - b; });
      if (!bs.length) return [];
      if (bs.length <= 8) {
        return bs.map(function (b) { return { b: b, r: shellRadius[b] }; });
      }
      var maxb = bs[bs.length - 1], step = niceStep(maxb / 4), ticks = [];
      for (var b = step; b <= maxb + 1e-6; b += step) ticks.push(b);
      if (ticks[ticks.length - 1] < maxb * 0.98) ticks.push(maxb);
      return ticks.map(function (b) {
        return { b: b, r: interpRadius(bs, shellRadius, b) };
      });
    }

    // Concentric b-value reference rings. Orthographic projection maps each
    // shell sphere to a circle of the same radius regardless of rotation, so
    // these stay fixed while the axes turn through them.
    function drawShells(ctx, cx, cy, s, shellRadius) {
      ctx.font = "10px ui-sans-serif,system-ui,sans-serif";
      ctx.textBaseline = "middle";
      ringTicks(shellRadius).forEach(function (t) {
        var r = t.r * s;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 6.2832);
        ctx.strokeStyle = "rgba(148,163,184,0.32)";
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        // Label up-and-right of the ring, with a white halo for legibility.
        var lx = cx + r * 0.7071 + 4, ly = cy - r * 0.7071;
        var label = "b=" + roundB(t.b);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(252,253,254,0.95)";
        ctx.strokeText(label, lx, ly);
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(label, lx, ly);
      });
      ctx.textBaseline = "alphabetic";
    }

    function drawAxes(ctx, cx, cy, s, maxR) {
      var units = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      ctx.strokeStyle = AXIS_COLOR;
      ctx.lineWidth = 1;
      ctx.font = "600 11px ui-sans-serif,system-ui,sans-serif";
      units.forEach(function (e, k) {
        var pos = project([e[0] * maxR, e[1] * maxR, e[2] * maxR], cx, cy, s);
        var neg = project([-e[0] * maxR, -e[1] * maxR, -e[2] * maxR], cx, cy, s);
        ctx.beginPath();
        ctx.moveTo(neg.x, neg.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.fillText(axisLabels[k], pos.x + 3, pos.y - 3);
        if (axisLabelsNeg) ctx.fillText(axisLabelsNeg[k], neg.x + 3, neg.y - 3);
      });
    }

    // ---- interaction ------------------------------------------------------
    var drag = null;
    stage.addEventListener("pointerdown", function (e) {
      drag = { x: e.clientX, y: e.clientY };
      st.spin = false; spinBox.checked = false;
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener("pointermove", function (e) {
      if (drag) {
        st.az += (e.clientX - drag.x) * 0.01;
        st.el += (e.clientY - drag.y) * 0.01;
        st.el = Math.max(-1.5, Math.min(1.5, st.el));
        drag.x = e.clientX; drag.y = e.clientY;
        render();
      } else {
        hoverAt(e);
      }
    });
    stage.addEventListener("pointerup", function () { drag = null; });
    stage.addEventListener("pointerleave", function () {
      drag = null; st.hover = null; tip.style.display = "none"; render();
    });
    stage.addEventListener("wheel", function (e) {
      e.preventDefault();
      st.scale *= e.deltaY < 0 ? 1.1 : 0.9;
      st.scale = Math.max(0.4, Math.min(4, st.scale));
      render();
    }, { passive: false });

    function hoverAt(e) {
      var found = null;
      for (var vi = 0; vi < views.length; vi++) {
        var v = views[vi];
        var rect = v.canvas.getBoundingClientRect();
        if (e.clientX < rect.left || e.clientX > rect.right ||
            e.clientY < rect.top || e.clientY > rect.bottom) continue;
        var mx = e.clientX - rect.left, my = e.clientY - rect.top;
        for (var j = v.proj.length - 1; j >= 0; j--) {      // nearest = topmost
          var d = v.proj[j];
          if (Math.hypot(d.x - mx, d.y - my) < 8) {
            found = { view: v, d: d, sx: e.clientX, sy: e.clientY };
            break;
          }
        }
        if (found) break;
      }
      var changed = (found && (!st.hover || st.hover.cl !== found.d.cl)) ||
                    (!found && st.hover);
      st.hover = found ? { view: found.view, cl: found.d.cl } : null;
      if (found) {
        // List every (visible) sample at this coordinate.
        var lines = found.d.vis.map(function (i) {
          var m = meta[i];
          return (files[m.file] || "?") + " &middot; PE " + m.pe +
                 " &middot; b=" + (isB0(m) ? 0 : roundB(m.b));
        });
        var head = found.d.vis.length > 1
          ? "<b>" + found.d.vis.length + " samples here</b><br>" : "";
        tip.innerHTML = head + lines.join("<br>");
        var box = root.getBoundingClientRect();
        tip.style.left = (found.sx - box.left + 12) + "px";
        tip.style.top = (found.sy - box.top + 12) + "px";
        tip.style.display = "block";
      } else {
        tip.style.display = "none";
      }
      if (changed) render();
    }

    function tick() {
      if (!st.spin) return;
      st.az += 0.006;
      render();
      requestAnimationFrame(tick);
    }

    if (window.ResizeObserver) {
      new ResizeObserver(function () { render(); }).observe(stage);
    }

    syncSeg();
    buildLegend();
    render();
  }

  function boot() {
    document.querySelectorAll(".qspace-viewer").forEach(init);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
