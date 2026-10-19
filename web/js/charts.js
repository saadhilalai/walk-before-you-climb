/* charts.js — the README's charts, redrawn for the page from data/site.json (exported from the saved
   results by export_web_data.py site: the same sources and calculations as make_charts.py).
   A is quiet stone, B is the accent; every mark is also labelled, so colour is never the only cue. */

const NS = "http://www.w3.org/2000/svg";
const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
let site = null;
let ppoMode = "steps";

const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const text = (parent, x, y, str, attrs = {}) => { const t = el("text", { x, y, ...attrs }, parent); t.textContent = str; return t; };
const pct = v => (v === 0 ? "0%" : `${(v * 100).toFixed(1)}%`);
const scale = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const pLabel = p => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(2)}`);

function svgFor(box, height) {
  box.innerHTML = "";
  const w = Math.max(280, box.clientWidth);
  const svg = el("svg", { viewBox: `0 0 ${w} ${height}`, width: w, height, role: "img" }, box);
  const tip = document.createElement("div");
  tip.className = "tip";
  box.appendChild(tip);
  return { svg, w, tip };
}

function showTip(box, tip, html, x, y) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const bw = box.clientWidth, tw = tip.offsetWidth;
  tip.style.left = `${Math.min(bw - tw, Math.max(0, x - tw / 2))}px`;
  tip.style.top = `${y - tip.offsetHeight - 14}px`;
}
const hideTip = tip => tip.classList.remove("on");

function animateLines(g) {
  if (RM) return;
  g.classList.add("draw");
  g.querySelectorAll("path.line").forEach(p => p.style.setProperty("--len", `${Math.ceil(p.getTotalLength()) + 2}`));
}

// ---- Finishing per training run (the honest breakdown under the 6×) ---------------------------------------
function seedsChart(box) {
  const h = site.hardcore;
  const rows = [["All 300 runs", h.a.pooled, h.b.pooled, 300],
                ...["42", "142", "242"].map((s, i) => [`Training run ${i + 1} · seed ${s}`, h.a.per_seed[s], h.b.per_seed[s], 100])];
  const narrow = box.clientWidth < 560;
  const rowH = 74, top = 8, height = top + rows.length * rowH;
  const { svg, w, tip } = svgFor(box, height);
  svg.setAttribute("aria-label", "Courses finished in the final test, per training run: " +
    rows.map(([n, a, b]) => `${n}: A ${pct(a.finished_rate)}, B ${pct(b.finished_rate)}`).join("; "));
  const left = narrow ? 0 : 200, right = 60;
  const x = scale(0, 0.35, left, w - right);
  const g = el("g", {}, svg);
  rows.forEach(([name, a, b, n], i) => {
    const y0 = top + i * rowH + (narrow ? 22 : 12);
    if (i === 1) el("line", { x1: 0, x2: w, y1: y0 - 18, y2: y0 - 18, class: "grid-line" }, g);
    text(g, 0, narrow ? y0 - 8 : y0 + 13, name, { class: i === 0 ? "lbl-strong" : "lbl" });
    [["a", a], ["b", b]].forEach(([k, v], j) => {
      const yy = y0 + j * 18;
      el("rect", { x: left, y: yy, width: Math.max(0, x(0.35) - left), height: 8, rx: 4, fill: "rgba(243,242,238,0.05)" }, g);
      const bar = el("rect", { x: left, y: yy, width: Math.max(3, x(v.finished_rate) - left), height: 8, rx: 4, class: `bar series-${k}` }, g);
      bar.style.transitionDelay = `${(i * 2 + j) * 70}ms`;
      text(g, Math.max(3, x(v.finished_rate)) + 10, yy + 8, `${k.toUpperCase()} ${pct(v.finished_rate)}`, { class: `lbl-${k}` });
    });
    const hit = el("rect", { x: 0, y: y0 - 16, width: w, height: rowH - 8, class: "hit" }, g);
    hit.addEventListener("pointerenter", () => showTip(box, tip,
      `<b>${name}</b><br>B finished ${Math.round(b.finished_rate * n)} of ${n} · A ${Math.round(a.finished_rate * n)} of ${n}`, w / 2, y0));
    hit.addEventListener("pointerleave", () => hideTip(tip));
  });
}

// ---- Three measures, three verdicts (HTML) ------------------------------------------------------------------
function verdicts(box) {
  const h = site.hardcore, t = h.paired;
  const seedsLower = ["42", "142", "242"].filter(s => h.b.per_seed[s].mean_score < h.a.per_seed[s].mean_score).length;
  const rows = [
    ["Finished the course", h.a.pooled.finished_rate * 100, h.b.pooled.finished_rate * 100, v => `${v.toFixed(1)}%`,
     "Confirmed", true, `Paired McNemar test, ${pLabel(t.finished_mcnemar_p)}. B finished ${t.finished_only_by_other} courses that A didn’t; A finished ${t.finished_only_by_reference} that B didn’t.`],
    ["Distance reached", h.a.pooled.mean_distance, h.b.pooled.mean_distance, v => `${v.toFixed(1)}%`,
     "Proven", true, `+${t.mean_distance_difference.toFixed(1)} points of the course. Paired Wilcoxon test, ${pLabel(t.distance_wilcoxon_p)}.`],
    ["Official score", h.a.pooled.mean_score, h.b.pooled.mean_score, v => v.toFixed(1),
     "Not proven", false, `${pLabel(t.score_wilcoxon_p)}, and B scored lower in ${seedsLower} of the 3 training runs.`],
  ];
  box.innerHTML = rows.map(([name, a, b, f, verdict, yes, why]) => {
    const max = Math.max(a, b) * 1.12;
    return `<div class="verdict"><h3>${name}</h3>
      <div class="vbar">
        <div class="vbar-row a"><span>A</span><div class="vbar-track"><div class="vbar-fill" style="--w:${(100 * Math.max(0, a) / max).toFixed(1)}%"></div></div><span class="v">${f(a)}</span></div>
        <div class="vbar-row b"><span>B</span><div class="vbar-track"><div class="vbar-fill" style="--w:${(100 * Math.max(0, b) / max).toFixed(1)}%"></div></div><span class="v">${f(b)}</span></div>
      </div>
      <div><p class="verdict-tag ${yes ? "yes" : "no"}">${verdict}</p><p class="why">${why}</p></div></div>`;
  }).join("");
}

// ---- Line charts with a band (learning curves) -----------------------------------------------------------------
function lineChart(box, { series, xMax, yMin, yMax, xTicks, xFmt, yTicks, refs = [], vline = null, tipFmt, label }) {
  const narrow = box.clientWidth < 560;
  const height = Math.round(Math.min(460, Math.max(280, box.clientWidth * 0.48)));
  const { svg, w, tip } = svgFor(box, height);
  svg.setAttribute("aria-label", label);
  const m = { l: 44, r: narrow ? 64 : 84, t: 14, b: 34 };
  const x = scale(0, xMax, m.l, w - m.r), y = scale(yMin, yMax, height - m.b, m.t);
  const g = el("g", {}, svg);
  for (const v of yTicks) {
    el("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v), class: v === 0 ? "axis-line" : "grid-line" }, g);
    text(g, m.l - 10, y(v) + 4, `${v}`, { "text-anchor": "end" });
  }
  for (const v of xTicks) text(g, x(v), height - 10, xFmt(v), { "text-anchor": "middle" });
  for (const r of refs) {
    el("line", { x1: m.l, x2: w - m.r, y1: y(r.y), y2: y(r.y), class: "ref-line" }, g);
    text(g, m.l + 6, y(r.y) - 7, r.label, { class: "lbl" });
  }
  if (vline) {
    el("line", { x1: x(vline.x), x2: x(vline.x), y1: m.t, y2: height - m.b, class: "ref-line" }, g);
    text(g, x(vline.x) + 8, height - m.b - 10, vline.label, { class: "lbl" });
  }
  const lines = el("g", {}, g);
  for (const s of series) {
    const pts = s.data.x.map((xv, i) => [x(xv), y(s.data.mean[i])]);
    const band = s.data.x.map((xv, i) => `${x(xv)},${y(s.data.hi[i])}`).concat(
      s.data.x.map((xv, i) => `${x(xv)},${y(s.data.lo[i])}`).reverse());
    el("path", { d: `M${band.join("L")}Z`, class: `band fade series-${s.cls}` }, lines);
    el("path", { d: `M${pts.map(p => p.join(",")).join("L")}`, class: `line series-${s.cls}` }, lines);
    const [ex, ey] = pts[pts.length - 1];
    el("circle", { cx: ex, cy: ey, r: 4, class: `fade series-${s.cls}` }, lines);
    text(lines, ex + 10, ey + 4 + (s.dy || 0), `${s.name} ${Math.round(s.data.mean[s.data.mean.length - 1])}`, { class: `fade lbl-${s.cls}` });
  }
  animateLines(lines);
  // Hover: a crosshair and every series' value at the nearest check-up.
  const cross = el("line", { y1: m.t, y2: height - m.b, class: "cross" }, g);
  const dots = series.map(s => el("circle", { r: 4.5, class: `hover-dot series-${s.cls}` }, g));
  const hit = el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: height - m.t - m.b, class: "hit" }, g);
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (w / r.width);
    const xv = (px - m.l) / (w - m.l - m.r) * xMax;
    const idx = series.map(s => { let b = 0; s.data.x.forEach((v, i) => { if (Math.abs(v - xv) < Math.abs(s.data.x[b] - xv)) b = i; }); return b; });
    const cx = x(series[0].data.x[idx[0]]);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.classList.add("on");
    series.forEach((s, k) => { dots[k].setAttribute("cx", x(s.data.x[idx[k]])); dots[k].setAttribute("cy", y(s.data.mean[idx[k]])); dots[k].classList.add("on"); });
    showTip(box, tip, tipFmt(series, idx), cx * (r.width / w), Math.min(...series.map((s, k) => y(s.data.mean[idx[k]]))) * (r.height / height));
  });
  hit.addEventListener("pointerleave", () => { cross.classList.remove("on"); dots.forEach(d => d.classList.remove("on")); hideTip(tip); });
}

function curvesChart(box) {
  const c = site.curves.hardcore;
  const toM = d => ({ ...d, x: d.x.map(v => v / 1e6) });
  lineChart(box, {
    series: [{ name: "A", cls: "a", data: toM(c.a), dy: 6 }, { name: "B", cls: "b", data: toM(c.b), dy: -4 }],
    xMax: 104, yMin: -140, yMax: 160, xTicks: [0, 20, 40, 60, 80, 100], xFmt: v => (v ? `${v}M` : "0"),
    yTicks: [-100, 0, 100], vline: { x: site.b_switch_steps / 1e6, label: "B starts obstacles (5M)" },
    label: "Learning curves on the obstacle course: average score of A and B over 100 million practice steps",
    tipFmt: (s, idx) => `<b>${(s[0].data.x[idx[0]]).toFixed(1)}M practice steps</b><br>` +
      s.map((q, k) => `${q.name}: ${q.data.mean[idx[k]].toFixed(0)} <span>(runs ${q.data.lo[idx[k]].toFixed(0)} to ${q.data.hi[idx[k]].toFixed(0)})</span>`).join("<br>"),
  });
}

function ppoChart(box) {
  const c = site.curves.normal, steps = ppoMode === "steps";
  const pick = run => {
    const d = c[run][ppoMode];
    if (!steps) return d;
    const n = c[run].steps.x.filter(v => v <= site.step2_steps).length;
    return { x: d.x.slice(0, n).map(v => v / 1e3), mean: d.mean.slice(0, n), lo: d.lo.slice(0, n), hi: d.hi.slice(0, n) };
  };
  lineChart(box, {
    series: [{ name: "SAC", cls: "1", data: pick("sac") }, { name: "PPO", cls: "2", data: pick("ppo"), dy: 4 }],
    xMax: steps ? 560 : 66, yMin: -150, yMax: 340,
    xTicks: steps ? [0, 100, 200, 300, 400, 500] : [0, 10, 20, 30, 40, 50, 60],
    xFmt: v => (steps ? (v ? `${v}k` : "0") : `${v} min`),
    yTicks: [-100, 0, 100, 200, 300], refs: [{ y: 300, label: "solved (300)" }],
    label: steps ? "PPO vs SAC with the same practice: SAC learns to walk within 524,288 steps, PPO doesn't"
                 : "PPO vs SAC with the same training time: PPO, continued to 5 million steps, learns faster per minute",
    tipFmt: (s, idx) => `<b>${steps ? `${Math.round(s[0].data.x[idx[0]])}k steps` : `${s[0].data.x[idx[0]].toFixed(1)} min`}</b><br>` +
      s.map((q, k) => `${q.name}: ${q.data.mean[idx[k]].toFixed(0)}`).join("<br>"),
  });
  const n = site.normal;
  document.querySelector("[data-ppo-note]").textContent = steps
    ? `Final test, 300 runs each: SAC finished ${pct(n.sac_524k.pooled.finished_rate)} after 524,288 steps; PPO finished ${pct(n.ppo_524k.pooled.finished_rate)}. It learned to stand still, which scores better than falling.`
    : `Given more steps, PPO catches up fast: continued to 5M steps (a third of SAC’s training time), it finished ${pct(n.ppo_5m.pooled.finished_rate)} of the final test courses.`;
}

// ---- Where they fall (grouped bars) -----------------------------------------------------------------------------
function fallsChart(box) {
  const kinds = [["stump", "Stumps"], ["pit", "Pits"], ["stairs_up", "Stairs up"], ["stairs_down", "Stairs down"]];
  const wa = site.where.a_100m, wb = site.where.b_100m;
  const narrow = box.clientWidth < 560;
  const rowH = 70, height = kinds.length * rowH + 6;
  const { svg, w, tip } = svgFor(box, height);
  svg.setAttribute("aria-label", "Falls per 100 obstacles reached: " +
    kinds.map(([k, n]) => `${n}: A ${wa.per_100_reached[k]}, B ${wb.per_100_reached[k]}`).join("; "));
  const left = narrow ? 0 : 140, right = 70;
  const x = scale(0, 22, left, w - right);
  const g = el("g", {}, svg);
  kinds.forEach(([k, name], i) => {
    const y0 = i * rowH + (narrow ? 22 : 10);
    text(g, 0, narrow ? y0 - 8 : y0 + 14, name, { class: "lbl" });
    [["a", wa], ["b", wb]].forEach(([s, d], j) => {
      const v = d.per_100_reached[k] || 0, yy = y0 + j * 18;
      const bar = el("rect", { x: left, y: yy, width: Math.max(3, x(v) - left), height: 8, rx: 4, class: `bar series-${s}` }, g);
      bar.style.transitionDelay = `${(i * 2 + j) * 70}ms`;
      text(g, Math.max(3, x(v)) + 10, yy + 8, `${s.toUpperCase()} ${v.toFixed(1)}`, { class: `lbl-${s}` });
    });
    const hit = el("rect", { x: 0, y: y0 - 16, width: w, height: rowH - 6, class: "hit" }, g);
    hit.addEventListener("pointerenter", () => showTip(box, tip,
      `<b>${name}</b><br>A fell at ${wa.falls[k] || 0} of the ${wa.reached[k]} it reached<br>B fell at ${wb.falls[k] || 0} of the ${wb.reached[k]} it reached`, w / 2, y0));
    hit.addEventListener("pointerleave", () => hideTip(tip));
  });
  const sa = wa.timed_out_stuck_at, sb = wb.timed_out_stuck_at;
  const tot = o => Object.values(o).reduce((a, b) => a + b, 0);
  document.querySelector("[data-falls-note]").textContent =
    `Runs that ran out of time mostly ended frozen at a stump: A in ${sa.stump} of ${tot(sa)}, B in ${sb.stump} of ${tot(sb)}. Standing still costs a little; falling costs 100 points.`;
}

// ---- Entry points -----------------------------------------------------------------------------------------------------
function drawAll() {
  if (!site) return;
  const box = name => document.querySelector(`[data-chart="${name}"]`);
  seedsChart(box("seeds"));
  verdicts(box("verdicts"));
  curvesChart(box("curves"));
  fallsChart(box("falls"));
  ppoChart(box("ppo"));
}

export function drawCharts(data) {
  site = data;
  document.querySelectorAll("[data-ppo-mode]").forEach(btn => btn.addEventListener("click", () => {
    ppoMode = btn.dataset.ppoMode;
    document.querySelectorAll("[data-ppo-mode]").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
    const box = document.querySelector('[data-chart="ppo"]');
    ppoChart(box);
  }));
  drawAll();
}

let lastW = 0;
export function redrawCharts() {
  if (!site || Math.abs(innerWidth - lastW) < 2) return;
  lastW = innerWidth;
  drawAll();
}
