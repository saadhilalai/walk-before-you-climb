/* app.js — the page: reveals, the film, the scroll story, the freeze-vs-climb moment, the course
   explorer and the charts. Every robot is a replay of a recorded test run (see replay.js). */

import { loadRuns, Stage, Player, duration, FPS } from "./replay.js";
import { drawCharts, redrawCharts } from "./charts.js";

const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
const LIGHT = matchMedia("(hover: none), (pointer: coarse)").matches;   // phones and tablets: lighter effects
const DPR = LIGHT ? 1.75 : 2;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);   // smooth in-out
const easeOut = t => 1 - Math.pow(1 - t, 3);
const RUN_NAMES = { 42: "training run 1", 142: "training run 2", 242: "training run 3" };
const SEEDS = [42, 142, 242];
const FINISH_X = (200 - 10) * 14 / 30, START_X = 14 / 30 * 20 / 2;
const progress = x => clamp(100 * (x - START_X) / (FINISH_X - START_X), 0, 100);
const fmt = (v, f) => f === "int" ? Math.round(v).toLocaleString("en-US") : f === "pct1" ? `${v.toFixed(1)}%` : `${Math.round(v)}`;

// ---- Film grain ---------------------------------------------------------------------------------------
(function grain() {
  const c = document.createElement("canvas");
  c.width = c.height = 180;
  const g = c.getContext("2d"), img = g.createImageData(180, 180);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = Math.random() < 0.5 ? 255 : 0;
  }
  g.putImageData(img, 0, 0);
  const el = document.createElement("div");
  el.className = "grain";
  el.setAttribute("aria-hidden", "true");
  el.style.setProperty("--grain", `url(${c.toDataURL()})`);
  document.body.appendChild(el);
})();

// ---- Count-ups ------------------------------------------------------------------------------------------
function countUp(el) {
  if (el.dataset.done) return;
  el.dataset.done = "1";
  const to = parseFloat(el.dataset.count), from = parseFloat(el.dataset.from || 0), f = el.dataset.format;
  if (RM) { el.textContent = fmt(to, f); return; }
  const dur = 1700, t0 = performance.now();
  const step = now => {
    const t = clamp((now - t0) / dur);
    el.textContent = fmt(lerp(from, to, easeOut(t)), f);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---- Reveal on scroll ------------------------------------------------------------------------------------
const revealer = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    $$("[data-count]", e.target).forEach(el => { if (!el.closest(".story-cap")) countUp(el); });
    e.target.querySelectorAll(".chart").forEach(c => c.classList.add("in"));
    revealer.unobserve(e.target);
  }
}, { rootMargin: "0px 0px -9% 0px", threshold: 0.12 });
$$("[data-reveal]").forEach(el => revealer.observe(el));

// ---- Navigation -------------------------------------------------------------------------------------------
const nav = $("[data-nav]");

// ---- 1. The film --------------------------------------------------------------------------------------------
const film = $("[data-film]"), video = $("[data-video]"), filmFrame = $("[data-film-frame]");
$("[data-video-play]").addEventListener("click", () => {
  film.classList.add("playing");
  video.controls = true;
  if (video.currentTime < 0.034) video.currentTime = 0.034;   // frame 0 is the poster, baked in for sharing
  video.play().catch(() => {});
});
video.addEventListener("ended", () => { video.currentTime = 0.034; });

// ---- 2. The learning story ------------------------------------------------------------------------------------
const story = (() => {
  const root = $("[data-story]"), sticky = $(".story-sticky", root), win = $("[data-story-window]", root);
  const names = ["falls", "wobbles", "walks"];
  const windows = { falls: [0, null], wobbles: [0, 10.5], walks: [0, null] };   // seconds of each run shown
  const stills = { falls: 2.3, wobbles: 6.0, walks: 8.0 };                       // reduced motion: one frame each
  const stages = {}, players = {}, data = {};
  let active = null, paused = false, ready = false;

  for (const n of names) stages[n] = new Stage($(`[data-story-canvas="${n}"]`, root), { maxDpr: DPR });

  const load = Promise.all(names.map(n => loadRuns(`data/stories/${n}.bin`).then(d => {
    data[n] = d;
    const run = d.runs[0], [a, b] = windows[n];
    players[n] = new Player({ from: a, to: b ?? duration(run), hold: n === "falls" ? 1.8 : 1.2,
                              render: t => stages[n].draw(d, run, t) });
    stages[n].draw(d, run, RM ? stills[n] : a);
  }))).then(() => { ready = true; if (active) activate(active, true); });

  const toggle = $("[data-story-toggle]", root);
  const setToggle = playing => {
    toggle.innerHTML = `<span class="${playing ? "icon-pause" : "icon-play"}" aria-hidden="true"></span>`;
    toggle.setAttribute("aria-label", playing ? "Pause the replay" : "Play the replay");
  };
  setToggle(!RM);
  toggle.addEventListener("click", () => {
    const p = players[active];
    if (!p) return;
    if (p.playing) { p.pause(); paused = true; setToggle(false); } else { p.play(); paused = false; setToggle(true); }
  });

  function activate(name, force = false) {
    if (name === active && !force) return;
    active = name;
    for (const n of names) {
      $(`[data-story-canvas="${n}"]`, root).classList.toggle("on", n === name);
      $(`[data-story-cap="${n}"]`, root).classList.remove("on");
      if (n !== name) players[n]?.pause();
    }
    if (!name) return;
    if (!ready) return;
    const p = players[name];
    if (RM || paused) { p.seek(RM ? stills[name] : p.from); setToggle(false); }
    else { p.restart(); setToggle(true); }
  }

  function caption(name) {
    for (const n of names) {
      const cap = $(`[data-story-cap="${n}"]`, root), on = n === name;
      if (on && !cap.classList.contains("on")) $$("[data-count]", cap).forEach(el => { el.dataset.done = ""; countUp(el); });
      cap.classList.toggle("on", on);
    }
  }

  function layout() {
    const portrait = innerHeight > innerWidth || innerWidth <= 760;
    root.dataset.s0 = portrait ? 0.86 : 0.56;
    names.forEach(n => stages[n].resize());
    if (ready && active) players[active].render(players[active].t);
    else if (ready) names.forEach(n => stages[n].draw(data[n], data[n].runs[0], RM ? stills[n] : 0));
  }

  function onScroll(vh) {
    const r = root.getBoundingClientRect();
    const p = clamp(-r.top / (r.height - vh));
    const s0 = +root.dataset.s0, winH = win.offsetHeight;
    const set = (k, v) => sticky.style.setProperty(k, v);
    // Title: rises in while the section scrolls into view, holds, then leaves (up + blur) as the window arrives.
    const tIn = RM ? 1 : ease(clamp((vh - r.top) / (vh * 0.85))), tOut = ease(clamp((p - 0.06) / 0.045));
    set("--title-o", (tIn * (1 - tOut)).toFixed(3));
    set("--title-y", `${((1 - tIn) * 24 - tOut * 30).toFixed(1)}px`);
    set("--title-b", `${(RM ? 0 : (1 - tIn) * 8 + tOut * 10).toFixed(1)}px`);
    // Window: appears small, then grows to fill the screen on "It walks."
    const wIn = ease(clamp((p - 0.085) / 0.055));
    const grow = ease(clamp((p - 0.69) / 0.13));
    const scale = RM ? (p > 0.75 ? 1 : s0) : lerp(lerp(s0 * 0.95, s0, wIn), 1, grow);
    const lift = -0.05 * vh * (1 - grow);
    set("--win-s", scale.toFixed(4));
    set("--win-r", `${(30 * (1 - grow)).toFixed(2)}px`);
    set("--win-o", (RM ? (p > 0.085 ? 1 : 0) : wIn).toFixed(3));
    set("--win-y", `${lift.toFixed(1)}px`);
    set("--cap-top", `${(winH * s0 / 2 - 0.05 * vh + 26).toFixed(0)}px`);
    set("--note-o", (wIn * (1 - grow) * 0.9).toFixed(3));
    const stage = p < 0.085 ? null : p < 0.34 ? "falls" : p < 0.56 ? "wobbles" : "walks";
    activate(stage);
    caption(stage && p > 0.125 && p < 0.675 ? stage : null);
    // Only draw while the section is on screen.
    if (r.bottom < 0 || r.top > vh) players[active]?.pause();
    else if (active && players[active] && !players[active].playing && !paused && !RM) players[active].play();
  }

  return { onScroll, layout, load };
})();

// ---- 4. Freeze vs climb ----------------------------------------------------------------------------------------
const moment = (() => {
  const root = $("[data-moment]"), pair = $("[data-moment-pair]", root);
  const stages = { a: new Stage($('[data-moment-canvas="a"]', root), { maxDpr: DPR }),
                   b: new Stage($('[data-moment-canvas="b"]', root), { maxDpr: DPR }) };
  const status = { a: $('[data-moment-status="a"]', root), b: $('[data-moment-status="b"]', root) };
  // Course 10016, training run 1 (seed 42), as in the film: A stops making progress at 4.8 s,
  // B clears the same stump at 6.6 s (make_videos.stump_seconds). The scene ends at 7.0 s.
  const T_A = 5.3, T_B = 6.6, END = 7.0;
  let player = null, course = null, runs = null, visible = false;

  const load = loadRuns("data/courses/10016.bin").then(d => {
    course = d;
    runs = { a: d.runs.find(r => r.model === "a_100m" && r.seed === 42), b: d.runs.find(r => r.model === "b_100m" && r.seed === 42) };
    player = new Player({ from: 0, to: END, hold: 3.2, render: t => {
      stages.a.draw(course, runs.a, t); stages.b.draw(course, runs.b, t);
      status.a.classList.toggle("on", t >= T_A); status.b.classList.toggle("on", t >= T_B);
    } });
    player.seek(RM ? END : 0);
    if (visible && !RM) player.play();
  });

  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (!player || RM) return;
    if (visible) player.restart(); else player.pause();
  }, { threshold: 0.35 }).observe(pair);

  return {
    load,
    layout() { stages.a.resize(); stages.b.resize(); if (player) player.render(player.t); },
    onScroll(vh) {
      if (RM) return;
      const r = pair.getBoundingClientRect();
      const t = clamp((vh * 0.95 - r.top) / (vh * 0.7));
      pair.style.setProperty("--pair-s", lerp(0.9, 1, ease(t)).toFixed(4));
    },
  };
})();

// ---- 5. The course explorer -------------------------------------------------------------------------------------
const explorer = (() => {
  const root = $("[data-explorer]");
  const stages = { a: new Stage($('[data-viewer-canvas="a"]', root), { maxDpr: DPR }),
                   b: new Stage($('[data-viewer-canvas="b"]', root), { maxDpr: DPR }) };
  const statusEl = { a: $('[data-viewer-status="a"]', root), b: $('[data-viewer-status="b"]', root) };
  const playBtn = $("[data-t-play]", root), scrub = $("[data-t-scrub]", root), timeEl = $("[data-t-time]", root);
  let index = null, course = null, runs = null, sel = null, player = null, speed = 1, started = false, loadToken = 0;
  const cells = new Map();
  const picks = { "b-only": [], "a-only": [], both: [] }, pickAt = { "b-only": 0, "a-only": 0, both: 0 };

  const where = { stump: "at a stump", pit: "at a pit", stairs_up: "on stairs going up", stairs_down: "on stairs going down", "open ground": "on open ground" };
  const outcomeText = r => {
    const t = (r.steps / FPS).toFixed(1);
    if (r.outcome === "finished") return `<b>Finished</b> in ${t} s`;
    if (r.outcome === "fell") return `<b>Fell</b> ${where[r.fell_at] || ""} · ${t} s · got ${Math.round(r.distance)}% of the way`;
    if (r.outcome === "backed_off") return `<b>Walked off the start</b> · ${t} s`;
    return `<b>Ran out of time</b> (40 s) · got ${Math.round(r.distance)}% of the way`;
  };

  function setPlaying(on) {
    playBtn.innerHTML = `<span class="${on ? "icon-pause" : "icon-play"}" aria-hidden="true"></span>`;
    playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  function render(t) {
    if (!course) return;
    for (const k of ["a", "b"]) {
      const run = runs[k], pose = stages[k].draw(course, run, t);
      const done = t >= duration(run) - 1e-6;
      statusEl[k].innerHTML = done ? outcomeText(run) : `${t.toFixed(1)} s · ${Math.round(progress(pose[0]))}% of the course`;
    }
    scrub.value = t;
    scrub.style.setProperty("--p", `${(100 * t / (+scrub.max || 1)).toFixed(2)}%`);
    timeEl.textContent = `${t.toFixed(1)} s`;
  }

  async function select(courseSeed, seed, { autoplay = true, scroll = false } = {}) {
    const token = ++loadToken;
    sel = { courseSeed, seed };
    for (const [k, el] of cells) el.setAttribute("aria-pressed", String(k === `${courseSeed}/${seed}`));
    $("[data-viewer-title]", root).textContent = `Course ${courseSeed} · ${RUN_NAMES[seed]} (seed ${seed})`;
    if (scroll) $("[data-viewer]", root).scrollIntoView({ behavior: RM ? "auto" : "smooth", block: "start" });
    const d = await loadRuns(`data/courses/${courseSeed}.bin`);
    if (token !== loadToken) return;
    course = d;
    runs = { a: d.runs.find(r => r.model === "a_100m" && r.seed === seed), b: d.runs.find(r => r.model === "b_100m" && r.seed === seed) };
    const end = Math.max(duration(runs.a), duration(runs.b));
    scrub.max = end.toFixed(2);
    player?.destroy();
    player = new Player({ from: 0, to: end, loop: false, speed, render, onEnd: () => setPlaying(false) });
    player.seek(0);
    if (autoplay && !RM) { player.play(); setPlaying(true); } else setPlaying(false);
  }

  function buildMap() {
    const grids = $("[data-map-grids]", root);
    const bySeed = Object.fromEntries(SEEDS.map(s => [s, []]));
    for (const c of index.courses) for (const s of SEEDS) {
      const a = c.runs.find(r => r.model === "a_100m" && r.seed === s), b = c.runs.find(r => r.model === "b_100m" && r.seed === s);
      bySeed[s].push({ course: c.course_seed, a, b });
      const fa = a.outcome === "finished", fb = b.outcome === "finished";
      if (fb && !fa) picks["b-only"].push([c.course_seed, s]);
      if (fa && !fb) picks["a-only"].push([c.course_seed, s]);
      if (fa && fb) picks.both.push([c.course_seed, s]);
    }
    const say = r => ({ finished: "finished", fell: "fell", timed_out: "ran out of time", backed_off: "walked off the start" }[r.outcome]);
    for (const s of SEEDS) {
      const list = bySeed[s];
      const na = list.filter(x => x.a.outcome === "finished").length, nb = list.filter(x => x.b.outcome === "finished").length;
      const block = document.createElement("div");
      block.className = "grid-block";
      block.innerHTML = `<h3>Training run ${SEEDS.indexOf(s) + 1}</h3><p class="grid-sub">seed ${s} · finished: <b>A ${na}</b> · <b class="b">B ${nb}</b> of 100</p>`;
      const wrap = document.createElement("div");
      wrap.className = "cells";
      wrap.setAttribute("role", "group");
      wrap.setAttribute("aria-label", `Training run ${SEEDS.indexOf(s) + 1}: 100 test courses`);
      for (const x of list) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "cell";
        b.innerHTML = `<i class="${x.a.outcome === "finished" ? "fa" : ""}"></i><i class="${x.b.outcome === "finished" ? "fb" : ""}"></i>`;
        b.setAttribute("aria-label", `Course ${x.course}: A ${say(x.a)}, B ${say(x.b)}`);
        b.setAttribute("aria-pressed", "false");
        b.title = `Course ${x.course} · A ${say(x.a)} · B ${say(x.b)}`;
        b.addEventListener("click", () => select(x.course, s, { scroll: true }));
        cells.set(`${x.course}/${s}`, b);
        wrap.appendChild(b);
      }
      block.appendChild(wrap);
      grids.appendChild(block);
    }
  }

  playBtn.addEventListener("click", () => {
    if (!player) return;
    if (player.playing) { player.pause(); setPlaying(false); } else { player.play(); setPlaying(true); }
  });
  $("[data-t-restart]", root).addEventListener("click", () => { if (player) { player.restart(); setPlaying(true); } });
  scrub.addEventListener("input", () => { if (player) { player.pause(); setPlaying(false); player.seek(+scrub.value); } });
  $$("[data-speed]", root).forEach(btn => btn.addEventListener("click", () => {
    speed = +btn.dataset.speed;
    if (player) player.speed = speed;
    $$("[data-speed]", root).forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  }));
  $$("[data-pick]", root).forEach(btn => btn.addEventListener("click", () => {
    const kind = btn.dataset.pick;
    let c;
    if (kind === "random") c = [index.courses[Math.floor(Math.random() * 100)].course_seed, SEEDS[Math.floor(Math.random() * 3)]];
    else { const list = picks[kind]; c = list[pickAt[kind] % list.length]; pickAt[kind]++; }
    select(c[0], c[1]);
  }));
  $$("[data-load-course]").forEach(a => a.addEventListener("click", e => {
    e.preventDefault();
    if (index) select(+a.dataset.loadCourse, +a.dataset.loadSeed, { scroll: true });
  }));

  const load = fetch("data/runs.json").then(r => r.json()).then(d => {
    index = d;
    buildMap();
    // Start on the course the README's A-vs-B video uses: the first test course, training run 1,
    // that B finishes and A doesn't.
    const first = picks["b-only"].find(([, s]) => s === 42) || picks["b-only"][0];
    pickAt["b-only"] = picks["b-only"].indexOf(first) + 1;
    return select(first[0], first[1], { autoplay: false });
  });

  new IntersectionObserver(([e]) => {
    if (!player) return;
    if (e.isIntersecting && !started && !RM) { started = true; player.restart(); setPlaying(true); }
    else if (!e.isIntersecting && player.playing) { player.pause(); setPlaying(false); }
  }, { threshold: 0.3 }).observe($("[data-viewer]", root));

  return { load, layout() { stages.a.resize(); stages.b.resize(); if (player) render(player.t); } };
})();

// ---- Scroll + resize ------------------------------------------------------------------------------------------------
let ticking = false, lastY = scrollY;
function onScroll() {
  ticking = false;
  const vh = innerHeight, y = scrollY;
  nav.classList.toggle("scrolled", y > 40);
  nav.classList.toggle("hidden", y > lastY + 4 && y > vh * 0.9);
  if (y < lastY - 4) nav.classList.remove("hidden");
  lastY = y;
  if (!RM) {   // the film frame grows from a framed window to its full width as it rises into view
    const r = filmFrame.parentElement.getBoundingClientRect();
    const t = clamp((vh - r.top) / (vh * 0.75));
    filmFrame.style.setProperty("--film-s", lerp(0.86, 1, ease(t)).toFixed(4));
  }
  story.onScroll(vh);
  moment.onScroll(vh);
}
addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });

let resizeTimer = 0;
function layout() { story.layout(); moment.layout(); explorer.layout(); onScroll(); }
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { layout(); redrawCharts(); }, 120); });

story.layout();
onScroll();
Promise.all([story.load, moment.load]).then(() => { layout(); });
explorer.load.catch(err => console.error(err));
fetch("data/site.json").then(r => r.json()).then(drawCharts).catch(err => console.error(err));
document.fonts?.ready.then(() => redrawCharts());
