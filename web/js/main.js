/* main.js — the page's scenes. Every robot is a replay of a recorded test run (replay.js), drawn
   with the designed look (look.js).

   Motion (pinned scenes, scroll-scrubbing, tilt, cursor) only runs when html.motion is set, i.e.
   reduce motion is off. Without it, every scene shows a clear still state. Phones get lighter
   effects: no blur filters, no tilt, no custom cursor, lower canvas resolution, simpler drawing.
   Nerd mode (html.nerd) shows the details, measurements and footnotes. */

import { loadRuns, Stage, Player, duration, poseAt, FPS } from "./replay.js";
import { THEMES, drawFigure } from "./look.js";
import { drawCharts, redrawCharts } from "./charts.js";

const { gsap, ScrollTrigger, SplitText } = window;
gsap.registerPlugin(ScrollTrigger, SplitText);
ScrollTrigger.config({ ignoreMobileResize: true });

const html = document.documentElement;
const MOTION = html.classList.contains("motion");
const FINE = html.classList.contains("fine");
const LIGHT = matchMedia("(hover: none), (pointer: coarse)").matches;
const DPR = LIGHT ? 1.6 : 2;
const narrow = () => innerWidth <= 760;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const SEEDS = [42, 142, 242];
const FINISH_X = (200 - 10) * 14 / 30, START_X = 14 / 30 * 20 / 2;
const progressOf = x => clamp(100 * (x - START_X) / (FINISH_X - START_X), 0, 100);
const stage = (canvas, theme) => new Stage(canvas, { maxDpr: DPR, theme, lite: LIGHT });
const onRefresh = [];
ScrollTrigger.addEventListener("refresh", () => onRefresh.forEach(f => f()));

// ---- Grain ------------------------------------------------------------------------------------------------
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

// ---- Nerd mode ---------------------------------------------------------------------------------------------
(function nerd() {
  const btn = $("[data-nerd]");
  const sync = () => btn.setAttribute("aria-checked", String(html.classList.contains("nerd")));
  sync();
  btn.addEventListener("click", () => {
    const on = html.classList.toggle("nerd");
    try { localStorage.setItem("nerd", on ? "1" : "0"); } catch (e) { /* private mode: just this visit */ }
    sync();
    requestAnimationFrame(() => ScrollTrigger.refresh());
  });
})();

// ---- Count-ups and entrances ---------------------------------------------------------------------------------
function countUp(el) {
  if (el.dataset.done) return;
  el.dataset.done = "1";
  const to = +el.dataset.count, dec = +(el.dataset.dec || 0);
  const f = v => (dec ? v.toFixed(dec) : Math.round(v).toLocaleString("en-US"));
  if (!MOTION) { el.textContent = f(to); return; }
  const o = { v: 0 };
  gsap.to(o, { v: to, duration: 1.8, ease: "power3.out", onUpdate: () => { el.textContent = f(o.v); } });
}

const entrance = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const el = e.target;
    el.classList.add("in");
    $$("[data-count]", el).forEach(countUp);
    $$(".chart, .viz", el).forEach(c => c.classList.add("in"));
    setTimeout(() => el.classList.add("settled"), 1700 + (parseFloat(el.style.getPropertyValue("--d")) || 0));
    entrance.unobserve(el);
  }
}, { rootMargin: "0px 0px -8% 0px", threshold: 0.1 });
$$("[data-up]").forEach(el => {
  const sibs = $$(":scope > [data-up]", el.parentElement);
  el.style.setProperty("--d", `${Math.min(6, sibs.indexOf(el)) * 90}ms`);
  entrance.observe(el);
});
const seen = new IntersectionObserver(entries => entries.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add("in"); seen.unobserve(e.target); }
}), { threshold: 0.25 });
$$(".chart, .g-viz").forEach(c => { if (!(MOTION && c.classList.contains("six-runs"))) seen.observe(c); });   // the 6× scene reveals its own

// ---- Navigation: hide on the way down, match the section underneath ----------------------------------------------
const nav = $("[data-nav]");
let lastY = scrollY;
addEventListener("scroll", () => {
  const y = scrollY;
  nav.classList.toggle("scrolled", y > 30);
  if (y > lastY + 6 && y > innerHeight) nav.classList.add("hidden");
  else if (y < lastY - 6) nav.classList.remove("hidden");
  lastY = y;
}, { passive: true });
$$("[data-theme]").forEach(sec => ScrollTrigger.create({
  trigger: sec, start: "top 56px", end: "bottom 56px",
  onToggle: self => { if (self.isActive) { const light = sec.dataset.theme === "light"; nav.classList.toggle("light", light); html.classList.toggle("on-light", light); } },
}));

// ---- Cursor, tilt and the magnetic button (mouse only) -------------------------------------------------------------
if (FINE && MOTION) {
  html.classList.add("cursor-on");
  const cur = $(".cursor"), dot = $(".cursor-dot"), ring = $(".cursor-ring");
  let mx = -100, my = -100, rx = -100, ry = -100;
  addEventListener("pointermove", e => { mx = e.clientX; my = e.clientY; dot.style.transform = `translate3d(${mx}px,${my}px,0)`; }, { passive: true });
  gsap.ticker.add(() => { rx += (mx - rx) * 0.2; ry += (my - ry) * 0.2; ring.style.transform = `translate3d(${rx}px,${ry}px,0)`; });
  const hot = "a, button, input, .cell, [data-cursor]";
  document.addEventListener("pointerover", e => cur.classList.toggle("hover", !!e.target.closest(hot)));
  addEventListener("pointerdown", () => cur.classList.add("down"));
  addEventListener("pointerup", () => cur.classList.remove("down"));
  document.documentElement.addEventListener("pointerleave", () => { mx = my = -100; });
}

function tilt(el, max = 7) {
  if (!FINE || !MOTION) return;
  gsap.set(el, { transformPerspective: 1000 });
  const rx = gsap.quickTo(el, "rotationX", { duration: 0.7, ease: "power3.out" });
  const ry = gsap.quickTo(el, "rotationY", { duration: 0.7, ease: "power3.out" });
  el.addEventListener("pointermove", e => {
    const r = el.getBoundingClientRect(), px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    ry((px - 0.5) * 2 * max); rx(-(py - 0.5) * 2 * max);
    el.style.setProperty("--gx", `${(px * 100).toFixed(1)}%`); el.style.setProperty("--gy", `${(py * 100).toFixed(1)}%`);
  });
  el.addEventListener("pointerleave", () => { rx(0); ry(0); });
}
$$("[data-tilt]").forEach(el => tilt(el, el.classList.contains("g-card") ? 5 : 7));

(function magnet() {
  const btn = $("[data-magnet]");
  if (!btn || !FINE || !MOTION) return;
  const qx = gsap.quickTo(btn, "x", { duration: 0.8, ease: "power3.out" }), qy = gsap.quickTo(btn, "y", { duration: 0.8, ease: "power3.out" });
  const zone = btn.closest("section");
  zone.addEventListener("pointermove", e => {
    const r = btn.getBoundingClientRect(), dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const d = Math.hypot(dx, dy), pull = clamp(1 - d / 320);
    qx(dx * 0.35 * pull); qy(dy * 0.35 * pull);
  });
  zone.addEventListener("pointerleave", () => { qx(0); qy(0); });
})();

/* A canvas sized to its box at the page's pixel density; draw(ctx, W, H) repaints it. */
function figureCanvas(canvas, draw) {
  const ctx = canvas.getContext("2d");
  const paint = () => {
    const dpr = Math.min(DPR, devicePixelRatio || 1), w = Math.round(canvas.offsetWidth * dpr), h = Math.round(canvas.offsetHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
  };
  onRefresh.push(paint);
  paint();
  return paint;
}

// ---- 01 Hero ------------------------------------------------------------------------------------------------------
function hero(pose) {
  const canvas = $("[data-hero-robot]");
  figureCanvas(canvas, (ctx, w, h) => drawFigure(ctx, w, h, pose, THEMES.dark, { scale: 1, cx: 0.5, floor: 0.92, lite: LIGHT }));
  if (!MOTION) return;
  // Intro (on load) and scroll animations never share a property on the same element.
  gsap.from(".ht > span", { yPercent: 118, duration: 1.7, ease: "expo.out", stagger: 0.12, delay: 0.1 });
  gsap.from([".hero-kicker", ".hero-foot"], { opacity: 0, y: 26, duration: 1.4, ease: "power3.out", stagger: 0.12, delay: 0.55, clearProps: "opacity,transform" });
  gsap.from(canvas, { opacity: 0, duration: 2.4, ease: "expo.out", delay: 0.3 });
  const tl = gsap.timeline({ defaults: { ease: "none" }, scrollTrigger: { trigger: ".s-hero", start: "top top", end: "bottom top", scrub: 0.6 } });
  tl.to(".ht.l1 > span", { xPercent: -26 }, 0)
    .to(".ht.l2 > span", { xPercent: 20 }, 0)
    .to(".ht.l3 > span", { xPercent: -18 }, 0)
    .to(".hero-title", { scale: 1.14, opacity: 0, ease: "power1.in", duration: 0.55 }, 0)
    .to(".hero-inner", { "--hf": 0, duration: 0.3 }, 0)
    .to(canvas, { y: () => -innerHeight * 0.3, scale: 1.16, duration: 1 }, 0)
    .to(".hero-glow", { scale: 1.4, opacity: 0.5, duration: 1 }, 0);
}

// ---- 02 Film ------------------------------------------------------------------------------------------------------
function film() {
  const sec = $(".s-film"), frame = $("[data-film-frame]"), video = $("[data-video]");
  $("[data-video-play]").addEventListener("click", () => {
    sec.classList.add("playing");
    video.controls = true;
    if (video.currentTime < 0.034) video.currentTime = 0.034;      // frame 0 is the poster, baked in for sharing
    video.play().catch(() => {});
  });
  video.addEventListener("ended", () => { video.currentTime = 0.034; });
  new IntersectionObserver(([e]) => { if (!e.isIntersecting && !video.paused) video.pause(); }, { threshold: 0.15 }).observe(frame);
  if (!MOTION) return;
  const s0 = () => (narrow() ? 0.86 : 0.56);
  const o = { s: s0() };
  const set = () => frame.style.setProperty("--fs", o.s.toFixed(4));
  set();
  gsap.timeline({ scrollTrigger: { trigger: sec, start: "top top", end: "bottom bottom", scrub: 0.6, invalidateOnRefresh: true } })
    .fromTo(o, { s: s0 }, { s: 1, duration: 0.62, ease: "power2.inOut", onUpdate: set }, 0)
    .to({}, { duration: 0.38 });
}

// ---- 03 The walker, lit: lidar rays draw on, then the motors light up --------------------------------------------
function reveal(pose) {
  const canvas = $("[data-reveal-robot]");
  const o = { lidar: MOTION ? 0 : 1, motors: MOTION ? 0 : 1 };
  const paint = figureCanvas(canvas, (ctx, w, h) => drawFigure(ctx, w, h, pose, THEMES.dark, {
    scale: narrow() ? 1.05 : 0.86, cx: narrow() ? 0.4 : 0.44, floor: narrow() ? 0.62 : 0.84, lidar: o.lidar, motors: o.motors, lite: LIGHT }));
  // The small visuals beside each spec.
  const fan = $("[data-fan]");
  fan.innerHTML = Array.from({ length: 10 }, (_, i) => {
    const a = 1.5 * i / 10, x = 6 + Math.sin(a) * 56, y = 4 + Math.cos(a) * 38;
    return `<line x1="6" y1="4" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" opacity="${(0.45 + 0.055 * i).toFixed(2)}"/>`;
  }).join("") + '<circle cx="6" cy="4" r="3"/>';
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  $("[data-bars24]").innerHTML = Array.from({ length: 24 }, () =>
    `<i style="--h:${(25 + rnd() * 60).toFixed(0)}%;--h2:${(25 + rnd() * 70).toFixed(0)}%;--dl:-${(rnd() * 0.9).toFixed(2)}s"></i>`).join("");
  const lit = on => $('[data-spec="motors"]').classList.toggle("lit", on);
  if (!MOTION) { lit(true); return; }
  const specs = s => `[data-spec="${s}"]`;
  const tl = gsap.timeline({ defaults: { ease: "power3.out" }, scrollTrigger: { trigger: ".s-reveal", start: "top 65%", end: "bottom bottom", scrub: 0.7 } });
  tl.fromTo('[data-rv="title"]', { opacity: 0, y: 50 }, { opacity: 1, y: 0, duration: 1.4 }, 0)
    .fromTo(".spot", { "--spot": 0 }, { "--spot": 1, duration: 2, ease: "power2.inOut" }, 0.6)
    .fromTo(canvas, { "--rs": 0.86, opacity: 0 }, { "--rs": 1, opacity: 1, duration: 2.4, ease: "expo.out" }, 0.9)
    .to(o, { lidar: 1, duration: 1.6, ease: "power2.out", onUpdate: paint }, 3)
    .fromTo(specs("eyes"), { opacity: 0, y: 34 }, { opacity: 1, y: 0, duration: 1 }, 3.2)
    .to(o, { motors: 1, duration: 1.4, ease: "power2.out", onUpdate: () => { paint(); lit(o.motors > 0.3); } }, 4.4)
    .fromTo(specs("motors"), { opacity: 0, y: 34 }, { opacity: 1, y: 0, duration: 1 }, 4.5)
    .fromTo(specs("senses"), { opacity: 0, y: 34 }, { opacity: 1, y: 0, duration: 1 }, 5.6)
    .fromTo(specs("score"), { opacity: 0, y: 34 }, { opacity: 1, y: 0, duration: 1 }, 6.6)
    .to(canvas, { "--rs": 1.04, duration: 2, ease: "sine.inOut" }, 6.6)
    .to({}, { duration: 1.2 });
  if (FINE) {   // the robot turns a little toward the mouse
    gsap.set(canvas, { transformPerspective: 1400 });
    const ry = gsap.quickTo(canvas, "rotationY", { duration: 1.2, ease: "power3.out" }), rx = gsap.quickTo(canvas, "rotationX", { duration: 1.2, ease: "power3.out" });
    $(".s-reveal").addEventListener("pointermove", e => { ry((e.clientX / innerWidth - 0.5) * 10); rx(-(e.clientY / innerHeight - 0.5) * 5); });
  }
}

// ---- 04 Learning to walk: scroll scrubs the recorded runs -------------------------------------------------------------
function learn(data) {
  const names = ["falls", "wobbles", "walks"];
  const stages = Object.fromEntries(names.map(n => [n, stage($(`[data-learn-canvas="${n}"]`), "light")]));
  const cam = () => ({ viewH: narrow() ? 9 : 8.4, minW: 6, followY: true, yFrac: narrow() ? 0.24 : 0.16, lead: 0.4 });
  const st = { falls: 0, wobbles: 0, walks: 0, steps: 0 };
  const draw = n => stages[n].draw(data[n], data[n].runs[0], st[n], cam());
  const counter = $("[data-learn-steps]");
  const setSteps = () => { counter.textContent = Math.round(st.steps).toLocaleString("en-US"); };
  onRefresh.push(() => names.forEach(n => { stages[n].resize(); draw(n); }));
  if (!MOTION) { st.walks = 8; draw("walks"); st.steps = 524288; setSteps(); return; }
  names.forEach(draw);
  const cv = n => $(`[data-learn-canvas="${n}"]`);
  cv("falls").classList.add("on");
  const words = Object.fromEntries(names.map(n => [n, $(`[data-lw="${n}"]`)]));
  gsap.set(Object.values(words), { yPercent: 160, y: 0 });
  const tw = (n, end) => ({ [n]: end, duration: 0, onUpdate: () => draw(n) });
  const tl = gsap.timeline({ defaults: { ease: "none" }, scrollTrigger: { trigger: ".s-learn", start: "top top", end: "bottom bottom", scrub: 0.5 } });
  tl.to("[data-learn-title] span", { yPercent: -140, opacity: 0, rotation: i => [-8, 5, -4][i], x: i => [-90, 10, 110][i], duration: 5, stagger: 0.7, ease: "power2.in" }, 3)
    .to(st, { ...tw("falls", duration(data.falls.runs[0])), duration: 19 }, 9)
    .to(words.falls, { yPercent: 0, duration: 3, ease: "power3.out" }, 9)
    .to(words.falls, { yPercent: -160, duration: 3, ease: "power3.in" }, 28)
    .to(cv("wobbles"), { opacity: 1, duration: 4, ease: "sine.inOut" }, 28)
    .to(cv("falls"), { opacity: 0, duration: 4, ease: "sine.inOut" }, 28)
    .to(st, { steps: 229376, duration: 6, ease: "power2.out", onUpdate: setSteps }, 28)
    .to(words.wobbles, { yPercent: 0, duration: 3, ease: "power3.out" }, 31)
    .to(st, { ...tw("wobbles", 10.5), duration: 25 }, 32)
    .to(words.wobbles, { yPercent: -160, duration: 3, ease: "power3.in" }, 57)
    .to(cv("walks"), { opacity: 1, duration: 4, ease: "sine.inOut" }, 57)
    .to(cv("wobbles"), { opacity: 0, duration: 4, ease: "sine.inOut" }, 57)
    .to(st, { steps: 524288, duration: 6, ease: "power2.out", onUpdate: setSteps }, 57)
    .to(words.walks, { yPercent: 0, duration: 3, ease: "power3.out" }, 60)
    .to(st, { ...tw("walks", duration(data.walks.runs[0])), duration: 33 }, 61)
    .fromTo("[data-learn-stamp]", { opacity: 0, y: 24, scale: 0.92 }, { opacity: 1, y: 0, scale: 1, duration: 3, ease: "expo.out" }, 94)
    .fromTo("[data-learn-fill]", { scaleX: 0 }, { scaleX: 1, duration: 85 }, 9)
    .to({}, { duration: 3 }, 97);
}

// ---- 05 A | B split -------------------------------------------------------------------------------------------------
function split() {
  if (!MOTION) return;
  const tl = gsap.timeline({ defaults: { ease: "power3.out" }, scrollTrigger: { trigger: ".s-split", start: "top 55%", end: "bottom bottom", scrub: 0.6 } });
  const dx = () => (narrow() ? 0 : 80), dy = () => (narrow() ? 40 : 0);
  tl.from(".half-a > :not(.half-letter)", { x: () => -dx(), y: dy, opacity: 0, duration: 2, stagger: 0.25 }, 0)
    .from(".half-b > :not(.half-letter)", { x: dx, y: dy, opacity: 0, duration: 2, stagger: 0.25 }, 0.2)
    .fromTo(".half-a .half-letter", { yPercent: 30, xPercent: -10 }, { yPercent: -4, xPercent: 0, duration: 7, ease: "none" }, 0)
    .fromTo(".half-b .half-letter", { yPercent: -20, xPercent: 10 }, { yPercent: 4, xPercent: 0, duration: 7, ease: "none" }, 0)
    .fromTo("[data-bar-a]", { scaleX: 0 }, { scaleX: 1, duration: 3, ease: "power2.inOut" }, 2)
    .fromTo("[data-bar-b1]", { scaleX: 0 }, { scaleX: 1, duration: 0.8, ease: "power2.out" }, 2)
    .fromTo("[data-bar-b2]", { scaleX: 0 }, { scaleX: 1, duration: 2.2, ease: "power2.inOut" }, 2.8)
    .fromTo("[data-split-mid] > span", { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 1.2, stagger: 0.4 }, 4.6)
    .to({}, { duration: 1.4 });
}

// ---- 06 Freeze vs climb: scrubbed -----------------------------------------------------------------------------------
function freeze(course) {
  const runA = course.runs.find(r => r.model === "a_100m" && r.seed === 42), runB = course.runs.find(r => r.model === "b_100m" && r.seed === 42);
  const sa = stage($('[data-freeze-canvas="a"]'), "dark"), sb = stage($('[data-freeze-canvas="b"]'), "dark");
  const wa = $('[data-freeze-word="a"]'), wb = $('[data-freeze-word="b"]');
  const cam = () => ({ viewH: narrow() ? 7.2 : 6.4, minW: 6, followY: true, yFrac: 0.12, lead: narrow() ? 0.3 : 0.36 });
  // Course 10016, training run 1: A stops making progress at 4.8 s; B clears the same stump at 6.6 s.
  const o = { t: MOTION ? 0 : 7 };
  const word = (el, t0) => { const k = clamp((o.t - t0) / 0.5); el.style.opacity = k; el.style.transform = `translate3d(0, ${(1 - k) * 40}px, 0) scale(${0.94 + 0.06 * k})`; };
  const draw = () => { sa.draw(course, runA, o.t, cam()); sb.draw(course, runB, o.t, cam()); word(wa, 4.9); word(wb, 6.3); };
  onRefresh.push(() => { sa.resize(); sb.resize(); draw(); });
  draw();
  if (!MOTION) return;
  gsap.timeline({ scrollTrigger: { trigger: ".s-freeze", start: "top top", end: "bottom bottom", scrub: 0.5 } })
    .to(o, { t: 7.0, duration: 0.85, ease: "none", onUpdate: draw }, 0)
    .to({}, { duration: 0.15 });
}

// ---- 07 Gallery: horizontal scroll, each card a real clip and a small chart -------------------------------------------
function cardCharts() {
  for (const box of $$(".g-viz")) {
    const q = `<p class="g-q">${box.dataset.q}</p>`;
    const rows = box.dataset.rows.split(",").map(r => { const [k, v] = r.split(":"); return [k, +v]; });
    if (box.dataset.viz === "dots") {
      const of = +box.dataset.of;
      box.innerHTML = q + rows.map(([k, v]) => `<div class="gd ${k.startsWith("B") ? "b" : "a"}"><span class="gb-k">${k}</span>
        <span class="gd-dots" role="img" aria-label="${k}: ${v} of ${of}">${Array.from({ length: of }, (_, i) => `<i class="${i < v ? "on" : ""}"></i>`).join("")}</span><span class="gb-v">${v}</span></div>`).join("");
    } else {
      const max = +box.dataset.max, unit = box.dataset.unit || "";
      box.innerHTML = q + rows.map(([k, v]) => `<div class="gb ${k.startsWith("B") ? "b" : "a"}"><span class="gb-k">${k}</span>
        <span class="gb-t" role="img" aria-label="${k}: ${v}${unit}"><i style="--w:${(100 * v / max).toFixed(1)}%"></i></span><span class="gb-v">${unit ? Math.round(v) : v.toFixed(1)}${unit}</span></div>`).join("");
    }
  }
}

async function gallery() {
  const clips = await (await fetch("data/gallery.json")).json();
  const cam = { viewH: 6.8, minW: 6, followY: true, yFrac: 0.14, lead: 0.36 };
  for (const card of $$("[data-clip]")) {
    const c = clips[card.dataset.clip];
    const course = await loadRuns(`data/courses/${c.course_seed}.bin`);
    const run = course.runs.find(r => r.model === c.model && r.seed === c.seed);
    const st = stage($("canvas", card), "light");
    const player = new Player({ from: c.t0, to: c.t1, hold: 1.2, loop: MOTION, render: t => st.draw(course, run, t, cam) });
    player.seek(MOTION ? c.t0 : c.t1);
    onRefresh.push(() => { st.resize(); player.render(player.t); });
    if (MOTION) new IntersectionObserver(([e]) => (e.isIntersecting ? player.play() : player.pause()), { threshold: 0.4 }).observe(card);
    else $(".g-media", card).addEventListener("click", () => player.restart());
  }
  if (!MOTION) return;
  const sec = $(".s-gallery"), track = $("[data-gallery-track]");
  const dist = () => Math.max(0, track.scrollWidth - innerWidth);
  const size = () => { sec.style.height = `${dist() + innerHeight}px`; };
  size();
  ScrollTrigger.addEventListener("refreshInit", size);
  gsap.to(track, { x: () => -dist(), ease: "none", scrollTrigger: { trigger: sec, start: "top top", end: "bottom bottom", scrub: 0.6, invalidateOnRefresh: true } });
  gsap.from(".g-card", { y: 60, opacity: 0, duration: 1.2, ease: "power3.out", stagger: 0.1, scrollTrigger: { trigger: sec, start: "top 70%" } });
  ScrollTrigger.refresh();
}

// ---- 08 Six ------------------------------------------------------------------------------------------------------------
// Where the 6× goes once the details rise in: as large as fits between the nav and the details.
function sixFit() {
  const hero = $("[data-six-hero]"), det = $("[data-six-details]");
  const h = hero.offsetHeight, top = 78, detTop = innerHeight - innerHeight * 0.06 - det.offsetHeight - 26;
  const s = Math.min(0.62, (detTop - top) / h);
  return { s, y: (top + detTop) / 2 - innerHeight / 2 };
}

function six() {
  if (!MOTION) return;
  const blur = !LIGHT;
  const tl = gsap.timeline({ scrollTrigger: { trigger: ".s-six", start: "top 35%", end: "bottom bottom", scrub: 0.6, invalidateOnRefresh: true } });
  tl.fromTo(".six-glow", { "--g": 0.25, "--go": 0 }, { "--g": 1.05, "--go": 1, duration: 3, ease: "power2.out" }, 0)
    .fromTo("[data-six-digit]", { scale: 2.8, opacity: 0, filter: blur ? "blur(36px)" : "blur(0px)" }, { scale: 1, opacity: 1, filter: "blur(0px)", duration: 3, ease: "expo.out" }, 0)
    .fromTo("[data-six-x]", { xPercent: 70, opacity: 0, rotation: 20 }, { xPercent: 0, opacity: 1, rotation: 0, duration: 2, ease: "power3.out" }, 1.1)
    .fromTo(".six-kicker", { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 1.4, ease: "power3.out" }, 1.6)
    .fromTo("[data-six-what]", { opacity: 0, y: 50 }, { opacity: 1, y: 0, duration: 1.6, ease: "power3.out" }, 1.8)
    .to({}, { duration: 1.4 }, 3.4)
    .to("[data-six-hero]", { y: () => sixFit().y, scale: () => sixFit().s, duration: 2.6, ease: "power2.inOut" }, 4.8)
    .fromTo("[data-six-details]", { opacity: 0, y: 70 }, { opacity: 1, y: 0, duration: 2, ease: "power3.out" }, 6.9)
    .add(() => $(".six-runs").classList.add("in"), 7.6)
    .to({}, { duration: 2 });
}

// ---- 09 Bento: a small visual for every number --------------------------------------------------------------------------
function bento() {
  for (const v of $$("[data-viz]:not(.g-viz)")) {
    const d = v.dataset, kind = d.viz;
    if (kind === "pair-dots") {
      const row = (k, n) => `<div class="pd-row ${k.toLowerCase()}"><span class="pd-k">${k}</span><span class="pd-dots">${Array.from({ length: n }, (_, i) => `<i style="transition-delay:${(i * 22)}ms"></i>`).join("")}</span></div>`;
      v.innerHTML = row("B", +d.b) + row("A", +d.a);
    } else if (kind === "track") {
      v.innerHTML = `<div class="tr"><span class="tr-line"></span><span class="tr-flag"></span>
        <span class="tr-run a" style="--x:${d.a}%"><span>${d.a}%</span>A</span><span class="tr-run b" style="--x:${d.b}%"><span>${d.b}%</span>B</span></div>`;
    } else if (kind === "bars2") {
      const max = +d.max, la = d.la || "A", lb = d.lb || "B";
      const goal = d.goal ? `<span class="b2-goal" style="left:${(100 * d.goal / max).toFixed(1)}%"><span>${d.goal} = solved</span></span>` : "";
      const row = (k, val, cls, g) => `<div class="b2 ${cls}"><span class="b2-k">${k}</span><span class="b2-t"><i style="--w:${(100 * val / max).toFixed(1)}%"></i>${g}</span><span class="b2-v">${val}</span></div>`;
      v.innerHTML = row(lb, +d.b, "b", goal) + row(la, +d.a, "a", "");
    } else if (kind === "stack") {
      const runs = d.runs.split(",").map(Number), max = Math.max(...runs);
      v.innerHTML = `<div class="st">${runs.map((r, i) => `<i class="${r < 50 ? "small" : ""}" style="--h:${(100 * r / max).toFixed(2)}%;transition-delay:${i * 60}ms"></i>`).join("")}</div>`;
    } else if (kind === "field") {
      v.innerHTML = `<div class="fd">${Array.from({ length: 305 }, (_, i) => `<i class="${i >= 293 ? "on" : ""}" style="transition-delay:${Math.floor(i / 25) * 60}ms"></i>`).join("")}</div>`;   // 1 dot = 200 runs
    }
  }
}

// ---- 10 The race: who gets further? ---------------------------------------------------------------------------------------
function race(index) {
  const root = $("[data-viewer]"), dock = $(".dock"), over = $("[data-race-over]");
  const stages = { a: stage($('[data-viewer-canvas="a"]'), "dark"), b: stage($('[data-viewer-canvas="b"]'), "dark") };
  const live = { a: $('[data-viewer-live="a"]'), b: $('[data-viewer-live="b"]') };
  const result = { a: $('[data-viewer-status="a"]'), b: $('[data-viewer-status="b"]') };
  const rail = { a: $('[data-rail="a"]'), b: $('[data-rail="b"]') };
  const playBtn = $("[data-t-play]", dock), scrub = $("[data-t-scrub]", dock);
  const cam = () => ({ viewH: narrow() ? 6.6 : 6.4, minW: 6, followY: true, yFrac: 0.14, lead: 0.36 });
  let course = null, runs = null, sel = null, player = null, speed = 2, token = 0, guess = null, countTl = null;
  const cells = new Map();
  const picks = { "b-only": [], "a-only": [], both: [] }, at = { "b-only": 0, "a-only": 0, both: 0 };
  const where = { stump: "at a stump", pit: "at a pit", stairs_up: "on stairs", stairs_down: "on stairs", "open ground": "on open ground" };
  const pctOf = r => Math.round(r.distance);
  const short = r => {
    if (r.outcome === "finished") return `finished in ${(r.steps / FPS).toFixed(1)} s`;
    if (r.outcome === "fell") return `fell ${where[r.fell_at] || ""}, ${pctOf(r)}%`;
    if (r.outcome === "backed_off") return "walked off the start";
    return `ran out of time, ${pctOf(r)}%`;
  };
  const badge = r => `<span class="${r.outcome === "finished" ? "ok" : r.outcome === "fell" ? "bad" : "meh"}">●</span> ${short(r)[0].toUpperCase()}${short(r).slice(1)}`;
  /* Who wins: whoever finishes (the faster one, if both do); otherwise whoever got further along the course. */
  const winner = (ra, rb) => {
    const fa = ra.outcome === "finished", fb = rb.outcome === "finished";
    if (fa && fb) return ra.steps === rb.steps ? "tie" : ra.steps < rb.steps ? "a" : "b";
    if (fa !== fb) return fa ? "a" : "b";
    if (Math.abs(ra.distance - rb.distance) < 0.5) return "tie";
    return ra.distance > rb.distance ? "a" : "b";
  };
  const setPlaying = on => { playBtn.innerHTML = `<span class="${on ? "i-pause" : "i-play"}" aria-hidden="true"></span>`; playBtn.setAttribute("aria-label", on ? "Pause" : "Play"); };
  const setState = s => { over.dataset.state = s; };

  function render(t) {
    if (!course) return;
    for (const k of ["a", "b"]) {
      const run = runs[k], pose = stages[k].draw(course, run, t, cam());
      const done = t >= duration(run) - 1e-6;
      const p = done ? run.distance : progressOf(pose[0]);
      live[k].textContent = `${Math.round(p)}%`;
      rail[k].style.left = `${(p * 0.97).toFixed(2)}%`;
      if (done && !result[k].classList.contains("on")) { result[k].innerHTML = badge(run); result[k].classList.add("on"); }
      if (!done) result[k].classList.remove("on");
    }
    scrub.value = t;
    scrub.style.setProperty("--p", `${(100 * t / (+scrub.max || 1)).toFixed(2)}%`);
  }

  function finish() {
    setPlaying(false);
    const w = winner(runs.a, runs.b);
    const name = k => (k === "b" ? '<span class="b">B</span>' : "A");
    $("[data-ro-win]").innerHTML = w === "tie" ? "A tie." : `${name(w)} wins.`;
    const order = w === "a" ? ["a", "b"] : ["b", "a"];
    $("[data-ro-sub]").textContent = order.map(k => `${k.toUpperCase()} ${short(runs[k])}`).join(" · ");
    $("[data-ro-you]").textContent = !guess || guess === "none" ? "" : guess === w ? "You called it." : "Not this time.";
    setState("result");
  }

  function go({ countdown = true } = {}) {
    if (!player) return;
    countTl?.kill();
    player.pause(); player.seek(0);
    if (!countdown || !MOTION) { setState("racing"); player.play(); setPlaying(true); return; }
    const n = $("[data-count-n]");
    setState("count");
    countTl = gsap.timeline({ onComplete: () => { setState("racing"); player.play(); setPlaying(true); } });
    ["3", "2", "1"].forEach((v, i) => countTl
      .call(() => { n.textContent = v; }, null, i * 0.62)
      .fromTo(n, { scale: 1.5, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.42, ease: "expo.out" }, i * 0.62));
    countTl.to(n, { opacity: 0, duration: 0.2 }, 1.86);
  }

  async function select(courseSeed, seed, { scroll = false } = {}) {
    const my = ++token;
    countTl?.kill();
    sel = { courseSeed, seed };
    for (const [k, el] of cells) el.setAttribute("aria-pressed", String(k === `${courseSeed}/${seed}`));
    $$("[data-run]", dock).forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.run === seed)));
    $("[data-course-n]", dock).textContent = courseSeed;
    $("[data-viewer-title]", dock).textContent = `Course ${courseSeed} · training run ${SEEDS.indexOf(seed) + 1} (seed ${seed})`;
    if (scroll) root.scrollIntoView({ behavior: MOTION ? "smooth" : "auto", block: "center" });
    const d = await loadRuns(`data/courses/${courseSeed}.bin`);
    if (my !== token) return;
    course = d;
    runs = { a: d.runs.find(r => r.model === "a_100m" && r.seed === seed), b: d.runs.find(r => r.model === "b_100m" && r.seed === seed) };
    result.a.classList.remove("on"); result.b.classList.remove("on");
    const end = Math.max(duration(runs.a), duration(runs.b));
    scrub.max = end.toFixed(2);
    player?.destroy();
    player = new Player({ from: 0, to: end, loop: false, speed, render, onEnd: finish });
    player.seek(0);
    setPlaying(false);
    guess = null;
    setState("pick");
  }

  // The map of all 300 races.
  const grids = $("[data-map-grids]");
  const say = r => ({ finished: "finished", fell: "fell", timed_out: "ran out of time", backed_off: "walked off the start" }[r.outcome]);
  for (const s of SEEDS) {
    const list = index.courses.map(c => ({ course: c.course_seed, a: c.runs.find(r => r.model === "a_100m" && r.seed === s), b: c.runs.find(r => r.model === "b_100m" && r.seed === s) }));
    for (const x of list) {
      const fa = x.a.outcome === "finished", fb = x.b.outcome === "finished";
      if (fb && !fa) picks["b-only"].push([x.course, s]);
      if (fa && !fb) picks["a-only"].push([x.course, s]);
      if (fa && fb) picks.both.push([x.course, s]);
    }
    const na = list.filter(x => x.a.outcome === "finished").length, nb = list.filter(x => x.b.outcome === "finished").length;
    const block = document.createElement("div");
    block.className = "grid-block";
    block.innerHTML = `<h3>Run ${SEEDS.indexOf(s) + 1}</h3><p class="grid-sub">A ${na} · <b class="b">B ${nb}</b></p>`;
    const wrap = document.createElement("div");
    wrap.className = "cells";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", `Training run ${SEEDS.indexOf(s) + 1}: 100 test courses`);
    for (const x of list) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "cell";
      b.innerHTML = `<i class="${x.a.outcome === "finished" ? "fa" : ""}"></i><i class="${x.b.outcome === "finished" ? "fb" : ""}"></i>`;
      b.setAttribute("aria-label", `Race on course ${x.course}: A ${say(x.a)}, B ${say(x.b)}`);
      b.setAttribute("aria-pressed", "false");
      b.title = `Course ${x.course} · A ${say(x.a)} · B ${say(x.b)}`;
      b.addEventListener("click", () => select(x.course, s, { scroll: true }));
      cells.set(`${x.course}/${s}`, b);
      wrap.appendChild(b);
    }
    block.appendChild(wrap);
    grids.appendChild(block);
  }

  const next = () => select(10000 + Math.floor(Math.random() * 100), SEEDS[Math.floor(Math.random() * 3)]);
  $$("[data-guess]", over).forEach(b => b.addEventListener("click", () => { guess = b.dataset.guess; go({ countdown: guess !== "none" }); }));
  $("[data-rematch]", over).addEventListener("click", () => go());
  $$("[data-next]").forEach(b => b.addEventListener("click", next));
  playBtn.addEventListener("click", () => {
    if (!player) return;
    if (over.dataset.state === "pick" || over.dataset.state === "result") { guess = "none"; go({ countdown: false }); return; }
    if (player.playing) { player.pause(); setPlaying(false); } else { setState("racing"); player.play(); setPlaying(true); }
  });
  $("[data-t-restart]", dock).addEventListener("click", () => go());
  scrub.addEventListener("input", () => { if (!player) return; countTl?.kill(); setState("racing"); player.pause(); setPlaying(false); player.seek(+scrub.value); });
  $$("[data-speed]", dock).forEach(btn => btn.addEventListener("click", () => {
    speed = +btn.dataset.speed;
    if (player) player.speed = speed;
    $$("[data-speed]", dock).forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  }));
  $$("[data-run]", dock).forEach(btn => btn.addEventListener("click", () => select(sel.courseSeed, +btn.dataset.run)));
  $$("[data-course-step]", dock).forEach(btn => btn.addEventListener("click", () => {
    const n = 10000 + ((sel.courseSeed - 10000 + +btn.dataset.courseStep + 100) % 100);
    select(n, sel.seed);
  }));
  $$("[data-pick]", dock).forEach(btn => btn.addEventListener("click", () => {
    const kind = btn.dataset.pick, list = picks[kind], c = list[at[kind] % list.length];
    at[kind]++;
    select(c[0], c[1]);
  }));
  $$("[data-load-course]").forEach(a => a.addEventListener("click", e => { e.preventDefault(); select(+a.dataset.loadCourse, +a.dataset.loadSeed, { scroll: true }); }));
  onRefresh.push(() => { stages.a.resize(); stages.b.resize(); if (player) render(player.t); });
  new IntersectionObserver(([e]) => { if (!e.isIntersecting && player?.playing) { player.pause(); setPlaying(false); } }, { threshold: 0.2 }).observe(root);

  // The first race: the first test course, training run 1, that B finishes and A doesn't (the README video's course).
  const first = picks["b-only"].find(([, s]) => s === 42);
  at["b-only"] = picks["b-only"].indexOf(first) + 1;
  return select(first[0], first[1]);
}

// ---- 11 Evidence: the heading follows the chart in view ------------------------------------------------------------------
function evidence() {
  const set = name => $$("[data-ev]").forEach(el => el.classList.toggle("on", el.dataset.ev === name));
  $$("[data-ev-fig]").forEach(fig => ScrollTrigger.create({ trigger: fig, start: "top 60%", end: "bottom 40%", onToggle: self => { if (self.isActive) set(fig.dataset.evFig); } }));
}

// ---- 13 Close ----------------------------------------------------------------------------------------------------------
function close() {
  if (!MOTION) return;
  const split = new SplitText("[data-close-title]", { type: "words,chars", wordsClass: "word", charsClass: "char", mask: "chars" });
  gsap.from(split.chars, { yPercent: 110, duration: 1.4, ease: "expo.out", stagger: 0.035, scrollTrigger: { trigger: ".s-close", start: "top 62%" } });
  gsap.from(".magnet", { opacity: 0, y: 30, duration: 1.2, ease: "power3.out", scrollTrigger: { trigger: ".s-close", start: "top 50%" } });
  gsap.fromTo(".close-glow", { scale: 0.6, opacity: 0 }, { scale: 1, opacity: 1, ease: "none", scrollTrigger: { trigger: ".s-close", start: "top bottom", end: "center center", scrub: 0.6 } });
}

// ---- Start ---------------------------------------------------------------------------------------------------------------
film();
split();
six();
bento();
cardCharts();
evidence();
close();
const stories = Promise.all(["falls", "wobbles", "walks"].map(n => loadRuns(`data/stories/${n}.bin`)))
  .then(([falls, wobbles, walks]) => {
    const pose = poseAt(walks.runs[0], 6.3, walks.xy_scale, walks.angle_scale);
    hero(pose);
    reveal(pose);
    learn({ falls, wobbles, walks });
  });
const freezeLoad = loadRuns("data/courses/10016.bin").then(freeze);
const raceLoad = fetch("data/runs.json").then(r => r.json()).then(race);
fetch("data/site.json").then(r => r.json()).then(site => { drawCharts(site); ScrollTrigger.refresh(); });
Promise.all([stories, freezeLoad, raceLoad]).then(() => gallery()).then(() => ScrollTrigger.refresh()).catch(err => console.error(err));
let resizeTimer = 0;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawCharts, 150); });
document.fonts?.ready.then(() => { redrawCharts(); ScrollTrigger.refresh(); });
