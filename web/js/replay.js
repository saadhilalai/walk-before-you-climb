/* replay.js — recorded test runs, drawn the way the game (and the film) draws them.

   Every robot on this page is a replay: the robot's pose at every 1/50 s of a real recorded
   test run, exported by export_web_data.py, which replayed each run in Python and checked it
   against the recorded result. Nothing is simulated here. The drawing copies render.py: same
   shapes, colours and camera, at whatever size the window is. */

const SCALE = 30;
const VIEW_H = 400 / SCALE;                         // the game shows 13.3 world units of height
const TERRAIN_STEP = 14 / SCALE;
const TERRAIN_HEIGHT = 400 / SCALE / 4;
const HULL = [[-30, 9], [6, 9], [34, 1], [34, -8], [-30, -8]].map(([x, y]) => [x / SCALE, y / SCALE]);
const LEG_W = 8 / SCALE, LEG_H = 34 / SCALE;
const box = (hx, hy) => [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]];
const PARTS = [                                     // drawing order: legs (-1 side first), then the body
  { shape: HULL, fill: "rgb(127,51,229)", edge: "rgb(76,76,127)", index: 0 },
];
const LEGS = [
  { shape: box(LEG_W / 2, LEG_H / 2), fill: "rgb(178,101,152)", edge: "rgb(127,76,101)", index: 1 },
  { shape: box(0.8 * LEG_W / 2, LEG_H / 2), fill: "rgb(178,101,152)", edge: "rgb(127,76,101)", index: 2 },
  { shape: box(LEG_W / 2, LEG_H / 2), fill: "rgb(128,51,102)", edge: "rgb(77,26,51)", index: 3 },
  { shape: box(0.8 * LEG_W / 2, LEG_H / 2), fill: "rgb(128,51,102)", edge: "rgb(77,26,51)", index: 4 },
];
const DRAW_ORDER = [...LEGS, ...PARTS];
export const FPS = 50;

// ---- Loading --------------------------------------------------------------------------------
const cache = new Map();

/* A .bin file: uint32 header length, JSON header, then each run's int16 poses
   (frames x [hull, 4 leg parts] x [x, y, angle]). */
export function loadRuns(url) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(r => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r.arrayBuffer();
    }).then(buf => {
      const n = new DataView(buf).getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
      const base = 4 + n;
      header.runs = header.runs.map(run => ({
        ...run, frames: new Int16Array(buf, base + run.offset, run.frames * 15), count: run.frames,
      }));
      header.groundPolys = [];
      for (let i = 0; i < header.terrain_x.length - 1; i++)
        header.groundPolys.push([header.terrain_x[i], header.terrain_y[i], header.terrain_x[i + 1], header.terrain_y[i + 1]]);
      return header;
    }));
  }
  return cache.get(url);
}

/* The pose at time t (seconds), smoothly between two recorded frames for high refresh rates. */
export function poseAt(run, t, xy, ang) {
  const f = Math.max(0, Math.min(run.count - 1, t * FPS));
  const i = Math.floor(f), j = Math.min(run.count - 1, i + 1), a = f - i;
  const A = run.frames, out = new Array(15);
  for (let p = 0; p < 5; p++) {
    const o = i * 15 + p * 3, q = j * 15 + p * 3;
    out[p * 3] = (A[o] + (A[q] - A[o]) * a) / xy;
    out[p * 3 + 1] = (A[o + 1] + (A[q + 1] - A[o + 1]) * a) / xy;
    let d = (A[q + 2] - A[o + 2]) / ang;
    if (d > Math.PI) d -= 2 * Math.PI; else if (d < -Math.PI) d += 2 * Math.PI;
    out[p * 3 + 2] = A[o + 2] / ang + d * a;
  }
  return out;
}

export const duration = run => (run.count - 1) / FPS;

/* Average ground height over a few tiles around x (a smooth camera height). */
function groundNear(course, x) {
  const xs = course.terrain_x, ys = course.terrain_y;
  let sum = 0, n = 0;
  for (let k = -6; k <= 6; k++) {
    const xx = x + k * TERRAIN_STEP * 0.75;
    let i = Math.floor(xx / TERRAIN_STEP);
    i = Math.max(0, Math.min(xs.length - 2, i));
    const f = Math.max(0, Math.min(1, (xx - xs[i]) / (xs[i + 1] - xs[i])));
    sum += ys[i] + (ys[i + 1] - ys[i]) * f;
    n++;
  }
  return sum / n;
}

// ---- Drawing ----------------------------------------------------------------------------------
export class Stage {
  /* One canvas showing one course. maxDpr caps the pixel density (lighter on phones). */
  constructor(canvas, { maxDpr = 2 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.maxDpr = maxDpr;
    this.course = null;
    this.resize();
  }

  resize() {
    // Layout size (ignores CSS transforms, so a window drawn at full size and shown scaled
    // down stays sharp when it grows), capped at about 4 million pixels per canvas.
    const cw = this.canvas.offsetWidth, ch = this.canvas.offsetHeight;
    const dpr = Math.min(this.maxDpr, window.devicePixelRatio || 1, Math.sqrt(4.2e6 / Math.max(1, cw * ch)));
    const w = Math.max(1, Math.round(cw * dpr)), h = Math.max(1, Math.round(ch * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.W = w;
    this.H = h;
  }

  /* cam: { viewH: world units shown top to bottom (default: the game's 13.3), minW: at least this
     many units across, followY: keep the ground under the robot at yFrac of the height (for
     close-ups; the ground is averaged over a few tiles so the camera glides over pits and stairs). */
  draw(course, run, t, cam = {}) {
    if (!course) return;
    const { ctx, W, H } = this;
    const pose = poseAt(run, t, course.xy_scale, course.angle_scale);
    const s = Math.min(H / (cam.viewH || VIEW_H), W / (cam.minW || 12));
    const viewW = W / s;
    const scroll = pose[0] - Math.min(4, viewW * (cam.lead ?? 0.28));
    const sx = Math.max(0, scroll);
    const sy = cam.followY ? groundNear(course, pose[0]) - (H / s) * (cam.yFrac ?? 0.22) : 0;
    const X = x => (x - sx) * s, Y = y => H - (y - sy) * s;
    const k = s / SCALE, lw = Math.max(1, k);

    ctx.fillStyle = "rgb(215,215,255)";
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = "#fff";                         // clouds move at half speed
    for (const poly of course.clouds) {
      ctx.beginPath();
      poly.forEach(([x, y], n) => (n ? ctx.lineTo : ctx.moveTo).call(ctx, (x + scroll / 2 - sx) * s, Y(y)));
      ctx.closePath();
      ctx.fill();
    }

    ctx.fillStyle = "rgb(102,153,76)";              // the ground
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of course.groundPolys) {
      if (x2 < sx - 1 || x1 > sx + viewW + 1) continue;
      ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.lineTo(X(x2), H + 2); ctx.lineTo(X(x1), H + 2); ctx.closePath();
    }
    ctx.fill();

    ctx.lineWidth = lw;
    ctx.lineCap = "round";                          // the grass edge, alternating greens
    course.groundPolys.forEach(([x1, y1, x2, y2], i) => {
      if (x2 < sx - 1 || x1 > sx + viewW + 1) return;
      ctx.strokeStyle = i % 2 ? "rgb(76,204,76)" : "rgb(76,255,76)";
      ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke();
    });

    for (const poly of course.blocks) {             // obstacles, drawn over the grass as in the game
      const xs = poly.map(v => v[0]);
      if (Math.max(...xs) < sx - 1 || Math.min(...xs) > sx + viewW + 1) continue;
      ctx.beginPath();
      poly.forEach(([x, y], n) => (n ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(y)));
      ctx.closePath();
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.strokeStyle = "rgb(153,153,153)"; ctx.stroke();
    }

    ctx.lineJoin = "round";                         // the robot
    for (const part of DRAW_ORDER) {
      const px = pose[part.index * 3], py = pose[part.index * 3 + 1], a = pose[part.index * 3 + 2];
      const c = Math.cos(a), sn = Math.sin(a);
      ctx.beginPath();
      part.shape.forEach(([vx, vy], n) => {
        const wx = px + c * vx - sn * vy, wy = py + sn * vx + c * vy;
        (n ? ctx.lineTo : ctx.moveTo).call(ctx, X(wx), Y(wy));
      });
      ctx.closePath();
      ctx.fillStyle = part.fill; ctx.fill();
      ctx.strokeStyle = part.edge; ctx.stroke();
    }

    const fx = X(TERRAIN_STEP * 3), fy1 = Y(TERRAIN_HEIGHT), fy2 = fy1 - 50 * k;   // the start flag
    if (fx > -30 * k) {
      ctx.strokeStyle = "#000";
      ctx.beginPath(); ctx.moveTo(fx, fy1); ctx.lineTo(fx, fy2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(fx, fy2); ctx.lineTo(fx, fy2 + 10 * k); ctx.lineTo(fx + 25 * k, fy2 + 5 * k); ctx.closePath();
      ctx.fillStyle = "rgb(230,51,0)"; ctx.fill(); ctx.stroke();
    }
    return pose;
  }
}

// ---- One clock for every playing replay ----------------------------------------------------------
const players = new Set();
let rafId = 0;

function tick(now) {
  rafId = 0;
  for (const p of players) p._tick(now);
  if ([...players].some(p => p.playing)) rafId = requestAnimationFrame(tick);
}

export function wake() { if (!rafId) rafId = requestAnimationFrame(tick); }

/* Plays one or more stages from a shared clock. render(t) draws; t runs from `from` to `to`,
   then holds for `hold` seconds and loops (or stops when loop is false). */
export class Player {
  constructor({ render, from = 0, to, hold = 1.5, loop = true, speed = 1, onTime = null, onEnd = null }) {
    Object.assign(this, { render, from, to, hold, loop, speed, onTime, onEnd });
    this.t = from;
    this.playing = false;
    this.last = null;
    this.ended = false;
    players.add(this);
  }

  play() { if (this.ended && !this.loop) { this.t = this.from; this.ended = false; } this.playing = true; this.last = null; wake(); }
  pause() { this.playing = false; }
  seek(t) { this.t = Math.max(this.from, Math.min(this.to, t)); this.ended = false; this.holdLeft = null; this.render(this.t); this.onTime?.(this.t); }
  restart() { this.seek(this.from); this.play(); }
  destroy() { players.delete(this); }

  _tick(now) {
    if (!this.playing) return;
    const dt = this.last === null ? 0 : Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.t >= this.to) {
      if (!this.ended) { this.ended = true; this.onEnd?.(); this.holdLeft = this.hold; }
      if (!this.loop) { this.playing = false; return; }
      this.holdLeft -= dt;
      if (this.holdLeft <= 0) { this.t = this.from; this.ended = false; }
    } else {
      this.t = Math.min(this.to, this.t + dt * this.speed);
    }
    this.render(this.t);
    this.onTime?.(this.t);
  }
}
