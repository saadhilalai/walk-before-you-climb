/* look.js — how the recorded runs are drawn: a designed robot (matte graphite body, metal joints,
   one small accent light) on a designed course (clean surfaces, soft shadows, a faint grid).

   Only the drawing lives here. Every part of the robot is placed exactly at its recorded position
   and angle, the course is the recorded course, and the finish line sits where the game ends a run
   as finished. Rounding, shading and shadows are paint: they never move anything. */

const SCALE = 30;
export const TERRAIN_STEP = 14 / SCALE;
export const FINISH_X = (200 - 10) * TERRAIN_STEP;          // the game's "finished" line: (TERRAIN_LENGTH - TERRAIN_GRASS) * step
const START_X = 3 * TERRAIN_STEP;                            // where the game draws its start flag
const HULL = [[-30, 9], [6, 9], [34, 1], [34, -8], [-30, -8]].map(([x, y]) => [x / SCALE, y / SCALE]);
const LEG_W = 8 / SCALE, LEG_H = 34 / SCALE;
const KEY = norm(-0.55, 1);                                  // key light from the upper left
const RIM = norm(0.45, 1);                                   // rim light from behind, upper right

function norm(x, y) { const l = Math.hypot(x, y); return [x / l, y / l]; }

export const THEMES = {
  dark: {
    sky: ["#121218", "#08080b"], glow: [150, 140, 255], glowA: 0.13, grid: [198, 192, 255], gridA: 0.06,
    ground: ["#1d1d24", "#09090c"], edge: "rgba(236,234,255,0.7)", face: "rgba(236,234,255,0.22)", bevel: "rgba(198,192,255,0.05)",
    block: ["#383843", "#202027"], blockSide: "rgba(255,255,255,0.08)",
    pit: ["rgba(6,6,9,0)", "#020203"], shadow: [0, 0, 0], shadowA: 0.62, castA: 0.5,
    body: ["#6a6a75", "#3a3a43", "#17171c"], legNear: ["#70707b", "#3f3f48", "#18181d"], legFar: ["#45454e", "#29292f", "#0e0e11"],
    pack: ["#3b3b44", "#222228", "#0c0c0f"], pelvis: ["#56565f", "#2e2e36", "#121216"], fist: ["#6a6a75", "#3a3a43", "#18181d"], torso: ["#7e7e8a", "#464651", "#1a1a20"],
    rim: [238, 235, 255], rimA: 1, chamfer: "rgba(255,255,255,0.10)", bounceA: 0.45, seam: "rgba(0,0,0,0.45)", seamHi: "rgba(255,255,255,0.08)", rubber: "#0b0b0e",
    accent: [198, 192, 255], accentCore: "#efedff", marker: [198, 192, 255], additive: true,
  },
  light: {
    sky: ["#f6f5f1", "#e9e7e1"], glow: [120, 110, 240], glowA: 0.07, grid: [40, 34, 90], gridA: 0.065,
    ground: ["#dcd9d2", "#c8c5bd"], edge: "rgba(255,255,255,1)", face: "rgba(80,74,100,0.25)", bevel: "rgba(255,255,255,0.3)",
    block: ["#ebe9e4", "#d2cfc8"], blockSide: "rgba(255,255,255,0.6)",
    pit: ["rgba(60,56,70,0)", "rgba(40,37,48,0.55)"], shadow: [30, 27, 45], shadowA: 0.42, castA: 0.26,
    body: ["#62626d", "#33333b", "#18181c"], legNear: ["#686873", "#3a3a42", "#18181c"], legFar: ["#4a4a53", "#2a2a30", "#111114"],
    pack: ["#3e3e46", "#24242a", "#0e0e11"], pelvis: ["#575761", "#2f2f36", "#131316"], fist: ["#62626d", "#33333b", "#18181c"], torso: ["#74747f", "#41414a", "#19191e"],
    rim: [255, 255, 255], rimA: 0.8, chamfer: "rgba(255,255,255,0.12)", bounceA: 0.2, seam: "rgba(0,0,0,0.45)", seamHi: "rgba(255,255,255,0.1)", rubber: "#0d0d10",
    accent: [124, 112, 245], accentCore: "#f3f1ff", marker: [107, 94, 242], additive: false,
  },
};

// ---- Soft sprites (pre-rendered once; drawing a scaled image is far cheaper than blur) ------------
const sprites = new Map();
function softDot([r, g, b]) {
  const key = `${r},${g},${b}`;
  if (!sprites.has(key)) {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const x = c.getContext("2d"), gr = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, `rgba(${key},1)`); gr.addColorStop(0.35, `rgba(${key},0.55)`); gr.addColorStop(1, `rgba(${key},0)`);
    x.fillStyle = gr; x.fillRect(0, 0, 128, 128);
    sprites.set(key, c);
  }
  return sprites.get(key);
}
function blob(ctx, rgb, cx, cy, rx, ry, alpha) {             // screen-space soft ellipse
  if (alpha <= 0.003 || rx <= 0 || ry <= 0) return;
  const prev = ctx.globalAlpha;                                  // respects an outer fade
  ctx.globalAlpha = prev * Math.min(1, alpha);
  ctx.drawImage(softDot(rgb), cx - rx, cy - ry, rx * 2, ry * 2);
  ctx.globalAlpha = prev;
}
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

// ---- The course, sorted once into what to draw -------------------------------------------------
function courseLook(course) {
  if (course._look) return course._look;
  const pits = (course.obstacles || []).filter(o => o.kind === "pit");
  const kindAt = x => (course.obstacles || []).find(o => x >= o.x_start - 1e-3 && x <= o.x_end + 1e-3)?.kind;
  const blocks = (course.blocks || []).map(poly => {
    const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
    const b = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    b.kind = kindAt((b.x0 + b.x1) / 2) || "stump";
    b.wall = b.kind === "pit";
    return b;
  });
  const holes = pits.map(p => {                                // the open part of a pit, between its two walls
    const walls = blocks.filter(b => b.wall && b.x0 >= p.x_start - 1e-3 && b.x1 <= p.x_end + 1e-3).sort((a, b) => a.x0 - b.x0);
    if (walls.length < 2) return null;
    const x0 = walls[0].x1, x1 = walls[walls.length - 1].x0;
    return { x0, x1, top: Math.max(walls[0].y1, walls[walls.length - 1].y1), floor: terrainAt(course, (x0 + x1) / 2) };
  }).filter(Boolean);
  return (course._look = { blocks, holes });
}

export function terrainAt(course, x) {
  const xs = course.terrain_x, ys = course.terrain_y;
  let i = Math.floor(x / TERRAIN_STEP);
  i = Math.max(0, Math.min(xs.length - 2, i));
  const f = Math.max(0, Math.min(1, (x - xs[i]) / (xs[i + 1] - xs[i])));
  return ys[i] + (ys[i + 1] - ys[i]) * f;
}

/* The real walking surface at x: the highest of the ground line, a stump or stair-slab top,
   or a pit's rim (the game's stairs are slabs set into a slope, so the slope shows between them). */
export function surfaceAt(course, x, blocks = courseLook(course).blocks) {
  let y = terrainAt(course, x);
  for (const b of blocks) if (x >= b.x0 && x <= b.x1) y = Math.max(y, b.y1);
  return y;
}

/* The surface outline across [left, right]: dense samples plus the exact corners of every block. */
function surfaceLine(course, left, right) {
  const look = courseLook(course), near = look.blocks.filter(b => b.x1 >= left - 0.1 && b.x0 <= right + 0.1);
  const xs = [];
  for (let x = left; x < right; x += 0.04) xs.push(x);
  xs.push(right);
  const t = course.terrain_x;
  for (let i = Math.max(0, Math.floor(left / TERRAIN_STEP)); i < t.length && t[i] <= right; i++) if (t[i] >= left) xs.push(t[i]);
  for (const b of near) for (const x of [b.x0 - 1e-4, b.x0 + 1e-4, b.x1 - 1e-4, b.x1 + 1e-4]) if (x >= left && x <= right) xs.push(x);
  xs.sort((a, b) => a - b);
  return xs.map(x => [x, surfaceAt(course, x, near)]);
}

// ---- A view: world units -> device pixels ---------------------------------------------------------
export function makeView(W, H, s, sx, sy) {
  return { W, H, s, sx, sy, X: x => (x - sx) * s, Y: y => H - (y - sy) * s };
}

// ---- The whole frame --------------------------------------------------------------------------------
/* opts: { lite: phones (no grid, fewer details), robot: true, markers: true } */
export function drawScene(ctx, v, course, pose, T, opts = {}) {
  const lite = !!opts.lite;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const sky = ctx.createLinearGradient(0, 0, 0, v.H);
  sky.addColorStop(0, T.sky[0]); sky.addColorStop(1, T.sky[1]);
  ctx.fillStyle = sky; ctx.fillRect(0, 0, v.W, v.H);
  if (!lite) drawGrid(ctx, v, T);
  if (pose) {                                                     // a soft glow that travels with the robot
    const gx = v.X(pose[0]), gy = v.Y(pose[1] + 0.4), r = 4.2 * v.s;
    if (T.additive) ctx.globalCompositeOperation = "lighter";
    blob(ctx, T.glow, gx, gy, r * 1.25, r, T.glowA);
    ctx.globalCompositeOperation = "source-over";
  }
  drawGround(ctx, v, course, T, lite);
  if (opts.markers !== false) drawMarkers(ctx, v, course, T);
  if (pose && opts.robot !== false) {
    drawRobotShadows(ctx, v, course, pose, T);
    drawRobot(ctx, v, pose, T, { lite, shell: opts.shell ?? 1 });
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawGrid(ctx, v, T) {                                    // a faint floor-to-sky grid, half-speed parallax
  const step = 1.5, par = 0.5;
  const bx = v.sx * par, by = v.sy * par;
  const fade = ctx.createLinearGradient(0, 0, 0, v.H);           // faint at the top, clearer near the ground
  fade.addColorStop(0, rgba(T.grid, 0)); fade.addColorStop(0.7, rgba(T.grid, T.gridA)); fade.addColorStop(1, rgba(T.grid, T.gridA));
  ctx.strokeStyle = fade; ctx.lineWidth = 1;
  ctx.beginPath();
  const x0 = Math.floor(bx / step) * step;
  for (let gx = x0; gx < bx + v.W / v.s + step; gx += step) { const X = (gx - bx) * v.s; ctx.moveTo(Math.round(X) + 0.5, 0); ctx.lineTo(Math.round(X) + 0.5, v.H); }
  const y0 = Math.floor(by / step) * step;
  for (let gy = y0; gy < by + v.H / v.s + step; gy += step) { const Y = v.H - (gy - by) * v.s; ctx.moveTo(0, Math.round(Y) + 0.5); ctx.lineTo(v.W, Math.round(Y) + 0.5); }
  ctx.stroke();
}

function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }

function drawGround(ctx, v, course, T, lite) {
  const left = v.sx - 0.5, right = v.sx + v.W / v.s + 0.5;
  const look = courseLook(course), line = surfaceLine(course, left, right);
  const mid = terrainAt(course, v.sx + v.W / v.s / 2), px = Math.max(1, v.s / 40);
  const g = ctx.createLinearGradient(0, v.Y(mid + 0.2), 0, v.Y(mid - 3.2));
  g.addColorStop(0, T.ground[0]); g.addColorStop(1, T.ground[1]);

  const body = new Path2D();                                      // the ground, under the real surface
  body.moveTo(v.X(line[0][0]), v.H + 2);
  for (const [x, y] of line) body.lineTo(v.X(x), v.Y(y));
  body.lineTo(v.X(line[line.length - 1][0]), v.H + 2); body.closePath();
  ctx.fillStyle = g; ctx.fill(body);

  const edge = new Path2D(), faces = new Path2D();                // the outline: tops bright, vertical faces dim
  let open = false;
  for (let i = 1; i < line.length; i++) {
    const [x0, y0] = line[i - 1], [x1, y1] = line[i];
    if (Math.abs(y1 - y0) > 2.5 * Math.abs(x1 - x0)) { faces.moveTo(v.X(x0), v.Y(y0)); faces.lineTo(v.X(x1), v.Y(y1)); open = false; continue; }
    if (!open) edge.moveTo(v.X(x0), v.Y(y0));
    edge.lineTo(v.X(x1), v.Y(y1)); open = true;
  }
  ctx.save(); ctx.clip(body);                                     // a lit top band, fading into the ground
  ctx.lineJoin = "round"; ctx.lineCap = "butt";
  ctx.strokeStyle = T.bevel; ctx.lineWidth = 0.5 * v.s; ctx.stroke(edge);
  ctx.lineWidth = 0.16 * v.s; ctx.stroke(edge);
  ctx.restore();

  for (const h of look.holes) {                                   // pits: open, darkening with depth
    if (h.x1 < left || h.x0 > right) continue;
    const pg = ctx.createLinearGradient(0, v.Y(h.top), 0, v.Y(h.floor));
    pg.addColorStop(0, T.pit[0]); pg.addColorStop(1, T.pit[1]);
    ctx.fillStyle = pg; ctx.fillRect(v.X(h.x0), v.Y(h.top), (h.x1 - h.x0) * v.s, (h.top - h.floor) * v.s + 1);
  }

  for (const b of look.blocks) {                                  // soft contact shade where blocks meet the ground
    if (b.wall || b.x1 < left - 1 || b.x0 > right + 1) continue;
    const base = v.Y(terrainAt(course, (b.x0 + b.x1) / 2));
    blob(ctx, T.shadow, v.X(b.x1 + 0.22), base, 0.5 * v.s, 0.07 * v.s, T.castA);
    blob(ctx, T.shadow, v.X((b.x0 + b.x1) / 2), base, ((b.x1 - b.x0) / 2 + 0.3) * v.s, 0.06 * v.s, T.shadowA * 0.6);
  }
  for (const b of look.blocks) {                                  // stumps and stair slabs: their own lighter material
    if (b.wall || b.x1 < left || b.x0 > right) continue;
    const x = v.X(b.x0), y = v.Y(b.y1), w = (b.x1 - b.x0) * v.s, h = (b.y1 - b.y0) * v.s, r = Math.min(0.04 * v.s, w / 4);
    const bg = ctx.createLinearGradient(0, y, 0, y + h);
    bg.addColorStop(0, T.block[0]); bg.addColorStop(1, T.block[1]);
    ctx.beginPath(); roundTop(ctx, x, y, w, h, r); ctx.fillStyle = bg; ctx.fill();
    ctx.beginPath(); ctx.moveTo(x + px, y + h); ctx.lineTo(x + px, y + r);
    ctx.strokeStyle = T.blockSide; ctx.lineWidth = 1.2 * px; ctx.stroke();
  }

  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = T.face; ctx.lineWidth = 1.2 * px; ctx.stroke(faces);
  ctx.strokeStyle = T.edge; ctx.lineWidth = 1.5 * px; ctx.stroke(edge);
}

function roundTop(ctx, x, y, w, h, r) {
  ctx.moveTo(x, y + h); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h); ctx.closePath();
}

function drawMarkers(ctx, v, course, T) {
  const px = Math.max(1, v.s / 40);
  for (const [x, strong] of [[START_X, false], [FINISH_X, true]]) {
    const X = v.X(x);
    if (X < -40 || X > v.W + 40) continue;
    const y = v.Y(terrainAt(course, x)), top = y - (strong ? 3.2 : 1.6) * v.s;
    const lg = ctx.createLinearGradient(0, y, 0, top);
    lg.addColorStop(0, rgba(T.marker, strong ? 0.95 : 0.5)); lg.addColorStop(1, rgba(T.marker, 0));
    ctx.fillStyle = lg; ctx.fillRect(X - (strong ? 1.5 : 0.75) * px, top, (strong ? 3 : 1.5) * px, y - top);
    if (strong) {
      if (T.additive) ctx.globalCompositeOperation = "lighter";
      blob(ctx, T.marker, X, y, 0.9 * v.s, 0.22 * v.s, 0.55);
      blob(ctx, T.marker, X, y - 1.2 * v.s, 0.35 * v.s, 1.6 * v.s, 0.12);
      ctx.globalCompositeOperation = "source-over";
    }
  }
}

// ---- The robot -------------------------------------------------------------------------------------
const part = (pose, k) => [pose[k * 3], pose[k * 3 + 1], pose[k * 3 + 2]];
const along = ([x, y, a], d) => [x - Math.sin(a) * d, y + Math.cos(a) * d];   // a point on a leg's long axis

function place(ctx, v, x, y, a) { ctx.setTransform(v.s, 0, 0, v.flip ? v.s : -v.s, v.X(x), v.Y(y)); ctx.rotate(a); }

/* Soft contact shadows: darker as a foot (or the body) gets closer to the surface under it. */
function drawRobotShadows(ctx, v, course, pose, T) {
  for (const k of [2, 4]) {
    const [fx, fy] = along(part(pose, k), -LEG_H / 2);
    const gy = surfaceAt(course, fx), h = Math.max(0, fy - gy);
    const a = T.shadowA * Math.max(0, 1 - h / 0.9);
    blob(ctx, T.shadow, v.X(fx), v.Y(gy), (0.26 + h * 0.25) * v.s, (0.06 + h * 0.03) * v.s, a);
  }
  const [hx, hy] = pose, gy = surfaceAt(course, hx), h = Math.max(0, hy - gy - 0.6);
  blob(ctx, T.shadow, v.X(hx), v.Y(gy), (0.95 + h * 0.3) * v.s, 0.12 * v.s, T.shadowA * 0.55 * Math.max(0, 1 - h / 2.2));
}

/* shell: 1 = the humanoid redesign, 0 = the real simulated body (the game's box), in between = a crossfade.
   The legs are the same in both. */
export function drawRobot(ctx, v, pose, T, { lite = false, shell = 1 } = {}) {
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  const px = 1 / v.s;                                              // one device pixel, in world units
  const H = part(pose, 0), fade = shell > 0.002 && shell < 0.998;
  if (shell > 0.002 && !fade) shellBack(ctx, v, H, T, px, lite);
  leg(ctx, v, part(pose, 2), 0.8 * LEG_W / 2, T.legFar, T, px, true, lite);
  leg(ctx, v, part(pose, 1), LEG_W / 2, T.legFar, T, px, false, lite);
  joint(ctx, v, along(part(pose, 1), -LEG_H / 2), 0.115, T, true, px);
  joint(ctx, v, along(part(pose, 1), LEG_H / 2), 0.13, T, true, px);
  if (shell < 0.998) { ctx.globalAlpha = 1 - shell; hull(ctx, v, H, T, px, lite); ctx.globalAlpha = 1; }
  if (fade) {                                                     // crossfade: the whole shell on its own layer
    const L = layerFor(ctx);
    L.setTransform(1, 0, 0, 1, 0, 0); L.clearRect(0, 0, L.canvas.width, L.canvas.height);
    shellBack(L, v, H, T, px, lite); shellBody(L, v, H, T, px, lite); shellFront(L, v, H, T, px, lite);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = shell; ctx.drawImage(L.canvas, 0, 0); ctx.globalAlpha = 1;
  } else if (shell >= 0.998) shellBody(ctx, v, H, T, px, lite);
  leg(ctx, v, part(pose, 4), 0.8 * LEG_W / 2, T.legNear, T, px, true, lite);
  leg(ctx, v, part(pose, 3), LEG_W / 2, T.legNear, T, px, false, lite);
  joint(ctx, v, along(part(pose, 3), -LEG_H / 2), 0.115, T, false, px);
  joint(ctx, v, along(part(pose, 3), LEG_H / 2), 0.13, T, false, px);
  if (shell >= 0.998) shellFront(ctx, v, H, T, px, lite);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

const layers = new WeakMap();
function layerFor(ctx) {
  let L = layers.get(ctx.canvas);
  if (!L || L.canvas.width !== ctx.canvas.width || L.canvas.height !== ctx.canvas.height) {
    const c = document.createElement("canvas"); c.width = ctx.canvas.width; c.height = ctx.canvas.height;
    L = c.getContext("2d"); layers.set(ctx.canvas, L);
  }
  return L;
}

function roundPoly(ctx, pts, r) {
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = pts[(i + n - 1) % n], [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % n];
    const a = Math.atan2(y0 - y1, x0 - x1), b = Math.atan2(y2 - y1, x2 - x1);
    const d1 = Math.hypot(x0 - x1, y0 - y1), d2 = Math.hypot(x2 - x1, y2 - y1), rr = Math.min(r, d1 / 2, d2 / 2);
    const p = [x1 + Math.cos(a) * rr, y1 + Math.sin(a) * rr], q = [x1 + Math.cos(b) * rr, y1 + Math.sin(b) * rr];
    i ? ctx.lineTo(...p) : ctx.moveTo(...p);
    ctx.quadraticCurveTo(x1, y1, ...q);
  }
  ctx.closePath();
}

function hull(ctx, v, [x, y, a], T, px, lite) {
  place(ctx, v, x, y, a);
  const ux = Math.sin(a), uy = Math.cos(a);                       // world "up", in the body's own frame
  ctx.beginPath(); roundPoly(ctx, HULL, 0.055);
  const g = ctx.createLinearGradient(ux * 0.32, uy * 0.32, -ux * 0.3, -uy * 0.3);
  g.addColorStop(0, T.body[0]); g.addColorStop(0.45, T.body[1]); g.addColorStop(1, T.body[2]);
  ctx.fillStyle = g; ctx.fill();
  const rx = RIM[0] * Math.cos(-a) - RIM[1] * Math.sin(-a), ry = RIM[0] * Math.sin(-a) + RIM[1] * Math.cos(-a);
  ctx.save(); ctx.clip();                                         // a soft chamfer band inside the edge, lit from above
  const cg = ctx.createLinearGradient(ux * 0.32, uy * 0.32, -ux * 0.1, -uy * 0.1);
  cg.addColorStop(0, T.chamfer); cg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.strokeStyle = cg; ctx.lineWidth = 0.075; ctx.stroke();
  ctx.restore();
  const rg = ctx.createLinearGradient(rx * 0.35, ry * 0.35, -rx * 0.1, -ry * 0.1);   // rim light on the edges facing it
  rg.addColorStop(0, rgba(T.rim, T.rimA)); rg.addColorStop(0.55, rgba(T.rim, 0.14)); rg.addColorStop(1, rgba(T.rim, 0));
  ctx.strokeStyle = rg; ctx.lineWidth = 2.2 * px; ctx.stroke();
  const bl = ctx.createLinearGradient(-ux * 0.32, -uy * 0.32, ux * 0.05, uy * 0.05);   // bounce light from below
  bl.addColorStop(0, rgba(T.accent, T.bounceA)); bl.addColorStop(1, rgba(T.accent, 0));
  ctx.strokeStyle = bl; ctx.lineWidth = 1.5 * px; ctx.stroke();
  if (lite) return;
  ctx.beginPath();                                                // panel seams
  ctx.moveTo(-27 / SCALE, -3 / SCALE); ctx.lineTo(29 / SCALE, -3 / SCALE);
  ctx.moveTo(-10 / SCALE, 7.5 / SCALE); ctx.lineTo(-10 / SCALE, -3 / SCALE);
  ctx.strokeStyle = T.seam; ctx.lineWidth = 1.1 * px; ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-27 / SCALE, -3 / SCALE - 1.1 * px); ctx.lineTo(29 / SCALE, -3 / SCALE - 1.1 * px);
  ctx.strokeStyle = T.seamHi; ctx.lineWidth = px; ctx.stroke();
  ctx.beginPath();                                                // three vents at the back
  for (let k = 0; k < 3; k++) { const vx = (-25 + k * 3) / SCALE; ctx.moveTo(vx, 5.5 / SCALE); ctx.lineTo(vx, 1 / SCALE); }
  ctx.strokeStyle = T.seam; ctx.lineWidth = 1.2 * px; ctx.stroke();
  const lx = 31.2 / SCALE, ly = -2.5 / SCALE;                      // the accent light on the nose
  if (T.additive) ctx.globalCompositeOperation = "lighter";
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  const sx = v.X(x + Math.cos(a) * lx - Math.sin(a) * ly), sy = v.Y(y + Math.sin(a) * lx + Math.cos(a) * ly);
  blob(ctx, T.accent, sx, sy, 0.3 * v.s, 0.3 * v.s, T.additive ? 0.55 : 0.4);
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
  ctx.beginPath(); roundRectPath(ctx, lx - 1.1 / SCALE, ly - 2.6 / SCALE, 2.2 / SCALE, 5.2 / SCALE, 1.1 / SCALE);
  ctx.fillStyle = T.accentCore; ctx.fill();
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

function leg(ctx, v, [x, y, a], hx, C, T, px, lower, lite) {
  const hy = LEG_H / 2;
  place(ctx, v, x, y, a);
  const kx = KEY[0] * Math.cos(-a) - KEY[1] * Math.sin(-a);       // which side of the leg faces the key light
  const side = kx >= 0 ? 1 : -1;
  ctx.beginPath(); roundRectPath(ctx, -hx, -hy, 2 * hx, 2 * hy, hx * 0.9);
  const g = ctx.createLinearGradient(hx * side, 0, -hx * side, 0);
  g.addColorStop(0, C[0]); g.addColorStop(0.35, C[1]); g.addColorStop(1, C[2]);
  ctx.fillStyle = g; ctx.fill();
  ctx.save(); ctx.clip();
  if (lower) {                                                    // a rubber foot pad at the bottom end
    ctx.fillStyle = T.rubber; ctx.fillRect(-hx, -hy, 2 * hx, 0.11);
    ctx.fillStyle = "rgba(255,255,255,0.08)"; ctx.fillRect(-hx, -hy + 0.11, 2 * hx, px);
  }
  ctx.restore();
  const rx = RIM[0] * Math.cos(-a) - RIM[1] * Math.sin(-a), rs = rx >= 0 ? 1 : -1;
  const rg = ctx.createLinearGradient(hx * rs, 0, -hx * rs * 0.2, 0);   // rim light along one side
  rg.addColorStop(0, rgba(T.rim, T.rimA * 0.8)); rg.addColorStop(1, rgba(T.rim, 0));
  ctx.strokeStyle = rg; ctx.lineWidth = 1.8 * px; ctx.stroke();
  if (lite) return;
  const dx = -hx * 0.3 * side, y0 = hy - 0.13, y1 = -hy + (lower ? 0.2 : 0.13);
  ctx.beginPath(); ctx.moveTo(dx, y0); ctx.lineTo(dx, y1);        // a thin panel line down the leg, with its lit lip
  ctx.strokeStyle = T.seam; ctx.lineWidth = 1.2 * px; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(dx + side * 1.2 * px, y0); ctx.lineTo(dx + side * 1.2 * px, y1);
  ctx.strokeStyle = T.seamHi; ctx.lineWidth = px; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-hx * 0.8, y0 + 0.02); ctx.lineTo(hx * 0.8, y0 + 0.02);   // a collar near the top joint
  ctx.strokeStyle = T.seam; ctx.lineWidth = 1.2 * px; ctx.stroke();
}

function joint(ctx, v, [x, y], r, T, far, px) {
  place(ctx, v, x, y, 0);
  const g = ctx.createRadialGradient(-r * 0.4, r * 0.45, 0, 0, 0, r);
  g.addColorStop(0, far ? "#b9b9c4" : "#f4f4fa"); g.addColorStop(0.4, far ? "#6c6c78" : "#a6a6b3");
  g.addColorStop(0.85, far ? "#2c2c33" : "#45454f"); g.addColorStop(1, far ? "#1a1a1f" : "#26262d");
  ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.lineWidth = px; ctx.stroke();
  ctx.beginPath(); ctx.arc(0, 0, r * 0.32, 0, Math.PI * 2); ctx.fillStyle = far ? "#121216" : "#1a1a20"; ctx.fill();
  ctx.beginPath(); ctx.arc(0, 0, r * 0.62, Math.PI * 0.15, Math.PI * 0.85);   // a machined ring catching the light
  ctx.strokeStyle = far ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.3)"; ctx.lineWidth = px; ctx.stroke();
}

// ---- The humanoid upper body: a visual redesign, drawn around the real body ------------------------
/* The simulated body is the game's hull: a long, low box (HULL), 2.1 units long, with the hips under its
   middle. The humanoid is drawn rigidly in the box's own frame, so it leans, wobbles and falls exactly
   with it. It is built so the box is covered wherever the box can touch the course: the power pack
   covers its back end, the hips its middle, the forearms its underside and the fists its front end.
   The torso, shoulders and head rise above it. The arms are fixed to the body: the AI never moved them. */
const HB = -8 / SCALE, HF = 34 / SCALE, HT = 1 / SCALE;          // the box's underside, front face, front-top corner
const ARM = { shoulder: [-0.07, 1.02], elbow: [0.25, HB + 0.133], wrist: [0.86, HB + 0.133], upper: 0.12, fore: 0.133 };
const SHELL = {
  pack:   [[-1.0, HB], [-0.42, HB], [-0.42, 0.8], [-0.55, 0.92], [-0.88, 0.88], [-1.0, 0.72]],
  pelvis: [[-0.47, HB], [0.42, HB], [0.45, 0.1], [0.36, 0.36], [-0.41, 0.36], [-0.49, 0.1]],
  torso:  [[-0.34, 0.3], [0.24, 0.3], [0.31, 0.55], [0.39, 0.9], [0.33, 1.08], [0.1, 1.17], [-0.33, 1.17], [-0.46, 1.05], [-0.47, 0.7], [-0.39, 0.45]],
  neck:   [[-0.08, 1.12], [0.1, 1.12], [0.1, 1.34], [-0.08, 1.34]],
  head:   [-0.17, 1.29, 0.45, 0.43, 0.15],                         // x, y, w, h, corner radius
  visor:  [0.05, 1.44, 0.25, 0.15, 0.07],
  fist:   [0.83, HB, HF - 0.83, HT - HB, 0.035],
};
const FAR = [-0.035, 0.05];                                        // the far arm sits a little behind and above

/* Points around the shell's outline, for the honesty check (where it could touch the course). */
let shellPts = null;
export function shellSamples() {
  if (shellPts) return shellPts;
  const out = [], poly = (P, n = 12) => P.forEach((p, i) => { const q = P[(i + 1) % P.length]; for (let k = 0; k < n; k++) out.push([p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n]); });
  const box = ([x, y, w, h]) => poly([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]);
  const capsule = ([x0, y0], [x1, y1], r, [dx, dy] = [0, 0]) => {
    for (let k = 0; k <= 16; k++) { const t = k / 16, cx = x0 + (x1 - x0) * t + dx, cy = y0 + (y1 - y0) * t + dy; for (let j = 0; j < 8; j++) out.push([cx + r * Math.cos(j * Math.PI / 4), cy + r * Math.sin(j * Math.PI / 4)]); }
  };
  ["pack", "pelvis", "torso", "neck"].forEach(k => poly(SHELL[k]));
  box(SHELL.head); box(SHELL.fist); box([SHELL.fist[0] + FAR[0], SHELL.fist[1] + FAR[1], SHELL.fist[2], SHELL.fist[3]]);
  for (const off of [[0, 0], FAR]) { capsule(ARM.shoulder, ARM.elbow, ARM.upper, off); capsule(ARM.elbow, ARM.wrist, ARM.fore, off); }
  for (let j = 0; j < 16; j++) out.push([ARM.shoulder[0] + 0.2 * Math.cos(j * Math.PI / 8), ARM.shoulder[1] + 0.02 + 0.16 * Math.sin(j * Math.PI / 8)]);
  return (shellPts = out);
}
let hullPts = null;
function hullSamples() {
  if (hullPts) return hullPts;
  hullPts = [];
  HULL.forEach((p, i) => { const q = HULL[(i + 1) % HULL.length]; for (let k = 0; k < 24; k++) hullPts.push([p[0] + (q[0] - p[0]) * k / 24, p[1] + (q[1] - p[1]) * k / 24]); });
  return hullPts;
}

/* The course's surface on a fine grid (for fast checks). */
function surfaceGrid(course) {
  if (course._grid) return course._grid;
  const dx = 0.01, n = Math.ceil(course.terrain_x[course.terrain_x.length - 1] / dx) + 2, g = new Float32Array(n);
  const blocks = courseLook(course).blocks;
  for (let i = 0; i < n; i++) g[i] = surfaceAt(course, i * dx, blocks);
  const f = x => { const u = Math.max(0, Math.min(n - 1.001, x / dx)), i = Math.floor(u); return g[i] + (g[i + 1] - g[i]) * (u - i); };
  return (course._grid = f);
}

/* For every recorded frame of a run: 1 where the humanoid shell is shown, 0 where it has to give way to the
   real body, because the shell would go into the course, or the real body touches the course somewhere the
   shell does not cover. (In the recorded runs this happens only when a robot flips onto its back as it falls.)
   The shell fades over 0.2 s around those frames. */
export function shellVisibility(course, run) {
  if (run._shell) return run._shell;
  const n = run.count, A = run.frames, xy = course.xy_scale, ang = course.angle_scale, S = surfaceGrid(course);
  const shellP = shellSamples(), hullP = hullSamples(), bad = new Uint8Array(n);
  const lowest = (P, x, y, c, s) => { let m = Infinity; for (const [px, py] of P) { const X = x + px * c - py * s, Y = y + px * s + py * c; m = Math.min(m, Y - S(X)); } return m; };
  let count = 0;
  for (let f = 0; f < n; f++) {
    const x = A[f * 15] / xy, y = A[f * 15 + 1] / xy, a = A[f * 15 + 2] / ang, c = Math.cos(a), s = Math.sin(a);
    const sh = lowest(shellP, x, y, c, s);
    if (sh > 0.06) continue;                                       // well clear of everything: fine
    const hu = lowest(hullP, x, y, c, s);
    if (sh < -0.03 || (hu < 0.03 && sh - hu > 0.045)) { bad[f] = 1; count++; }
  }
  const vis = new Float32Array(n).fill(1), FADE = 10;
  if (count) {
    let last = -1e9;
    for (let f = 0; f < n; f++) { if (bad[f]) last = f; vis[f] = Math.min(1, (f - last) / FADE); }
    last = 1e9;
    for (let f = n - 1; f >= 0; f--) { if (bad[f]) last = f; vis[f] = Math.min(vis[f], (last - f) / FADE); }
    for (let f = 0; f < n; f++) { const t = vis[f]; vis[f] = t * t * (3 - 2 * t); }
  }
  vis.bad = count;
  return (run._shell = vis);
}

function toWorld([x, y, a], [lx, ly]) { const c = Math.cos(a), s = Math.sin(a); return [x + lx * c - ly * s, y + lx * s + ly * c]; }

/* One shell panel: graphite, lit from above in the world, with a chamfer, a rim light and a bounce light. */
function panel(ctx, a, C, T, px, extent, lite = false) {
  if (lite) { ctx.fillStyle = C[1]; ctx.fill(); ctx.strokeStyle = rgba(T.rim, 0.35); ctx.lineWidth = 1.4 * px; ctx.stroke(); return; }
  const ux = Math.sin(a), uy = Math.cos(a);
  let lo = Infinity, hi = -Infinity;
  for (const [x, y] of extent) { const d = x * ux + y * uy; lo = Math.min(lo, d); hi = Math.max(hi, d); }
  const g = ctx.createLinearGradient(ux * hi, uy * hi, ux * lo, uy * lo);
  g.addColorStop(0, C[0]); g.addColorStop(0.5, C[1]); g.addColorStop(1, C[2]);
  ctx.fillStyle = g; ctx.fill();
  ctx.save(); ctx.clip();
  const cg = ctx.createLinearGradient(ux * hi, uy * hi, ux * (hi - 0.35), uy * (hi - 0.35));
  cg.addColorStop(0, T.chamfer); cg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.strokeStyle = cg; ctx.lineWidth = 0.07; ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1.6 * px; ctx.stroke();   // a thin contact line between parts
  const rx = RIM[0] * Math.cos(-a) - RIM[1] * Math.sin(-a), ry = RIM[0] * Math.sin(-a) + RIM[1] * Math.cos(-a);
  let rlo = Infinity, rhi = -Infinity;
  for (const [x, y] of extent) { const d = x * rx + y * ry; rlo = Math.min(rlo, d); rhi = Math.max(rhi, d); }
  const rg = ctx.createLinearGradient(rx * rhi, ry * rhi, rx * (rhi - 0.3), ry * (rhi - 0.3));
  rg.addColorStop(0, rgba(T.rim, T.rimA)); rg.addColorStop(1, rgba(T.rim, 0));
  ctx.strokeStyle = rg; ctx.lineWidth = 2 * px; ctx.stroke();
  const bl = ctx.createLinearGradient(ux * lo, uy * lo, ux * (lo + 0.3), uy * (lo + 0.3));
  bl.addColorStop(0, rgba(T.accent, T.bounceA)); bl.addColorStop(1, rgba(T.accent, 0));
  ctx.strokeStyle = bl; ctx.lineWidth = 1.5 * px; ctx.stroke();
}
const rectPts = ([x, y, w, h]) => [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];

function seams(ctx, lines, T, px) {
  ctx.beginPath(); for (const [x0, y0, x1, y1] of lines) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
  ctx.strokeStyle = T.seam; ctx.lineWidth = 1.2 * px; ctx.stroke();
  ctx.beginPath(); for (const [x0, y0, x1, y1] of lines) { ctx.moveTo(x0, y0 - 1.2 * px); ctx.lineTo(x1, y1 - 1.2 * px); }
  ctx.strokeStyle = T.seamHi; ctx.lineWidth = px; ctx.stroke();
}

function arm(ctx, v, H, T, px, C, off, lite, far) {
  const [hx, hy, a] = H, sh = [ARM.shoulder[0] + off[0], ARM.shoulder[1] + off[1]];
  const el = [ARM.elbow[0] + off[0], ARM.elbow[1] + off[1]], wr = [ARM.wrist[0] + off[0], ARM.wrist[1] + off[1]];
  const limb = (p0, p1, r) => {                                    // a shaded capsule, like the legs
    place(ctx, v, hx, hy, a);
    const th = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]), len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]), phi = a + th;
    ctx.translate(p0[0], p0[1]); ctx.rotate(th);
    const ky = -KEY[0] * Math.sin(phi) + KEY[1] * Math.cos(phi), side = ky >= 0 ? 1 : -1;
    ctx.beginPath(); roundRectPath(ctx, -r * 0.6, -r, len + r * 1.2, 2 * r, r * 0.9);
    const g = ctx.createLinearGradient(0, r * side, 0, -r * side);
    g.addColorStop(0, C[0]); g.addColorStop(0.35, C[1]); g.addColorStop(1, C[2]);
    ctx.fillStyle = g; ctx.fill();
    const ry = -RIM[0] * Math.sin(phi) + RIM[1] * Math.cos(phi), rs = ry >= 0 ? 1 : -1;
    const rg = ctx.createLinearGradient(0, r * rs, 0, -r * rs * 0.2);
    rg.addColorStop(0, rgba(T.rim, T.rimA * (far ? 0.4 : 0.8))); rg.addColorStop(1, rgba(T.rim, 0));
    ctx.strokeStyle = rg; ctx.lineWidth = 1.8 * px; ctx.stroke();
    if (!lite) { ctx.beginPath(); ctx.moveTo(r * 0.5, -r * 0.3 * side); ctx.lineTo(len - r * 0.5, -r * 0.3 * side); ctx.strokeStyle = T.seam; ctx.lineWidth = 1.1 * px; ctx.stroke(); }
  };
  limb(sh, el, ARM.upper);
  limb(el, wr, ARM.fore);
  place(ctx, v, hx, hy, a);                                        // the fist covers the box's front end
  const F = [SHELL.fist[0] + off[0], SHELL.fist[1] + off[1], SHELL.fist[2], SHELL.fist[3]];
  ctx.beginPath(); roundRectPath(ctx, F[0], F[1], F[2], F[3], SHELL.fist[4]);
  panel(ctx, a, far ? T.legFar : T.fist, T, px, rectPts(F), lite);
  if (!lite) seams(ctx, [[F[0] + F[2] * 0.62, F[1] + 0.05, F[0] + F[2] * 0.62, F[1] + F[3] - 0.05], [F[0] + 0.03, F[1] + F[3] * 0.55, F[0] + F[2] * 0.55, F[1] + F[3] * 0.55]], T, px);
  joint(ctx, v, toWorld(H, el), 0.125, T, far, px);
  place(ctx, v, hx, hy, a);                                        // shoulder cap, then the shoulder joint
  ctx.beginPath(); ctx.ellipse(sh[0], sh[1] + 0.02, 0.2, 0.16, 0, 0, Math.PI * 2);
  panel(ctx, a, far ? T.legFar : T.body, T, px, [[sh[0] - 0.2, sh[1] - 0.14], [sh[0] + 0.2, sh[1] + 0.18]], lite);
  joint(ctx, v, toWorld(H, sh), 0.11, T, far, px);
}

function shellBack(ctx, v, H, T, px, lite) { arm(ctx, v, H, T, px, T.legFar, FAR, lite, true); }
function shellFront(ctx, v, H, T, px, lite) { arm(ctx, v, H, T, px, T.legNear, [0, 0], lite, false); }

function shellBody(ctx, v, H, T, px, lite) {
  const [hx, hy, a] = H;
  place(ctx, v, hx, hy, a);
  ctx.beginPath(); roundPoly(ctx, SHELL.pack, 0.05); panel(ctx, a, T.pack, T, px, SHELL.pack, lite);
  if (!lite) {
    seams(ctx, [[-0.97, 0.12, -0.45, 0.12], [-0.97, 0.46, -0.45, 0.46]], T, px);
    ctx.beginPath(); for (let k = 0; k < 3; k++) { const y = 0.58 + k * 0.06; ctx.moveTo(-0.9, y); ctx.lineTo(-0.6, y); }
    ctx.strokeStyle = T.seam; ctx.lineWidth = 1.4 * px; ctx.stroke();
  }
  ctx.beginPath(); roundPoly(ctx, SHELL.pelvis, 0.05); panel(ctx, a, T.pelvis, T, px, SHELL.pelvis, lite);
  if (!lite) seams(ctx, [[-0.46, 0.15, 0.43, 0.15]], T, px);
  ctx.beginPath(); roundPoly(ctx, SHELL.neck, 0.03); panel(ctx, a, T.pelvis, T, px, SHELL.neck, lite);
  ctx.beginPath(); roundPoly(ctx, SHELL.torso, 0.09); panel(ctx, a, T.torso, T, px, SHELL.torso, lite);
  if (!lite) seams(ctx, [[-0.42, 0.5, 0.34, 0.5], [-0.5, 0.66, 0.38, 0.66], [-0.12, 0.68, -0.14, 1.15]], T, px);
  const [x, y, w, h, r] = SHELL.head;                              // the head: a helmet with a glass visor
  ctx.beginPath(); roundRectPath(ctx, x, y, w, h, r); panel(ctx, a, T.torso, T, px, rectPts([x, y, w, h]), lite);
  const [vx, vy, vw, vh, vr] = SHELL.visor;
  ctx.beginPath(); roundRectPath(ctx, vx, vy, vw, vh, vr);
  const vg = ctx.createLinearGradient(0, vy + vh, 0, vy);
  vg.addColorStop(0, "#05050a"); vg.addColorStop(1, "#24242f");
  ctx.fillStyle = vg; ctx.fill();
  ctx.strokeStyle = rgba(T.rim, 0.35); ctx.lineWidth = px; ctx.stroke();
  const lx = vx + vw * 0.62, ly = vy + vh * 0.48;                  // the accent light behind the glass
  if (T.additive) ctx.globalCompositeOperation = "lighter";
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  const [wx, wy] = toWorld(H, [lx, ly]);
  blob(ctx, T.accent, v.X(wx), v.Y(wy), 0.24 * v.s, 0.2 * v.s, T.additive ? 0.6 : 0.35);
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
  ctx.beginPath(); roundRectPath(ctx, vx + vw * 0.28, ly - 0.014, vw * 0.62, 0.028, 0.014);
  ctx.fillStyle = T.accentCore; ctx.fill();
  if (!lite) { ctx.beginPath(); ctx.moveTo(vx + 0.03, vy + vh - 0.03); ctx.lineTo(vx + vw - 0.05, vy + vh - 0.03); ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = px; ctx.stroke(); }
}

// ---- Product shot: one recorded pose, levelled, in a studio ---------------------------------------
/* Rotates a recorded pose as a whole so the body is level and the feet stand on y = 0 at x = 0
   (the joints' angles relative to each other are untouched). */
export function levelPose(pose) {
  const hx = pose[0], hy = pose[1], ha = pose[2], c = Math.cos(-ha), s = Math.sin(-ha);
  const out = pose.slice();
  for (let k = 0; k < 5; k++) {
    const dx = pose[k * 3] - hx, dy = pose[k * 3 + 1] - hy;
    out[k * 3] = c * dx - s * dy; out[k * 3 + 1] = s * dx + c * dy; out[k * 3 + 2] = pose[k * 3 + 2] - ha;
  }
  let floor = Infinity;                                          // the lowest corner of either foot
  for (const k of [2, 4]) {
    const [x, y, a] = part(out, k), hx = 0.8 * LEG_W / 2, hy = LEG_H / 2;
    for (const [cx, cy] of [[-hx, -hy], [hx, -hy]]) floor = Math.min(floor, y + Math.sin(a) * cx + Math.cos(a) * cy);
  }
  for (let k = 0; k < 5; k++) out[k * 3 + 1] -= floor;
  return out;
}

/* The robot alone on a transparent canvas, standing on a floor line at `floor` (fraction of the height):
   a soft contact shadow, an optional reflection, the 10 lidar rays drawn on as `lidar` goes 0 -> 1, and
   the 4 motors (both hips, both knees) lit as `motors` goes 0 -> 1. One recorded pose, levelled. */
export function drawFigure(ctx, W, H, pose, T, { scale = 1, cx = 0.46, floor = 0.8, lidar = 0, motors = 0, reflect = true, lite = false } = {}) {
  const P = levelPose(pose);
  const s = Math.min(W / 7.2, H / 5.6) * scale, floorY = H * floor;
  const v = { W, H, s, X: x => W * cx + x * s, Y: y => floorY - y * s };
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (reflect && !lite) {                                         // the reflection, fading into the floor
    const L = layerFor(ctx);
    L.setTransform(1, 0, 0, 1, 0, 0); L.clearRect(0, 0, W, H);
    drawRobot(L, { ...v, Y: y => floorY + y * s, flip: true }, P, T, { lite: true });
    L.setTransform(1, 0, 0, 1, 0, 0); L.globalCompositeOperation = "destination-in";
    const m = L.createLinearGradient(0, floorY, 0, floorY + 1.1 * s);
    m.addColorStop(0, "rgba(0,0,0,0.22)"); m.addColorStop(1, "rgba(0,0,0,0)");
    L.fillStyle = m; L.fillRect(0, floorY, W, H - floorY); L.clearRect(0, 0, W, floorY);
    L.globalCompositeOperation = "source-over";
    ctx.drawImage(L.canvas, 0, 0);
  }
  blob(ctx, T.shadow, v.X(P[0]), floorY, 1.5 * s, 0.11 * s, T.shadowA * 0.9);
  for (const k of [2, 4]) { const [fx] = along(part(P, k), -LEG_H / 2); blob(ctx, T.shadow, v.X(fx), floorY, 0.32 * s, 0.05 * s, T.shadowA); }
  if (lidar > 0) {                                                // the 10 lidar rays, at the game's own angles, from the body's centre
    const cx0 = v.X(P[0]), cy0 = v.Y(P[1]), range = 160 / SCALE * s;
    ctx.lineCap = "round";
    for (let i = 0; i < 10; i++) {
      const k = Math.min(1, Math.max(0, lidar * 12 - i * 1.1));
      if (k <= 0) continue;
      const ang = 1.5 * i / 10, dx = Math.sin(ang), dy = -Math.cos(ang);
      const len = range * Math.min(1, P[1] / -dy / (160 / SCALE)) * k;
      const lg = ctx.createLinearGradient(cx0, cy0, cx0 + dx * len, cy0 - dy * len);
      lg.addColorStop(0, rgba(T.accent, 0.85)); lg.addColorStop(1, rgba(T.accent, 0.1));
      ctx.strokeStyle = lg; ctx.lineWidth = Math.max(1, s / 110);
      ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0 + dx * len, cy0 - dy * len); ctx.stroke();
      if (k >= 1 && len < range * 0.999) blob(ctx, T.accent, cx0 + dx * len, cy0 - dy * len, 0.07 * s, 0.035 * s, 0.9);
    }
  }
  drawRobot(ctx, v, P, T, { lite });
  if (motors > 0) {                                               // the 4 motors: both hips (one spot), both knees
    const spots = [along(part(P, 1), LEG_H / 2), along(part(P, 1), -LEG_H / 2), along(part(P, 3), -LEG_H / 2)];
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (T.additive) ctx.globalCompositeOperation = "lighter";
    spots.forEach(([x, y], i) => {
      const k = Math.min(1, Math.max(0, motors * 3.2 - i * 0.9));
      blob(ctx, T.accent, v.X(x), v.Y(y), 0.42 * s * (0.8 + 0.2 * k), 0.42 * s * (0.8 + 0.2 * k), 0.75 * k);
    });
    ctx.globalCompositeOperation = "source-over";
    spots.forEach(([x, y], i) => {
      const k = Math.min(1, Math.max(0, motors * 3.2 - i * 0.9));
      if (k <= 0) return;
      ctx.beginPath(); ctx.arc(v.X(x), v.Y(y), 0.2 * s, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * k);
      ctx.strokeStyle = rgba(T.accent, 0.9); ctx.lineWidth = Math.max(1.5, s / 70); ctx.stroke();
    });
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return v;
}

/* The same robot in a studio: background, floor and figure (used for stills). */
export function drawStudio(ctx, W, H, pose, T, { lidar = 0, scale = 1 } = {}) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const floorY = H * 0.8;
  const bg = ctx.createRadialGradient(W / 2, H * 0.35, 0, W / 2, H * 0.45, Math.max(W, H) * 0.75);
  bg.addColorStop(0, T.sky[0]); bg.addColorStop(1, T.sky[1]);
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
  const s = Math.min(W / 7.2, H / 5.6) * scale;
  if (T.additive) ctx.globalCompositeOperation = "lighter";
  blob(ctx, T.glow, W / 2, floorY - 0.9 * s, 2.6 * s, 1.6 * s, T.glowA * 1.4);
  ctx.globalCompositeOperation = "source-over";
  const fl = ctx.createLinearGradient(0, floorY, 0, H);
  fl.addColorStop(0, rgba(hexRgb(T.ground[0]), 0.9)); fl.addColorStop(1, rgba(hexRgb(T.sky[1]), 1));
  ctx.fillStyle = fl; ctx.fillRect(0, floorY, W, H - floorY);
  ctx.fillStyle = rgba(T.rim, 0.22); ctx.fillRect(0, floorY, W, 1);
  drawFigure(ctx, W, H, pose, T, { scale, cx: 0.42, floor: 0.8, lidar });
}
