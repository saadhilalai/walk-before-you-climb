/* robot.js — the walker as a product shot: the real robot geometry (the game's body and leg
   shapes), posed from a recorded run, lit like a hero object: gradient shading, a rim light, a
   contact shadow and a floor reflection. The 10 lidar rays use the game's own angles. */

const SCALE = 30;
const HULL = [[-30, 9], [6, 9], [34, 1], [34, -8], [-30, -8]].map(([x, y]) => [x / SCALE, y / SCALE]);
const LEG_W = 8 / SCALE, LEG_H = 34 / SCALE;
const box = (hx, hy) => [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]];
const SHAPES = [HULL, box(LEG_W / 2, LEG_H / 2), box(0.8 * LEG_W / 2, LEG_H / 2), box(LEG_W / 2, LEG_H / 2), box(0.8 * LEG_W / 2, LEG_H / 2)];
const LIDAR_RANGE = 160 / SCALE;
let uid = 0;

/* pose: 15 numbers [x, y, angle] for hull, leg(-1), lower(-1), leg(+1), lower(+1).
   Returns { svg, joints, lidar } with the body levelled and standing on y = 0. */
export function robotSVG(pose, { variant = "lit", lidar = true } = {}) {
  const id = `r${++uid}`;
  const hx = pose[0], hy = pose[1], ha = pose[2];
  const c = Math.cos(-ha), s = Math.sin(-ha);
  const level = (x, y) => { const dx = x - hx, dy = y - hy; return [c * dx - s * dy, s * dx + c * dy]; };   // body upright
  const parts = SHAPES.map((shape, k) => {
    const px = pose[k * 3], py = pose[k * 3 + 1], a = pose[k * 3 + 2], ca = Math.cos(a), sa = Math.sin(a);
    return shape.map(([vx, vy]) => level(px + ca * vx - sa * vy, py + sa * vx + ca * vy));
  });
  const floor = Math.min(...parts.flat().map(p => p[1]));                 // feet on the floor
  const P = parts.map(poly => poly.map(([x, y]) => [x, y - floor]));
  const all = P.flat();
  const minX = Math.min(...all.map(p => p[0])) - 0.4, maxX = Math.max(...all.map(p => p[0])) + 0.4 + (lidar ? 2.6 : 0);
  const maxY = Math.max(...all.map(p => p[1])) + 0.35;
  const W = maxX - minX, depth = maxY * 0.55;                              // room for the reflection
  const pts = poly => poly.map(([x, y]) => `${(x - minX).toFixed(3)},${(maxY - y).toFixed(3)}`).join(" ");
  const hull = P[0], hc = hull.reduce((a, p) => [a[0] + p[0] / hull.length, a[1] + p[1] / hull.length], [0, 0]);
  const joint = (k, end) => { const poly = P[k], [p0, p1, p2, p3] = poly; return end === "top" ? [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2] : [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]; };
  const joints = [joint(1, "top"), joint(2, "top"), joint(3, "top"), joint(4, "top")].map(([x, y]) => [x - minX, maxY - y]);

  // Lidar: 10 rays from the body centre, the game's angles (1.5 * i / 10 rad from straight down), cut at the floor.
  const rays = [];
  if (lidar) for (let i = 0; i < 10; i++) {
    const a = 1.5 * i / 10, dx = Math.sin(a) * LIDAR_RANGE, dy = -Math.cos(a) * LIDAR_RANGE;
    const f = Math.min(1, hc[1] / -dy);
    rays.push([hc[0] - minX, maxY - hc[1], hc[0] + dx * f - minX, maxY - (hc[1] + dy * f)]);
  }

  const lit = variant === "lit";
  const legFill = k => (k <= 2 ? `url(#${id}-legback)` : `url(#${id}-legfront)`);
  const body = `
    <polygon points="${pts(P[1])}" fill="${legFill(1)}" stroke="url(#${id}-rim)" stroke-width="0.022"/>
    <polygon points="${pts(P[2])}" fill="${legFill(2)}" stroke="url(#${id}-rim)" stroke-width="0.022"/>
    <polygon points="${pts(P[3])}" fill="${legFill(3)}" stroke="url(#${id}-rim)" stroke-width="0.022"/>
    <polygon points="${pts(P[4])}" fill="${legFill(4)}" stroke="url(#${id}-rim)" stroke-width="0.022"/>
    <polygon points="${pts(hull)}" fill="url(#${id}-hull)" stroke="url(#${id}-rim)" stroke-width="0.028"/>
    <polygon points="${pts(hull)}" fill="url(#${id}-sheen)"/>`;
  const svg = `
<svg viewBox="0 0 ${W.toFixed(3)} ${(maxY + depth).toFixed(3)}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="${id}-hull" x1="0" y1="0" x2="0" y2="1">
      ${lit ? `<stop offset="0" stop-color="#b993ff"/><stop offset="0.45" stop-color="#7f3df0"/><stop offset="1" stop-color="#3d1694"/>`
            : `<stop offset="0" stop-color="#1b1a24"/><stop offset="1" stop-color="#0a0a0f"/>`}
    </linearGradient>
    <linearGradient id="${id}-sheen" x1="0" y1="0" x2="1" y2="0.4">
      <stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="0.52" stop-color="#fff" stop-opacity="${lit ? 0.28 : 0.06}"/><stop offset="0.6" stop-color="#fff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="${id}-legback" x1="0" y1="0" x2="0" y2="1">
      ${lit ? `<stop offset="0" stop-color="#c47aaf"/><stop offset="1" stop-color="#6e2f5c"/>` : `<stop offset="0" stop-color="#16151d"/><stop offset="1" stop-color="#09090d"/>`}
    </linearGradient>
    <linearGradient id="${id}-legfront" x1="0" y1="0" x2="0" y2="1">
      ${lit ? `<stop offset="0" stop-color="#9c4a86"/><stop offset="1" stop-color="#43153a"/>` : `<stop offset="0" stop-color="#1a1922"/><stop offset="1" stop-color="#0b0b10"/>`}
    </linearGradient>
    <linearGradient id="${id}-rim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#efeaff" stop-opacity="${lit ? 0.85 : 0.75}"/><stop offset="0.5" stop-color="#c6c0ff" stop-opacity="${lit ? 0.25 : 0.35}"/><stop offset="1" stop-color="#c6c0ff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="${id}-ray" gradientUnits="userSpaceOnUse" x1="${(hc[0] - minX).toFixed(2)}" y1="0" x2="${(hc[0] - minX + LIDAR_RANGE).toFixed(2)}" y2="0">
      <stop offset="0" stop-color="#e9e5ff" stop-opacity="0.9"/><stop offset="1" stop-color="#c6c0ff" stop-opacity="0.15"/>
    </linearGradient>
    <radialGradient id="${id}-shadow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#000" stop-opacity="0.75"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
    <linearGradient id="${id}-fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.32"/><stop offset="0.7" stop-color="#fff" stop-opacity="0"/></linearGradient>
    <mask id="${id}-mask" maskUnits="userSpaceOnUse" x="0" y="${maxY.toFixed(3)}" width="${W.toFixed(3)}" height="${depth.toFixed(3)}">
      <rect x="0" y="${maxY.toFixed(3)}" width="${W.toFixed(3)}" height="${depth.toFixed(3)}" fill="url(#${id}-fade)"/>
    </mask>
  </defs>
  <ellipse cx="${(hc[0] - minX).toFixed(3)}" cy="${maxY.toFixed(3)}" rx="${(W * 0.32).toFixed(3)}" ry="0.16" fill="url(#${id}-shadow)"/>
  <g mask="url(#${id}-mask)"><g transform="translate(0 ${(2 * maxY).toFixed(3)}) scale(1 -1)">${body}</g></g>
  <g class="rays">${rays.map(([x1, y1, x2, y2]) => `<line x1="${x1.toFixed(3)}" y1="${y1.toFixed(3)}" x2="${x2.toFixed(3)}" y2="${y2.toFixed(3)}" stroke="url(#${id}-ray)" stroke-width="0.025" stroke-linecap="round" pathLength="1"/>`).join("")}</g>
  <g class="body">${body}</g>
  ${lit ? `<g class="joints">${joints.map(([x, y]) => `<circle class="halo" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.11" fill="#c6c0ff" fill-opacity="0.28"/><circle class="core" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.042" fill="#fff"/>`).join("")}</g>` : ""}
</svg>`;
  return { svg, joints, rays: rays.length };
}
