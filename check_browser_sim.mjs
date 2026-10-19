/* check_browser_sim.mjs — checks the browser simulator experiment (browser_sim/) against Python.

     node check_browser_sim.mjs trace FILE.json   one run, step by step, against a trace from
                                                  `python export_web_data.py trace ...`
     node check_browser_sim.mjs final [--workers 4] [--out FILE.json]
                                                  all 600 hardcore test runs (A and B, 3 seeds,
                                                  courses 10000-10099) against runs/final/episodes.csv

   "trace" checks, in order: the course (terrain, obstacles, clouds), the brain (Python's
   observation in -> the same move out?), the physics with Python's own moves (open loop), then the
   whole run with the browser's own moves (closed loop). "final" reports how often the browser run
   ends the same way as the recorded test run. */

import fs from "node:fs";
import os from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { Walker } from "./browser_sim/walker.js";
import { Brain } from "./browser_sim/policy.js";

const brainsJson = JSON.parse(fs.readFileSync(new URL("./browser_sim/brains.json", import.meta.url)));
const brainFor = (model, seed) => new Brain(brainsJson.agents[model].seeds[String(seed)]);

function runEpisode(brain, courseSeed, hardcore = true) {
  const w = new Walker(courseSeed, hardcore);
  while (!w.done) w.step(brain.act(w.obs));
  return { score: w.score, outcome: w.outcome, distance: w.distance, steps: w.steps, end_x: w.endX };
}

const maxAbsDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

function checkTrace(file) {
  const t = JSON.parse(fs.readFileSync(file));
  const brain = brainFor(t.model, t.seed);
  const w = new Walker(t.course_seed, true);
  const terrain = Math.max(maxAbsDiff(w.terrainX, t.terrain_x), maxAbsDiff(w.terrainY, t.terrain_y));
  const jsObst = w.obstacles.flatMap(o => o.polys);
  const pyXs = t.obstacles.map(p => Math.min(...p.map(v => v[0]))).sort((a, b) => a - b);
  const jsXs = jsObst.map(p => Math.min(...p.map(v => v[0]))).sort((a, b) => a - b);
  const clouds = maxAbsDiff(w.clouds.flat(2), t.clouds.flat(2));
  console.log(`course ${t.course_seed}: terrain max diff ${terrain.toExponential(2)} | obstacle blocks JS ${jsObst.length} / Python ${t.obstacles.length}` +
              ` (left edges max diff ${jsXs.length === pyXs.length ? maxAbsDiff(jsXs, pyXs).toExponential(2) : "n/a"}) | clouds max diff ${clouds.toExponential(2)}`);
  console.log(`after reset: observation max diff ${maxAbsDiff(Array.from(w.obs), t.reset_obs).toExponential(2)}`);

  // The brain alone: Python's observation in, compare the move out.
  let worstAct = 0, prevObs = t.reset_obs;
  for (const s of t.steps) { worstAct = Math.max(worstAct, maxAbsDiff(Array.from(brain.act(prevObs)), s.action)); prevObs = s.obs; }
  console.log(`brain: largest move difference over ${t.steps.length} steps (same inputs) ${worstAct.toExponential(2)}`);

  // Physics alone: Python's moves in (open loop).
  const open = new Walker(t.course_seed, true);
  let firstDrift = null, n = 0;
  for (const s of t.steps) {
    open.step(s.action); n++;
    const d = Math.max(maxAbsDiff(open.pose()[0], s.hull), ...open.pose().slice(1).map((p, i) => maxAbsDiff(p, s.legs[i])));
    if (firstDrift === null && d > 0.01) firstDrift = { step: n, d };
    if ([1, 10, 50, 100, 250, 500, 1000, 1500, 2000].includes(n))
      console.log(`  open loop step ${String(n).padStart(4)}: pose max diff ${d.toExponential(2)}, hull x JS ${open.hull.getPosition().x.toFixed(4)} / Python ${s.hull[0].toFixed(4)}`);
    if (open.done) break;
  }
  console.log(`open loop: first step with a part more than 1 cm off: ${firstDrift ? firstDrift.step : "none"}`);

  const r = runEpisode(brain, t.course_seed);
  const p = t.result;
  console.log(`closed loop (browser's own moves): ${r.outcome} ${r.score.toFixed(4)} pts, ${r.steps} steps, distance ${r.distance.toFixed(2)}%` +
              `  | Python: ${p.outcome} ${p.score.toFixed(4)} pts, ${p.steps} steps, distance ${p.distance.toFixed(2)}%`);
}

function readFinal() {
  const lines = fs.readFileSync(new URL("./runs/final/episodes.csv", import.meta.url), "utf8").trim().split("\n");
  const head = lines[0].split(",");
  return lines.slice(1).map(l => Object.fromEntries(l.split(",").map((v, i) => [head[i], v])))
    .filter(r => r.course === "hardcore" && (r.model === "a_100m" || r.model === "b_100m"))
    .map(r => ({ model: r.model, seed: +r.train_seed, course_seed: +r.course_seed, score: +r.score,
                 outcome: r.outcome, distance: +r.distance, steps: +r.steps }));
}

async function checkFinal(nWorkers, outFile) {
  const recs = readFinal();
  const t0 = Date.now();
  const chunks = Array.from({ length: nWorkers }, (_, k) => recs.filter((_, i) => i % nWorkers === k));
  const results = (await Promise.all(chunks.map(chunk => new Promise((res, rej) => {
    const wk = new Worker(new URL(import.meta.url), { workerData: chunk });
    wk.on("message", res); wk.on("error", rej);
  })))).flat();
  const secs = (Date.now() - t0) / 1000;
  const key = r => `${r.model}/${r.seed}/${r.course_seed}`;
  const js = new Map(results.map(r => [key(r), r]));
  let same = 0, identical = 0;
  const confusion = {};
  const rates = {};
  for (const r of recs) {
    const b = js.get(key(r));
    if (b.outcome === r.outcome) same++;
    if (b.outcome === r.outcome && b.steps === r.steps && Math.abs(b.score - r.score) < 0.01) identical++;
    const c = `${r.outcome} -> ${b.outcome}`;
    confusion[c] = (confusion[c] || 0) + 1;
    for (const [src, o] of [["python", r.outcome], ["browser", b.outcome]]) {
      const k = `${r.model} ${src}`;
      rates[k] = rates[k] || { n: 0, finished: 0 };
      rates[k].n++; if (o === "finished") rates[k].finished++;
    }
  }
  const n = recs.length;
  console.log(`${n} runs in ${secs.toFixed(0)} s on ${nWorkers} workers`);
  console.log(`same outcome as the recorded test run: ${same}/${n} = ${(100 * same / n).toFixed(1)}%`);
  console.log(`same outcome, same number of steps and the same score (within 0.01): ${identical}/${n} = ${(100 * identical / n).toFixed(1)}%`);
  for (const [k, v] of Object.entries(rates).sort()) console.log(`  finished, ${k}: ${v.finished}/${v.n} = ${(100 * v.finished / v.n).toFixed(1)}%`);
  console.log("  recorded -> browser:", Object.entries(confusion).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}: ${v}`).join(" | "));
  if (outFile) fs.writeFileSync(outFile, JSON.stringify({ recs, browser: results }));
}

if (!isMainThread) {
  const brains = {};
  parentPort.postMessage(workerData.map(r => {
    const k = `${r.model}/${r.seed}`;
    brains[k] = brains[k] || brainFor(r.model, r.seed);
    return { model: r.model, seed: r.seed, course_seed: r.course_seed, ...runEpisode(brains[k], r.course_seed) };
  }));
} else {
  const [cmd, arg] = process.argv.slice(2);
  if (cmd === "trace") checkTrace(arg);
  else if (cmd === "final") {
    const wi = process.argv.indexOf("--workers"), oi = process.argv.indexOf("--out");
    const nWorkers = Math.min(4, wi > 0 ? +process.argv[wi + 1] : 4, os.cpus().length);   // the Mac's 4-core cap
    await checkFinal(nWorkers, oi > 0 ? process.argv[oi + 1] : null);
  } else console.log("usage: node check_browser_sim.mjs trace FILE.json | final [--workers 4] [--out FILE.json]");
}
