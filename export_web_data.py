"""
export_web_data.py — exports what the website needs from the trained agents and the saved results.

    python export_web_data.py replays                 # web/data/courses/*.bin + stories/*.bin + runs.json
    python export_web_data.py site                    # web/data/site.json: the numbers and charts on the page
    python export_web_data.py gallery                 # web/data/gallery.json: the obstacle gallery's clips (after replays)
    python export_web_data.py brains                  # browser_sim/brains.json: A and B's networks (3 seeds each)
    python export_web_data.py trace MODEL SEED COURSE OUT.json [--steps N]
                                                     # one run step by step, to check the browser simulator

replays: every hardcore test run of A and B (100 hidden test courses x 3 training seeds) and the
three "collapse -> wobble -> walk" runs from the videos are played again in Python, exactly like
make_videos.py does. Each replay must match its recorded test run (runs/final/episodes.csv)
exactly: the same score to the last digit, outcome, number of steps and distance. If one doesn't,
nothing is written. The site then draws the robot from these recorded poses (it doesn't simulate).

brains/trace: for the browser simulator experiment (browser_sim/, check_browser_sim.mjs). It matched
the courses and the brains exactly, but its physics drifted from Python's: only 50.8% of the 600
test runs ended the same way, so the site replays recorded runs instead.
"""

import argparse
import base64
import csv
import json
import pickle
import struct
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config

config.limit_cpu_threads(1)

import Box2D  # noqa: E402

from evaluate import load_agent  # noqa: E402
from walker_env import FINISH_X, START_X, STEP, course_obstacles, make_env  # noqa: E402

WEB_DATA = Path("web") / "data"
HARDCORE_AGENTS = {"a_100m": "A: obstacles from the start", "b_100m": "B: learned to walk first"}
TEST_COURSES = range(config.TEST_SEED, config.TEST_SEED + config.N_TEST_EPISODES)
# The three runs of the "collapse -> wobble -> walk" video clip (videos/final/manifest.json).
STORIES = {"falls": ("untrained", 42, 10000), "wobbles": ("sac_229k", 42, 10000), "walks": ("sac_524k", 42, 10001)}
# Poses are stored as int16: x and y in 1/300 of a world unit, angles in 1/10000 of a radian.
XY_SCALE, ANGLE_SCALE = 300, 10000


def recorded_runs():
    return {(r["model"], int(r["train_seed"]), r["course"], int(r["course_seed"])): r
            for r in csv.DictReader(open(Path("runs") / "final" / "episodes.csv"))}


def course_layout(game):
    """The course as drawing data: ground profile, obstacle blocks, clouds, obstacle list."""
    blocks = []
    for body in game.terrain:
        shape = body.fixtures[0].shape
        if isinstance(shape, Box2D.b2PolygonShape):
            blocks.append([[round(float(v[0]), 5), round(float(v[1]), 5)] for v in shape.vertices])
    blocks.sort(key=lambda p: min(v[0] for v in p))
    return dict(terrain_x=[round(float(v), 5) for v in game.terrain_x],
                terrain_y=[round(float(v), 5) for v in game.terrain_y],
                blocks=blocks,
                clouds=[[[round(float(a), 4), round(float(b), 4)] for a, b in poly] for poly, _, _ in game.cloud_poly],
                obstacles=[dict(kind=o["kind"], x_start=round(o["x_start"], 4), x_end=round(o["x_end"], 4))
                           for o in (course_obstacles(game) if game.hardcore else [])])


def pose(game):
    out = []
    for body in [game.hull] + game.legs:
        a = (float(body.angle) + np.pi) % (2 * np.pi) - np.pi          # wrapped to [-pi, pi) for int16
        out += [float(body.position.x) * XY_SCALE, float(body.position.y) * XY_SCALE, a * ANGLE_SCALE]
    return out


def replay(agent, course, course_seed, rec):
    """Play one recorded run again, keep every frame's pose, and check it against the record."""
    env = make_env(course)
    obs, _ = env.reset(seed=course_seed)
    game = env.unwrapped
    layout = course_layout(game)
    frames = [pose(game)]
    score, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        score += float(reward)
        steps += 1
        done = terminated or truncated
        frames.append(pose(game))
    env.close()
    got = dict(score=score, outcome=info["outcome"], steps=steps, distance=info["distance"])
    want = dict(score=float(rec["score"]), outcome=rec["outcome"], steps=int(rec["steps"]), distance=float(rec["distance"]))
    if got != want:
        raise RuntimeError(f"replay of {rec['model']} seed {rec['train_seed']} course {course_seed} does not match "
                           f"the recorded test run: {got} vs {want}")
    data = np.rint(np.array(frames)).astype("<i2")
    assert np.all(np.abs(np.array(frames)) < 32767), "pose out of int16 range"
    summary = dict(model=rec["model"], seed=int(rec["train_seed"]), score=score, outcome=info["outcome"], steps=steps,
                   distance=info["distance"], end_x=info["end_x"], fell_at=info["fell_at"])
    return layout, summary, data


def pack(path, header, arrays):
    """One file: uint32 header length, the JSON header, padding to 4 bytes, then the int16 pose arrays."""
    offset = 0
    for entry, arr in zip(header["runs"], arrays):
        entry["offset"], entry["frames"] = offset, int(arr.shape[0])
        offset += arr.nbytes
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-(4 + len(head)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(head)) + head)
        for arr in arrays:
            f.write(arr.tobytes())


_agents = {}


def agent_for(model, seed):
    if (model, seed) not in _agents:
        _agents[(model, seed)] = load_agent(Path("models") / model / f"seed{seed}")
    return _agents[(model, seed)]


def export_course(course_seed):
    recs = recorded_runs()
    layout, runs, arrays = None, [], []
    for model in HARDCORE_AGENTS:
        for seed in config.TRAIN_SEEDS:
            lay, summary, data = replay(agent_for(model, seed), "hardcore", course_seed,
                                        recs[(model, seed, "hardcore", course_seed)])
            assert layout is None or lay == layout, "a course must be identical for every agent"
            layout = lay
            runs.append(summary)
            arrays.append(data)
    pack(WEB_DATA / "courses" / f"{course_seed}.bin", dict(course_seed=course_seed, hardcore=True, **layout,
         xy_scale=XY_SCALE, angle_scale=ANGLE_SCALE, runs=runs), arrays)
    return course_seed, layout["obstacles"], runs


def export_story(name):
    model, seed, course_seed = STORIES[name]
    rec = recorded_runs()[(model, seed, "normal", course_seed)]
    layout, summary, data = replay(agent_for(model, seed), "normal", course_seed, rec)
    pack(WEB_DATA / "stories" / f"{name}.bin", dict(name=name, course_seed=course_seed, hardcore=False, **layout,
         xy_scale=XY_SCALE, angle_scale=ANGLE_SCALE, runs=[summary]), [data])
    return name, summary


def replays(workers=config.MAX_CORES):
    (WEB_DATA / "courses").mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "stories").mkdir(parents=True, exist_ok=True)
    with Pool(min(workers, config.MAX_CORES)) as pool:
        stories = pool.map(export_story, list(STORIES))
        courses = pool.map(export_course, list(TEST_COURSES), chunksize=5)
    index = dict(note="Every hardcore test run of A and B, replayed and checked against runs/final/episodes.csv",
                 finish_x=FINISH_X, start_x=START_X, tile=STEP, stories={n: s for n, s in stories},
                 courses=[dict(course_seed=c, obstacles=obs, runs=[{k: r[k] for k in ("model", "seed", "outcome",
                          "score", "steps", "distance", "fell_at")} for r in runs]) for c, obs, runs in courses])
    (WEB_DATA / "runs.json").write_text(json.dumps(index, separators=(",", ":")))
    total = sum(p.stat().st_size for p in (WEB_DATA / "courses").glob("*.bin"))
    print(f"{len(courses)} courses x 6 runs and {len(stories)} story runs replayed; every one matches its recorded "
          f"test run exactly. courses/: {total / 1e6:.1f} MB")


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def export_brain(model_dir):
    """One PPO agent: normalization stats (float64) and its policy network (float32)."""
    from stable_baselines3 import PPO

    model = PPO.load(Path(model_dir) / "model.zip", device="cpu")
    with open(Path(model_dir) / "vecnormalize.pkl", "rb") as f:
        vecnorm = pickle.load(f)
    assert vecnorm.norm_obs and model.policy.squash_output is False
    sd = {k: v.detach().numpy() for k, v in model.policy.state_dict().items()}
    layers = []
    for w, b in (("mlp_extractor.policy_net.0", "tanh"), ("mlp_extractor.policy_net.2", "tanh"), ("action_net", None)):
        weight, bias = sd[f"{w}.weight"], sd[f"{w}.bias"]
        layers.append(dict(rows=int(weight.shape[0]), cols=int(weight.shape[1]), activation=b,
                           weight=b64(weight, "<f4"), bias=b64(bias, "<f4")))
    return dict(obs_mean=b64(vecnorm.obs_rms.mean, "<f8"), obs_var=b64(vecnorm.obs_rms.var, "<f8"),
                epsilon=float(vecnorm.epsilon), clip_obs=float(vecnorm.clip_obs), layers=layers)


def brains():
    out = dict(note="PPO policy networks (float32, little-endian, row-major) and observation-normalization "
                    "stats (float64), exported by export_web_data.py from models/<agent>/seed<N>/",
               agents={})
    for agent, label in HARDCORE_AGENTS.items():
        out["agents"][agent] = dict(label=label, seeds={str(s): export_brain(Path("models") / agent / f"seed{s}")
                                                         for s in config.TRAIN_SEEDS})
    path = Path("browser_sim") / "brains.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


def body_state(body):
    return [float(body.position.x), float(body.position.y), float(body.angle)]


def trace(model, seed, course_seed, out, steps):
    """Run one recorded test run and save the course layout plus every step's state, for comparing
    the browser simulator with Python step by step."""
    import Box2D

    agent = load_agent(Path("models") / model / f"seed{seed}")
    course = "hardcore" if model in HARDCORE_AGENTS else "normal"
    env = make_env(course)
    obs, _ = env.reset(seed=course_seed)
    game = env.unwrapped
    polys = []
    for body in game.terrain:
        shape = body.fixtures[0].shape
        if isinstance(shape, Box2D.b2PolygonShape):
            polys.append([[float(v[0]), float(v[1])] for v in shape.vertices])
    rec = dict(model=model, seed=seed, course_seed=course_seed, terrain_x=[float(v) for v in game.terrain_x],
               terrain_y=[float(v) for v in game.terrain_y], obstacles=polys,
               clouds=[[[float(a), float(b)] for a, b in poly] for poly, _, _ in game.cloud_poly],
               reset_obs=obs.tolist(), steps=[])
    score, done, n = 0.0, False, 0
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += float(reward)
        n += 1
        done = terminated or truncated
        if n <= steps:
            rec["steps"].append(dict(action=[float(a) for a in action], obs=obs.tolist(), reward=float(reward),
                                     hull=body_state(game.hull), legs=[body_state(b) for b in game.legs]))
    rec["result"] = dict(score=score, steps=n, outcome=info["outcome"], distance=info["distance"], end_x=info["end_x"])
    Path(out).write_text(json.dumps(rec))
    print(f"wrote {out}: {rec['result']}")


def curve_series(c, x, scores):
    r = lambda a: [round(float(v), 2) for v in a]
    return dict(x=r(x), mean=r(scores.mean(0)), lo=r(scores.min(0)), hi=r(scores.max(0)))


def site():
    """web/data/site.json: every number and chart on the page, from the saved results only
    (the same sources and calculations as make_charts.py)."""
    import make_charts as mc
    from falls import analyse, obstacles_by_course

    r = json.loads((Path("runs") / "final" / "results.json").read_text())
    keep = ("mean_score", "finished_rate", "fell_rate", "timed_out_rate", "backed_off_rate", "mean_distance", "mean_steps")
    agent = lambda key: dict(pooled={k: r["agents"][key]["pooled"][k] for k in keep},
                             per_seed={s: {k: v[k] for k in keep} for s, v in r["agents"][key]["per_seed"].items()})
    t = r["paired"]["b_100m vs a_100m / hardcore"]

    curves = dict(normal={}, hardcore={})
    for run in ("sac", "ppo"):
        c = mc.load_curves(run, "normal")
        curves["normal"][run] = dict(steps=curve_series(c, c["steps"], c["score"]),
                                     minutes=curve_series(c, c["minutes"], c["score"]))
    for run in ("a", "b"):
        c = mc.load_curves(run, "hardcore")
        curves["hardcore"][run] = curve_series(c, c["steps"], mc.rolling(c["score"], 5))

    falls = json.loads((Path("runs") / "final" / "falls.json").read_text())
    rows = [x for x in csv.DictReader((Path("runs") / "final" / "episodes.csv").open())
            if x["course"] == "hardcore" and x["model"] in HARDCORE_AGENTS]
    corrected = analyse(rows, obstacles_by_course({int(x["course_seed"]) for x in rows}))
    where = {}
    for m in HARDCORE_AGENTS:
        f = falls[m]
        n_falls = sum(f["falls"].values())
        where[m] = dict(per_100_reached=f["falls_per_100_reached"], reached=f["obstacles_reached"], falls=f["falls"],
                        share={k: f["falls"].get(k, 0) / n_falls for k in ("stump", "pit", "stairs_up", "stairs_down", "open ground")},
                        timed_out_stuck_at=corrected[m]["timed_out_stuck_at"])

    out = dict(
        hardcore=dict(a=agent("a_100m / hardcore"), b=agent("b_100m / hardcore"),
                      paired={k: t[k] for k in ("finished_only_by_reference", "finished_only_by_other", "finished_by_both",
                                                 "finished_by_neither", "finished_mcnemar_p", "score_wilcoxon_p",
                                                 "distance_wilcoxon_p", "mean_distance_difference", "verdict")}),
        normal={m: agent(f"{m} / normal") for m in ("sac_524k", "ppo_524k", "ppo_5m", "sac_229k", "untrained")},
        zoo_reference=r["zoo_reference"],
        curves=curves, b_switch_steps=config.B_NORMAL_STEPS, step2_steps=config.STEP2_STEPS,
        total_steps=config.STEP3_TOTAL_STEPS, where=where)
    path = WEB_DATA / "site.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


def read_course_bin(course_seed):
    """A course file written by pack(): the header, and each run's hull x per frame (world units)."""
    raw = (WEB_DATA / "courses" / f"{course_seed}.bin").read_bytes()
    n = struct.unpack("<I", raw[:4])[0]
    header = json.loads(raw[4:4 + n])
    base = 4 + n
    for run in header["runs"]:
        a = np.frombuffer(raw, "<i2", count=run["frames"] * 15, offset=base + run["offset"]).reshape(-1, 15)
        run["hull_x"] = a[:, 0] / header["xy_scale"]
    return header


def gallery():
    """web/data/gallery.json: the obstacle gallery's clips, each a time window of one recorded test
    run, picked by a fixed rule: the first test course (in order, training run 1 = seed 42) that
    shows the thing. Run after `replays` (it reads web/data/courses/)."""
    index = json.loads((WEB_DATA / "runs.json").read_text())
    seed = config.TRAIN_SEEDS[0]
    run_of = lambda c, model: next(r for r in c["runs"] if r["model"] == model and r["seed"] == seed)

    def first_fall(model, kind):
        for c in index["courses"]:
            r = run_of(c, model)
            if r["outcome"] == "fell" and r["fell_at"] == kind:
                end = r["steps"] / 50
                return dict(course_seed=c["course_seed"], model=model, t0=max(0.0, end - 4.0), t1=end)

    def first_pass(model, kind):
        for c in index["courses"]:
            for ob in c["obstacles"]:
                if ob["kind"] != kind:
                    continue
                x = next(r for r in read_course_bin(c["course_seed"])["runs"]
                         if r["model"] == model and r["seed"] == seed)["hull_x"]
                past = np.flatnonzero(x > ob["x_end"] + STEP)
                if len(past):
                    near = np.flatnonzero(x > ob["x_start"] - 3 * STEP)
                    return dict(course_seed=c["course_seed"], model=model,
                                t0=max(0.0, near[0] / 50 - 0.6), t1=min(len(x) - 1, past[0] + 40) / 50)

    def first_finish(model):
        for c in index["courses"]:
            r = run_of(c, model)
            if r["outcome"] == "finished":
                end = r["steps"] / 50
                return dict(course_seed=c["course_seed"], model=model, t0=max(0.0, end - 6.0), t1=end)

    clips = {
        "stump_fall": dict(first_fall("b_100m", "stump"), rule="the first test course where B fell at a stump"),
        "freeze": dict(course_seed=10016, model="a_100m", t0=2.0, t1=12.0,
                       rule="course 10016, the film's example: A stops making progress at 4.8 s and never moves on"),
        "pit_fall": dict(first_fall("a_100m", "pit"), rule="the first test course where A fell at a pit"),
        "stairs_up": dict(first_pass("b_100m", "stairs_up"), rule="the first test course where B climbed a staircase"),
        "finish": dict(first_finish("b_100m"), rule="the first test course B finished"),
    }
    for c in clips.values():
        c.update(seed=seed, t0=round(c["t0"], 2), t1=round(c["t1"], 2))
    (WEB_DATA / "gallery.json").write_text(json.dumps(clips, indent=1))
    for k, c in clips.items():
        print(f"{k:11s} course {c['course_seed']} {c['model']} seed {seed}: {c['t0']:.2f}-{c['t1']:.2f} s  ({c['rule']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replays")
    sub.add_parser("site")
    sub.add_parser("gallery")
    sub.add_parser("brains")
    t = sub.add_parser("trace")
    t.add_argument("model")
    t.add_argument("seed", type=int)
    t.add_argument("course_seed", type=int)
    t.add_argument("out")
    t.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    if args.cmd == "replays":
        replays()
    elif args.cmd == "site":
        site()
    elif args.cmd == "gallery":
        gallery()
    elif args.cmd == "brains":
        brains()
    else:
        trace(args.model, args.seed, args.course_seed, args.out, args.steps)


if __name__ == "__main__":
    main()
