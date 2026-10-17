"""
review.py — Step 4 checkpoint review: how does each agent walk, and what are its odd habits?

Replays recorded analysis-course runs (checking each replay against its recorded score) with
the physics state logged at every step, then measures:
  gaits    how each trained agent walks on the normal course: speed, body lean, foot contacts,
           steps per second, how the two legs share the work, motor effort
  stumps   when A or B runs out of time stuck at a stump: is it in front of the stump or
           straddling it, and is it really standing still?
  flat     falls that walker_env labels "flat ground": where are they, relative to the
           obstacles just behind and ahead?

Usage:
    python review.py gaits | stumps | flat
Outputs: runs/review/<what>.json and printed tables.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config

config.limit_cpu_threads(1)

from evaluate import load_agent  # noqa: E402
from walker_env import FINISH_X, START_X, STEP, make_env  # noqa: E402

OUT = Path("runs") / "review"
FPS = 50
GAIT_AGENTS = (("sac_524k", "step2"), ("ppo_5m", "step2"), ("a_100m", "step3_normal"), ("b_100m", "step3_normal"))


def episodes(name):
    rows = list(csv.DictReader(open(Path("runs") / "compare" / f"{name}_episodes.csv")))
    return {(r["model"], int(r["train_seed"]), int(r["course_seed"])): r for r in rows}


def replay(agent, course, course_seed, rec=None):
    """One episode with the physics state logged every step. Checks it against the recorded run."""
    env = make_env(course)
    obs, _ = env.reset(seed=course_seed)
    g = env.unwrapped
    log, score, done = defaultdict(list), 0.0, False
    while not done:
        action = agent.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        score += float(reward)
        done = term or trunc
        log["x"].append(g.hull.position.x)
        log["vx"].append(g.hull.linearVelocity.x)
        log["angle"].append(math.degrees(g.hull.angle))
        log["contact1"].append(g.legs[1].ground_contact)
        log["contact2"].append(g.legs[3].ground_contact)
        for name, j in zip(("hip1", "knee1", "hip2", "knee2"), g.joints):
            log[name].append(math.degrees(j.angle))
            log[name + "_speed"].append(j.speed)
        log["effort"].append(float(np.abs(np.clip(action, -1, 1)).mean()))
    env.close()
    if rec is not None:
        assert info["outcome"] == rec["outcome"] and abs(score - float(rec["score"])) < 1e-9, "replay mismatch"
    return {k: np.array(v) for k, v in log.items()}, dict(score=score, outcome=info["outcome"], steps=len(log["x"]),
                                                          end_x=info["end_x"])


def touchdowns(contact):
    return int(np.sum(np.diff(contact.astype(int)) == 1))


def gait(log, skip=50):
    """Gait numbers over one episode, leaving out the first second (getting going)."""
    s = slice(skip, None)
    c1, c2 = log["contact1"][s], log["contact2"][s]
    secs = len(c1) / FPS
    t1, t2 = touchdowns(c1), touchdowns(c2)
    return dict(
        speed_course_pct_per_10s=float(100 * (log["x"][-1] - log["x"][skip]) / (FINISH_X - START_X) / secs * 10),
        lean_deg=float(log["angle"][s].mean()), lean_wobble_deg=float(log["angle"][s].std()),
        one_foot_down=float(np.mean(c1 ^ c2)), both_feet_down=float(np.mean(c1 & c2)),
        airborne=float(np.mean(~c1 & ~c2)),
        steps_per_sec=float((t1 + t2) / secs), share_of_steps_leg1=float(t1 / (t1 + t2)) if t1 + t2 else float("nan"),
        hip1_mean_deg=float(log["hip1"][s].mean()), hip2_mean_deg=float(log["hip2"][s].mean()),
        hip_swing_deg=float(np.mean([log["hip1"][s].std(), log["hip2"][s].std()])),
        knee1_mean_deg=float(log["knee1"][s].mean()), knee2_mean_deg=float(log["knee2"][s].mean()),
        effort=float(log["effort"][s].mean()),
    )


def run_gaits(n=10):
    """Each agent x seed on the first n normal analysis courses it FINISHED (steady walking)."""
    report = {}
    for model, name in GAIT_AGENTS:
        eps = episodes(name)
        report[model] = {}
        for seed in config.TRAIN_SEEDS:
            agent = load_agent(Path("models") / model / f"seed{seed}")
            done = [k for k, r in eps.items() if k[0] == model and k[1] == seed and r["outcome"] == "finished"]
            stats = [gait(replay(agent, "normal", cs, eps[(m, s, cs)])[0]) for m, s, cs in sorted(done)[:n]]
            report[model][seed] = {k: float(np.mean([st[k] for st in stats])) for k in stats[0]}
            report[model][seed]["courses"] = len(stats)
    keys = list(next(iter(report["sac_524k"].values())))
    print(f"{'':26s}" + "".join(f"{m:>22s}" for m in report))
    for k in keys:
        print(f"{k:26s}" + "".join("".join(f"{report[m][s][k]:7.2f}" for s in config.TRAIN_SEEDS) + " "
                                   for m in report))
    return report


def run_stumps(per_seed=20):
    """Runs that timed out stuck at a stump: where is the robot, and is it moving?"""
    from falls import obstacles_by_course
    eps = episodes("step3")
    obs = obstacles_by_course({k[2] for k in eps})
    report = {}
    for model in ("a_100m", "b_100m"):
        poses, report[model] = Counter(), dict(examples=[])
        for seed in config.TRAIN_SEEDS:
            agent = load_agent(Path("models") / model / f"seed{seed}")
            picked = 0
            for (m, s, cs), r in sorted(eps.items()):
                if m != model or s != seed or r["outcome"] != "timed_out" or picked >= per_seed:
                    continue
                x_far = START_X + float(r["distance"]) / 100 * (FINISH_X - START_X)
                ahead = [o for o in obs[cs] if o["x_end"] + STEP >= x_far]
                if not ahead or ahead[0]["kind"] != "stump":
                    continue
                picked += 1
                stump = ahead[0]
                log, res = replay(agent, "hardcore", cs, r)
                last = slice(-500, None)                                 # the last 10 seconds
                x = res["end_x"]
                where = ("in front of it" if x < stump["x_start"] else
                         "straddling it" if x < stump["x_end"] + STEP else "past it, stuck further on")
                still = float(np.mean([np.abs(log[j + "_speed"][last]).mean() for j in ("hip1", "knee1", "hip2", "knee2")]))
                feet = (log["contact1"][-1], log["contact2"][-1])
                poses[where] += 1
                report[model]["examples"].append(dict(
                    seed=seed, course=cs, where=where, body_minus_stump_start_tiles=round((x - stump["x_start"]) / STEP, 2),
                    stump_tiles=round((stump["x_end"] - stump["x_start"]) / STEP), joint_speed_last_10s=round(still, 4),
                    moved_last_10s=round(float(np.ptp(log["x"][last])), 3), feet_down=[bool(f) for f in feet],
                    lean_deg=round(float(log["angle"][-1]), 1)))
        ex = report[model]["examples"]
        report[model]["pose"] = dict(poses)
        report[model]["median_joint_speed_last_10s"] = float(np.median([e["joint_speed_last_10s"] for e in ex]))
        report[model]["median_moved_last_10s"] = float(np.median([e["moved_last_10s"] for e in ex]))
        print(f"{model}: {len(ex)} runs stuck at a stump -> {dict(poses)}; in the last 10 s: median joint speed "
              f"{report[model]['median_joint_speed_last_10s']:.4f}, body moved {report[model]['median_moved_last_10s']:.3f} "
              f"units (a stump is {STEP:.2f} wide)")
    return report


def run_flat():
    """Falls labelled "flat ground": distance from the obstacle just behind and the one just ahead."""
    from falls import obstacles_by_course
    eps = episodes("step3")
    obs = obstacles_by_course({k[2] for k in eps})
    report = {}
    for model in ("a_100m", "b_100m"):
        behind_kind, gaps_behind, gaps_ahead = Counter(), [], []
        for seed in config.TRAIN_SEEDS:
            agent = load_agent(Path("models") / model / f"seed{seed}")
            for (m, s, cs), r in sorted(eps.items()):
                if m != model or s != seed or r["fell_at"] != "flat ground":
                    continue
                _, res = replay(agent, "hardcore", cs, r)
                x = res["end_x"]
                before = [o for o in obs[cs] if o["x_end"] < x]
                after = [o for o in obs[cs] if o["x_start"] > x]
                if before:
                    behind_kind[before[-1]["kind"]] += 1
                    gaps_behind.append((x - before[-1]["x_end"]) / STEP)
                else:
                    behind_kind["start pad (no obstacle yet)"] += 1
                if after:
                    gaps_ahead.append((after[0]["x_start"] - x) / STEP)
        gb, ga = np.array(gaps_behind), np.array(gaps_ahead)
        report[model] = dict(falls=int(sum(behind_kind.values())), obstacle_just_behind=dict(behind_kind),
                             tiles_after_previous_obstacle_quartiles=np.percentile(gb, [25, 50, 75]).round(1).tolist(),
                             within_3_tiles_after_an_obstacle=float(np.mean(gb <= 3)),
                             tiles_before_next_obstacle_quartiles=np.percentile(ga, [25, 50, 75]).round(1).tolist())
        print(f"{model}: {report[model]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["gaits", "stumps", "flat"])
    args = parser.parse_args()
    report = {"gaits": run_gaits, "stumps": run_stumps, "flat": run_flat}[args.what]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.what}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {OUT}/{args.what}.json")


if __name__ == "__main__":
    main()
