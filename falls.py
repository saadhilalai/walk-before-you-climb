"""
falls.py — where do the agents fall, and where do they get stuck, on the hardcore course?

Reads an episodes file written by compare.py (e.g. runs/compare/step3_episodes.csv), rebuilds
each course from its seed, and for every model counts:
  * obstacles REACHED, by kind: an agent that gets further meets more obstacles, so raw fall
    counts alone would be unfair. An obstacle counts as reached once the robot's furthest
    point is within 2 tiles of its start (the same window walker_env uses for "fell at").
  * falls by kind (walker_env's "fell_at"), and falls per 100 obstacles of that kind reached.
  * for timed-out episodes, the obstacle the robot was stuck at (the first one not yet passed).

Usage:
    python falls.py --episodes runs/compare/step3_episodes.csv --name step3
Outputs: runs/compare/<name>_falls.json and printed tables.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from walker_env import FINISH_X, START_X, STEP, make_env

KINDS = ("stump", "pit", "stairs_up", "stairs_down")


def obstacles_by_course(course_seeds):
    env = make_env("hardcore")
    out = {}
    for s in sorted(course_seeds):
        env.reset(seed=s)
        out[s] = env.obstacles
    env.close()
    return out


def furthest_x(distance_pct):
    return START_X + distance_pct / 100 * (FINISH_X - START_X)


def analyse(rows, obstacles):
    by_model = defaultdict(lambda: dict(episodes=0, reached=Counter(), falls=Counter(), stuck=Counter()))
    for r in rows:
        m = by_model[r["model"]]
        m["episodes"] += 1
        x = furthest_x(float(r["distance"]))
        obs = obstacles[int(r["course_seed"])]
        m["reached"].update(o["kind"] for o in obs if o["x_start"] - 2 * STEP <= x)
        if r["outcome"] == "fell":
            m["falls"][r["fell_at"]] += 1   # walker_env's label (config.FALL_WINDOW_TILES)
        elif r["outcome"] == "timed_out":
            ahead = [o for o in obs if o["x_end"] + STEP >= x]
            m["stuck"][ahead[0]["kind"] if ahead else "past the last obstacle"] += 1
    report = {}
    for model, m in by_model.items():
        report[model] = dict(
            episodes=m["episodes"], obstacles_reached=dict(m["reached"]), falls=dict(m["falls"]),
            falls_per_100_reached={k: round(100 * m["falls"][k] / m["reached"][k], 1) if m["reached"][k] else None
                                   for k in KINDS},
            timed_out_stuck_at=dict(m["stuck"]),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.episodes)))
    report = analyse(rows, obstacles_by_course({int(r["course_seed"]) for r in rows}))

    for model, m in report.items():
        n_falls = sum(m["falls"].values())
        print(f"\n{model}: {m['episodes']:,} episodes, {n_falls:,} falls")
        print(f"  {'':14s} {'reached':>8s} {'falls':>6s} {'share of falls':>15s} {'falls per 100 reached':>22s}")
        for k in KINDS + ("open ground",):
            reached = m["obstacles_reached"].get(k, "")
            f = m["falls"].get(k, 0)
            rate = m["falls_per_100_reached"].get(k)
            print(f"  {k:14s} {reached!s:>8s} {f:>6,} {f / n_falls if n_falls else 0:>14.0%} "
                  f"{'' if rate is None else rate:>22}")
        stuck = m["timed_out_stuck_at"]
        n_stuck = sum(stuck.values())
        print(f"  timed out ({n_stuck:,}): stuck at " +
              ", ".join(f"{k} {v / n_stuck:.0%}" for k, v in sorted(stuck.items(), key=lambda kv: -kv[1])))
    out = Path("runs") / "compare" / f"{args.name}_falls.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
