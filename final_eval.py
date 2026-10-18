"""
final_eval.py — Step 5: the ONE-TIME final test on the hidden test courses (seeds 10000-10099).

The plan is TEST_PLAN.md, committed to git BEFORE this runs. This script:
  * refuses to run if runs/final/ already has anything in it: the test runs exactly once;
  * refuses to run unless every file in the project is committed, so the result belongs to one
    git commit (recorded in the results);
  * runs every agent x training seed x course in the plan with AVERAGE (deterministic) moves,
    3 at a time, 1 CPU core each;
  * writes runs/final/episodes.csv (every run), results.json, results.md and falls.json.

Usage:
    python final_eval.py                                   # THE final test (once)
    python final_eval.py --dry-run --out <scratch folder>  # test the code on 10 ANALYSIS courses
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import config
from compare import evaluate_job, p_text, paired, summarize_model

OUT = Path("runs") / "final"
SEEDS = list(config.TRAIN_SEEDS)
VIDEO_SEED = 42
# (model, course, seeds): exactly the table in TEST_PLAN.md, section 2
PLAN = [("a_100m", "hardcore", SEEDS), ("b_100m", "hardcore", SEEDS)] + \
       [(m, "normal", SEEDS) for m in ("sac_524k", "ppo_524k", "ppo_5m", "a_100m", "b_100m")] + \
       [("untrained", "normal", [VIDEO_SEED]), ("untrained", "hardcore", [VIDEO_SEED]),
        ("sac_229k", "normal", [VIDEO_SEED])]
VIDEO_ONLY = {"untrained", "sac_229k"}
# (other, reference, course, primary?): TEST_PLAN.md, section 3
PAIRED_TESTS = [("b_100m", "a_100m", "hardcore", True),
                ("sac_524k", "ppo_524k", "normal", False),
                ("ppo_5m", "sac_524k", "normal", False),
                ("b_100m", "a_100m", "normal", False)]
ZOO_PPO_HARDCORE = 122.374   # RL Zoo benchmark.md, PPO after 100M hardcore steps (their agent, their courses)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def per_seed_ahead(ref_eps, other_eps):
    """In how many seeds the other agent has the higher finish rate / score (ties count for neither)."""
    fin = sum(np.mean([e["outcome"] == "finished" for e in other_eps[s]]) >
              np.mean([e["outcome"] == "finished" for e in ref_eps[s]]) for s in ref_eps)
    score_diffs = {s: float(np.mean([e["score"] for e in other_eps[s]]) - np.mean([e["score"] for e in ref_eps[s]]))
                   for s in ref_eps}
    return int(fin), score_diffs


def primary_verdict(t):
    if t["finished_mcnemar_p"] < 0.05:
        return "CONFIRMED" if t["finished_only_by_other"] > t["finished_only_by_reference"] else "REVERSED"
    return "NOT CONFIRMED"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="10 analysis courses per seed instead of the test")
    parser.add_argument("--out", default=None, help="output folder for a dry run")
    args = parser.parse_args()

    commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if args.dry_run:
        out, first_seed, n = Path(args.out), config.ANALYSIS_SEED, 10
        assert out.resolve() != OUT.resolve(), "a dry run must not write to runs/final"
        print(f"DRY RUN on {n} analysis courses per seed (not the test). Uncommitted files: {len(dirty.splitlines())}")
    else:
        out, first_seed, n = OUT, config.TEST_SEED, config.N_TEST_EPISODES
        if out.exists() and any(out.iterdir()):
            raise SystemExit("runs/final/ already has results. The final test runs exactly once.")
        if dirty:
            raise SystemExit("Some files aren't committed. Commit TEST_PLAN.md and everything else first:\n" + dirty)
        assert not config.touches_test_courses(config.ANALYSIS_SEED, config.N_ANALYSIS_EPISODES)
    out.mkdir(parents=True, exist_ok=True)
    started = dict(git_commit=commit, started=time.strftime("%Y-%m-%d %H:%M:%S"), dry_run=args.dry_run,
                   course_seeds=[first_seed, first_seed + n - 1], moves="average (deterministic)")
    (out / "started.json").write_text(json.dumps(started, indent=2))   # from now on, a second run refuses

    jobs = [(m, s, course, first_seed, n, False) for m, course, seeds in PLAN for s in seeds]
    print(f"{len(jobs)} model files x {n} courses, average moves, 3 at a time (commit {commit[:7]})", flush=True)
    with get_context("spawn").Pool(3) as pool:
        results = pool.map(evaluate_job, jobs)

    episodes = {(job[0], job[2], job[1]): eps for job, _, eps in results}     # (model, course, seed)
    infos = {(job[0], job[2], job[1]): info for job, info, _ in results}
    with (out / "episodes.csv").open("w", newline="") as f:
        fields = ["model", "train_seed", "course", "course_seed", "score", "outcome", "distance", "steps", "end_x",
                  "fell_at"]
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for (m, course, s), eps in episodes.items():
            for e in eps:
                w.writerow(dict(model=m, train_seed=s, course=course, **e))

    report = dict(run=dict(**started, finished=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                           machine=f"{platform.system()} {platform.machine()}"),
                  agents={}, paired={})
    for m, course, seeds in PLAN:
        report["agents"][f"{m} / {course}"] = dict(
            video_only=m in VIDEO_ONLY,
            **summarize_model({s: episodes[(m, course, s)] for s in seeds}, {s: infos[(m, course, s)] for s in seeds}))
    for other, ref, course, primary in PAIRED_TESTS:
        r = {s: episodes[(ref, course, s)] for s in SEEDS}
        o = {s: episodes[(other, course, s)] for s in SEEDS}
        t = paired(r, o)
        t["seeds_where_other_finishes_more"], t["score_difference_per_seed"] = per_seed_ahead(r, o)
        t["score_result_robust"] = bool(t["score_wilcoxon_p"] < 0.05 and
                                        len({np.sign(d) for d in t["score_difference_per_seed"].values()}) == 1)
        t["primary"] = primary
        if primary:
            t["verdict"] = primary_verdict(t)
        report["paired"][f"{other} vs {ref} / {course}"] = t
    a_hardcore = report["agents"]["a_100m / hardcore"]["pooled"]["mean_score"]
    report["zoo_reference"] = dict(zoo_ppo_100m=ZOO_PPO_HARDCORE, our_a_100m=a_hardcore,
                                   note="rough comparison only: their single agent, their courses; no test")

    from falls import analyse, obstacles_by_course
    hard_rows = [dict(model=m, course_seed=e["course_seed"], **{k: e[k] for k in ("outcome", "distance", "fell_at")})
                 for (m, course, s), eps in episodes.items() if course == "hardcore" and m not in VIDEO_ONLY for e in eps]
    falls = analyse(hard_rows, obstacles_by_course({r["course_seed"] for r in hard_rows}))
    (out / "falls.json").write_text(json.dumps(falls, indent=2))
    (out / "results.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Final test{' (DRY RUN, analysis courses)' if args.dry_run else ''}: courses {first_seed}-"
             f"{first_seed + n - 1}, average moves, git commit {commit[:7]}", "",
             "| Agent | Course | Score (± across seeds) | Finished | Fell | Backed off | Timed out | Distance | Solved? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, a in report["agents"].items():
        p, sd = a["pooled"], a["across_seeds"]["mean_score"]["std"]
        lines.append(f"| {key.split(' / ')[0]}{' (video only)' if a['video_only'] else ''} | {key.split(' / ')[1]} | "
                     f"{p['mean_score']:.1f} ± {sd:.1f} | {p['finished_rate']:.1%} | {p['fell_rate']:.1%} | "
                     f"{p['backed_off_rate']:.1%} | {p['timed_out_rate']:.1%} | {p['mean_distance']:.1f}% | "
                     f"{'yes' if p['solved'] else 'no'} |")
    lines += ["", "Paired tests (same training seed, same course):", ""]
    for key, t in report["paired"].items():
        other, ref = key.split(" / ")[0].split(" vs ")
        lines.append(f"- {'**PRIMARY** ' if t['primary'] else ''}**{key}**: finished only by {other} "
                     f"{t['finished_only_by_other']}, only by {ref} {t['finished_only_by_reference']} "
                     f"({p_text(t['finished_mcnemar_p'])}); {other} finishes more in {t['seeds_where_other_finishes_more']} of 3 seeds; "
                     f"score {t['mean_score_difference']:+.1f} mean, {t['median_score_difference']:+.1f} median "
                     f"({p_text(t['score_wilcoxon_p'])}; per seed "
                     f"{', '.join(f'{d:+.1f}' for d in t['score_difference_per_seed'].values())}; "
                     f"{'robust' if t['score_result_robust'] else 'not robust'}); distance "
                     f"{t['mean_distance_difference']:+.1f} points ({p_text(t['distance_wilcoxon_p'])})"
                     + (f" -> **{t['verdict']}**" if t["primary"] else ""))
    lines += ["", f"A on hardcore: {a_hardcore:.1f} vs the RL Zoo's {ZOO_PPO_HARDCORE:.0f} (rough comparison, no test).",
              "", "Where they fall on hardcore (falls per 100 obstacles reached):", ""]
    for m in ("a_100m", "b_100m"):
        f = falls[m]
        lines.append(f"- {m}: " + ", ".join(f"{k} {v}" for k, v in f["falls_per_100_reached"].items()) +
                     f"; open ground {f['falls'].get('open ground', 0)} falls; timed out stuck at "
                     + ", ".join(f"{k} {v}" for k, v in f["timed_out_stuck_at"].items()))
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}/episodes.csv, results.json, results.md, falls.json")


if __name__ == "__main__":
    main()
