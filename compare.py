"""
compare.py — compares trained models on the exact same analysis courses, for every training seed.

Usage:
    python compare.py --models sac_524k ppo_524k ppo_5m --reference sac_524k --name step2
    python compare.py --models a b --course hardcore --episodes 500 --name step3

Every model is graded on the OFFICIAL score plus finished / fell / backed off / timed out
rates and distance, on the analysis courses (config.ANALYSIS_SEED onwards). The test
courses stay hidden. 3 models run at a time (--workers), 1 CPU core each.

For each model vs each reference model, matched on the same course and the same training
seed, it reports paired tests:
  * finished (yes/no): McNemar's exact test. Only courses that exactly one of the two
    finished are evidence; if those split about 50/50 there's no real difference, and the
    p-value says how unlikely the actual split would be by luck.
  * score and distance: Wilcoxon signed-rank test on the per-course differences (a paired
    test for numbers that doesn't assume a bell curve).
  * per seed: in how many of the 3 seed pairs the model beats the reference. With only 3
    seeds this can't be "proven" on its own, but it shows whether the result holds for every run.

Outputs: runs/compare/<name>_episodes.csv (every episode), <name>.json, <name>.md
"""

import argparse
import csv
import json
from collections import Counter
from math import comb
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import config

OUT = Path("runs") / "compare"


def evaluate_job(job):
    """One model, one seed, all analysis courses. Runs in its own process on 1 core."""
    config.limit_cpu_threads(1)
    from evaluate import load_agent, run_episodes

    model, seed, course, first_seed, n, random_moves = job
    agent = load_agent(Path("models") / model / f"seed{seed}", deterministic=not random_moves)
    return job, agent.info, run_episodes(agent.act, course, first_seed, n, start_episode=agent.start_episode)


def mcnemar_p(only_a, only_b):
    """Exact two-sided McNemar test: chance of a split at least this lopsided if both were equal."""
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def wilcoxon_p(differences):
    """Two-sided Wilcoxon signed-rank test; courses with exactly zero difference are left out."""
    from scipy.stats import wilcoxon
    d = np.asarray(differences)
    d = d[d != 0]
    return float(wilcoxon(d).pvalue) if len(d) else 1.0


def summarize_model(episodes_by_seed, infos):
    from evaluate import summarize
    per_seed = {s: summarize(eps) for s, eps in episodes_by_seed.items()}
    pooled = summarize([e for eps in episodes_by_seed.values() for e in eps])
    spread = {k: dict(mean=float(np.mean([p[k] for p in per_seed.values()])),
                      std=float(np.std([p[k] for p in per_seed.values()])))
              for k in ("mean_score", "finished_rate", "mean_distance")}
    training = dict(steps=[infos[s]["steps"] for s in per_seed],
                    train_minutes=[infos[s]["train_minutes"] for s in per_seed])
    return dict(pooled=pooled, per_seed=per_seed, across_seeds=spread, training=training)


def paired(ref_eps, other_eps):
    """Paired tests: other vs ref, same training seed, same courses."""
    only_ref = only_other = both = neither = 0
    score_diff, dist_diff, seeds_better = [], [], 0
    for seed in ref_eps:
        r_list, o_list = ref_eps[seed], other_eps[seed]
        assert [e["course_seed"] for e in r_list] == [e["course_seed"] for e in o_list]
        for r, o in zip(r_list, o_list):
            rf, of = r["outcome"] == "finished", o["outcome"] == "finished"
            only_ref += rf and not of
            only_other += of and not rf
            both += rf and of
            neither += not rf and not of
            score_diff.append(o["score"] - r["score"])
            dist_diff.append(o["distance"] - r["distance"])
        seeds_better += np.mean([e["score"] for e in o_list]) > np.mean([e["score"] for e in r_list])
    return dict(
        finished_only_by_reference=only_ref, finished_only_by_other=only_other,
        finished_by_both=both, finished_by_neither=neither, finished_mcnemar_p=mcnemar_p(only_ref, only_other),
        mean_score_difference=float(np.mean(score_diff)), median_score_difference=float(np.median(score_diff)),
        score_wilcoxon_p=wilcoxon_p(score_diff),
        mean_distance_difference=float(np.mean(dist_diff)), distance_wilcoxon_p=wilcoxon_p(dist_diff),
        seeds_where_other_scores_higher=int(seeds_better), n_seeds=len(ref_eps), n_pairs=len(score_diff),
    )


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reference", nargs="+", default=None, help="model(s) the others are compared against")
    parser.add_argument("--course", default="normal", choices=list(config.COURSES))
    parser.add_argument("--episodes", type=int, default=config.N_ANALYSIS_EPISODES)
    parser.add_argument("--first-seed", type=int, default=config.ANALYSIS_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=3, help="models tested at the same time (1 core each)")
    parser.add_argument("--random-moves", action="store_true",
                        help="draw moves at random around the average move (default: always the average move)")
    args = parser.parse_args()
    assert not config.touches_test_courses(args.first_seed, args.episodes), "test courses are for the final test only"
    references = args.reference or args.models[:1]

    jobs = [(m, s, args.course, args.first_seed, args.episodes, args.random_moves) for m in args.models for s in args.seeds]
    moves = "randomized moves (seeded by course)" if args.random_moves else "average (deterministic) moves"
    print(f"{len(jobs)} model files x {args.episodes:,} {args.course} courses "
          f"(seeds {args.first_seed}-{args.first_seed + args.episodes - 1}), {moves}, {args.workers} at a time", flush=True)
    assert args.workers < config.MAX_CORES
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(evaluate_job, jobs)
    episodes = {(job[0], job[1]): eps for job, _, eps in results}
    infos = {(job[0], job[1]): info for job, info, _ in results}

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.name}_episodes.csv").open("w", newline="") as f:
        fields = ["model", "train_seed", "course_seed", "score", "outcome", "distance", "steps", "end_x", "fell_at"]
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for (m, s), eps in episodes.items():
            for e in eps:
                w.writerow(dict(model=m, train_seed=s, **e))

    report = dict(course=args.course, moves=moves, episodes_per_seed=args.episodes, first_course_seed=args.first_seed,
                  seeds=args.seeds, models={}, paired={})
    for m in args.models:
        report["models"][m] = summarize_model({s: episodes[(m, s)] for s in args.seeds},
                                              {s: infos[(m, s)] for s in args.seeds})
    for ref in references:
        for other in args.models:
            if other != ref and f"{ref}_vs_{other}" not in report["paired"]:
                report["paired"][f"{other}_vs_{ref}"] = paired({s: episodes[(ref, s)] for s in args.seeds},
                                                               {s: episodes[(other, s)] for s in args.seeds})

    # ---- Readable tables --------------------------------------------------------------
    n_total = args.episodes * len(args.seeds)
    lines = [f"## {args.name}: {args.course} course, {args.episodes:,} analysis courses x {len(args.seeds)} seeds "
             f"= {n_total:,} episodes per model, {moves}", "",
             "| Model | Steps | Training time (min, per seed) | Score (± across seeds) | Finished | Fell | "
             "Backed off | Timed out | Distance | Solved (300+)? |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in report["models"].items():
        p, a, t = r["pooled"], r["across_seeds"], r["training"]
        lines.append(f"| {m} | {np.mean(t['steps']):,.0f} | {np.mean(t['train_minutes']):.0f} | "
                     f"{p['mean_score']:.1f} ± {a['mean_score']['std']:.1f} | {p['finished_rate']:.1%} | "
                     f"{p['fell_rate']:.1%} | {p['backed_off_rate']:.1%} | {p['timed_out_rate']:.1%} | "
                     f"{p['mean_distance']:.1f}% | {'yes' if p['solved'] else 'no'} |")
    lines += ["", "Per seed (score / finished):", ""]
    for m, r in report["models"].items():
        cells = [f"seed {s}: {v['mean_score']:.1f} / {v['finished_rate']:.1%}" for s, v in r["per_seed"].items()]
        lines.append(f"- {m}: " + ", ".join(cells))
    lines += ["", "Paired tests (same course, same training seed):", ""]
    for key, t in report["paired"].items():
        other, ref = key.split("_vs_")
        lines.append(f"- **{other} vs {ref}**: finished only by {other} {t['finished_only_by_other']}, only by {ref} "
                     f"{t['finished_only_by_reference']} ({p_text(t['finished_mcnemar_p'])}); score difference "
                     f"{t['mean_score_difference']:+.1f} mean, {t['median_score_difference']:+.1f} median "
                     f"({p_text(t['score_wilcoxon_p'])}); distance {t['mean_distance_difference']:+.1f} points "
                     f"({p_text(t['distance_wilcoxon_p'])}); higher score in {t['seeds_where_other_scores_higher']} "
                     f"of {t['n_seeds']} seeds")
    text = "\n".join(lines)
    (OUT / f"{args.name}.md").write_text(text + "\n")
    (OUT / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    print(text)
    print(f"\nSaved {OUT}/{args.name}.json, .md and _episodes.csv")


if __name__ == "__main__":
    main()
