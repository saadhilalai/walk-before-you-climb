"""
make_charts.py — draws the charts from saved results (never re-runs training or tests).

Inputs:  runs/<run>/seed<N>/curve.csv (learning curves), runs/compare/step2.json (analysis results)
Outputs: charts/2_ppo_vs_sac.png  (more charts are added in Step 6)

Style: one fixed color per agent across every chart (palette checked with a
colorblind-safety validator), thin lines, light gridlines, text in ink colors
(never the series color), values labelled directly. Same style as the rocket project.

Usage:
    python make_charts.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

OUT = Path("charts")

# ---- Style tokens -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
RUN_STYLE = {   # name shown on charts, fixed color (follows the agent everywhere)
    "sac": ("SAC", "#2a78d6"),
    "ppo": ("PPO", "#eb6834"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 10,
})


def title(fig, main, sub):
    fig.text(0.01, 0.975, main, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.915, sub, fontsize=10.5, color=INK_2, va="top")


def load_curves(run, course="normal"):
    """Learning curves (check-ups on one course) of all seeds of a run, trimmed to the check-ups
    every seed has reached. Returns steps, minutes (mean over seeds) and a (seeds x check-ups)
    array per column. Step 2's curve files have no "course" column: they are all normal course."""
    rows = []
    for seed in config.TRAIN_SEEDS:
        path = Path("runs") / run / f"seed{seed}" / "curve.csv"
        if path.exists():
            rows.append([r for r in csv.DictReader(path.open()) if r.get("course", "normal") == course])
    if not rows:
        return None
    n = min(len(r) for r in rows)
    col = lambda k: np.array([[float(x[k]) for x in r[:n]] for r in rows])
    return dict(steps=col("steps")[0], minutes=col("train_minutes").mean(0), score=col("mean_score"),
                finished=col("finished_rate"), n_seeds=len(rows))


def draw_curve(ax, x, scores, color, label, end_label=True):
    mean = scores.mean(0)
    ax.fill_between(x, scores.min(0), scores.max(0), color=color, alpha=0.13, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2, solid_capstyle="round", label=label)
    ax.plot(x[-1], mean[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    if end_label:
        ax.annotate(f"{label}: {mean[-1]:.0f}", (x[-1], mean[-1]), xytext=(6, 6), textcoords="offset points",
                    fontsize=10, color=INK)


# ---- 2. PPO vs SAC ------------------------------------------------------------------
def ppo_vs_sac():
    curves = {run: load_curves(run) for run in ("sac", "ppo")}
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

    for run, c in curves.items():
        if c is None:
            continue
        label, color = RUN_STYLE[run]
        equal = c["steps"] <= config.STEP2_STEPS
        draw_curve(left, c["steps"][equal] / 1e3, c["score"][:, equal], color, label)
        draw_curve(right, c["minutes"], c["score"], color, label)
        if run == "ppo" and c["steps"][-1] > config.STEP2_STEPS:
            i = int(np.flatnonzero(equal)[-1])
            right.plot(c["minutes"][i], c["score"][:, i].mean(), "o", color=color, markersize=7,
                       markerfacecolor=SURFACE, markeredgewidth=1.8, linestyle="none",
                       label="PPO at 524,288 steps")

    for ax in (left, right):
        ax.axhline(config.SOLVED_SCORE, color=INK_2, linewidth=0.9)
        ax.set_ylim(-140, 345)
    left.text(4, config.SOLVED_SCORE + 7, "solved (300)", fontsize=9, color=INK_2)
    left.set_xlim(0, config.STEP2_STEPS / 1e3 * 1.12)
    left.set_xlabel("practice steps (thousands)")
    left.set_ylabel("official score (10 check-up courses)")
    left.set_title("Same practice: 524,288 steps each", loc="left", fontsize=11.5)
    right.set_xlabel("training time (minutes, 1 CPU core per run)")
    right.set_title("Same clock: PPO continued to 5M steps", loc="left", fontsize=11.5)
    right.set_xlim(0, right.get_xlim()[1] * 1.08)
    left.legend(loc="lower right")
    right.legend(loc="lower right")
    n = min(c["n_seeds"] for c in curves.values() if c)
    title(fig, "PPO vs SAC on the normal course",
          f"Line = mean of {n} training seeds, band = best to worst seed, each point = 10 check-up courses. "
          "Check-ups: SAC every 16,384 steps; PPO after every brain update (65,536 steps), every 131,072 after 524k.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.87))
    results = Path("runs") / "compare" / "step2.json"
    if results.exists():                 # the real result: 1,000 fresh analysis courses per seed
        models = json.loads(results.read_text())["models"]
        cells = [f"{label} {models[m]['pooled']['mean_score']:.0f} (finishes {models[m]['pooled']['finished_rate']:.0%})"
                 for m, label in (("sac_524k", "SAC 524k"), ("ppo_524k", "PPO 524k"), ("ppo_5m", "PPO 5M"))]
        fig.text(0.01, 0.015, "Final score on 1,000 analysis courses x 3 seeds: " + "  ·  ".join(cells) +
                 ". The check-up curves above use only 10 courses, so they are noisier.", fontsize=10, color=INK_2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "2_ppo_vs_sac.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/2_ppo_vs_sac.png")


if __name__ == "__main__":
    ppo_vs_sac()
