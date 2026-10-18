"""
make_charts.py — draws the charts from saved results (never re-runs training or tests).

Inputs:  runs/<run>/seed<N>/curve.csv (learning curves), runs/compare/step2.json (analysis results)
Outputs (Step 6): charts/1_learning_curves.png, 2_ppo_vs_sac.png, 3_a_vs_b_hardcore.png,
         3b_a_vs_b_curves.png, 4_where_agents_fall.png
Final-test numbers come from runs/final/ (the one-time test); curves from the training check-ups.

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
    "a": ("A: hardcore from scratch", "#1baf7a"),
    "b": ("B: curriculum (5M normal first)", "#eda100"),
}
ZOO_PPO_HARDCORE = 122.374   # RL Zoo benchmark.md: their single PPO agent after 100M hardcore steps
FINAL = Path("runs") / "final" / "results.json"
FINAL_FALLS = Path("runs") / "final" / "falls.json"
SEED_MARKERS = {42: "o", 142: "s", 242: "D"}

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
                finished=col("finished_rate"), distance=col("mean_distance"), fell=col("fell_rate"),
                n_seeds=len(rows))


def rolling(a, k):
    """Average of the last k check-ups (fewer at the start), per seed."""
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    n = np.minimum(np.arange(1, a.shape[1] + 1), k)
    return (c[:, 1:] - c[:, np.maximum(np.arange(1, a.shape[1] + 1) - k, 0)]) / n


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
    if FINAL.exists():                   # the real result: the one-time final test
        models = {k.split(" / ")[0]: v for k, v in json.loads(FINAL.read_text())["agents"].items()
                  if k.endswith(" / normal")}
        cells = [f"{label} {models[m]['pooled']['mean_score']:.0f} (finishes {models[m]['pooled']['finished_rate']:.0%})"
                 for m, label in (("sac_524k", "SAC 524k"), ("ppo_524k", "PPO 524k"), ("ppo_5m", "PPO 5M"))]
        fig.text(0.01, 0.015, "Final test (100 hidden test courses x 3 seeds): " + "  ·  ".join(cells) +
                 ". The check-up curves above use only 10 courses, so they are noisier.", fontsize=10, color=INK_2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "2_ppo_vs_sac.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/2_ppo_vs_sac.png")


# ---- 3 (draft for Step 3). A vs B learning curves on hardcore ---------------------------
def a_vs_b_curves(k=5):
    curves = {run: load_curves(run, "hardcore") for run in ("a", "b")}
    panels = (("score", "official score", 1, (-110, 160)), ("finished", "finished the course (%)", 100, (0, 50)),
              ("distance", "distance reached (% of course)", 1, (0, 80)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (key, ylabel, scale, ylim) in zip(axes, panels):
        for run, c in curves.items():
            if c is None:
                continue
            label, color = RUN_STYLE[run]
            draw_curve(ax, c["steps"] / 1e6, rolling(c[key] * scale, k), color, label.split(":")[0])
        ax.axvline(config.B_NORMAL_STEPS / 1e6, color=INK_2, linewidth=0.9, linestyle=":")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, config.STEP3_TOTAL_STEPS / 1e6 * 1.1)
        ax.set_xlabel("practice steps (millions)")
        ax.set_title(ylabel, loc="left", fontsize=11.5)
    axes[0].text(config.B_NORMAL_STEPS / 1e6 + 1.5, -100, "B switches to hardcore (5M)", fontsize=9, color=INK_2)
    axes[0].axhline(ZOO_PPO_HARDCORE, color=INK_2, linewidth=0.9, linestyle="--")
    axes[0].text(config.B_NORMAL_STEPS / 1e6 + 1.5, ZOO_PPO_HARDCORE + 5, f"RL Zoo's PPO after 100M: {ZOO_PPO_HARDCORE:.0f} (their courses, 1 agent)",
                 fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], color=RUN_STYLE[r][1], linewidth=2, label=RUN_STYLE[r][0]) for r in ("a", "b")]
    axes[2].legend(handles=handles, loc="lower right")
    n = min(c["n_seeds"] for c in curves.values() if c)
    title(fig, "Hardcore course: from scratch (A) vs curriculum (B), same 100M practice steps",
          f"Learning curves. Check-ups on 20 hardcore courses. Line = mean of {n} training seeds, averaged over the last {k} check-ups "
          "(~5M steps) to reduce noise; band = best to worst seed.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    if FINAL.exists():                   # the real result: the one-time final test
        agents = json.loads(FINAL.read_text())["agents"]
        cells = [f"{label} score {agents[m + ' / hardcore']['pooled']['mean_score']:.0f}, finished "
                 f"{agents[m + ' / hardcore']['pooled']['finished_rate']:.1%}, distance "
                 f"{agents[m + ' / hardcore']['pooled']['mean_distance']:.0f}%" for m, label in (("a_100m", "A"), ("b_100m", "B"))]
        fig.text(0.01, 0.012, "Final test (100 hidden hardcore courses x 3 seeds): " + "  ·  ".join(cells) +
                 ". The check-up curves above use only 20 courses.", fontsize=10, color=INK_2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "3b_a_vs_b_curves.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/3b_a_vs_b_curves.png")


# ---- 1. Learning curves of every agent -------------------------------------------------------
def learning_curves(k=5):
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for run in ("sac", "ppo"):
        c = load_curves(run, "normal")
        label, color = RUN_STYLE[run]
        draw_curve(left, c["steps"] / 1e6, c["score"], color, label)
    left.axhline(config.SOLVED_SCORE, color=INK_2, linewidth=0.9)
    left.text(3.3, config.SOLVED_SCORE + 7, "solved (300)", fontsize=9, color=INK_2)
    left.set_xlim(0, 5.6)
    left.set_xlabel("practice steps (millions)")
    left.set_ylabel("official score (check-up courses)")
    left.set_title("Normal course (Step 2)", loc="left", fontsize=11.5)
    for run in ("a", "b"):
        c = load_curves(run, "hardcore")
        label, color = RUN_STYLE[run]
        draw_curve(right, c["steps"] / 1e6, rolling(c["score"], k), color, label.split(":")[0])
    right.axvline(config.B_NORMAL_STEPS / 1e6, color=INK_2, linewidth=0.9, linestyle=":")
    right.text(config.B_NORMAL_STEPS / 1e6 + 1.5, -120, "B switches to hardcore (5M)", fontsize=9, color=INK_2)
    right.set_xlim(0, config.STEP3_TOTAL_STEPS / 1e6 * 1.1)
    right.set_xlabel("practice steps (millions)")
    right.set_title(f"Hardcore course (Step 3), averaged over the last {k} check-ups", loc="left", fontsize=11.5)
    right.set_ylim(-140, 345)
    for ax, runs in ((left, ("sac", "ppo")), (right, ("a", "b"))):
        handles = [plt.Line2D([], [], color=RUN_STYLE[r][1], linewidth=2, label=RUN_STYLE[r][0]) for r in runs]
        ax.legend(handles=handles, loc="lower right" if ax is left else "upper left")
    title(fig, "How every agent learned",
          "Line = mean of 3 training seeds, band = best to worst seed. Normal: 10 check-up courses; hardcore: 20. "
          "SAC learns the normal course in a tenth of PPO's practice; nobody gets close to solving hardcore.")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "1_learning_curves.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/1_learning_curves.png")


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


# ---- 3. A vs B on the hardcore final test ------------------------------------------------------------
def a_vs_b_hardcore():
    r = json.loads(FINAL.read_text())
    t = r["paired"]["b_100m vs a_100m / hardcore"]
    panels = (("mean_score", "official score", 1, t["score_wilcoxon_p"], "Wilcoxon", "not shown"),
              ("finished_rate", "finished the course (%)", 100, t["finished_mcnemar_p"], "McNemar", "confirmed"),
              ("mean_distance", "distance reached (% of course)", 1, t["distance_wilcoxon_p"], "Wilcoxon", "proven"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (key, label, scale, p, test, verdict) in zip(axes, panels):
        for i, m in enumerate(("a_100m", "b_100m")):
            run = m[0]
            name, color = RUN_STYLE[run]
            a = r["agents"][f"{m} / hardcore"]
            v = a["pooled"][key] * scale
            ax.bar(i, v, width=0.55, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
            for j, (seed, ps) in enumerate(a["per_seed"].items()):
                ax.plot(i + (j - 1) * 0.13, ps[key] * scale, SEED_MARKERS[int(seed)], color=INK, markersize=6,
                        markerfacecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        unit = "%" if key != "mean_score" else ""
        vals = [r["agents"][f"{m} / hardcore"]["pooled"][key] * scale for m in ("a_100m", "b_100m")]
        ax.set_xticks([0, 1], [f"A: from scratch\n{vals[0]:.1f}{unit}", f"B: curriculum\n{vals[1]:.1f}{unit}"])
        ax.tick_params(axis="x", labelsize=11.5)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(label, loc="left", fontsize=11.5)
        ax.text(0.5, -0.24, f"B vs A, paired {test}: {p_label(p)} ({verdict})", transform=ax.transAxes,
                ha="center", fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylim(-30, 110)
    axes[1].set_ylim(0, 35)
    axes[2].set_ylim(0, 70)
    handles = [plt.Line2D([], [], marker=SEED_MARKERS[s], linestyle="none", color=INK, markerfacecolor=SURFACE,
                          markersize=6, label=f"seed {s}") for s in SEED_MARKERS]
    axes[2].legend(handles=handles, loc="upper left", title="each training seed", title_fontsize=9)
    title(fig, "Final test on the hardcore course: curriculum (B) vs from scratch (A)",
          "100 hidden test courses x 3 training seeds = 300 runs each, same courses for both. Bar = all 300 runs; "
          "markers = each seed. Primary question (finishing): confirmed. Score difference: not shown.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.87))
    fig.text(0.01, 0.012, "Most of B's finishing lead comes from seed 242 (31% vs 0%); in seed 42, A finished slightly more "
             "(4% vs 3%). Neither agent comes close to solving hardcore (300+).", fontsize=10, color=INK_2)
    fig.savefig(OUT / "3_a_vs_b_hardcore.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/3_a_vs_b_hardcore.png")


# ---- 4. Where they fall (hardcore final test) --------------------------------------------------------
def where_agents_fall():
    falls = json.loads(FINAL_FALLS.read_text())
    kinds = (("stump", "stumps"), ("pit", "pits"), ("stairs_up", "stairs up"), ("stairs_down", "stairs down"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.4))
    h = 0.36
    for i, m in enumerate(("a_100m", "b_100m")):
        name, color = RUN_STYLE[m[0]]
        f = falls[m]
        rates = [f["falls_per_100_reached"][k] or 0 for k, _ in kinds]
        y = np.arange(len(kinds)) + (i - 0.5) * h
        left.barh(y, rates, height=h, color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=2)
        for yy, v in zip(y, rates):
            left.text(v + 0.3, yy, f"{v:.1f}", va="center", fontsize=10, color=INK)
        n_falls = sum(f["falls"].values())
        share = [f["falls"].get(k, 0) / n_falls * 100 for k, _ in kinds] + [f["falls"].get("open ground", 0) / n_falls * 100]
        y2 = np.arange(len(kinds) + 1) + (i - 0.5) * h
        right.barh(y2, share, height=h, color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=2)
        for yy, v in zip(y2, share):
            right.text(v + 0.6, yy, f"{v:.0f}%", va="center", fontsize=10, color=INK)
    left.set_yticks(range(len(kinds)), [lbl for _, lbl in kinds])
    right.set_yticks(range(len(kinds) + 1), [lbl for _, lbl in kinds] + ["open ground"])
    for ax in (left, right):
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    left.set_title("Falls per 100 obstacles reached (fair: B reaches more)", loc="left", fontsize=11.5)
    left.set_xlabel("falls per 100 obstacles of that kind reached")
    right.set_title("Where each agent's falls happen (share of its falls)", loc="left", fontsize=11.5)
    right.set_xlabel("% of the agent's falls")
    left.legend(loc="lower right")
    # Where timed-out runs got stuck, judged by where the run ENDED (falls.py, fixed in Step 7). The recorded
    # falls.json used the furthest point instead (A: 145 of 188 at a stump); the corrected count is used here.
    from falls import analyse, obstacles_by_course
    rows = [r for r in csv.DictReader((Path("runs") / "final" / "episodes.csv").open())
            if r["course"] == "hardcore" and r["model"] in ("a_100m", "b_100m")]
    corrected = analyse(rows, obstacles_by_course({int(r["course_seed"]) for r in rows}))
    stuck = {m: corrected[m]["timed_out_stuck_at"] for m in ("a_100m", "b_100m")}
    note = "  ·  ".join(f"{RUN_STYLE[m[0]][0].split(':')[0]}: {s.get('stump', 0)} of {sum(s.values())} at a stump"
                        for m, s in stuck.items())
    title(fig, "Where the agents fall on the hardcore course (final test)",
          "300 test runs each. A fall belongs to an obstacle if the body was from 2 tiles before it to 3 tiles after it "
          "(one body length), otherwise 'open ground'. Rule fixed before the test.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.87))
    fig.text(0.01, 0.012, f"Runs that ran out of time mostly ended frozen at a stump: {note}.",
             fontsize=10, color=INK_2)
    fig.savefig(OUT / "4_where_agents_fall.png", dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}/4_where_agents_fall.png")


if __name__ == "__main__":
    learning_curves()
    ppo_vs_sac()
    a_vs_b_hardcore()
    a_vs_b_curves()
    where_agents_fall()
