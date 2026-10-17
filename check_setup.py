"""
check_setup.py — Step 1 sanity check.

Proves that:
  1. Both courses load (normal and hardcore) and a random robot can run on them.
  2. Rendering works (frames saved to runs/setup/).
  3. The hardcore course really has obstacles, and the normal one has none.
  4. The same course seed always builds the same course (needed for fair, paired tests).
  5. The courses are walkable: Gymnasium's built-in hand-written controller is run on both.
  6. WalkerStats labels every episode (finished / fell / backed_off / timed_out) and its distance.

Uses only the setup courses (seeds 0-99), never the validation/analysis/test ones.

Run it with:  .venv/bin/python check_setup.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from walker_env import FINISH_X, START_X, course_obstacles, make_env  # noqa: E402

OUT = Path("runs") / "setup"
OBS_NAMES = (
    ["body tilt", "body spin speed", "forward speed", "up/down speed"]
    + [f"leg {n}: {p}" for n in (1, 2) for p in ("hip angle", "hip speed", "knee angle", "knee speed",
                                                 "foot touching ground")]
    + [f"lidar {i} ({round(np.degrees(1.5 * i / 10))} deg from straight down)" for i in range(10)]
)


def run_episode(env, seed, policy):
    """One episode on course `seed`. policy(obs) -> action."""
    obs, _ = env.reset(seed=seed)
    score, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        score += float(reward)
        steps += 1
        done = terminated or truncated
    return dict(seed=seed, score=round(score, 1), steps=steps, outcome=info["outcome"],
                distance=round(info["distance"], 1), fell_at=info["fell_at"])


def random_policy(env):
    return lambda obs: env.action_space.sample()


def heuristic_policy():
    h = BipedalWalkerHeuristics()   # fresh walking-cycle state for each episode
    return lambda obs: h.step_heuristic(obs)


def summary(episodes):
    return dict(mean_score=round(float(np.mean([e["score"] for e in episodes])), 1),
                outcomes=dict(Counter(e["outcome"] for e in episodes)),
                mean_distance=round(float(np.mean([e["distance"] for e in episodes])), 1))


def draw_courses(seed):
    """Side view of the whole normal and hardcore course for one seed, obstacles coloured by type."""
    colors = {"stump": "#e34948", "pit": "#2a78d6", "stairs_up": "#1baf7a", "stairs_down": "#eda100"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for ax, course in zip(axes, ("normal", "hardcore")):
        env = make_env(course)
        env.reset(seed=seed)
        game = env.unwrapped
        ax.plot(game.terrain_x, game.terrain_y, color="#52514e", lw=1)
        for body in game.terrain:
            shape = body.fixtures[0].shape
            if hasattr(shape, "vertices") and len(shape.vertices) == 4:
                ax.fill(*zip(*shape.vertices), color="#d0cfc8", lw=0)
        for ob in env.obstacles:
            ax.axvspan(ob["x_start"], ob["x_end"], color=colors[ob["kind"]], alpha=0.35, lw=0,
                       label=ob["kind"].replace("_", " "))
        ax.axvline(START_X, color="#6250d6", ls="--", lw=1)
        ax.axvline(FINISH_X, color="#0b0b0b", ls="--", lw=1)
        ax.set_title(f"{course} course, seed {seed}: {len(env.obstacles)} obstacles "
                     f"(dashed: start position and finish line)", loc="left", fontsize=10)
        ax.set_ylim(0, 10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", ncols=4, fontsize=8, frameon=False)
        env.close()
    axes[1].set_xlabel("position along the course (simulator units)")
    fig.tight_layout()
    fig.savefig(OUT / "courses_overview.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    # --- 1. What the robot sees and controls --------------------------------
    env = make_env("normal", render_mode="rgb_array")
    print(f"Observation: {env.observation_space.shape[0]} numbers")
    for i, name in enumerate(OBS_NAMES):
        print(f"  [{i:2d}] {name}")
    print(f"Action: {env.action_space.shape[0]} numbers in [{env.action_space.low[0]:.0f}, "
          f"{env.action_space.high[0]:.0f}]: leg 1 hip, leg 1 knee, leg 2 hip, leg 2 knee\n")

    # --- 2. Rendering --------------------------------------------------------
    for course in ("normal", "hardcore"):
        e = make_env(course, render_mode="rgb_array")
        e.reset(seed=config.SETUP_SEED)
        frame = e.render()
        Image.fromarray(frame).save(OUT / f"frame_{course}.png")
        print(f"Rendered a {course} frame of size {frame.shape} -> {OUT}/frame_{course}.png")
        e.close()
    env.close()

    # --- 3. Random robot and hand-written controller on both courses ----------
    seeds = range(config.SETUP_SEED, config.SETUP_SEED + 20)
    for course in ("normal", "hardcore"):
        env = make_env(course)
        env.action_space.seed(0)
        rnd = [run_episode(env, s, random_policy(env)) for s in seeds]
        heur = [run_episode(env, s, heuristic_policy()) for s in seeds]
        report[course] = dict(random=summary(rnd), heuristic=summary(heur),
                              random_example=rnd[0], heuristic_example=heur[0])
        print(f"\n{course} course, 20 setup courses:")
        print(f"  random robot:          {summary(rnd)}")
        print(f"  hand-written walker:   {summary(heur)}")
        if course == "hardcore":
            print(f"  where the hand-written walker fell: "
                  f"{dict(Counter(e['fell_at'] for e in heur if e['fell_at']))}")
        env.close()

    # --- 4. Obstacles: count them on 100 courses of each kind ---------------
    print()
    for course in ("normal", "hardcore"):
        env = make_env(course)
        counts, per_course = Counter(), []
        for s in range(config.SETUP_SEED, config.SETUP_SEED + 100):
            env.reset(seed=s)
            obs_list = course_obstacles(env.unwrapped)
            per_course.append(len(obs_list))
            counts.update(o["kind"] for o in obs_list)
        report[course]["obstacles_100_courses"] = dict(per_course_min=min(per_course),
                                                       per_course_mean=float(np.mean(per_course)),
                                                       per_course_max=max(per_course), by_kind=dict(counts))
        print(f"{course}: obstacles per course min {min(per_course)}, mean {np.mean(per_course):.1f}, "
              f"max {max(per_course)} | by kind over 100 courses: {dict(counts)}")
        env.close()
    assert report["normal"]["obstacles_100_courses"]["per_course_max"] == 0, "normal course has obstacles?!"
    assert report["hardcore"]["obstacles_100_courses"]["per_course_min"] > 0, "hardcore course without obstacles?!"
    draw_courses(config.SETUP_SEED)
    print(f"Saved a side view of both courses -> {OUT}/courses_overview.png")

    # --- 5. Same seed -> same course, same result ----------------------------
    env = make_env("hardcore")
    env.action_space.seed(123)
    actions = [env.action_space.sample() for _ in range(300)]
    results = []
    for _ in range(2):
        env.reset(seed=7)
        terrain = list(env.unwrapped.terrain_y)
        total = 0.0
        for a in actions:
            _, r, term, trunc, _ = env.step(a)
            total += float(r)
            if term or trunc:
                break
        results.append((terrain, total))
    same = bool(results[0][0] == results[1][0] and results[0][1] == results[1][1])
    report["reproducible"] = same
    print(f"Same course seed + same moves -> identical course and score: {same}")
    env.close()
    assert same

    # --- 5b. ...no matter which courses ran before (fresh physics world per episode) ---
    seeds = list(range(config.SETUP_SEED, config.SETUP_SEED + 10))
    env = make_env("normal")
    forward = {s: run_episode(env, s, heuristic_policy())["score"] for s in seeds}
    backward = {s: run_episode(env, s, heuristic_policy())["score"] for s in reversed(seeds)}
    fresh = {s: run_episode(make_env("normal"), s, heuristic_policy())["score"] for s in seeds}
    order_free = forward == backward == fresh
    report["history_independent"] = order_free
    print(f"Hand-written walker on 10 courses, in order / reversed / fresh simulator each time -> "
          f"identical scores: {order_free}")
    env.close()
    assert order_free

    (OUT / "check_setup.json").write_text(json.dumps(report, indent=2))
    print(f"\nAll checks passed. Saved {OUT}/check_setup.json")


if __name__ == "__main__":
    main()
