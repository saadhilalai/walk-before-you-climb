"""
make_untrained.py — saves the untrained robot used in the "untrained collapse" video (Step 5 plan).

It is A's exact starting state: PPO with the Zoo's hardcore settings and training seed 42, built
the same way train.py builds run "a", including the normalization stats after the first look at
the 16 practice courses (what A's step-0 check-up used). Its brain weights are also B's starting
weights (B's normalization stats started from normal courses instead).

Check: it must reproduce A's step-0 check-up scores exactly (runs/a/seed42/curve.csv).

Usage:
    python make_untrained.py
Output: models/untrained/seed42/ (model.zip, vecnormalize.pkl, model_info.json)
"""

import csv
import json
import shutil
from pathlib import Path

import config

config.limit_cpu_threads(1)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from evaluate import load_agent, run_episodes, summarize  # noqa: E402
from train import resolve  # noqa: E402
from walker_env import make_env  # noqa: E402

SEED = 42


def main():
    run = config.RUNS["a"]
    hp = run["hp"]
    course = run["phases"][0]["course"]
    env = DummyVecEnv([lambda: Monitor(make_env(course)) for _ in range(hp["n_envs"])])
    env = VecNormalize(env, gamma=hp["kwargs"]["gamma"])
    model = PPO(env=env, seed=SEED, device="cpu", verbose=0, **resolve(hp["kwargs"]))
    env.reset()                       # learn() does this first: the stats see the 16 starting positions

    out = Path("models") / "untrained" / f"seed{SEED}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    model.save(out / "model.zip")
    env.save(str(out / "vecnormalize.pkl"))
    (out / "model_info.json").write_text(json.dumps(dict(
        name="untrained", run="untrained", algo="ppo", normalize=True, course=None, train_seed=SEED, steps=0,
        train_minutes=0.0, wall_minutes=0.0, train_seconds=0.0, wall_seconds=0.0,
        note="A's exact starting state (Zoo hardcore PPO settings, seed 42), never trained"), indent=2))

    agent = load_agent(out)
    rows = [r for r in csv.DictReader(open(Path("runs") / "a" / f"seed{SEED}" / "curve.csv")) if r["steps"] == "0"]
    ok = True
    for r in rows:
        c, n = r["course"], config.STEP3_CHECKUPS[r["course"]]
        s = summarize(run_episodes(agent.act, c, config.CURVE_EVAL_SEED, n))
        same = round(s["mean_score"], 4) == float(r["mean_score"])
        ok &= same
        print(f"{c}: untrained robot {s['mean_score']:.4f} vs A's step-0 check-up {float(r['mean_score']):.4f} -> "
              f"{'identical' if same else 'DIFFERENT'}")
    print(f"Saved {out}" if ok else "MISMATCH: this is not A's starting state")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
