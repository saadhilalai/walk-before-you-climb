"""
train.py — trains one run (an algorithm on a course), all seeds side by side, 1 CPU core each.

Usage:
    python train.py --run ppo                 # seeds 42, 142, 242 at the same time (3 cores)
    python train.py --run sac --seeds 42      # just one seed
    python train.py --run sac --smoke --root /tmp/x   # tiny test run, written somewhere else

What happens for each seed:
  1. The robots practise with the exact RL Zoo settings from config.py.
  2. Regular check-ups on 10 fixed curve courses (2024-2033) become the learning curve
     (runs/<run>/seed<N>/curve.csv). A check-up always tests the latest brain: PPO only
     changes its brain once per practice round (65,536 steps), SAC every 64 steps.
  3. At each save point in config.RUNS (e.g. 524,288 steps) the model is saved to
     models/<name>/seed<N>/ (with its normalization stats, if any), loaded back from disk,
     and tested on the 100 validation courses (5000-5099).
  4. runs/<run>/seed<N>/run_info.json records every setting, seed, library version and result.

Every practice episode is logged to runs/<run>/seed<N>/practice_episodes.csv (official score,
outcome, distance), and progress lines go to runs/<run>/seed<N>/train.log.
"""

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import config

SMOKE = dict(saves_ppo=(65_536, 131_072), saves_sac=(12_288,), every=4_096, every_late=32_768,
             curve_episodes=2, validation_episodes=3)
CURVE_FIELDS = ["steps", "train_minutes", "wall_minutes", "steps_per_sec", "mean_score", "finished_rate",
                "fell_rate", "backed_off_rate", "timed_out_rate", "mean_distance"]
EPISODE_FIELDS = ["steps", "score", "length", "outcome", "distance"]


def save_points(run, smoke):
    saves = dict(run["saves"])
    if smoke:
        small = SMOKE["saves_ppo"] if run["algo"] == "ppo" else SMOKE["saves_sac"]
        saves = dict(zip(saves, small))
    return saves


def train_one(run_name, seed, root, smoke):
    """Train one seed of one run. Runs in its own process with 1 CPU thread."""
    config.limit_cpu_threads(1)
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from evaluate import format_summary, load_agent, run_episodes, summarize
    from walker_env import make_env

    run = config.RUNS[run_name]
    hp = config.ALGOS[run["algo"]]
    course = run["course"]
    saves = save_points(run, smoke)
    first_save = min(saves.values())
    every = SMOKE["every"] if smoke else config.CURVE_EVAL_EVERY
    every_late = SMOKE["every_late"] if smoke else config.CURVE_EVAL_EVERY_LATE
    curve_episodes = SMOKE["curve_episodes"] if smoke else config.CURVE_EVAL_EPISODES
    validation_episodes = SMOKE["validation_episodes"] if smoke else config.N_VALIDATION_EPISODES

    run_dir = root / "runs" / run_name / f"seed{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)    # re-running replaces the old run
    run_dir.mkdir(parents=True)
    log = (run_dir / "train.log").open("w", buffering=1)

    def say(msg):
        print(msg, file=log, flush=True)

    # Practice worlds. Monitor sits right on the course, so it records the OFFICIAL score.
    env = DummyVecEnv([lambda: Monitor(make_env(course)) for _ in range(hp["n_envs"])])
    if hp["normalize"]:
        env = VecNormalize(env, gamma=hp["kwargs"]["gamma"])   # as the Zoo does it
    algo = {"ppo": PPO, "sac": SAC}[run["algo"]]
    model = algo(env=env, seed=seed, device="cpu", verbose=0, **hp["kwargs"])
    model.set_logger(configure(str(run_dir), ["csv"]))       # SB3's own training stats: progress.csv

    curve_file = (run_dir / "curve.csv").open("w", newline="", buffering=1)
    curve_csv = csv.DictWriter(curve_file, CURVE_FIELDS)
    curve_csv.writeheader()
    episode_file = (run_dir / "practice_episodes.csv").open("w", newline="")
    episode_csv = csv.DictWriter(episode_file, EPISODE_FIELDS)
    episode_csv.writeheader()

    class Tracker(BaseCallback):
        """Learning-curve check-ups, a log of every practice episode, and timing."""

        def __init__(self):
            super().__init__()
            self.start = time.perf_counter()
            self.train_seconds = 0.0          # practising + learning, check-ups excluded
            self.mark = None
            self.next_checkup = 0
            self.last_checkup = (-1, 0.0)     # (steps, train_seconds) at the previous check-up
            self.curve_env = make_env(course)

        def _on_training_start(self):
            self.mark = time.perf_counter()

        def _on_rollout_start(self):
            if self.num_timesteps >= self.next_checkup:
                self.checkup()

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done and "outcome" in info:
                    episode_csv.writerow(dict(steps=self.num_timesteps, score=round(info["episode"]["r"], 2),
                                              length=info["episode"]["l"], outcome=info["outcome"],
                                              distance=round(info["distance"], 2)))
            return True

        def _on_training_end(self):
            if self.num_timesteps > self.last_checkup[0]:
                self.checkup()                # always test the final brain
            self.pause()

        def pause(self):
            now = time.perf_counter()
            self.train_seconds += now - self.mark
            self.mark = now

        def checkup(self):
            self.pause()
            vecnorm = self.model.get_vec_normalize_env()

            def act(obs):
                if vecnorm is not None:
                    obs = vecnorm.normalize_obs(obs)
                return self.model.predict(obs, deterministic=True)[0]

            s = summarize(run_episodes(act, course, config.CURVE_EVAL_SEED, curve_episodes, env=self.curve_env))
            steps_done = self.num_timesteps - self.last_checkup[0]
            secs = self.train_seconds - self.last_checkup[1]
            row = dict(steps=self.num_timesteps, train_minutes=round(self.train_seconds / 60, 3),
                       wall_minutes=round((time.perf_counter() - self.start) / 60, 3),
                       steps_per_sec=round(steps_done / secs, 1) if self.num_timesteps and secs > 0 else "",
                       **{k: round(s[k], 4) for k in CURVE_FIELDS[4:]})
            curve_csv.writerow(row)
            episode_file.flush()
            say(f"{self.num_timesteps:>9,} steps | {row['train_minutes']:6.1f} min | "
                f"{row['steps_per_sec'] or 0:6.0f} steps/s | {format_summary(s)}")
            self.last_checkup = (self.num_timesteps, self.train_seconds)
            gap = every if self.num_timesteps < first_save else every_late
            self.next_checkup = self.num_timesteps + gap
            self.mark = time.perf_counter()   # check-up time doesn't count as training time

    tracker = Tracker()
    say(f"[{run_name} seed {seed}] {run['algo'].upper()} on the {course} course, saves at "
        f"{', '.join(f'{n}={s:,}' for n, s in saves.items())} steps")

    saved = {}
    for name, target in sorted(saves.items(), key=lambda kv: kv[1]):
        model.learn(target - model.num_timesteps, callback=tracker, reset_num_timesteps=False, progress_bar=False)
        model_dir = root / "models" / name / f"seed{seed}"
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True)
        model.save(model_dir / "model.zip")
        if hp["normalize"]:
            model.get_vec_normalize_env().save(str(model_dir / "vecnormalize.pkl"))
        info = dict(name=name, run=run_name, algo=run["algo"], normalize=hp["normalize"], course=course,
                    train_seed=seed, steps=model.num_timesteps,
                    train_minutes=round(tracker.train_seconds / 60, 2),
                    wall_minutes=round((time.perf_counter() - tracker.start) / 60, 2))
        (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))

        # Validation: load the model back FROM DISK, so the save/load path (and the
        # normalization stats) is exactly what every later test and video will use.
        val = summarize(run_episodes(load_agent(model_dir).act, course, config.VALIDATION_SEED, validation_episodes))
        saved[name] = dict(**info, validation=val)
        say(f"  saved {model_dir.relative_to(root)} at {model.num_timesteps:,} steps "
            f"({info['train_minutes']:.1f} min of training)")
        say(f"  validation ({validation_episodes} courses): {format_summary(val)} -> "
            f"{'SOLVED' if val['solved'] else 'not solved'}")

    curve_file.close()
    episode_file.close()
    run_info = dict(
        run=run_name, train_seed=seed, smoke=smoke, total_steps=model.num_timesteps,
        train_minutes=round(tracker.train_seconds / 60, 2),
        wall_minutes=round((time.perf_counter() - tracker.start) / 60, 2),
        saved_models=saved,
        settings=dict(
            algo=run["algo"], course=course, env_id=config.COURSES[course]["env_id"],
            time_limit=config.COURSES[course]["time_limit"], n_envs=hp["n_envs"], normalize=hp["normalize"],
            hyperparameters=hp["kwargs"], hyperparameter_source="RL Baselines3 Zoo v2.9.1, BipedalWalker-v3",
            save_points=saves, train_seed=seed, curve_eval_seed=config.CURVE_EVAL_SEED,
            curve_eval_episodes=curve_episodes, curve_eval_every=every, curve_eval_every_late=every_late,
            validation_seed=config.VALIDATION_SEED, validation_episodes=validation_episodes,
            solved_score=config.SOLVED_SCORE, cpu_threads=torch.get_num_threads(),
        ),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
                      numpy=__import__("numpy").__version__, machine=f"{platform.system()} {platform.machine()}"),
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    say(f"done: {model.num_timesteps:,} steps in {run_info['train_minutes']:.1f} min of training "
        f"({run_info['wall_minutes']:.1f} min including check-ups)")
    env.close()
    return run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=list(config.RUNS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    parser.add_argument("--root", default=".", help="where runs/ and models/ go (default: this folder)")
    parser.add_argument("--smoke", action="store_true", help="tiny budgets, to test the code")
    args = parser.parse_args()
    assert len(args.seeds) < config.MAX_CORES, "keep at least one core free (shared Mac)"
    root = Path(args.root)

    print(f"[{time.strftime('%H:%M:%S')}] {args.run}: seeds {args.seeds} side by side, 1 core each", flush=True)
    with get_context("spawn").Pool(len(args.seeds)) as pool:
        runs = pool.starmap(train_one, [(args.run, s, root, args.smoke) for s in args.seeds])

    print(f"[{time.strftime('%H:%M:%S')}] done")
    from evaluate import format_summary
    for r in runs:
        for name, m in r["saved_models"].items():
            print(f"  {name:9s} seed {r['train_seed']:>3} | {m['steps']:>9,} steps | {m['train_minutes']:6.1f} min | "
                  f"{format_summary(m['validation'])}")


if __name__ == "__main__":
    sys.exit(main())
