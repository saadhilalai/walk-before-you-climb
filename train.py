"""
train.py — trains one run, all seeds side by side, 1 CPU core each.

Usage:
    python train.py --run ppo                 # Step 2: seeds 42, 142, 242 at the same time (3 cores)
    python train.py --run sac --seeds 42      # just one seed
    python train.py --run b --phase 1         # Step 3: B's normal phase
    python train.py --run b --phase 2         # ...then B's hardcore phase (continues from B's 5M save)
    python train.py --run a                   # Step 3: A, hardcore from scratch
    python train.py --run b --smoke --root /tmp/x   # tiny test run, written somewhere else

What happens for each seed:
  1. The robots practise with the exact RL Zoo settings named in config.RUNS. Settings
     that shrink ("lin_" in the Zoo) shrink over the WHOLE run, across save points and phases.
  2. Regular check-ups on fixed curve courses (2024 onwards) become the learning curve
     (runs/<run>/seed<N>/curve.csv, one row per check-up and course). A check-up always
     tests the latest brain: PPO only changes its brain once per practice round.
  3. At each save point the model is saved to models/<name>/seed<N>/ (with its
     normalization stats, if any), loaded back from disk, and tested on the 100
     validation courses (5000-5099) of the phase's course.
  4. A later phase starts from the previous phase's last saved model, loaded from disk:
     same brain, same optimizer memory, same normalization stats (which keep updating),
     same step counter. Only the course changes.
  5. runs/<run>/seed<N>/run_info.json records every setting, seed, library version and result.

Every practice episode is logged to runs/<run>/seed<N>/practice_episodes.csv.gz (official score,
outcome, distance; Step 2's runs wrote an uncompressed .csv), and progress lines go to
runs/<run>/seed<N>/train.log.
"""

import argparse
import csv
import gzip
import json
import platform
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import config

SMOKE = dict(sac_round=12_288, every=4_096, every_late=32_768, checkup_courses=2, validation_episodes=3)
CURVE_FIELDS = ["steps", "course", "train_minutes", "wall_minutes", "steps_per_sec", "mean_score", "finished_rate",
                "fell_rate", "backed_off_rate", "timed_out_rate", "mean_distance"]
EPISODE_FIELDS = ["steps", "course", "score", "length", "outcome", "distance"]


def phase_saves(run, smoke):
    """Save points of every phase, [{name: steps}, ...]. In a smoke test the k-th save of the
    run comes after k practice rounds (PPO) or k x 12,288 steps (SAC)."""
    phases = [dict(p["saves"]) for p in run["phases"]]
    if smoke:
        hp = run["hp"]
        size = hp["n_envs"] * hp["kwargs"]["n_steps"] if run["algo"] == "ppo" else SMOKE["sac_round"]
        k = 0
        for saves in phases:
            for name in sorted(saves, key=saves.get):
                k += 1
                saves[name] = k * size
    return phases


def resolve(kwargs):
    """The Zoo's ("lin", x) -> a straight line from x down to 0 (SB3's LinearSchedule)."""
    from stable_baselines3.common.utils import LinearSchedule
    return {k: LinearSchedule(v[1], 0.0, 1.0) if isinstance(v, tuple) and v[0] == "lin" else v
            for k, v in kwargs.items()}


def open_log(path, mode):
    return gzip.open(path, mode + "t", newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def trim_log(path, last_step):
    """Keep only the rows logged up to `last_step` (re-running a later phase replaces it)."""
    with open_log(path, "r") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, [r for r in reader if int(r["steps"]) <= last_step]
    with open_log(path, "w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def train_one(run_name, seed, root, smoke, phase):
    """Train one phase of one run for one seed. Runs in its own process with 1 CPU thread."""
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

    class WholeRun:
        """Shrinking settings follow the WHOLE run (1 - steps done / run_total_steps).
        SB3 would otherwise restart them at every learn() call, i.e. at every save point."""
        run_total_steps = None

        def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
            super()._update_current_progress_remaining(num_timesteps, self.run_total_steps or total_timesteps)

    class WholeRunPPO(WholeRun, PPO):
        pass

    class WholeRunSAC(WholeRun, SAC):
        pass

    run = config.RUNS[run_name]
    hp = run["hp"]
    all_saves = phase_saves(run, smoke)
    saves = all_saves[phase - 1]
    run_total = max(all_saves[-1].values())
    this = run["phases"][phase - 1]
    course = this["course"]
    first_save = min(saves.values())
    every = SMOKE["every"] if smoke else this["every"]
    every_late = SMOKE["every_late"] if smoke else this.get("every_late", this["every"])
    checkups = {c: SMOKE["checkup_courses"] if smoke else n for c, n in run["checkups"].items()}
    validation_episodes = SMOKE["validation_episodes"] if smoke else config.N_VALIDATION_EPISODES
    algo = {"ppo": WholeRunPPO, "sac": WholeRunSAC}[run["algo"]]

    run_dir = root / "runs" / run_name / f"seed{seed}"
    if phase == 1:
        shutil.rmtree(run_dir, ignore_errors=True)    # re-running replaces the old run
        run_dir.mkdir(parents=True)
        start_steps, prev_train_s, prev_wall_s = 0, 0.0, 0.0
    else:
        prev_name, start_steps = max(all_saves[phase - 2].items(), key=lambda kv: kv[1])
        prev_dir = root / "models" / prev_name / f"seed{seed}"
        prev_info = json.loads((prev_dir / "model_info.json").read_text())
        prev_train_s, prev_wall_s = prev_info["train_seconds"], prev_info["wall_seconds"]
        trim_log(run_dir / "curve.csv", start_steps)
        trim_log(run_dir / "practice_episodes.csv.gz", start_steps)
    log = (run_dir / "train.log").open("w" if phase == 1 else "a", buffering=1)

    def say(msg):
        print(msg, file=log, flush=True)

    # Practice worlds. Monitor sits right on the course, so it records the OFFICIAL score.
    env = DummyVecEnv([lambda: Monitor(make_env(course)) for _ in range(hp["n_envs"])])
    if phase == 1:
        if hp["normalize"]:
            env = VecNormalize(env, gamma=hp["kwargs"]["gamma"])   # as the Zoo does it
        model = algo(env=env, seed=seed, device="cpu", verbose=0, **resolve(hp["kwargs"]))
        model.run_total_steps = run_total                        # saved with the model
    else:
        # Continue from the previous phase's saved model: brain, optimizer memory, step counter
        # and normalization stats all carry over. Loading re-seeds everything with the run's seed.
        if hp["normalize"]:
            env = VecNormalize.load(str(prev_dir / "vecnormalize.pkl"), env)
            assert env.training and env.norm_obs and env.norm_reward
        model = algo.load(prev_dir / "model.zip", env=env, device="cpu")
        assert model.num_timesteps == start_steps and model.run_total_steps == run_total and model.seed == seed
    log_dir = run_dir if len(run["phases"]) == 1 else run_dir / f"phase{phase}"
    model.set_logger(configure(str(log_dir), ["csv"]))          # SB3's own training stats: progress.csv

    curve_path, episode_path = run_dir / "curve.csv", run_dir / "practice_episodes.csv.gz"
    new_files = phase == 1
    curve_file = curve_path.open("a", newline="", buffering=1)
    curve_csv = csv.DictWriter(curve_file, CURVE_FIELDS)
    episode_file = open_log(episode_path, "a")
    episode_csv = csv.DictWriter(episode_file, EPISODE_FIELDS)
    if new_files:
        curve_csv.writeheader()
        episode_csv.writeheader()

    class Tracker(BaseCallback):
        """Learning-curve check-ups, a log of every practice episode, and timing."""

        def __init__(self):
            super().__init__()
            self.start = time.perf_counter() - prev_wall_s
            self.train_seconds = prev_train_s   # practising + learning, check-ups excluded
            self.mark = None
            if start_steps == 0:                # check the untrained brain too
                self.last_checkup, self.next_checkup = (-1, 0.0), 0
            else:                               # the previous phase already checked this exact brain
                self.last_checkup = (start_steps, prev_train_s)
                self.next_checkup = (start_steps // every + 1) * every
            self.curve_envs = {c: make_env(c) for c in checkups}

        def _on_training_start(self):
            self.mark = time.perf_counter()

        def _on_rollout_start(self):
            if self.num_timesteps >= self.next_checkup:
                self.checkup()

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done and "outcome" in info:
                    episode_csv.writerow(dict(steps=self.num_timesteps, course=course,
                                              score=round(info["episode"]["r"], 2), length=info["episode"]["l"],
                                              outcome=info["outcome"], distance=round(info["distance"], 2)))
            return True

        def _on_training_end(self):
            if self.num_timesteps > self.last_checkup[0]:
                self.checkup()                  # always test the final brain
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

            steps_done = self.num_timesteps - self.last_checkup[0]
            secs = self.train_seconds - self.last_checkup[1]
            base = dict(steps=self.num_timesteps, train_minutes=round(self.train_seconds / 60, 3),
                        wall_minutes=round((time.perf_counter() - self.start) / 60, 3),
                        steps_per_sec=round(steps_done / secs, 1) if self.last_checkup[0] >= 0 and secs > 0 else "")
            progress = self.model._current_progress_remaining   # what the latest brain update used
            rates = f"lr {self.model.lr_schedule(progress):.3g}"
            if run["algo"] == "ppo":
                rates += f", clip {self.model.clip_range(progress):.3g}"
            say(f"{self.num_timesteps:>11,} steps | {base['train_minutes']:7.1f} min | "
                f"{base['steps_per_sec'] or 0:6.0f} steps/s | {rates}")
            for c, n in checkups.items():
                s = summarize(run_episodes(act, c, config.CURVE_EVAL_SEED, n, env=self.curve_envs[c]))
                curve_csv.writerow(dict(base, course=c, **{k: round(s[k], 4) for k in CURVE_FIELDS[5:]}))
                say(f"    {c:8s} ({n} courses): {format_summary(s)}")
            episode_file.flush()
            self.last_checkup = (self.num_timesteps, self.train_seconds)
            gap = every if self.num_timesteps < first_save else every_late
            self.next_checkup = (self.num_timesteps // gap + 1) * gap
            self.mark = time.perf_counter()     # check-up time doesn't count as training time

    tracker = Tracker()
    say(f"[{run_name} seed {seed}] phase {phase} of {len(run['phases'])}: {run['algo'].upper()} "
        f"({hp['zoo_entry']} settings) on the {course} course, from {model.num_timesteps:,} steps; saves at "
        f"{', '.join(f'{n}={s:,}' for n, s in saves.items())}; whole run {run_total:,} steps")

    saved = {}
    for name, target in sorted(saves.items(), key=lambda kv: kv[1]):
        model.learn(target - model.num_timesteps, callback=tracker, reset_num_timesteps=False, progress_bar=False)
        model_dir = root / "models" / name / f"seed{seed}"
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True)
        model.save(model_dir / "model.zip")
        if hp["normalize"]:
            model.get_vec_normalize_env().save(str(model_dir / "vecnormalize.pkl"))
        wall_s = time.perf_counter() - tracker.start
        info = dict(name=name, run=run_name, phase=phase, algo=run["algo"], normalize=hp["normalize"], course=course,
                    train_seed=seed, steps=model.num_timesteps,
                    train_minutes=round(tracker.train_seconds / 60, 2), wall_minutes=round(wall_s / 60, 2),
                    train_seconds=round(tracker.train_seconds, 3), wall_seconds=round(wall_s, 3))
        (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))

        # Validation: load the model back FROM DISK, so the save/load path (and the
        # normalization stats) is exactly what every later test and video will use.
        val = summarize(run_episodes(load_agent(model_dir).act, course, config.VALIDATION_SEED, validation_episodes))
        saved[name] = dict(**info, validation=val)
        say(f"  saved {model_dir.relative_to(root)} at {model.num_timesteps:,} steps "
            f"({info['train_minutes']:.1f} min of training)")
        say(f"  validation ({validation_episodes} {course} courses): {format_summary(val)} -> "
            f"{'SOLVED' if val['solved'] else 'not solved'}")

    curve_file.close()
    episode_file.close()
    info_path = run_dir / "run_info.json"
    earlier = json.loads(info_path.read_text()) if phase > 1 else {}
    earlier_names = {n for p in all_saves[:phase - 1] for n in p}
    run_info = dict(
        run=run_name, train_seed=seed, smoke=smoke, phases_done=phase, total_steps=model.num_timesteps,
        train_minutes=round(tracker.train_seconds / 60, 2),
        wall_minutes=round((time.perf_counter() - tracker.start) / 60, 2),
        saved_models={**{n: m for n, m in earlier.get("saved_models", {}).items() if n in earlier_names}, **saved},
        settings=dict(
            algo=run["algo"], n_envs=hp["n_envs"], normalize=hp["normalize"], hyperparameters=hp["kwargs"],
            hyperparameter_source=f"RL Baselines3 Zoo v2.9.1, {hp['zoo_entry']}",
            schedules='["lin", x] = straight line from x down to 0 over the whole run (run_total_steps)',
            run_total_steps=run_total, train_seed=seed,
            phases=[dict(course=p["course"], env_id=config.COURSES[p["course"]]["env_id"],
                         time_limit=config.COURSES[p["course"]]["time_limit"], saves=s,
                         checkup_every=SMOKE["every"] if smoke else p["every"],
                         checkup_every_late=SMOKE["every_late"] if smoke else p.get("every_late", p["every"]))
                    for p, s in zip(run["phases"], all_saves)],
            checkups=checkups, curve_eval_seed=config.CURVE_EVAL_SEED,
            validation_seed=config.VALIDATION_SEED, validation_episodes=validation_episodes,
            solved_score=config.SOLVED_SCORE, cpu_threads=torch.get_num_threads(),
        ),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
                      numpy=__import__("numpy").__version__, machine=f"{platform.system()} {platform.machine()}"),
    )
    info_path.write_text(json.dumps(run_info, indent=2))
    say(f"phase {phase} done: {model.num_timesteps:,} steps in {run_info['train_minutes']:.1f} min of training "
        f"({run_info['wall_minutes']:.1f} min including check-ups)")
    env.close()
    return run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=list(config.RUNS), required=True)
    parser.add_argument("--phase", type=int, default=1, help="which phase of the run (B has 2)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    parser.add_argument("--root", default=".", help="where runs/ and models/ go (default: this folder)")
    parser.add_argument("--smoke", action="store_true", help="tiny budgets, to test the code")
    args = parser.parse_args()
    assert len(args.seeds) < config.MAX_CORES, "keep at least one core free (shared Mac)"
    assert 1 <= args.phase <= len(config.RUNS[args.run]["phases"])
    root = Path(args.root)

    print(f"[{time.strftime('%H:%M:%S')}] {args.run} phase {args.phase}: seeds {args.seeds} side by side, "
          f"1 core each", flush=True)
    with get_context("spawn").Pool(len(args.seeds)) as pool:
        runs = pool.starmap(train_one, [(args.run, s, root, args.smoke, args.phase) for s in args.seeds])

    print(f"[{time.strftime('%H:%M:%S')}] done")
    from evaluate import format_summary
    for r in runs:
        for name, m in r["saved_models"].items():
            print(f"  {name:9s} seed {r['train_seed']:>3} | {m['course']:8s} | {m['steps']:>11,} steps | "
                  f"{m['train_minutes']:6.1f} min | {format_summary(m['validation'])}")


if __name__ == "__main__":
    sys.exit(main())
