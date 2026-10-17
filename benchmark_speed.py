"""
benchmark_speed.py — Step 1: how many practice steps per second can we train at?

Measures PPO and SAC with the exact RL Zoo settings from config.py on the normal
course. Every result is appended to runs/setup/benchmark.jsonl.

Usage:
    python benchmark_speed.py env                          # raw simulator speed, no learning
    python benchmark_speed.py ppo --threads 1              # one PPO run on 1 CPU core
    python benchmark_speed.py sac --threads 4              # one SAC run on 4 CPU cores
    python benchmark_speed.py sac --threads 1 --tag par3   # (started 3 times at once = 3 seeds side by side)
    python benchmark_speed.py sac --hardcore               # the Zoo's hardcore settings on the hardcore course

PPO is timed over 2 full practice-then-learn rounds (normal: 2 x 65,536 steps; hardcore: 2 x 32,768).
SAC is timed AFTER its first 10,000 random warm-up steps, when it learns on every step.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import config

OUT = Path("runs") / "setup" / "benchmark.jsonl"


def save(result):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.update(machine=f"{platform.system()} {platform.machine()}", time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with OUT.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def bench_env(steps=20_000):
    import numpy as np
    from walker_env import make_env
    for course in ("normal", "hardcore"):
        env = make_env(course)
        env.action_space.seed(0)
        env.reset(seed=config.SETUP_SEED)
        start = time.perf_counter()
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
        sps = steps / (time.perf_counter() - start)
        save(dict(test="env", course=course, steps_per_sec=round(sps)))
        env.close()


def settings(hardcore, normal_hp, hardcore_hp):
    """(course, settings) for a benchmark. A ("lin", x) schedule is timed at its start
    value x: the value changes the learning, not the speed."""
    if not hardcore:
        return "normal", normal_hp
    kwargs = {k: v[1] if isinstance(v, tuple) else v for k, v in hardcore_hp["kwargs"].items()}
    return "hardcore", dict(hardcore_hp, kwargs=kwargs)


def bench_ppo(threads, tag, rounds=2, hardcore=False):
    config.limit_cpu_threads(threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from walker_env import make_env

    class Timer(BaseCallback):
        def _on_training_start(self):
            self.t, self.rollout, self.update = time.perf_counter(), 0.0, 0.0

        def _on_rollout_start(self):
            now = time.perf_counter()
            self.update += now - self.t       # time since last rollout ended = learning update
            self.t = now

        def _on_rollout_end(self):
            now = time.perf_counter()
            self.rollout += now - self.t      # practising in the simulators
            self.t = now

        def _on_step(self):
            return True

    course, hp = settings(hardcore, config.PPO_ZOO, config.PPO_ZOO_HARDCORE)
    env = DummyVecEnv([lambda: make_env(course) for _ in range(hp["n_envs"])])
    env = VecNormalize(env, gamma=hp["kwargs"]["gamma"])
    model = PPO(env=env, seed=42, device="cpu", verbose=0, **hp["kwargs"])
    steps = rounds * hp["n_envs"] * hp["kwargs"]["n_steps"]
    timer = Timer()
    start = time.perf_counter()
    model.learn(steps, callback=timer)
    total = time.perf_counter() - start
    timer.update += time.perf_counter() - timer.t
    save(dict(test="ppo", course=course, tag=tag, threads=threads, steps=steps, seconds=round(total, 1),
              steps_per_sec=round(steps / total), practice_seconds=round(timer.rollout, 1),
              learning_seconds=round(timer.update, 1)))


def bench_sac(threads, tag, device, steps_after_warmup, hardcore=False):
    config.limit_cpu_threads(threads)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from walker_env import make_env

    course, hp = settings(hardcore, config.SAC_ZOO, config.SAC_ZOO_HARDCORE)
    warmup = hp["kwargs"]["learning_starts"]

    class Timer(BaseCallback):
        marks = {}

        def _on_step(self):
            if self.num_timesteps in (warmup, warmup + steps_after_warmup // 2, warmup + steps_after_warmup):
                self.marks[self.num_timesteps] = time.perf_counter()
            return True

    model = SAC(env=make_env(course), seed=42, device=device, verbose=0, **hp["kwargs"])
    timer = Timer()
    model.learn(warmup + steps_after_warmup, callback=timer)
    m = [timer.marks[k] for k in sorted(timer.marks)]
    half = steps_after_warmup // 2
    save(dict(test="sac", course=course, tag=tag, threads=threads, device=device, steps=steps_after_warmup,
              seconds=round(m[2] - m[0], 1), steps_per_sec=round(steps_after_warmup / (m[2] - m[0]), 1),
              first_half_sps=round(half / (m[1] - m[0]), 1), second_half_sps=round(half / (m[2] - m[1]), 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["env", "ppo", "sac"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tag", default="single")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sac-steps", type=int, default=4_096)
    parser.add_argument("--ppo-rounds", type=int, default=2)
    parser.add_argument("--hardcore", action="store_true",
                        help="the Zoo's hardcore settings on the hardcore course (added before Step 3)")
    args = parser.parse_args()
    assert args.threads <= config.MAX_CORES
    if args.what == "env":
        bench_env()
    elif args.what == "ppo":
        bench_ppo(args.threads, args.tag, args.ppo_rounds, args.hardcore)
    else:
        bench_sac(args.threads, args.tag, args.device, args.sac_steps, args.hardcore)


if __name__ == "__main__":
    main()
