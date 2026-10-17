"""
evaluate.py — loads a trained walker and runs it on a fixed list of courses.

Every agent is graded on the OFFICIAL score, on the same list of course seeds,
so they all face exactly the same courses.

A saved model is a folder:  models/<model name>/seed<N>/
    model.zip            the trained network
    vecnormalize.pkl     the observation-normalization stats (only for agents trained with them)
    model_info.json      algorithm, steps, course, and whether normalization is used
load_agent() always loads the stats when the model needs them, and refuses to run without
them, so no test or video can forget them (a classic bug: the network would see numbers on
a completely different scale from the ones it learned on).

Usage:
    python evaluate.py models/sac_524k/seed42                       # 100 validation courses
    python evaluate.py models/ppo_524k/seed42 --episodes 20 --course normal
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

import config
from walker_env import make_env


class Agent:
    """A trained model plus (if it was trained with them) its normalization stats.

    deterministic=True (the default, used for every result so far): the robot always makes
    its "average" move. deterministic=False: it draws its moves at random around that average,
    as it did while practising. The random draws are seeded by the course (start_episode), so a
    randomized run can still be replayed exactly."""

    def __init__(self, model_dir, deterministic=True):
        from stable_baselines3 import PPO, SAC

        self.dir = Path(model_dir)
        self.deterministic = deterministic
        self.info = json.loads((self.dir / "model_info.json").read_text())
        algo = {"ppo": PPO, "sac": SAC}[self.info["algo"]]
        self.model = algo.load(self.dir / "model.zip", device="cpu")
        self.vecnorm = None
        if self.info["normalize"]:
            stats = self.dir / "vecnormalize.pkl"
            if not stats.exists():
                raise FileNotFoundError(f"{stats} is missing: this agent can't be tested without its "
                                        "normalization stats")
            with stats.open("rb") as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False        # never update the stats while testing

    def start_episode(self, course_seed):
        if not self.deterministic:
            import torch
            torch.manual_seed(course_seed)

    def act(self, obs):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def load_agent(model_dir, deterministic=True):
    return Agent(model_dir, deterministic)


def run_episodes(act, course, first_seed, n_episodes, env=None, start_episode=None):
    """Run `n_episodes` courses: course i uses seed `first_seed + i`. act(obs) -> action.
    start_episode(course_seed), if given, is called at the start of each course."""
    own_env = env is None
    env = env or make_env(course)
    episodes = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=first_seed + i)
        if start_episode:
            start_episode(first_seed + i)
        score, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs))
            score += float(reward)
            steps += 1
            done = terminated or truncated
        episodes.append(dict(course_seed=first_seed + i, score=score, outcome=info["outcome"],
                             distance=info["distance"], steps=steps, end_x=info["end_x"], fell_at=info["fell_at"]))
    if own_env:
        env.close()
    return episodes


def summarize(episodes):
    """Boil a list of episodes down to the headline numbers."""
    scores = np.array([e["score"] for e in episodes])
    outcomes = Counter(e["outcome"] for e in episodes)
    n = len(episodes)
    summary = dict(episodes=n, mean_score=float(scores.mean()), std_score=float(scores.std()),
                   solved=bool(scores.mean() >= config.SOLVED_SCORE))
    for o in config.OUTCOMES:
        summary[f"{o}_rate"] = outcomes[o] / n
    summary["mean_distance"] = float(np.mean([e["distance"] for e in episodes]))
    summary["mean_steps"] = float(np.mean([e["steps"] for e in episodes]))
    return summary


def format_summary(s):
    return (f"score {s['mean_score']:6.1f} ± {s['std_score']:5.1f} | finished {s['finished_rate']:6.1%} | "
            f"fell {s['fell_rate']:6.1%} | backed off {s['backed_off_rate']:5.1%} | "
            f"timed out {s['timed_out_rate']:6.1%} | distance {s['mean_distance']:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir")
    parser.add_argument("--course", default="normal", choices=list(config.COURSES))
    parser.add_argument("--episodes", type=int, default=config.N_VALIDATION_EPISODES)
    parser.add_argument("--seed", type=int, default=config.VALIDATION_SEED)
    args = parser.parse_args()
    assert not config.touches_test_courses(args.seed, args.episodes), "test courses are for the final test only"
    config.limit_cpu_threads(1)
    agent = load_agent(args.model_dir)
    print(format_summary(summarize(run_episodes(agent.act, args.course, args.seed, args.episodes))))


if __name__ == "__main__":
    main()
