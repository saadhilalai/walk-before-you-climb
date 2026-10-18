"""
config.py — every setting used to train and test the walkers, in one place.

All agents of the same kind use EXACTLY these settings, so any difference in
results comes from the idea being tested, not from luck or different knobs.
"""

import os

# ---- The two courses ----------------------------------------------------------
# Gymnasium 1.4.0 registers both names; BipedalWalkerHardcore-v3 is simply
# BipedalWalker-v3 with hardcore=True and a longer time limit.
COURSES = {
    "normal":   dict(env_id="BipedalWalker-v3",         time_limit=1600),  # gently uneven grass
    "hardcore": dict(env_id="BipedalWalkerHardcore-v3", time_limit=2000),  # + stumps, pits, stairs
}

# ---- Random seeds (fixed, so every run can be reproduced) -----------------------
# A training seed sets the network's starting weights and the practice courses.
# The same seed gives every agent type the same starting point, so comparisons are paired.
TRAIN_SEEDS = (42, 142, 242)

# Course seeds. Course seed s always builds the exact same course (terrain + starting push).
SETUP_SEED = 0            # courses 0-99: Step 1 sanity checks only
CURVE_EVAL_SEED = 2_024   # the quick check-ups drawn on the learning curve (a hardcore course with seed s
                          # is a different course from the normal one with seed s)
VALIDATION_SEED = 5_000   # courses 5000-5099: used during training and to pick settings
ANALYSIS_SEED = 20_000    # courses 20000+: comparing agents along the way
TEST_SEED = 10_000        # courses 10000-10099: HIDDEN until the final test, used exactly once
N_VALIDATION_EPISODES = 100
N_TEST_EPISODES = 100


def touches_test_courses(first_seed, n_episodes):
    """True if courses first_seed .. first_seed + n_episodes - 1 overlap the hidden test courses."""
    return first_seed < TEST_SEED + N_TEST_EPISODES and first_seed + n_episodes > TEST_SEED

# ---- Definitions (from the project brief) ---------------------------------------
SOLVED_SCORE = 300        # official mark: average score of 300+ on the normal course
# Outcome of an episode (see walker_env.WalkerStats):
#   finished    the body crossed the finish line
#   fell        the body touched the ground (-100 penalty)
#   backed_off  walked backwards off the start of the course (the game's other -100 case)
#   timed_out   neither, before the step limit
OUTCOMES = ("finished", "fell", "backed_off", "timed_out")

# Where a fall happened on the hardcore course (decided in Step 4, before the final test).
# A fall belongs to an obstacle if the robot's body centre was from 2 tiles before it to 3 tiles
# after it: about one body length, so falls while landing after a stump or pit count too.
# If two obstacles qualify, the nearer one wins (the earlier one on a tie). Otherwise "open ground".
FALL_WINDOW_TILES = dict(before=2, after=3)

# ---- CPU budget (shared MacBook Air: at most 4 cores in total) --------------------
MAX_CORES = 4


def limit_cpu_threads(n):
    """Cap how many CPU threads this process's maths libraries may use."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    import torch
    torch.set_num_threads(n)


# ---- Algorithm settings ---------------------------------------------------------
# Starting point: RL Baselines3 Zoo v2.9.1 (rl_zoo3/hyperparams/ppo.yml and sac.yml),
# entries "BipedalWalker-v3". Copied exactly; the Zoo's own step budget is noted but
# NOT used (each step of this project sets its own budget).

# PPO (Proximal Policy Optimization): learns only from fresh practice, then throws it away.
PPO_ZOO = dict(
    n_envs=32,            # 32 robots practise side by side
    normalize=True,       # VecNormalize: rescale observations and rewards (stats saved with the model)
    zoo_n_timesteps=5_000_000,
    kwargs=dict(
        policy="MlpPolicy",   # default network: 2 layers of 64 (actor and critic separately)
        n_steps=2048,         # each robot plays 2048 steps before every learning update
        batch_size=64,        # learn from 64 moments at a time
        gae_lambda=0.95,      # smoothing for "was this action better than expected?"
        gamma=0.999,          # how much future rewards matter (0.999 = very far-sighted)
        n_epochs=10,          # go over each round of experience 10 times
        ent_coef=0.0,         # no extra bonus for random exploration
        learning_rate=3e-4,   # how big each brain update is
        clip_range=0.18,      # PPO's safety limit: never change the policy too much at once
    ),
)

# SAC (Soft Actor-Critic): keeps a memory of past experience and keeps re-learning from it.
SAC_ZOO = dict(
    n_envs=1,
    normalize=False,
    zoo_n_timesteps=500_000,
    kwargs=dict(
        policy="MlpPolicy",
        learning_rate=7.3e-4,
        buffer_size=300_000,     # remembers the last 300k moments
        batch_size=256,          # learns from 256 remembered moments at a time
        ent_coef="auto",         # automatically balances trying new things vs. exploiting
        gamma=0.98,
        tau=0.02,                # how fast the "target" critic follows the main one
        train_freq=64,           # every 64 steps...
        gradient_steps=64,       # ...do 64 learning updates
        learning_starts=10_000,  # first 10k steps: random moves, just to fill the memory
        use_sde=True,            # gSDE: smoother, more consistent exploration noise
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),  # 2 layers: 400 and 300
    ),
)

# The Zoo's separate settings for "BipedalWalkerHardcore-v3", copied exactly.
# Step 3 uses PPO_ZOO_HARDCORE for BOTH agents (director's decision, 2026-10-16).
# ("lin", x) is the Zoo's "lin_x": start at x and shrink in a straight line to 0 by the end
# of training. train.py measures it over the WHOLE run (SB3's own version restarts every
# time training is resumed, e.g. at a save point or when B switches courses).
PPO_ZOO_HARDCORE = dict(
    n_envs=16, normalize=True, zoo_n_timesteps=100_000_000,
    kwargs=dict(policy="MlpPolicy", n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10,
                ent_coef=0.001, learning_rate=("lin", 2.5e-4), clip_range=("lin", 0.2)),
)
SAC_ZOO_HARDCORE = dict(
    n_envs=1, normalize=False, zoo_n_timesteps=10_000_000,
    kwargs=dict(policy="MlpPolicy", learning_rate=("lin", 7.3e-4), buffer_size=1_000_000, batch_size=256,
                ent_coef=0.005, gamma=0.99, tau=0.01, train_freq=1, gradient_steps=1, learning_starts=10_000,
                policy_kwargs=dict(net_arch=[400, 300])),
)
PPO_ZOO["zoo_entry"] = SAC_ZOO["zoo_entry"] = "BipedalWalker-v3"
PPO_ZOO_HARDCORE["zoo_entry"] = SAC_ZOO_HARDCORE["zoo_entry"] = "BipedalWalkerHardcore-v3"


# ---- Training runs ----------------------------------------------------------------
# A run trains one algorithm with one set of settings. It has one or more phases, each on
# one course; every phase saves a model at each listed step count (counted from the start
# of the run). Check-ups happen every `every` steps (`every_late` after the phase's first
# save) on the run's `checkups` courses: {course: number of curve courses}.

# Step 2. 524,288 = 8 full PPO practice rounds (32 robots x 2,048 steps). PPO can only stop
# at the end of a round, so "500k" is rounded up to this for BOTH algorithms.
STEP2_STEPS = 524_288
PPO_LONG_STEPS = 5_000_000   # PPO continued to its Zoo budget (it stops at the end of a round: 5,046,272)

# Step 3. PPO's practice round with the hardcore settings is 16 robots x 2,048 = 32,768 steps,
# so every budget is rounded up to whole rounds. A and B get exactly the same total.
STEP3_ROUND = PPO_ZOO_HARDCORE["n_envs"] * PPO_ZOO_HARDCORE["kwargs"]["n_steps"]


def whole_rounds(steps, round_size=STEP3_ROUND):
    return -(-steps // round_size) * round_size


STEP3_TOTAL_STEPS = whole_rounds(100_000_000)   # 100,007,936 for both A and B
B_NORMAL_STEPS = whole_rounds(5_000_000)        # 5,013,504: B's normal phase (counts toward its total)
STEP3_CHECKUPS = {"hardcore": 20, "normal": 10}  # both agents are checked on both courses
STEP3_EVERY = 1_048_576                         # check-up every 32 rounds (~1M steps)
B_NORMAL_EVERY = 262_144                        # more often while B learns to walk (8 rounds)


def step3_saves(prefix):
    saves = {f"{prefix}_{m}m": whole_rounds(m * 1_000_000) for m in (25, 50, 75)}
    saves[f"{prefix}_100m"] = STEP3_TOTAL_STEPS
    return saves


RUNS = {
    # Step 2: PPO vs SAC on the normal course
    "ppo": dict(algo="ppo", hp=PPO_ZOO, checkups={"normal": 10}, phases=[
        dict(course="normal", every=16_384, every_late=131_072,
             saves={"ppo_524k": STEP2_STEPS, "ppo_5m": PPO_LONG_STEPS})]),
    "sac": dict(algo="sac", hp=SAC_ZOO, checkups={"normal": 10}, phases=[
        dict(course="normal", every=16_384, every_late=131_072, saves={"sac_524k": STEP2_STEPS})]),
    # Step 5 (video only): Step 2's SAC seed 42 re-created up to 229,376 steps, for the "early
    # wobbly steps" clip. Same settings and check-up schedule, so its curve must match Step 2's exactly.
    "sac_early": dict(algo="sac", hp=SAC_ZOO, checkups={"normal": 10}, phases=[
        dict(course="normal", every=16_384, every_late=131_072, saves={"sac_229k": 229_376})]),
    # Step 3: hardcore. A = from scratch, B = curriculum (normal first, then hardcore).
    "a": dict(algo="ppo", hp=PPO_ZOO_HARDCORE, checkups=STEP3_CHECKUPS, phases=[
        dict(course="hardcore", every=STEP3_EVERY, saves=step3_saves("a"))]),
    "b": dict(algo="ppo", hp=PPO_ZOO_HARDCORE, checkups=STEP3_CHECKUPS, phases=[
        dict(course="normal", every=B_NORMAL_EVERY, saves={"b_5m": B_NORMAL_STEPS}),
        dict(course="hardcore", every=STEP3_EVERY, saves=step3_saves("b"))]),
}

# Safety check between B's two phases (agreed before training): B only moves on to hardcore
# if every seed finishes at least 90% of the 100 normal validation courses at 5M steps.
B_GATE = dict(run="b", model="b_5m", min_finished=0.90)

# ---- Comparisons ------------------------------------------------------------------------
N_ANALYSIS_EPISODES = 1_000   # analysis courses per trained seed
