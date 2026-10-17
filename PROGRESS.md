# Robot Walker: Progress Notes

_Last updated 2026-10-17, 00:15. Step 3 is paused: B's safety check failed for 2 of 3 seeds. Waiting for the director's decision (Step 3, batch 0)._

| Step | Status |
|---|---|
| 1. Setup, speed test, time budget | ✅ Done |
| 2. First walker: PPO vs SAC (normal course) | ✅ Done (PPO picked for Step 3) |
| 3. Hardcore: from scratch (A) vs curriculum (B) | 🔄 Training |
| 4. Checkpoint review | Not started |
| 5. Final test (once) | Not started |
| 6. Charts and videos | Not started |
| 7. Package it | Not started |

---

## How to pick up again

```bash
cd ~/robot-walker
.venv/bin/python check_setup.py          # Step 1 sanity check (~10 s)
```

On a fresh clone (no `.venv`):
```bash
brew install python@3.11 swig ffmpeg
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Files

| File | What it does |
|---|---|
| `config.py` | **Every** setting: courses, seeds, outcome names, CPU limit, PPO and SAC settings (from RL Zoo) |
| `walker_env.py` | Builds a course and judges each episode: finished / fell / backed_off / timed_out, distance, and which obstacle it fell at |
| `check_setup.py` | Step 1 sanity check (random robot, rendering, obstacles, reproducibility) |
| `benchmark_speed.py` | Step 1 speed test. Results in `runs/setup/benchmark.jsonl` |
| `evaluate.py` | Loads a saved model (always with its normalization stats, if it has them) and runs it on a list of courses |
| `train.py` | Trains a run (`ppo` or `sac`), all 3 seeds side by side on 1 core each. Saves models, curves, logs, `run_info.json` |
| `compare.py` | Tests models on the same analysis courses and runs the paired tests. Results in `runs/compare/<name>.md/.json/_episodes.csv` |
| `make_charts.py` | Draws the charts in `charts/` from saved results (never re-runs anything) |
| `gate.py` | Step 3 safety check between B's two phases (B moves on only if every seed finishes 90%+ of normal validation courses) |
| `models/<name>/seed<N>/` | A trained model: `model.zip`, `vecnormalize.pkl` (PPO only) and `model_info.json` |
| `runs/<run>/seed<N>/` | `train.log`, `curve.csv` (learning curve), `practice_episodes.csv.gz` (every practice episode; Step 2: `.csv`), `progress.csv` (SB3's own stats; B has one per phase), `run_info.json` |
| `runs/setup/` | Step 1 outputs: `check_setup.json`, rendered frames, `courses_overview.png`, `benchmark.jsonl` |

---

## Rules that keep the comparison fair

- **3 training seeds per agent: 42, 142, 242.**
- **Separate sets of courses** (course seed `s` always builds the exact same course):
  - setup, 0–99: Step 1 checks only
  - learning-curve check-ups, 2024+
  - validation, 5000–5099: used during training and to pick settings
  - analysis, 20000+: comparing agents along the way
  - **test, 10000–10099: hidden until the final test, used exactly once**
- **Graded on the official score**, plus finished / fell / timed-out rates and distance.
- **Observation normalization stats are saved with the model** and loaded for every test and video (PPO uses them; SAC doesn't).
- **At most 4 CPU cores** in total on this shared Mac.

---

## Step 1: Setup (done 2026-10-16)

### Software
- Homebrew Python 3.11.17, `.venv` in this folder. All versions pinned in `requirements.txt`. They match the rocket project, which ran fine on this Mac yesterday.
- **Hardcore course:** Gymnasium 1.4.0 has **both** `BipedalWalkerHardcore-v3` and `BipedalWalker-v3` with `hardcore=True`. They're the same course. We use the `BipedalWalkerHardcore-v3` name.
- **Time limits differ:** the normal course allows **1,600 steps** (32 seconds of simulated time), hardcore allows **2,000** (40 s).
- **Settings source:** RL Baselines3 Zoo **v2.9.1**, files `hyperparams/ppo.yml` and `hyperparams/sac.yml`, entries `BipedalWalker-v3`. They're copied exactly into `config.py`.
  - The Zoo's own training lengths, for reference:

    | Course | PPO | SAC |
    |---|---|---|
    | Normal | 5M steps | 500k steps |
    | Hardcore | **100M** steps | **10M** steps |

  - The Zoo also has separate, different settings for hardcore. Which ones to use is a Step 3 decision.

### Sanity checks (`check_setup.py`, setup courses 0–99 only)
- **Rendering works:** 600×400 frames, saved in `runs/setup/frame_*.png`.
- **The hardcore course really has obstacles.** Over 100 courses: 10 to 19 per course, 13.4 on average.
  - The three kinds are equally common: 440 stumps, 456 pits, and 442 staircases (243 up, 199 down).
  - The normal course has none on any of the 100 courses. `runs/setup/courses_overview.png` shows a side view of each.
- **Random robot (20 courses each; numbers updated after the Step 2 physics fix below):**

  | Course | Score | Fell | Timed out | Distance |
  |---|---|---|---|---|
  | Normal | −105.9 | 16 | 4 | 1.2% |
  | Hardcore | −111.6 | 15 | 5 | 1.6% |

- **Gymnasium's built-in hand-written walking controller is NOT a good walker.**
  - Normal course: it finished only **2 of 20** courses and fell on 16. It got 51% of the way on average, with a score of 78.
  - Hardcore: 12.8% of the way, and it fell on 19 of 20.
  - So it's a reference point, not proof that the course can be finished. (Published RL results show it can be.)
- **Same course seed + same moves → exactly the same course and score, no matter which courses ran before**, so paired comparisons work. (The "no matter what ran before" part was only true after the Step 2 fix below.)

### Bug found and fixed while building the obstacle detector
- The simulator doesn't keep a list of obstacles, so `walker_env.course_obstacles()` works them out from the block shapes.
- My first version counted **every stair step as its own staircase**, and found no down-staircases at all.
- The cause: the simulator stores the blocks **finish-to-start**, so neighbouring steps never looked adjacent.
- After sorting the blocks by position, the counts are balanced (see above), and the side-view picture matches the rendered frame.

### Speed test (`benchmark_speed.py`, normal course, exact Zoo settings)
The Mac was not idle: background apps (video call / screen encoder) used roughly 1.5 cores the whole time.

| Setup | PPO steps/sec | SAC steps/sec |
|---|---|---|
| Raw simulator, random moves, 1 core | 9,800 (hardcore: 9,600) | |
| 1 run, 1 core | 4,209 | 202 |
| 1 run, 4 cores | 4,014 (no gain) | 177 (slower) |
| 1 run, Mac GPU ("mps") | | 61 (3× slower) |
| **3 runs side by side, 1 core each** | **3,340 each = 10,030 total** | **94 each = 281 total** |

- **PPO's network is tiny** (2 layers of 64), so extra cores don't help one run. Running 3 seeds side by side gets 2.4× the total throughput.
- **SAC does 1 learning update per practice step** on a bigger network (400 + 300), so it's about **20× slower per step** than PPO.
  - 3 SAC runs side by side only get 1.4× the total. Likely reason (untested): they compete for the chip's shared matrix-maths unit and memory.
- **Heat:** within 3 minutes, the 3 side-by-side SAC runs slowed from ~98 to ~90 steps/sec.
  - The estimates below **add 25% for heat** on long runs. That's an assumption; the first long run will measure the real number.

### Time budget (3 seeds side by side = 3 cores, +25% for heat)

| Step | What | Mac estimate |
|---|---|---|
| 2 | SAC, 500k steps × 3 seeds (the Zoo's SAC budget) | ~1 h 50 min |
| 2 | PPO, 500k steps × 3 seeds (same steps as SAC) | ~3 min |
| 2 | *Optional:* PPO continued to its Zoo budget of 5M × 3 seeds | ~30 min |
| 2 | *Smaller alternative:* both at 300k steps | ~1 h 10 min total |
| 3 | If SAC wins: A + B, 6 runs × 500k / 1M / 2M steps | ~3.7 h / ~7.4 h / ~15 h |
| 3 | If PPO wins: A + B, 6 runs × 10M / 20M steps | ~2.1 h / ~4.2 h |
| 3 | Zoo hardcore budgets (SAC 10M, PPO 100M) | ~74 h / ~21 h: not realistic on the Mac |
| 4 | Analysis-course tests + a few videos to watch | ~20–30 min |
| 5 | Final test | ~15 min |
| 6 | Charts + videos (rendering) | ~30 min |
| 7 | Writing only | none |

- **Step 2 fits on the Mac** (about 2 hours).
- **Step 3 doesn't fit in 2 hours on the Mac** at any budget likely to teach hardcore. It's the step that may belong on the gaming PC.
  - To estimate it there, run `benchmark_speed.py` on the PC (~10 min).
- **Honest expectation:** the Zoo trains hardcore for 10M (SAC) or 100M (PPO) steps. At a budget we can afford, **neither agent is likely to "solve" hardcore** (300+).
  - The A-vs-B comparison is still valid (same steps, same seeds, paired test).
  - It answers "which is better *with a small practice budget*", and the README must say so.

### What the robot sees, controls, and how it's scored
- **Sees: 24 numbers, 50 times a second.** No position or map: it doesn't know where it is on the course.
  - **Body (4):** tilt, how fast it's tipping, forward speed, up/down speed.
  - **Each leg (5 × 2):** hip angle, hip speed, knee angle, knee speed, and "is this foot touching the ground" (1 or 0).
  - **Lidar (10):** like 10 laser pointers fanning out from the body, from straight down (0°) to 77° forward, reaching about 2.5 body-lengths.
    - Each reports how far along the beam it hits something: 1.0 means nothing within reach, 0.3 means something close.
    - This is the **only way it can "see" a stump, pit or stair** coming.
- **Controls: 4 motors (leg 1 hip, leg 1 knee, leg 2 hip, leg 2 knee),** each a number from −1 to +1.
  - The game is a side view, so the legs are "leg 1" and "leg 2", not left and right.
  - The sign picks the direction the joint turns. The size picks how hard it pushes (up to full motor strength).
- **Score (the official reward), added up every step:**
  - **+ for moving forward:** start to finish is worth about **+364** in total. Walking backwards takes points away.
  - **− for tilting** the body (keep the head level).
  - **− a small cost for motor effort:** all 4 motors at full power cost 0.11 per step, so a whole run of max effort would cost about −180.
  - **−100 if the body touches the ground.** The episode ends right there.
  - A clean, efficient walk finishes at about **300–330**. **Solved = average 300+ on the normal course.**
- **Possible trap to watch for:**
  - Falling costs −100, but standing still costs almost nothing.
  - A robot can learn to "freeze" safely until time runs out (a timed-out episode) instead of risking a fall.
  - This is a known local optimum, and it's why we report timed-out rates, not just scores.

### Decisions made after Step 1 (2026-10-16)
1. **3 seeds side by side** (3 cores) count as one job. OK.
2. **"Backed off"** (walking backwards off the start, the game's second −100 case) stays its own outcome.
3. **Step 2 budget:** 500k steps for both algorithms, plus PPO continued to its Zoo budget of 5M.
4. **The whole project stays on the MacBook.** Time doesn't matter: for Step 3, pick the budget that gives the best, clearest result, even overnight. Before Step 3, show the options again with Mac times measured from Step 2's real speed and heat.

---

## Step 2: PPO vs SAC on the normal course (done 2026-10-16)

### Setup choices (made before training)
- **"500k" = 524,288 steps for both algorithms.** PPO can only stop at the end of a full practice round (32 robots × 2,048 = 65,536 steps), and 8 rounds = 524,288. SAC gets exactly the same number, so the budget is equal.
- **PPO's 5M model is the same 3 runs continued.** Each PPO run saves `ppo_524k` at 524,288 steps and keeps going to `ppo_5m` (5,046,272 steps = 77 rounds).
- **Learning-curve check-ups:** 10 fixed curve courses (2024–2033).
  - SAC: every 16,384 steps.
  - PPO: after every brain update (every 65,536 steps; testing more often would re-test the same brain), and every 131,072 steps after 524,288.
- **Validation:** at every save point, the model is **loaded back from disk** (with its normalization stats) and run on the 100 validation courses (5000–5099). Solved = average 300+.
- **Model kept:** the final model at each save point (no picking the best check-up).

### Bug found and fixed before training: results depended on what ran before
- **Symptom:** a model loaded from disk scored −111.5395 on two curve courses, but the same model's in-training check-up said −111.5638.
- **Cause:** not the model. The game reuses one Box2D physics world for every episode. Leftover internal bookkeeping from earlier episodes slightly changes later ones. Walking is chaotic, so tiny differences grow.
  - The hand-written walker scored differently on **6 of 10 courses** just by running them in reverse order.
- **Fix (`walker_env.py`):** every episode starts in a **brand-new physics world**, exactly like a freshly created simulator. The game's rules are unchanged.
- **Checks after the fix:**
  - Same scores in order, reversed, and on a fresh simulator. This is now a permanent check in `check_setup.py`.
  - In-training check-ups and loaded-from-disk tests match exactly.
  - Two SAC runs with the same seed give identical learning curves and practice episodes.
- **Why it matters:** the Step 6 videos must replay a test run and match its result exactly, and paired tests need "same course = same conditions".
- Step 1's sanity-check numbers were re-run with the fix (updated above).

### Wall-clock time (3 seeds side by side, 1 core each)
| Run | Practice steps | Training time per seed | Speed per seed |
|---|---|---|---|
| PPO to 524k | 524,288 | 2.2 min | ~4,000 steps/s |
| PPO to 5M (same runs, continued) | 5,046,272 | 21.0–21.5 min (24 min on the clock, with check-ups) | ~3,900–4,000 steps/s |
| SAC | 524,288 | 58.8–61.6 min (63 min on the clock, with check-ups) | ~142–149 steps/s |

- **PPO practises about 27× more steps per second than SAC.**
- **Heat:** no slowdown over SAC's hour of training (speed bounced between 115 and 180 steps/s, with no downward trend). PPO dipped ~10% around minutes 4–12, then sped back up. The likely reason is the other session's work, not heat. Runs longer than an hour haven't been measured yet.
- SAC ran faster than Step 1's benchmark predicted (145 vs 94 steps/s). The likely reason is the Mac being busier during that benchmark.

### Validation (100 courses, 5000–5099; model loaded from disk)
| Model | Seed 42 | Seed 142 | Seed 242 |
|---|---|---|---|
| SAC 524k | 236.3 (finished 71%) | **312.2 (99%): solved** | 289.5 (88%) |
| PPO 524k | −27.5 (timed out 100%) | −37.9 (timed out 100%) | −24.4 (timed out 100%) |
| PPO 5M | 284.5 (98%) | 297.1 (99%) | 278.0 (99%) |

### Analysis courses (1,000 courses, 20000–20999, × 3 seeds = 3,000 episodes per model)
| Model | Steps | Training min/seed | Score (± across seeds) | Finished | Fell | Backed off | Timed out | Distance | Solved? |
|---|---|---|---|---|---|---|---|---|---|
| SAC 524k | 524,288 | 60 | 282.2 ± 28.7 | 87.3% | 12.7% | 0.0% | 0.0% | 94.2% | 1 of 3 seeds (seed 142: 310.1) |
| PPO 524k | 524,288 | 2 | −30.0 ± 6.4 | 0.0% | 0.0% | 0.0% | 100.0% | 1.4% | no |
| PPO 5M | 5,046,272 | 21 | 287.8 ± 8.7 | 99.1% | 0.9% | 0.0% | 0.0% | 99.7% | no (best seed 298.2) |

Paired tests (same course, same training seed). Full output: `runs/compare/step2.md`.
- **Same practice (SAC 524k vs PPO 524k): SAC is far better. Proven.**
  - SAC finished 2,619 courses that PPO didn't; PPO finished none that SAC didn't (McNemar p < 0.001).
  - Score: +312 per course for SAC (Wilcoxon p < 0.001). Distance: +92.8 points. SAC is ahead in 3 of 3 seeds.
- **PPO with 10× the practice (PPO 5M vs SAC 524k): a split result.**
  - **PPO 5M is more reliable. Proven:** it finished 379 courses that SAC didn't, and SAC finished only 26 that PPO didn't (p < 0.001). It falls on 0.9% of courses vs SAC's 12.7%.
  - **SAC scores higher on a typical course. Proven:** SAC is ahead on 87% of courses (median +20.5 for SAC; Wilcoxon p < 0.001).
    - On the 2,593 courses both finished, SAC scored higher on **every one**, by 24.7 points on average.
    - SAC also crossed the line in 760 steps (15 s of game time) vs PPO's 1,248 (25 s), so **SAC walks ~1.6× faster**.
    - Likely reason for the higher score (not tested): fewer steps means less motor-energy cost, which the score subtracts at every step.
  - **Average score is about equal:** PPO +5.7 overall, because SAC's falls (about −100 each) pull its average down. PPO is ahead in only 1 of 3 seeds (seed 42, SAC's weakest; it had 263 of SAC's 381 falls).
  - PPO 5M used about a third of SAC's clock time (21 vs 60 min per seed).

Chart: `charts/2_ppo_vs_sac.png`. Left: same practice. Right: same clock.

### Why SAC learned more from the same practice (plain English)
- **SAC keeps a memory** of its last 300,000 moments. After every practice step it makes a small brain update from 256 remembered moments. Each moment gets reused many times.
  - In 524,288 steps that's **524,288 small brain updates**.
- **PPO learns only from fresh practice.** 32 robots practise 65,536 steps, PPO makes one careful brain update from them, then **throws them away**.
  - In 524,288 steps that's only **8 brain updates**. That wasn't enough to get past the "stand still" trap (next point).
- **PPO's advantage is speed.** Nothing to re-learn from memory, so each step is cheap: 27× more steps per second. With 10× more steps, it caught up in a third of the time.

### What the learning curves show
- **Both fell into the "freeze" trap first.** Standing still until time runs out scores about −10 to −40, which beats falling (−100).
  - SAC escaped after ~150–260k steps, depending on the seed.
  - PPO was still frozen at 524k (100% timed out on every seed) and escaped at about 1M steps.
- **Odd thing: PPO's "slow walker" phase.** From about 1M to 3M steps, PPO walked forward but too slowly to reach the finish in the 32 s allowed (timed out 100%, while getting 52–96% of the way). We'll watch the gait in Step 4's videos.
- **SAC's check-up scores jump around** (seed 42: between 95 and 317 from one check-up to the next). Likely reason: its brain changes after every step, so a check-up can catch it in a bad moment. PPO's curve is smooth after 3M steps.

### Measured speed for planning Step 3 (2026-10-16, 22:00)
- **Hardcore speed test** (`benchmark_speed.py --hardcore`, the Zoo's hardcore settings on the hardcore course, 3 side by side):
  - SAC: 134–174 steps/s, vs 124–181 for the normal settings in the same test. About 5% slower.
  - PPO: ~4,360, vs ~4,550 for the normal settings. About 4% slower.
- **Applied to Step 2's real speed:** SAC ~138 steps/s per run and PPO ~3,800 per run, plus check-up time (~5% for SAC, ~12% for PPO).
- Step 3 = 6 runs (A and B × 3 seeds), 3 at a time, so **2 batches back to back**:

| Practice steps per agent | SAC: total time | PPO: total time |
|---|---|---|
| 1M | ~4 h | ~10 min |
| 2M | ~8.5 h | ~20 min |
| 5M | ~21 h | ~50 min |
| 10M (Zoo's SAC hardcore budget) | ~42 h | ~1.6 h |
| 25M | | ~4 h |
| 50M | | ~8 h |
| 100M (Zoo's PPO hardcore budget) | | ~16.5 h |

- **Heat risk:** runs over an hour haven't been tested. The time could be ~20% longer if the Mac slows down overnight. I'll watch the speed in the logs.
- **What the Zoo's own trained agents scored on hardcore** ([RL Zoo benchmark](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md), one published agent each):
  - PPO after 100M steps: **122 ± 118**
  - SAC after 10M steps: **4 ± 104**
  - (The Zoo's best is TQC, an upgraded SAC: 235 after 2M. It wasn't in our comparison.)
- **The Zoo's separate hardcore settings** are written down exactly in `config.py` (`PPO_ZOO_HARDCORE`, `SAC_ZOO_HARDCORE`).
  - Both use learning rates that shrink to 0 by the end of training. This has to be measured over the **whole** run, so I'll write that myself: SB3's built-in version restarts each time training is resumed.
  - Which settings Step 3 uses is still to be decided.

---

## Step 3: hardcore, from scratch (A) vs curriculum (B) (in progress)

### Decisions (director, 2026-10-16)
1. **Algorithm: PPO.** SAC learned more from the same practice, but PPO wins on time on this Mac, and the Zoo's own results show SAC barely learns hardcore (4 after 10M steps).
2. **Budget: 100M steps per agent** (~16.5 h). It matches the Zoo's PPO hardcore budget, so our A can be compared with the Zoo's 122 (a rough comparison only: their single agent, different courses).
3. **Settings: the Zoo's hardcore settings (`PPO_ZOO_HARDCORE`), identical for A and B.**
   - 16 robots, gamma 0.99, a small exploration bonus (0.001).
   - The learning rate (2.5e-4) and the clip range (0.2) shrink in a straight line to 0 over the whole run.
4. **B's split: 5M normal, then 95M hardcore.**
   - Exact: 5,013,504 normal + 94,994,432 hardcore = 100,007,936. A gets 100,007,936 hardcore.
   - PPO stops only at the end of a 32,768-step practice round, so every budget is whole rounds.
   - Why 5M: in Step 2, PPO learned to finish normal courses by ~3M steps. 5M adds margin, is the Zoo's normal budget, and costs B only 5% of its hardcore practice.

### How it's run
- **What B keeps at the switch:** everything (brain, optimizer memory, step counter, normalization stats, which keep updating). Phase 2 starts from B's 5M model loaded from disk, so the switch is exactly reproducible.
- **Same start:** A and B with the same seed begin with identical brain weights (checked: all 13 weight tables equal).
- **Check-ups:**
  - Both agents, on 20 hardcore and 10 normal curve courses, every 1,048,576 steps (32 rounds).
  - During B's normal phase: every 262,144 steps.
- **Saves:** 25M, 50M, 75M and 100M (plus B at 5M), each validated on 100 validation courses of its phase's course. We keep the final 100M model.
- **Safety check (`gate.py`):** B moves on to hardcore only if every seed finishes at least 90% of the 100 normal validation courses at 5M. If not: stop and ask.
- **Order:**
  - Batch 0: B's normal phase (~25 min)
  - Safety check
  - Batch 1: B's hardcore phase (~7 h)
  - Batch 2: A (~7.5 h)

### Smoke test (tiny budgets, scratch folder, seed 42): all passed
- The whole chain runs: B phase 1 → safety check → B phase 2, plus A.
  - The safety check correctly **failed** the barely trained smoke model (0% finished) and exits with an error, so a chained command stops there.
- **Exactly reproducible:** B run twice in separate folders gave an identical curve, practice log and final brain.
- **The schedule shrinks over the whole run, not per piece:**
  - Learning rate 2.5e-4 → 2e-4 at the switch → … → 0 at the end, matching the whole-run formula.
  - Before this fix, SB3's own schedule would have restarted at every save point.
- **Re-running phase 2 replaces it cleanly:** identical files to a clean run, no duplicate rows.
- **Step 2's runs still work with the new code,** and give the same check-up numbers as before (e.g. −109.9656 at 65,536 steps).

### Batch 0: B's normal phase (2026-10-16 23:37 to 00:09): safety check FAILED for 2 of 3 seeds
| Seed | Normal validation score | Finished | Fell | Distance | First 10-course check-up with 90%+ finished | Hardcore check-up at 5M |
|---|---|---|---|---|---|---|
| 42 | 301.5 | **99%** (pass) | 1% | 99.8% | 1.8M steps | −80.2, 0% finished, 9.8% of the way |
| 142 | 258.5 | **81%** (fail) | 19% | 91.8% | 1.6M steps | −83.2, 0%, 10.0% |
| 242 | 247.4 | **78%** (fail) | 22% | 89.5% | 1.0M steps | −81.6, 0%, 11.6% |

- **All 3 seeds learned to walk**, and all reached 90%+ on a check-up early (by 1.0–1.8M steps). But their reliability then swung between 40% and 100% from one check-up to the next.
  - Likely reason (not tested): the learning rate is still 95% of its starting value, so the brain keeps changing a lot.
- **On hardcore, a 5M normal walker gets ~10% of the way and falls.** That's expected: it has never seen an obstacle.
- **Speed: 29 min instead of ~21.** It started at ~4,200 steps/s per seed, dropped to ~1,800–2,500 between minutes ~12 and ~25, then recovered to 3,000–4,100.
  - Likely reason: other apps using CPU (the process list showed a video-call / video-encoding service and Chrome; Mac load ~9). Not tested.
- Nothing has been trained since. The director decides how to continue.

