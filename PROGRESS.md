# Robot Walker: Progress Notes

_Last updated 2026-10-16. Nothing is running._

| Step | Status |
|---|---|
| 1. Setup, speed test, time budget | ✅ Done (waiting for the director's decisions below) |
| 2. First walker: PPO vs SAC (normal course) | Not started |
| 3. Hardcore: from scratch (A) vs curriculum (B) | Not started |
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
- **Random robot (20 courses each):**

  | Course | Score | Fell | Timed out | Distance |
  |---|---|---|---|---|
  | Normal | −99.3 | 12 | 8 | 1.3% |
  | Hardcore | −109.9 | 17 | 3 | 1.4% |

- **Gymnasium's built-in hand-written walking controller is NOT a good walker.**
  - Normal course: it fell on **all 20** courses. It got 41.5% of the way on average, with a score of 27.
  - Hardcore: 10.9% of the way.
  - So it's a reference point, not proof that the course can be finished. (Published RL results show it can be.)
- **Same course seed + same moves → exactly the same course and score**, so paired comparisons work.

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

### Decisions needed before Step 2
1. Is running **3 seeds side by side (3 cores) as one job** OK? It's 2.4× faster for PPO and 1.4× for SAC than one seed at a time.
2. What to call the game's second −100 case (**walking backwards off the start of the course**): its own outcome "backed off", or count it as "fell"?
3. **Step 2 budget:** 500k steps each (~2 h) or 300k (~1 h 10 min)? Optionally, PPO continued to 5M (+30 min)?
4. **Gaming PC:** what CPU does it have? This decides whether Step 3 moves there.
