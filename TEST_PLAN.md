# Step 5: Final test plan

Written on 2026-10-17 and **approved by the director on 2026-10-17** (all three new points: the video-only agents, testing Step 2's agents, the video-pick rules). Committed to git before the test runs. Everything
below is fixed in advance. Nothing about the agents, the measurements or the statistics may
change after the test has run.

## 1. The rules
- **Test courses:** seeds 10000–10099, normal and hardcore. They have never been used for training, tuning, checking or any decision.
- **Run once.** `final_eval.py` refuses to run if any final results already exist. It also refuses to run unless every file in the project is committed, so the result is tied to one git commit (recorded in the results).
- **The agents are frozen.** Every model file was committed before this plan. No training, settings, labels or code change after the test.
- **Average moves.** Every agent makes its "average" (deterministic) move, as in all results so far and as in the RL Zoo's benchmark.
  - Director's decision, 2026-10-17. Step 4's randomized-move check is reported on the analysis courses only.
- **Fall labels:** as locked in at Step 4 (`config.FALL_WINDOW_TILES`). A fall belongs to an obstacle if the body was from 2 tiles before it to 3 tiles after it, the nearer one if two qualify; otherwise "open ground".
- **Scoring:** the official reward. For every run we record the outcome (finished / fell / backed off / timed out), the distance (% of the course), the steps, where the body ended up, and the obstacle at a fall.

## 2. What gets tested
Every trained seed (42, 142, 242), 100 test courses each.

| Agents | Course | Runs | Why |
|---|---|---|---|
| A (`a_100m`), B (`b_100m`) | hardcore 10000–10099 | 2 × 3 × 100 = 600 | **The main question:** does the curriculum (B) beat training from scratch (A)? |
| SAC 524k, PPO 524k, PPO 5M (`sac_524k`, `ppo_524k`, `ppo_5m`), A, B | normal 10000–10099 | 5 × 3 × 100 = 1,500 | Step 2's PPO vs SAC; does A or B walk better on plain ground? |
| Untrained PPO (`untrained`, seed 42 only) | normal and hardcore 10000–10099 | 200 | Video only ("untrained collapse"). It's A's and B's exact starting brain, saved before any practice. |
| SAC at 229,376 steps (`sac_229k`, seed 42 only) | normal 10000–10099 | 100 | Video only ("early wobbly steps"). Re-created by re-running Step 2's SAC seed 42 up to that point; it must match Step 2's check-ups exactly. |

The two video-only agents are reported, but they're not part of any comparison.

## 3. Comparisons and statistics, decided now
Every comparison is paired: the same training seed on the same test course, so 300 pairs (3 seeds × 100 courses).
- **Finished (yes/no):** McNemar's exact test, two-sided. Only courses that exactly one of the two agents finished count as evidence.
- **Score and distance:** Wilcoxon signed-rank test on the per-course differences, two-sided (courses with no difference are left out).
- **Per seed:** in how many of the 3 seeds each agent is ahead.
- **"Proven"** means p < 0.05 in the paired test. Otherwise we say "not shown". We never say "proven" for anything else.

**Primary question (the only one the project stands on):**
- **"B finishes hardcore courses more often than A."**
  - **Confirmed** if McNemar p < 0.05 with B ahead.
  - **Not confirmed** if p ≥ 0.05, even though the analysis courses showed it (13.1% vs 2.4%).
  - **Reversed** if A is ahead with p < 0.05.
  - Either way, the README reports exactly that.

**Secondary results (reported with their p-values, clearly marked as secondary):**
- **Hardcore, B vs A:**
  - Score and distance (Wilcoxon). A score result counts as **robust** only if p < 0.05 **and** the same agent is ahead in all 3 seeds. On the analysis courses it was not robust.
  - Fell, timed-out and backed-off rates.
  - Falls per 100 obstacles reached, and where timed-out runs got stuck.
- **Hardcore, A vs the RL Zoo's 122:** A's average score is shown next to it. This is a rough comparison only (their single agent, their courses), with no test.
- **Normal course:**
  - SAC 524k vs PPO 524k (same practice).
  - PPO 5M vs SAC 524k.
  - B vs A.
  - Each with finished (McNemar), score and distance (Wilcoxon), and per seed.
- **Solved** = average 300+ on the normal course, pooled over the 3 seeds and per seed.

**What we expect** (from the analysis courses, written down before the test):
- B finishes hardcore more often than A.
- The B-vs-A score difference may not hold up.
- SAC 524k beats PPO 524k (PPO is still frozen at 524k).
- PPO 5M finishes the normal course more often than SAC 524k, but scores lower on courses both finish.
- A and B are about equal on the normal course.
- No agent solves hardcore.

## 4. Videos (Step 6) replay test runs, picked by fixed rules
Every clip replays a run recorded by this test. The video is only saved if the replay's score and outcome match the recorded ones exactly. The seed-42 model of each agent is used, and the pick is the **first test course in seed order** that matches the rule:

| Clip | Agent (seed 42) | Rule |
|---|---|---|
| Untrained collapse | untrained | normal course 10000 (the first one, whatever happens) |
| Early wobbly steps | sac_229k | first normal course where it gets at least 20% of the way and doesn't finish |
| Trained normal walker | sac_524k | first normal course it finishes |
| Hardcore A vs B, side by side | a_100m and b_100m | first hardcore course that B finishes and A doesn't (labelled on screen as chosen to illustrate the main result, with the overall finish rates shown) |
| A freezes, B gets past (side by side) | a_100m and b_100m | first hardcore course where A times out stuck at a stump and B gets past that stump |
| B-roll 1 | b_100m | first hardcore course it finishes |
| B-roll 2 | a_100m | first hardcore course where it falls at a pit |
| B-roll 3 | b_100m | first hardcore course where it falls at a stump |
| B-roll 4 | ppo_5m | first normal course it finishes |
| B-roll 5 | a_100m | first normal course it finishes |

If no test course matches a rule, the clip is dropped and the README says so. We don't pick a different course.

## 5. Steps, in order
1. **Prepare the two video-only agents:**
   - Save the untrained PPO (seconds).
   - Re-create SAC seed 42 at 229,376 steps (~25 min on 1 core), and check that its learning-curve check-ups match Step 2's exactly.
2. **Dry run** of `final_eval.py` on 10 *analysis* courses per seed, written to a scratch folder, to test the code without touching the test courses.
3. **Commit** this plan, `final_eval.py`, the two new agents and everything else.
4. **Run the final test once:** `python final_eval.py` (~5–10 min, 3 cores). Results go to `runs/final/`: `episodes.csv` (every run), `results.json`, `results.md`, `falls.json`.
5. **Report** in plain English with a simple table, update PROGRESS.md, and commit.

## 6. Preparation, done before this commit
- **Untrained robot** (`make_untrained.py`, `models/untrained/seed42`): on the check-up courses it scores exactly what A scored before any practice (hardcore −92.0965, normal −92.0546), so it is A's exact starting state.
- **Early SAC** (`train.py --run sac_early --seeds 42`, `models/sac_229k/seed42`, 15.4 min of training):
  - All 15 learning-curve check-ups (0 to 229,376 steps) are identical to Step 2's SAC seed 42.
  - Validation: −15.1, 0% finished, fell 78%, 33% of the way.
- **Dry run** (`final_eval.py --dry-run`, 10 analysis courses per seed, scratch folder): all 24 model files ran, and all outputs and tests were produced. The test courses were not touched.

