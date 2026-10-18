# Robot Walker: interview questions, answered plainly

**1. What did you build, in one breath?**
I trained a simulated two-legged robot to walk with reinforcement learning, then tested one idea: if it learns to walk on flat ground first (a curriculum), does it handle an obstacle course better than a robot that practises on obstacles from the start, with exactly the same total practice? On 300 hidden test runs each, the curriculum robot finished the obstacle course 13.3% of the time vs 2.0% (p < 0.001), but its score advantage wasn't statistically solid.

**2. Why compare PPO and SAC first, and why did you pick PPO when SAC won?**
SAC won the fair fight at equal practice: after 524k steps it finished 85% of flat test courses, while PPO was still standing still. SAC re-learns from a memory of old moves, so it squeezes more out of each step.

But my real limit was a laptop CPU, and PPO runs 27× more steps per second. Given 10× the steps, which still took a third of SAC's time, PPO finished 99% of courses. The RL Zoo's own results also showed SAC barely learning the hardcore course even at 10M steps. So "best per step" and "best per hour" pointed to different algorithms, and per hour mattered more for a 100M-step experiment.

**3. How did you keep the curriculum comparison fair?**
- **Same total practice:** both agents got exactly 100,007,936 steps, and B's flat-ground steps counted toward its total.
- **Same everything else:** settings, training seeds, and starting brain (I checked the weights were identical at step 0).
- **Paired testing:** both were tested on the same courses, with paired statistical tests.
- **Hidden test, fixed in advance:** the test courses were never used until the end. The test plan, including the verdict rule and my predictions, was committed to git before the one-shot test.

**4. What exactly did the final test show?**
- **Finishing (confirmed):** B finished 39 courses that A didn't, and A finished only 5 that B didn't. McNemar's exact test gives p < 0.001, which met the rule set in advance.
- **Distance (proven):** B got 15 points further along the course on average.
- **Score (not proven):** p = 0.09, and B actually scored lower in 2 of the 3 seeds.
- **Caveat:** most of the finishing lead came from one training seed (31% vs 0%). In another seed, A finished slightly more. So the honest summary is "it clearly helped these robots", not "it always helps".

**5. Why not call the score difference proven? B's average was almost three times A's.**
Averages can be dragged around by a few big wins. On a typical course they were even: on the test, B scored higher on 48% of courses and A on 52%. The paired test gave p = 0.09, and the direction flipped from seed to seed. I wrote down before the test that a score result only counts as robust if p < 0.05 *and* the same agent wins in all three seeds. It met neither condition.

**6. What's the "freeze" behaviour, and what does it tell you?**
- **What happens:** when a robot got stuck, it was almost always at a stump. It stands still until time runs out: in 60 checked runs per agent, the joints were completely motionless for the last 10 seconds in 56 (A) and 53 (B).
- **Why:** falling costs −100, and standing still costs almost nothing, so freezing is a "safe" habit the robot found during practice.
- **Not a testing artefact:** testing with randomized moves didn't break the freeze; it just turned some freezes into falls.
- **The lesson:** a learner optimises exactly the reward you give it, and a safe-but-useless habit can be a stable local optimum. The same trap showed up in the very first experiment, when PPO stood still on flat ground.

**7. A safety check you set yourself failed. What did you do?**
Before training, I required every curriculum seed to finish at least 90% of flat validation courses before switching to obstacles. Two of three seeds came in at 81% and 78%.

I looked at the data: all three had clearly learned to walk (each had hit 90%+ earlier) and were swinging up and down from check-up to check-up. So I decided to continue, and wrote the failure, the table and my reasoning into the project notes and the README. The point of a rule set in advance isn't that it can never be overridden. It's that any override is visible and explained, not quietly forgotten.

**8. What bugs did you find?**
- **The physics engine had memory:** it reused its world between episodes, so a run's result depended on which courses ran before. The same hand-written walker scored differently on 6 of 10 courses just by running them in reverse order. That would have broken both the paired tests and the video replays, so every episode now starts in a fresh physics world.
- **The learning-rate schedule restarted:** Stable-Baselines3's shrinking learning rate silently restarts whenever training resumes, which would have broken the curriculum agent's schedule. I made it shrink over the whole run.
- **The "stuck at" label misfired:** after the test, a video titled "A freezes at the stump" showed A stuck in a pit. The label had used the robot's furthest point rather than where it ended up. I fixed the label, kept the recorded results unchanged, and reported both numbers (145 vs 139 of 188).

**9. Why are your agents so far below the RL Zoo's 122?**
Honestly, I don't know, and I haven't tested why. Their 122 is a single trained agent on their own courses, so it's a rough comparison. My three from-scratch seeds alone ranged from −17 to 30, so seed-to-seed luck is large. The most likely candidates would be seed variation and small differences in setup. I'd want more seeds before claiming anything.

**10. What would you do next?**
- **More training seeds** (5–10), to tell whether the curriculum's finishing gain is reliable or mostly one lucky seed.
- **Attack the freeze directly,** for example by penalising long stretches without progress during practice, while still scoring with the official reward.
- **Try a gradual curriculum,** where obstacles fade in rather than switching all at once.
- **Try a stronger algorithm** like TQC, which the Zoo reports at 235 after only 2M hardcore steps, on a GPU.
