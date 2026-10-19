# Browser simulator experiment (not used by the website)

Before building the website, I tried to run the walker game and the two trained agents live in the browser, the way the LANDFALL site does. The rule for using it was set **before** measuring: the browser had to end the same way as the recorded test run on at least 95% of the 600 hardcore test runs, and land within 1 point of the recorded finish rates (13.3% and 2.0%).

- `walker.js`: Gymnasium 1.4.0's BipedalWalker(-Hardcore) rewritten in JavaScript on [planck.js](https://github.com/piqnt/planck.js) 1.5.0 (MIT, in `vendor/`).
- `rng.js`: NumPy's random-number generator (SeedSequence + PCG64), adapted from LANDFALL's.
- `policy.js`: A and B's trained networks with their observation normalization (`brains.json`, from `python export_web_data.py brains`).
- `../check_browser_sim.mjs`: compares the browser with Python.

**What matched exactly:** every course (terrain, obstacles, clouds: 0 difference) and the brains (the same observation gives the same move, within 0.000001).

**What didn't:** the physics. After the very first move, parts of the robot were already off by about 1% of a leg's length, and the gap grows from there. A likely reason (not tested further): planck.js computes in 64-bit numbers and Box2D in 32-bit, which can tip the physics solver's yes/no decisions differently. Over a 40-second obstacle course that changes the ending often:

| `node check_browser_sim.mjs final` | Result | Rule |
|---|---|---|
| Same outcome as the recorded test run | 305 of 600 = **50.8%** | at least 95% |
| B finished (browser vs recorded) | 18.3% vs 13.3% | within 1 point |
| A finished (browser vs recorded) | 2.0% vs 2.0% | within 1 point |

It failed the rule, so the website replays the recorded test runs instead (`export_web_data.py replays`, each checked against `runs/final/episodes.csv` to the last digit). A likely reason it worked for LANDFALL: a landing lasts about 5 seconds, a walk up to 40.
