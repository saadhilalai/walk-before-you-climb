## step3: hardcore course, 1,000 analysis courses x 3 seeds = 3,000 episodes per model, average (deterministic) moves

| Model | Steps | Training time (min, per seed) | Score (± across seeds) | Finished | Fell | Backed off | Timed out | Distance | Solved (300+)? |
|---|---|---|---|---|---|---|---|---|---|
| a_100m | 100,007,936 | 380 | 15.4 ± 24.3 | 2.4% | 34.7% | 0.0% | 62.9% | 29.4% | no |
| b_100m | 100,007,936 | 474 | 42.4 ± 30.4 | 13.1% | 48.2% | 0.0% | 38.7% | 45.8% | no |

Per seed (score / finished):

- a_100m: seed 42: 44.3 / 4.1%, seed 142: 16.8 / 2.5%, seed 242: -15.1 / 0.5%
- b_100m: seed 42: 24.3 / 6.9%, seed 142: 17.7 / 7.6%, seed 242: 85.2 / 24.7%

Paired tests (same course, same training seed):

- **b_100m vs a_100m**: finished only by b_100m 364, only by a_100m 43 (p < 0.001); score difference +27.0 mean, +1.7 median (p < 0.001); distance +16.4 points (p < 0.001); higher score in 2 of 3 seeds
