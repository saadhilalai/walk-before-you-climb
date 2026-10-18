# Final test: courses 10000-10099, average moves, git commit 012ffcf

| Agent | Course | Score (± across seeds) | Finished | Fell | Backed off | Timed out | Distance | Solved? |
|---|---|---|---|---|---|---|---|---|
| a_100m | hardcore | 13.1 ± 21.2 | 2.0% | 35.3% | 0.0% | 62.7% | 29.1% | no |
| b_100m | hardcore | 37.4 ± 40.0 | 13.3% | 45.7% | 0.0% | 41.0% | 44.4% | no |
| sac_524k | normal | 274.9 ± 34.4 | 85.3% | 14.7% | 0.0% | 0.0% | 92.6% | no |
| ppo_524k | normal | -30.6 ± 6.5 | 0.0% | 0.0% | 0.0% | 100.0% | 1.4% | no |
| ppo_5m | normal | 288.4 ± 9.3 | 99.3% | 0.3% | 0.0% | 0.3% | 99.7% | no |
| a_100m | normal | 260.8 ± 31.0 | 89.7% | 9.0% | 0.0% | 1.3% | 95.2% | no |
| b_100m | normal | 269.1 ± 17.1 | 90.7% | 9.3% | 0.0% | 0.0% | 97.2% | no |
| untrained (video only) | normal | -92.1 ± 0.0 | 0.0% | 100.0% | 0.0% | 0.0% | 2.8% | no |
| untrained (video only) | hardcore | -92.1 ± 0.0 | 0.0% | 100.0% | 0.0% | 0.0% | 2.8% | no |
| sac_229k (video only) | normal | 2.6 ± 0.0 | 0.0% | 65.0% | 0.0% | 35.0% | 35.2% | no |

Paired tests (same training seed, same course):

- **PRIMARY** **b_100m vs a_100m / hardcore**: finished only by b_100m 39, only by a_100m 5 (p < 0.001); b_100m finishes more in 2 of 3 seeds; score +24.2 mean, -4.8 median (p = 0.093; per seed -11.7, -25.6, +110.0; not robust); distance +15.3 points (p < 0.001) -> **CONFIRMED**
- **sac_524k vs ppo_524k / normal**: finished only by sac_524k 256, only by ppo_524k 0 (p < 0.001); sac_524k finishes more in 3 of 3 seeds; score +305.5 mean, +344.3 median (p < 0.001; per seed +255.5, +347.3, +313.7; robust); distance +91.2 points (p < 0.001)
- **ppo_5m vs sac_524k / normal**: finished only by ppo_5m 44, only by sac_524k 2 (p < 0.001); ppo_5m finishes more in 3 of 3 seeds; score +13.5 mean, -19.5 median (p < 0.001; per seed +61.3, -8.2, -12.7; not robust); distance +7.0 points (p < 0.001)
- **b_100m vs a_100m / normal**: finished only by b_100m 26, only by a_100m 23 (p = 0.775); b_100m finishes more in 1 of 3 seeds; score +8.3 mean, +2.6 median (p < 0.001; per seed -9.6, +4.2, +30.1; not robust); distance +2.0 points (p = 0.069)

A on hardcore: 13.1 vs the RL Zoo's 122 (rough comparison, no test).

Where they fall on hardcore (falls per 100 obstacles reached):

- a_100m: stump 6.8, pit 20.5, stairs_up 1.0, stairs_down 4.8; open ground 10 falls; timed out stuck at stump 145, pit 31, stairs_down 4, stairs_up 8
- b_100m: stump 9.4, pit 9.5, stairs_up 3.6, stairs_down 2.0; open ground 22 falls; timed out stuck at stump 120, pit 3
