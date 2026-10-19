# Hyperframes Composition Brief: Walk Before You Climb (v2)

## Output
- **Composition:** `brag-output/composition/`
- **Video:** `brag-output/brag.mp4`
  - Authored at 1920×1080 and rendered at 3840×2160 (`--resolution landscape-4k --quality delivery --fps 30 --workers 2`).
- **Duration:** 39.30 s (1,179 frames).

## Source material
| Clip (`videos/final`, clean) | Window used | Role |
|---|---|---|
| `untrained_normal_clean.mp4` | 0.00–3.28 s, then a hold on its last frame | collapse |
| `sac_229k_normal_clean.mp4` | 0.40–5.35 s | wobble |
| `sac_524k_normal_clean.mp4` | 4.00–10.40 s | walk; grows to full screen |
| `a_freezes_b_gets_past_clean.mp4` | 1.20–6.90 s, then a hold on frame 345 | A vs B |

For A vs B, the left panel is cropped as A and the right panel as B. Panel bounds in the source are x 34–1901 / 1938–3805 and y 422–1667.

## Copy (verbatim)
- It couldn't stand.
- 0 practice steps / It falls.
- 229,376 practice steps / It wobbles.
- 524,288 practice steps / It walks.
- Then we gave it obstacles.
- A Obstacles from the start · B Learned to walk first
- Stopped at the stump. / Climbed over it.
- One example test run (course 10016), not the typical result.
- Learned to walk first · 6× · more courses finished
- 13.3% vs 2.0% · tested once on courses it had never seen
- Same practice for both. Most of the gap came from 1 of 3 training runs.
- Walk Before You Climb · Saad Hilal · S4D · github.com/saadhilalai/walk-before-you-climb

## Visual identity
- **Colours:**
  - background: flat `#0b0b0c` (a soft radial lift to `#141417` banded after compression, so it was dropped)
  - ink `#f3f2ee`, secondary `#9b9a95`, footnote `#7f7e7a`
  - one accent `#c6c0ff`, used only for "learned to walk first"
- **Fonts:** Geist and Geist Mono variable woff2 (SIL OFL 1.1, © 2023 Vercel, in collaboration with basement.studio).

## Audio
- **Track:** `assets/music/meanwhile-scott-buckley.mp3` ("Meanwhile" by Scott Buckley, CC BY 4.0).
  - `data-media-start` 127.05 and volume 0.9.
  - GSAP fade-in over 0–2.2 s and fade-out over 36.3–39.3 s.
- **Swell:** at 141.85 s in the source, which lands at 14.80 s in the composition.
- **No sound effects.**

## Gate
- `npx hyperframes check`: 0 errors and 17/17 contrast checks pass.
- 12 warnings remain, all reviewed:
  - duplicate-media notices from the two intentional crops of one clip
  - Studio-only layout notes
  - a volume-tween notice
