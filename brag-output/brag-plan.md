# Brag plan v2: Walk Before You Climb

The director rejected v1: it looked generic and AI-made, and it reused the rocket video's music. This is the redo, approved by the director on 2026-10-19.

## Direction (from the director)
- **Look:** calm, confident and premium, inspired by modern AI-lab launch films. It is a style reference only, with no names, logos or assets from anyone else.
- **Space:** lots of empty space and one idea at a time. Smooth easing everywhere, with nothing snappy, bouncy, glitchy or neon, and no fast counters.
- **Footage:** framed small and clean in the middle, like a product reveal, then growing to full screen at the key moments.
- **Length:** about 30–40 s. **Actual:** 39.3 s.
- **Music:** a different track from the rocket video. Minimal and cinematic, building gently, CC0 or CC BY only.

## Design decisions
- **Canvas:** flat near-black `#0b0b0c`.
  - A very soft radial lift behind the centre was tried and dropped: it showed faint rings (banding) after video compression.
  - No grain: the footage is a clean vector render, and grain would only muddy it.
- **Type:** Geist (statements) and Geist Mono (small data labels), both SIL OFL 1.1.
  - Sentence case, weight 500, tight tracking.
- **Accent:** a single soft lavender, `#c6c0ff`, taken from the footage's own sky.
  - It is used only to mark "learned to walk first" (B's dot and the result kicker).
- **Motion:** words rise, unblur and fade in (about 1.1–1.6 s, `power3.out` / `expo.out`), and exits are shorter fades.
  - The windows grow with `power2.inOut`.
  - A very slow push-in runs while the robot learns.
- **No sound effects:** music only, which suits the calm brief.

## Storyboard (seconds)
| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–3.0 | Black | "It couldn't stand." |
| 2 | 3.0–7.4 | Window (small) | Untrained robot collapses (clip, then a hold on its own last frame). "0 practice steps" / "It falls." |
| 3 | 7.4–11.6 | Same window | Dissolve to the early learner. "229,376 practice steps" / "It wobbles." |
| 4 | 11.6–17.8 | Same window | Dissolve to the trained walker. "524,288 practice steps" / "It walks." **The window grows to full screen at 14.6–16.4 s, on the music's strings swell (14.8 s).** |
| 5 | 17.8–20.6 | Black | "Then we gave it obstacles." |
| 6 | 20.6–28.2 | Two windows, side by side | Hardcore test course 10016, same course for both. Labels "A Obstacles from the start" and "B Learned to walk first". **The pair grows to full width at 24.4–26.0 s, as A stops at the stump and B climbs it.** "Stopped at the stump." / "Climbed over it." The note "One example test run (course 10016), not the typical result." is on screen throughout. |
| 7 | 28.2–35.3 | Black | Kicker "Learned to walk first" leads into **6×**, "more courses finished", then "13.3% vs 2.0% · tested once on courses it had never seen", then the footnote "Same practice for both. Most of the gap came from 1 of 3 training runs." |
| 8 | 35.3–39.3 | Black | "Walk Before You Climb", "Saad Hilal · S4D", "github.com/saadhilalai/walk-before-you-climb". The music fades out. |

## Honesty checks
- **Footage:** every moving image is a real, verified replay from `videos/final` (clean versions).
  - The two holds are each clip's own frame: the untrained clip's last frame (source frame 163), and the A-vs-B clip at 6.90 s (frame 345).
- **Numbers (`runs/final/results.json`):**
  - B finished 40/300 = 13.3% and A finished 6/300 = 2.0%, a ratio of 6.7, shown as "6×".
  - B's three training runs finished 3%, 6% and 31%, so "most of the gap came from 1 of 3 training runs".
- **The A-vs-B scene shows source 1.2–6.9 s.**
  - A stops at the small stump at about 5.2 s; B climbs it at 5.8–6.2 s.
  - B then gets stuck at the next, taller stump at about 7.2 s, which is after the window ends. B did not finish course 10016 (it timed out at 25.0 m).
  - Nothing on screen claims it did. The scene is labelled as one example, not the typical result.

## Music
"Meanwhile" by Scott Buckley, CC BY 4.0 ([library page](https://www.scottbuckley.com.au/library/meanwhile/)).

- **Excerpt:** 2:07.05 to 2:46.35 of the track, with a 2.2 s fade-in and a 3.0 s fade-out.
- **How it was chosen:** by measuring the candidates' loudness over time, since the music could not be auditioned by ear.
  - This window has a quiet stretch (about −22 to −26 dB) under the learning arc.
  - Then comes the strings swell at 2:21.85 (+12 dB within 0.4 s), timed to land on the walk going full screen.
  - The track then stays full (about −13 to −16 dB) through the result.
- **Runner-up:** "Luminance" by Scott Buckley (CC BY 4.0). Its opening is soft and gentle, but it stays fairly flat for the first 40 s.
- **Not chosen:**
  - "Breathe" by Shane Ivers (CC BY 4.0) is loud from the first second.
  - "She Moved Mountains" by Scott Buckley is very sparse and dark.
