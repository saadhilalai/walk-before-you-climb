# Credits

## Project video (`brag-output/brag.mp4`)
- **Made by:** Saad Hilal, with [Hyperframes](https://hyperframes.heygen.com/) and GSAP, via the /brag workflow.
- **Footage:** only this project's own clips (`videos/final`, clean versions). Each is a replay of a recorded final-test run, checked to match its recorded score exactly.
- **Music:** "Meanwhile" by Scott Buckley ([library page](https://www.scottbuckley.com.au/library/meanwhile/)), an excerpt from 2:07 to 2:46.
  - **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Free for public and commercial use, with credit.
  - **YouTube:** the credit must be in the video description, or his system may place a copyright claim on the video. If a claim appears anyway, release it with his [claims release form](https://www.scottbuckley.com.au/library/copyright-claims-release).
  - **Not in this repo:** the music is already mixed into `brag.mp4`, so the separate track isn't committed. To re-render the video, download it from the library page and save it as `brag-output/composition/assets/music/meanwhile-scott-buckley.mp3`.
- **Fonts:** Geist and Geist Mono, © 2023 Vercel, in collaboration with basement.studio, [SIL Open Font License 1.1](https://github.com/vercel/geist-font/blob/main/OFL.txt).
- **Sound effects:** none.

**When you post the video, put this credit in the post or description:**
'Meanwhile' by Scott Buckley - released under CC-BY 4.0. www.scottbuckley.com.au

## Website (`web/`, live at https://s4dai-walk-before-you-climb.static.hf.space)
- **Made by:** Saad Hilal · S4D. HTML, CSS and JavaScript, with no build step.
- **Motion:** [GSAP](https://gsap.com) 3.14.2 with ScrollTrigger and SplitText, free under GreenSock's [standard "no charge" license](https://gsap.com/standard-license) (note in `web/vendor/gsap/`).
- **Robots:** replays of this project's recorded test runs, drawn in the game's own style (see `render.py`).
- **Fonts:** Geist and Geist Mono, © 2023 Vercel, in collaboration with basement.studio, [SIL Open Font License 1.1](https://github.com/vercel/geist-font/blob/main/OFL.txt) (copy in `web/fonts/OFL.txt`).
- **Film on the page:** the project video above, with the same music credit.
- **Browser-simulator experiment** (`browser_sim/`, not used by the site): [planck.js](https://github.com/piqnt/planck.js) 1.5.0, MIT license (copy in `browser_sim/vendor/`). The random-number generator is adapted from LANDFALL's (same author).

## The project
- **The simulator:** [Gymnasium](https://gymnasium.farama.org/)'s BipedalWalker by the Farama Foundation, running on the Box2D physics engine.
- **Learning algorithms:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
- **Hyperparameters and the RL Zoo benchmark:** [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
