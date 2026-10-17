"""
make_videos.py — videos made by replaying REAL recorded runs (Step 4 review clips; Step 6 adds the rest).

Every clip replays an episode recorded by compare.py (runs/compare/<name>_episodes.csv): the same
model, training seed and course seed. The WHOLE episode is replayed (only the frames the clip needs
are drawn), and its official score and outcome must match the recorded ones exactly, or no video
is written. Drawing never touches the physics (see render.py).

Usage:
    python make_videos.py stump     # Step 4: A freezes at a stump, B gets past it (same course, side by side)
    python make_videos.py stump --course-seed 20821 --out step4_stump_straddle_a_vs_b   # A straddling a stump
    python make_videos.py gaits     # Step 4: the four trained agents walking the same normal course (2 x 2)

Output: videos/*.mp4 (H.264, CRF 16, 50 frames per second = real time).
"""

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import config

config.limit_cpu_threads(1)

from gymnasium.envs.box2d import bipedal_walker as bw  # noqa: E402

from evaluate import load_agent  # noqa: E402
from render import font, render_hires  # noqa: E402
from walker_env import STEP, make_env, progress  # noqa: E402

OUT = Path("videos")
FPS = 50                      # the simulator's own frame rate: 1 step = 1 frame = 1/50 s
K = 3                         # draw 3x the game's size: each panel 1800 x 1200
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
AGENTS = {   # name on screen, colour tab (same colours as the charts)
    "sac_524k": ("SAC: 524k steps, normal course", (42, 120, 214)),
    "ppo_5m": ("PPO: 5M steps, normal course", (235, 104, 52)),
    "a_100m": ("A: hardcore from scratch", (27, 175, 122)),
    "b_100m": ("B: curriculum (5M normal first)", (237, 161, 0)),
}
STILL = 0.05                  # "no forward progress" = furthest point grew by less than this (world units)


def recorded(name, model, train_seed, course_seed):
    for r in csv.DictReader(open(Path("runs") / "compare" / f"{name}_episodes.csv")):
        if (r["model"], int(r["train_seed"]), int(r["course_seed"])) == (model, train_seed, course_seed):
            return r
    raise KeyError(f"{model} seed {train_seed} course {course_seed} is not in {name}_episodes.csv")


class Replay:
    """Replays one recorded episode step by step. frame() draws the current moment;
    finish() plays the rest without drawing. Frames are never stored, so long clips fit in memory."""

    def __init__(self, model, train_seed, course, course_seed):
        self.agent = load_agent(Path("models") / model / f"seed{train_seed}")
        self.env = make_env(course)
        self.obs, _ = self.env.reset(seed=course_seed)
        self.game = self.env.unwrapped
        self.score, self.steps, self.done, self.info = 0.0, 0, False, {}
        self.best_x, self.best_t = self.game.hull.position.x, 0

    def step(self):
        if self.done:
            return
        self.obs, reward, terminated, truncated, self.info = self.env.step(self.agent.act(self.obs))
        self.score += float(reward)
        self.steps += 1
        x = self.game.hull.position.x
        if x > self.best_x + STILL:
            self.best_x, self.best_t = x, self.steps
        self.best_x = max(self.best_x, x)
        self.done = terminated or truncated

    def frame(self, k=K):
        status = dict(t=self.steps / FPS, distance=progress(self.best_x), score=self.score,
                      stuck=0.0 if self.done else (self.steps - self.best_t) / FPS)
        return render_hires(self.game, k), status

    def robot_box(self, k, left=3.0, right=4.0, up=2.6, down=2.6):
        """Pixel box around the robot in a frame drawn at size k (for filmstrips)."""
        S, H = bw.SCALE * k, bw.VIEWPORT_H * k
        x = (self.game.hull.position.x - max(0.0, self.game.scroll)) * S
        y = H - self.game.hull.position.y * S
        return round(x - left * S), round(y - up * S), round(x + right * S), round(y + down * S)

    def finish(self):
        while not self.done:
            self.step()
        self.env.close()
        return dict(score=self.score, outcome=self.info["outcome"], steps=self.steps, distance=self.info["distance"])


def check(result, rec, label):
    same = result["outcome"] == rec["outcome"] and abs(result["score"] - float(rec["score"])) < 1e-9
    print(f"  {label}: replay {result['outcome']}, score {result['score']:.6f}, {result['steps']} steps | "
          f"recorded {rec['outcome']}, score {float(rec['score']):.6f} -> {'MATCH' if same else 'MISMATCH'}")
    if not same:
        raise SystemExit("The replay doesn't match the recorded run: no video written.")


def outcome_text(result):
    secs = result["steps"] / FPS
    return {"finished": f"finished in {secs:.1f} s",
            "fell": f"fell after {secs:.1f} s, {result['distance']:.0f}% of the way",
            "timed_out": f"ran out of time after {secs:.0f} s, {result['distance']:.0f}% of the way",
            "backed_off": f"walked off the start after {secs:.1f} s"}[result["outcome"]]


def panel(frame, st, name, color, subtitle, header_h, final=None, z=1.0):
    """One agent's view: header (colour tab, name, subtitle) above the game, live status box on it.
    z scales the text and spacing (1.0 = sized for 3x panels)."""
    def u(v):
        return round(v * z)
    header_h = u(header_h)
    h, w = frame.shape[:2]
    img = Image.new("RGB", (w, h + header_h), SURFACE)
    img.paste(Image.fromarray(frame), (0, header_h))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, u(14), header_h), fill=color)
    d.text((u(36), u(12)), name, font=font(u(40), bold=True), fill=INK)
    d.text((u(36), u(58)), subtitle, font=font(u(26)), fill=INK_2)

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    line = f"{st['t']:4.1f} s   ·   {st['distance']:3.0f}% of the course   ·   score {st['score']:+.1f}"
    lines = [(line, font(u(32)), INK)]
    if st["stuck"] >= 1.0:
        lines.append((f"no forward progress for {st['stuck']:.1f} s", font(u(32), bold=True), INK))
    if final:
        lines.append((final, font(u(32), bold=True), INK))
    box_h = u(24) + u(46) * len(lines)
    box_w = u(36) + max(ld.textlength(t, font=f) for t, f, _ in lines)
    ld.rounded_rectangle((u(24), header_h + u(24), u(24) + box_w, header_h + u(24) + box_h), radius=u(14),
                         fill=(255, 255, 255, 215), outline=(190, 189, 182, 255), width=max(1, u(2)))
    for i, (t, f, c) in enumerate(lines):
        ld.text((u(42), header_h + u(36) + u(46) * i), t, font=f, fill=c)
    return np.array(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))


def open_writer(path):
    OUT.mkdir(exist_ok=True)
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                              macro_block_size=16,
                              output_params=["-crf", "16", "-preset", "slow", "-movflags", "+faststart"])


def side_by_side(panels, caption, gap=16, footer_h=48, cols=None):
    """Panels in a grid (one row unless cols is given), with a caption underneath."""
    cols = cols or len(panels)
    rows = -(-len(panels) // cols)
    ph, pw = panels[0].shape[:2]
    canvas = Image.new("RGB", (cols * pw + (cols - 1) * gap, rows * ph + (rows - 1) * gap + footer_h), SURFACE)
    for i, p in enumerate(panels):
        canvas.paste(Image.fromarray(p), ((i % cols) * (pw + gap), (i // cols) * (ph + gap)))
    ImageDraw.Draw(canvas).text((24, rows * ph + (rows - 1) * gap + 10), caption, font=font(24), fill=INK_2)
    return np.array(canvas)


def stump_timing(train_seed, course_seed):
    """Quick pass without drawing: when A stops making progress, and when B clears that same stump."""
    a, b = Replay("a_100m", train_seed, "hardcore", course_seed), Replay("b_100m", train_seed, "hardcore", course_seed)
    a.finish()
    stump = next(o for o in a.env.obstacles if o["x_end"] + STEP >= a.best_x)
    assert stump["kind"] == "stump", f"A is not stuck at a stump but at {stump['kind']}"
    while not b.done and b.game.hull.position.x <= stump["x_end"] + STEP:
        b.step()
    assert not b.done, "B never got past the stump"
    return a.best_t / FPS, b.steps / FPS


def stump(train_seed=142, course_seed=20107, out_name="step4_stump_a_vs_b", hold=3):
    """A freezes at a stump while B, same course and same training seed, gets past it. The clip
    runs until A has been stuck for 6 s and B is 4 s past the stump (10 to 24 s)."""
    t_stuck, t_pass = stump_timing(train_seed, course_seed)
    seconds = int(np.clip(np.ceil(max(t_stuck + 6, t_pass + 4)), 10, 24))
    print(f"A stops making progress at {t_stuck:.1f} s; B clears the same stump at {t_pass:.1f} s -> {seconds} s clip")
    models, header_h = ("a_100m", "b_100m"), 96
    subtitle = f"training seed {train_seed}  ·  100M practice steps"
    caption = (f"Same hardcore analysis course (#{course_seed}) and same training seed ({train_seed}). Real runs, "
               f"replayed exactly: their scores and outcomes match the recorded results. First {seconds} s shown.")
    path = OUT / f"{out_name}.mp4"
    partial = path.with_name(path.stem + ".partial.mp4")
    print(f"Replaying hardcore analysis course {course_seed}, training seed {train_seed}:")
    runs = {m: Replay(m, train_seed, "hardcore", course_seed) for m in models}
    writer, n_frames, last = open_writer(partial), seconds * FPS + 1, {}
    for i in range(n_frames):
        if i:
            for r in runs.values():
                r.step()
        last = {m: r.frame() for m, r in runs.items()}
        frame = side_by_side([panel(*last[m], *AGENTS[m], subtitle, header_h) for m in models], caption)
        writer.append_data(frame)
        if i == min(n_frames - 1, round((t_pass + 1) * FPS)):
            Image.fromarray(frame).save(OUT / f"{out_name}_still.png")

    finals = {}
    try:
        for m, r in runs.items():
            result = r.finish()
            check(result, recorded("step3", m, train_seed, course_seed), m)
            finals[m] = f"whole run: {outcome_text(result)}, score {result['score']:+.1f}"
    except SystemExit:
        writer.close()
        partial.unlink()
        raise
    end = side_by_side([panel(*last[m], *AGENTS[m], subtitle, header_h, final=finals[m]) for m in models], caption)
    for _ in range(hold * FPS):
        writer.append_data(end)
    writer.close()
    partial.rename(path)
    Image.fromarray(end).save(OUT / f"{out_name}_end.png")
    print(f"Saved {path} ({n_frames + hold * FPS} frames, {(n_frames + hold * FPS) / FPS:.1f} s, "
          f"{end.shape[1]}x{end.shape[0]})")


GAITS = (("sac_524k", "step2"), ("ppo_5m", "step2"), ("a_100m", "step3_normal"), ("b_100m", "step3_normal"))


def gaits(train_seed=42, course_seed=20000, seconds=12, hold=2, k=2, strip_from=6.0, strip_n=10, strip_every=5):
    """The four trained agents walking the same normal course (2 x 2 video), plus a filmstrip:
    one row per agent, a frame every strip_every steps (0.1 s), cropped around the robot."""
    models = [m for m, _ in GAITS]
    runs = {m: Replay(m, train_seed, "normal", course_seed) for m in models}
    path = OUT / "step4_gaits_normal.mp4"
    partial = path.with_name(path.stem + ".partial.mp4")
    caption = (f"The four trained agents on the same normal analysis course (#{course_seed}), training seed "
               f"{train_seed}. Real runs, replayed exactly: their scores and outcomes match the recorded results.")
    subtitle = f"training seed {train_seed}"
    z = k / K
    writer, n_frames, last = open_writer(partial), seconds * FPS + 1, {}
    strip_at = {round(strip_from * FPS) + i * strip_every for i in range(strip_n)}
    strip = {m: [] for m in models}
    for i in range(n_frames):
        if i:
            for r in runs.values():
                r.step()
        last = {m: r.frame(k) for m, r in runs.items()}
        if i in strip_at:
            for m, r in runs.items():
                strip[m].append(Image.fromarray(last[m][0]).crop(r.robot_box(k)))
        writer.append_data(side_by_side([panel(*last[m], *AGENTS[m], subtitle, 96, z=z) for m in models],
                                        caption, cols=2))
    finals = {}
    try:
        for (m, name), r in zip(GAITS, runs.values()):
            result = r.finish()
            check(result, recorded(name, m, train_seed, course_seed), m)
            finals[m] = f"whole run: {outcome_text(result)}, score {result['score']:+.1f}"
    except SystemExit:
        writer.close()
        partial.unlink()
        raise
    end = side_by_side([panel(*last[m], *AGENTS[m], subtitle, 96, final=finals[m], z=z) for m in models],
                       caption, cols=2)
    for _ in range(hold * FPS):
        writer.append_data(end)
    writer.close()
    partial.rename(path)
    print(f"Saved {path} ({end.shape[1]}x{end.shape[0]})")

    tw, th = strip[models[0]][0].size
    label_w, gap = 330, 6
    sheet = Image.new("RGB", (label_w + strip_n * (tw + gap), len(models) * (th + gap) + 40), SURFACE)
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), f"Filmstrip: one frame every {strip_every / FPS:.1f} s from {strip_from:.0f} s, normal course "
                    f"#{course_seed}, training seed {train_seed}", font=font(22), fill=INK)
    for row, m in enumerate(models):
        y = 40 + row * (th + gap)
        d.rectangle((0, y, 8, y + th), fill=AGENTS[m][1])
        d.text((18, y + th // 2 - 14), AGENTS[m][0].split(",")[0].split(":")[0] + ": " +
               AGENTS[m][0].split(": ", 1)[1] if ": " in AGENTS[m][0] else AGENTS[m][0], font=font(20), fill=INK)
        for col, img in enumerate(strip[m]):
            sheet.paste(img, (label_w + col * (tw + gap), y))
    sheet.save(OUT / "step4_gaits_filmstrip.png")
    print(f"Saved {OUT}/step4_gaits_filmstrip.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["stump", "gaits"])
    parser.add_argument("--train-seed", type=int, default=142)
    parser.add_argument("--course-seed", type=int, default=20107)
    parser.add_argument("--out", default="step4_stump_a_vs_b")
    args = parser.parse_args()
    if args.what == "stump":
        stump(args.train_seed, args.course_seed, args.out)
    elif args.what == "gaits":
        gaits()


if __name__ == "__main__":
    main()
