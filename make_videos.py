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
    python make_videos.py final     # Step 6: every final clip, 4K, replaying TEST runs picked by TEST_PLAN.md rules

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

    def frame(self, k=K, width=None):
        status = dict(t=self.steps / FPS, distance=progress(self.best_x), score=self.score,
                      stuck=0.0 if self.done else (self.steps - self.best_t) / FPS)
        return render_hires(self.game, k, width), status

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


def open_writer(path, crf=16, threads=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = ["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"]
    if threads:
        params += ["-tune", "animation", "-threads", str(threads)]
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                              macro_block_size=16, output_params=params)


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
    stump = next(o for o in a.env.obstacles if o["x_end"] + STEP >= a.game.hull.position.x)   # where A ended up
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


# ---- Step 6: the final videos, replaying TEST runs (TEST_PLAN.md, section 4) ------------------
FINAL_DIR = OUT / "final"
UHD_W, UHD_H = 3840, 2160                      # 4K, 16:9
K_UHD = UHD_H / bw.VIEWPORT_H                  # 5.4x the game's size; the camera is widened to 16:9
CRF_FINAL = 12                                 # very high quality H.264 for editing
HOLD = FPS                                     # 1 s on the last frame
NAMES = {   # on-screen name, practice steps
    "untrained": ("Untrained robot", "0"), "sac_229k": ("SAC, early in training", "229,376"),
    "sac_524k": ("SAC, trained", "524,288"), "ppo_5m": ("PPO, trained", "5,046,272"),
    "a_100m": ("A: hardcore from scratch", "100M"), "b_100m": ("B: curriculum (5M normal first)", "100M"),
}


def final_record(model, course, train_seed, course_seed):
    for r in csv.DictReader(open(Path("runs") / "final" / "episodes.csv")):
        if (r["model"], r["course"], int(r["train_seed"]), int(r["course_seed"])) == \
                (model, course, train_seed, course_seed):
            return r
    raise KeyError((model, course, train_seed, course_seed))


def final_picks(seed=42):
    """TEST_PLAN.md section 4: the first test course in seed order that matches each rule (seed 42)."""
    from falls import furthest_x, obstacles_by_course
    rows = list(csv.DictReader(open(Path("runs") / "final" / "episodes.csv")))
    ep = {(r["model"], r["course"], int(r["train_seed"]), int(r["course_seed"])): r for r in rows}
    seeds = range(config.TEST_SEED, config.TEST_SEED + config.N_TEST_EPISODES)
    obs = obstacles_by_course(seeds)

    def g(m, c, cs):
        return ep[(m, c, seed, cs)]

    def first(rule):
        return next((cs for cs in seeds if rule(cs)), None)

    def a_stuck_b_past(cs):
        a, b = g("a_100m", "hardcore", cs), g("b_100m", "hardcore", cs)
        if a["outcome"] != "timed_out":
            return False
        ahead = [o for o in obs[cs] if o["x_end"] + STEP >= float(a["end_x"])]   # where A ended up (fixed in Step 7)
        return bool(ahead) and ahead[0]["kind"] == "stump" and \
            furthest_x(float(b["distance"])) > ahead[0]["x_end"] + STEP

    return {
        "untrained_collapse": 10000,
        "early_wobbly": first(lambda cs: float(g("sac_229k", "normal", cs)["distance"]) >= 20
                              and g("sac_229k", "normal", cs)["outcome"] != "finished"),
        "trained_walker": first(lambda cs: g("sac_524k", "normal", cs)["outcome"] == "finished"),
        "a_vs_b": first(lambda cs: g("b_100m", "hardcore", cs)["outcome"] == "finished"
                        and g("a_100m", "hardcore", cs)["outcome"] != "finished"),
        "a_freezes_b_gets_past": first(a_stuck_b_past),
        "broll_1": first(lambda cs: g("b_100m", "hardcore", cs)["outcome"] == "finished"),
        "broll_2": first(lambda cs: g("a_100m", "hardcore", cs)["fell_at"] == "pit"),
        "broll_3": first(lambda cs: g("b_100m", "hardcore", cs)["fell_at"] == "stump"),
        "broll_4": first(lambda cs: g("ppo_5m", "normal", cs)["outcome"] == "finished"),
        "broll_5": first(lambda cs: g("a_100m", "normal", cs)["outcome"] == "finished"),
    }


class Outputs:
    """Writes the same frames to several files at once (e.g. labelled and clean). Files are
    written as *.partial.mp4 and only renamed into place by keep(), after the replay checks."""

    def __init__(self, names):
        self.paths = {n: FINAL_DIR / f"{n}.mp4" for n in names}
        self.partial = {n: p.with_name(p.stem + ".partial.mp4") for n, p in self.paths.items()}
        self.writers = {n: open_writer(p, CRF_FINAL, threads=1) for n, p in self.partial.items()}
        self.frames = 0

    def write(self, frames):
        for n, f in frames.items():
            self.writers[n].append_data(f)
        self.frames += 1

    def close(self, keep):
        for n, w in self.writers.items():
            w.close()
            if keep:
                self.partial[n].rename(self.paths[n])
            else:
                self.partial[n].unlink()


def verified(runs_and_records):
    """Finish every replay and compare it with its recorded TEST run; stop if anything differs."""
    sources = []
    for (model, course, seed, cs), run in runs_and_records:
        result = run.finish()
        rec = final_record(model, course, seed, cs)
        same = result["outcome"] == rec["outcome"] and abs(result["score"] - float(rec["score"])) < 1e-9
        print(f"  {model} {course} seed {seed} course {cs}: replay {result['outcome']} {result['score']:.6f} | "
              f"recorded {rec['outcome']} {float(rec['score']):.6f} -> {'MATCH' if same else 'MISMATCH'}")
        sources.append(dict(agent=model, course=course, train_seed=seed, course_seed=cs, outcome=result["outcome"],
                            score=result["score"], steps=result["steps"], distance=result["distance"], match=same))
        if not same:
            raise SystemExit("A replay doesn't match its recorded test run: no video kept.")
    return sources


def text_box(img, lines, xy, z=1.0):
    """A white rounded box with lines of (text, size, bold) at xy, drawn on a PIL image."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fonts = [font(round(sz * z), bold=b) for _, sz, b in lines]
    heights = [round(sz * z * 1.35) for _, sz, _ in lines]
    pad = round(28 * z)
    w = max(d.textlength(t, font=f) for (t, _, _), f in zip(lines, fonts)) + 2 * pad
    h = sum(heights) + 2 * pad - round(8 * z)
    x, y = xy
    d.rounded_rectangle((x, y, x + w, y + h), radius=round(18 * z), fill=(255, 255, 255, 222),
                        outline=(190, 189, 182, 255), width=max(2, round(2 * z)))
    yy = y + pad
    for (t, _, _), f, lh in zip(lines, fonts, heights):
        d.text((x + pad, yy), t, font=f, fill=INK)
        yy += lh
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def live_line(st):
    return f"{st['t']:4.1f} s   ·   {st['distance']:3.0f}% of the course   ·   score {st['score']:+.1f}"


def single_labelled(frame, title, subtitle, st, final=None):
    lines = [(title, 84, True), (subtitle, 46, False), (live_line(st), 50, False)]
    if st["stuck"] >= 1.0:
        lines.append((f"no forward progress for {st['stuck']:.1f} s", 50, True))
    if final:
        lines.append((final, 50, True))
    return np.array(text_box(Image.fromarray(frame), lines, (64, 64)))


def run_single(model, course, seed, cs, n_frames=None):
    """Yield (frame, status) for a whole run (or its first n_frames), then the run itself."""
    run = Replay(model, seed, course, cs)
    i = 0
    while True:
        frame, st = run.frame(K_UHD, UHD_W)
        yield frame, st, run
        i += 1
        if run.done or (n_frames and i >= n_frames):
            return
        run.step()


def clip_collapse_to_walking(picks, seed=42):
    """Untrained -> early wobbly SAC -> trained SAC (labelled sequence + clean sequence + 3 clean singles)."""
    segments = [("untrained", picks["untrained_collapse"], None, "No practice yet"),
                ("sac_229k", picks["early_wobbly"], 10 * FPS, "After 229,376 practice steps"),
                ("sac_524k", picks["trained_walker"], None, "After 524,288 practice steps")]
    seq = Outputs(["collapse_to_walking", "collapse_to_walking_clean"])
    singles = {m: Outputs([f"{m}_normal_clean"]) for m, *_ in segments}
    checks = []
    for model, cs, limit, title in segments:
        name, steps = NAMES[model]
        subtitle = f"{name}  ·  normal test course {cs}  ·  training seed {seed}"
        last = None
        for frame, st, run in run_single(model, "normal", seed, cs):
            if limit is None or run.steps < limit:
                seq.write({"collapse_to_walking": single_labelled(frame, title, subtitle, st),
                           "collapse_to_walking_clean": frame})
                last = (frame, st)
            singles[model].write({f"{model}_normal_clean": frame})
        rec = verified([((model, "normal", seed, cs), run)])
        checks += rec
        r = rec[0]
        final = f"whole run: {outcome_text(r)}, score {r['score']:+.1f}"
        for _ in range(HOLD):
            seq.write({"collapse_to_walking": single_labelled(*last[:1], title, subtitle, last[1], final),
                       "collapse_to_walking_clean": last[0]})
            singles[model].write({f"{model}_normal_clean": frame})
    seq.close(keep=True)
    for o in singles.values():
        o.close(keep=True)
    return [dict(files=list(seq.paths.values()), frames=seq.frames, sources=checks,
                 rule="untrained: course 10000; early SAC: first normal course >= 20% and not finished "
                      "(first 10 s shown in the sequence); trained SAC: first normal course it finishes")] + \
           [dict(files=list(o.paths.values()), frames=o.frames, sources=[c for c in checks if c["agent"] == m],
                 rule="whole run") for m, o in singles.items()]


PAIR_PANEL_W, PAIR_GAP, PAIR_TOP = 1872, 32, 420
PAIR_K = PAIR_PANEL_W / bw.VIEWPORT_W          # 3.12x, a 3:2 panel per agent


def pair_frame(frames, statuses, models, seed, title, caption, labelled, finals=None):
    z = PAIR_K / K
    canvas = Image.new("RGB", (UHD_W, UHD_H), SURFACE)
    header_h = round(96 * z)
    y0 = PAIR_TOP
    for i, m in enumerate(models):
        x = 32 + i * (PAIR_PANEL_W + PAIR_GAP)
        if labelled:
            p = panel(frames[i], statuses[i], NAMES[m][0], AGENTS[m][1], f"training seed {seed}  ·  100M practice steps",
                      96, final=finals[i] if finals else None, z=z)
            canvas.paste(Image.fromarray(p), (x, y0 - header_h))
        else:
            canvas.paste(Image.fromarray(frames[i]), (x, y0))
    if labelled:
        d = ImageDraw.Draw(canvas)
        d.text((40, 170), title, font=font(72, bold=True), fill=INK)
        ph = frames[0].shape[0]
        for j, line in enumerate(caption):
            d.text((40, y0 + ph + 60 + j * 62), line, font=font(44), fill=INK_2)
    return np.array(canvas)


def clip_pair(name, cs, seed, title, caption, seconds=None):
    """A and B side by side on the same hardcore test course (labelled + clean)."""
    models = ("a_100m", "b_100m")
    runs = {m: Replay(m, seed, "hardcore", cs) for m in models}
    out = Outputs([name, f"{name}_clean"])
    n_frames = seconds * FPS + 1 if seconds else None
    i = 0
    while True:
        fr = {m: runs[m].frame(PAIR_K) for m in models}
        frames, sts = [fr[m][0] for m in models], [fr[m][1] for m in models]
        out.write({name: pair_frame(frames, sts, models, seed, title, caption, True),
                   f"{name}_clean": pair_frame(frames, sts, models, seed, title, caption, False)})
        i += 1
        if (n_frames and i >= n_frames) or (not n_frames and all(r.done for r in runs.values())):
            break
        for r in runs.values():
            r.step()
    try:
        checks = verified([((m, "hardcore", seed, cs), runs[m]) for m in models])
    except SystemExit:
        out.close(keep=False)
        raise
    finals = [f"whole run: {outcome_text(c)}, score {c['score']:+.1f}" for c in checks]
    end = pair_frame(frames, sts, models, seed, title, caption, True, finals)
    clean_end = pair_frame(frames, sts, models, seed, title, caption, False)
    for _ in range(2 * HOLD):
        out.write({name: end, f"{name}_clean": clean_end})
    out.close(keep=True)
    return [dict(files=list(out.paths.values()), frames=out.frames, sources=checks)]


def clip_broll(name, model, course, cs, seed=42):
    out = Outputs([name])
    for frame, st, run in run_single(model, course, seed, cs):
        out.write({name: frame})
    try:
        checks = verified([((model, course, seed, cs), run)])
    except SystemExit:
        out.close(keep=False)
        raise
    for _ in range(HOLD):
        out.write({name: frame})
    out.close(keep=True)
    return [dict(files=list(out.paths.values()), frames=out.frames, sources=checks, rule="whole run")]


def final_videos(only=None):
    import json
    import time
    picks = final_picks()
    print("Picked by the TEST_PLAN.md rules (seed 42):", picks)
    results = json.loads((Path("runs") / "final" / "results.json").read_text())
    rate = {m: results["agents"][f"{m} / hardcore"]["pooled"]["finished_rate"] for m in ("a_100m", "b_100m")}
    jobs = {
        "collapse_to_walking": lambda: clip_collapse_to_walking(picks),
        "a_freezes_b_gets_past": lambda: clip_pair(
            "a_freezes_b_gets_past", picks["a_freezes_b_gets_past"], 42,
            "Hardcore course: A freezes at the stump, B gets over it",
            [f"Hardcore test course {picks['a_freezes_b_gets_past']}, training seed 42. Picked by a rule fixed before the "
             "test: the first test course where A gets stuck at a stump and B gets past it.",
             "Real test runs, replayed exactly: scores and outcomes match the recorded results."],
            seconds=stump_seconds(picks["a_freezes_b_gets_past"])),
        "a_vs_b_hardcore": lambda: clip_pair(
            "a_vs_b_hardcore", picks["a_vs_b"], 42,
            "Hardcore course: from scratch (A) vs curriculum (B), same course, same seed",
            [f"Hardcore test course {picks['a_vs_b']}, training seed 42. Chosen to illustrate the main result, by a rule "
             "fixed before the test: the first test course that B finishes and A doesn't.",
             f"Across all 300 hardcore test runs: B finished {rate['b_100m']:.1%}, A finished {rate['a_100m']:.1%}. "
             "Real test runs, replayed exactly."]),
        "broll_1_b_finishes_hardcore": lambda: clip_broll("broll_1_b_finishes_hardcore", "b_100m", "hardcore",
                                                          picks["broll_1"]),
        "broll_2_a_falls_at_pit": lambda: clip_broll("broll_2_a_falls_at_pit", "a_100m", "hardcore", picks["broll_2"]),
        "broll_3_b_falls_at_stump": lambda: clip_broll("broll_3_b_falls_at_stump", "b_100m", "hardcore",
                                                       picks["broll_3"]),
        "broll_4_ppo_walks_normal": lambda: clip_broll("broll_4_ppo_walks_normal", "ppo_5m", "normal", picks["broll_4"]),
        "broll_5_a_walks_normal": lambda: clip_broll("broll_5_a_walks_normal", "a_100m", "normal", picks["broll_5"]),
    }
    manifest_path = FINAL_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, job in jobs.items():
        if only and key not in only:
            continue
        t0 = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] {key}", flush=True)
        entries = job()
        for e in entries:
            e["files"] = [str(f) for f in e["files"]]
            e.update(resolution=f"{UHD_W}x{UHD_H}", fps=FPS, seconds=round(e["frames"] / FPS, 2),
                     codec=f"H.264 (libx264, CRF {CRF_FINAL}, preset slow, yuv420p)")
        manifest[key] = entries
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"  done in {time.perf_counter() - t0:.0f} s: " + ", ".join(f for e in entries for f in e["files"]))
    print(f"Saved {manifest_path}")


def stump_seconds(cs, seed=42):
    """As in Step 4: until A has been stuck for 6 s and B is 4 s past the stump (10 to 24 s); if B's run ends
    sooner, B's panel holds its last frame."""
    a, b = Replay("a_100m", seed, "hardcore", cs), Replay("b_100m", seed, "hardcore", cs)
    a.finish()
    stump = next(o for o in a.env.obstacles if o["x_end"] + STEP >= a.game.hull.position.x)   # where A ended up
    while not b.done and b.game.hull.position.x <= stump["x_end"] + STEP:
        b.step()
    t_stuck, t_pass = a.best_t / FPS, b.steps / FPS
    print(f"  A is stuck at the {stump['kind']} at x={stump['x_start']:.2f}; A stops making progress at {t_stuck:.1f} s; B clears the stump at {t_pass:.1f} s")
    return int(np.clip(np.ceil(max(t_stuck + 6, t_pass + 4)), 10, 24))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["stump", "gaits", "final"])
    parser.add_argument("--only", nargs="+", help="final: just these clips")
    parser.add_argument("--train-seed", type=int, default=142)
    parser.add_argument("--course-seed", type=int, default=20107)
    parser.add_argument("--out", default="step4_stump_a_vs_b")
    args = parser.parse_args()
    if args.what == "stump":
        stump(args.train_seed, args.course_seed, args.out)
    elif args.what == "gaits":
        gaits()
    elif args.what == "final":
        final_videos(args.only)


if __name__ == "__main__":
    main()
