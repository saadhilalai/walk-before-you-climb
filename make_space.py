"""
make_space.py — packs the website into a folder that runs as a free Hugging Face Space, and
(optionally) uploads it.

The site needs no server, so the Space is a free "static" Space, just files: the page (web/), the
recorded test runs it replays (web/data/, from export_web_data.py), the film (web/media/, plus the
4K original from brag-output/), the S4D marks and the Geist fonts. Training logs, models and the
browser-simulator experiment (browser_sim/) stay out.

Usage:
    python export_web_data.py replays && python export_web_data.py site && python export_web_data.py gallery
    python make_space.py                                                   # build build/hf_space/ only
    python make_space.py --push USERNAME/walk-before-you-climb             # build, then upload
                                                                           # (pip install huggingface_hub; hf auth login)
"""

import argparse
import shutil
from pathlib import Path

OUT = Path("build") / "hf_space"
FILM_4K = Path("brag-output") / "brag.mp4"

SPACE_CARD = """\
---
title: Walk Before You Climb
colorFrom: gray
colorTo: purple
sdk: static
app_file: index.html
pinned: false
short_description: Robots that learned to walk first, 600 recorded test runs
---

# Walk Before You Climb

Two simulated robots (Gymnasium's BipedalWalker) taught themselves to cross obstacle courses with
exactly the same amount of practice. The one that learned to walk on flat ground first finished 6x
more courses it had never seen (13.3% vs 2.0%, tested once; most of the gap came from 1 of 3
training runs). Pick any of the 100 hidden test courses and watch both robots side by side: every
run on the page is a replay of the recorded test run, checked against the recorded result.

Designed & built by Saad Hilal · S4D. Code, test plan and lab notebook:
https://github.com/saadhilalai/walk-before-you-climb

Simulator: Gymnasium BipedalWalker (Box2D). Learning: Stable-Baselines3 (PPO, SAC), with RL Baselines3
Zoo settings. Typeface: Geist and Geist Mono by Vercel (SIL Open Font License, see fonts/OFL.txt).
Motion: GSAP 3.14.2 with ScrollTrigger and SplitText (GreenSock standard "no charge" license).
Film music: "Meanwhile" by Scott Buckley - released under CC-BY 4.0. www.scottbuckley.com.au
"""


def check_inputs():
    courses = sorted((Path("web") / "data" / "courses").glob("*.bin"))
    stories = sorted((Path("web") / "data" / "stories").glob("*.bin"))
    missing = [p for p in (Path("web") / "data" / "runs.json", Path("web") / "data" / "site.json", Path("web") / "data" / "gallery.json", FILM_4K)
               if not p.exists()]
    if len(courses) != 100 or len(stories) != 3 or missing:
        raise SystemExit(f"Missing site data ({len(courses)}/100 courses, {len(stories)}/3 stories, missing {missing}). "
                         "Run: python export_web_data.py replays && python export_web_data.py site && python export_web_data.py gallery")


def build():
    check_inputs()
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree("web", OUT, ignore=shutil.ignore_patterns(".DS_Store"))
    shutil.copy(FILM_4K, OUT / "media" / "film-4k.mp4")
    (OUT / "README.md").write_text(SPACE_CARD)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"Built {OUT}/ ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")


def push(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Walk Before You Climb", delete_patterns="*")
    user, name = repo_id.split("/")
    print(f"Uploaded. Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"Live site: https://{user.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", metavar="USERNAME/SPACE", help="upload to this Hugging Face Space")
    args = parser.parse_args()
    build()
    if args.push:
        push(args.push)
