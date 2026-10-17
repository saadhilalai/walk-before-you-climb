"""
gate.py — Step 3 safety check between B's normal phase and its hardcore phase.

Rule (agreed before training, config.B_GATE): B only moves on to hardcore if EVERY seed
finishes at least 90% of the 100 normal validation courses with its 5M model. Otherwise
it exits with an error, so a chained command stops there and the director decides.

Usage:
    python gate.py                 # reads runs/b/seed<N>/run_info.json
    python gate.py --root /tmp/x   # a smoke-test folder
"""

import argparse
import json
import sys
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    args = parser.parse_args()
    gate = config.B_GATE
    passed = True
    for seed in args.seeds:
        info = json.loads((Path(args.root) / "runs" / gate["run"] / f"seed{seed}" / "run_info.json").read_text())
        val = info["saved_models"][gate["model"]]["validation"]
        ok = val["finished_rate"] >= gate["min_finished"]
        passed &= ok
        print(f"seed {seed}: {gate['model']} finished {val['finished_rate']:.0%} of {val['episodes']} "
              f"{info['saved_models'][gate['model']]['course']} validation courses "
              f"(score {val['mean_score']:.1f}) -> {'pass' if ok else 'FAIL'}")
    print(f"Gate (every seed finishes at least {gate['min_finished']:.0%}): {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
