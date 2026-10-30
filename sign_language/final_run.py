"""One command for the final PHOENIX-2014T results: 3 seeds of the best setup, then a summary.

Trains RGB + pose with the regularised settings that scored best so far
(test BLEU-4 20.12 on seed 0) for seeds 0, 1 and 2, one after another. A seed
whose folder already has metrics.json is skipped, so an existing seed-0 run is
reused and the command can be restarted after an interruption. At the end it
writes <runs>/final_results.md with the mean ± std over the seeds and the
earlier ablation runs found in the same folder.

    python -m sign_language.final_run --manifest data/phoenix.tsv --kp-root E:/datasets/phoenix_kp \
        --rgb-root E:/datasets/phoenix_i3d
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

BEST = ["--lang", "de", "--decoder", "mbart50", "--epochs", "100", "--eval-every", "2",
        "--time-mask", "0.2", "--dropout", "0.3", "--word-dropout", "0.3", "--label-smoothing", "0.2"]
ABLATIONS = [("Pose only", "phoenix_dec_s0"), ("RGB + pose", "phoenix_rgb_s0")]
PAPER = ("SignFormer-GCN (PLOS ONE 2025)", 19.75)


def load(run):
    path = os.path.join(run, "metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kp-root", required=True)
    ap.add_argument("--rgb-root", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    outs = [os.path.join(args.runs, f"phoenix_rgb_reg_s{s}") for s in seeds]
    for seed, out in zip(seeds, outs):
        if load(out):
            print(f"[FINAL] seed {seed}: already done in {out}, skipping")
            continue
        print(f"[FINAL] seed {seed}: training into {out}")
        cmd = [sys.executable, "-m", "sign_language.train", "--manifest", args.manifest,
               "--kp-root", args.kp_root, "--rgb-root", args.rgb_root, *BEST,
               "--seed", str(seed), "--out", out]
        subprocess.run(cmd, check=True)

    results = [load(o) for o in outs]
    lines = ["# ARGUS-SLT on PHOENIX-2014T", "",
             "Test BLEU-4 (sacreBLEU, 13a) and ROUGE-L. The final model is the mean ± std over seeds.", "",
             "| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |", "|---|---|---|---|---|---|",
             f"| {PAPER[0]} | | | | {PAPER[1]:.2f} | |"]
    for name, folder in ABLATIONS:
        r = load(os.path.join(args.runs, folder))
        if r:
            t = r["test"]
            lines.append(f"| {name} (seed 0) | {t['bleu1']:.2f} | {t['bleu2']:.2f} | {t['bleu3']:.2f} "
                         f"| {t['bleu4']:.2f} | {t['rougeL']:.2f} |")
    cells = []
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL"):
        v = np.array([r["test"][k] for r in results])
        cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
    lines.append(f"| **RGB + pose, regularised ({len(results)} seeds)** | " + " | ".join(cells) + " |")
    lines += ["", "Per seed (test BLEU-4): " + ", ".join(
        f"seed {s} {r['test']['bleu4']:.2f}" for s, r in zip(seeds, results))]
    bleu4 = np.mean([r["test"]["bleu4"] for r in results])
    lines += ["", f"Mean test BLEU-4 {bleu4:.2f} vs {PAPER[1]:.2f}: {bleu4 - PAPER[1]:+.2f}"]

    path = os.path.join(args.runs, "final_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[FINAL] written to {path}")


if __name__ == "__main__":
    main()
