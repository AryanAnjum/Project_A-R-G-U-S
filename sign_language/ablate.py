"""One command for the PHOENIX-2014T ablations, model size and speed.

Each ablation removes one part of the final model and is trained with the same
regularised settings as final_run.py (seed 0):

    RGB only                no keypoints (shows what the pose branch adds)
    Pose only               no RGB (shows what the I3D features add)
    Body keypoints + RGB    no hands or lips (shows what hand and lip keypoints add)
    Small decoder           a 3-layer decoder trained from scratch instead of mBART-50, the
                            decoder size of SignFormer-GCN; 200 epochs and 3 seeds, since a
                            decoder without pretraining needs far longer to learn the language

Finished runs (folder with metrics.json) are skipped, so the command can be
restarted. At the end it times the final model on test samples one at a time
and writes <runs>/phoenix_ablations.md.

    python -m sign_language.ablate --manifest data/phoenix.tsv --kp-root E:/datasets/phoenix_kp \
        --rgb-root E:/datasets/phoenix_i3d
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

from sign_language.final_run import BEST, PAPER, load

# (name, folder, extra train.py arguments, uses RGB, seeds)
SMALL = "Pose + RGB, 3-layer decoder from scratch (SignFormer-GCN size)"
RUNS = [
    ("RGB only", "ablation_rgb_only", ["--parts="], True, [0]),
    ("Pose only", "ablation_pose_only", [], False, [0]),
    ("Body keypoints + RGB (no hands, no lips)", "ablation_body_rgb", ["--parts=body"], True, [0]),
    (SMALL, "ablation_small_decoder",
     ["--decoder", "small", "--epochs", "200", "--lr", "5e-4", "--patience", "25"], True, [0, 1, 2]),
]


def folders(folder, seeds):
    return [folder if len(seeds) == 1 else f"{folder}_s{s}" for s in seeds]


FINAL = "phoenix_rgb_reg_s{}"


@torch.no_grad()
def time_model(run, n=100):
    """Milliseconds per sample, one sample at a time with 5 beams, as in live use."""
    from torch.utils.data import DataLoader
    from sign_language.dataset import KeypointDataset, collate
    from sign_language.model import load_checkpoint
    from sign_language.train import _rgb, autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, targs = load_checkpoint(os.path.join(run, "best"), device)
    ds = KeypointDataset(targs["manifest"], targs["kp_root"], "test", targs["max_frames"],
                         rgb_root=targs.get("rgb_root"))
    times = []
    for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=collate)):
        if i >= n + 5:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(device, True):
            model.translate(batch["features"].to(device), batch["lengths"], rgb=_rgb(batch, device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:  # skip warm-up
            times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times))


def row(name, results):
    """Table row; with several seeds each score is mean ± std."""
    cells = []
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL"):
        v = np.array([r["test"][k] for r in results])
        cells.append(f"{v.mean():.2f}" + (f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else ""))
    r = results[0]
    return (f"| {name} | " + " | ".join(cells)
            + f" | {r['params_total_m']:.1f}M / {r['params_trainable_m']:.1f}M |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kp-root", required=True)
    ap.add_argument("--rgb-root", required=True)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    for name, folder, extra, rgb, seeds in RUNS:
        for seed, sub in zip(seeds, folders(folder, seeds)):
            out = os.path.join(args.runs, sub)
            if load(out):
                print(f"[ABLATE] {name}, seed {seed}: already done in {out}, skipping")
                continue
            print(f"[ABLATE] {name}, seed {seed}: training into {out}")
            cmd = [sys.executable, "-m", "sign_language.train", "--manifest", args.manifest,
                   "--kp-root", args.kp_root, *(["--rgb-root", args.rgb_root] if rgb else []),
                   *BEST, *extra, "--seed", str(seed), "--out", out]
            subprocess.run(cmd, check=True)

    final = os.path.join(args.runs, FINAL.format(0))
    print("[ABLATE] timing the final model")
    ms = time_model(final)

    lines = ["# ARGUS-SLT ablations on PHOENIX-2014T (test)", "",
             "Each row removes one part of the final model; all use the same training settings.",
             "Parameters are total / trained. The I3D extractor (about 12M) and MediaPipe run before the model "
             "and are not counted.", "",
             "| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Params |", "|---|---|---|---|---|---|---|",
             f"| {PAPER[0]} | 41.19 | 30.89 | 24.23 | {PAPER[1]:.2f} | | 9.4M |"]
    seeds = [r for r in (load(os.path.join(args.runs, FINAL.format(s))) for s in range(3)) if r]
    lines.append(row(f"**Full model: pose + RGB, mBART-50 ({len(seeds)} seeds)**", seeds))
    for name, folder, _, _, run_seeds in RUNS:
        results = [load(os.path.join(args.runs, f)) for f in folders(folder, run_seeds)]
        label = name + (f" ({len(results)} seeds)" if len(results) > 1 else " (seed 0)")
        lines.append(row(label, results))
    b4 = np.array([r["test"]["bleu4"] for r in results])  # the small decoder is the last row
    lines += ["", f"3-layer decoder per seed (test BLEU-4): "
              + ", ".join(f"{v:.2f}" for v in b4)
              + f"; mean {b4.mean():.2f} vs {PAPER[1]:.2f} ({b4.mean() - PAPER[1]:+.2f})"]
    lines += ["", f"Speed of the full model: {ms:.0f} ms per sentence (median, one sentence at a time, "
                  f"5 beams, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}), "
                  "after keypoint and I3D extraction."]

    path = os.path.join(args.runs, "phoenix_ablations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ABLATE] written to {path}")


if __name__ == "__main__":
    main()
