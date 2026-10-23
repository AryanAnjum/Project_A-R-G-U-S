"""Average metrics.json over several seeds: mean and standard deviation per metric.

    python -m sign_language.summarize runs/bornil_mbart_s0 runs/bornil_mbart_s1 runs/bornil_mbart_s2
"""

import json
import os
import sys

import numpy as np

KEYS = ["bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "ms_per_sample"]


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    results = []
    for run in runs:
        with open(os.path.join(run, "metrics.json"), encoding="utf-8") as f:
            results.append(json.load(f))
    print(f"{len(runs)} runs")
    for split in ("dev", "test"):
        cells = []
        for k in KEYS:
            vals = np.array([r[split][k] for r in results])
            cells.append(f"{k} {vals.mean():.2f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.2f}")
        print(f"{split:5s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
