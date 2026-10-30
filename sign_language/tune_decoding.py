"""Pick beam size, length penalty and repeat blocking on dev for a trained model.

No retraining: the saved checkpoint is decoded with each setting on dev, the
best setting by dev BLEU-4 is kept, and only that one is scored on test, so
the test set is not used for selection.

    python -m sign_language.tune_decoding --run runs/phoenix_rgb_s0
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from sign_language.dataset import KeypointDataset, collate
from sign_language.model import load_checkpoint
from sign_language.train import evaluate, write_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of a training run (holds best/)")
    ap.add_argument("--beams", default="4,5,8")
    ap.add_argument("--length-penalties", default="0.8,1.0,1.2,1.5")
    ap.add_argument("--no-repeat", default="0,3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, targs = load_checkpoint(os.path.join(args.run, "best"), device)
    amp = not targs.get("no_amp", False)

    def loader(split):
        ds = KeypointDataset(targs["manifest"], targs["kp_root"], split, targs["max_frames"],
                             rgb_root=targs.get("rgb_root"))
        return DataLoader(ds, batch_size=args.batch, collate_fn=collate, num_workers=args.workers)

    dev_dl, test_dl = loader("dev"), loader("test")
    grid = [(b, lp, nr) for b in map(int, args.beams.split(","))
            for lp in map(float, args.length_penalties.split(","))
            for nr in map(int, args.no_repeat.split(","))]
    results, best = [], None
    for b, lp, nr in grid:
        scores, _ = evaluate(model, vocab, dev_dl, device, b, amp, targs["bleu_tokenize"],
                             no_repeat=nr, length_penalty=lp)
        results.append({"beams": b, "length_penalty": lp, "no_repeat_ngram": nr, "dev": scores})
        print(f"[TUNE] beams={b} length_penalty={lp} no_repeat={nr} | dev BLEU-4 {scores['bleu4']:.2f} "
              f"ROUGE-L {scores['rougeL']:.2f}")
        if best is None or (scores["bleu4"], scores["rougeL"]) > (best["dev"]["bleu4"], best["dev"]["rougeL"]):
            best = results[-1]

    test, rows = evaluate(model, vocab, test_dl, device, best["beams"], amp, targs["bleu_tokenize"],
                          no_repeat=best["no_repeat_ngram"], length_penalty=best["length_penalty"])
    best["test"] = test
    write_predictions(rows, os.path.join(args.run, "predictions_test_tuned.tsv"))
    with open(os.path.join(args.run, "decoding.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "grid": results}, f, indent=1)
    print(f"[TUNE] best on dev: beams={best['beams']} length_penalty={best['length_penalty']} "
          f"no_repeat={best['no_repeat_ngram']} | dev BLEU-4 {best['dev']['bleu4']:.2f} "
          f"| test BLEU-4 {test['bleu4']:.2f} ROUGE-L {test['rougeL']:.2f}")


if __name__ == "__main__":
    main()
