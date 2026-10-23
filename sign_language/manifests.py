"""Build the manifest TSV that training reads: id, keypoints, text, split.

`keypoints` is a path relative to the keypoint root passed to train.py.

    # BornilDB or any CSV with a video column and a sentence column
    python -m sign_language.manifests csv --csv bornil.csv --video-col video --text-col sentence \
        --split-col split --out data/bornil.tsv

    # How2Sign realigned CSVs (tab separated, SENTENCE_NAME / SENTENCE)
    python -m sign_language.manifests how2sign --train how2sign_realigned_train.csv \
        --dev how2sign_realigned_val.csv --test how2sign_realigned_test.csv --out data/how2sign.tsv

    # PHOENIX-2014T annotation files (pipe separated, name / translation)
    python -m sign_language.manifests phoenix --annotations PHOENIX-2014-T/annotations/manual --out data/phoenix.tsv
"""

import argparse
import csv
import os
import random
import sys

SPLITS = ("train", "dev", "test")
SPLIT_ALIASES = {"val": "dev", "valid": "dev", "validation": "dev"}


def write_manifest(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "keypoints", "text", "split"])
        for r in rows:
            w.writerow([r["id"], r["keypoints"], " ".join(r["text"].split()), r["split"]])
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    print(f"[MANIFEST] wrote {out}: {counts}")


def read_manifest(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL))


def from_csv(args):
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        rows = list(csv.DictReader(f, dialect=dialect))
    out = []
    for r in rows:
        video = r[args.video_col].strip()
        stem = os.path.splitext(video)[0]
        split = r[args.split_col].strip().lower() if args.split_col else ""
        out.append({"id": os.path.basename(stem), "keypoints": stem + ".npy",
                    "text": r[args.text_col], "split": SPLIT_ALIASES.get(split, split)})
    if not args.split_col:
        print("[MANIFEST] WARNING: no split column, making a random 80/10/10 split. "
              "Use the official split when comparing against a paper.", file=sys.stderr)
        random.Random(args.seed).shuffle(out)
        n = len(out)
        for i, r in enumerate(out):
            r["split"] = "train" if i < 0.8 * n else "dev" if i < 0.9 * n else "test"
    write_manifest(out, args.out)


def from_how2sign(args):
    out = []
    for split, path in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                name = r["SENTENCE_NAME"].strip()
                out.append({"id": name, "keypoints": f"{split}/{name}.npy", "text": r["SENTENCE"], "split": split})
    write_manifest(out, args.out)


def from_phoenix(args):
    out = []
    for split in SPLITS:
        path = os.path.join(args.annotations, f"PHOENIX-2014-T.{split}.corpus.csv")
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                name = r["name"].strip()
                out.append({"id": name, "keypoints": f"{split}/{name}.npy", "text": r["translation"], "split": split})
    write_manifest(out, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("csv")
    c.add_argument("--csv", required=True)
    c.add_argument("--video-col", required=True, help="column with the video path or name, relative to the video root")
    c.add_argument("--text-col", required=True)
    c.add_argument("--split-col", help="column with train/dev/test; omit for a random split")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=from_csv)

    h = sub.add_parser("how2sign")
    for s in ("train", "dev", "test"):
        h.add_argument(f"--{s}", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(fn=from_how2sign)

    p = sub.add_parser("phoenix")
    p.add_argument("--annotations", required=True, help="folder with PHOENIX-2014-T.{train,dev,test}.corpus.csv")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=from_phoenix)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
