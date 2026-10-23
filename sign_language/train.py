"""Train ARGUS-SLT on a keypoint manifest and report dev/test BLEU.

    python -m sign_language.train --manifest data/bornil.tsv --kp-root D:/BornilDB/keypoints \
        --lang bn --decoder mbart50 --out runs/bornil_mbart_s0 --seed 0

The best checkpoint by dev BLEU-4 is kept in <out>/best; at the end it is scored
on the test split and the numbers go to <out>/metrics.json.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sign_language.dataset import KeypointDataset, collate, worker_init
from sign_language.manifests import read_manifest
from sign_language.metrics import all_metrics
from sign_language.model import (LANG_CODES, PrunedVocab, SignTranslator, build_text_model,
                                 load_checkpoint, load_tokenizer, save_checkpoint)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_labels(vocab, texts, max_len, device):
    ids = vocab.encode(texts, max_len)
    labels = torch.full((len(ids), max(map(len, ids))), -100, dtype=torch.long)
    for i, seq in enumerate(ids):
        labels[i, :len(seq)] = torch.tensor(seq)
    return labels.to(device)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def evaluate(model, vocab, loader, device, beams=5, amp=True, tokenize="13a", max_batches=None):
    model.eval()
    hyps, refs, ids = [], [], []
    start, n = time.perf_counter(), 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        with autocast(device, amp):
            out = model.translate(batch["features"].to(device), batch["lengths"], num_beams=beams)
        hyps += vocab.decode(out.tolist())
        refs += batch["texts"]
        ids += batch["ids"]
        n += len(batch["ids"])
    seconds = time.perf_counter() - start
    scores = all_metrics(hyps, refs, tokenize)
    scores["ms_per_sample"] = 1000.0 * seconds / max(1, n)
    return scores, list(zip(ids, refs, hyps))


def write_predictions(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "reference", "hypothesis"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kp-root", required=True)
    ap.add_argument("--lang", default="bn", help="bn, en, de, zh or an mBART-50 code such as bn_IN")
    ap.add_argument("--decoder", choices=["mbart50", "small"], default="mbart50")
    ap.add_argument("--finetune", choices=["lora", "full"], default="lora", help="how mbart50 is trained")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--d-pose", type=int, default=512)
    ap.add_argument("--pose-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4, help="text model learning rate")
    ap.add_argument("--pose-lr", type=float, default=1e-3, help="pose encoder learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--max-text-len", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=None, help="limit dev batches during training")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--bleu-tokenize", default="13a")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp
    lang_code = LANG_CODES.get(args.lang, args.lang)

    tok = load_tokenizer(lang_code)
    train_texts = [r["text"] for r in read_manifest(args.manifest) if r["split"] == "train"]
    vocab = PrunedVocab.build(tok, train_texts, lang_code)
    print(f"[TRAIN] vocabulary pruned to {len(vocab)} tokens")

    text_model = build_text_model(args.decoder, vocab, args.finetune, args.lora_r)
    model = SignTranslator(text_model, args.d_pose, args.pose_layers, args.dropout).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] parameters: {n_train / 1e6:.1f}M trainable of {n_all / 1e6:.1f}M")

    def loader(split, train):
        ds = KeypointDataset(args.manifest, args.kp_root, split, args.max_frames, train, args.seed)
        return DataLoader(ds, batch_size=args.batch, shuffle=train, collate_fn=collate, drop_last=train,
                          num_workers=args.workers, worker_init_fn=worker_init, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)

    train_dl, dev_dl, test_dl = loader("train", True), loader("dev", False), loader("test", False)

    pose_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("text.")]
    text_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("text.")]
    opt = torch.optim.AdamW([{"params": pose_params, "lr": args.pose_lr},
                             {"params": text_params, "lr": args.lr}], weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl) // args.grad_accum
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    best, history = (-1.0, -1.0), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for step, batch in enumerate(train_dl):
            labels = make_labels(vocab, batch["texts"], args.max_text_len, device)
            with autocast(device, amp):
                logits = model(batch["features"].to(device, non_blocking=True), batch["lengths"], labels)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=-100, label_smoothing=args.label_smoothing)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "minutes": (time.perf_counter() - t0) / 60}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            scores, _ = evaluate(model, vocab, dev_dl, device, args.beams, amp, args.bleu_tokenize, args.eval_batches)
            record.update({f"dev_{k}": v for k, v in scores.items()})
            # ROUGE-L breaks ties, e.g. while BLEU-4 is still 0 early in training
            if (scores["bleu4"], scores["rougeL"]) > best:
                best = (scores["bleu4"], scores["rougeL"])
                save_checkpoint(model, vocab, args, os.path.join(args.out, "best"))
                record["saved"] = True
        history.append(record)
        print("[TRAIN] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    model, vocab, _ = load_checkpoint(os.path.join(args.out, "best"), device)
    results = {}
    for split, dl in (("dev", dev_dl), ("test", test_dl)):
        scores, rows = evaluate(model, vocab, dl, device, args.beams, amp, args.bleu_tokenize)
        results[split] = scores
        write_predictions(rows, os.path.join(args.out, f"predictions_{split}.tsv"))
    results["params_trainable_m"] = n_train / 1e6
    results["params_total_m"] = n_all / 1e6
    results["args"] = vars(args)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"[TRAIN] best dev BLEU-4 {results['dev']['bleu4']:.2f} | test BLEU-4 {results['test']['bleu4']:.2f} "
          f"| ROUGE-L {results['test']['rougeL']:.2f}")


if __name__ == "__main__":
    main()
