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


class EMA:
    """Exponential moving average of the trainable weights.

    Dev scores jumped around from one evaluation to the next while the train
    loss fell smoothly; evaluating and saving the averaged weights smooths that out.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {n: p.detach().clone() for n, p in self.params.items()}

    @torch.no_grad()
    def update(self):
        for n, p in self.params.items():
            self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap(self):
        """Swap the live and averaged weights; call twice to swap back."""
        for n, p in self.params.items():
            tmp = p.detach().clone()
            p.copy_(self.shadow[n])
            self.shadow[n] = tmp


def _rgb(batch, device, roll=0):
    return batch["rgb"].roll(roll, 0).to(device) if "rgb" in batch else None


def make_labels(vocab, texts, max_len, device):
    ids = vocab.encode(texts, max_len)
    labels = torch.full((len(ids), max(map(len, ids))), -100, dtype=torch.long)
    for i, seq in enumerate(ids):
        labels[i, :len(seq)] = torch.tensor(seq)
    return labels.to(device)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def evaluate(model, vocab, loader, device, beams=5, amp=True, tokenize="13a", max_batches=None, no_repeat=3,
             length_penalty=1.0):
    model.eval()
    hyps, refs, ids = [], [], []
    start, n = time.perf_counter(), 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        with autocast(device, amp):
            out = model.translate(batch["features"].to(device), batch["lengths"], num_beams=beams,
                                  no_repeat_ngram_size=no_repeat, length_penalty=length_penalty,
                                  rgb=_rgb(batch, device))
        hyps += vocab.decode(out.tolist())
        refs += batch["texts"]
        ids += batch["ids"]
        n += len(batch["ids"])
    seconds = time.perf_counter() - start
    scores = all_metrics(hyps, refs, tokenize)
    scores["ms_per_sample"] = 1000.0 * seconds / max(1, n)
    return scores, list(zip(ids, refs, hyps))


@torch.no_grad()
def dev_loss(model, vocab, loader, device, max_text_len, amp=True):
    """Teacher-forced dev loss, with the real signing and with signing from another sample.

    Rising real loss while train loss falls means overfitting. If the two numbers
    are close, the model is not using the signing and only guesses from the text.
    """
    model.eval()
    total, shuffled, count = 0.0, 0.0, 0
    for batch in loader:
        labels = make_labels(vocab, batch["texts"], max_text_len, device)
        feats, lengths = batch["features"].to(device), batch["lengths"]
        for roll in (0, 1):
            with autocast(device, amp):
                logits = model(feats.roll(roll, 0), lengths.roll(roll, 0), labels, rgb=_rgb(batch, device, roll))
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=-100, reduction="sum").item()
            if roll:
                shuffled += loss
            else:
                total += loss
        count += int((labels != -100).sum())
    return total / max(1, count), shuffled / max(1, count)


def write_predictions(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "reference", "hypothesis"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kp-root", required=True)
    ap.add_argument("--rgb-root", help="I3D features from extract_i3d.py; adds the RGB branch")
    ap.add_argument("--rgb-dim", type=int, default=1024, help="I3D feature size (used with --rgb-root)")
    ap.add_argument("--lang", default="bn", help="bn, en, de, zh or an mBART-50 code such as bn_IN")
    ap.add_argument("--decoder", choices=["mbart50", "small"], default="mbart50")
    ap.add_argument("--finetune", choices=["lora", "full"], default="lora", help="how mbart50 is trained")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--bridge", choices=["decoder", "encoder"], default="decoder",
                    help="feed pose features to mBART's decoder directly, or through its text encoder")
    ap.add_argument("--word-dropout", type=float, default=0.2,
                    help="fraction of previous words hidden from the decoder in training")
    ap.add_argument("--d-pose", type=int, default=512)
    ap.add_argument("--pose-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2, help="pose encoder dropout")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4, help="text model learning rate")
    ap.add_argument("--pose-lr", type=float, default=3e-4, help="pose encoder learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--time-mask", type=float, default=0.0,
                    help="fraction of each training sample blanked in two random spans (0 = off)")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--max-text-len", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=None, help="limit dev batches during training")
    ap.add_argument("--ema", type=float, default=0.999, help="weight averaging decay for eval (0 = off)")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--no-repeat-ngram", type=int, default=3, help="block repeated n-grams when decoding (0 = off)")
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many dev evaluations without a new best (0 = never)")
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
    if not args.rgb_root:
        args.rgb_dim = 0
    model = SignTranslator(text_model, args.d_pose, args.pose_layers, args.dropout, args.bridge,
                           args.rgb_dim).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] parameters: {n_train / 1e6:.1f}M trainable of {n_all / 1e6:.1f}M")

    def loader(split, train):
        ds = KeypointDataset(args.manifest, args.kp_root, split, args.max_frames, train, args.seed, args.rgb_root,
                             args.time_mask)
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

    ema = EMA(model, args.ema) if args.ema > 0 else None
    best, history, stale = (-1.0, -1.0), [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for step, batch in enumerate(train_dl):
            labels = make_labels(vocab, batch["texts"], args.max_text_len, device)
            with autocast(device, amp):
                logits = model(batch["features"].to(device, non_blocking=True), batch["lengths"], labels,
                               args.word_dropout, _rgb(batch, device))
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=-100, label_smoothing=args.label_smoothing)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update()
            losses.append(loss.item())
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "minutes": (time.perf_counter() - t0) / 60}

        evaluating = epoch % args.eval_every == 0 or epoch == args.epochs
        if evaluating and ema:
            ema.swap()
        if evaluating:
            record["dev_loss"], record["dev_loss_shuffled"] = dev_loss(model, vocab, dev_dl, device,
                                                                      args.max_text_len, amp)
            scores, _ = evaluate(model, vocab, dev_dl, device, args.beams, amp, args.bleu_tokenize,
                                 args.eval_batches, args.no_repeat_ngram)
            record.update({f"dev_{k}": v for k, v in scores.items()})
            # ROUGE-L breaks ties, e.g. while BLEU-4 is still 0 early in training
            if (scores["bleu4"], scores["rougeL"]) > best:
                best = (scores["bleu4"], scores["rougeL"])
                save_checkpoint(model, vocab, args, os.path.join(args.out, "best"))
                record["saved"] = True
                stale = 0
            else:
                stale += 1
        if evaluating and ema:
            ema.swap()
        history.append(record)
        print("[TRAIN] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if args.patience and stale >= args.patience:
            print(f"[TRAIN] no dev improvement in {stale} evaluations, stopping")
            break

    model, vocab, _ = load_checkpoint(os.path.join(args.out, "best"), device)
    results = {}
    for split, dl in (("dev", dev_dl), ("test", test_dl)):
        scores, rows = evaluate(model, vocab, dl, device, args.beams, amp, args.bleu_tokenize,
                                no_repeat=args.no_repeat_ngram)
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
