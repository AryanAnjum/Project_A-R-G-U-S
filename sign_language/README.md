# ARGUS-SLT: sign language translation

Translates continuous sign language into a sentence, from body, hand and face keypoints.
It is the model behind the planned live "translate the signer" mode in ARGUS, and the
research track that aims to beat SignFormer-GCN (PLOS ONE 2025):

| Dataset | Sign language | SignFormer-GCN test BLEU-4 |
| --- | --- | --- |
| BornilDB v1.0 | Bangla (BdSL) | 0.58 |
| How2Sign | American (ASL) | 8.53 |
| RWTH-PHOENIX-2014T | German (DGS) | 19.75 |

What is different from SignFormer-GCN:

- 107 keypoints per frame (upper body, both hands, lips, eyebrows) instead of 33 body points.
- One spatio-temporal graph network per body part, hands sharing weights, then a transformer.
- A pretrained multilingual text model (mBART-50, covers Bengali, English and German),
  trained with LoRA, with its vocabulary cut to the tokens in the training text.
- Pose augmentation: rotation, scale, shear, speed change, dropping a hand.
- Plain PyTorch, so it runs on RTX 50-series cards (no fairseq).

## Setup (Windows, RTX 5060 Ti)

Use a separate environment from the ARGUS server (Python 3.10 or newer).

```
python -m venv slt_venv
slt_venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r sign_language/requirements.txt
```

RTX 50-series cards need PyTorch 2.7 or newer built for CUDA 12.8; older builds fail with
"no kernel image is available". Run every command below from the repository root.

## 1. Extract keypoints

```
python -m sign_language.extract_keypoints --src D:/BornilDB/videos --out D:/BornilDB/keypoints
```

The first run downloads MediaPipe's `holistic_landmarker.task` into `models/` (the phone app will use the same model). Videos are subsampled to 25 fps. PHOENIX-2014T ships frame folders instead of videos:
add `--frame-dirs` and point `--src` at `features/fullFrame-210x260px`. Re-running skips
files already done.

## 2. Build the manifest

```
python -m sign_language.manifests csv --csv D:/BornilDB/annotations.csv ^
    --video-col video --text-col sentence --split-col split --out data/bornil.tsv
```

Use the same train/dev/test split as SignFormer-GCN, or the comparison is not fair.
`how2sign` and `phoenix` subcommands read those datasets' own annotation files.

## 3. Train and evaluate

```
python -m sign_language.train --manifest data/bornil.tsv --kp-root D:/BornilDB/keypoints ^
    --lang bn --decoder mbart50 --out runs/bornil_mbart_s0 --seed 0
```

Run seeds 0, 1 and 2, then report the mean and spread:

```
python -m sign_language.summarize runs/bornil_mbart_s0 runs/bornil_mbart_s1 runs/bornil_mbart_s2
```

Each run writes `best/` (checkpoint), `history.json`, `metrics.json` (BLEU-1 to 4,
ROUGE-L, ms per sample, parameter counts) and `predictions_{dev,test}.tsv`.

Useful switches:

- `--decoder small`: a small text model trained from scratch, about SignFormer-GCN's size. Use as an ablation.
- `--finetune full`: fine-tune all of mBART-50 instead of LoRA. Fits in 16 GB with `--batch 8 --grad-accum 2`.
- `--eval-batches 20`: score only part of dev during training, to save time.
- `--bleu-tokenize`: sacreBLEU tokenizer. Keep the default `13a` unless the paper you compare with used another.
