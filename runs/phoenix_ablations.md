# ARGUS-SLT ablations on PHOENIX-2014T (test)

Each row removes one part of the final model; all use the same training settings.
Parameters are total / trained. The I3D extractor (about 12M) and MediaPipe run before the model and are not counted.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Params |
|---|---|---|---|---|---|---|
| SignFormer-GCN (PLOS ONE 2025) | 41.19 | 30.89 | 24.23 | 19.75 | | 9.4M |
| **Full model: pose + RGB, mBART-50 (3 seeds)** | 43.88 ± 0.37 | 32.59 ± 0.44 | 25.29 ± 0.41 | 20.54 ± 0.39 | 42.90 ± 0.39 | 225.2M / 19.4M |
| RGB only (seed 0) | 42.99 | 31.68 | 24.57 | 19.85 | 41.11 | 221.5M / 15.6M |
| Pose only (seed 0) | 36.67 | 26.26 | 20.09 | 16.08 | 35.06 | 224.7M / 18.8M |
| Body keypoints + RGB (no hands, no lips) (seed 0) | 43.16 | 31.92 | 24.67 | 20.02 | 41.88 | 222.7M / 16.8M |
| Pose + RGB, 3-layer decoder from scratch (SignFormer-GCN size) (3 seeds) | 42.35 ± 0.46 | 31.78 ± 0.46 | 24.92 ± 0.41 | 20.46 ± 0.40 | 42.09 ± 0.45 | 18.1M / 18.1M |

3-layer decoder per seed (test BLEU-4): 20.05, 20.48, 20.85; mean 20.46 vs 19.75 (+0.71)

Speed of the full model: 418 ms per sentence (median, one sentence at a time, 5 beams, NVIDIA GeForce RTX 5060 Ti), after keypoint and I3D extraction.
