# ARGUS-SLT on PHOENIX-2014T

Test BLEU-4 (sacreBLEU, 13a) and ROUGE-L. The final model is the mean ± std over seeds.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| SignFormer-GCN (PLOS ONE 2025) | | | | 19.75 | |
| RGB + pose (seed 0) | 43.41 | 31.93 | 24.61 | 19.93 | 41.34 |
| **RGB + pose, regularised (3 seeds)** | 43.88 ± 0.37 | 32.59 ± 0.44 | 25.29 ± 0.41 | 20.54 ± 0.39 | 42.90 ± 0.39 |

Per seed (test BLEU-4): seed 0 20.12, seed 1 20.60, seed 2 20.91

Mean test BLEU-4 20.54 vs 19.75: +0.79
