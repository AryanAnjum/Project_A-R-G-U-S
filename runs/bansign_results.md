# ARGUS-SLT on Ban-Sign-Sent-9K (banglagov, Bangla Sign Language)

Test scores, sacreBLEU (13a) on Bangla text with punctuation removed. No official split exists, so two protocols are reported: unseen sentences (train/dev/test sentences are disjoint, 80/10/10 by sentence, seed 0) and unseen signers (dev and test are signers absent from training).

| Protocol | Model (3-layer decoder, 18M) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|---|
| unseen sentences (real translation) | RGB + pose (3 seeds) | 78.37 ± 0.69 | 69.33 ± 0.81 | 62.98 ± 0.94 | 55.12 ± 0.91 | 78.04 ± 0.65 |
| unseen sentences (real translation) | Pose only, live ARGUS model (seed 0) | 78.49 | 69.55 | 63.27 | 55.50 | 78.25 |
| unseen signers | RGB + pose (3 seeds) | 66.04 ± 0.74 | 58.32 ± 0.84 | 52.58 ± 0.92 | 47.22 ± 0.76 | 66.71 ± 0.81 |
| unseen signers | Pose only, live ARGUS model (seed 0) | 71.58 | 63.45 | 57.93 | 52.07 | 71.91 |
