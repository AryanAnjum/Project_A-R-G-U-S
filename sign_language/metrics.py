"""Translation metrics as reported in sign language translation papers.

BLEU-n is sacreBLEU corpus BLEU with max n-gram order n (BLEU-4 is the headline
number). ROUGE-L is the sentence-level LCS F-score averaged over the corpus,
with beta = 1.2, following the Camgoz et al. evaluation code.
"""

from sacrebleu.metrics import BLEU


def bleu_scores(hyps, refs, tokenize="13a"):
    return {f"bleu{n}": BLEU(max_ngram_order=n, tokenize=tokenize).corpus_score(hyps, [refs]).score
            for n in range(1, 5)}


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyps, refs, beta=1.2):
    total = 0.0
    for h, r in zip(hyps, refs):
        h, r = h.split(), r.split()
        lcs = _lcs(h, r)
        if not h or not r or lcs == 0:
            continue
        p, rec = lcs / len(h), lcs / len(r)
        total += (1 + beta ** 2) * p * rec / (rec + beta ** 2 * p)
    return 100.0 * total / max(1, len(refs))


def all_metrics(hyps, refs, tokenize="13a"):
    out = bleu_scores(hyps, refs, tokenize)
    out["rougeL"] = rouge_l(hyps, refs)
    return out
