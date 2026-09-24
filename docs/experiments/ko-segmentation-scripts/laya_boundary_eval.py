"""Offline: can laya-multilingual tell where a Korean sentence ends?

Ground truth: human subtitle cue boundaries in artifacts/testset/ko-*.ref.json.
Candidates: every word (eojeol) boundary in the joined cue text. Punctuation is
stripped, as ASR mostly omits it. Each prompt variant scores every candidate;
we report AUROC and the best F1 over thresholds.

    USE_TF=0 HF_HOME=artifacts/laya/hf artifacts/laya/venv/bin/python artifacts/laya/boundary_eval.py ko-1 [variant ...]
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import laya

ROOT = Path(__file__).resolve().parents[1]
LEFT, RIGHT = 60, 3        # chars of left context, words of right context
PUNCT = re.compile(r"[^\w\s]|[ㅋㅎㅠㅜ]+")

A_B = {"A": "yes", "B": "no"}

def choice(ins, yes, no):
    return {"type": "choice", "instructions": ins, "criteria": {"A": yes, "B": no}}

VARIANTS = {
    # state = text up to the candidate only
    "left_complete": (lambda l, r: l, choice(
        "Is this text a complete sentence or utterance that can be translated on its own?",
        "yes, it ends at a complete sentence", "no, it is cut off and continues")),
    # state = left ‖ right, question about the marker
    "mark_end": (lambda l, r: f"{l} ‖ {r}", choice(
        "The mark ‖ is a possible cut point in a live speech transcript. Does a sentence end at ‖?",
        "yes, the words before ‖ finish a sentence and the words after ‖ start a new one",
        "no, the sentence continues across ‖")),
    "json_end": (lambda l, r: {"transcript_so_far": l, "next_words": r}, choice(
        "Speech transcript without punctuation. Does a sentence end right after transcript_so_far, before next_words?",
        "yes, transcript_so_far ends a sentence", "no, next_words continue the same sentence")),
    "left_desc": (lambda l, r: l, choice(
        "Korean speech. Does the utterance end with a finished predicate (e.g. -요, -다, -죠, -네요, -까) or is it cut mid-clause?",
        "finished utterance", "cut mid-clause, more words must follow")),
    "mark_desc": (lambda l, r: f"{l} ‖ {r}", choice(
        "Korean speech transcript. Does the utterance before ‖ end with a finished predicate (e.g. -요, -다, -죠, -네요, -까), so a new utterance starts after ‖?",
        "finished utterance before ‖", "cut mid-clause, the words after ‖ continue it")),
    "mark_noul": (lambda l, r: f"{l} ‖ {r}", {"type": "noul",
        "instructions": "In this speech transcript, does a sentence end at the mark ‖?"}),
}


def candidates(clip):
    cues = json.loads((ROOT / "testset" / f"{clip}.ref.json").read_text())
    words, labels = [], []
    for cue in cues:
        w = PUNCT.sub(" ", cue["text"]).split()
        if not w:
            continue
        words += w
        labels += [0] * (len(w) - 1) + [1]
    return words, labels


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def best_f1(y, s):
    y, s = np.asarray(y), np.asarray(s)
    best = (0, 0, 0, 0)
    for t in np.unique(s):
        p = s >= t
        tp = (p & (y == 1)).sum()
        if not tp:
            continue
        prec, rec = tp / p.sum(), tp / (y == 1).sum()
        f = 2 * prec * rec / (prec + rec)
        best = max(best, (f, prec, rec, t))
    return best


def main():
    clip, names = sys.argv[1], sys.argv[2:] or list(VARIANTS)
    words, labels = candidates(clip)
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    for name in names:
        make, q = VARIANTS[name]
        states = []
        for i in range(len(words)):
            left = " ".join(words[: i + 1])[-LEFT:]
            right = " ".join(words[i + 1: i + 1 + RIGHT])
            states.append(make(left, right))
        t = time.time()
        res = agent.predict_batch(states, {"q": q}, batch_size=16)
        ms = (time.time() - t) * 1000 / len(states)
        a = res[0]["answers"]["q"]
        key = "noul" if a["type"] == "noul" else None
        s = [r["answers"]["q"]["noul"] if key else r["answers"]["q"]["probabilities"]["A"] for r in res]
        f, p, r, th = best_f1(labels, s)
        out = {"clip": clip, "variant": name, "n": len(s), "pos": sum(labels),
               "auroc": round(auroc(labels, s), 3), "best_f1": round(f, 3), "p": round(p, 3),
               "r": round(r, 3), "thr": round(float(th), 3), "ms_per": round(ms, 1),
               "score_pos_mean": round(float(np.mean([x for x, y in zip(s, labels) if y])), 3),
               "score_neg_mean": round(float(np.mean([x for x, y in zip(s, labels) if not y])), 3)}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        Path(ROOT / "laya" / "out").mkdir(exist_ok=True)
        (ROOT / "laya" / "out" / f"{clip}-{name}.json").write_text(json.dumps(
            {"summary": out, "words": words, "labels": labels, "scores": s}, ensure_ascii=False))


if __name__ == "__main__":
    main()
