"""Offline: can Jev tell where a Korean sentence ends in unpunctuated ASR-like text?

Follows TypeSafe's structure-recovery cookbook: the state is a window of words
with an inline id at every candidate boundary, and one Noul per id goes in the
same request, so the model reads the whole window once.

Ground truth from human subtitle cues (artifacts/testset/ko-*.ref.json):
  all    every cue boundary is positive, every in-cue word gap negative
  strong only cue boundaries that end in . ? ! … or have a >= 0.3 s gap count
         as positive; other cue boundaries (display splits) are left out

    python3 artifacts/jev/boundary_eval.py ko-1 [variant ...]
"""
import json, re, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = (ROOT / "jev" / ".jev_key").read_text().strip()
PUNCT = re.compile(r"[^\w\s]|[ㅋㅎㅠㅜ]+")
WINDOW, STRIDE = 50, 40      # words per request; boundaries scored once each
GAP = 0.3


def candidates(clip):
    cues = json.loads((ROOT / "testset" / f"{clip}.ref.json").read_text())
    cues = [c for c in cues if PUNCT.sub(" ", c["text"]).split()]
    words, labels, strong = [], [], []
    for n, cue in enumerate(cues):
        w = PUNCT.sub(" ", cue["text"]).split()
        nxt = cues[n + 1]["start"] if n + 1 < len(cues) else cue["end"] + 1
        hard = bool(re.search(r"[.?!…。？！]\s*$", cue["text"])) or nxt - cue["end"] >= GAP
        words += w
        labels += [0] * (len(w) - 1) + [1]
        strong += [0] * (len(w) - 1) + [1 if hard else None]
    return words, labels, strong


VARIANTS = {
    "noul_new": lambda k: {"type": "noul", "instructions":
        f"In this Korean speech transcript (no punctuation), does a new sentence begin right after marker [{k}], "
        f"rather than the words after [{k}] continuing the sentence before it?",
        "criteria": {"true": f"The words before [{k}] finish a sentence or utterance; a new one starts after [{k}]",
                     "false": f"The sentence is unfinished at [{k}]; the words after [{k}] continue it"}},
    "noul_mid": lambda k: {"type": "noul", "instructions":
        f"Does marker [{k}] fall in the middle of a sentence, splitting one sentence of this Korean speech transcript in two?",
        "criteria": {"true": f"[{k}] tears a sentence apart; the words after it continue the sentence before it",
                     "false": f"[{k}] sits between two sentences or utterances"}},
}
INVERT = {"noul_mid"}


def call(state, questions):
    body = json.dumps({"model": "jev-1.13.0", "state": state, "questions": questions}).encode()
    for attempt in range(6):
        req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", body,
                                     {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{e.code} {e.read()[:300]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
    raise RuntimeError("retries exhausted")


def score(words, variant):
    make, n = VARIANTS[variant], len(words)
    out, tokens, secs = [None] * n, 0, []
    start = 0
    while start < n - 1:
        end = min(n, start + WINDOW)
        # score boundaries after words [lo, hi); leave context on both sides
        lo = start if start == 0 else start + (WINDOW - STRIDE) // 2
        hi = n - 1 if end == n else end - (WINDOW - STRIDE) // 2
        parts = []
        for i in range(start, end):
            parts.append(words[i])
            if lo <= i < hi:
                parts.append(f"[{i}]")
        t = time.time()
        r = call(" ".join(parts), {f"b{i}": make(i) for i in range(lo, hi)})
        secs.append(time.time() - t)
        tokens += r["usage"]["input_tokens"]
        for i in range(lo, hi):
            p = r["answers"][f"b{i}"]["noul"]
            out[i] = 1 - p if variant in INVERT else p
        if end == n:
            break
        start += STRIDE
    out[n - 1] = out[n - 1] if out[n - 1] is not None else 1.0
    return out, tokens, secs


def auroc(y, s):
    pairs = sorted(zip(s, y))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    pos = sum(y)
    rsum = sum(ranks[k] for k, (_, lab) in enumerate(pairs) if lab)
    return (rsum - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def best_f1(y, s):
    best = (0, 0, 0, 0)
    for t in sorted(set(s)):
        tp = sum(1 for a, b in zip(s, y) if a >= t and b)
        pp = sum(1 for a in s if a >= t)
        if tp:
            p, r = tp / pp, tp / sum(y)
            best = max(best, (2 * p * r / (p + r), p, r, t))
    return best


def report(clip, variant, labels, strong, s, extra=None):
    rows = {}
    for name, lab in (("all", labels), ("strong", strong)):
        keep = [(a, b) for a, b in zip(s[:-1], lab[:-1]) if b is not None]
        ss, yy = [a for a, _ in keep], [b for _, b in keep]
        f, p, r, t = best_f1(yy, ss)
        rows[name] = {"n": len(yy), "pos": sum(yy), "auroc": round(auroc(yy, ss), 3),
                      "f1": round(f, 3), "p": round(p, 3), "r": round(r, 3), "thr": t}
    print(json.dumps({"clip": clip, "variant": variant, **rows, **(extra or {})}, ensure_ascii=False), flush=True)


def main():
    clip, names = sys.argv[1], sys.argv[2:] or list(VARIANTS)
    words, labels, strong = candidates(clip)
    outdir = ROOT / "jev" / "out"
    outdir.mkdir(exist_ok=True)
    for v in names:
        s, tokens, secs = score(words, v)
        secs.sort()
        report(clip, v, labels, strong, s, {"requests": len(secs), "tokens": tokens,
               "req_s_p50": round(secs[len(secs) // 2], 2), "req_s_max": round(secs[-1], 2)})
        (outdir / f"{clip}-{v}.json").write_text(json.dumps(
            {"words": words, "labels": labels, "strong": strong, "scores": s}, ensure_ascii=False))


if __name__ == "__main__":
    main()
