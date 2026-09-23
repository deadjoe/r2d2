"""Score smoke.py event files against the test set's subtitle references.

    python scripts/score.py artifacts/bench/q8/*.json [--testset artifacts/testset]

Korean and Japanese are scored by character (spaces ignored), English by word.
Both sides are normalised the same way: NFKC, lower case, no punctuation, no
laughter marks, no unambiguous fillers. Subtitles tidy speech up, so absolute
numbers overstate the error; compare configurations, not against zero.
"""
import argparse
import json
from pathlib import Path
import re
import unicodedata

FILLERS = {
    "en": {"uh", "um", "uhm", "umm", "hmm", "mm", "mhm", "erm", "ah"},
    "ko": {"음", "어", "으음", "음음"},
    "ja": set(),
}
_LAUGH = re.compile(r"[ㅋㅎㅠㅜ]+")


def units(text, lang, keep_index=False):
    """Scoring units: words for English, characters otherwise. With
    keep_index, each unit carries the offset of its last char in `text`."""
    def clean(chunk):
        chunk = _LAUGH.sub("", unicodedata.normalize("NFKC", chunk).lower())
        return "".join(c for c in chunk if unicodedata.category(c)[0] not in "PSZC")

    out = []
    for match in re.finditer(r"\S+", text):
        word = clean(match.group())
        if not word or word in FILLERS[lang]:
            continue
        if lang == "en":
            out.append((word, match.end() - 1) if keep_index else word)
            continue
        # Per character, so each unit keeps its own offset (Japanese has no spaces).
        for offset, char in enumerate(match.group(), match.start()):
            out.extend((c, offset) if keep_index else c for c in clean(char))
    return out


def align(ref, hyp):
    """Levenshtein alignment. Returns counts and the ref index each matched hyp unit hit."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row, r = back[i], ref[i - 1]
        row[0] = 1
        for j in range(1, m + 1):
            best, op = prev[j - 1] + (r != hyp[j - 1]), 0
            if prev[j] + 1 < best:
                best, op = prev[j] + 1, 1       # deletion
            if cur[j - 1] + 1 < best:
                best, op = cur[j - 1] + 1, 2    # insertion
            cur[j], row[j] = best, op
        prev = cur
    i, j, sub, dele, ins, hits = n, m, 0, 0, 0, {}
    while i or j:
        op = back[i][j]
        if op == 0:
            if ref[i - 1] == hyp[j - 1]:
                hits[j - 1] = i - 1
            else:
                sub += 1
            i, j = i - 1, j - 1
        elif op == 1:
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return {"ref": n, "hyp": m, "sub": sub, "del": dele, "ins": ins,
            "err": round((sub + dele + ins) / max(n, 1), 4)}, hits


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def score(path, testset):
    data = json.loads(Path(path).read_text())
    events = data["events"]
    clip = Path(path).stem
    lang = clip.split("-")[0]
    cues = json.loads((testset / f"{clip}.ref.json").read_text())
    updates = [e for e in events if e["type"] == "transcript"]
    text = updates[-1]["text"]

    # Reference units with the time each was spoken (spread evenly over its cue).
    ref, spoken = [], []
    for cue in cues:
        cue_units = units(cue["text"], lang)
        for k, unit in enumerate(cue_units):
            ref.append(unit)
            spoken.append(cue["start"] + (cue["end"] - cue["start"]) * (k + 1) / len(cue_units))
    hyp_indexed = units(text, lang, keep_index=True)
    counts, hits = align(ref, [u for u, _ in hyp_indexed])

    # When each char of the confirmed text first appeared (wall clock of that update).
    confirmed_at, known = [], 0
    for e in updates:
        if len(e["text"]) > known:
            confirmed_at.extend([e["wall_ms"]] * (len(e["text"]) - known))
            known = len(e["text"])
    latency = [confirmed_at[hyp_indexed[j][1]] / 1000 - spoken[i] for j, i in hits.items()]

    steps = [e for e in updates if not e["final"]]
    decode = [e["decode_ms"] for e in steps]
    timings = [e.get("timings") or {} for e in steps]
    field = lambda key: [t[key] for t in timings if key in t]
    result = {
        "clip": clip, **counts,
        "latency_s": {"p50": round(pct(latency, .5), 2), "p90": round(pct(latency, .9), 2)} if latency else None,
        "steps": len(steps),
        "decode_ms": {"p50": pct(decode, .5), "p90": pct(decode, .9), "p99": pct(decode, .99)},
        "over_160": round(sum(d > 160 for d in decode) / len(decode), 3),
        "merged": sum(e["hops"] > 1 for e in steps),
        "max_backlog_ms": max(e["backlog_ms"] for e in updates),
        "resets": [(round(e["audio_ms"] / 1000), e["reset"]) for e in steps if e["reset"]],
        "prompt_n": pct(field("prompt_n"), .5), "prompt_ms": pct(field("prompt_ms"), .5),
        "predicted_n": pct(field("predicted_n"), .5), "predicted_ms": pct(field("predicted_ms"), .5),
        "empty_steps": round(sum(1 for a, b in zip(updates, updates[1:]) if a["text"] == b["text"]) / len(updates), 3),
    }
    final = [e for e in events if e["type"] == "translation" and e.get("final")]
    if final:
        sentences = final[-1]["sentences"]
        calls = [s["translate_ms"] for s in sentences if s["translate_ms"]]
        lags = [s["lag_ms"] for s in sentences]
        result["mt"] = {
            "sentences": len(sentences), "reused_draft": sum(1 for s in sentences if not s["translate_ms"]),
            "translate_ms": {"p50": pct(calls, .5), "p90": pct(calls, .9)},
            "lag_ms": {"p50": pct(lags, .5), "p90": pct(lags, .9), "max": max(lags, default=None)},
            "src_chars_p50": pct([len(s["source"]) for s in sentences], .5),
            "drafts": sum(1 for e in events if e["type"] == "translation" and e["lag_ms"] is None
                          and not e.get("final")),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--testset", default="artifacts/testset")
    args = parser.parse_args()
    for path in args.files:
        print(json.dumps(score(path, Path(args.testset)), ensure_ascii=False))


if __name__ == "__main__":
    main()
