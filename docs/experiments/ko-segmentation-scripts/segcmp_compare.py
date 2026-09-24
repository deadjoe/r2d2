"""Compare cut sets on the real ASR transcript: main vs Jev vs laya.

A model becomes a cut set by thresholding its scores. The threshold is taken
from the *other* clip (best F1 there), so neither clip grades its own tuning;
the per-clip best threshold is shown too as an optimistic ceiling.

    python3 artifacts/segcmp/compare.py
"""
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
CLIPS = ("ko-1", "ko-2")


def prf(cuts, labels, strict):
    # strict: only strong boundaries count; display splits are left out of both sides.
    pos = {i for i, l in enumerate(labels[:-1]) if l == 2 or (l == 1 and not strict)}
    skip = {i for i, l in enumerate(labels[:-1]) if l == 1 and strict}
    cuts = set(cuts) - skip - {len(labels) - 1}
    tp = len(cuts & pos)
    p = tp / len(cuts) if cuts else 0
    r = tp / len(pos) if pos else 0
    return round(p, 2), round(r, 2), round(2 * p * r / (p + r), 2) if tp else 0


def best_thr(scores, labels, strict):
    best = (0, 0.5)
    for t in sorted(set(scores)):
        f = prf([i for i, s in enumerate(scores) if s >= t], labels, strict)[2]
        best = max(best, (f, t))
    return best[1]


def seg_stats(cuts, words):
    bounds = [-1] + sorted(cuts) + [len(words) - 1]
    lens = [sum(len(w) for w in words[a + 1:b + 1]) for a, b in zip(bounds, bounds[1:])]
    lens = [l for l in lens if l]
    lens.sort()
    return {"segs": len(lens), "chars_p50": lens[len(lens) // 2], "chars_max": lens[-1]}


def main():
    data = {c: json.loads((ART / f"segcmp/{c}.data.json").read_text()) for c in CLIPS}
    methods = [k for k in ("jev", "laya_left_complete", "laya_mark_noul") if k in data[CLIPS[0]]]
    for strict in (False, True):
        print("\n== truth:", "strong boundaries only" if strict else "all cue boundaries")
        print(f"{'clip':6}{'method':22}{'P':>6}{'R':>6}{'F1':>6}   {'segments':>9}{'chars p50':>10}{'max':>6}   ceiling F1")
        for c in CLIPS:
            d, other = data[c], data[CLIPS[1 - CLIPS.index(c)]]
            print(f"{c:6}{'main (as is)':22}{'%6.2f%6.2f%6.2f' % prf(d['main_cuts'], d['labels'], strict)}   "
                  + "{segs:>9}{chars_p50:>10}{chars_max:>6}".format(**seg_stats(d["main_cuts"], d["words"])))
            for m in methods:
                t = best_thr(other[m], other["labels"], strict)
                cuts = [i for i, s in enumerate(d[m]) if s >= t]
                ceil = prf([i for i, s in enumerate(d[m]) if s >= best_thr(d[m], d["labels"], strict)], d["labels"], strict)[2]
                print(f"{c:6}{m:22}{'%6.2f%6.2f%6.2f' % prf(cuts, d['labels'], strict)}   "
                      + "{segs:>9}{chars_p50:>10}{chars_max:>6}".format(**seg_stats(cuts, d["words"]))
                      + f"   {ceil:.2f} (thr {t:.2f} from other clip)")
            # Human cues themselves, for scale.
            truth = [i for i, l in enumerate(d["labels"]) if l]
            print(f"{c:6}{'(human cues)':22}{'':18}   " + "{segs:>9}{chars_p50:>10}{chars_max:>6}".format(**seg_stats(truth, d["words"])))


if __name__ == "__main__":
    main()
