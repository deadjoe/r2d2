"""Put main's splitter, Jev and laya on the same footing: real Q8 ASR output.

For each Korean clip, take the confirmed transcript of the real-time run
(artifacts/bench/q8-ab), map the human subtitle cue boundaries onto the word
gaps of that transcript by character alignment, and replay the transcript
updates through main's split_sentences exactly as LiveTranslation does.

    .venv/bin/python artifacts/segcmp/prepare.py
"""
import bisect, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from r2d2.translate import split_sentences
from score import units, align

GAP = 0.3


def main():
    for clip in ("ko-1", "ko-2"):
        events = json.loads((REPO / f"artifacts/bench/q8-ab/{clip}.json").read_text())["events"]
        updates = [e for e in events if e["type"] == "transcript"]
        text = updates[-1]["text"]
        cues = json.loads((REPO / f"artifacts/testset/{clip}.ref.json").read_text())

        # Hyp words and the char offset where each ends.
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        ends = [e for _, e in spans]
        word_of = lambda off: bisect.bisect_right(ends, off)   # word containing char offset

        # Reference units, remembering which unit closes each cue.
        ref, cue_last, cue_strong = [], [], []
        for n, cue in enumerate(cues):
            u = units(cue["text"], "ko")
            if not u:
                continue
            ref += u
            nxt = cues[n + 1]["start"] if n + 1 < len(cues) else cue["end"] + 1
            cue_last.append(len(ref) - 1)
            cue_strong.append(bool(re.search(r"[.?!…。？！]\s*$", cue["text"])) or nxt - cue["end"] >= GAP)
        hyp = units(text, "ko", keep_index=True)
        _, hits = align(ref, [u for u, _ in hyp])
        ref_to_off = {i: hyp[j][1] for j, i in hits.items()}

        # A cue boundary lands after the hyp word holding the cue's last matched
        # char, or before the word holding the next cue's first matched char.
        labels = [0] * len(spans)
        mapped = 0
        for last, strong in zip(cue_last, cue_strong):
            gap = None
            for i in (last, last - 1):
                if i in ref_to_off:
                    gap = word_of(ref_to_off[i])
                    break
            if gap is None:
                for i in (last + 1, last + 2):
                    if i in ref_to_off:
                        gap = word_of(ref_to_off[i]) - 1
                        break
            if gap is None or gap < 0:
                continue
            mapped += 1
            labels[gap] = max(labels[gap], 2 if strong else 1)
        # 2 = strong boundary, 1 = display split only, 0 = inside a cue

        # main: replay updates through split_sentences as LiveTranslation.update does.
        consumed, cuts, midword = 0, [], 0
        for e in updates:
            sentences, _ = split_sentences(e["text"][consumed:], e["final"])
            for s in sentences:
                consumed += len(s)
                end = len(text[:consumed].rstrip())
                inside = 0 < consumed < len(text) and not text[consumed - 1].isspace() and not text[consumed].isspace()
                midword += inside
                cuts.append(word_of(end - 1))
        cuts = sorted(set(cuts) - {len(spans) - 1})

        out = {"clip": clip, "words": text.split(), "labels": labels, "main_cuts": cuts,
               "main_midword": midword, "cues": len(cue_last), "cues_mapped": mapped}
        (REPO / f"artifacts/segcmp/{clip}.data.json").write_text(json.dumps(out, ensure_ascii=False))
        print(clip, "words", len(spans), "cues", len(cue_last), "mapped", mapped,
              "strong", labels.count(2), "weak", labels.count(1), "main cuts", len(cuts), "midword", midword)


if __name__ == "__main__":
    main()
