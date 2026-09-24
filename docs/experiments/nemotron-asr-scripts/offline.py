"""Nemotron 3.5 streaming accuracy on the r2d2 test set, via the CLI's --stream
path (160 ms input chunks, cache-aware RNNT). Scores with scripts/score.py's
normalisation and alignment so numbers line up with the R2T2 runs.

    python artifacts/nemotron/offline.py [TAG] [extra CLI args...]
"""
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from score import units, align

HERE = Path(__file__).resolve().parent
CLI = next(HERE.glob("src/build/**/bin/nemo-speech"), None) or next(HERE.glob("src/build/**/nemo-speech"))
MODEL = HERE / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
LANG = {"ko": "ko-KR", "ja": "ja-JP", "en": "en-US"}
CLIPS = ["ko-1", "ko-2", "ja-1", "ja-2", "en-1", "en-2"]


def main():
    tag, extra = (sys.argv[1] if len(sys.argv) > 1 else "r1"), sys.argv[2:]
    out = HERE / "out" / tag
    out.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        lang = clip.split("-")[0]
        wav = REPO / "artifacts/testset" / f"{clip}.wav"
        t = time.time()
        p = subprocess.run([str(CLI), "--json", "--quiet", "transcribe", str(wav), "--stream",
                            "--model", str(MODEL), "--language", LANG[lang], *extra],
                           capture_output=True, text=True)
        wall = time.time() - t
        if p.returncode:
            print(clip, "failed", p.stderr[-800:])
            continue
        result = json.loads(p.stdout)
        (out / f"{clip}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        text = result.get("text") or result.get("transcript") or ""
        cues = json.loads((REPO / "artifacts/testset" / f"{clip}.ref.json").read_text())
        ref = [u for c in cues for u in units(c["text"], lang)]
        counts, _ = align(ref, units(text, lang))
        print(json.dumps({"clip": clip, **counts, "wall_s": round(wall, 1)}), flush=True)


if __name__ == "__main__":
    main()
