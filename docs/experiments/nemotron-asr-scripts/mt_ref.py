"""Translation ceiling: HY-MT on the human Korean reference, one subtitle cue at
a time, scored with the same chrF as the live runs. The gap between this and a
live run is what recognition and live segmentation cost the translation.

    .venv/bin/python artifacts/bench/mt_ref.py ko-3 ko-4
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from r2d2.translate import Translator
from score import chrf

translator = Translator()
translator.load()
try:
    for clip in sys.argv[1:]:
        cues = json.loads((REPO / f"artifacts/testset/{clip}.ref.json").read_text())
        zh = json.loads((REPO / f"artifacts/testset/{clip}.zh.json").read_text())
        out = [translator.translate(c["text"]) for c in cues]
        (REPO / f"artifacts/bench/mt-ref-{clip}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps({"clip": clip, "cues": len(cues),
                          "zh_chrf": chrf("".join(out), "".join(c["text"] for c in zh))}), flush=True)
finally:
    translator.close()
