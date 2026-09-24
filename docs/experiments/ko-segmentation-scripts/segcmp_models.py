"""Score every word gap of the real ASR transcript with Jev or laya.

    python3 artifacts/segcmp/models.py jev
    USE_TF=0 HF_HOME=artifacts/laya/hf HF_HUB_OFFLINE=1 artifacts/laya/venv/bin/python artifacts/segcmp/models.py laya
"""
import json, sys
from pathlib import Path

ART = Path(__file__).resolve().parents[1]


def main():
    which = sys.argv[1]
    for clip in ("ko-1", "ko-2"):
        path = ART / f"segcmp/{clip}.data.json"
        data = json.loads(path.read_text())
        words = data["words"]
        if which == "jev":
            sys.path.insert(0, str(ART / "jev"))
            import boundary_eval as J
            # The ASR text carries some punctuation, so drop the "no punctuation" claim.
            J.VARIANTS["asr_new"] = lambda k: {"type": "noul", "instructions":
                f"In this Korean speech recognition transcript, does a new sentence begin right after marker [{k}], "
                f"rather than the words after [{k}] continuing the sentence before it?",
                "criteria": {"true": f"The words before [{k}] finish a sentence or utterance; a new one starts after [{k}]",
                             "false": f"The sentence is unfinished at [{k}]; the words after [{k}] continue it"}}
            scores, tokens, _ = J.score(words, "asr_new")
            data["jev"] = scores
            print(clip, "jev tokens", tokens)
        else:
            sys.path.insert(0, str(ART / "laya"))
            import boundary_eval as L
            import laya
            agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
            for v in ("left_complete", "mark_noul"):
                make, q = L.VARIANTS[v]
                states = [make(" ".join(words[:i + 1])[-L.LEFT:], " ".join(words[i + 1:i + 1 + L.RIGHT]))
                          for i in range(len(words))]
                res = agent.predict_batch(states, {"q": q}, batch_size=16)
                a = [r["answers"]["q"] for r in res]
                data[f"laya_{v}"] = [x["noul"] if x["type"] == "noul" else x["probabilities"]["A"] for x in a]
                print(clip, "laya", v, "done")
        path.write_text(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
