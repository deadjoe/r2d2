"""Build the multi-speaker benchmark set: 5-minute 16 kHz clips plus subtitle references.

Output goes to artifacts/testset/ (gitignored). Needs yt-dlp and ffmpeg on PATH.
References come from the videos' YouTube subtitles; quality is noted per source.
A source with a human Chinese track also gets <clip>.zh.json / .zh.txt, the
reference for the live translation.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

SOURCES = {
    "ko": {"id": "BD4ggrUx7tE", "lang": "ko", "subs": "ko", "auto": False,
           "note": "human Korean captions (브런치는 핑계고 EP.118)"},
    # Four-person variety talk with edited-in music and effects. The Korean
    # track is near verbatim; the human Chinese track stops at 10:30.
    "ko-mmtg": {"id": "trnQ7dvE1t0", "lang": "ko", "subs": "ko-FmoQciUtYSc", "auto": False,
                "translation": "zh-CN-M2-_869_lzY",
                "note": "human Korean captions + human Chinese translation (문명특급 MMTG, 2026-09-17)"},
    "ja": {"id": "fa_MfqT_SWk", "lang": "ja", "subs": "ja-orig", "auto": True,
           "note": "YouTube auto captions only (Kansai-dialect interview); rough reference"},
    "en": {"id": "NYFGCESmikA", "lang": "en", "subs": "en", "auto": False,
           "note": "human English captions (Lex Fridman #501)"},
}
# clip -> (source, offset in seconds)
CLIPS = {"ko-1": ("ko", 480), "ko-2": ("ko", 2280), "ko-3": ("ko-mmtg", 0), "ko-4": ("ko-mmtg", 300),
         "ja-1": ("ja", 300), "ja-2": ("ja", 2100), "en-1": ("en", 1200), "en-2": ("en", 7680)}
LENGTH = 300


def stamp(text):
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(path, auto):
    cues = []
    for block in path.read_text().split("\n\n"):
        lines = block.strip().splitlines()
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        start, end = (stamp(part.split()[0]) for part in lines[timing].split("-->"))
        body = [re.sub(r"<[^>]+>", "", line).strip() for line in lines[timing + 1:]]
        body = [line for line in body if line]
        if auto:
            # Rolling captions repeat the previous line; the newest text is the last line.
            if end - start < 0.05 or not body:
                continue
            body = body[-1:]
            if cues and cues[-1]["text"] == body[0]:
                continue
        # Editorial captions ([촬영 일시 ...], 【...】) and sound notes are not speech.
        text = re.sub(r"\[[^\]]*\]|【[^】]*】|\([^)]*\)", " ", " ".join(body))
        text = re.sub(r"^- |\s+- ", " ", text)
        text = " ".join(text.split())
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def fetch(root, key, source):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = f"https://youtu.be/{source['id']}"
    audio = next(raw.glob(f"{key}-full.*"), None)
    if audio is None:
        subprocess.run(["yt-dlp", "-q", "-f", "bestaudio", "-o", f"{key}-full.%(ext)s", url],
                       cwd=raw, check=True)
        audio = next(raw.glob(f"{key}-full.*"))
    tracks = {}
    for kind in ("subs", "translation"):
        if kind not in source:
            continue
        vtt = raw / f"{key}.{source[kind]}.vtt"
        if not vtt.exists():
            flag = "--write-auto-subs" if kind == "subs" and source["auto"] else "--write-subs"
            subprocess.run(["yt-dlp", "-q", "--skip-download", flag, "--sub-langs", source[kind],
                            "--sub-format", "vtt", "-o", f"{key}.%(ext)s", url], cwd=raw, check=True)
        tracks[kind] = vtt
    return audio, tracks


def cut(cues, offset):
    # Keep cues that start inside the clip; times are relative to the clip.
    return [{**cue, "start": round(cue["start"] - offset, 3),
             "end": round(min(cue["end"], offset + LENGTH) - offset, 3)}
            for cue in cues if offset <= cue["start"] < offset + LENGTH]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/testset")
    root = Path(parser.parse_args().out)
    manifest = {}
    for key, source in SOURCES.items():
        audio, tracks = fetch(root, key, source)
        cues = parse_vtt(tracks["subs"], source["auto"])
        translated = parse_vtt(tracks["translation"], False) if "translation" in tracks else None
        lang = source["lang"]
        for name, (clip_source, offset) in CLIPS.items():
            if clip_source != key:
                continue
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(offset), "-t", str(LENGTH),
                            "-i", str(audio), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                            str(root / f"{name}.wav")], check=True)
            clip = cut(cues, offset)
            joiner = "" if lang == "ja" else " "
            (root / f"{name}.ref.txt").write_text(joiner.join(c["text"] for c in clip) + "\n")
            (root / f"{name}.ref.json").write_text(json.dumps(clip, ensure_ascii=False, indent=1))
            manifest[name] = {"language": lang, "url": f"https://youtu.be/{source['id']}?t={offset}",
                              "offset": offset, "seconds": LENGTH, "cues": len(clip),
                              "reference": source["note"]}
            if translated is not None:
                zh = cut(translated, offset)
                (root / f"{name}.zh.txt").write_text("".join(c["text"] for c in zh) + "\n")
                (root / f"{name}.zh.json").write_text(json.dumps(zh, ensure_ascii=False, indent=1))
                manifest[name]["translation_cues"] = len(zh)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
