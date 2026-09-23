"""Build the multi-speaker benchmark set: 5-minute 16 kHz clips plus subtitle references.

Output goes to artifacts/testset/ (gitignored). Needs yt-dlp and ffmpeg on PATH.
References come from the videos' YouTube subtitles; quality is noted per source.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

SOURCES = {
    "ko": {"id": "BD4ggrUx7tE", "subs": "ko", "auto": False,
           "note": "human Korean captions (브런치는 핑계고 EP.118)"},
    "ja": {"id": "fa_MfqT_SWk", "subs": "ja-orig", "auto": True,
           "note": "YouTube auto captions only (Kansai-dialect interview); rough reference"},
    "en": {"id": "NYFGCESmikA", "subs": "en", "auto": False,
           "note": "human English captions (Lex Fridman #501)"},
}
CLIPS = {"ko-1": 480, "ko-2": 2280, "ja-1": 300, "ja-2": 2100, "en-1": 1200, "en-2": 7680}
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
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", " ".join(body))
        text = re.sub(r"^- |\s+- ", " ", text)
        text = " ".join(text.split())
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def fetch(root, lang, source):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = f"https://youtu.be/{source['id']}"
    audio = next(raw.glob(f"{lang}-full.*"), None)
    if audio is None:
        subprocess.run(["yt-dlp", "-q", "-f", "bestaudio", "-o", f"{lang}-full.%(ext)s", url],
                       cwd=raw, check=True)
        audio = next(raw.glob(f"{lang}-full.*"))
    vtt = raw / f"{lang}.{source['subs']}.vtt"
    if not vtt.exists():
        flag = "--write-auto-subs" if source["auto"] else "--write-subs"
        subprocess.run(["yt-dlp", "-q", "--skip-download", flag, "--sub-langs", source["subs"],
                        "--sub-format", "vtt", "-o", f"{lang}.%(ext)s", url], cwd=raw, check=True)
    return audio, vtt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/testset")
    root = Path(parser.parse_args().out)
    manifest = {}
    for lang, source in SOURCES.items():
        audio, vtt = fetch(root, lang, source)
        cues = parse_vtt(vtt, source["auto"])
        for name, offset in CLIPS.items():
            if not name.startswith(lang):
                continue
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(offset), "-t", str(LENGTH),
                            "-i", str(audio), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                            str(root / f"{name}.wav")], check=True)
            # Keep cues that start inside the clip; times are relative to the clip.
            clip = [{**cue, "start": round(cue["start"] - offset, 3),
                     "end": round(min(cue["end"], offset + LENGTH) - offset, 3)}
                    for cue in cues if offset <= cue["start"] < offset + LENGTH]
            joiner = "" if lang == "ja" else " "
            (root / f"{name}.ref.txt").write_text(joiner.join(c["text"] for c in clip) + "\n")
            (root / f"{name}.ref.json").write_text(json.dumps(clip, ensure_ascii=False, indent=1))
            manifest[name] = {"language": lang, "url": f"https://youtu.be/{source['id']}?t={offset}",
                              "offset": offset, "seconds": LENGTH, "cues": len(clip),
                              "reference": source["note"]}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
