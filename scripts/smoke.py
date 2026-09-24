"""Replay PCM in real time through the actual web API. Writes timing evidence."""
import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
import wave

import websockets


async def run(args):
    with wave.open(args.audio) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
    if args.repeat > 1:
        pcm *= args.repeat
    async with websockets.connect(args.url, max_size=2**20, proxy=None) as ws:
        await ws.send(json.dumps({"backend": args.backend, "language": args.language,
                                  "translate": args.translate}))
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "error":
                raise RuntimeError(message)
            if message["type"] == "ready":
                break
        start = time.perf_counter()

        async def send():
            for pos in range(0, len(pcm), 5120):
                part = pcm[pos:pos + 5120]
                due = (pos + len(part)) / 32000
                await asyncio.sleep(max(0, due - (time.perf_counter() - start)))
                await ws.send(part)
            await ws.send("stop")

        sender = asyncio.create_task(send())
        events = []
        async for payload in ws:
            event = json.loads(payload)
            event["wall_ms"] = round((time.perf_counter() - start) * 1000, 1)
            events.append(event)
            if event["type"] == "error":
                sender.cancel()
                raise RuntimeError(event)
            if event["type"] == "transcript":
                print(json.dumps({k: event.get(k) for k in ("text", "draft", "decode_ms", "backlog_ms")}, ensure_ascii=False), flush=True)
            if event["type"] in ("translation", "translation_error"):
                print("MT", json.dumps({k: event.get(k) for k in ("text", "draft", "translate_ms", "lag_ms", "message")}, ensure_ascii=False), flush=True)
        await sender
    updates = [e for e in events if e["type"] == "transcript"]
    assert updates and updates[-1]["final"] and events[-1]["type"] == "done"
    for prev, next_ in zip(updates, updates[1:]):
        assert next_["text"].startswith(prev["text"]), "confirmed text must never be rewritten"
    summary = {"backend": args.backend, "language": args.language, "audio_seconds": len(pcm) / 32000,
               "text": updates[-1]["text"], "updates": len(updates),
               "first_text_ms": next((e["wall_ms"] for e in updates if e["text"]), None),
               "median_decode_ms": statistics.median(e["decode_ms"] for e in updates),
               "max_backlog_ms": max(e["backlog_ms"] for e in updates),
               "wall_seconds": events[-1]["wall_ms"] / 1000}
    translations = [e for e in events if e["type"] == "translation"]
    if args.translate not in ("off", args.language):
        assert translations and translations[-1]["final"], "translation must settle before done"
        for prev, next_ in zip(translations, translations[1:]):
            assert next_["text"].startswith(prev["text"]), "settled translation must never be rewritten"
        sentences = translations[-1]["sentences"]
        lags = [s["lag_ms"] for s in sentences]
        calls = [s["translate_ms"] for s in sentences if s["translate_ms"]]
        summary["translation"] = {
            "text": translations[-1]["text"], "sentences": len(sentences), "updates": len(translations),
            "median_lag_ms": statistics.median(lags) if lags else None, "max_lag_ms": max(lags, default=None),
            "median_translate_ms": statistics.median(calls) if calls else None,
            "drafts": sum(1 for e in translations if e["lag_ms"] is None and not e["final"]),
            "last_translation_after_final_ms": round(translations[-1]["wall_ms"] - updates[-1]["wall_ms"])}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "events": events}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gguf", "gguf_q8", "gguf_q4", "mlx"], required=True)
    parser.add_argument("--audio", default="tests/fixtures/official-test.wav")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--translate", nargs="?", const="Chinese", default="off",
                        choices=["off", "Chinese", "English", "Japanese", "Korean", "Spanish"],
                        help="also stream a translation (bare flag: Chinese)")
    parser.add_argument("--url", default="ws://localhost:8765/api/stream")
    parser.add_argument("--output", default="artifacts/smoke.json")
    asyncio.run(run(parser.parse_args()))
