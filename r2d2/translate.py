"""Live translation of the streaming transcript into Chinese.

Mirrors the transcript's two layers. A sentence the recogniser has confirmed
and closed is translated once and becomes settled Chinese text that is only
ever appended to. The open sentence, confirmed tail plus the recogniser's
grey draft, is re-translated latest-wins whenever it changes and shown as a
grey draft. Word order differs across languages, so a settled translation has
to wait for its sentence to close; the draft is what keeps the panel moving.

The model is Tencent HY-MT1.5-1.8B (Q4_K_M) in its own llama-server on the
CPU. Measured on an M1 Max: on Metal, back-to-back translation pushed 63 % of
Q8 recogniser steps past the 160 ms budget and forced merges; on the CPU the
recogniser was indistinguishable from running alone. See docs/translation.md.
"""
from __future__ import annotations

import asyncio
import os
import re
import time

from .backends import MODELS, start_llama_server, stop_llama_server

MT_PATH = MODELS / "HY-MT1.5-1.8B-GGUF"
MT_FILE = "HY-MT1.5-1.8B-Q4_K_M.gguf"
THREADS = int(os.environ.get("R2D2_MT_THREADS", "6"))
# The model card's XX=>ZH template with the model's own chat framing. Its
# contextual template was tried and rejected: the 1.8B model translates the
# supplied context into the output as well.
PROMPT = ("<｜hy_begin▁of▁sentence｜><｜hy_User｜>"
          "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{}"
          "<｜hy_Assistant｜>")

# Marks that close a sentence immediately. A full stop is ambiguous (3.5, Mr.,
# "..."), so it closes only once whitespace follows it or the stream ends.
_HARD = "。！？!?"
_CLOSERS = "\"'”’」』）)】]"
_SOFT = "，,、；;：:"
# Weighted length (CJK counts double) past which an unclosed run is cut anyway,
# about seven seconds of speech in either script.
MAX_SEGMENT = 120
_WIDE = re.compile(r"[⺀-鿿가-힯぀-ヿ＀-￯]")
_HAN = re.compile(r"[一-鿿]")
_KANA_HANGUL = re.compile(r"[぀-ヿ가-힯ᄀ-ᇿ]")
_LETTER = re.compile(r"[^\W\d_]")
_TRAILING_ELLIPSIS = re.compile(r"(?:…+|\.{3,})$")


def weight(text):
    return len(text) + len(_WIDE.findall(text))


def split_sentences(text, final=False):
    """Split off every closed sentence at the front of `text`.

    Returns (sentences, rest). Sentences keep their surrounding whitespace so
    their lengths add back up to what was consumed; callers strip them.
    """
    sentences, start, i = [], 0, 0
    while i < len(text):
        char = text[i]
        end = None
        if char in _HARD:
            end = i + 1
        elif char == "." and i + 1 < len(text) and text[i + 1].isspace() and text[i - 1:i] != ".":
            end = i + 1
        if end is not None:
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            sentences.append(text[start:end])
            start = i = end
            continue
        i += 1
    rest = text[start:]
    while weight(rest) > MAX_SEGMENT:
        cut = _long_cut(rest)
        sentences.append(rest[:cut])
        rest = rest[cut:]
    if final and rest.strip():
        sentences.append(rest)
        rest = ""
    return sentences, rest


def _long_cut(text):
    """Where to break a run that never closed: last clause mark, else last
    space, inside the length limit; a hard cut only as the last resort."""
    limit, total = 0, 0
    for index, char in enumerate(text):
        total += 2 if _WIDE.match(char) else 1
        if total > MAX_SEGMENT:
            break
        limit = index + 1
    head = text[:limit]
    for marks in (_SOFT, " "):
        cut = max(head.rfind(mark) for mark in marks)
        if cut >= limit // 4:
            return cut + 1
    return limit


def needs_translation(text):
    """False for text that is already Chinese (auto mode, code-switching) or
    has nothing to translate. Kana or Hangul always means translate."""
    if _KANA_HANGUL.search(text):
        return True
    letters = _LETTER.findall(text)
    if not letters:
        return False
    return len(_HAN.findall(text)) * 2 < len(letters)


def tidy(source, target):
    target = target.strip()
    # The model marks an unfinished fragment with "……"; a sentence cut for
    # length or a draft is unfinished by construction, and the caret says so.
    if not _TRAILING_ELLIPSIS.search(source.strip()):
        target = _TRAILING_ELLIPSIS.sub("", target).rstrip()
    return target


class Translator:
    """HY-MT in a private llama-server, CPU only. Blocking; callers run it off
    the event loop in a thread of their own, never the recogniser's."""

    def __init__(self):
        self.process = self.client = self.log = None

    @staticmethod
    def available():
        return (MT_PATH / MT_FILE).is_file()

    def load(self):
        model = MT_PATH / MT_FILE
        if not model.is_file():
            raise RuntimeError(f"缺少本地翻译模型：{model}")
        self.process, self.client, self.log = start_llama_server(
            # -ngl 0 keeps the Metal queue for the recogniser; see module doc.
            [str(model), "-ngl", "0", "-t", str(THREADS), "-c", "2048", "--cache-ram", "0"],
            "translate-server.log", "翻译模型", timeout=60)
        self.translate("Hello.")

    def translate(self, text):
        # Greedy, not the card's sampling: a draft re-translated many times
        # must not change for no reason other than the dice.
        response = self.client.post("/completion", json={
            "prompt": PROMPT.format(text), "temperature": 0, "cache_prompt": False,
            "n_predict": min(256, 32 + 2 * len(text)), "stream": False,
        })
        response.raise_for_status()
        return response.json()["content"]

    def close(self):
        stop_llama_server(self.process, self.client, self.log)
        self.process = self.client = self.log = None


class LiveTranslation:
    """Schedules translation of one streaming session.

    `translate` is an async str -> str; `emit` an async callback taking the
    update dict. Settled sentences always go first, in order; the draft is
    only translated when nothing settled is waiting, and only its newest
    version, so a slow translator falls behind in draft freshness, never in
    settled text order.
    """

    def __init__(self, translate, emit):
        self._translate, self._emit = translate, emit
        self.consumed = 0  # chars of the transcript's confirmed text already segmented
        self.queue = []    # (source, confirmed_at, start offset) waiting to be settled
        self.text = ""
        self.segments = []
        self.draft = ""
        self.draft_source = ""
        self.pending_draft = ""
        # Offsets into the confirmed text where the pending and the shown draft begin.
        self.pending_start = self.draft_start = 0
        self.last_draft = ("", "")  # (source, raw target): reused if that sentence then closes
        self.final = False
        self.error = ""
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    def update(self, confirmed, draft="", final=False):
        sentences, rest = split_sentences(confirmed[self.consumed:], final)
        now = time.perf_counter()
        for sentence in sentences:
            if sentence.strip():
                self.queue.append((sentence.strip(), now, self.consumed))
            self.consumed += len(sentence)
        self.pending_draft = "" if final else (rest + draft).strip()
        self.pending_start = self.consumed
        self.final = self.final or final
        self._wake.set()

    async def finish(self, timeout=30):
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except asyncio.TimeoutError:
            self.error = self.error or "翻译未能在停止后 30 秒内完成，已放弃余下句子"
            await self.close()

    async def close(self):
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _call(self, source):
        if not needs_translation(source):
            return source, 0.0
        begin = time.perf_counter()
        target = await self._translate(source)
        return target, (time.perf_counter() - begin) * 1000

    async def _run(self):
        try:
            while True:
                if self.queue:
                    source, confirmed_at, start = self.queue.pop(0)
                    if self.last_draft[0] == source:
                        # The sentence closed exactly as last drafted: same
                        # prompt, same greedy output, no second call needed.
                        target, ms = self.last_draft[1], 0.0
                    else:
                        target, ms = await self._call(source)
                    target = tidy(source, target)
                    self.text += target
                    lag = (time.perf_counter() - confirmed_at) * 1000
                    self.segments.append({"source": source, "target": target,
                                          "translate_ms": round(ms, 1), "lag_ms": round(lag)})
                    if self.draft and self.draft_start <= start:
                        # That draft covered the sentence just settled, in
                        # whatever wording the recogniser had then; showing it
                        # would print the sentence twice until redrafted.
                        self.draft, self.draft_source = "", ""
                    await self._emit(self._message(ms, lag))
                    continue
                if self.pending_draft != self.draft_source:
                    source, start = self.pending_draft, self.pending_start
                    target, ms = await self._call(source) if source else ("", 0.0)
                    if source:
                        self.last_draft = (source, target)
                    # A sentence that settled while this ran supersedes the draft.
                    if self.queue:
                        continue
                    self.draft, self.draft_source = tidy(source, target), source
                    self.draft_start = start
                    await self._emit(self._message(ms, None))
                    continue
                if self.final:
                    return
                self._wake.clear()
                await self._wake.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"翻译出错，已停止翻译（识别继续）：{exc}"
            await self._emit({"type": "translation_error", "message": self.error})

    def _message(self, ms, lag, final=False):
        return {"type": "translation", "text": self.text, "draft": self.draft,
                "translate_ms": round(ms, 1), "lag_ms": None if lag is None else round(lag),
                "segments": len(self.segments), "final": final}

    def snapshot(self):
        """The closing message after finish(): settled text only."""
        return {**self._message(0.0, None, final=True), "draft": "", "error": self.error,
                "sentences": self.segments}
