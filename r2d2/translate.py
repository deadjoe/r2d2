"""Live translation of the streaming transcript into a chosen language.

Mirrors the transcript's two layers. A sentence the recogniser has confirmed
and closed is translated once and becomes settled text that is only ever
appended to. The open sentence, confirmed tail plus the recogniser's
grey draft, is re-translated latest-wins whenever it changes and shown as a
grey draft. Word order differs across languages, so a settled translation has
to wait for its sentence to close; the draft is what keeps the panel moving.

The model is Tencent HY-MT1.5-1.8B (Q4_K_M) in its own llama-server, on the
CPU on a Mac and on the GPU elsewhere (see DEVICE). Measured on an M1 Max: on
Metal, back-to-back translation pushed 63 % of Q8 recogniser steps past the
160 ms budget and forced merges; on the CPU the recogniser was
indistinguishable from running alone. On CUDA the GPU translated 5-10x faster
without changing recognition. See docs/translation.md.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time

from .backends import MODELS, start_llama_server, stop_llama_server

MT_PATH = MODELS / "HY-MT1.5-1.8B-GGUF"
MT_FILE = "HY-MT1.5-1.8B-Q4_K_M.gguf"
THREADS = int(os.environ.get("R2D2_MT_THREADS", "6"))
# Where HY-MT runs. On a Mac, the CPU: on Metal it pushed the recogniser past
# its step budget (see module doc). On CUDA, the GPU: 5-10x faster translation
# while recognition kept its accuracy and never merged a step (docs/translation.md).
# R2D2_MT_DEVICE overrides.
DEVICE = os.environ.get("R2D2_MT_DEVICE") or ("cpu" if sys.platform == "darwin" else "gpu")
if DEVICE not in ("cpu", "gpu"):
    raise ValueError(f"R2D2_MT_DEVICE must be cpu or gpu, not {DEVICE!r}")
# The model card's two templates with the model's own chat framing: the
# Chinese instruction for ZH<=>XX, naming the target in Chinese, and the
# English one for every other pair. Its contextual template was tried and
# rejected: the 1.8B model translates the supplied context into the output too.
TARGETS = {"Chinese": "中文", "English": "英语", "Japanese": "日语", "Korean": "韩语", "Spanish": "西班牙语"}
_FRAME = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>{}<｜hy_Assistant｜>"
_ZH_PAIR = "将以下文本翻译为{}，注意只需要输出翻译后的结果，不要额外解释：\n\n{}"
_OTHER_PAIR = "Translate the following segment into {}, without additional explanation.\n\n{}"

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
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
_KANA_HANGUL = re.compile(r"[぀-ヿ가-힯ᄀ-ᇿ]")
_LETTER = re.compile(r"[^\W\d_]")
_TRAILING_ELLIPSIS = re.compile(r"(?:…+|\.{3,})$")
# A draft is re-translated once it has grown by this much weighted text since
# the last translated draft (about four CJK characters or one or two words),
# or once it has stopped changing for DRAFT_IDLE seconds. Re-translating on
# every recogniser step kept the CPU translator busy about 75 % of the time,
# almost all of it on drafts that were replaced before anyone could read them.
DRAFT_STEP = 8
DRAFT_IDLE = 0.6


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


def is_chinese(text):
    """Han makes up most letters and there is no kana or Hangul."""
    letters = _LETTER.findall(text)
    return bool(letters) and not _KANA_HANGUL.search(text) and len(_HAN.findall(text)) * 2 >= len(letters)


def needs_translation(text, target="Chinese"):
    """False for text with nothing to translate, or already written in the
    target (auto mode, code-switching) as far as its script tells: Chinese by
    Han without kana or Hangul, Japanese by kana, Korean by Hangul. English
    and Spanish share a script, so they are left to the detected language."""
    if not _LETTER.search(text):
        return False
    if target == "Chinese":
        return not is_chinese(text)
    if target == "Japanese":
        return not _KANA.search(text)
    if target == "Korean":
        return not _HANGUL.search(text)
    return True


def prompt(text, target="Chinese"):
    if target == "Chinese" or is_chinese(text):
        return _FRAME.format(_ZH_PAIR.format(TARGETS[target], text))
    return _FRAME.format(_OTHER_PAIR.format(target, text))


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
            raise RuntimeError(f"Missing local translation model: {model}")
        self.process, self.client, self.log = start_llama_server(
            # -ngl 0 keeps the GPU queue for the recogniser; see DEVICE.
            [str(model), "-ngl", "99" if DEVICE == "gpu" else "0", "-t", str(THREADS),
             "-c", "2048", "--cache-ram", "0"],
            "translate-server.log", "Translation model", timeout=60)
        self.translate("Hello.")

    def translate(self, text, abort=None, target="Chinese"):
        """`text` in `target`, or None when `abort` (a threading.Event) is set
        first. Streams, so an abort closes the connection mid-generation and
        llama-server stops decoding instead of finishing a discarded draft."""
        # Greedy, not the card's sampling: a draft re-translated many times
        # must not change for no reason other than the dice.
        # cache_prompt reuses the slot's KV for the prefix shared with the last
        # call: the fixed template, and for a growing draft most of its source.
        # 30-40 % less time per call on the CPU. Reuse changes the batch split,
        # so wording can differ slightly from an uncached call; no quality
        # difference was seen, and a settled sentence identical to its last
        # draft still reuses that draft's output.
        body = {"prompt": prompt(text, target), "temperature": 0, "cache_prompt": True,
                "n_predict": min(256, 32 + 2 * len(text)), "stream": True}
        parts = []
        with self.client.stream("POST", "/completion", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if abort is not None and abort.is_set():
                    return None
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                parts.append(chunk.get("content", ""))
                if chunk.get("stop"):
                    break
        return "".join(parts)

    def close(self):
        stop_llama_server(self.process, self.client, self.log)
        self.process = self.client = self.log = None


class LiveTranslation:
    """Schedules translation of one streaming session.

    `translate` is an async (text, abort) -> str, returning None once the
    threading.Event `abort` is set; `emit` an async callback taking the update
    dict. Settled sentences always go first, in order; the draft is only
    translated when nothing settled is waiting, and only its newest version,
    so a slow translator falls behind in draft freshness, never in settled
    text order. A draft still being translated when a different sentence
    settles is aborted, so the settled sentence does not wait behind it.
    """

    def __init__(self, translate, emit, target="Chinese"):
        self._translate, self._emit, self.target = translate, emit, target
        # The recogniser's language, when known; text in the target passes through.
        self.language = ""
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
        self.pending_since = time.perf_counter()  # when pending_draft last changed
        self._drafting = None  # (source, abort event) of the draft call in flight
        self.final = False
        self.error = ""
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    def update(self, confirmed, draft="", final=False, language=""):
        self.language = language or self.language
        sentences, rest = split_sentences(confirmed[self.consumed:], final)
        now = time.perf_counter()
        for sentence in sentences:
            if sentence.strip():
                self.queue.append((sentence.strip(), now, self.consumed))
            self.consumed += len(sentence)
        pending = "" if final else (rest + draft).strip()
        if pending != self.pending_draft:
            self.pending_draft, self.pending_since = pending, now
        self.pending_start = self.consumed
        self.final = self.final or final
        if self._drafting and any(s.strip() != self._drafting[0] for s in sentences if s.strip()):
            # The draft in flight is about to be superseded; free the translator.
            self._drafting[1].set()
        self._wake.set()

    async def finish(self, timeout=30):
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except asyncio.TimeoutError:
            self.error = self.error or "Translation did not finish within 30 s of stopping; the remaining sentences were dropped"
            await self.close()

    async def close(self):
        if self._drafting:
            self._drafting[1].set()
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _call(self, source, abort=None):
        if self.language == self.target or not needs_translation(source, self.target):
            return source, 0.0
        begin = time.perf_counter()
        target = await self._translate(source, abort)
        return target, (time.perf_counter() - begin) * 1000

    def _draft_delay(self):
        """Seconds until the pending draft is worth translating; 0 for now."""
        pending, shown = self.pending_draft, self.draft_source
        if not pending:
            return 0.0
        common = 0
        while common < min(len(pending), len(shown)) and pending[common] == shown[common]:
            common += 1
        if weight(pending) - weight(pending[:common]) >= DRAFT_STEP:
            return 0.0
        return max(0.0, DRAFT_IDLE - (time.perf_counter() - self.pending_since))

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
                delay = None
                if self.pending_draft != self.draft_source:
                    delay = self._draft_delay()
                if delay == 0.0:
                    source, start = self.pending_draft, self.pending_start
                    abort = threading.Event()
                    self._drafting = (source, abort)
                    try:
                        target, ms = await self._call(source, abort) if source else ("", 0.0)
                    finally:
                        self._drafting = None
                    if target is None:
                        continue
                    if source:
                        self.last_draft = (source, target)
                    # A sentence that settled while this ran supersedes the draft.
                    if self.queue:
                        continue
                    self.draft, self.draft_source = tidy(source, target), source
                    self.draft_start = start
                    await self._emit(self._message(ms, None))
                    continue
                if self.final and delay is None:
                    return
                self._wake.clear()
                try:
                    # A draft too small to re-translate yet is picked up once it
                    # stops changing, even if no further update arrives.
                    await asyncio.wait_for(self._wake.wait(), delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Translation failed and has stopped (recognition continues): {exc}"
            await self._emit({"type": "translation_error", "message": self.error})

    def _message(self, ms, lag, final=False):
        return {"type": "translation", "text": self.text, "draft": self.draft,
                "translate_ms": round(ms, 1), "lag_ms": None if lag is None else round(lag),
                "segments": len(self.segments), "final": final}

    def snapshot(self):
        """The closing message after finish(): settled text only."""
        return {**self._message(0.0, None, final=True), "draft": "", "error": self.error,
                "sentences": self.segments}
