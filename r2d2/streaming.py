# Portions adapted from Copyright 2026 The NetEase Youdao team.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral adaptation of Youdao's streaming_transcribe_no_reset policy.

160 ms hops, 160 ms initial lookahead, rolling audio window with eviction,
one-token rollback, adaptive decode budget, context-aware punctuation, and the
decoder rebuild upstream performs on a repetition loop or an over-long run.
No silence gate or hidden denoising.

Two deliberate departures from upstream, both measured on Apple Silicon:
the window is shorter than upstream's 16 s / 8 s because encoder and prefill
cost scale with it, and a backlog merges several hops into one decode instead
of letting the queue overflow. See docs/streaming.md.

A rebuild discards only the state the next step decodes from. Confirmed text is
kept and never rewritten, so this is the loop-breaking half of upstream's
handling, not its repetition rewriter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
import numpy as np

RATE = 16000
HOP = 2560
FIRST = 5120
# Audio kept in front of the model. Cost per step is roughly linear in this
# length on both backends, so it is the main real-time lever.
WINDOW = 8 * RATE
EVICT = 4 * RATE
# Most hops merged into a single decode while catching up, and the ceiling on
# the per-step token budget that merging scales up.
MAX_HOPS = 3
MAX_BUDGET = 16
# Upstream SOFT_RESET_SEC on the no-VAD path: a stream that never sees a silence
# boundary is rebuilt anyway, so accumulated drift cannot feed itself forever.
SAFETY_RESET = 60 * RATE
# Scripts dense enough that one token covers about one spoken syllable. Upstream
# doubles the budget for Chinese only; Japanese passed through its kanji, while
# Korean fell to the English budget and lagged until the window evicted speech.
_DENSE = "一-鿿぀-ヿㄱ-ㆎ가-힯ᄀ-ᇿ"
_DENSE_TAIL = re.compile(rf"[{_DENSE}][^a-zA-Z{_DENSE}]*$")
# Scripts written with spaces between words. A rebuilt decoder starts a fresh
# utterance, whose first word carries no leading space.
_SPACED_END = re.compile(r"[0-9A-Za-zÀ-ɏ가-힯,.!?;:)\"'%]$")
_SPACED_START = re.compile(r"^[0-9A-Za-zÀ-ɏ가-힯(\"']")

_EN2ZH_PUNCT = {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：", "(": "（", ")": "）"}
_ZH2EN_PUNCT = {v: k for k, v in _EN2ZH_PUNCT.items()}
_ALL_PUNCT = re.compile(r"[,\.!?;:()，。！？；：（）]")
# Upstream's hallucination alphabet, wider than the pairs normalised above.
_PUNCT_CHARS = "，。！？、；：,.!?;:~…·\"'()（）《》—-"
_PUNCT_RE = re.compile(r"[%s\s]+" % re.escape(_PUNCT_CHARS))


def normalize_punct(text):
    """Upstream _normalize_punct_by_context: each mark follows the script of the
    character before it. One char in, one char out, so slice offsets taken
    against the text stay valid and re-running it on settled text is a no-op.
    """
    def replace(match: re.Match) -> str:
        punct = str(match.group())
        for index in range(match.start() - 1, -1, -1):
            if not text[index].isspace():
                previous = text[index]
                break
        else:
            return punct
        if "一" <= previous <= "鿿":
            return _EN2ZH_PUNCT.get(punct, punct)
        if previous.isascii() and (previous.isalnum() or previous in "\"'"):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct

    return _ALL_PUNCT.sub(replace, text)


def detect_hallucination(text, threshold=5, max_pattern=50, tail=256):
    """Upstream detect_hallucination: a tail that is the same pattern repeated
    `threshold` times is a decoder loop, not speech. Returns a reason or "".
    Reports only; the caller decides what to do with the text already emitted.
    """
    if not text:
        return ""
    text = text[-tail:]
    for size in range(1, max_pattern + 1):
        if len(text) < size * threshold:
            continue
        pattern = text[-size:]
        if not pattern.strip(_PUNCT_CHARS + " \t"):
            continue
        if all(text[-(r + 1) * size:-r * size] == pattern for r in range(1, threshold)):
            return f"tail_pattern:'{pattern}'x{threshold}+"
    # Same search once punctuation and case stop hiding a loop ("Okay. O kay, okay.").
    plain = _PUNCT_RE.sub("", text).lower()
    for size in range(3, max_pattern + 1):
        if len(plain) < size * threshold:
            continue
        pattern = plain[-size:]
        if all(plain[-(r + 1) * size:-r * size] == pattern for r in range(1, threshold)):
            return f"tail_pattern_norm:'{pattern}'x{threshold}+"
    return ""


def rollback(text, tokenizer, count=1):
    ids = tokenizer.encode(text, add_special_tokens=False)
    end = max(0, len(ids) - count)
    while end:
        result = tokenizer.decode(ids[:end], skip_special_tokens=False)
        if "�" not in result:
            return result
        end -= 1
    return ""


def parse_text(raw, language):
    raw = raw.split("|")[0].replace("�", "")
    if "<asr_text>" in raw:
        meta, raw = raw.split("<asr_text>", 1)
        match = re.search(r"language\s+(\w+)", meta)
        language = match.group(1) if match else language
    elif not language:
        return "", ""
    if language == "None":
        return "", ""
    if language == "Chinese":
        raw = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", raw)
    return language or "", raw


@dataclass
class Stream:
    backend: object
    language: str | None = "Chinese"
    context: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    pending: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    entries: list = field(default_factory=list)
    confirmed: str = ""
    draft: str = ""
    detected: str = ""
    processed: int = 0
    window_start: int = 0
    steps: int = 0
    budget: float = 4
    reset_at: int = 0
    # Offset into `confirmed` where the current decoder state began. Upstream
    # checks for loops in last_fixed_asr_text, which a rebuild clears; checking
    # all of `confirmed` would re-fire on the same tail after every rebuild.
    segment: int = 0

    def feed(self, pcm):
        self.pending = np.concatenate([self.pending, pcm])
        results = []
        required = FIRST if self.steps == 0 else HOP
        while len(self.pending) >= required:
            take, hops = required, 1
            if self.steps:
                # Audio waiting in `pending` means the caller is behind. Encoder
                # and prefill are paid once per decode regardless of how much
                # audio advanced, so merging hops is what actually catches up.
                hops = min(MAX_HOPS, len(self.pending) // HOP)
                take = HOP * hops
            chunk, self.pending = self.pending[:take], self.pending[take:]
            results.append(self._step(chunk, hops=hops))
            required = HOP
        return results

    def _step(self, chunk, final=False, hops=1):
        begin = time.perf_counter()
        self.audio = np.concatenate([self.audio, chunk])
        self.processed += len(chunk)
        # Entries carry audio positions, avoiding the official hard-coded first 49 / next 50 count.
        while len(self.audio) > WINDOW:
            self.audio = self.audio[EVICT:]
            self.window_start += EVICT
            self.entries = [(end, text) for end, text in self.entries if end > self.window_start]
        prefix_text = "".join(text for _, text in self.entries)
        prefix = prefix_text
        if not self.language and self.detected:
            prefix = f"language {self.detected}<asr_text>" + prefix
        # A merged step covers several hops of speech, so it needs proportionally
        # more tokens to stay caught up in text as well as in audio.
        budget = MAX_BUDGET if final else min(MAX_BUDGET, int(self.budget) * hops)
        generated = self.backend.decode(
            self.audio, prefix, self.language, self.context, budget
        )
        # Normalising prefix and generation together is what lets a mark that
        # opens the generation see the character before it; the pass is length
        # preserving and idempotent, so the settled prefix comes back unchanged.
        raw = normalize_punct(prefix + generated).split("|")[0].replace("�", "")
        lang, text = parse_text(raw, self.language)
        fixed_raw = raw if final else rollback(raw, self.backend.tokenizer)
        _, fixed = parse_text(fixed_raw, self.language)
        self.detected = lang or self.detected
        delta = fixed[len(prefix_text):] if fixed.startswith(prefix_text) else ""
        self.entries.append((self.processed, delta))
        local_fixed = prefix_text + delta
        self.draft = text[len(local_fixed):] if text.startswith(local_fixed) else ""
        if final:
            self.draft = ""
        # After a rebuild the model's context is empty, so the first text of the
        # new segment would be glued to the last word before it ("chargedChinese").
        # The model keeps its own text in `entries`; only the transcript gets the space.
        if not prefix_text and _SPACED_END.search(self.confirmed):
            if _SPACED_START.search(delta):
                delta = " " + delta
            elif not delta and _SPACED_START.search(self.draft):
                self.draft = " " + self.draft
        self.confirmed += delta
        # Same adaptive budget as ws_server.py: base 2, dense script x2, upper bound 4.
        dense = bool(_DENSE_TAIL.search(self.confirmed))
        self.budget = 2 if delta or dense else self.budget + 0.5
        if dense:
            self.budget *= 2
        self.budget = min(4, self.budget)
        self.steps += 1
        # Upstream rebuilds the decoder on a loop or on an over-long run. Text
        # already confirmed is kept and never rewritten; only the state the next
        # step decodes from is discarded, which is what breaks a loop.
        reset = "" if final else detect_hallucination(self.confirmed[self.segment:])
        if not reset and not final and self.processed - self.reset_at > SAFETY_RESET:
            reset = f"elapsed:{(self.processed - self.reset_at) // RATE}s"
        if reset:
            self.audio = np.empty(0, np.float32)
            self.entries = []
            self.draft = ""
            self.window_start = self.processed
            self.reset_at = self.processed
            self.segment = len(self.confirmed)
            self.budget = 4
        return {"type": "transcript", "text": self.confirmed, "draft": self.draft,
                "language": self.detected, "audio_ms": round(self.processed / RATE * 1000),
                "decode_ms": round((time.perf_counter() - begin) * 1000, 1),
                "step": self.steps, "final": final, "hops": hops, "reset": reset,
                "timings": getattr(self.backend, "timings", None)}

    def finish(self):
        # Deliberate UX fix vs upstream: decode even on an exact hop boundary so
        # the withheld final token is not lost. No artificial silence is appended.
        if not self.processed and not len(self.pending):
            return {"type": "transcript", "text": "", "draft": "", "final": True,
                    "language": "", "audio_ms": 0, "decode_ms": 0, "step": 0, "hops": 0,
                    "reset": ""}
        tail, self.pending = self.pending, np.empty(0, np.float32)
        return self._step(tail, final=True)
