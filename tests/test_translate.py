import asyncio

import pytest

from r2d2.translate import LiveTranslation, MAX_SEGMENT, needs_translation, split_sentences, tidy, weight


@pytest.mark.parametrize("text, sentences, rest", [
    ("Hello there. How are", ["Hello there."], " How are"),
    # A full stop closes only once whitespace follows: the next token may be a digit.
    ("It costs 3.", [], "It costs 3."),
    ("It costs 3.5 dollars. Ok", ["It costs 3.5 dollars."], " Ok"),
    ("Really? Yes!", ["Really?", " Yes!"], ""),
    ("Wait... what", [], "Wait... what"),
    ("今日は暑い。明日は", ["今日は暑い。"], "明日は"),
    ("他说「好的！」然后", ["他说「好的！」"], "然后"),
    ("¿Dónde está? Aquí", ["¿Dónde está?"], " Aquí"),
])
def test_sentences_close_on_final_marks_only(text, sentences, rest):
    assert split_sentences(text) == (sentences, rest)


def test_final_flush_hands_over_the_open_sentence():
    assert split_sentences("Hello there. And then", final=True) == (["Hello there.", " And then"], "")
    assert split_sentences("It costs 3.", final=True) == (["It costs 3."], "")
    assert split_sentences("   ", final=True) == ([], "   ")


def test_unclosed_run_is_cut_at_a_clause_mark_within_the_limit():
    words = "so we went there, and then we kept talking about the plan " * 4
    sentences, rest = split_sentences(words)
    assert sentences and all(weight(s) <= MAX_SEGMENT for s in sentences)
    assert sentences[0].endswith(",")
    assert "".join(sentences) + rest == words
    wide = "あ" * 200
    sentences, rest = split_sentences(wide)
    assert "".join(sentences) + rest == wide and weight(rest) <= MAX_SEGMENT


@pytest.mark.parametrize("text, expected", [
    ("Hello world", True), ("今日は晴れ", True), ("안녕하세요", True), ("今天天气不错", False),
    ("我们今天用Python写代码", False), ("123, 456.", False), ("OK。", True),
])
def test_only_text_that_is_not_already_chinese_is_translated(text, expected):
    assert needs_translation(text) is expected


def test_tidy_drops_the_fragment_ellipsis_unless_the_source_trails_off():
    assert tidy("So the thing is, we", "所以，实际上，我们是……") == "所以，实际上，我们是"
    assert tidy("And then...", "然后……") == "然后……"


class FakeMT:
    def __init__(self, delay=0.0):
        self.calls, self.delay = [], delay

    async def __call__(self, text):
        self.calls.append(text)
        await asyncio.sleep(self.delay)
        return f"<{text}>"


async def drive(steps, delay=0.0):
    mt, messages = FakeMT(delay), []

    async def emit(message):
        messages.append(message)

    live = LiveTranslation(mt, emit)
    for confirmed, draft in steps:
        live.update(confirmed, draft)
        await asyncio.sleep(0)
    live.update(steps[-1][0], final=True)
    await live.finish(timeout=5)
    return live, mt, messages


def test_settled_translation_only_appends_and_keeps_sentence_order():
    steps = [("Hello", " there"), ("Hello there.", ""), ("Hello there. How", " are"),
             ("Hello there. How are you?", ""), ("Hello there. How are you? Fine", "")]
    live, mt, messages = asyncio.run(drive(steps, delay=0.01))
    settled = [m["text"] for m in messages if m["type"] == "translation"]
    assert all(b.startswith(a) for a, b in zip(settled, settled[1:]))
    assert live.text == "<Hello there.><How are you?><Fine>"
    assert [s["source"] for s in live.segments] == ["Hello there.", "How are you?", "Fine"]
    assert live.snapshot()["final"] and live.snapshot()["draft"] == ""


def test_a_draft_identical_to_the_closed_sentence_is_not_translated_twice():
    async def run():
        mt, messages = FakeMT(), []

        async def emit(message):
            messages.append(message)

        live = LiveTranslation(mt, emit)
        live.update("Good morning", ".")
        await asyncio.sleep(0.01)
        assert live.draft == "<Good morning.>"
        live.update("Good morning. ", "")
        live.update("Good morning. ", final=True)
        await live.finish(timeout=5)
        return live, mt

    live, mt = asyncio.run(run())
    assert mt.calls == ["Good morning."]
    assert live.segments[0]["translate_ms"] == 0 and live.text == "<Good morning.>"


def test_slow_translator_skips_stale_drafts_but_never_settled_text():
    steps = [("", "w" * n) for n in range(1, 30)] + [("Done.", "")]
    live, mt, _ = asyncio.run(drive(steps, delay=0.02))
    assert len(mt.calls) < 10
    assert live.text == "<Done.>"


def test_chinese_passes_through_without_a_model_call():
    live, mt, _ = asyncio.run(drive([("今天天气不错。Nice day.", "")]))
    assert live.text == "今天天气不错。<Nice day.>"
    assert mt.calls == ["Nice day."]


def test_a_translator_failure_is_reported_once_and_recognition_is_unaffected():
    async def run():
        messages = []

        async def boom(text):
            raise RuntimeError("down")

        async def emit(message):
            messages.append(message)

        live = LiveTranslation(boom, emit)
        live.update("Hello.", "")
        live.update("Hello. Again.", "", final=True)
        await live.finish(timeout=5)
        return live, messages

    live, messages = asyncio.run(run())
    assert [m["type"] for m in messages] == ["translation_error"]
    assert "down" in live.snapshot()["error"]


def test_settling_drops_a_draft_worded_differently_from_the_settled_sentence():
    async def run():
        mt, messages = FakeMT(), []

        async def emit(message):
            messages.append(message)

        live = LiveTranslation(mt, emit)
        live.update("I spent ten", " years")
        await asyncio.sleep(0.01)
        assert live.draft == "<I spent ten years>"
        live.update("I spent 10 years here. And", "")
        await asyncio.sleep(0.01)
        return messages

    messages = asyncio.run(run())
    settled = next(m for m in messages if m["lag_ms"] is not None)
    assert settled["text"] == "<I spent 10 years here.>" and settled["draft"] == ""
