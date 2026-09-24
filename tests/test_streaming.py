import numpy as np
import pytest

from r2d2.streaming import (FIRST, HOP, MAX_BUDGET, MAX_HOPS, SAFETY_RESET,
                            WINDOW, Stream, detect_hallucination,
                            normalize_punct, parse_text, rollback)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return "".join(tokens)


class Backend:
    tokenizer = Tokenizer()

    def __init__(self, output="甲乙"):
        self.output = output
        self.calls = []

    def decode(self, audio, prefix, language, context, budget):
        self.calls.append((len(audio), prefix, language, budget))
        return self.output


def test_first_lookahead_and_packet_boundaries():
    backend = Backend()
    stream = Stream(backend)
    assert stream.feed(np.zeros(HOP)) == []
    assert len(stream.feed(np.zeros(HOP))) == 1
    assert backend.calls[0][0] == FIRST
    assert stream.confirmed == "甲" and stream.draft == "乙"
    stream.feed(np.zeros(HOP))
    assert backend.calls[1][1] == "甲"
    assert stream.confirmed == "甲甲"


@pytest.mark.parametrize("length", [1, 1599, 2560, 5120, 5121, 7680])
def test_eos_preserves_short_tail_and_withheld_token(length):
    stream = Stream(Backend())
    stream.feed(np.zeros(length))
    before = stream.confirmed
    result = stream.finish()
    assert result["final"] and result["draft"] == ""
    assert result["text"].startswith(before)
    assert result["text"].endswith("乙")
    assert stream.processed == length


def test_empty_stop_does_not_invoke_model():
    backend = Backend()
    assert Stream(backend).finish()["text"] == ""
    assert backend.calls == []


def test_rolling_window_has_bounded_audio_and_stable_global_transcript():
    backend = Backend()
    stream = Stream(backend)
    previous = ""
    for _ in range(250):
        stream.feed(np.zeros(HOP))
        assert stream.confirmed.startswith(previous)
        previous = stream.confirmed
        assert len(stream.audio) <= WINDOW
    assert stream.window_start >= 250 * HOP - WINDOW
    assert len(stream.entries) <= WINDOW // HOP + 1
    assert stream.processed == 250 * HOP


def test_backlog_merges_hops_into_one_decode_without_losing_audio():
    backend = Backend()
    stream = Stream(backend)
    stream.feed(np.zeros(FIRST))
    assert [c[0] for c in backend.calls] == [FIRST]
    # A caller that is behind hands over several hops at once; they must become
    # one decode covering all of them, not one decode per hop.
    updates = stream.feed(np.zeros(HOP * MAX_HOPS))
    assert len(updates) == 1 and updates[0]["hops"] == MAX_HOPS
    assert stream.processed == FIRST + HOP * MAX_HOPS
    # Beyond the cap the remainder is still consumed, never dropped.
    backend.calls.clear()
    stream.feed(np.zeros(HOP * (MAX_HOPS + 2)))
    assert stream.processed == FIRST + HOP * (2 * MAX_HOPS + 2)
    assert all(len(stream.audio) <= WINDOW for _ in backend.calls)


def test_merged_step_scales_the_token_budget_with_the_audio_it_covers():
    backend = Backend()
    stream = Stream(backend)
    stream.feed(np.zeros(FIRST))
    single = backend.calls[-1][3]
    stream.feed(np.zeros(HOP * MAX_HOPS))
    assert backend.calls[-1][3] == min(MAX_BUDGET, single * MAX_HOPS)


def test_single_hop_feeding_never_merges():
    backend = Backend()
    stream = Stream(backend)
    stream.feed(np.zeros(FIRST))
    for _ in range(5):
        updates = stream.feed(np.zeros(HOP))
        assert len(updates) == 1 and updates[0]["hops"] == 1


def test_utf8_rollback_does_not_emit_replacement_character():
    class SplitTokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids, **kwargs):
            return "\ufffd" if len(ids) == 2 else "你"
    assert rollback("你好", SplitTokenizer()) == "你"


def test_auto_language_hides_incomplete_metadata():
    assert parse_text("language Chi", None) == ("", "")
    assert parse_text("language Chinese<asr_text>你 好", None) == ("Chinese", "你好")
    assert parse_text("language None<asr_text>", None) == ("", "")


def test_auto_language_is_retained_in_assistant_prefix():
    backend = Backend("language English<asr_text>Hello")
    stream = Stream(backend, language=None)
    stream.feed(np.zeros(FIRST))
    backend.output = " world"
    stream.feed(np.zeros(HOP))
    assert backend.calls[-1][1] == "language English<asr_text>Hell"
    assert stream.detected == "English"


def test_english_spaces_survive_prefix_continuation():
    backend = Backend("Hello ")
    stream = Stream(backend, language="English")
    stream.feed(np.zeros(FIRST))
    backend.output = " world!"
    stream.feed(np.zeros(HOP))
    assert stream.confirmed == "Hello world"


def test_punctuation_follows_the_script_of_the_preceding_character():
    # Upstream normalises the mark by what precedes it, not by the session language.
    assert normalize_punct("延迟,大概两百毫秒.") == "延迟，大概两百毫秒。"
    assert normalize_punct("latency is 200ms，ok") == "latency is 200ms,ok"
    assert normalize_punct("他说(小声)") == "他说（小声）"
    # A leading mark has nothing to take its script from, so it is left alone.
    assert normalize_punct(",开头") == ",开头"


@pytest.mark.parametrize("text", ["延迟,大概两百毫秒.", "这个 API 的 latency, 大概 200ms。", ""])
def test_punctuation_pass_is_length_preserving_and_idempotent(text):
    # Confirmed text is sliced by offset and never rewritten, so both must hold.
    once = normalize_punct(text)
    assert len(once) == len(text)
    assert normalize_punct(once) == once


def test_punctuation_is_normalised_against_the_prefix_not_the_generation_alone():
    stream = Stream(Backend(output=","))
    stream.feed(np.zeros(FIRST))
    stream.entries = [(stream.processed, "延迟")]
    stream.confirmed = "延迟"
    update = stream.feed(np.zeros(HOP))[0]
    # The comma opens the generation; only the prefix can say it follows a 汉字.
    # It is still the withheld token here, so read it from the draft side.
    combined = update["text"] + update["draft"]
    assert "，" in combined and "," not in combined


def test_repetition_loop_is_detected_and_pure_punctuation_is_not():
    assert detect_hallucination("好的。" * 5).startswith("tail_pattern:")
    assert detect_hallucination("Okay. O kay, okay. okay! OKAY").startswith("tail_pattern_norm:")
    assert detect_hallucination("，" * 10) == ""
    assert detect_hallucination("之前有顾客自己带酒水也没加收钱或者不让喝") == ""
    assert detect_hallucination("") == ""


def test_loop_rebuilds_decoder_state_but_keeps_confirmed_text():
    stream = Stream(Backend(output="好的。"))
    for _ in range(8):
        if any(u["reset"] for u in stream.feed(np.zeros(HOP))):
            break
    # State as left by the rebuild, before the next step refills it.
    assert stream.entries == [] and len(stream.audio) == 0
    assert stream.window_start == stream.processed == stream.reset_at
    # The loop is broken by discarding state, never by rewriting what was sent.
    # The trailing mark stays withheld each step, so the loop reads as 好的x5.
    assert stream.confirmed.startswith("好的") and len(stream.confirmed) >= 10


def test_reset_is_reported_on_the_update_that_triggered_it():
    stream = Stream(Backend(output="好的。"))
    reasons = []
    for _ in range(8):
        reasons += [u["reset"] for u in stream.feed(np.zeros(HOP)) if u["reset"]]
    assert reasons and reasons[0].startswith("tail_pattern")


def test_overlong_run_without_a_loop_still_rebuilds():
    stream = Stream(Backend(output="甲"))
    stream.feed(np.zeros(FIRST))
    # Jump the clock past the safety window instead of feeding 60 s of audio.
    stream.reset_at = stream.processed - (SAFETY_RESET + 1)
    update = stream.feed(np.zeros(HOP))[0]
    assert update["reset"].startswith("elapsed:")
    assert stream.entries == [] and stream.reset_at == stream.processed


@pytest.mark.parametrize("output,budget", [("갑을", 4), ("あい", 4), ("甲乙", 4), ("ab", 2)])
def test_dense_scripts_get_the_doubled_budget(output, budget):
    backend = Backend(output=output)
    stream = Stream(backend)
    stream.feed(np.zeros(FIRST))
    stream.feed(np.zeros(HOP))
    assert backend.calls[-1][3] == budget


def test_final_step_never_rebuilds():
    stream = Stream(Backend(output="好的。"))
    stream.feed(np.zeros(FIRST))
    stream.reset_at = stream.processed - (SAFETY_RESET + 1)
    assert stream.finish()["reset"] == ""


def test_rebuild_does_not_refire_on_the_tail_it_already_handled():
    # Regression: checking all of `confirmed` re-detected the same loop on every
    # step after a rebuild, leaving the decoder only one hop of audio each time.
    stream = Stream(Backend(output="好的。"))
    stream.feed(np.zeros(FIRST))
    resets = []
    for _ in range(20):
        resets += [u["step"] for u in stream.feed(np.zeros(HOP)) if u["reset"]]
    assert resets, "the loop should be caught at least once"
    # A fresh segment needs the pattern to repeat five times again before firing.
    assert all(b - a >= 4 for a, b in zip(resets, resets[1:]))


class Script:
    """A backend whose next generation is popped from a list."""
    tokenizer = Tokenizer()

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def decode(self, audio, prefix, language, context, budget):
        return self.outputs.pop(0) if self.outputs else ""


@pytest.mark.parametrize("before,after,joined", [
    ("The DOJ charged", "Chinese nationals", "The DOJ charged Chinese nationals"),
    ("the twenty-first century.", "in that relationship", "the twenty-first century. in that relationship"),
    ("그래서 우리가", "밥을 먹었어요", "그래서 우리가 밥을 먹었어요"),
    ("今天天气不错。", "我们出去吧", "今天天气不错。我们出去吧"),
    ("今日はいい天気", "ですね", "今日はいい天気ですね"),
])
def test_text_after_a_rebuild_is_spaced_only_in_spaced_scripts(before, after, joined):
    # Each output ends in a character the rollback withholds, so it confirms whole.
    stream = Stream(Script([before + "@", "@", after + "@"]))
    stream.feed(np.zeros(FIRST))
    assert stream.confirmed == before
    stream.reset_at = stream.processed - (SAFETY_RESET + 1)
    stream.feed(np.zeros(HOP))
    assert stream.entries == []
    stream.feed(np.zeros(HOP))
    assert stream.confirmed == joined
    # The model's own context keeps its text as generated.
    assert stream.entries[0][1] == after


def test_a_draft_after_a_rebuild_is_spaced_too():
    # One character: the rollback withholds it all, so it is draft only.
    stream = Stream(Script(["The DOJ charged@", "@", "C"]))
    stream.feed(np.zeros(FIRST))
    stream.reset_at = stream.processed - (SAFETY_RESET + 1)
    stream.feed(np.zeros(HOP))
    stream.feed(np.zeros(HOP))
    assert stream.confirmed == "The DOJ charged" and stream.draft == " C"
