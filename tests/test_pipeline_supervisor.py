from __future__ import annotations

from pipeline.supervisor import detect_repetition


def test_detect_repetition_flags_literal_repeats():
    chunk = "Let me reconsider the approach carefully and think about edge cases. " * 1
    text = chunk * 5  # same ~70-char chunk repeated 5x, well over the window+min_repeats default
    assert detect_repetition(text, window_chars=50, min_repeats=3) is True


def test_detect_repetition_ignores_varied_reasoning():
    text = (
        "First I need to understand the requirements. "
        "The function takes an input list and should return the sorted output. "
        "I'll use a comparison-based sort here since stability matters. "
        "Let me check the edge case of an empty list. "
        "Now I'll write the implementation and verify it against the examples given. "
        "This looks correct, moving on to write the tests next."
    )
    assert detect_repetition(text, window_chars=50, min_repeats=3) is False


def test_detect_repetition_false_below_length_threshold():
    # Too short to possibly contain min_repeats non-overlapping windows.
    assert detect_repetition("short text", window_chars=400, min_repeats=3) is False


def test_detect_repetition_empty_string():
    assert detect_repetition("", window_chars=400, min_repeats=3) is False


def test_detect_repetition_default_thresholds_on_realistic_loop():
    # A ~280-char reasoning fragment repeated enough times to clear the
    # default gate (len >= window_chars * min_repeats = 400*3 = 1200).
    fragment = (
        "Wait, I need to reconsider this from the beginning. Let me re-examine "
        "whether the approach handles the null case correctly, because if it "
        "doesn't the whole chain of reasoning falls apart and I should restart "
        "the analysis from first principles before proceeding any further here."
    )
    assert len(fragment) * 4 < 1200, "fixture assumption changed - test no longer exercises the gate"
    text = fragment * 6
    assert detect_repetition(text) is True
