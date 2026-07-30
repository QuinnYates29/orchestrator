"""Token accounting: the estimator, the reported/estimated/unknown distinction,
and the fact that the client actually asks for usage in the first place."""
from __future__ import annotations

from pipeline.client import OrchestratorClient
from pipeline.events import EventLog
from pipeline.tokens import estimate_message_tokens, estimate_text_tokens, estimate_usage


def test_streaming_requests_usage():
    """The whole reason every token figure used to be unreported: nobody asked.

    Both fleet backends return usage on a stream, but only when
    stream_options.include_usage is set.
    """
    client = OrchestratorClient("http://x/v1", "http://x")
    body = client._build_body("m", [{"role": "user", "content": "hi"}], tools=None,
                              tool_choice=None, max_tokens=None, temperature=None, stream=True)
    assert body["stream_options"] == {"include_usage": True}


def test_non_streaming_does_not_send_stream_options():
    client = OrchestratorClient("http://x/v1", "http://x")
    body = client._build_body("m", [{"role": "user", "content": "hi"}], tools=None,
                              tool_choice=None, max_tokens=None, temperature=None, stream=False)
    assert "stream_options" not in body


def test_caller_can_override_stream_options():
    """A backend that rejects the field must remain drivable."""
    client = OrchestratorClient("http://x/v1", "http://x")
    body = client._build_body("m", [], tools=None, tool_choice=None, max_tokens=None,
                              temperature=None, stream=True, stream_options=None)
    assert body["stream_options"] is None


def test_estimate_scales_with_length():
    short = estimate_text_tokens("hello")
    long = estimate_text_tokens("hello " * 200)
    assert 0 < short < long


def test_empty_text_estimates_zero():
    assert estimate_text_tokens("") == 0


def test_message_estimate_includes_tool_call_payloads():
    """Tool-call arguments are frequently bigger than the prose around them; an
    estimator that ignores them understates a coding agent badly."""
    plain = [{"role": "assistant", "content": "ok"}]
    with_call = [{
        "role": "assistant", "content": "ok",
        "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "write_file", "arguments": '{"path": "a.py", "content": "' + "x" * 500 + '"}'}}],
    }]
    assert estimate_message_tokens(with_call) > estimate_message_tokens(plain) + 100


def test_estimate_counts_reasoning_separately():
    """reasoning_content is generated but never appears in the visible answer,
    and on ornith it dwarfs it."""
    without = estimate_usage([{"role": "user", "content": "q"}], "short answer")
    with_thinking = estimate_usage([{"role": "user", "content": "q"}], "short answer",
                                   reasoning_chars=40_000)
    assert with_thinking["completion_tokens"] > without["completion_tokens"] * 100


def test_estimated_usage_is_labelled_and_not_zero(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.emit_usage("worker", "ornith", None,
                   estimate={"prompt_tokens": 1200, "completion_tokens": 300})
    (event,) = [e for e in EventLog.read(log.path) if e["kind"] == "usage"]
    assert event["reported"] is False      # the backend did not count it
    assert event["estimated"] is True      # but we did
    assert event["prompt_tokens"] == 1200


def test_unreported_with_no_estimate_stays_zero_and_unlabelled(tmp_path):
    """Three states must stay distinguishable, not collapse into two."""
    log = EventLog(tmp_path / "events.jsonl")
    log.emit_usage("worker", "ornith", None)
    (event,) = [e for e in EventLog.read(log.path) if e["kind"] == "usage"]
    assert event["reported"] is False
    assert event["estimated"] is False
    assert event["prompt_tokens"] == 0


def test_reported_usage_wins_over_an_estimate(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.emit_usage("worker", "ornith", {"prompt_tokens": 10, "completion_tokens": 2},
                   estimate={"prompt_tokens": 9999, "completion_tokens": 9999})
    (event,) = [e for e in EventLog.read(log.path) if e["kind"] == "usage"]
    assert event["reported"] is True
    assert event["estimated"] is False
    assert event["prompt_tokens"] == 10


def test_token_totals_separates_estimated_from_uncounted(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.emit_usage("worker", "ornith", {"prompt_tokens": 100, "completion_tokens": 10})
    log.emit_usage("worker", "ornith", None, estimate={"prompt_tokens": 50, "completion_tokens": 5})
    log.emit_usage("worker", "ornith", None)
    totals = EventLog.token_totals(log.path)["ornith"]
    assert totals["calls"] == 3
    assert totals["unreported_calls"] == 2
    assert totals["estimated_calls"] == 1
    assert totals["prompt_tokens"] == 150  # the uncounted call contributes nothing
