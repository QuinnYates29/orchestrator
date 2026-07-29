from __future__ import annotations

import asyncio

import pytest

from pipeline.client import OrchestratorError, StreamEvent
from pipeline.events import EventLog
from pipeline.solo import ensure_model_resident, run_solo


class FakeClient:
    """Replays scripted turns. Each turn is (content, tool_calls, usage);
    tool_calls uses the already-accumulated shape chat_stream yields."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.loads = []

    async def chat_stream(self, model, messages, **kwargs):
        self.calls.append((model, len(messages)))
        if not self.turns:
            raise AssertionError("model was called more times than the script allows")
        content, tool_calls, usage = self.turns.pop(0)
        if isinstance(content, Exception):
            raise content
        yield StreamEvent(reasoning_delta="thinking...")
        if content:
            yield StreamEvent(content_delta=content)
        yield StreamEvent(finish_reason="stop", tool_calls=tool_calls or None, usage=usage)

    async def admin_load(self, model, profile=None, wait_s=180.0):
        self.loads.append((model, profile))
        return {"ok": True}


def _run(coro):
    return asyncio.run(coro)


def test_stops_when_model_stops_calling_tools(tmp_path):
    client = FakeClient([("all done, I changed nothing", None, {"completion_tokens": 7})])
    result = _run(run_solo(client, model="ds4-full", repo=tmp_path, task="do a thing"))
    assert result.ok
    assert result.stop_reason == "finished"
    assert result.turns == 1
    assert result.final_message == "all done, I changed nothing"
    assert result.completion_tokens == 7


def test_executes_tool_calls_and_feeds_results_back(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    client = FakeClient([
        ("", [{"id": "c1", "name": "read_file", "arguments": {"path": "a.txt"}}], None),
        ("read it", None, None),
    ])
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="read a.txt"))
    assert result.ok
    assert result.tool_calls == 1
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "hello"
    # Second turn must see the grown conversation, not a fresh one.
    assert client.calls[1][1] > client.calls[0][1]


def test_writes_are_applied_to_the_real_directory(tmp_path):
    client = FakeClient([
        ("", [{"id": "c1", "name": "write_file",
               "arguments": {"path": "out/new.txt", "content": "written"}}], None),
        ("done", None, None),
    ])
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="write a file"))
    assert result.ok
    assert (tmp_path / "out" / "new.txt").read_text() == "written"


def test_turn_ceiling_is_not_reported_as_success(tmp_path):
    tool_turn = ("", [{"id": "c", "name": "list_dir", "arguments": {"path": "."}}], None)
    client = FakeClient([tool_turn] * 3)
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="loop", max_turns=3))
    assert not result.ok
    assert result.stop_reason == "max_turns"
    assert result.turns == 3


def test_orchestrator_error_ends_the_run_without_raising(tmp_path):
    client = FakeClient([(OrchestratorError(503, {"error": "model_not_resident"}), None, None)])
    result = _run(run_solo(client, model="ds4-full", repo=tmp_path, task="x"))
    assert not result.ok
    assert result.stop_reason == "error"
    assert "503" in result.final_message


def test_tokens_accumulate_across_turns(tmp_path):
    client = FakeClient([
        ("", [{"id": "c1", "name": "list_dir", "arguments": {}}],
         {"prompt_tokens": 100, "completion_tokens": 10}),
        ("done", None, {"prompt_tokens": 150, "completion_tokens": 5}),
    ])
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="x"))
    assert result.prompt_tokens == 250
    assert result.completion_tokens == 15


def test_events_are_written(tmp_path):
    events = EventLog(tmp_path / "events.jsonl", run_id="r1")
    client = FakeClient([
        ("", [{"id": "c1", "name": "list_dir", "arguments": {}}], {"completion_tokens": 3}),
        ("done", None, None),
    ])
    _run(run_solo(client, model="ornith", repo=tmp_path, task="x", events=events))
    kinds = [e["kind"] for e in EventLog.read(events.path)]
    assert kinds[0] == "solo_start"
    assert "tool_call" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "solo_end"


def test_unknown_tool_is_reported_to_the_model_not_fatal(tmp_path):
    client = FakeClient([
        ("", [{"id": "c1", "name": "teleport", "arguments": {}}], None),
        ("ok, my mistake", None, None),
    ])
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="x"))
    assert result.ok
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]


@pytest.mark.parametrize("model,expect_profile_attempt", [
    ("ds4-full", True),
    ("ornith", False),
])
def test_residency_tries_profile_then_falls_back(model, expect_profile_attempt):
    class Rejecting(FakeClient):
        async def admin_load(self, model, profile=None, wait_s=180.0):
            self.loads.append((model, profile))
            if profile:
                raise OrchestratorError(400, {"error": "unknown launch profile"})
            return {"ok": True}

    client = Rejecting([])
    _run(ensure_model_resident(client, model))
    if expect_profile_attempt:
        assert client.loads[0] == ("ds4", "ds4-full")   # profile attempted first
        assert client.loads[-1] == (model, None)        # then plain load
    else:
        assert client.loads == [("ornith", None)]


def test_residency_stops_after_successful_profile_load():
    client = FakeClient([])
    _run(ensure_model_resident(client, "ds4-full"))
    assert client.loads == [("ds4", "ds4-full")]


def test_empty_turn_is_nudged_not_reported_as_finished(tmp_path):
    """Ornith reasons at length and then returns an empty message. Treating
    that as a finished task reports a phase complete having written nothing."""
    (tmp_path / "a.txt").write_text("hello")
    client = FakeClient([
        ("", None, None),                                             # empty
        ("", [{"id": "c1", "name": "read_file",
               "arguments": {"path": "a.txt"}}], None),               # recovers
        ("done", None, None),
    ])
    result = _run(run_solo(client, model="ornith", repo=tmp_path, task="read a.txt"))
    assert result.ok
    assert result.stop_reason == "finished"
    assert result.tool_calls == 1
    nudged = [m for m in result.messages
              if m["role"] == "user" and "empty" in (m["content"] or "")]
    assert len(nudged) == 1
    assert not any(m["role"] == "assistant" and not m.get("content")
                   and not m.get("tool_calls") for m in result.messages)


def test_persistent_empty_turns_fail_rather_than_claim_success(tmp_path):
    client = FakeClient([("", None, None)] * 4)
    result = _run(run_solo(client, model="ornith", repo=tmp_path,
                           task="x", max_empty_retries=2))
    assert not result.ok
    assert result.stop_reason == "empty_response"
    assert "empty responses" in result.final_message


def test_whitespace_only_content_counts_as_empty(tmp_path):
    client = FakeClient([("   \n  ", None, None)] * 4)
    result = _run(run_solo(client, model="ornith", repo=tmp_path,
                           task="x", max_empty_retries=1))
    assert not result.ok
    assert result.stop_reason == "empty_response"
