from __future__ import annotations

import asyncio
from pathlib import Path

from pipeline.client import StreamEvent
from pipeline.config import VerifyCfg
from pipeline.models import AgentState, AgentStatus, Plan, PlanChunk, RunConfig
from pipeline.worker import build_initial_messages, run_agent


def _run(coro):
    return asyncio.run(coro)


def _chunk(**overrides) -> PlanChunk:
    defaults = dict(id="agent-1", title="Add auth", description="Implement the login flow.")
    defaults.update(overrides)
    return PlanChunk(**defaults)


def _make_state(chunk: PlanChunk, attempt: int = 1) -> AgentState:
    return AgentState(chunk=chunk, attempt=attempt)


def _make_config(repo: Path, verify_cfg: VerifyCfg | None = None) -> RunConfig:
    return RunConfig(
        repo=repo,
        task="test task",
        max_agent_turns=10,
        verify=verify_cfg or VerifyCfg(),
    )


def _make_initial_messages(chunk, plan, retry_context=""):
    return build_initial_messages(chunk, plan, retry_context)


class FakeClient:
    """Replays scripted turns for the worker loop."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    async def chat_stream(self, model, messages, **kwargs):
        self.calls.append((model, len(messages)))
        if not self.turns:
            raise AssertionError("model was called more times than the script allows")
        content, tool_calls = self.turns.pop(0)
        if isinstance(content, Exception):
            raise content
        yield StreamEvent(reasoning_delta="thinking...")
        if content:
            yield StreamEvent(content_delta=content)
        yield StreamEvent(finish_reason="stop", tool_calls=tool_calls or None, usage={"completion_tokens": 5})

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


# --- Existing message-building tests ---

def test_build_initial_messages_basic_shape():
    plan = Plan(chunks=[_chunk()])
    messages = build_initial_messages(plan.chunks[0], plan)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Implement the login flow." in messages[1]["content"]
    assert "Add auth" in messages[1]["content"]


def test_build_initial_messages_includes_scope_and_context():
    chunk = _chunk(scope=["auth/", "tests/auth/"], context="Use the existing JWT helper.")
    plan = Plan(chunks=[chunk])
    messages = build_initial_messages(chunk, plan)
    user = messages[1]["content"]
    assert "auth/" in user
    assert "tests/auth/" in user
    assert "Use the existing JWT helper." in user


def test_build_initial_messages_omits_scope_and_context_when_absent():
    chunk = _chunk()
    plan = Plan(chunks=[chunk])
    messages = build_initial_messages(chunk, plan)
    user = messages[1]["content"]
    assert "Expected scope" not in user
    assert "Additional context" not in user


def test_build_initial_messages_includes_shared_context():
    chunk = _chunk()
    plan = Plan(chunks=[chunk], shared_context="All agents must follow PEP 8.")
    messages = build_initial_messages(chunk, plan)
    assert "All agents must follow PEP 8." in messages[0]["content"]


def test_build_initial_messages_includes_retry_context_on_retry():
    chunk = _chunk()
    plan = Plan(chunks=[chunk])
    retry_note = "A previous attempt at this chunk was stopped: looped on the same file read."
    messages = build_initial_messages(chunk, plan, retry_context=retry_note)
    assert retry_note in messages[0]["content"]


def test_build_initial_messages_no_retry_context_by_default():
    chunk = _chunk()
    plan = Plan(chunks=[chunk])
    messages = build_initial_messages(chunk, plan)
    assert "previous attempt" not in messages[0]["content"]


# --- New loop tests ---

def test_submit_work_with_passing_verify_completes(tmp_path):
    """submit_work with passing verify -> COMPLETED."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path

    verify_cfg = VerifyCfg(command="echo ok", timeout_s=30.0)
    config = _make_config(tmp_path, verify_cfg=verify_cfg)

    submit_args = {"summary": "done", "files_changed": [], "verified": "ran echo ok", "blocked": ""}
    client = FakeClient([
        ("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}]),
    ])
    _run(run_agent(client, config, plan, state))
    assert state.status == AgentStatus.COMPLETED
    assert state.submitted == submit_args


def test_failing_verify_exhausts_budget_verify_failed(tmp_path):
    """Failing verify when repair attempts exceed max -> VERIFY_FAILED."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path

    verify_cfg = VerifyCfg(command="false", timeout_s=30.0, max_repair_attempts=1)
    config = _make_config(tmp_path, verify_cfg=verify_cfg)

    submit_args = {"summary": "done", "files_changed": [], "verified": "ran tests", "blocked": ""}
    # With max_repair_attempts=1, first failure increments to 1 (not > 1),
    # second failure increments to 2 (2 > 1) -> VERIFY_FAILED
    client = FakeClient([
        ("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}]),
        ("", [{"id": "s2", "name": "submit_work", "arguments": submit_args}]),
    ])
    _run(run_agent(client, config, plan, state))
    assert state.status == AgentStatus.VERIFY_FAILED
    assert "verification failed" in state.kill_reason
    assert state.repair_attempts == 2


def test_failing_verify_loop_continues(tmp_path):
    """After a verify failure the loop continues (more turns are taken)."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path

    # max_repair_attempts=2, so one failure is within budget and the loop continues
    verify_cfg = VerifyCfg(command="false", timeout_s=30.0, max_repair_attempts=2)
    config = _make_config(tmp_path, verify_cfg=verify_cfg)

    submit_args = {"summary": "done", "files_changed": [], "verified": "ran tests", "blocked": ""}
    # Give 2 submit_work calls. Both fail. repair_attempts becomes 2, which is NOT > 2,
    # so the loop continues. It tries a 3rd chat_stream call and our FakeClient raises
    # AssertionError because we exhausted the scripted turns.
    client = FakeClient([
        ("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}]),
        ("", [{"id": "s2", "name": "submit_work", "arguments": submit_args}]),
    ])
    try:
        _run(run_agent(client, config, plan, state))
        # Should not reach here — the loop should try a 3rd turn
        assert False, "expected AssertionError from loop continuation"
    except AssertionError as e:
        assert "more times" in str(e)
        assert state.repair_attempts == 2
        # The loop tried to continue, which is what we want to verify


def test_repair_message_in_conversation(tmp_path):
    """After a verify failure, the conversation has a repair message."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path

    verify_cfg = VerifyCfg(command="false", timeout_s=30.0, max_repair_attempts=1)
    config = _make_config(tmp_path, verify_cfg=verify_cfg)

    submit_args = {"summary": "done", "files_changed": [], "verified": "ran tests", "blocked": ""}
    # Need 2 submit calls so repair_attempts becomes 2 > 1 -> VERIFY_FAILED
    client = FakeClient([
        ("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}]),
        ("", [{"id": "s2", "name": "submit_work", "arguments": submit_args}]),
    ])
    _run(run_agent(client, config, plan, state))
    assert state.status == AgentStatus.VERIFY_FAILED
    repair_msgs = [m for m in state.messages if m.get("role") == "user" and "Verification failed" in m.get("content", "")]
    assert len(repair_msgs) > 0


def test_two_consecutive_no_tool_calls_fails_after_nudge(tmp_path):
    """Two turns with no tool calls -> FAILED after one nudge."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path

    config = _make_config(tmp_path)

    client = FakeClient([
        ("I'm thinking...", None),
        ("I'm still thinking...", None),
    ])
    _run(run_agent(client, config, plan, state))
    assert state.status == AgentStatus.FAILED
    assert "agent stopped producing tool calls" in state.kill_reason


# --- Turn-budget warning ---

def _run_shell_stub_turns(n):
    """n turns of a harmless tool call, so the agent neither finishes nor
    triggers the no-tool-call failure path."""
    return [("", [{"id": f"c{i}", "name": "list_dir", "arguments": {"path": "."}}])
            for i in range(n)]


def test_agent_is_warned_before_the_turn_ceiling(tmp_path):
    """An agent that hits max_agent_turns loses the whole chunk. It gets told
    to land the work while there is still budget to do it."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path
    config = _make_config(tmp_path)
    config.max_agent_turns = 12   # lead is 10, so the warning lands on turn 2

    client = FakeClient(_run_shell_stub_turns(12))
    _run(run_agent(client, config, plan, state))

    warnings = [m for m in state.messages
                if m.get("role") == "user" and "Budget warning" in (m.get("content") or "")]
    assert len(warnings) == 1, "the warning must fire exactly once, not every turn"
    assert "submit_work" in warnings[0]["content"]


def test_turn_warning_lands_with_budget_left_to_act_on_it(tmp_path):
    """A warning delivered on the final turn is useless - check where it lands."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path
    config = _make_config(tmp_path)
    config.max_agent_turns = 12

    turns_at_warning = []

    class WatchingClient(FakeClient):
        async def chat_stream(self, model, messages, **kwargs):
            if any("Budget warning" in (m.get("content") or "") for m in messages):
                turns_at_warning.append(state.turns)
            async for ev in super().chat_stream(model, messages, **kwargs):
                yield ev

    _run(run_agent(WatchingClient(_run_shell_stub_turns(12)), config, plan, state))
    assert turns_at_warning[0] == 2          # 12 - TURN_WARNING_LEAD
    assert config.max_agent_turns - turns_at_warning[0] == 10


def test_short_budget_still_gets_a_warning(tmp_path):
    """With fewer turns than the lead, warn immediately rather than never."""
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path
    config = _make_config(tmp_path)
    config.max_agent_turns = 3

    _run(run_agent(FakeClient(_run_shell_stub_turns(3)), config, plan, state))
    assert any("Budget warning" in (m.get("content") or "")
               for m in state.messages if m.get("role") == "user")


def test_agent_that_finishes_early_is_never_warned(tmp_path):
    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path
    config = _make_config(tmp_path)
    config.max_agent_turns = 60

    submit_args = {"summary": "done", "files_changed": [], "verified": "", "blocked": ""}
    _run(run_agent(FakeClient([("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}])]),
                   config, plan, state))
    assert state.status == AgentStatus.COMPLETED
    assert not any("Budget warning" in (m.get("content") or "") for m in state.messages)


def test_worker_emits_usage_per_turn(tmp_path):
    from pipeline.events import EventLog

    plan = Plan(chunks=[_chunk()])
    chunk = plan.chunks[0]
    state = _make_state(chunk)
    state.messages = _make_initial_messages(chunk, plan)
    state.workspace = tmp_path
    config = _make_config(tmp_path)
    events = EventLog(tmp_path / "events.jsonl", run_id="r")

    submit_args = {"summary": "done", "files_changed": [], "verified": "", "blocked": ""}
    client = FakeClient([
        ("", [{"id": "c1", "name": "list_dir", "arguments": {"path": "."}}]),
        ("", [{"id": "s1", "name": "submit_work", "arguments": submit_args}]),
    ])
    _run(run_agent(client, config, plan, state, events))

    usage = [e for e in EventLog.read(events.path) if e["kind"] == "usage"]
    assert len(usage) == 2
    assert all(e["role"] == "worker" and e["chunk"] == "agent-1" for e in usage)
    assert [e["turn"] for e in usage] == [1, 2]
