from __future__ import annotations

import json

import pytest

from pipeline.planner import _parse_plan


def test_parse_plan_from_dict_arguments():
    args = {
        "chunks": [
            {"id": "agent-1", "title": "Add auth", "description": "Implement login flow",
             "scope": ["auth/"], "context": "use JWT"},
            {"id": "agent-2", "title": "Add tests", "description": "Write tests for auth"},
        ],
        "shared_context": "Follow existing code style.",
    }
    plan = _parse_plan(args)
    assert len(plan.chunks) == 2
    assert plan.chunks[0].id == "agent-1"
    assert plan.chunks[0].scope == ["auth/"]
    assert plan.chunks[0].context == "use JWT"
    assert plan.chunks[1].scope == []  # defaults to empty list when omitted
    assert plan.chunks[1].context == ""
    assert plan.shared_context == "Follow existing code style."


def test_parse_plan_from_json_string_arguments():
    # llama-server's tool_call arguments arrive as a JSON string, not a dict -
    # this is the realistic path from client.py's tool-call accumulator.
    raw = json.dumps({"chunks": [{"id": "agent-1", "title": "T", "description": "D"}]})
    plan = _parse_plan(raw)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].title == "T"


def test_parse_plan_single_chunk_when_work_does_not_split():
    args = {"chunks": [{"id": "agent-1", "title": "Small fix", "description": "One-line change"}]}
    plan = _parse_plan(args)
    assert len(plan.chunks) == 1


def test_parse_plan_accepts_zero_chunks():
    """_parse_plan itself no longer rejects an empty plan - the zero-chunks
    check moved to _plan_structure_error so it's fed back to the model as a
    retry prompt instead of crashing the run outright."""
    plan = _parse_plan({"chunks": []})
    assert plan.chunks == []


def test_parse_plan_title_defaults_to_id_when_missing():
    args = {"chunks": [{"id": "agent-1", "description": "D"}]}
    plan = _parse_plan(args)
    assert plan.chunks[0].title == "agent-1"


def test_parse_plan_shared_context_defaults_to_empty():
    args = {"chunks": [{"id": "agent-1", "title": "T", "description": "D"}]}
    plan = _parse_plan(args)
    assert plan.shared_context == ""


# --- Plan structure validation ---

from pipeline.models import Plan as _Plan, PlanChunk as _PlanChunk, RunConfig as _RunConfig
from pipeline.planner import _plan_structure_error, _system_prompt


def _plan(*chunks) -> _Plan:
    return _Plan(chunks=list(chunks))


def _c(cid, depends_on=()) -> _PlanChunk:
    return _PlanChunk(id=cid, title=cid, description="d", depends_on=list(depends_on))


def test_valid_plan_has_no_structure_error():
    assert _plan_structure_error(_plan(_c("a"), _c("b", ["a"]))) is None


def test_zero_chunks_is_reported():
    problem = _plan_structure_error(_plan())
    assert problem and "zero chunks" in problem


def test_dangling_dependency_is_reported_with_the_real_ids():
    problem = _plan_structure_error(_plan(_c("a"), _c("b", ["nope"])))
    assert problem and "nope" in problem
    # The planner can only fix this if it is told what the ids actually are.
    assert "a" in problem and "b" in problem


def test_self_dependency_is_reported():
    problem = _plan_structure_error(_plan(_c("a", ["a"])))
    assert problem and "itself" in problem


def test_duplicate_ids_are_reported():
    problem = _plan_structure_error(_plan(_c("a"), _c("a")))
    assert problem and "more than once" in problem


def test_cycle_is_reported():
    problem = _plan_structure_error(_plan(_c("a", ["b"]), _c("b", ["a"])))
    assert problem is not None


def test_a_merely_badly_split_plan_is_not_rejected():
    """Structural validation must not second-guess the split - rejecting a plan
    a human would accept costs a whole exploration round-trip."""
    assert _plan_structure_error(_plan(_c("impl"), _c("tests"), _c("verify"))) is None


def test_planner_prompt_forbids_the_splits_that_produced_broken_runs(tmp_path):
    prompt = _system_prompt(_RunConfig(repo=tmp_path, task="t"))
    lowered = prompt.lower()
    # Splitting tests away from implementation: the test agent cannot see the
    # code it is testing, so it invents a mismatched API.
    assert "never create one chunk that writes code and another that writes the tests" in lowered
    # A verify-only chunk produces an empty diff and burns an agent; the harness
    # already runs verification after the merge.
    assert "never create a chunk whose job is to verify" in lowered
    # Two chunks editing one file conflict every time.
    assert "same file" in lowered


# --- create_plan: a structurally broken plan is sent back, not fatal ---

import asyncio as _asyncio

from pipeline.events import EventLog as _EventLog
from pipeline.planner import create_plan as _create_plan


class _FakePlannerClient:
    """Replays scripted chat_once completions."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = 0

    async def chat_once(self, model, messages, **kwargs):
        self.calls += 1
        if not self.completions:
            raise AssertionError("planner called more times than the script allows")
        return self.completions.pop(0)


def _submit(chunks, usage=None):
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"p{id(chunks)}", "type": "function", "function": {
                "name": "submit_plan", "arguments": json.dumps({"chunks": chunks})}},
        ]}}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 30},
    }


def test_create_plan_sends_an_unexecutable_plan_back_to_the_planner(tmp_path):
    """A dangling depends_on used to take the whole run down with a ValueError
    out of topological_waves, discarding every exploration turn that produced
    it. The planner is the only thing that can fix it and it is still right
    here, so ask."""
    bad = [{"id": "a", "title": "A", "description": "d"},
           {"id": "b", "title": "B", "description": "d", "depends_on": ["ghost"]}]
    good = [{"id": "a", "title": "A", "description": "d"},
            {"id": "b", "title": "B", "description": "d", "depends_on": ["a"]}]
    client = _FakePlannerClient([_submit(bad), _submit(good)])
    config = _RunConfig(repo=tmp_path, task="t")

    plan = _asyncio.run(_create_plan(client, config))
    assert client.calls == 2
    assert [c.id for c in plan.chunks] == ["a", "b"]
    assert plan.chunks[1].depends_on == ["a"]


def test_rejection_is_recorded_with_its_reason(tmp_path):
    bad = [{"id": "a", "title": "A", "description": "d", "depends_on": ["ghost"]}]
    good = [{"id": "a", "title": "A", "description": "d"}]
    events = _EventLog(tmp_path / "events.jsonl", run_id="r")
    _asyncio.run(_create_plan(_FakePlannerClient([_submit(bad), _submit(good)]),
                              _RunConfig(repo=tmp_path, task="t"), events))

    rejections = [e for e in _EventLog.read(events.path) if e["kind"] == "plan_rejected"]
    assert len(rejections) == 1
    assert "ghost" in rejections[0]["reason"]


def test_create_plan_emits_usage_per_call(tmp_path):
    events = _EventLog(tmp_path / "events.jsonl", run_id="r")
    good = [{"id": "a", "title": "A", "description": "d"}]
    _asyncio.run(_create_plan(_FakePlannerClient([_submit(good)]),
                              _RunConfig(repo=tmp_path, task="t"), events))

    (usage,) = [e for e in _EventLog.read(events.path) if e["kind"] == "usage"]
    assert usage["role"] == "planner"
    assert usage["reported"] is True
    assert usage["prompt_tokens"] == 100


# --- The forced final turn has to actually force ---

from pipeline.planner import FINAL_TURN_SYSTEM_PROMPT, MAX_EXPLORATION_TURNS


class _RecordingClient(_FakePlannerClient):
    def __init__(self, completions):
        super().__init__(completions)
        self.seen = []

    async def chat_once(self, model, messages, **kwargs):
        self.seen.append({"messages": [dict(m) for m in messages],
                          "tools": kwargs.get("tools"),
                          "tool_choice": kwargs.get("tool_choice")})
        return await super().chat_once(model, messages, **kwargs)


def _explore_turn():
    """A turn that calls a read-only tool instead of submitting."""
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function",
         "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_final_turn_withdraws_the_exploration_tools_and_the_instruction(tmp_path):
    """ds4 ignores tool_choice when the system prompt still says to explore
    first - it answered a forced submit_plan with a fabricated list_dir call.
    Forcing only works if both the other tools and that instruction are gone."""
    good = [{"id": "a", "title": "A", "description": "d"}]
    script = [_explore_turn()] * (MAX_EXPLORATION_TURNS - 1) + [_submit(good)]
    client = _RecordingClient(script)

    plan = _asyncio.run(_create_plan(client, _RunConfig(repo=tmp_path, task="t")))
    assert len(plan.chunks) == 1

    final = client.seen[-1]
    assert [t["function"]["name"] for t in final["tools"]] == ["submit_plan"]
    assert final["messages"][0]["content"] == FINAL_TURN_SYSTEM_PROMPT
    assert "Explore the repository first" not in final["messages"][0]["content"]
    assert final["tool_choice"]["function"]["name"] == "submit_plan"


def test_ordinary_turns_keep_the_read_only_tools(tmp_path):
    good = [{"id": "a", "title": "A", "description": "d"}]
    client = _RecordingClient([_explore_turn(), _submit(good)])
    _asyncio.run(_create_plan(client, _RunConfig(repo=tmp_path, task="t")))

    first = client.seen[0]
    assert len(first["tools"]) > 1
    assert first["tool_choice"] == "auto"
    assert "Explore the repository first" in first["messages"][0]["content"]


def test_conversation_history_survives_the_final_turn_swap(tmp_path):
    """Only the system message is replaced - everything the planner learned
    while exploring has to still be there, or forcing it to submit produces a
    plan built on nothing."""
    good = [{"id": "a", "title": "A", "description": "d"}]
    script = [_explore_turn()] * (MAX_EXPLORATION_TURNS - 1) + [_submit(good)]
    client = _RecordingClient(script)
    _asyncio.run(_create_plan(client, _RunConfig(repo=tmp_path, task="the original task")))

    final = client.seen[-1]
    assert final["messages"][1] == {"role": "user", "content": "the original task"}
    assert any(m.get("role") == "tool" for m in final["messages"])
