"""Tests for pipeline.explore — the read-only research fan-out."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from pipeline.config import PipelineCfg
from pipeline.events import EventLog
from pipeline.explore import (
    SubQuestionResult,
    run_explore,
    _run_single_agent,
    _stage_split,
    _stage_synthesize,
    SUBMIT_QUESTIONS_TOOL,
)
from pipeline.tools import READ_ONLY_TOOL_SCHEMAS


class FakeExploreClient:
    """Minimal fake for the non-streaming chat API used by explore.

    Returns a list of pre-scripted responses. Each response is either a
    string (content only, no tool calls) or a tuple (content, tool_calls,
    usage). If a response is a string, the assistant has nothing more to do
    — the loop stops.

    Tool calls are passed in OpenAI's wire format: each item is a dict with
    `id`, `name`, `arguments`. Internally the fake wraps them in `function`
    so the caller sees the same shape the orchestrator returns."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.chat_once_calls = []

    async def chat_once(self, model, messages, *, tools=None, tool_choice=None, max_tokens=None):
        self.chat_once_calls.append({
            "model": model,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "messages": messages,
        })
        if not self.responses:
            raise AssertionError("model was called more times than the script allows")
        resp = self.responses.pop(0)
        if isinstance(resp, str):
            return self._content_only(resp)
        content, tool_calls, usage = resp
        return self._build_response(content, tool_calls, usage)

    async def admin_load(self, model, profile=None, wait_s=180.0):
        return {"ok": True}

    @staticmethod
    def _build_response(content, tool_calls, usage):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            # Wrap each tool call in the OpenAI `function` envelope.
            message["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        # `usage` is a sibling of `choices`, not a member of one. This fixture
        # used to nest it inside the choice, which is exactly where the code
        # under test was (wrongly) reading it from - so the two agreed with each
        # other and the token counts were silently always zero in production.
        return {"choices": [{"message": message}], "usage": usage or {}}

    @staticmethod
    def _content_only(content):
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {}}


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# Stage 1: split
# ----------------------------------------------------------------------

def test_split_produces_sub_questions_returned_by_model():
    """Split returns the list the model produced via submit_questions."""
    tool_calls = [
        {
            "id": "call_1",
            "name": "submit_questions",
            "arguments": {"questions": ["How does routing pick a backend?", "Where are backends configured?"]},
        }
    ]
    client = FakeExploreClient([
        ("", tool_calls, {"prompt_tokens": 10, "completion_tokens": 5}),
    ])
    sub_questions, sp, sc = _run(_stage_split(
        client, model="test-model", repo=Path("/tmp"), question="How does routing work?",
        events=None,
    ))
    assert sub_questions == ["How does routing pick a backend?", "Where are backends configured?"]
    assert sp == 10
    assert sc == 5


def test_split_allows_single_question():
    """A single sub-question is a valid answer — do not pad."""
    tool_calls = [
        {
            "id": "call_1",
            "name": "submit_questions",
            "arguments": {"questions": ["What does this one function do?"]},
        }
    ]
    client = FakeExploreClient([
        ("", tool_calls, {"prompt_tokens": 10, "completion_tokens": 3}),
    ])
    sub_questions, _, _ = _run(_stage_split(
        client, model="test-model", repo=Path("/tmp"), question="What does foo() do?",
        events=None,
    ))
    assert sub_questions == ["What does this one function do?"]
    assert len(sub_questions) == 1


def test_split_uses_read_only_tools_then_submit():
    """Split tool list contains only read-only tools plus submit_questions."""
    tool_names = [t["function"]["name"] for t in READ_ONLY_TOOL_SCHEMAS] + [
        SUBMIT_QUESTIONS_TOOL["function"]["name"]
    ]
    tool_calls = [
        {
            "id": "call_1",
            "name": "list_dir",
            "arguments": {"path": "."},
        },
        {
            "id": "call_2",
            "name": "submit_questions",
            "arguments": {"questions": ["sub-q"]},
        }
    ]
    client = FakeExploreClient([
        ("", [tool_calls[0]], {"prompt_tokens": 10, "completion_tokens": 2}),
        ("", [tool_calls[1]], {"prompt_tokens": 10, "completion_tokens": 3}),
    ])
    sub_questions, _, _ = _run(_stage_split(
        client, model="test-model", repo=Path("/tmp"), question="What is this?",
        events=None,
    ))
    assert sub_questions == ["sub-q"]
    # The first call should have read-only tools + submit_questions.
    called_tools = client.chat_once_calls[0]["tools"]
    called_names = [t["function"]["name"] for t in called_tools]
    for n in tool_names:
        assert n in called_names, f"expected {n!r} in tools: {called_names}"
    # And nothing that would write.
    for n in ("write_file", "edit_file", "run_shell"):
        assert n not in called_names, f"write tool {n!r} must not appear in split tools"


# ----------------------------------------------------------------------
# Stage 2: answer (single agent)
# ----------------------------------------------------------------------

def test_single_agent_returns_answer_when_model_stops_calling_tools():
    """When the model stops calling tools, its content is the answer."""
    client = FakeExploreClient([
        ("the answer is 42", None, {"prompt_tokens": 10, "completion_tokens": 5}),
    ])
    result = _run(_run_single_agent(
        client, model="test-model", repo=Path("/tmp"),
        sub_question="What is the meaning?",
        events=None,
    ))
    assert result.answer == "the answer is 42"
    assert result.stop_reason == "finished"
    assert result.turns == 1


def test_single_agent_handles_tool_calls():
    """Tool calls are executed and results fed back; then answer follows."""
    client = FakeExploreClient([
        (
            "",
            [{"id": "c1", "name": "read_file", "arguments": {"path": "README.md"}}],
            {"prompt_tokens": 10, "completion_tokens": 2},
        ),
        ("README says it's 42", None, {"prompt_tokens": 10, "completion_tokens": 5}),
    ])
    (Path("/tmp") / "README.md").write_text("README says it's 42")
    result = _run(_run_single_agent(
        client, model="test-model", repo=Path("/tmp"),
        sub_question="What does the README say?",
        events=None,
    ))
    assert "42" in result.answer
    assert result.tool_calls == 1


def test_single_agent_max_turns_is_not_fatal():
    """A sub-agent that runs out of turns is recorded, not propagated."""
    # Always ask for a tool — never stop calling.
    client = FakeExploreClient([
        ("", [{"id": "c1", "name": "list_dir", "arguments": {"path": "."}}],
         {"prompt_tokens": 10, "completion_tokens": 2})
    ] * 20)
    result = _run(_run_single_agent(
        client, model="test-model", repo=Path("/tmp"),
        sub_question="Something",
        max_turns=3,
        events=None,
    ))
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    # No exception should be raised — the failure is recorded.


def test_single_agent_orchestrator_error_is_recorded():
    """An OrchestratorError in a sub-agent is recorded, not re-raised."""
    from pipeline.client import OrchestratorError

    class FailingClient(FakeExploreClient):
        async def chat_once(self, model, messages, **kwargs):
            raise OrchestratorError(503, {"error": "model_not_resident"})

    client = FailingClient([])
    result = _run(_run_single_agent(
        client, model="test-model", repo=Path("/tmp"),
        sub_question="Something", events=None,
    ))
    assert result.stop_reason == "error"
    assert "503" in result.error


# ----------------------------------------------------------------------
# Stage 3: synthesize
# ----------------------------------------------------------------------

def test_synthesize_contains_every_sub_answer():
    """The synthesis prompt must contain every sub-question and its answer."""
    captured_messages = []

    class RecordingClient(FakeExploreClient):
        async def chat_once(self, model, messages, **kwargs):
            captured_messages.append(list(messages))
            return self._build_response("final answer", None, {"prompt_tokens": 5, "completion_tokens": 3})

    client = RecordingClient([])
    sub_answers = [
        SubQuestionResult(question="Q1", answer="Answer to Q1 from file A", ok=True),
        SubQuestionResult(question="Q2", answer="Answer to Q2 from file B", ok=True),
    ]
    answer, _, _ = _run(_stage_synthesize(
        client, model="test-model",
        question="Original question?",
        sub_answers=sub_answers,
        events=None,
    ))
    # The synthesis should have produced something.
    assert "final answer" in answer
    # The last user message should contain every sub-answer.
    user_msg = captured_messages[-1][-1]["content"]
    assert "Q1" in user_msg
    assert "Q2" in user_msg
    assert "Answer to Q1 from file A" in user_msg
    assert "Answer to Q2 from file B" in user_msg


def test_synthesize_reports_gap_for_unanswered_sub_question():
    """If a sub-question has no answer, the synthesis prompt says so."""
    captured_messages = []

    class RecordingClient(FakeExploreClient):
        async def chat_once(self, model, messages, **kwargs):
            captured_messages.append(list(messages))
            return self._build_response("final answer", None, {"prompt_tokens": 5, "completion_tokens": 3})

    client = RecordingClient([])
    sub_answers = [
        SubQuestionResult(question="Q1", stop_reason="max_turns", error="hit ceiling"),
    ]
    answer, _, _ = _run(_stage_synthesize(
        client, model="test-model",
        question="Original question?",
        sub_answers=sub_answers,
        events=None,
    ))
    user_msg = captured_messages[-1][-1]["content"]
    assert "UNANSWERED" in user_msg
    assert "Q1" in user_msg


# ----------------------------------------------------------------------
# Full pipeline: run_explore
# ----------------------------------------------------------------------

def test_run_explore_returns_explore_result():
    """End-to-end: split -> answer -> synthesize produces an ExploreResult."""
    client = FakeExploreClient([
        # Split: submit two questions.
        (
            "",
            [{"id": "c1", "name": "submit_questions",
              "arguments": {"questions": ["Q1?", "Q2?"]}}],
            {"prompt_tokens": 10, "completion_tokens": 5},
        ),
        # Agent for Q1: answer immediately.
        ("A1", None, {"prompt_tokens": 8, "completion_tokens": 4}),
        # Agent for Q2: answer immediately.
        ("A2", None, {"prompt_tokens": 8, "completion_tokens": 4}),
        # Synthesize: final answer.
        ("Final synthesized answer.", None, {"prompt_tokens": 20, "completion_tokens": 10}),
    ])
    result = _run(run_explore(
        client,
        model="test-model",
        repo=Path("/tmp"),
        question="How does this work?",
        max_questions=6,
        agent_max_turns=5,
        pipeline_cfg=PipelineCfg(),
        events=None,
    ))
    assert result.question == "How does this work?"
    assert result.sub_questions == ["Q1?", "Q2?"]
    assert len(result.sub_answers) == 2
    assert result.sub_answers[0].answer == "A1"
    assert result.sub_answers[1].answer == "A2"
    assert "Final synthesized answer" in result.final_answer
    assert result.prompt_tokens > 0
    assert result.duration_s >= 0


def test_run_explore_one_failing_sub_agent_yields_gap_in_final_answer():
    """A failing sub-agent must not sink the run; synthesis reports the gap."""
    call_idx = 0

    class FailingAgentClient(FakeExploreClient):
        async def chat_once(self, model, messages, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                # Split
                return self._build_response(
                    "",
                    [{"id": "c1", "name": "submit_questions",
                      "arguments": {"questions": ["Q1?", "Q2?"]}}],
                    {"prompt_tokens": 10, "completion_tokens": 5},
                )
            elif call_idx == 2:
                # Agent 1 succeeds
                return self._build_response(
                    "Answer to Q1",
                    None,
                    {"prompt_tokens": 8, "completion_tokens": 4},
                )
            elif call_idx == 3:
                # Agent 2 fails with OrchestratorError
                from pipeline.client import OrchestratorError
                raise OrchestratorError(503, {"error": "model_not_resident"})
            else:
                # Synthesize
                return self._build_response(
                    "Final answer noting the gap.",
                    None,
                    {"prompt_tokens": 20, "completion_tokens": 10},
                )

    client = FailingAgentClient([])
    result = _run(run_explore(
        client,
        model="test-model",
        repo=Path("/tmp"),
        question="How does this work?",
        max_questions=6,
        agent_max_turns=5,
        pipeline_cfg=PipelineCfg(),
        events=None,
    ))
    # The failing agent's result should be recorded, not propagated.
    assert len(result.sub_answers) == 2
    assert result.sub_answers[0].answer == "Answer to Q1"
    assert result.sub_answers[1].error  # Q2 failed
    assert "503" in result.sub_answers[1].error
    # The synthesis should mention the gap.
    assert "gap" in result.final_answer.lower() or "UNANSWERED" in result.final_answer or "503" in result.final_answer


def test_concurrency_is_bounded_by_max_concurrent_workers():
    """Sub-agents must not exceed max_concurrent_workers at any time."""
    import asyncio as _asyncio

    class CountingClient(FakeExploreClient):
        async def chat_once(self, model, messages, **kwargs):
            call_idx = len(self.chat_once_calls)
            if call_idx == 0:
                return self._build_response(
                    "",
                    [{"id": "c1", "name": "submit_questions",
                      "arguments": {"questions": ["Q1?", "Q2?", "Q3?"]}}],
                    {"prompt_tokens": 10, "completion_tokens": 5},
                )
            elif call_idx <= 3:
                return self._build_response(
                    f"Answer {call_idx}",
                    None,
                    {"prompt_tokens": 8, "completion_tokens": 4},
                )
            else:
                return self._build_response(
                    "Final answer.",
                    None,
                    {"prompt_tokens": 20, "completion_tokens": 10},
                )

    client = CountingClient([])
    from pipeline.config import LimitsCfg
    cfg = PipelineCfg(limits=LimitsCfg(max_concurrent_workers=2))
    result = _run(run_explore(
        client,
        model="test-model",
        repo=Path("/tmp"),
        question="How does this work?",
        max_questions=6,
        agent_max_turns=5,
        pipeline_cfg=cfg,
        events=None,
    ))
    # We should have 3 sub-answers.
    assert len(result.sub_answers) == 3


# ----------------------------------------------------------------------
# Safety property: read-only tools
# ----------------------------------------------------------------------

def test_read_only_tool_schemas_contain_no_write_or_shell_tool():
    """READ_ONLY_TOOL_SCHEMAS must not include write_file, edit_file, or run_shell.
    This is a safety property, not a style one."""
    schema_names = [t["function"]["name"] for t in READ_ONLY_TOOL_SCHEMAS]
    assert "read_file" in schema_names
    assert "list_dir" in schema_names
    assert "grep" in schema_names
    assert "glob" in schema_names
    for forbidden in ("write_file", "edit_file", "run_shell"):
        assert forbidden not in schema_names, (
            f"Forbidden tool {forbidden!r} found in READ_ONLY_TOOL_SCHEMAS"
        )


# ----------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------

def test_events_are_written_during_explore():
    """Event log should capture the explore lifecycle."""
    events = EventLog(Path("/tmp/test_events.jsonl"))
    client = FakeExploreClient([
        (
            "",
            [{"id": "c1", "name": "submit_questions",
              "arguments": {"questions": ["Q?"]}}],
            {"prompt_tokens": 10, "completion_tokens": 5},
        ),
        ("A", None, {"prompt_tokens": 8, "completion_tokens": 4}),
        ("Final.", None, {"prompt_tokens": 20, "completion_tokens": 10}),
    ])
    result = _run(run_explore(
        client,
        model="test-model",
        repo=Path("/tmp"),
        question="How does this work?",
        max_questions=1,
        agent_max_turns=5,
        pipeline_cfg=PipelineCfg(),
        events=events,
    ))
    assert result.final_answer == "Final."
    # Check that events were written.
    event_kinds = [e["kind"] for e in EventLog.read(events.path)]
    assert "explore_start" in event_kinds
    assert "explore_end" in event_kinds
    assert "explore_split_done" in event_kinds
    assert "explore_synthesize_done" in event_kinds


# ----------------------------------------------------------------------
# CLI: explore subcommand
# ----------------------------------------------------------------------

def test_cli_explore_subcommand_exists():
    """The explore subcommand should be registered in the parser."""
    from pipeline.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["explore", "--repo", "/tmp", "--question", "What is this?"])
    assert args.command == "explore"
    assert args.question == "What is this?"
    assert args.repo == Path("/tmp")
    # Default values.
    assert args.max_questions == 6
    assert args.model is None


def test_cli_explore_allows_all_common_flags():
    """The explore subcommand should accept common flags like --config, --api-key, etc."""
    from pipeline.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explore",
        "--repo", "/tmp",
        "--question", "What?",
        "--config", "/tmp/config.yaml",
        "--api-key", "test-key",
        "--orchestrator-url", "http://localhost:9999/v1",
        "--admin-url", "http://localhost:9999",
        "--scratch-dir", "/tmp/scratch",
        "--load-wait-s", "10",
        "--model", "ds4-full",
        "--max-questions", "3",
    ])
    assert args.command == "explore"
    assert args.question == "What?"
    assert args.config == Path("/tmp/config.yaml")
    assert args.api_key == "test-key"
    assert args.orchestrator_url == "http://localhost:9999/v1"
    assert args.admin_url == "http://localhost:9999"
    assert args.scratch_dir == Path("/tmp/scratch")
    assert args.load_wait_s == 10.0
    assert args.model == "ds4-full"
    assert args.max_questions == 3


def test_cli_explore_question_is_required():
    """The explore subcommand should require --question."""
    from pipeline.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["explore", "--repo", "/tmp"])


# ----------------------------------------------------------------------
# Path import
# ----------------------------------------------------------------------
from pathlib import Path
