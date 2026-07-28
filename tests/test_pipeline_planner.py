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


def test_parse_plan_rejects_zero_chunks():
    with pytest.raises(RuntimeError, match="zero chunks"):
        _parse_plan({"chunks": []})


def test_parse_plan_title_defaults_to_id_when_missing():
    args = {"chunks": [{"id": "agent-1", "description": "D"}]}
    plan = _parse_plan(args)
    assert plan.chunks[0].title == "agent-1"


def test_parse_plan_shared_context_defaults_to_empty():
    args = {"chunks": [{"id": "agent-1", "title": "T", "description": "D"}]}
    plan = _parse_plan(args)
    assert plan.shared_context == ""
