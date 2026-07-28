from __future__ import annotations

from pipeline.models import Plan, PlanChunk
from pipeline.worker import build_initial_messages


def _chunk(**overrides) -> PlanChunk:
    defaults = dict(id="agent-1", title="Add auth", description="Implement the login flow.")
    defaults.update(overrides)
    return PlanChunk(**defaults)


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
