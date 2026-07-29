"""Tests for pipeline.state: round-trip, atomic write, truncation, find_runs,
resume-selection, and token totals."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pipeline.events import EventLog
from pipeline.models import AgentStatus, ChunkOutcome, Plan, PlanChunk, VerifyResult
from pipeline.state import (
    StateCorruptError,
    RunState,
    find_runs,
    load_state,
    save_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, depends_on: list[str] | None = None, title: str | None = None):
    return PlanChunk(
        id=id,
        title=title or f"Chunk {id}",
        description=f"desc for {id}",
        depends_on=depends_on or [],
    )


def _outcome(chunk, status=AgentStatus.COMPLETED, workspace=None,
             kill_reason="", attempts=1):
    return ChunkOutcome(
        chunk=chunk,
        status=status,
        workspace=workspace,
        kill_reason=kill_reason,
        attempts=attempts,
        verify=None,
    )


# ---------------------------------------------------------------------------
# Round-trip with depends_on and mixed statuses
# ---------------------------------------------------------------------------


def test_round_trip_preserves_depends_on_and_statuses(tmp_path):
    """A plan with depends_on survives a save/load cycle unchanged."""
    a = _chunk("a")
    b = _chunk("b", depends_on=["a"])
    c = _chunk("c", depends_on=["a", "b"])
    plan = Plan(chunks=[a, b, c], shared_context="x")

    outcome_a = _outcome(a, AgentStatus.COMPLETED)
    outcome_b = _outcome(b, AgentStatus.FAILED, kill_reason="boom")
    outcome_c = _outcome(c, AgentStatus.SKIPPED, kill_reason="dep failed")

    path = tmp_path / "run1" / "state.json"
    save_state(path, "run1", "build feature X", tmp_path / "repo", plan,
               [outcome_a, outcome_b, outcome_c], "deadbeef")

    assert path.exists()
    state = load_state(path)

    assert state.run_id == "run1"
    assert state.task == "build feature X"
    assert state.repo == tmp_path / "repo"
    assert state.base_commit == "deadbeef"

    chunks_by_id = {c.id: c for c in state.plan.chunks}
    assert set(chunks_by_id) == {"a", "b", "c"}
    assert chunks_by_id["b"].depends_on == ["a"]
    assert set(chunks_by_id["c"].depends_on) == {"a", "b"}

    outcomes_by_id = {o.chunk.id: o for o in state.outcomes}
    assert outcomes_by_id["a"].status == AgentStatus.COMPLETED
    assert outcomes_by_id["b"].status == AgentStatus.FAILED
    assert outcomes_by_id["b"].kill_reason == "boom"
    assert outcomes_by_id["c"].status == AgentStatus.SKIPPED


# ---------------------------------------------------------------------------
# Atomic write: no partial file on failure
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_file_on_failure(tmp_path):
    """If os.replace fails (read-only dir), no state.json or .tmp should exist."""
    state_path = tmp_path / "run1" / "state.json"
    parent = tmp_path / "run1"
    parent.mkdir()

    # Make parent read-only so os.replace (rename) fails.
    os.chmod(parent, 0o500)
    try:
        # save_state swallows OSError, but no partial file should be left.
        save_state(state_path, "run1", "t", tmp_path / "repo",
                   Plan(chunks=[]), [], "abc")
        # After failure the state file must NOT exist.
        assert not state_path.exists(), \
            "partial state file left on disk after failed save"
        # No .tmp left either.
        assert not (parent / "state.json.tmp").exists()
    finally:
        os.chmod(parent, 0o700)


def test_atomic_write_completes_cleanly(tmp_path):
    """A successful save leaves only state.json (no .tmp leftover)."""
    path = tmp_path / "state.json"
    save_state(path, "r", "t", tmp_path / "repo", Plan(chunks=[]), [], "abc")
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# load_state on truncated / corrupt files
# ---------------------------------------------------------------------------


def test_load_state_raises_on_missing_file(tmp_path):
    path = tmp_path / "nope.json"
    with pytest.raises(StateCorruptError, match="not found"):
        load_state(path)


def test_load_state_raises_on_truncated_file(tmp_path):
    path = tmp_path / "truncated.json"
    path.write_text('{"run_id": "r", "plan": {', encoding="utf-8")
    with pytest.raises(StateCorruptError, match="truncated or malformed"):
        load_state(path)


def test_load_state_raises_on_garbage(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_bytes(b"\x00\x01\x02not json at all")
    with pytest.raises(StateCorruptError):
        load_state(path)


def test_load_state_raises_on_missing_required_field(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"run_id": "r"}), encoding="utf-8")
    with pytest.raises(StateCorruptError, match="missing required field"):
        load_state(path)


# ---------------------------------------------------------------------------
# find_runs ordering and tolerance
# ---------------------------------------------------------------------------


def test_find_runs_orders_newest_first(tmp_path):
    """find_runs returns runs ordered by mtime, newest first."""
    runs = []
    for name in ("run_a", "run_b", "run_c"):
        d = tmp_path / name
        d.mkdir()
        path = d / "state.json"
        state = {
            "run_id": name,
            "task": "task",
            "repo": str(tmp_path / "repo"),
            "base_commit": "abc",
            "plan": {"shared_context": "", "chunks": []},
            "outcomes": [],
            "created_at": time.time(),
        }
        path.write_text(json.dumps(state), encoding="utf-8")
        runs.append(d)
        # Stagger mtimes so ordering is deterministic.
        t = time.time()
        os.utime(d, (t, t))

    found = find_runs(tmp_path)
    assert [r.run_id for r in found] == ["run_c", "run_b", "run_a"]


def test_find_runs_skips_dirs_without_state_json(tmp_path):
    """Directories without state.json are ignored, not errored."""
    (tmp_path / "no_state").mkdir()
    d = tmp_path / "has_state"
    d.mkdir()
    state = {
        "run_id": "has_state",
        "task": "task",
        "repo": str(tmp_path / "repo"),
        "base_commit": "abc",
        "plan": {"shared_context": "", "chunks": []},
        "outcomes": [],
        "created_at": time.time(),
    }
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")

    found = find_runs(tmp_path)
    assert [r.run_id for r in found] == ["has_state"]


def test_find_runs_tolerates_corrupt_state_json(tmp_path):
    """Corrupt state.json is skipped, not raised."""
    d = tmp_path / "broken"
    d.mkdir()
    (d / "state.json").write_bytes(b"not json")
    # Also add a good one so we can confirm we still get results.
    good = tmp_path / "good"
    good.mkdir()
    state = {
        "run_id": "good",
        "task": "t",
        "repo": str(tmp_path / "repo"),
        "base_commit": "abc",
        "plan": {"shared_context": "", "chunks": []},
        "outcomes": [],
        "created_at": time.time(),
    }
    (good / "state.json").write_text(json.dumps(state), encoding="utf-8")

    found = find_runs(tmp_path)
    assert len(found) == 1
    assert found[0].run_id == "good"


def test_find_runs_tolerates_missing_directory(tmp_path):
    found = find_runs(tmp_path / "does_not_exist")
    assert found == []


# ---------------------------------------------------------------------------
# Resume selection picks exactly the non-COMPLETED chunks
# ---------------------------------------------------------------------------


def _make_state(outcomes, run_id="r1", task="t", base="abc"):
    chunks = [o.chunk for o in outcomes]
    plan = Plan(chunks=chunks)
    return RunState(
        run_id=run_id,
        task=task,
        repo=Path("/fake/repo"),
        base_commit=base,
        plan=plan,
        outcomes=list(outcomes),
        created_at=time.time(),
    )


def test_resume_selection_picks_non_completed(tmp_path):
    """`incomplete_chunks` returns exactly the non-COMPLETED ones."""
    a = _chunk("a", title="A")
    b = _chunk("b", title="B")
    c = _chunk("c", title="C")

    state = _make_state([
        _outcome(a, AgentStatus.COMPLETED),
        _outcome(b, AgentStatus.FAILED, kill_reason="nope"),
        _outcome(c, AgentStatus.SKIPPED),
    ])

    incomplete = state.incomplete_chunks
    assert {o.chunk.id for o in incomplete} == {"b", "c"}


def test_resume_selection_picks_all_when_none_complete(tmp_path):
    """If nothing is completed, every chunk is selected."""
    a = _chunk("a", title="A")
    state = _make_state([
        _outcome(a, AgentStatus.RUNNING),
    ])
    assert [o.chunk.id for o in state.incomplete_chunks] == ["a"]


def test_resume_selection_picks_none_when_all_complete(tmp_path):
    """If everything is COMPLETED, no chunks are selected."""
    a = _chunk("a", title="A")
    state = _make_state([
        _outcome(a, AgentStatus.COMPLETED),
    ])
    assert state.incomplete_chunks == []


# ---------------------------------------------------------------------------
# A SKIPPED chunk whose dependency is now COMPLETED is selected
# ---------------------------------------------------------------------------


def test_skipped_chunk_selected_when_dependency_completed(tmp_path):
    """
    Resume scenario: chunk B was SKIPPED because A had failed, but on the
    resumed run A now succeeds. B must be re-selected (it's not COMPLETED).
    """
    a = _chunk("a", title="A")
    b = _chunk("b", depends_on=["a"], title="B")

    state = _make_state([
        _outcome(a, AgentStatus.COMPLETED),
        _outcome(b, AgentStatus.SKIPPED, kill_reason="dep 'a' failed"),
    ])

    incomplete = state.incomplete_chunks
    assert len(incomplete) == 1
    assert incomplete[0].chunk.id == "b"
    assert incomplete[0].status == AgentStatus.SKIPPED


def test_skipped_chunk_stays_skipped_if_dependency_still_failed(tmp_path):
    """B is still selected when A remains FAILED."""
    a = _chunk("a", title="A")
    b = _chunk("b", depends_on=["a"], title="B")

    state = _make_state([
        _outcome(a, AgentStatus.FAILED, kill_reason="oops"),
        _outcome(b, AgentStatus.SKIPPED, kill_reason="dep 'a' failed"),
    ])

    incomplete = state.incomplete_chunks
    assert len(incomplete) == 2


# ---------------------------------------------------------------------------
# Token totals surface unreported calls
# ---------------------------------------------------------------------------


def test_token_totals_surfaces_unreported_calls(tmp_path):
    """EventLog.token_totals reports unreported_calls when present."""
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps({
            "ts": 1.0,
            "run_id": "r1",
            "kind": "usage",
            "model": "ds4-full",
            "reported": False,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "calls": 2,
            "unreported_calls": 1,
        }) + "\n",
        encoding="utf-8",
    )

    totals = EventLog.token_totals(events_path)
    assert totals["ds4-full"]["prompt_tokens"] == 100
    assert totals["ds4-full"]["completion_tokens"] == 50
    assert totals["ds4-full"]["unreported_calls"] == 1
    assert totals["ds4-full"]["calls"] == 1  # token_totals counts events, not raw "calls" field


def test_token_totals_empty_file(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    totals = EventLog.token_totals(events_path)
    assert totals == {}
