"""Tests for pipeline.executor: topological_waves (pure) and execute_plan."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.events import EventLog, NullEventLog
from pipeline.executor import topological_waves, execute_plan
from pipeline.models import (
    AgentStatus, ChunkOutcome, Plan, PlanChunk, RunConfig, LimitsCfg,
)


# ---------------------------------------------------------------------------
# Pure function: topological_waves
# ---------------------------------------------------------------------------

def _mk(id: str, deps: list[str] | None = None) -> PlanChunk:
    return PlanChunk(id=id, title="t", description="d", depends_on=deps or [])


def test_topological_waves_empty():
    assert topological_waves([]) == []


def test_topological_waves_no_deps_single_wave():
    """All chunks with no deps → one wave in planner order."""
    chunks = [_mk("a"), _mk("b"), _mk("c")]
    waves = topological_waves(chunks)
    assert len(waves) == 1
    assert [c.id for c in waves[0]] == ["a", "b", "c"]


def test_topological_waves_chain_one_per_wave():
    """a → b → c → one chunk per wave."""
    chunks = [_mk("a"), _mk("b", ["a"]), _mk("c", ["b"])]
    waves = topological_waves(chunks)
    assert len(waves) == 3
    assert [c.id for c in waves[0]] == ["a"]
    assert [c.id for c in waves[1]] == ["b"]
    assert [c.id for c in waves[2]] == ["c"]


def test_topological_waves_diamond_three_waves():
    """a → b, a → c, b → d, c → d → three waves: [a], [b, c], [d]."""
    chunks = [_mk("a"), _mk("b", ["a"]), _mk("c", ["a"]), _mk("d", ["b", "c"])]
    waves = topological_waves(chunks)
    assert len(waves) == 3
    assert [c.id for c in waves[0]] == ["a"]
    assert sorted(c.id for c in waves[1]) == ["b", "c"]
    assert [c.id for c in waves[2]] == ["d"]


def test_topological_waves_stable_order_within_wave():
    """Within a wave, chunks preserve planner order."""
    # d and e both depend only on a; they should appear in original order.
    chunks = [_mk("a"), _mk("d", ["a"]), _mk("e", ["a"])]
    waves = topological_waves(chunks)
    assert [c.id for c in waves[0]] == ["a"]
    assert [c.id for c in waves[1]] == ["d", "e"]


def test_topological_waves_unknown_dep_raises():
    """A chunk depending on an id not in the plan raises ValueError naming it."""
    chunks = [_mk("a"), _mk("b", ["zzz"])]
    with pytest.raises(ValueError, match="zzz"):
        topological_waves(chunks)


def test_topological_waves_self_dep_raises():
    """A chunk that depends on itself raises ValueError naming it."""
    with pytest.raises(ValueError, match="'a' depends on itself"):
        topological_waves([_mk("a", ["a"])])


def test_topological_waves_2_cycle_raises():
    """a→b, b→a raises, naming both members."""
    chunks = [_mk("a", ["b"]), _mk("b", ["a"])]
    with pytest.raises(ValueError, match="cycle") as exc:
        topological_waves(chunks)
    msg = str(exc.value)
    assert "a" in msg and "b" in msg


def test_topological_waves_3_cycle_raises():
    """a→b→c→a raises, naming all three members."""
    chunks = [_mk("a", ["c"]), _mk("b", ["a"]), _mk("c", ["b"])]
    with pytest.raises(ValueError, match="cycle") as exc:
        topological_waves(chunks)
    msg = str(exc.value)
    assert all(letter in msg for letter in "abc")


# ---------------------------------------------------------------------------
# Executor behavior with a fake client
# ---------------------------------------------------------------------------

def _make_fake_client():
    """Build a fake OrchestratorClient with minimal async methods."""
    client = MagicMock()
    client.admin_load = AsyncMock(return_value={"ok": True})
    client.aclose = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat_stream = AsyncMock()
    client.chat_once = AsyncMock()
    return client


def _make_plan(chunks):
    return Plan(chunks=chunks, shared_context="")


def _make_run_config(tmp_path):
    return RunConfig(
        repo=tmp_path,
        task="test task",
        limits=LimitsCfg(max_concurrent_workers=2),
    )


def _make_event_log(tmp_path):
    return NullEventLog()


def test_execute_plan_no_deps_all_complete(tmp_path):
    """A plan with no dependencies: all chunks run, all complete."""
    chunks = [
        _mk("a"),
        _mk("b"),
    ]
    plan = _make_plan(chunks)
    client = _make_fake_client()
    config = _make_run_config(tmp_path)
    events = _make_event_log(tmp_path)
    base_commit = "abc123"

    # Mock create_agent_workspace and run_agent to simulate success
    with patch("pipeline.executor.create_agent_workspace", new_callable=AsyncMock) as mock_ws, \
         patch("pipeline.executor.run_agent", new_callable=AsyncMock) as mock_run, \
         patch("pipeline.executor.supervise", new_callable=AsyncMock) as mock_supervise, \
         patch("pipeline.executor._integrate_chunk", new_callable=AsyncMock) as mock_integrate, \
         patch("pipeline.executor._init_integration_repo", new_callable=AsyncMock), \
         patch("pipeline.executor._get_integration_head", new_callable=AsyncMock) as mock_get_head:
        
        # Setup mocks
        mock_ws.return_value = (Path("/fake/workspace"), "branch")
        mock_run.return_value = None  # run_agent mutates state
        
        # Create AgentState objects that run_agent would create
        from pipeline.models import AgentState
        state_a = AgentState(chunk=chunks[0], attempt=1, workspace=Path("/fake/workspace"))
        state_a.status = AgentStatus.COMPLETED
        state_a.finished_at = 1234567890
        state_a.turns = 5
        state_a.tool_call_log = []
        
        state_b = AgentState(chunk=chunks[1], attempt=1, workspace=Path("/fake/workspace2"))
        state_b.status = AgentStatus.COMPLETED
        state_b.finished_at = 1234567890
        state_b.turns = 3
        state_b.tool_call_log = []
        
        # run_agent mutates state in place, so we need to set it up
        async def fake_run_agent(client, config, plan, state):
            state.status = AgentStatus.COMPLETED
            state.finished_at = 1234567890
            state.turns = 5
            state.tool_call_log = []
        
        mock_run.side_effect = fake_run_agent
        mock_integrate.return_value = None  # No integration for simplicity
        
        outcomes = asyncio.run(execute_plan(
            client, config, plan, events,
            load_wait_s=180.0, base_commit=base_commit,
        ))
        
        # All chunks should have outcomes
        assert len(outcomes) == 2
        assert all(o.status == AgentStatus.COMPLETED for o in outcomes)
        
        # Check that run_agent was called for each chunk
        assert mock_run.call_count == 2


def test_execute_plan_wave_dependency(tmp_path):
    """Wave 2 chunks don't start until wave 1 finishes."""
    chunks = [
        _mk("a"),
        _mk("b", ["a"]),
    ]
    plan = _make_plan(chunks)
    client = _make_fake_client()
    config = _make_run_config(tmp_path)
    events = _make_event_log(tmp_path)
    base_commit = "abc123"
    
    with patch("pipeline.executor.create_agent_workspace", new_callable=AsyncMock) as mock_ws, \
         patch("pipeline.executor.run_agent", new_callable=AsyncMock) as mock_run, \
         patch("pipeline.executor.supervise", new_callable=AsyncMock) as mock_supervise, \
         patch("pipeline.executor._integrate_chunk", new_callable=AsyncMock) as mock_integrate, \
         patch("pipeline.executor._init_integration_repo", new_callable=AsyncMock), \
         patch("pipeline.executor._get_integration_head", new_callable=AsyncMock) as mock_get_head:
        
        mock_ws.return_value = (Path("/fake/workspace"), "branch")
        mock_integrate.return_value = None
        
        execution_order = []
        
        async def fake_run_agent(client, config, plan, state):
            # Track when this chunk starts
            execution_order.append(("start", state.chunk.id))
            await asyncio.sleep(0.01)  # Simulate some work
            execution_order.append(("end", state.chunk.id))
            state.status = AgentStatus.COMPLETED
            state.finished_at = 1234567890
            state.turns = 5
            state.tool_call_log = []
        
        mock_run.side_effect = fake_run_agent
        
        outcomes = asyncio.run(execute_plan(
            client, config, plan, events,
            load_wait_s=180.0, base_commit=base_commit,
        ))
        
        # All chunks should complete
        assert len(outcomes) == 2
        assert all(o.status == AgentStatus.COMPLETED for o in outcomes)
        
        # Verify wave ordering: 'a' should finish before 'b' starts
        assert execution_order[0] == ("start", "a")
        assert execution_order[1] == ("end", "a")
        assert execution_order[2] == ("start", "b")
        assert execution_order[3] == ("end", "b")


def test_execute_plan_dependency_failure_skips_downstream(tmp_path):
    """If a dependency fails, downstream chunks are SKIPPED with a reason."""
    chunks = [
        _mk("a"),
        _mk("b", ["a"]),
    ]
    plan = _make_plan(chunks)
    client = _make_fake_client()
    config = _make_run_config(tmp_path)
    events = _make_event_log(tmp_path)
    base_commit = "abc123"
    
    with patch("pipeline.executor.create_agent_workspace", new_callable=AsyncMock) as mock_ws, \
         patch("pipeline.executor.run_agent", new_callable=AsyncMock) as mock_run, \
         patch("pipeline.executor.supervise", new_callable=AsyncMock) as mock_supervise, \
         patch("pipeline.executor._integrate_chunk", new_callable=AsyncMock) as mock_integrate, \
         patch("pipeline.executor._init_integration_repo", new_callable=AsyncMock), \
         patch("pipeline.executor._get_integration_head", new_callable=AsyncMock) as mock_get_head:
        
        mock_ws.return_value = (Path("/fake/workspace"), "branch")
        mock_integrate.return_value = None
        
        call_count = 0
        
        async def fake_run_agent(client, config, plan, state):
            nonlocal call_count
            call_count += 1
            if state.chunk.id == "a":
                # First chunk fails
                state.status = AgentStatus.FAILED
                state.kill_reason = "test failure"
                state.finished_at = 1234567890
                state.turns = 5
                state.tool_call_log = []
            else:
                # This should not be reached for 'b' since 'a' failed
                state.status = AgentStatus.COMPLETED
                state.finished_at = 1234567890
                state.turns = 3
                state.tool_call_log = []
        
        mock_run.side_effect = fake_run_agent
        
        outcomes = asyncio.run(execute_plan(
            client, config, plan, events,
            load_wait_s=180.0, base_commit=base_commit,
        ))
        
        # 'a' should have failed
        a_outcome = next(o for o in outcomes if o.chunk.id == "a")
        assert a_outcome.status == AgentStatus.FAILED
        
        # 'b' should be SKIPPED
        b_outcome = next(o for o in outcomes if o.chunk.id == "b")
        assert b_outcome.status == AgentStatus.SKIPPED
        assert "a" in b_outcome.kill_reason
        
        # run_agent should only have been called for 'a', not 'b'
        assert call_count == 1


def test_execute_plan_semaphore_caps_concurrency(tmp_path):
    """The semaphore should prevent more than max_concurrent_workers running at once."""
    # Create more chunks than the concurrency limit
    chunks = [_mk(str(i)) for i in range(5)]
    plan = _make_plan(chunks)
    client = _make_fake_client()
    config = _make_run_config(tmp_path)
    config.limits.max_concurrent_workers = 2  # Limit to 2 concurrent
    events = _make_event_log(tmp_path)
    base_commit = "abc123"
    
    with patch("pipeline.executor.create_agent_workspace", new_callable=AsyncMock) as mock_ws, \
         patch("pipeline.executor.run_agent", new_callable=AsyncMock) as mock_run, \
         patch("pipeline.executor.supervise", new_callable=AsyncMock) as mock_supervise, \
         patch("pipeline.executor._integrate_chunk", new_callable=AsyncMock) as mock_integrate, \
         patch("pipeline.executor._init_integration_repo", new_callable=AsyncMock), \
         patch("pipeline.executor._get_integration_head", new_callable=AsyncMock) as mock_get_head:
        
        mock_ws.return_value = (Path("/fake/workspace"), "branch")
        mock_integrate.return_value = None
        
        # Track concurrency
        current_in_flight = 0
        peak_in_flight = 0
        
        async def fake_run_agent(client, config, plan, state):
            nonlocal current_in_flight, peak_in_flight
            current_in_flight += 1
            peak_in_flight = max(peak_in_flight, current_in_flight)
            
            # Simulate variable work time
            await asyncio.sleep(0.01)
            
            state.status = AgentStatus.COMPLETED
            state.finished_at = 1234567890
            state.turns = 5
            state.tool_call_log = []
            
            current_in_flight -= 1
        
        mock_run.side_effect = fake_run_agent
        
        outcomes = asyncio.run(execute_plan(
            client, config, plan, events,
            load_wait_s=180.0, base_commit=base_commit,
        ))
        
        # All chunks should complete
        assert len(outcomes) == 5
        assert all(o.status == AgentStatus.COMPLETED for o in outcomes)
        
        # Peak concurrency should not exceed limit
        assert peak_in_flight <= config.limits.max_concurrent_workers


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
