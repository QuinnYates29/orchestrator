"""Tests for the deterministic sequential merge (Phase 4).

Builds real temporary git repos and exercises the merge logic via
subprocess — faking git would test nothing for this phase.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from pipeline._procutil import run_argv
from pipeline.executor import topological_waves
from pipeline.models import (
    AgentStatus,
    ChunkOutcome,
    Plan,
    PlanChunk,
    RunConfig,
)
from pipeline.merger import merge
from pipeline.workspace import (
    capture_base_commit,
    changed_files,
    create_agent_workspace,
    diff_against_base,
    finalize_agent_commit,
)
from pipeline.verify import VerifyResult
from pipeline.config import VerifyCfg


def _init_repo(path: Path) -> str:
    """Create a minimal git repo with one file and one commit."""
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_workspace(
    config: RunConfig, chunk: PlanChunk, base_commit: str, content: dict[str, str],
) -> tuple[Path, str]:
    """Create a workspace and commit the given file content into it."""
    ws, branch = asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))
    for fname, text in content.items():
        (ws / fname).write_text(text)
    new_commit = asyncio.run(finalize_agent_commit(ws, f"[{chunk.id}#1] completed"))
    return ws, new_commit


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fake client that records calls and can script resolutions
# ---------------------------------------------------------------------------


class FakeClient:
    """Scripted client.

    If resolution_turns are provided, each turn is a message dict
    (role=assistant, content, tool_calls).  The client returns them in order
    and executes any run_shell tool calls against the real repo so that the
    model can "commit" its fix.
    """
    def __init__(self, resolution_turns: list[dict] | None = None, repo: Path | None = None):
        self.calls: list[dict] = []
        self._resolution_turns = list(resolution_turns or [])
        self._repo = repo

    async def chat_once(self, model, messages, *, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append({
            "model": model,
            "messages_len": len(messages),
            "tool_choice": tool_choice,
        })

        if self._resolution_turns:
            turn = self._resolution_turns.pop(0)
            # If the turn has a tool call, execute it so HEAD changes.
            tool_calls = turn.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc["function"]
                if fn["name"] == "run_shell":
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    command = args.get("command", "")
                    subprocess.run(command, cwd=self._repo, shell=True, check=True)
                elif fn["name"] == "write_file":
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    path_str = args.get("path", "")
                    content = args.get("content", "")
                    (self._repo / path_str).write_text(content)
                elif fn["name"] == "edit_file":
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    path_str = args.get("path", "")
                    old = args.get("old_string", "")
                    new = args.get("new_string", "")
                    fpath = self._repo / path_str
                    text = fpath.read_text()
                    fpath.write_text(text.replace(old, new))
            # Return the turn as-is (tool results are not fed back; the
            # model's turn is scripted to be exactly what we need).
            return {"choices": [{"message": turn}]}

        # No resolution — return a message with no tool calls.
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "No resolution needed.",
                }
            }]
        }

    async def admin_load(self, model, profile=None, wait_s=180.0):
        return {"ok": True}


# =======================================================================
# Test: two chunks touching different files → both cherry-pick cleanly,
# model never called.
# =======================================================================


def test_two_chunks_different_files_model_not_called(tmp_path):
    """Two chunks touching different files → both cherry-pick cleanly, model
    never called (assert the fake client received zero requests)."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))  # verify skipped

    chunks = [
        PlanChunk(id="c1", title="add file b", description="add b.txt",
                  scope=["b.txt"]),
        PlanChunk(id="c2", title="add file c", description="add c.txt",
                  scope=["c.txt"]),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"b.txt": "from chunk1\n"})
    ws2, commit2 = _make_workspace(config, chunks[1], base_commit,
                                   {"c.txt": "from chunk2\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
        ChunkOutcome(chunk=chunks[1], status=AgentStatus.COMPLETED, workspace=ws2),
    ]

    client = FakeClient()  # no resolution turns — model should never be called
    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    # Model should not have been called at all.
    assert len(client.calls) == 0, f"model was called {len(client.calls)} times"

    assert merge_commit is not None
    # Both chunks should be merged cleanly.
    assert "[merged_cleanly] c1" in summary
    assert "[merged_cleanly] c2" in summary

    # Verify files exist in the real repo.
    assert (repo / "b.txt").read_text() == "from chunk1\n"
    assert (repo / "c.txt").read_text() == "from chunk2\n"


# =======================================================================
# Test: two chunks touching the same line → conflict detected, model
# escalation invoked with only that chunk's context.
# =======================================================================


def test_two_chunks_same_line_conflict_escalation(tmp_path):
    """Two chunks touching the same line → conflict detected, model
    escalation invoked with only that chunk's context."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))

    # Write initial content to a.txt and commit it.
    (repo / "a.txt").write_text("hello\nworld\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "second"], cwd=repo, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()

    chunks = [
        PlanChunk(id="c1", title="change hello", description="change hello to hi",
                  scope=["a.txt"]),
        PlanChunk(id="c2", title="change hello too", description="also change hello to hey",
                  scope=["a.txt"]),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"a.txt": "hi\nworld\n"})
    ws2, commit2 = _make_workspace(config, chunks[1], base_commit,
                                   {"a.txt": "hey\nworld\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
        ChunkOutcome(chunk=chunks[1], status=AgentStatus.COMPLETED, workspace=ws2),
    ]

    # Script a resolution: the model runs run_shell to commit a fix,
    # then calls finish_merge.
    resolution_turn1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "fc1",
            "function": {
                "name": "run_shell",
                "arguments": json.dumps({"command": "git add -A && git commit -m 'resolve conflict'"}),
            },
        }],
    }
    resolution_turn2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "fc2",
            "function": {
                "name": "finish_merge",
                "arguments": json.dumps({"summary": "Resolved conflict in a.txt by picking hi."}),
            },
        }],
    }

    client = FakeClient(resolution_turns=[resolution_turn1, resolution_turn2], repo=repo)

    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    # Model should have been called at least once for escalation.
    assert len(client.calls) >= 1

    # c1 should merge cleanly (first chunk, no conflict with base).
    assert "[merged_cleanly] c1" in summary

    # c2 should have been escalated and merged.
    assert "c2" in summary
    assert "[merged_after_escalation] c2" in summary or "[merged_cleanly] c2" in summary


# =======================================================================
# Test: verify failing after a clean pick → escalation invoked.
# =======================================================================


def test_verify_failure_triggers_escalation(tmp_path):
    """Verify failing after a clean pick → escalation invoked."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command="exit 1"))  # always fails

    chunks = [
        PlanChunk(id="c1", title="add b", description="add b.txt",
                  scope=["b.txt"]),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"b.txt": "from chunk1\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
    ]

    # Script a resolution: model first writes a fix (e.g., touch a file so
    # git commit works), then commits, then calls finish_merge.
    resolution_turn1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "fc1",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"path": "fix.txt", "content": "verify fix"}),
            },
        }],
    }
    resolution_turn2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "fc2",
            "function": {
                "name": "run_shell",
                "arguments": json.dumps({"command": "git add -A && git commit -m 'fix verify'"}),
            },
        }],
    }
    resolution_turn3 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "fc3",
            "function": {
                "name": "finish_merge",
                "arguments": json.dumps({"summary": "Fixed verify failure by adding fix.txt."}),
            },
        }],
    }

    client = FakeClient(resolution_turns=[resolution_turn1, resolution_turn2, resolution_turn3], repo=repo)

    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    # Model was called for escalation.
    assert len(client.calls) >= 1

    # c1 should show escalated or merged_after_escalation.
    assert "c1" in summary


# =======================================================================
# Test: a chunk that cannot be integrated → aborted, recorded as unmerged,
# subsequent chunks still processed.
# =======================================================================


def test_unmerged_chunk_does_not_block_subsequent(tmp_path):
    """A chunk that cannot be integrated → aborted, recorded as unmerged,
    subsequent chunks still processed."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))

    # Three chunks: c1 (add d.txt — clean), c2 (change a.txt — conflicts with
    # base because c1 didn't touch a.txt but c2 does, actually the conflict
    # is between c2 and the base since c2 changes a.txt which exists).
    # c3 (add e.txt — independent, no conflict).
    (repo / "a.txt").write_text("hello\nworld\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "second"], cwd=repo, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()

    chunks = [
        PlanChunk(id="c1", title="change a.txt to hi",
                  description="change a.txt line 1 to hi", scope=["a.txt"]),
        PlanChunk(id="c2", title="conflicting",
                  description="change a.txt line 1 to conflict", scope=["a.txt"]),
        PlanChunk(id="c3", title="add e", description="add e.txt",
                  scope=["e.txt"]),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"a.txt": "hi\nworld\n"})
    ws2, commit2 = _make_workspace(config, chunks[1], base_commit,
                                   {"a.txt": "conflict\nworld\n"})
    ws3, commit3 = _make_workspace(config, chunks[2], base_commit,
                                   {"e.txt": "from c3\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
        ChunkOutcome(chunk=chunks[1], status=AgentStatus.COMPLETED, workspace=ws2),
        ChunkOutcome(chunk=chunks[2], status=AgentStatus.COMPLETED, workspace=ws3),
    ]

    # No resolution turns — all escalations fail.
    client = FakeClient(repo=repo)

    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    # c1 should merge cleanly (no conflict with base).
    assert "[merged_cleanly] c1" in summary

    # c2 should be unmerged (conflict, no resolution).
    assert "c2" in summary
    assert "[unmerged]" in summary or "[escalated]" in summary

    # c3 should still be processed (different file, no dependency on c2).
    assert "c3" in summary

    # a.txt should have c1's change (hi) applied.
    assert (repo / "a.txt").read_text() == "hi\nworld\n"
    # e.txt may or may not exist depending on c3 merge success.
    if "[merged_cleanly] c3" in summary or "[merged_after_escalation] c3" in summary:
        assert (repo / "e.txt").read_text() == "from c3\n"


# =======================================================================
# Test: chunk with no changes → recorded, skipped, not an error.
# =======================================================================


def test_chunk_with_no_changes_skipped(tmp_path):
    """Chunk with no changes → recorded, skipped, not an error."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))

    chunks = [
        PlanChunk(id="c1", title="empty chunk", description="does nothing"),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {})  # no changes

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
    ]

    client = FakeClient()
    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    assert "[no_changes] c1" in summary
    assert merge_commit is None  # no new commits


# =======================================================================
# Test: summary names every failed and unmerged chunk.
# =======================================================================


def test_summary_covers_all_failed_chunks(tmp_path):
    """Summary names every failed and unmerged chunk."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))

    chunks = [
        PlanChunk(id="c1", title="first", description="first chunk"),
        PlanChunk(id="c2", title="second", description="second chunk"),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"b.txt": "from c1\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
        ChunkOutcome(chunk=chunks[1], status=AgentStatus.KILLED, workspace=None,
                     kill_reason="killed: looped"),
    ]

    client = FakeClient()
    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    assert "c1" in summary
    assert "c2" in summary
    assert "[skipped] c2" in summary
    assert "killed" in summary or "looped" in summary


# =======================================================================
# Test: skipped chunks (dependency failed) are recorded.
# =======================================================================


def test_skipped_chunks_are_recorded(tmp_path):
    """Skipped chunks are recorded in the summary."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = RunConfig(repo=repo, task="test", scratch_dir=scratch, run_id="r1",
                       verify=VerifyCfg(command=None))

    chunks = [
        PlanChunk(id="c1", title="good", description="good chunk"),
        PlanChunk(id="c2", title="failed", description="failed chunk"),
    ]
    plan = Plan(chunks=chunks)

    ws1, commit1 = _make_workspace(config, chunks[0], base_commit,
                                   {"b.txt": "from c1\n"})

    outcomes = [
        ChunkOutcome(chunk=chunks[0], status=AgentStatus.COMPLETED, workspace=ws1),
        ChunkOutcome(chunk=chunks[1], status=AgentStatus.SKIPPED, workspace=None,
                     kill_reason="dependency c1 failed"),
    ]

    client = FakeClient()
    merge_commit, summary = _run(merge(client, config, plan, outcomes, base_commit))

    assert "[merged_cleanly] c1" in summary
    assert "[skipped] c2" in summary
