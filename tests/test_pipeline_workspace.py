"""Exercises real git operations in temp repos - no orchestrator/models
needed, just the `git` binary. This is the trickiest infrastructure piece
(clone-per-agent isolation), so it's worth verifying against real git
rather than only mocking it."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from pipeline.models import PlanChunk, RunConfig
from pipeline.workspace import (
    capture_base_commit,
    changed_files,
    cleanup_workspace,
    create_agent_workspace,
    diff_against_base,
    finalize_agent_commit,
)


def _init_repo(path: Path) -> str:
    """Synchronous test-scaffolding helper (not the code under test) - sets
    up a minimal real repo with local git identity so `commit` works
    regardless of global git config on the machine running the tests."""
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_capture_base_commit_matches_real_head(tmp_path):
    repo = tmp_path / "repo"
    expected = _init_repo(repo)
    got = asyncio.run(capture_base_commit(repo))
    assert got == expected


def test_create_agent_workspace_clones_isolated_and_checks_out_branch(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")

    ws, branch = asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))

    assert ws.exists()
    assert (ws / ".git").exists()
    assert (ws / "README.md").read_text() == "hello\n"
    assert branch == "agent/run1/agent-1-1"
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ws, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert current_branch == branch


def test_create_agent_workspace_does_not_pollute_origin_repo(tmp_path):
    """The whole point of local clones over worktrees: zero visibility from
    the real repo. Confirm the source repo gains no new branches/refs."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")

    asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))

    branches = subprocess.run(
        ["git", "branch", "--list"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert "agent/" not in branches
    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert head_after == base_commit


def test_create_agent_workspace_rejects_existing_destination(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")

    asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))
    with pytest.raises(RuntimeError, match="already exists"):
        asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))


def test_finalize_commit_and_diff_capture_new_file(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")
    ws, _ = asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))

    (ws / "new_feature.py").write_text("def hello():\n    return 42\n")

    new_commit = asyncio.run(finalize_agent_commit(ws, "[agent-1#1] completed"))
    assert new_commit != base_commit

    diff = asyncio.run(diff_against_base(ws, base_commit))
    assert "new_feature.py" in diff
    assert "def hello" in diff

    files = asyncio.run(changed_files(ws, base_commit))
    assert files == ["new_feature.py"]


def test_finalize_commit_allows_empty_when_agent_made_no_changes(tmp_path):
    """A killed/timed-out agent may have made no changes at all - the commit
    must still succeed (--allow-empty) so there's always something to diff."""
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")
    ws, _ = asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))

    new_commit = asyncio.run(finalize_agent_commit(ws, "[agent-1#1] killed: looped"))
    assert new_commit != base_commit  # a real (empty) commit was still created

    files = asyncio.run(changed_files(ws, base_commit))
    assert files == []


def test_cleanup_workspace_removes_directory(tmp_path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    config = RunConfig(repo=repo, task="x", scratch_dir=tmp_path / "scratch", run_id="run1")
    chunk = PlanChunk(id="agent-1", title="T", description="D")
    ws, _ = asyncio.run(create_agent_workspace(config, chunk, 1, base_commit))

    assert ws.exists()
    cleanup_workspace(ws)
    assert not ws.exists()
