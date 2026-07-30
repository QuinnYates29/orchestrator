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


# --- Fetching must not leave litter in the user's repository ---

from pipeline._procutil import run_argv
from pipeline.workspace import fetch_from_workspace, prune_workspace_remotes


def _plain_repo(path: Path) -> Path:
    """A minimal committed repo. Distinct from _init_repo above, which returns
    a commit sha; redefining that name here would silently rebind it for every
    test in the file."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "f.txt").write_text("base\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "base"], cwd=path, check=True)
    return path


def _branch_with_work(real: Path, ws: Path, branch: str, filename: str = "new.txt") -> str:
    subprocess.run(["git", "clone", "--local", "--quiet", str(real), str(ws)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "-b", branch], cwd=ws, check=True)
    (ws / filename).write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "work"], cwd=ws, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ws,
                          capture_output=True, text=True).stdout.strip()


def _remotes(repo: Path) -> list[str]:
    out = subprocess.run(["git", "remote"], cwd=repo, capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_fetch_from_workspace_registers_no_remote(tmp_path):
    """It used to add a `ws-<chunk>` remote per chunk and never remove it, so
    every run left dangling remotes in the user's repo, each pointing at a
    scratch clone that keep_scratch=False would then delete."""
    real = _plain_repo(tmp_path / "real")
    _branch_with_work(real, tmp_path / "ws", "agent/x")

    before = _remotes(real)
    asyncio.run(fetch_from_workspace(real, tmp_path / "ws", "agent/x"))
    assert _remotes(real) == before, "fetching must not add a remote"


def test_fetched_commit_is_still_usable_without_a_remote(tmp_path):
    """Dropping the remote must not break the thing the remote was there for."""
    real = _plain_repo(tmp_path / "real")
    sha = _branch_with_work(real, tmp_path / "ws", "agent/x")

    asyncio.run(fetch_from_workspace(real, tmp_path / "ws", "agent/x"))
    code, _, _ = asyncio.run(run_argv(["git", "cat-file", "-e", sha], cwd=real))
    assert code == 0, "the commit was not fetched into the real repo"
    code, _, err = asyncio.run(run_argv(["git", "cherry-pick", sha], cwd=real))
    assert code == 0, err
    assert (real / "new.txt").exists()


def test_fetch_failure_still_raises(tmp_path):
    real = _plain_repo(tmp_path / "real")
    with pytest.raises(RuntimeError, match="git fetch"):
        asyncio.run(fetch_from_workspace(real, tmp_path / "nonexistent", "agent/x"))


def test_prune_removes_only_ws_remotes(tmp_path):
    """Repos that already went through earlier runs carry the litter."""
    real = _plain_repo(tmp_path / "real")
    subprocess.run(["git", "remote", "add", "origin", "/somewhere"], cwd=real, check=True)
    subprocess.run(["git", "remote", "add", "ws-chunk-1", "/gone/a"], cwd=real, check=True)
    subprocess.run(["git", "remote", "add", "ws-chunk-2", "/gone/b"], cwd=real, check=True)

    assert asyncio.run(prune_workspace_remotes(real)) == 2
    assert _remotes(real) == ["origin"], "a real remote must survive pruning"


def test_prune_on_a_clean_repo_is_a_no_op(tmp_path):
    real = _plain_repo(tmp_path / "real")
    assert asyncio.run(prune_workspace_remotes(real)) == 0
    assert _remotes(real) == []
