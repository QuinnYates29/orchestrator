"""Per-agent isolated git workspaces.

Deliberately local `git clone`, not `git worktree`: a worktree is visible
from and shares refs with the origin repo (shows up in `git branch`/`git
worktree list` right there in the repo you're actively using), which is
exactly the noise this avoids. A local clone is a fully separate .git -
zero visibility from or effect on the real repo - while still being cheap,
since git auto-hardlinks the object store for same-filesystem local clones.
"""
from __future__ import annotations

from pathlib import Path

from ._procutil import run_argv
from .models import PlanChunk, RunConfig


async def capture_base_commit(repo: Path) -> str:
    code, out, err = await run_argv(["git", "rev-parse", "HEAD"], cwd=repo)
    if code != 0:
        raise RuntimeError(f"failed to read HEAD of {repo}: {err.strip()}")
    return out.strip()


async def create_agent_workspace(config: RunConfig, chunk: PlanChunk, attempt: int,
                                  base_commit: str) -> tuple[Path, str]:
    """Clone the repo and check out a fresh branch from base_commit. Returns
    (workspace_path, branch_name)."""
    dest = config.resolved_scratch_dir() / config.run_id / f"{chunk.id}-attempt{attempt}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise RuntimeError(f"workspace already exists: {dest}")

    code, _, err = await run_argv(["git", "clone", "--local", "--quiet", str(config.repo), str(dest)])
    if code != 0:
        raise RuntimeError(f"git clone failed for {chunk.id} attempt {attempt}: {err.strip()}")

    branch = f"agent/{config.run_id}/{chunk.id}-{attempt}"
    code, _, err = await run_argv(["git", "checkout", "--quiet", "-b", branch, base_commit], cwd=dest)
    if code != 0:
        raise RuntimeError(f"git checkout failed for {chunk.id} attempt {attempt}: {err.strip()}")

    return dest, branch


async def finalize_agent_commit(workspace: Path, message: str) -> str:
    """Stage + commit everything the agent left behind, even if nothing
    changed (--allow-empty), so there is always a clean commit to diff
    against - regardless of whether the agent itself ran any git commands.
    Returns the new commit hash."""
    await run_argv(["git", "add", "-A"], cwd=workspace)
    code, _, err = await run_argv(
        ["git", "commit", "--quiet", "--allow-empty", "-m", message], cwd=workspace,
    )
    if code != 0:
        raise RuntimeError(f"git commit failed in {workspace}: {err.strip()}")
    _, out, _ = await run_argv(["git", "rev-parse", "HEAD"], cwd=workspace)
    return out.strip()


async def diff_against_base(workspace: Path, base_commit: str) -> str:
    _, out, _ = await run_argv(["git", "diff", base_commit, "HEAD"], cwd=workspace)
    return out


async def changed_files(workspace: Path, base_commit: str) -> list[str]:
    _, out, _ = await run_argv(["git", "diff", "--name-only", base_commit, "HEAD"], cwd=workspace)
    return [line.strip() for line in out.splitlines() if line.strip()]


def cleanup_workspace(workspace: Path) -> None:
    """Not called by default - keep_scratch defaults to True since the
    per-agent clones are the actual audit trail when something looks off."""
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)
