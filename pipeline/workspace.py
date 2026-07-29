"""Per-agent isolated git workspaces.

Deliberately local `git clone`, not `git worktree`: a worktree is visible
from and shares refs with the origin repo (shows up in `git branch`/`git
worktree list` right there in the repo you're actively using), which is
exactly the noise this avoids. A local clone is a fully separate .git -
zero visibility from or effect on the real repo - while still being cheap,
since git auto-hardlinks the object store for same-filesystem local clones.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ._procutil import run_argv
from .models import PlanChunk, RunConfig

log = logging.getLogger("pipeline.workspace")


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

    _write_workspace_excludes(dest)
    return dest, branch


# Build artifacts an agent generates by *running* the code it is writing.
# finalize_agent_commit does `git add -A`, so without this every agent that
# runs pytest commits its own .pyc files - and two chunks that both import the
# same module each commit their own binary copy of that module's .pyc, which
# conflicts on merge every single time. Written to .git/info/exclude rather
# than a .gitignore so the agent's diff stays free of files it did not author.
_WORKSPACE_EXCLUDES = """\
# Written by pipeline.workspace: artifacts from running the code under test.
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
*.egg-info/
"""


def _write_workspace_excludes(workspace: Path) -> None:
    exclude = workspace / ".git" / "info" / "exclude"
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text() if exclude.exists() else ""
        exclude.write_text(existing + "\n" + _WORKSPACE_EXCLUDES)
    except OSError as e:  # never fail a run over a hygiene nicety
        log.warning("could not write %s: %s", exclude, e)


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


# -- Merge-phase helpers (operate on the real repo, not a workspace) --------

async def fetch_from_workspace(real_repo: Path, workspace: Path, branch: str) -> None:
    """Add the workspace clone as a remote in the real repo and fetch its
    branch. The remote is named after the workspace path to avoid collisions."""
    remote_name = f"ws-{workspace.name}"
    # Ensure the remote is not already present (from a prior fetch).
    code, _, err = await run_argv(
        ["git", "remote", "add", remote_name, str(workspace)], cwd=real_repo,
    )
    if code != 0:
        # Remote may already exist; that's fine.
        pass
    code, _, err = await run_argv(
        ["git", "fetch", "--quiet", remote_name, branch], cwd=real_repo,
    )
    if code != 0:
        raise RuntimeError(f"git fetch from workspace {workspace} failed: {err.strip()}")


async def cherry_pick_commit(real_repo: Path, commit: str) -> tuple[int, str, str]:
    """Cherry-pick a single commit into the real repo. Returns (exit_code,
    stdout, stderr). On conflict the caller should inspect conflicted_files
    and decide whether to abort or escalate."""
    return await run_argv(
        ["git", "cherry-pick", "--quiet", commit], cwd=real_repo,
    )


async def conflicted_files(real_repo: Path) -> list[str]:
    """Return a list of files currently in a conflicted state (unmerged paths
    from a cherry-pick or merge). Returns empty list if no conflict."""
    code, out, _ = await run_argv(
        ["git", "diff", "--name-only", "--diff-filter=U", "HEAD"], cwd=real_repo,
    )
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


async def abort_cherry_pick(real_repo: Path) -> None:
    """Abort a cherry-pick that is in progress, restoring the pre-cherry-pick
    state. Raises if no cherry-pick is in progress."""
    code, _, err = await run_argv(
        ["git", "cherry-pick", "--abort"], cwd=real_repo,
    )
    if code != 0:
        raise RuntimeError(f"git cherry-pick --abort failed: {err.strip()}")


async def reset_to_commit(real_repo: Path, commit: str) -> None:
    """Reset the real repo to a specific commit (soft reset, preserving
    working tree). Used to discard failed merge attempts."""
    code, _, err = await run_argv(
        ["git", "reset", "--hard", commit], cwd=real_repo,
    )
    if code != 0:
        raise RuntimeError(f"git reset --hard {commit[:12]} failed: {err.strip()}")


def cleanup_workspace(workspace: Path) -> None:
    """Not called by default - keep_scratch defaults to True since the
    per-agent clones are the actual audit trail when something looks off."""
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)
