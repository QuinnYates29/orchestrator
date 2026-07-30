"""Shared async subprocess execution, used by workspace.py (git) and
tools.py (run_shell)."""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path


# An agent shell has no terminal and nobody to answer a prompt, but plenty of
# tools assume otherwise. `git cherry-pick --continue` opens $EDITOR for the
# commit message and waits forever; a pager does the same to stdout; git will
# prompt for credentials on a network remote. Observed for real: a merge run
# spent its whole 15-minute dead-man's-switch with /usr/bin/editor parked on
# .git/COMMIT_EDITMSG. The timeout does eventually fire, so this is not a hang
# - it is fifteen minutes bought for nothing, repeatedly.
#
# Every value here makes an interactive tool choose the non-interactive path.
# Combined with stdin on /dev/null, anything that still tries to read gets EOF
# immediately rather than blocking.
_NONINTERACTIVE_ENV = {
    "GIT_EDITOR": "true",
    "GIT_SEQUENCE_EDITOR": "true",
    "EDITOR": "true",
    "VISUAL": "true",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "true",
    "SSH_ASKPASS": "true",
    "DEBIAN_FRONTEND": "noninteractive",
}


def _noninteractive_env() -> dict[str, str]:
    return {**os.environ, **_NONINTERACTIVE_ENV}


class ProcessTimeout(RuntimeError):
    def __init__(self, cmd: str, timeout_s: float):
        self.cmd = cmd
        self.timeout_s = timeout_s
        super().__init__(f"command timed out after {timeout_s}s: {cmd}")


async def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the process *and everything it spawned*.

    `proc.kill()` alone only kills the direct child. For a shell command that
    is `/bin/sh`, whose own children survive - and they inherit the stdout and
    stderr pipes, so asyncio keeps waiting for an EOF that will not arrive
    until the orphan finishes on its own. `sleep 100` with a 0.01s timeout
    therefore took the full 100 seconds, which made the timeout no timeout at
    all. Both callers start a new session so the whole group can be signalled.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=5.0)


async def run_argv(argv: list[str], cwd: Path | None = None,
                    timeout_s: float | None = None) -> tuple[int, str, str]:
    """Run a fixed-argv command (no shell interpolation) - used for git."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd) if cwd else None,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, env=_noninteractive_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _kill_group(proc)
        raise ProcessTimeout(" ".join(argv), timeout_s or 0.0)
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def run_shell(command: str, cwd: Path | None = None,
                     timeout_s: float | None = None) -> tuple[int, str, str]:
    """Run an arbitrary shell command string - used for the agent run_shell tool."""
    proc = await asyncio.create_subprocess_shell(
        command, cwd=str(cwd) if cwd else None,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, env=_noninteractive_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _kill_group(proc)
        raise ProcessTimeout(command, timeout_s or 0.0)
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
