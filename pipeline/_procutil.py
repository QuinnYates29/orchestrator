"""Shared async subprocess execution, used by workspace.py (git) and
tools.py (run_shell)."""
from __future__ import annotations

import asyncio
from pathlib import Path


class ProcessTimeout(RuntimeError):
    def __init__(self, cmd: str, timeout_s: float):
        self.cmd = cmd
        self.timeout_s = timeout_s
        super().__init__(f"command timed out after {timeout_s}s: {cmd}")


async def run_argv(argv: list[str], cwd: Path | None = None,
                    timeout_s: float | None = None) -> tuple[int, str, str]:
    """Run a fixed-argv command (no shell interpolation) - used for git."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ProcessTimeout(" ".join(argv), timeout_s or 0.0)
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def run_shell(command: str, cwd: Path | None = None,
                     timeout_s: float | None = None) -> tuple[int, str, str]:
    """Run an arbitrary shell command string - used for the agent run_shell tool."""
    proc = await asyncio.create_subprocess_shell(
        command, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ProcessTimeout(command, timeout_s or 0.0)
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
