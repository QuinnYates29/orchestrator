"""Verification: run a configured command and report pass/fail/skip.

This module must not pretend a skipped check is evidence the work is good.
Skipped means no check was run — callers decide how to interpret that.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ._procutil import ProcessTimeout, run_shell
from .config import VerifyCfg


@dataclass
class VerifyResult:
    ok: bool
    skipped: bool  # no command configured — absence of a check, not evidence
    output_tail: str
    duration_s: float
    exit_code: int | None


# Maximum characters to keep from the output tail
_MAX_TAIL_CHARS = 2000


async def run_verify(cfg: VerifyCfg, cwd: Path) -> VerifyResult:
    """Run the configured verify command and return a result.

    If no command is configured, returns skipped=True with ok=False — absence
    of a check is never evidence the work is good.
    """
    if not cfg.configured:
        return VerifyResult(
            ok=False,
            skipped=True,
            output_tail="",
            duration_s=0.0,
            exit_code=None,
        )

    started = time.monotonic()
    try:
        exit_code, stdout, stderr = await run_shell(cfg.command, cwd=cwd, timeout_s=cfg.timeout_s)
    except ProcessTimeout as e:
        duration_s = time.monotonic() - started
        return VerifyResult(
            ok=False,
            skipped=False,
            output_tail=f"verify timed out after {cfg.timeout_s}s\ncommand: {cfg.command}\n{e}",
            duration_s=duration_s,
            exit_code=None,
        )

    duration_s = time.monotonic() - started
    combined = stdout + "\n" + stderr if stderr else stdout
    tail = combined[-_MAX_TAIL_CHARS:] if combined else ""
    ok = exit_code == 0

    return VerifyResult(
        ok=ok,
        skipped=False,
        output_tail=tail,
        duration_s=duration_s,
        exit_code=exit_code,
    )
