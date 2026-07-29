from __future__ import annotations

import asyncio
from pathlib import Path

from pipeline._procutil import ProcessTimeout
from pipeline.config import VerifyCfg
from pipeline.verify import VerifyResult, run_verify


def _run(coro):
    return asyncio.run(coro)


def test_skipped_is_not_ok():
    """A skipped verify (no command configured) must never be reported as ok."""
    cfg = VerifyCfg(command=None)
    result = _run(run_verify(cfg, Path(".")))
    assert result.skipped
    assert not result.ok


def test_skipped_is_not_ok_blank():
    """A blank/whitespace-only command also skips."""
    cfg = VerifyCfg(command="   ")
    result = _run(run_verify(cfg, Path(".")))
    assert result.skipped
    assert not result.ok


def test_passing_command():
    cfg = VerifyCfg(command="echo ok", timeout_s=30.0)
    result = _run(run_verify(cfg, Path(".")))
    assert result.ok
    assert not result.skipped
    assert result.exit_code == 0
    assert result.output_tail


def test_failing_command_captures_exit_code_and_tail():
    cfg = VerifyCfg(command="false", timeout_s=30.0)
    result = _run(run_verify(cfg, Path(".")))
    assert not result.ok
    assert not result.skipped
    assert result.exit_code == 1
    assert isinstance(result.output_tail, str)


def test_timeout_becomes_failure_not_exception():
    """A ProcessTimeout must be caught and reported as ok=False, not raised."""
    cfg = VerifyCfg(command="sleep 100", timeout_s=0.01)
    result = _run(run_verify(cfg, Path(".")))
    assert not result.ok
    assert not result.skipped
    assert "timed out" in result.output_tail
    assert result.exit_code is None


def test_tail_truncation_keeps_the_end():
    """Long output should be truncated to the last ~2000 chars."""
    cfg = VerifyCfg(command="python3 -c 'print(\"x\" * 5000)'", timeout_s=30.0)
    result = _run(run_verify(cfg, Path(".")))
    assert result.ok
    assert not result.skipped
    # Tail should be the last ~2000 chars, not the first 2000
    assert len(result.output_tail) <= 2000
    # The end of the long output is x's — the tail should contain x's
    assert "x" in result.output_tail
