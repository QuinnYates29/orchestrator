"""Append-only JSONL event log for a run.

Everything a run learns currently dies with the process: RunReport is printed
and dropped, and token usage arrives on every StreamEvent (client.py) only to be
discarded. This is the durable record — it's what `pipeline resume` reads back,
what `pipeline runs` summarizes, and the only way to compare the token cost of
two runs rather than assuming an optimization helped.

Written with an open-append-close per event rather than a held handle: events
are low-frequency (a handful per agent turn, against seconds-per-token
generation), and a run that is killed mid-flight — which is exactly when the log
matters most — must not lose its tail to an unflushed buffer.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("pipeline.events")

EVENTS_FILENAME = "events.jsonl"


class EventLog:
    """Thread- and task-safe append-only writer. `run_id` is stamped on every
    record so logs can be concatenated across runs without ambiguity."""

    def __init__(self, path: Path, run_id: str = ""):
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_run(cls, scratch_dir: Path, run_id: str) -> "EventLog":
        return cls(Path(scratch_dir) / run_id / EVENTS_FILENAME, run_id=run_id)

    def emit(self, kind: str, **fields: Any) -> None:
        """Record one event. Never raises: losing an observability record must
        not take down the run that produced it."""
        record = {"ts": time.time(), "run_id": self.run_id, "kind": kind}
        record.update(fields)
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError):
            log.exception("could not serialize event %r", kind)
            return
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            log.exception("could not append to %s", self.path)

    def emit_usage(self, role: str, model: str, usage: dict | None,
                   estimate: dict | None = None, **fields: Any) -> None:
        """Token accounting. `usage` is the raw OpenAI usage object.

        A backend that reports nothing falls back to `estimate` (see
        pipeline.tokens) rather than to zeros, because a zero is indistinguishable
        from a genuinely cheap call when totalling a run. Three states are kept
        distinguishable: `reported` (the backend counted), `estimated` (we did),
        and neither (nothing is known and the figures really are zero)."""
        usage = usage or {}
        reported = bool(usage)
        source = usage if reported else (estimate or {})
        self.emit(
            "usage",
            role=role,
            model=model,
            reported=reported,
            estimated=not reported and bool(estimate),
            prompt_tokens=int(source.get("prompt_tokens") or 0),
            completion_tokens=int(source.get("completion_tokens") or 0),
            **fields,
        )

    # -- reading back -------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        return list(EventLog.iter_events(path))

    @staticmethod
    def iter_events(path: str | Path) -> Iterator[dict]:
        """Yields well-formed records, skipping any torn final line — a run
        killed mid-write leaves one, and it shouldn't make the log unreadable."""
        target = Path(path)
        if not target.exists():
            return
        with target.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed event at %s:%d", target, lineno)

    @staticmethod
    def token_totals(path: str | Path) -> dict[str, dict[str, int]]:
        """Per-model prompt/completion totals over a run's log.

        `unreported_calls` counts calls the backend gave no usage for, and
        `estimated_calls` how many of those contributed an estimate instead of a
        zero — so a total can be read as exact, approximate, or a floor rather
        than being silently all three at once."""
        totals: dict[str, dict[str, int]] = {}
        for event in EventLog.iter_events(path):
            if event.get("kind") != "usage":
                continue
            bucket = totals.setdefault(
                str(event.get("model", "unknown")),
                {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                 "unreported_calls": 0, "estimated_calls": 0},
            )
            bucket["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
            bucket["completion_tokens"] += int(event.get("completion_tokens") or 0)
            bucket["calls"] += 1
            if not event.get("reported", True):
                bucket["unreported_calls"] += 1
            if event.get("estimated"):
                bucket["estimated_calls"] += 1
        return totals


class NullEventLog(EventLog):
    """No-op sink for tests and for callers with nowhere to write."""

    def __init__(self):  # noqa: D107 - deliberately skips EventLog.__init__
        self.path = Path(os.devnull)
        self.run_id = ""
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields: Any) -> None:
        return
