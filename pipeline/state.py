"""Persisted pipeline state.

``<scratch>/<run_id>/state.json`` is written after **every** state transition,
not only at the end — a run that dies is exactly the run whose state matters.
Without this, hours of agent work sitting in per-agent clones on disk would
be unrecoverable.

Round-tripping preserves enough to resume: the plan with its ``depends_on``,
each chunk's status/attempts/kill reason/verify result, the per-chunk workspace
path, and the base commit. Enum values serialize as their string values (
``AgentStatus`` extends ``str``).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from .events import EventLog
from .models import AgentStatus, ChunkOutcome, Plan, PlanChunk, VerifyResult

log = logging.getLogger("pipeline.state")

STATE_FILENAME = "state.json"


class StateCorruptError(Exception):
    """Raised when state.json is missing, malformed, or truncated."""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _chunk_to_dict(chunk: PlanChunk) -> dict:
    return {
        "id": chunk.id,
        "title": chunk.title,
        "description": chunk.description,
        "scope": list(chunk.scope),
        "context": chunk.context,
        "depends_on": list(chunk.depends_on),
    }


def _chunk_from_dict(data: dict) -> PlanChunk:
    return PlanChunk(
        id=data["id"],
        title=data.get("title", data["id"]),
        description=data.get("description", ""),
        scope=list(data.get("scope") or []),
        context=data.get("context", ""),
        depends_on=list(data.get("depends_on") or []),
    )


def _outcome_to_dict(outcome: ChunkOutcome) -> dict:
    verify = None
    if outcome.verify is not None:
        verify = asdict(outcome.verify)
    return {
        "chunk": _chunk_to_dict(outcome.chunk),
        "status": outcome.status.value,
        "workspace": str(outcome.workspace) if outcome.workspace else None,
        "kill_reason": outcome.kill_reason or "",
        "attempts": outcome.attempts,
        "verify": verify,
        "submitted": outcome.submitted,
    }


def _outcome_from_dict(data: dict, chunks_by_id: dict[str, PlanChunk]) -> ChunkOutcome:
    chunk_data = data.get("chunk", {})
    chunk_id = chunk_data.get("id", "unknown")

    # Use the chunk from the plan so identity is preserved (outcome.chunk IS
    # plan.chunks[i], not a copy).
    chunk = chunks_by_id.get(chunk_id)
    if chunk is None:
        chunk = _chunk_from_dict(chunk_data)

    verify = None
    if data.get("verify"):
        v = data["verify"]
        verify = VerifyResult(
            ok=bool(v.get("ok", False)),
            skipped=bool(v.get("skipped", False)),
            output_tail=v.get("output_tail", "") or "",
            duration_s=float(v.get("duration_s") or 0),
            exit_code=v.get("exit_code"),
        )
    status_str = data.get("status", "failed")
    try:
        status = AgentStatus(status_str)
    except ValueError:
        status = AgentStatus.FAILED
    return ChunkOutcome(
        chunk=chunk,
        status=status,
        workspace=Path(data["workspace"]) if data.get("workspace") else None,
        kill_reason=data.get("kill_reason", "") or "",
        attempts=int(data.get("attempts", 1) or 1),
        verify=verify,
        submitted=data.get("submitted"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_state(path: Path, run_id: str, task: str, repo: Path,
               plan: Plan, outcomes: list[ChunkOutcome],
               base_commit: str) -> None:
    """Write state.json atomically (temp file + ``os.replace``).

    A half-written ``state.json`` is worse than none — a run killed mid-write
    is the expected case. The temp file is cleaned up on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "task": task,
        "repo": str(repo),
        "base_commit": base_commit,
        "plan": {
            "shared_context": plan.shared_context,
            "chunks": [_chunk_to_dict(c) for c in plan.chunks],
        },
        "outcomes": [_outcome_to_dict(o) for o in outcomes],
        "created_at": time.time(),
    }
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2, default=str),
                            encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        log.exception("failed to write state to %s", path)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_state(path: Path) -> "RunState":
    """Read state.json and reconstruct a ``RunState``.

    Raises ``StateCorruptError`` if the file is missing, malformed, or
    truncated — never silently returns junk.
    """
    path = Path(path)
    if not path.exists():
        raise StateCorruptError(f"state file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StateCorruptError(
            f"state file {path} is truncated or malformed: {e}") from e
    except OSError as e:
        raise StateCorruptError(f"could not read state file {path}: {e}") from e

    try:
        plan_data = raw["plan"]
        chunks = [_chunk_from_dict(c)
                  for c in plan_data.get("chunks", [])]
        plan = Plan(chunks=chunks,
                    shared_context=plan_data.get("shared_context", ""))
        chunks_by_id = {c.id: c for c in chunks}

        outcomes: list[ChunkOutcome] = []
        for odata in raw.get("outcomes", []):
            outcomes.append(_outcome_from_dict(odata, chunks_by_id))

        return RunState(
            run_id=raw["run_id"],
            task=raw["task"],
            repo=Path(raw["repo"]),
            base_commit=raw.get("base_commit", ""),
            plan=plan,
            outcomes=outcomes,
            created_at=float(raw.get("created_at", 0) or 0),
        )
    except KeyError as e:
        raise StateCorruptError(
            f"state file {path} missing required field {e!r}") from e
    except ValueError as e:
        raise StateCorruptError(
            f"state file {path} has invalid value: {e}") from e


def find_runs(scratch_dir: Path) -> list["RunSummary"]:
    """List runs in ``scratch_dir``, newest first.

    Tolerates a missing directory, a directory with no ``state.json`` files,
    and corrupt ``state.json`` (skipped silently).
    """
    scratch_dir = Path(scratch_dir)
    if not scratch_dir.exists():
        return []

    runs: list[RunSummary] = []
    for entry in sorted(
            scratch_dir.iterdir(),
            key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
            reverse=True,
    ):
        if not entry.is_dir():
            continue
        state_path = entry / STATE_FILENAME
        if not state_path.exists():
            continue
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        outcomes_raw = raw.get("outcomes", [])
        status_counts: dict[str, int] = {}
        for o in outcomes_raw:
            s = o.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        runs.append(RunSummary(
            run_id=raw.get("run_id", entry.name),
            repo=Path(raw.get("repo", "")),
            task=raw.get("task", ""),
            status_counts=status_counts,
            merge_commit=raw.get("merge_commit"),
            base_commit=raw.get("base_commit", ""),
            created_at=float(raw.get("created_at", 0) or 0),
        ))

    return runs


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class RunState:
    """A fully-loaded state that can be resumed."""

    def __init__(self, run_id: str, task: str, repo: Path, base_commit: str,
                 plan: Plan, outcomes: list[ChunkOutcome],
                 created_at: float = 0.0):
        self.run_id = run_id
        self.task = task
        self.repo = repo
        self.base_commit = base_commit
        self.plan = plan
        self.outcomes = outcomes
        self.created_at = created_at

    @property
    def completed_chunks(self) -> list[ChunkOutcome]:
        return [o for o in self.outcomes
                if o.status == AgentStatus.COMPLETED]

    @property
    def incomplete_chunks(self) -> list[ChunkOutcome]:
        return [o for o in self.outcomes
                if o.status != AgentStatus.COMPLETED]

    def outcome_by_id(self, chunk_id: str) -> ChunkOutcome | None:
        for o in self.outcomes:
            if o.chunk.id == chunk_id:
                return o
        return None


class RunSummary:
    """Lightweight info about a run — used by ``pipeline runs`` listing."""

    def __init__(self, run_id: str, repo: Path, task: str,
                 status_counts: dict[str, int],
                 merge_commit: str | None, base_commit: str,
                 created_at: float):
        self.run_id = run_id
        self.repo = repo
        self.task = task
        self.status_counts = status_counts
        self.merge_commit = merge_commit
        self.base_commit = base_commit
        self.created_at = created_at

    @property
    def total_chunks(self) -> int:
        return sum(self.status_counts.values())
