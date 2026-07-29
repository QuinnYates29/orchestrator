"""Shared dataclasses for the pipeline. No I/O here - pure data + a few
small pure-logic helpers that don't belong to any one phase."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass
class RunConfig:
    repo: Path
    task: str
    orchestrator_url: str = "http://127.0.0.1:8080/v1"
    admin_url: str = "http://127.0.0.1:8080"
    api_key: str | None = None
    scratch_dir: Path | None = None          # default: <repo>/.pipeline-runs
    max_agents: int | None = None            # soft cap passed to the planner as a hint, not enforced
    run_shell_timeout_s: float = 900.0       # 15min dead-man's-switch, not a routine restriction
    fast_repetition_tick_s: float = 3.0      # mechanical duplicate-text check cadence
    review_tick_s: float = 30.0              # baseline periodic ds4-light review cadence
    max_agent_turns: int = 60                # hard ceiling on tool-call turns per agent (belt-and-suspenders)
    keep_scratch: bool = True                # per-agent clones are the audit trail; keep by default
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    # Which fleet model plays each role, from config.yaml's `pipeline.roles`.
    # Defaults kept here so a RunConfig built by hand (tests, embedding callers)
    # still works without loading a config file.
    roles: dict = field(default_factory=lambda: {
        "planner": "ds4-full", "worker": "ornith", "supervisor": "ds4-light",
        "merger": "ds4-full", "explorer": "ornith",
    })

    def resolved_scratch_dir(self) -> Path:
        return self.scratch_dir or (self.repo / ".pipeline-runs")

    def model_for(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"no model configured for role {role!r} "
                           f"(have: {', '.join(sorted(self.roles))})") from None


@dataclass
class PlanChunk:
    id: str
    title: str
    description: str
    scope: list[str] = field(default_factory=list)   # files/dirs this chunk is expected to touch
    context: str = ""                                  # anything not obvious from the repo itself


@dataclass
class Plan:
    chunks: list[PlanChunk]
    shared_context: str = ""


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    KILLED = "killed"
    TIMED_OUT = "timed_out"     # run_shell dead-man's-switch fired
    FAILED = "failed"           # terminal - retry exhausted, or a non-recoverable error


@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: dict
    result_summary: str = ""    # short, for supervisor review - not the full tool output
    ts: float = field(default_factory=time.time)


@dataclass
class AgentState:
    chunk: PlanChunk
    attempt: int
    status: AgentStatus = AgentStatus.PENDING
    workspace: Path | None = None
    branch: str = ""
    base_commit: str = ""
    messages: list[dict] = field(default_factory=list)   # full running conversation, needed to resume the loop
    reasoning_buffer: str = ""                             # live-appended reasoning_content for the current turn
    tool_call_log: list[ToolCallRecord] = field(default_factory=list)
    turns: int = 0
    kill_reason: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def label(self) -> str:
        return f"{self.chunk.id}#{self.attempt}"


@dataclass
class ReviewVerdict:
    verdict: str   # "continue" | "kill"
    reason: str = ""

    @property
    def should_kill(self) -> bool:
        return self.verdict == "kill"


@dataclass
class ChunkOutcome:
    chunk: PlanChunk
    status: AgentStatus
    workspace: Path | None
    kill_reason: str = ""
    attempts: int = 1


@dataclass
class RunReport:
    run_id: str
    plan: Plan
    outcomes: list[ChunkOutcome]
    merge_commit: str | None = None
    merge_summary: str = ""

    @property
    def succeeded(self) -> list[ChunkOutcome]:
        return [o for o in self.outcomes if o.status == AgentStatus.COMPLETED]

    @property
    def failed(self) -> list[ChunkOutcome]:
        return [o for o in self.outcomes if o.status != AgentStatus.COMPLETED]
