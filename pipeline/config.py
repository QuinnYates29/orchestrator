"""Loads the `pipeline:` block out of the orchestrator's config.yaml.

Deliberately independent of orchestrator.config.load(): that function validates
the whole routing/model graph and raises if anything about the *fleet* is
inconsistent, which has nothing to do with whether a pipeline run is well
configured. Reading the one block we own keeps a pipeline run from failing for
reasons it can't act on.

Unknown keys are an error rather than being ignored. A typo in a role name
(`suprevisor: ds4-light`) would otherwise silently fall back to the default and
be near-impossible to spot in a run that takes hours.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# config.yaml lives next to the package, not inside it.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

ROLE_NAMES = ("planner", "worker", "supervisor", "merger", "explorer")


def _build(cls, raw: dict | None, where: str):
    """Construct a dataclass from a config mapping, rejecting unknown keys."""
    data = dict(raw or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in {where}: {', '.join(sorted(unknown))} "
            f"(valid: {', '.join(sorted(known))})"
        )
    return cls(**data)


@dataclass
class RolesCfg:
    """Which fleet model plays each role in a run.

    Values are whatever the orchestrator accepts as a `model` — a real backend
    name (`ornith`), or a launch profile id (`ds4-full`), which is also a valid
    OpenAI model id per the orchestrator's profile support.
    """
    planner: str = "ds4-full"
    worker: str = "ornith"
    supervisor: str = "ds4-light"
    merger: str = "ds4-full"
    explorer: str = "ornith"

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ROLE_NAMES}


@dataclass
class VerifyCfg:
    command: str | None = None
    timeout_s: float = 600.0
    max_repair_attempts: int = 2

    @property
    def configured(self) -> bool:
        """False means verification is *skipped*, which callers must report as
        such. Absence of a check is never evidence the work is good."""
        return bool(self.command and self.command.strip())


@dataclass
class LimitsCfg:
    max_concurrent_workers: int = 4
    max_agent_turns: int = 60
    tool_output_chars: int = 8000


@dataclass
class PipelineCfg:
    roles: RolesCfg = field(default_factory=RolesCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    limits: LimitsCfg = field(default_factory=LimitsCfg)

    def model_for(self, role: str) -> str:
        if role not in ROLE_NAMES:
            raise KeyError(f"unknown role {role!r} (valid: {', '.join(ROLE_NAMES)})")
        return getattr(self.roles, role)


def from_raw(raw: dict | None) -> PipelineCfg:
    """Build a PipelineCfg from an already-parsed `pipeline:` mapping."""
    data = dict(raw or {})
    unknown = set(data) - {"roles", "verify", "limits"}
    if unknown:
        raise ValueError(
            f"unknown key(s) under `pipeline:`: {', '.join(sorted(unknown))} "
            "(valid: roles, verify, limits)"
        )
    return PipelineCfg(
        roles=_build(RolesCfg, data.get("roles"), "`pipeline.roles`"),
        verify=_build(VerifyCfg, data.get("verify"), "`pipeline.verify`"),
        limits=_build(LimitsCfg, data.get("limits"), "`pipeline.limits`"),
    )


def load(path: str | Path | None = None) -> PipelineCfg:
    """Load the `pipeline:` block. A config file with no such block is fine and
    yields all defaults — the pipeline predates the block and still works
    without it. A *missing file* is also fine for the same reason; only a
    malformed one is an error."""
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    if not target.exists():
        return PipelineCfg()
    raw = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{target} did not parse to a mapping")
    return from_raw(raw.get("pipeline"))
