from __future__ import annotations

from pathlib import Path

from pipeline.models import AgentStatus, ChunkOutcome, Plan, PlanChunk, RunConfig, RunReport


def _outcome(chunk_id: str, status: AgentStatus) -> ChunkOutcome:
    return ChunkOutcome(chunk=PlanChunk(id=chunk_id, title=chunk_id, description=""),
                        status=status, workspace=None)


def test_run_report_succeeded_and_failed_split_correctly():
    outcomes = [
        _outcome("a", AgentStatus.COMPLETED),
        _outcome("b", AgentStatus.FAILED),
        _outcome("c", AgentStatus.COMPLETED),
        _outcome("d", AgentStatus.TIMED_OUT),
        _outcome("e", AgentStatus.KILLED),
    ]
    report = RunReport(run_id="r1", plan=Plan(chunks=[]), outcomes=outcomes)
    assert [o.chunk.id for o in report.succeeded] == ["a", "c"]
    assert [o.chunk.id for o in report.failed] == ["b", "d", "e"]


def test_run_report_all_succeeded():
    outcomes = [_outcome("a", AgentStatus.COMPLETED)]
    report = RunReport(run_id="r1", plan=Plan(chunks=[]), outcomes=outcomes)
    assert len(report.succeeded) == 1
    assert report.failed == []


def test_run_config_resolved_scratch_dir_defaults_under_repo():
    config = RunConfig(repo=Path("/tmp/myrepo"), task="do something")
    assert config.resolved_scratch_dir() == Path("/tmp/myrepo/.pipeline-runs")


def test_run_config_resolved_scratch_dir_respects_explicit_override():
    config = RunConfig(repo=Path("/tmp/myrepo"), task="x", scratch_dir=Path("/var/scratch"))
    assert config.resolved_scratch_dir() == Path("/var/scratch")


def test_run_config_generates_unique_run_ids():
    a = RunConfig(repo=Path("/tmp/x"), task="x")
    b = RunConfig(repo=Path("/tmp/x"), task="x")
    assert a.run_id != b.run_id
