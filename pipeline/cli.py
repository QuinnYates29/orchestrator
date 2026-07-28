"""Entrypoint: python -m pipeline.cli --repo PATH --task "..."

Sequences: plan (ds4-full) -> prepare per-agent git clones -> swap to
ds4-light + ornith -> parallel execute under supervision (with one retry per
chunk on kill/timeout) -> swap back to ds4-full -> merge -> report.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .client import OrchestratorClient
from .merger import merge
from .models import AgentState, AgentStatus, ChunkOutcome, RunConfig, RunReport
from .planner import create_plan
from .supervisor import supervise
from .worker import build_initial_messages, run_agent
from .workspace import capture_base_commit, cleanup_workspace, create_agent_workspace

log = logging.getLogger("pipeline.cli")


def parse_args(argv=None) -> tuple[RunConfig, float]:
    p = argparse.ArgumentParser(
        description="Multi-agent implementation pipeline: ds4-full plans, N Ornith agents implement "
                    "chunks in parallel under ds4-light supervision, ds4-full reviews and merges.",
    )
    p.add_argument("--repo", type=Path, default=Path.cwd(),
                   help="Target repository root. Defaults to the current directory. Must be a git repo.")
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Free-text description of what to implement.")
    task_group.add_argument("--task-file", type=Path, help="Read the task description from a file.")
    p.add_argument("--agents-max", type=int, default=None,
                   help="Soft upper bound on chunk count passed to the planner - it may choose fewer.")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--admin-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default=os.environ.get("ORCHESTRATOR_API_KEY"))
    p.add_argument("--scratch-dir", type=Path, default=None,
                   help="Where per-agent clones live. Defaults to <repo>/.pipeline-runs.")
    p.add_argument("--load-wait-s", type=float, default=180.0,
                   help="How long each /admin/load call blocks waiting for a model to become resident.")
    p.add_argument("--run-shell-timeout-s", type=float, default=900.0,
                   help="Dead-man's-switch on a single run_shell call (default 15min).")
    p.add_argument("--fast-tick-s", type=float, default=3.0,
                   help="Mechanical repetition-check cadence.")
    p.add_argument("--review-tick-s", type=float, default=30.0,
                   help="Baseline periodic ds4-light review cadence.")
    p.add_argument("--max-agent-turns", type=int, default=60,
                   help="Hard ceiling on tool-call turns per agent (belt-and-suspenders, distinct from "
                       "ds4-light's judgment-based kill).")
    p.add_argument("--no-keep-scratch", action="store_true",
                   help="Delete per-agent clones after the run. Default: keep them, they're the audit trail.")
    args = p.parse_args(argv)

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a git repository (no .git found) - "
                         "this pipeline requires git for per-agent isolation.")

    task = args.task if args.task is not None else args.task_file.read_text()
    config = RunConfig(
        repo=repo, task=task, orchestrator_url=args.orchestrator_url, admin_url=args.admin_url,
        api_key=args.api_key, scratch_dir=args.scratch_dir, max_agents=args.agents_max,
        run_shell_timeout_s=args.run_shell_timeout_s, fast_repetition_tick_s=args.fast_tick_s,
        review_tick_s=args.review_tick_s, max_agent_turns=args.max_agent_turns,
        keep_scratch=not args.no_keep_scratch,
    )
    return config, args.load_wait_s


async def run_pipeline(config: RunConfig, load_wait_s: float) -> RunReport:
    async with OrchestratorClient(config.orchestrator_url, config.admin_url, config.api_key) as client:
        log.info("run %s: loading ds4-full to plan", config.run_id)
        await client.admin_load("ds4", profile="ds4-full", wait_s=load_wait_s)
        plan = await create_plan(client, config)

        base_commit = await capture_base_commit(config.repo)
        log.info("preparing %d agent workspace(s) from %s", len(plan.chunks), base_commit[:10])

        agents: dict[str, tuple[AgentState, asyncio.Task]] = {}
        for chunk in plan.chunks:
            ws, branch = await create_agent_workspace(config, chunk, 1, base_commit)
            state = AgentState(chunk=chunk, attempt=1, workspace=ws, branch=branch, base_commit=base_commit)
            state.messages = build_initial_messages(chunk, plan)
            task = asyncio.create_task(run_agent(client, config, plan, state), name=state.label)
            agents[chunk.id] = (state, task)

        log.info("loading ds4-light + ornith for parallel execution")
        await asyncio.gather(
            client.admin_load("ds4", profile="ds4-light", wait_s=load_wait_s),
            client.admin_load("ornith", wait_s=load_wait_s),
        )

        supervisor_task = asyncio.create_task(supervise(client, config, agents), name="supervisor")

        outcomes: list[ChunkOutcome] = []
        pending = dict(agents)
        while pending:
            done, _ = await asyncio.wait([t for _, t in pending.values()], return_when=asyncio.FIRST_COMPLETED)
            for chunk_id in list(pending):
                state, task = pending[chunk_id]
                if task not in done:
                    continue
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    log.exception("agent %s crashed unexpectedly", state.label)
                    if state.status == AgentStatus.RUNNING:
                        state.status = AgentStatus.FAILED
                        state.kill_reason = state.kill_reason or "unexpected crash - see logs"

                if state.status in (AgentStatus.KILLED, AgentStatus.TIMED_OUT) and state.attempt == 1:
                    log.info("retrying %s (attempt 2) after %s: %s",
                             chunk_id, state.status.value, state.kill_reason)
                    retry_ws, retry_branch = await create_agent_workspace(config, state.chunk, 2, base_commit)
                    retry_state = AgentState(chunk=state.chunk, attempt=2, workspace=retry_ws,
                                             branch=retry_branch, base_commit=base_commit)
                    retry_context = (f"A previous attempt at this chunk was stopped: {state.kill_reason}. "
                                     "Avoid repeating whatever caused that.")
                    retry_state.messages = build_initial_messages(state.chunk, plan, retry_context)
                    retry_task = asyncio.create_task(
                        run_agent(client, config, plan, retry_state), name=retry_state.label,
                    )
                    pending[chunk_id] = (retry_state, retry_task)
                    agents[chunk_id] = (retry_state, retry_task)
                else:
                    outcomes.append(ChunkOutcome(
                        chunk=state.chunk, status=state.status, workspace=state.workspace,
                        kill_reason=state.kill_reason, attempts=state.attempt,
                    ))
                    del pending[chunk_id]

        supervisor_task.cancel()
        try:
            await supervisor_task
        except asyncio.CancelledError:
            pass

        succeeded = sum(1 for o in outcomes if o.status == AgentStatus.COMPLETED)
        log.info("execution phase done: %d/%d succeeded - loading ds4-full to merge",
                 succeeded, len(outcomes))
        await client.admin_load("ds4", profile="ds4-full", wait_s=load_wait_s)
        merge_commit, merge_summary = await merge(client, config, plan, outcomes, base_commit)

        if not config.keep_scratch:
            for outcome in outcomes:
                if outcome.workspace is not None:
                    cleanup_workspace(outcome.workspace)

        return RunReport(run_id=config.run_id, plan=plan, outcomes=outcomes,
                         merge_commit=merge_commit, merge_summary=merge_summary)


def _print_report(report: RunReport) -> None:
    print(f"\n=== Run {report.run_id} ===")
    print(f"Plan: {len(report.plan.chunks)} chunk(s)")
    for outcome in report.outcomes:
        line = f"  [{outcome.status.value}] {outcome.chunk.id}: {outcome.chunk.title} (attempts: {outcome.attempts})"
        if outcome.kill_reason:
            line += f" - {outcome.kill_reason}"
        print(line)
    print(f"\nSucceeded: {len(report.succeeded)}/{len(report.outcomes)}")
    print(f"Merge commit: {report.merge_commit}" if report.merge_commit
          else "No new commit was created during the merge pass.")
    print(f"Merge summary: {report.merge_summary}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config, load_wait_s = parse_args(argv)
    report = asyncio.run(run_pipeline(config, load_wait_s))
    _print_report(report)
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
