"""Entrypoint for the pipeline's workflows.

    pipeline run     --repo PATH --task "..."    fan-out/merge (plan -> N agents -> merge)
    pipeline solo    --repo PATH --task "..."    single agent, one working directory
    pipeline explore --repo PATH --question "..."   read-only research fan-out

Later phases add `resume` and `runs`. `run` keeps every flag it had
before the subcommand split, so existing invocations work by inserting the word
`run`.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from . import config as pipeline_config
from .client import OrchestratorClient
from .events import EventLog
from .merger import merge
from .models import AgentState, AgentStatus, ChunkOutcome, RunConfig, RunReport
from .explore import explore_session
from .planner import create_plan
from .solo import solo_session
from .supervisor import supervise
from .worker import build_initial_messages, run_agent
from .workspace import capture_base_commit, cleanup_workspace, create_agent_workspace

log = logging.getLogger("pipeline.cli")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", type=Path, default=Path.cwd(),
                   help="Target repository root. Defaults to the current directory.")
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Free-text description of what to implement.")
    task_group.add_argument("--task-file", type=Path, help="Read the task description from a file.")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--admin-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default=os.environ.get("ORCHESTRATOR_API_KEY"))
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config.yaml holding the `pipeline:` block. Defaults to the one "
                        "next to the package.")
    p.add_argument("--scratch-dir", type=Path, default=None,
                   help="Where run artifacts (event log, agent clones) live. Defaults to "
                        "<repo>/.pipeline-runs.")
    p.add_argument("--load-wait-s", type=float, default=180.0,
                   help="How long each /admin/load call blocks waiting for a model to become resident.")
    p.add_argument("--run-shell-timeout-s", type=float, default=900.0,
                   help="Dead-man's-switch on a single run_shell call (default 15min).")
    p.add_argument("--max-agent-turns", type=int, default=None,
                   help="Override pipeline.limits.max_agent_turns from config.yaml.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Multi-agent implementation workflows over the local model fleet.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="Plan, fan out N agents in parallel isolated clones, then merge.",
        description="The planner splits the task, N worker agents implement chunks in parallel under "
                    "supervision, and the merger reviews and integrates the result.",
    )
    _add_common_args(run_p)
    run_p.add_argument("--agents-max", type=int, default=None,
                       help="Soft upper bound on chunk count passed to the planner - it may choose fewer.")
    run_p.add_argument("--fast-tick-s", type=float, default=3.0,
                       help="Mechanical repetition-check cadence.")
    run_p.add_argument("--review-tick-s", type=float, default=30.0,
                       help="Baseline periodic supervisor review cadence.")
    run_p.add_argument("--no-keep-scratch", action="store_true",
                       help="Delete per-agent clones after the run. Default: keep them as the audit trail.")

    solo_p = sub.add_parser(
        "solo", help="Run a single agent directly in one working directory.",
        description="One model, one working directory, no clones and no supervisor. The workspace is "
                    "edited in place - point it at a scratch clone, not a repo you care about.",
    )
    _add_common_args(solo_p)
    solo_p.add_argument("--model", default=None,
                        help="Model or launch profile to drive. Defaults to pipeline.roles.planner "
                             "from config.yaml.")
    solo_p.add_argument("--no-load", action="store_true",
                        help="Skip the /admin/load residency check (the backend is already up).")

    explore_p = sub.add_parser(
        "explore", help="Read-only research fan-out: split a question, answer sub-questions "
                        "with read-only tools, synthesize a final answer.",
        description="Ask a question about a repository without cloning or writing anything. The "
                    "explored repo is read with read_file, list_dir, grep, and glob only. The "
                    "answer fans out to N independent sub-questions answered concurrently, then "
                    "synthesizes a final answer with file-level attribution.",
    )
    # explore uses --question instead of --task; do not call _add_common_args
    # because it forces --task/--task-file to be required.
    explore_p.add_argument("--repo", type=Path, default=Path.cwd(),
                   help="Target repository root. Defaults to the current directory.")
    explore_p.add_argument("--question", required=True,
                           help="Free-text question to answer by reading the repo.")
    explore_p.add_argument("--orchestrator-url", default="http://127.0.0.1:8080/v1")
    explore_p.add_argument("--admin-url", default="http://127.0.0.1:8080")
    explore_p.add_argument("--api-key", default=os.environ.get("ORCHESTRATOR_API_KEY"))
    explore_p.add_argument("--config", type=Path, default=None,
                   help="Path to config.yaml holding the `pipeline:` block. Defaults to the one "
                        "next to the package.")
    explore_p.add_argument("--scratch-dir", type=Path, default=None,
                   help="Where run artifacts (event log, agent clones) live. Defaults to "
                        "<repo>/.pipeline-runs.")
    explore_p.add_argument("--load-wait-s", type=float, default=180.0,
                           help="How long each /admin/load call blocks waiting for a model to become resident.")
    explore_p.add_argument("--run-shell-timeout-s", type=float, default=900.0,
                           help="Dead-man's-switch on a single run_shell call (default 15min).")
    explore_p.add_argument("--max-agent-turns", type=int, default=None,
                   help="Override pipeline.limits.max_agent_turns from config.yaml.")
    explore_p.add_argument("--model", default=None,
                           help="Model or launch profile to use for the explorer role. "
                                "Defaults to pipeline.roles.explorer from config.yaml.")
    explore_p.add_argument("--max-questions", type=int, default=6,
                           help="Maximum number of sub-questions the explorer may produce "
                                "(default 6).")
    explore_p.add_argument("--no-load", action="store_true",
                           help="Skip the /admin/load residency check (the backend is already up).")
    return p


def _read_task(args) -> str:
    if args.task is not None:
        return args.task
    if not args.task_file.exists():
        raise SystemExit(f"task file not found: {args.task_file}")
    return args.task_file.read_text()


def _require_git_repo(repo: Path) -> None:
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a git repository (no .git found) - "
                         "this workflow requires git for per-agent isolation.")


def _build_run_config(args, pcfg) -> RunConfig:
    return RunConfig(
        repo=args.repo.resolve(), task=_read_task(args),
        orchestrator_url=args.orchestrator_url, admin_url=args.admin_url,
        api_key=args.api_key, scratch_dir=args.scratch_dir,
        max_agents=getattr(args, "agents_max", None),
        run_shell_timeout_s=args.run_shell_timeout_s,
        fast_repetition_tick_s=getattr(args, "fast_tick_s", 3.0),
        review_tick_s=getattr(args, "review_tick_s", 30.0),
        max_agent_turns=args.max_agent_turns or pcfg.limits.max_agent_turns,
        keep_scratch=not getattr(args, "no_keep_scratch", False),
        roles=pcfg.roles.as_dict(),
    )


# -- run: fan-out/merge -------------------------------------------------

async def run_pipeline(config: RunConfig, load_wait_s: float, events: EventLog) -> RunReport:
    roles = config.roles
    async with OrchestratorClient(config.orchestrator_url, config.admin_url, config.api_key) as client:
        log.info("run %s: loading %s to plan", config.run_id, roles["planner"])
        events.emit("run_start", repo=str(config.repo), roles=roles, task=config.task[:500])
        await client.admin_load("ds4", profile=roles["planner"], wait_s=load_wait_s)
        plan = await create_plan(client, config)
        events.emit("plan", chunks=[{"id": c.id, "title": c.title} for c in plan.chunks])

        base_commit = await capture_base_commit(config.repo)
        log.info("preparing %d agent workspace(s) from %s", len(plan.chunks), base_commit[:10])

        agents: dict[str, tuple[AgentState, asyncio.Task]] = {}
        for chunk in plan.chunks:
            ws, branch = await create_agent_workspace(config, chunk, 1, base_commit)
            state = AgentState(chunk=chunk, attempt=1, workspace=ws, branch=branch, base_commit=base_commit)
            state.messages = build_initial_messages(chunk, plan)
            task = asyncio.create_task(run_agent(client, config, plan, state), name=state.label)
            agents[chunk.id] = (state, task)

        log.info("loading %s + %s for parallel execution", roles["supervisor"], roles["worker"])
        await asyncio.gather(
            client.admin_load("ds4", profile=roles["supervisor"], wait_s=load_wait_s),
            client.admin_load(roles["worker"], wait_s=load_wait_s),
        )

        supervisor_task = asyncio.create_task(supervise(client, config, agents), name="supervisor")

        outcomes: list[ChunkOutcome] = []
        pending = dict(agents)
        while pending:
            done, _ = await asyncio.wait([t for _, t in pending.values()],
                                          return_when=asyncio.FIRST_COMPLETED)
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

                events.emit("chunk_finished", chunk=chunk_id, status=state.status.value,
                            attempt=state.attempt, turns=state.turns, kill_reason=state.kill_reason)

                if state.status in (AgentStatus.KILLED, AgentStatus.TIMED_OUT) and state.attempt == 1:
                    log.info("retrying %s (attempt 2) after %s: %s",
                             chunk_id, state.status.value, state.kill_reason)
                    retry_ws, retry_branch = await create_agent_workspace(
                        config, state.chunk, 2, base_commit)
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
        log.info("execution phase done: %d/%d succeeded - loading %s to merge",
                 succeeded, len(outcomes), roles["merger"])
        await client.admin_load("ds4", profile=roles["merger"], wait_s=load_wait_s)
        merge_commit, merge_summary = await merge(client, config, plan, outcomes, base_commit)
        events.emit("run_end", succeeded=succeeded, total=len(outcomes), merge_commit=merge_commit)

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
        line = (f"  [{outcome.status.value}] {outcome.chunk.id}: {outcome.chunk.title} "
                f"(attempts: {outcome.attempts})")
        if outcome.kill_reason:
            line += f" - {outcome.kill_reason}"
        print(line)
    print(f"\nSucceeded: {len(report.succeeded)}/{len(report.outcomes)}")
    print(f"Merge commit: {report.merge_commit}" if report.merge_commit
          else "No new commit was created during the merge pass.")
    print(f"Merge summary: {report.merge_summary}")


# -- subcommands --------------------------------------------------------

def _cmd_solo(args, pcfg) -> int:
    repo = args.repo.resolve()
    if not repo.exists():
        raise SystemExit(f"{repo} does not exist")
    config = _build_run_config(args, pcfg)
    model = args.model or pcfg.roles.planner
    events = EventLog.for_run(config.resolved_scratch_dir(), config.run_id)

    print(f"solo run {config.run_id}: {model} in {repo}", file=sys.stderr)
    print(f"events: {events.path}", file=sys.stderr)

    result = asyncio.run(solo_session(
        model=model, repo=repo, task=config.task,
        orchestrator_url=args.orchestrator_url, admin_url=args.admin_url,
        api_key=args.api_key, pipeline_cfg=pcfg, events=events,
        load_wait_s=args.load_wait_s, run_shell_timeout_s=args.run_shell_timeout_s,
        ensure_resident=not args.no_load,
    ))

    print(f"\n=== solo {config.run_id} ===")
    print(f"model: {model}  stop_reason: {result.stop_reason}  turns: {result.turns}  "
          f"tool calls: {result.tool_calls}")
    print(f"tokens: {result.prompt_tokens} prompt / {result.completion_tokens} completion  "
          f"({result.duration_s / 60:.1f} min)")
    print(f"\n{result.final_message}")
    return 0 if result.ok else 1


def _cmd_run(args, pcfg) -> int:
    _require_git_repo(args.repo.resolve())
    config = _build_run_config(args, pcfg)
    events = EventLog.for_run(config.resolved_scratch_dir(), config.run_id)
    print(f"run {config.run_id}: events at {events.path}", file=sys.stderr)
    report = asyncio.run(run_pipeline(config, args.load_wait_s, events))
    _print_report(report)
    return 0 if not report.failed else 1


def _cmd_explore(args, pcfg) -> int:
    repo = args.repo.resolve()
    if not repo.exists():
        raise SystemExit(f"{repo} does not exist")
    config = _build_run_config(args, pcfg)
    config.task = args.question  # use --question instead of --task
    model = args.model or pcfg.roles.explorer
    events = EventLog.for_run(config.resolved_scratch_dir(), config.run_id)

    print(f"explore run {config.run_id}: {model} in {repo}", file=sys.stderr)
    print(f"events: {events.path}", file=sys.stderr)

    result = asyncio.run(explore_session(
        model=model, repo=repo, question=args.question,
        orchestrator_url=args.orchestrator_url, admin_url=args.admin_url,
        api_key=args.api_key, pipeline_cfg=pcfg,
        max_questions=args.max_questions,
        agent_max_turns=args.max_agent_turns or pcfg.limits.max_agent_turns,
        events=events,
        load_wait_s=args.load_wait_s,
        ensure_resident=not args.no_load,
    ))

    # Print the final synthesized answer to stdout so it can be piped.
    print(result.final_answer)
    # Print the per-sub-question breakdown to stderr.
    print(f"\n=== explore {config.run_id} ===", file=sys.stderr)
    print(f"model: {model}  turns: {result.duration_s / 60:.1f} min", file=sys.stderr)
    print(f"tokens: {result.prompt_tokens} prompt / {result.completion_tokens} completion",
          file=sys.stderr)
    print(f"\nSub-questions ({len(result.sub_questions)}):", file=sys.stderr)
    for sa in result.sub_answers:
        if sa.answer:
            print(f"  - {sa.question}", file=sys.stderr)
        else:
            reason = sa.error or sa.stop_reason or "no answer"
            print(f"  - {sa.question} [UNANSWERED: {reason}]", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    pcfg = pipeline_config.load(args.config)
    if args.command == "solo":
        return _cmd_solo(args, pcfg)
    if args.command == "explore":
        return _cmd_explore(args, pcfg)
    return _cmd_run(args, pcfg)


if __name__ == "__main__":
    sys.exit(main())
