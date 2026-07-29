"""Entrypoint for the pipeline's workflows.

    pipeline run     --repo PATH --task "..."    fan-out/merge (plan -> N agents -> merge)
    pipeline solo    --repo PATH --task "..."    single agent, one working directory
    pipeline explore --repo PATH --question "..."   read-only research fan-out
    pipeline resume  --repo PATH <run_id>         resume an interrupted run
    pipeline runs    --repo PATH                  list runs

`run` keeps every flag it had before the subcommand split, so existing
invocations work by inserting the word `run`.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from . import config as pipeline_config
from .client import OrchestratorClient
from .events import EventLog
from .executor import execute_plan
from .merger import merge
from .models import AgentStatus, ChunkOutcome, Plan, RunConfig, RunReport
from .explore import explore_session
from .planner import create_plan
from .solo import solo_session
from .state import RunState, RunSummary, find_runs, load_state, save_state
from .workspace import capture_base_commit, cleanup_workspace

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
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Per-turn generation cap. Heavy reasoners need headroom: ornith "
                        "has produced 35k characters of reasoning_content and hit this "
                        "cap mid-thought, ending the turn with no answer (finish_reason="
                        "length). Raise it if you see empty-response nudges in the log.")


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

    resume_p = sub.add_parser(
        "resume", help="Resume a previously interrupted run from saved state.",
        description="Reloads the saved plan and per-chunk outcomes from state.json under the "
                    "scratch dir, re-runs only the chunks that did not complete (including any "
                    "SKIPPED chunks whose dependencies have since completed), and proceeds to the "
                    "merge phase as normal. --replan re-runs the planner from scratch instead of "
                    "reusing the stored plan.",
    )
    resume_p.add_argument("run_id", help="Run id to resume.")
    resume_p.add_argument("--repo", type=Path, required=True,
                          help="Target repository root.")
    resume_p.add_argument("--scratch-dir", type=Path, default=None,
                          help="Where run artifacts live. Defaults to <repo>/.pipeline-runs.")
    resume_p.add_argument("--replan", action="store_true",
                          help="Re-run the planner from scratch instead of reusing the stored plan.")
    resume_p.add_argument("--orchestrator-url", default="http://127.0.0.1:8080/v1")
    resume_p.add_argument("--admin-url", default="http://127.0.0.1:8080")
    resume_p.add_argument("--api-key", default=os.environ.get("ORCHESTRATOR_API_KEY"))
    resume_p.add_argument("--config", type=Path, default=None,
                          help="Path to config.yaml holding the `pipeline:` block. Defaults to the one "
                               "next to the package.")
    resume_p.add_argument("--load-wait-s", type=float, default=180.0,
                          help="How long each /admin/load call blocks waiting for a model to become resident.")
    resume_p.add_argument("--run-shell-timeout-s", type=float, default=900.0,
                          help="Dead-man's-switch on a single run_shell call (default 15min).")
    resume_p.add_argument("--max-agent-turns", type=int, default=None,
                          help="Override pipeline.limits.max_agent_turns from config.yaml.")
    resume_p.add_argument("--no-load", action="store_true",
                          help="Skip the /admin/load residency check (the backend is already up).")

    runs_p = sub.add_parser(
        "runs", help="List runs in the scratch directory.",
        description="Lists runs newest first: run id, repo, task (truncated), chunk counts by "
                    "status, whether a merge commit exists, wall-clock duration, and token totals "
                    "per model from the event log.",
    )
    runs_p.add_argument("--repo", type=Path, default=None,
                        help="Target repository root - scratch dir defaults to <repo>/.pipeline-runs.")
    runs_p.add_argument("--scratch-dir", type=Path, default=None,
                        help="Scratch directory to list runs from. Defaults to <repo>/.pipeline-runs "
                             "or ./.pipeline-runs if --repo is not given.")
    runs_p.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml holding the `pipeline:` block. Defaults to the one "
                             "next to the package.")
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

        outcomes = await execute_plan(
            client, config, plan, events,
            load_wait_s=load_wait_s, base_commit=base_commit,
        )

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
        ensure_resident=not args.no_load, max_tokens=args.max_tokens,
        max_turns=args.max_agent_turns or pcfg.limits.max_agent_turns,
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


# -- resume -------------------------------------------------------------


def _resolve_scratch(args, repo: Path) -> Path:
    """Pick the scratch dir: explicit --scratch-dir, or <repo>/.pipeline-runs."""
    if args.scratch_dir:
        return args.scratch_dir
    return repo / ".pipeline-runs"


def _cmd_resume(args, pcfg) -> int:
    repo = args.repo.resolve()
    if not repo.exists():
        raise SystemExit(f"{repo} does not exist")
    _require_git_repo(repo)
    scratch = _resolve_scratch(args, repo)
    run_state_path = scratch / args.run_id / "state.json"

    try:
        state = load_state(run_state_path)
    except Exception as e:
        existing = find_runs(scratch)
        msg = f"could not load state for run {args.run_id!r}: {e}"
        if existing:
            msg += "\nExisting runs in " + str(scratch) + ":"
            for r in existing:
                msg += f"\n  {r.run_id}  task={r.task[:50]}"
        raise SystemExit(msg)

    # Build a RunConfig that preserves the original run id so event log and
    # workspace layout match what's already on disk.
    config = RunConfig(
        repo=repo,
        task=state.task,
        orchestrator_url=args.orchestrator_url,
        admin_url=args.admin_url,
        api_key=args.api_key,
        scratch_dir=scratch,
        run_id=state.run_id,
        run_shell_timeout_s=args.run_shell_timeout_s,
        max_agent_turns=args.max_agent_turns or pcfg.limits.max_agent_turns,
        roles=pcfg.roles.as_dict(),
    )

    events = EventLog.for_run(config.resolved_scratch_dir(), config.run_id)
    print(f"resume run {config.run_id}: events at {events.path}", file=sys.stderr)

    if args.replan:
        print("re-running planner from scratch (--replan)", file=sys.stderr)
        new_plan = asyncio.run(create_plan_from_config(config))
        state.plan = new_plan
        # Clear outcomes for chunks we are re-planning so they get re-run.
        for o in state.outcomes:
            o.status = AgentStatus.PENDING

    # Determine which chunks still need work.
    incomplete = [o for o in state.outcomes if o.status != AgentStatus.COMPLETED]
    if not incomplete:
        print(f"all {len(state.outcomes)} chunk(s) already completed — nothing to do")
        return 0

    print(f"resuming with {len(incomplete)} incomplete chunk(s) out of {len(state.outcomes)}:",
          file=sys.stderr)
    for o in incomplete:
        print(f"  [{o.status.value}] {o.chunk.id}: {o.chunk.title}", file=sys.stderr)

    # Find a workspace for the highest existing attempt per chunk so we don't
    # clobber anything, and so new attempts get a fresh number.
    max_attempts: dict[str, int] = {}
    for o in state.outcomes:
        if o.workspace and o.workspace.exists():
            # Parse attempt number from path: .../<chunk_id>-attempt<N>/
            try:
                attempt_str = o.workspace.name.split("-attempt")[-1]
                attempt = int(attempt_str)
            except (ValueError, IndexError):
                attempt = 0
            max_attempts[o.chunk.id] = max(max_attempts.get(o.chunk.id, 0), attempt)

    # Re-run the plan through the executor. The executor will create new
    # workspaces with incremented attempt numbers.
    report = asyncio.run(run_pipeline(config, args.load_wait_s, events, plan=state.plan,
                                      base_commit=state.base_commit,
                                      existing_outcomes=state.outcomes,
                                      max_attempts=max_attempts))

    print(f"\n=== resume {report.run_id} ===")
    print(f"Plan: {len(report.plan.chunks)} chunk(s)")
    for outcome in report.outcomes:
        line = (f"  [{outcome.status.value}] {outcome.chunk.id}: {outcome.chunk.title} "
                f"(attempts: {outcome.attempts})")
        if outcome.kill_reason:
            line += f" - {outcome.kill_reason}"
        print(line)
    print(f"\nSucceeded: {len(report.succeeded)}/{len(report.outcomes)}")
    if report.merge_commit:
        print(f"Merge commit: {report.merge_commit}")
    else:
        print("No new commit was created during the merge pass.")
    print(f"Merge summary: {report.merge_summary}")
    return 0 if not report.failed else 1


async def create_plan_from_config(config: RunConfig) -> Plan:
    """Async helper: create a plan from a RunConfig, used by --replan."""
    async with OrchestratorClient(config.orchestrator_url, config.admin_url, config.api_key) as client:
        return await create_plan(client, config)


# -- runs list ----------------------------------------------------------


def _format_tokens(totals: dict[str, dict[str, int]]) -> str:
    """Format token totals, flagging unreported calls honestly."""
    if not totals:
        return "(no usage events recorded)"
    parts = []
    for model, data in totals.items():
        prompt = int(data["prompt_tokens"] or 0)
        completion = int(data["completion_tokens"] or 0)
        total = prompt + completion
        calls = int(data["calls"] or 0)
        unreported = int(data["unreported_calls"] or 0)
        if unreported:
            parts.append(f"{model}: ≥ {total:,} tokens ({calls - unreported + unreported} calls, "
                         f"{unreported} unreported)")
        else:
            parts.append(f"{model}: {total:,} tokens ({calls} calls)")
    return "  |  ".join(parts)


def _format_counts(status_counts: dict[str, int]) -> str:
    """Format chunk status counts: '5 completed, 1 failed, 2 skipped'."""
    if not status_counts:
        return "(no chunks)"
    parts = []
    for status, count in sorted(status_counts.items()):
        parts.append(f"{count} {status}")
    return ", ".join(parts)


def _format_duration(created_at: float) -> str:
    """Wall-clock duration from created_at to now."""
    if not created_at:
        return "unknown"
    delta = time.time() - created_at
    if delta < 60:
        return f"{delta:.0f}s"
    if delta < 3600:
        return f"{delta / 60:.1f}min"
    return f"{delta / 3600:.1f}h"


def _cmd_runs(args, pcfg) -> int:
    repo = Path(args.repo) if args.repo else Path.cwd()
    if args.scratch_dir:
        scratch = Path(args.scratch_dir)
    else:
        scratch = _resolve_scratch(args, repo)
    if not scratch.exists():
        print(f"no runs found (scratch dir does not exist: {scratch})")
        return 0

    runs = find_runs(scratch)
    if not runs:
        print(f"no runs found in {scratch}")
        return 0

    print(f"{'run_id':<32} {'repo':<30} {'task':<30} {'chunks':<40} "
          f"{'merge':<8} {'duration':<10}")
    print("-" * 150)
    for r in runs:
        run_dir = scratch / r.run_id
        events_path = run_dir / "events.jsonl"
        token_info = ""
        if events_path.exists():
            token_info = _format_tokens(EventLog.token_totals(events_path))
        task_short = r.task[:27] + "..." if len(r.task) > 30 else r.task
        merge_flag = "✓" if r.merge_commit else " "
        print(f"{r.run_id:<32} {str(r.repo):<30} {task_short:<30} "
              f"{_format_counts(r.status_counts):<40} "
              f"{merge_flag:<8} {_format_duration(r.created_at):<10}")
        if token_info:
            print(f"  tokens: {token_info}")

    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    pcfg = pipeline_config.load(args.config)
    if args.command == "solo":
        return _cmd_solo(args, pcfg)
    if args.command == "explore":
        return _cmd_explore(args, pcfg)
    if args.command == "resume":
        return _cmd_resume(args, pcfg)
    if args.command == "runs":
        return _cmd_runs(args, pcfg)
    return _cmd_run(args, pcfg)


if __name__ == "__main__":
    sys.exit(main())
