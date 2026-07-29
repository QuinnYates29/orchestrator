"""Wave-based execution of a Plan over isolated agent workspaces.

This module owns two things:

1. ``topological_waves`` — a pure, deterministic function that turns a plan
   into an ordered list of waves. Pure means it is trivially testable and has
   no side effects. It is the only place where dependency graph validation
   (unknown ids, cycles, self-deps) happens.

2. ``execute_plan`` — the async workhorse that runs those waves, bounded by
   ``max_concurrent_workers``, with the existing supervisor covering the
   whole execution phase, one blind retry on ``KILLED``/``TIMED_OUT``, and
   ``SKIPPED`` outcomes for chunks whose dependencies failed.

The wave integration step (merging a completed wave's work into a new commit
that downstream waves branch from) is intentionally lightweight: we clone
the real repo into a scratch-area "integration repo" and rebase each
successful chunk onto it. If a rebase conflicts we mark the affected
downstream chunks ``SKIPPED`` and keep going — an LLM-driven merge is
Phase 4's job.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from .client import OrchestratorClient
from .events import EventLog
from .models import AgentState, AgentStatus, ChunkOutcome, Plan, PlanChunk, RunConfig
from .state import save_state
from .supervisor import supervise
from .worker import build_initial_messages, run_agent
from .workspace import create_agent_workspace

log = logging.getLogger("pipeline.executor")

STATE_PATH = "state.json"


def _save_run_state(config: "RunConfig", plan: "Plan", outcomes: list["ChunkOutcome"],
                    base_commit: str, task: str = "", repo: Path | None = None) -> None:
    """Persist state atomically, swallowing any error so it can't take down the run."""
    state_path = config.resolved_scratch_dir() / config.run_id / STATE_PATH
    try:
        save_state(state_path, config.run_id, task or config.task,
                   repo or config.repo, plan, outcomes, base_commit)
    except Exception:
        log.exception("failed to persist state for run %s", config.run_id)


# ---------------------------------------------------------------------------
# Pure graph logic
# ---------------------------------------------------------------------------

def topological_waves(chunks: list[PlanChunk]) -> list[list[PlanChunk]]:
    """Group chunks into waves where every chunk in wave *N* depends only on
    chunks in waves < *N*.

    Raises
    ------
    ValueError
        If any chunk depends on an unknown id, depends on itself, or the
        dependency graph contains a cycle. The message names the offending
        id(s) so the planner (or a human) knows what to fix.

    The returned list preserves the planner's original ordering within each
    wave — ties broken by first appearance in ``chunks``.
    """
    if not chunks:
        return []

    # --- index by id, preserving order ----------------------------------
    by_id: dict[str, PlanChunk] = {}
    for c in chunks:
        if c.id in by_id:
            raise ValueError(f"duplicate chunk id {c.id!r}")
        by_id[c.id] = c

    # --- validate edges --------------------------------------------------
    for c in chunks:
        for dep in c.depends_on:
            if dep == c.id:
                raise ValueError(f"chunk {c.id!r} depends on itself")
            if dep not in by_id:
                raise ValueError(
                    f"chunk {c.id!r} depends on unknown chunk {dep!r}")

    # --- Kahn's algorithm with ordered queues ---------------------------
    remaining_deps: dict[str, set[str]] = {c.id: set(c.depends_on) for c in chunks}
    dependents: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        for dep in c.depends_on:
            dependents[dep].append(c.id)

    order = {c.id: i for i, c in enumerate(chunks)}
    waves: list[list[str]] = []
    queue: deque[str] = deque()
    # Initial wave: chunks with no dependencies, in planner order.
    for c in chunks:
        if not c.depends_on:
            queue.append(c.id)

    processed = 0
    total = len(chunks)

    while queue:
        wave_ids = list(queue)
        queue.clear()
        waves.append(wave_ids)
        processed += len(wave_ids)

        # Collect next-wave candidates, preserving planner order.
        next_wave: list[str] = []
        for wave_id in wave_ids:
            for dep_id in dependents[wave_id]:
                remaining_deps[dep_id].discard(wave_id)
                if not remaining_deps[dep_id]:
                    next_wave.append(dep_id)

        # Stabilize: reorder next_wave by first appearance in ``chunks``.
        next_wave.sort(key=lambda cid: order[cid])
        for cid in next_wave:
            queue.append(cid)

    if processed != total:
        cycle_chunks = [c for c in chunks if remaining_deps[c.id]]
        raise ValueError(
            f"cycle detected among chunks: "
            f"{', '.join(c.id for c in cycle_chunks)}")

    return [[by_id[cid] for cid in wave] for wave in waves]


# ---------------------------------------------------------------------------
# Executor internals
# ---------------------------------------------------------------------------

def _skipped_outcome(chunk: PlanChunk, reason: str) -> ChunkOutcome:
    return ChunkOutcome(
        chunk=chunk,
        status=AgentStatus.SKIPPED,
        workspace=None,
        kill_reason=reason,
        attempts=0,
    )


def _find_failed_dep(
    chunk: PlanChunk,
    outcomes_by_id: dict[str, ChunkOutcome],
) -> str | None:
    """Return the first dependency id that didn't complete, or None."""
    terminal_bad = {
        AgentStatus.FAILED, AgentStatus.KILLED, AgentStatus.TIMED_OUT,
        AgentStatus.SKIPPED, AgentStatus.VERIFY_FAILED,
    }
    for dep_id in chunk.depends_on:
        dep_outcome = outcomes_by_id.get(dep_id)
        if dep_outcome and dep_outcome.status in terminal_bad:
            return dep_id
    return None


async def _init_integration_repo(
    integration_path: Path, repo: Path, base_commit: str,
) -> None:
    """Create or reset the scratch integration repo at ``base_commit``."""
    from ._procutil import run_argv

    if integration_path.exists() and (integration_path / ".git").exists():
        # Reset to base_commit cleanly.
        await run_argv(["git", "checkout", "--quiet", base_commit], cwd=integration_path)
        await run_argv(["git", "reset", "--hard", "HEAD"], cwd=integration_path)
    else:
        if integration_path.exists():
            import shutil
            shutil.rmtree(integration_path)
        integration_path.mkdir(parents=True, exist_ok=True)
        await run_argv(["git", "clone", "--local", "--quiet", str(repo), str(integration_path)])
        await run_argv(["git", "checkout", "--quiet", base_commit], cwd=integration_path)


async def _get_integration_head(integration_path: Path) -> str:
    from ._procutil import run_argv
    _, out, _ = await run_argv(["git", "rev-parse", "HEAD"], cwd=integration_path)
    return out.strip()


async def _integrate_chunk(
    integration_path: Path,
    chunk: PlanChunk,
    outcome: ChunkOutcome,
    events: EventLog,
) -> str | None:
    """Try to merge a successful chunk's work into the integration repo.

    Returns the new integration HEAD on success, or ``None`` if the merge
    conflicted.
    """
    from ._procutil import run_argv

    if outcome.workspace is None or outcome.status != AgentStatus.COMPLETED:
        return None

    # Find the chunk's commit in its workspace.
    _, commit_out, _ = await run_argv(["git", "rev-parse", "HEAD"], cwd=outcome.workspace)
    chunk_commit = commit_out.strip()
    if not chunk_commit:
        return None

    # The integration repo is a clone of the *original* repo, so the chunk's
    # new commits do not exist in it yet. Fetch them across before merging -
    # without this, `git merge <sha>` fails with "not something we can merge".
    code, _, err = await run_argv(
        ["git", "fetch", "--quiet", str(outcome.workspace), chunk_commit],
        cwd=integration_path,
    )
    if code != 0:
        log.warning("could not fetch chunk %s from its workspace: %s", chunk.id, err.strip())
        events.emit("merge_conflict", chunk=chunk.id, reason=f"fetch failed: {err.strip()}")
        return None

    # Try to merge chunk_commit into the integration repo.
    code, _, err = await run_argv(
        ["git", "merge", "--no-ff", "--quiet", chunk_commit,
         "-m", f"Merge chunk {chunk.id}"],
        cwd=integration_path,
    )
    if code != 0:
        log.warning("merge conflict integrating chunk %s: %s", chunk.id, err.strip())
        events.emit("merge_conflict", chunk=chunk.id, reason=err.strip())
        # Leave the integration repo usable. Without this the conflict markers
        # stay staged and every later chunk fails with "Merging is not
        # possible because you have unmerged files" - one real conflict was
        # turning into a cascade of fake ones.
        await run_argv(["git", "merge", "--abort"], cwd=integration_path)
        return None

    _, new_head, _ = await run_argv(["git", "rev-parse", "HEAD"], cwd=integration_path)
    return new_head.strip()


async def _run_one_chunk(
    client: OrchestratorClient,
    config: RunConfig,
    plan: Plan,
    chunk: PlanChunk,
    base_commit: str,
    events: EventLog,
    attempt: int,
    semaphore: asyncio.Semaphore | None = None,
) -> AgentState:
    """Run one attempt of one chunk. Returns the final AgentState."""
    ws_path, branch = await create_agent_workspace(config, chunk, attempt, base_commit)
    state = AgentState(
        chunk=chunk, attempt=attempt,
        workspace=ws_path, branch=branch, base_commit=base_commit,
    )
    state.messages = build_initial_messages(chunk, plan)
    events.emit("chunk_started", chunk=chunk.id, attempt=attempt,
                base_commit=base_commit[:10])

    if semaphore is not None:
        async with semaphore:
            task = asyncio.create_task(run_agent(client, config, plan, state),
                                       name=state.label)
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("agent %s crashed unexpectedly", state.label)
                if state.status == AgentStatus.RUNNING:
                    state.status = AgentStatus.FAILED
                    state.kill_reason = state.kill_reason or "unexpected crash - see logs"
    else:
        task = asyncio.create_task(run_agent(client, config, plan, state),
                                   name=state.label)
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("agent %s crashed unexpectedly", state.label)
            if state.status == AgentStatus.RUNNING:
                state.status = AgentStatus.FAILED
                state.kill_reason = state.kill_reason or "unexpected crash - see logs"

    events.emit("chunk_finished", chunk=chunk.id, status=state.status.value,
                attempt=state.attempt, turns=state.turns,
                kill_reason=state.kill_reason)
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def execute_plan(
    client: OrchestratorClient,
    config: RunConfig,
    plan: Plan,
    events: EventLog,
    *,
    load_wait_s: float,
    base_commit: str,
    completed_outcomes: dict[str, ChunkOutcome] | None = None,
) -> list[ChunkOutcome]:
    """Run ``plan`` wave by wave, bounded by ``max_concurrent_workers``.

    Returns the full list of ``ChunkOutcome`` — completed, failed, and
    skipped — in planner order, never silently dropping SKIPPED chunks.

    ``completed_outcomes`` maps chunk id -> outcome for chunks a previous run
    already finished. Those are carried through untouched rather than re-run,
    which is the whole point of ``pipeline resume``: their work is already in
    their clone and re-running would discard it. They still count as satisfied
    dependencies, so a SKIPPED chunk whose dependency has since completed
    becomes runnable again.
    """
    completed_outcomes = completed_outcomes or {}
    waves = topological_waves(plan.chunks)
    semaphore = asyncio.Semaphore(config.limits.max_concurrent_workers)

    # Map chunk id -> integration commit once a wave has produced one.
    effective_base: dict[str, str] = {}
    for c in plan.chunks:
        effective_base[c.id] = base_commit

    # Integration repo under scratch so it's cleaned up with the run.
    integration_path = config.resolved_scratch_dir() / config.run_id / "integration"

    outcomes_by_id: dict[str, ChunkOutcome] = {}
    outcomes: list[ChunkOutcome] = []

    # Seed with work a previous run already finished. Doing this before the
    # wave loop means _find_failed_dep sees them as satisfied dependencies,
    # so downstream chunks that were SKIPPED last time can run now.
    for chunk in plan.chunks:
        prior = completed_outcomes.get(chunk.id)
        if prior is not None:
            outcomes_by_id[chunk.id] = prior
            outcomes.append(prior)
            events.emit("chunk_reused", chunk=chunk.id, status=prior.status.value)

    # The integration repo must exist before anything tries to merge into it.
    # _init_integration_repo was written but never called, so _integrate_chunk
    # ran git in a directory that did not exist. Initialise unconditionally:
    # integration runs per completed chunk, not per wave boundary, so a
    # single-wave plan reaches it too. It is a local clone, so this is cheap.
    await _init_integration_repo(integration_path, config.repo, base_commit)

    # --- Supervisor covers the *entire* execution phase across all waves.
    agents: dict[str, tuple[AgentState, asyncio.Task]] = {}
    supervisor_task = asyncio.create_task(
        supervise(client, config, agents), name="supervisor")

    chunk_order = {c.id: i for i, c in enumerate(plan.chunks)}

    # Persist state after every transition — a run that dies is the one we
    # need the state for. Never take down the run because state persistence
    # failed.
    for wave in waves:
        # Decide which chunks in this wave can actually run.
        ready: list[PlanChunk] = []
        for chunk in wave:
            if chunk.id in completed_outcomes:
                continue  # already done by the run being resumed
            failed_dep = _find_failed_dep(chunk, outcomes_by_id)
            if failed_dep:
                outcome = _skipped_outcome(
                    chunk,
                    f"dependency {failed_dep!r} did not succeed "
                    f"(status: {outcomes_by_id[failed_dep].status.value})",
                )
                outcomes_by_id[chunk.id] = outcome
                outcomes.append(outcome)
                events.emit("chunk_skipped", chunk=chunk.id,
                            reason=f"dependency {failed_dep!r} failed")
                _save_run_state(config, plan, outcomes, base_commit)
                # If downstream chunks depend on this one, mark them
                # skipped too (they'll be caught in their own wave).
                # We also propagate: any chunk that depends on this
                # skipped chunk will be detected when its wave runs.
            else:
                ready.append(chunk)

        if ready:
            # Determine the base for each ready chunk: if it depends on
            # earlier waves, use the latest integration commit; otherwise
            # use base_commit.
            coros: list[tuple[PlanChunk, asyncio.Task[AgentState]]] = []
            pending_tasks: dict[str, asyncio.Task[AgentState]] = {}
            for chunk in ready:
                base = effective_base[chunk.id]
                task = asyncio.create_task(
                    _run_one_chunk(
                        client, config, plan, chunk, base, events,
                        1, semaphore,
                    ), name=chunk.id)
                pending_tasks[chunk.id] = task

            # Wait for all to finish.
            for cid, task in pending_tasks.items():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    log.exception("wave task %s raised", cid)
                finally:
                    state = task.result() if not task.cancelled() else None  # type: ignore[possibly-undefined]

            # Now process outcomes.
            for chunk in ready:
                task = pending_tasks[chunk.id]
                if task.cancelled():
                    continue
                state = task.result()  # type: ignore[possibly-undefined]

                if state.status in (AgentStatus.KILLED, AgentStatus.TIMED_OUT) \
                        and state.attempt == 1:
                    # One blind retry from the same base.
                    log.info("retrying %s (attempt 2) after %s: %s",
                             chunk.id, state.status.value, state.kill_reason)
                    retry_state = await _run_one_chunk(
                        client, config, plan, chunk,
                        effective_base[chunk.id], events, 2, semaphore,
                    )
                    state = retry_state

                outcome = ChunkOutcome(
                    chunk=chunk, status=state.status, workspace=state.workspace,
                    kill_reason=state.kill_reason, attempts=state.attempt,
                )
                outcomes_by_id[chunk.id] = outcome
                outcomes.append(outcome)

                # If completed, try integrating into the integration repo.
                if outcome.status == AgentStatus.COMPLETED:
                    new_head = await _integrate_chunk(
                        integration_path, chunk, outcome, events,
                    )
                    if new_head:
                        effective_base[chunk.id] = new_head

                _save_run_state(config, plan, outcomes, base_commit)

    supervisor_task.cancel()
    try:
        await supervisor_task
    except asyncio.CancelledError:
        pass

    # Sort outcomes in planner order.
    outcomes.sort(key=lambda o: chunk_order[o.chunk.id])

    return outcomes
