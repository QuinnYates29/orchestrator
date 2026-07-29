# Phase 3 — DAG chunks and a wave executor

Working in the `orchestrator` repository. This phase adds
`pipeline/executor.py`, modifies `pipeline/models.py`, `pipeline/planner.py`,
`pipeline/cli.py`, and adds `tests/test_pipeline_executor.py`.

## Why

The planner is currently instructed that chunks must be *genuinely
independent*, and told that two chunks needing each other means the split is
wrong. Real tasks routinely have one dependency edge — "add the config field,
then use it" — and under that rule the whole task collapses to a single chunk
and loses all parallelism. Letting the planner declare edges keeps the
parallelism it can actually get while allowing the ordering it needs.

Separately, `cli.py` currently launches every chunk at once. Ornith's
llama-server runs with `--parallel 4`; more concurrent workers than slots just
queue behind each other while consuming context.

## Read first

- `pipeline/cli.py` — `run_pipeline`, especially the scheduling/retry loop.
- `pipeline/planner.py` — `SUBMIT_PLAN_TOOL`, `_system_prompt`, `_parse_plan`.
- `pipeline/models.py` — `PlanChunk`, `Plan`, `ChunkOutcome`, `AgentStatus`.
- `pipeline/workspace.py` — `create_agent_workspace` already takes a base commit.

## What to implement

### 1. `PlanChunk.depends_on: list[str]`

Add the field. Expose it in `SUBMIT_PLAN_TOOL`'s schema with a description
making clear it holds chunk ids that must complete first. Parse it in
`_parse_plan`.

Rewrite the relevant part of `_system_prompt`: chunks should be parallel where
possible, but the planner should declare `depends_on` where one chunk genuinely
builds on another, rather than merging them into one chunk or pretending they
are independent.

### 2. `AgentStatus.SKIPPED = "skipped"`

For a chunk whose dependency failed. It must appear in the final report with
the reason — never silently dropped.

### 3. `pipeline/executor.py`

Move the scheduling loop out of `cli.py` into a reusable executor.

- `def topological_waves(chunks: list[PlanChunk]) -> list[list[PlanChunk]]`
  Pure function, easy to test. Groups chunks into waves where every chunk in
  wave N depends only on chunks in waves < N.
  - Unknown dependency id → raise `ValueError` naming the chunk and the missing
    id. Fail loudly at plan time, not hours later.
  - A cycle → raise `ValueError` naming the chunks involved. Do not silently
    drop an edge to break it.
- `async def execute_plan(...) -> list[ChunkOutcome]`
  Runs the waves. Within a wave, run chunks concurrently bounded by an
  `asyncio.Semaphore(max_concurrent_workers)`. Preserve all existing behavior:
  the supervisor task running across the whole execution phase, one blind retry
  on `KILLED`/`TIMED_OUT`, event emission per chunk.
  - A chunk with a failed/skipped dependency does not run: mark `SKIPPED` with
    a reason naming which dependency failed.
  - **Wave N chunks branch from the accumulated result of their dependencies,
    not from `base_commit`.** After a wave completes, fast-forward the
    dependency work into an integration branch/commit in the scratch area and
    use that as the base for the next wave. `create_agent_workspace` already
    accepts a base commit — this is a parameter change, not a rewrite. If
    combining a wave's outputs conflicts, mark the affected downstream chunks
    `SKIPPED` with the conflict named and carry on; do not attempt an LLM merge
    here (that is Phase 4's job).

### 4. `cli.py`

Replace the inline scheduling loop with a call to `execute_plan`. Behavior for
a plan with no `depends_on` anywhere must be identical to today's, apart from
the concurrency cap.

## Constraints

- `topological_waves` must be deterministic: stable ordering for equal-depth
  chunks (preserve the planner's original order).
- Do not change `worker.py`, `merger.py`, or `supervisor.py`.
- Keep the supervisor covering the entire execution phase, across all waves.

## Tests — `tests/test_pipeline_executor.py`

`topological_waves` is pure, so test it hard: no dependencies → one wave; a
chain → one chunk per wave; a diamond → three waves; stable ordering within a
wave; unknown dependency id raises naming the id; a 2-cycle and a 3-cycle each
raise naming the members; self-dependency raises.

Then executor behavior with a fake client: a wave-2 chunk is not started until
wave 1 finishes; a chunk whose dependency failed is SKIPPED with a reason and
never runs; the semaphore caps concurrency (assert peak in-flight never exceeds
the limit).

## Definition of done

`python -m pytest tests/ -q` fully green, then summarize.
