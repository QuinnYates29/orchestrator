# Phase 6 — Resume and run listing

Working in the `orchestrator` repository. This phase adds `pipeline/state.py`,
extends `pipeline/cli.py` with `resume` and `runs` subcommands, wires state
persistence into `pipeline/executor.py`, and adds `tests/test_pipeline_state.py`.

## Why

A run currently exists only in memory. A crash, a killed process, or a machine
that needs its GPU back destroys hours of agent work that is still sitting in
the per-agent clones on disk. The clones are kept by default precisely because
they are the audit trail — but nothing can read them back. Persisting the plan
and per-chunk outcomes makes that work recoverable, and makes token cost
comparable across runs.

## Read first

- `pipeline/events.py` — `EventLog`, `token_totals`, and why it appends
  per-event rather than holding a handle.
- `pipeline/models.py` — `Plan`, `PlanChunk`, `ChunkOutcome`, `AgentStatus`,
  `RunConfig.resolved_scratch_dir`.
- `pipeline/executor.py` (Phase 3) — where outcomes are produced.
- `pipeline/workspace.py` — the run/attempt directory layout.

## What to implement

### 1. `pipeline/state.py`

`<scratch>/<run_id>/state.json`, written after **every** state transition, not
only at the end — a run that dies is exactly the run whose state matters.

```python
def save_state(path: Path, run_id: str, task: str, repo: Path,
               plan: Plan, outcomes: list[ChunkOutcome],
               base_commit: str) -> None
def load_state(path: Path) -> RunState
def find_runs(scratch_dir: Path) -> list[RunSummary]   # newest first
```

Write atomically: serialize to a temp file in the same directory, then
`os.replace`. A half-written `state.json` is worse than none, and a run being
killed mid-write is the expected case, not a rare one.

Round-tripping must preserve enough to resume: plan chunks with their
`depends_on`, each chunk's status, attempts, workspace path, kill reason, and
verify result. Enum values serialize as their string values.

### 2. `pipeline resume <run_id>`

- Locate the run under the scratch dir; a missing run is a clear error listing
  what runs do exist.
- Reload plan and outcomes. Re-run only chunks that are not `COMPLETED`.
  `SKIPPED` chunks whose dependencies have since completed become runnable
  again — that is much of the point.
- Reuse the existing per-agent clones as the audit trail; new attempts get new
  attempt numbers so nothing is overwritten. `create_agent_workspace` already
  raises if a workspace exists — do not defeat that by deleting; increment.
- Proceed to the merge phase as normal once execution finishes.
- `--replan` re-runs the planner from scratch instead of reusing the stored
  plan, for when the plan itself was the problem.

### 3. `pipeline runs`

List runs newest first: run id, repo, task (truncated), chunk counts by status,
whether a merge commit exists, wall-clock duration, and token totals per model
from `EventLog.token_totals`.

Report unreported-usage calls honestly (`EventLog.emit_usage` records
`reported: false` when a backend omits usage). A total that silently omits
calls the backend never accounted for would read as cheaper than it was — show
it as a floor, e.g. `≥ 12,400 tokens (3 calls unreported)`.

Add `--scratch-dir` since there is no `--repo` to derive it from, defaulting to
`./.pipeline-runs`.

## Constraints

- State writing must never take down a run. Wrap in try/except and log, the way
  `EventLog.emit` does.
- Do not change the on-disk layout of the agent clones.
- `resume` must not re-run the planner unless `--replan` is passed. Re-planning
  by default would discard completed work.

## Tests — `tests/test_pipeline_state.py`

Round-trip a plan with `depends_on` and mixed outcome statuses; atomic write
leaves no partial file on failure; `load_state` on a truncated file raises a
clear error rather than returning junk; `find_runs` orders newest first and
tolerates a directory with no `state.json`; resume selection logic picks exactly
the non-COMPLETED chunks; a SKIPPED chunk whose dependency is now COMPLETED is
selected; token totals surface unreported calls.

## Environment — read this before running anything

There is no `python` on PATH. The interpreter with the test dependencies is:

```
/home/quinna/tools/orchestrator/.venv/bin/python
```

Run the suite with exactly:

```
/home/quinna/tools/orchestrator/.venv/bin/python -m pytest tests/ -q
```

`pytest-asyncio` is **not installed** and will not be installed. Do not write
bare `async def test_*` — pytest cannot run them and reports them as failures.
The repo convention, in `tests/test_pipeline_solo.py`, is a synchronous test
that drives the coroutine itself:

```python
def _run(coro):
    return asyncio.run(coro)

def test_something():
    result = _run(some_async_fn(...))
```

You have a 60-turn budget and each turn is slow. Do not spend turns guessing at
the environment. Call `submit_work` as soon as the suite is green — running out
of turns loses the summary.

## Definition of done

`python -m pytest tests/ -q` fully green, then summarize.
