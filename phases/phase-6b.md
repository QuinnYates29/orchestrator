# Phase 6b — Finish resume/runs and fix what it broke

Working in `/home/quinna/tools/orchestrator-p6`. **Never `cd` somewhere else.**
All your files are here and only here.

A previous agent wrote `pipeline/state.py` and the `resume`/`runs` subcommands,
then stopped. It left two live bugs and skipped the tests. Do these in order.

## 1. `executor.py` does not run at all (NameError)

`pipeline/executor.py`, at the end of `execute_plan`:

```python
state_path = config.resolved_scratch_dir() / config.run_id / STATE_PATH
try:
    save_state(state_path, config.run_id, "", plan, outcomes, base_commit)
```

Two bugs in three lines:

- `STATE_PATH` is defined nowhere in the codebase. This raises `NameError` at
  the end of *every* `execute_plan` call. It currently breaks all four tests in
  `tests/test_pipeline_executor.py`.
- `save_state` takes `(path, run_id, task, repo, plan, outcomes, base_commit)`
  — seven arguments. The call passes six: `repo` is missing entirely and `task`
  is hardcoded to `""`, so a resumed run would forget what it was doing.

Fix both. Use the literal filename `"state.json"` (define a module-level
constant if you like, but define it). Pass the real task and repo from
`config`.

Verify with:

```
/home/quinna/tools/orchestrator/.venv/bin/python -m pytest tests/test_pipeline_executor.py -q
```

All 13 must pass. They passed before this phase touched the file.

## 2. Persist after every transition, not once at the end

The task required state to be written after **every** state transition — a run
that dies is exactly the run whose state matters. It is currently written once,
after all waves finish, which is the one moment the state is least useful. The
comment above it claims otherwise; make the code true rather than deleting the
comment.

Write state after each wave completes and after each chunk outcome is recorded.
Keep the `try/except` — state persistence must never take down a run.

## 3. `tests/test_pipeline_state.py`

Round-trip a plan with `depends_on` and mixed outcome statuses. Atomic write
leaves no partial file on failure. `load_state` on a truncated file raises a
clear error rather than returning junk. `find_runs` orders newest first and
tolerates a directory with no `state.json`. Resume selection picks exactly the
non-COMPLETED chunks. A `SKIPPED` chunk whose dependency is now `COMPLETED` is
selected. Token totals surface unreported calls.

## Environment

There is no `python` on PATH. The interpreter lives in a *different* directory
than the repo — that is only where the dependencies happen to be installed, it
is NOT the code you are editing. Run the suite from where you already are:

```
/home/quinna/tools/orchestrator/.venv/bin/python -m pytest tests/ -q
```

`pytest-asyncio` is **not installed** and will not be installed. Do not write
bare `async def test_*`. The convention, in `tests/test_pipeline_solo.py`, is a
synchronous test that drives the coroutine itself:

```python
def _run(coro):
    return asyncio.run(coro)
```

`tests/test_pipeline_tools.py` has 22 pre-existing failures from an earlier
phase that ignored this rule. They are **not yours** — do not fix them and do
not count them as your own. Any failure outside that file *is* yours.

One test in `tests/test_pipeline_verify.py` takes 100 seconds, so iterate on
your own test file and run the full suite once at the end.

## Definition of done

`tests/test_pipeline_executor.py` back to 13 passing, your own state tests
green, and the full suite showing exactly 22 failures — all in
`test_pipeline_tools.py`. Then call `submit_work`. Do not write a summary of
what remains to be done instead of doing it.
