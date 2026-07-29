# Phase 3b — Finish the wave executor

Working in `/home/quinna/tools/orchestrator-p3`. **Never `cd` somewhere else.**
All your files are here and only here.

A previous agent wrote `pipeline/executor.py` and stopped before finishing.
Three things remain. Do them in this order — the first is a live bug and the
tests in step 3 are what would have caught it.

## 1. Fix `topological_waves` (it does not work at all)

`pipeline/executor.py`, in `topological_waves`:

```python
order = {cid: i for i, cid in enumerate(chunks)}
```

`enumerate(chunks)` yields `PlanChunk` objects, not ids. `PlanChunk` is an
unhashable dataclass, so this raises `TypeError` on any plan that has a
dependency. It should map chunk **id** to index. Fix it, then read the rest of
the function and check it for the same confusion between a chunk and its id.

Confirm by hand before moving on:

```
/home/quinna/tools/orchestrator/.venv/bin/python -c "
from pipeline.executor import topological_waves
from pipeline.models import PlanChunk
mk=lambda i,d=None: PlanChunk(id=i,title='t',description='d',depends_on=d or [])
print([[c.id for c in w] for w in topological_waves([mk('a'),mk('b',['a'])])])
"
```

That must print `[['a'], ['b']]`.

## 2. Wire `execute_plan` into `cli.py`

`cli.py` already imports `execute_plan` but `run_pipeline` still contains the
old inline scheduling loop. Replace that loop with the call. Behavior for a
plan with no `depends_on` anywhere must be identical to today's, apart from the
concurrency cap.

Do not change `worker.py`, `merger.py`, or `supervisor.py`.

## 3. `tests/test_pipeline_executor.py`

`topological_waves` is pure, so test it hard: no dependencies → one wave; a
chain → one chunk per wave; a diamond → three waves; stable ordering within a
wave preserves the planner's original order; an unknown dependency id raises
`ValueError` naming the id; a 2-cycle and a 3-cycle each raise naming the
members; self-dependency raises.

Then executor behavior with a fake client in the style of
`tests/test_pipeline_solo.py`: a wave-2 chunk is not started until wave 1
finishes; a chunk whose dependency failed is `SKIPPED` with a reason and never
runs; the semaphore caps concurrency (assert peak in-flight never exceeds the
limit).

## Environment

There is no `python` on PATH. The interpreter lives in a *different* directory
than the repo — that is only where the dependencies happen to be installed, it
is NOT the code you are editing. Run the suite from where you already are:

```
/home/quinna/tools/orchestrator/.venv/bin/python -m pytest tests/ -q
```

`pytest-asyncio` is **not installed** and will not be installed. Do not write
bare `async def test_*`. The repo convention, in `tests/test_pipeline_solo.py`,
is a synchronous test that drives the coroutine itself:

```python
def _run(coro):
    return asyncio.run(coro)
```

`tests/test_pipeline_tools.py` has 22 pre-existing failures from an earlier
phase that ignored this rule. They are **not yours** — do not fix them, do not
let them stop you, and do not count them as your own failures.

One test in `tests/test_pipeline_verify.py` takes 100 seconds of real time, so
the full suite is slow. Run `pytest tests/test_pipeline_executor.py -q` while
iterating and the full suite only at the end.

## Definition of done

Your own tests green and the full suite showing no *new* failures, then call
`submit_work`. Do not write a summary of what remains to be done instead of
doing it.
