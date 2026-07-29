# Phase 5 — Read-only research fan-out (`pipeline explore`)

Working in the `orchestrator` repository. This phase adds
`pipeline/explore.py`, extends `pipeline/cli.py` with an `explore`
subcommand, and adds `tests/test_pipeline_explore.py`.

## Why

Not every question needs a git clone and a merge. "How does routing pick a
backend?" is answered by reading, and reading fans out perfectly. This is the
cheapest workflow in the harness — no clones, no writes, no supervisor, no
merge — and therefore the one that will get used most often.

## Read first

- `pipeline/planner.py` — the explore/submit tool-loop shape to mirror.
- `pipeline/tools.py` — `READ_ONLY_TOOL_SCHEMAS` (includes grep/glob after
  Phase 1).
- `pipeline/solo.py` — the single-agent loop, and `ensure_model_resident`.
- `pipeline/cli.py` — subcommand structure and `_add_common_args`.

## What to implement

### 1. `pipeline/explore.py`

Three stages:

**Split.** One call to the `explorer` role model with a `submit_questions`
tool: given the user's question and a listing of the repo root, produce 2-6
independent sub-questions. If the question is simple, one sub-question is a
valid answer — do not pad to hit a number.

**Answer.** One agent per sub-question, concurrently, bounded by
`limits.max_concurrent_workers`. Each gets `READ_ONLY_TOOL_SCHEMAS` against the
real repo directory. No writes, no `run_shell` — read-only means read-only,
and `run_shell` would make it trivially not so. Each agent has its own turn
budget (default lower than a worker's — call it 15; exploration that needs 60
turns is a sign the question was wrong).

**Synthesize.** One call giving the original question and every sub-answer,
producing the final answer. It must attribute claims to files where it can, and
must say plainly when a sub-question went unanswered rather than papering over
it.

Return a dataclass with the question, sub-questions, per-question answers, the
final answer, and token totals.

### 2. `cli.py` — `explore` subcommand

```
pipeline explore --repo PATH --question "..."
```

Shares `--orchestrator-url`, `--admin-url`, `--api-key`, `--config`,
`--scratch-dir`, `--load-wait-s`. Adds `--model` (defaults to
`pipeline.roles.explorer`) and `--max-questions`. Note `--question` rather than
`--task`; do not force it through `_add_common_args` if that makes the flag
wrong. Writes an event log like the other workflows. Prints the final answer to
stdout and the per-sub-question breakdown to stderr, so the answer can be piped.

## Constraints

- The repo is opened read-only in spirit: only `READ_ONLY_TOOL_SCHEMAS` is ever
  passed, and no clone is made. Do not add a write tool "just in case".
- A sub-agent that fails must not sink the run: record the failure, keep the
  others, and let the synthesis stage report the gap.
- Explore must work when the repo is not a git repository at all — there is no
  isolation requirement here, so do not call `_require_git_repo`.

## Tests — `tests/test_pipeline_explore.py`

With a fake client in the style of `tests/test_pipeline_solo.py`: splitting
produces the sub-questions the model returned; a single-question split is
allowed; sub-agents run concurrently under the cap; one failing sub-agent still
yields a final answer that mentions the gap; the synthesis prompt actually
contains every sub-answer; read-only tool schemas contain no write/shell tool
(assert on the schema list — this is a safety property, not a style one).

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
