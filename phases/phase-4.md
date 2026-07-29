# Phase 4 — Deterministic sequential merge

Working in the `orchestrator` repository. This phase rewrites
`pipeline/merger.py`, extends `pipeline/workspace.py`, and adds
`tests/test_pipeline_merger.py`.

## Why

The merge pass currently inlines every chunk's full diff into one prompt and
asks the model to hand-integrate everything and re-emit file contents. That is
the most expensive single generation in a run, and it asks a model to do the
one job git already does correctly and for free. Git should apply the patches;
the model should only be consulted where git genuinely cannot decide.

## Read first

- `pipeline/merger.py` — current prompt-driven merge, `FINISH_MERGE_TOOL`.
- `pipeline/workspace.py` — `capture_base_commit`, `diff_against_base`,
  `changed_files`, and the note on why local clones are used over worktrees.
- `pipeline/verify.py` (Phase 2) — `run_verify`, `VerifyResult`.
- `pipeline/executor.py` (Phase 3) — `topological_waves` for ordering.

## What to implement

Merge becomes a deterministic loop with a model escape hatch.

For each completed chunk, in topological order:

1. Add the chunk's clone as a git remote (or use `git fetch <path> <branch>`)
   and fetch its branch into the real repo.
2. Apply the chunk's commit range with `git cherry-pick`. Add helpers to
   `workspace.py` as needed (`fetch_from_workspace`, `cherry_pick_range`,
   `conflicted_files`, `abort_cherry_pick`).
3. If the cherry-pick applies cleanly, run `run_verify` in the real repo.
   - Verify passes or is skipped → keep the commit, move to the next chunk.
   - Verify fails → escalate (step 4).
4. **Escalation — only here does the model get called.** Give it a scoped
   prompt: this chunk's description, the conflicted files (with markers) or the
   verify output tail, and full tool access to the real repo. It resolves,
   commits, and calls `finish_merge`. It must NOT be given every other chunk's
   diff — the whole point is that the model sees only the problem it is being
   asked to solve.
5. If a chunk cannot be integrated even after escalation, `git cherry-pick
   --abort` (or reset to the last good commit), record the chunk as unmerged,
   and continue with the remaining chunks. One bad chunk must not sink the run.

Return `(merge_commit_or_None, summary)` exactly as now. The summary is
**assembled from the per-chunk records**, not generated prose: which chunks
merged cleanly, which needed escalation, which failed and why, and every chunk
that failed earlier in the run. Escalation summaries from the model are quoted
into it, not relied upon as the whole thing.

## Constraints

- Keep `FINISH_MERGE_TOOL` and the existing rule that failed chunks are never
  silently dropped from the summary — a human must be able to see what still
  needs redoing.
- Keep the existing behavior of verifying afterward that a commit actually
  happened and warning if not. Never report success without checking.
- `merge()`'s signature stays compatible with `cli.py`'s call.
- Chunks that produced no changes at all are normal, not errors: record and skip.
- Never `git push`, never touch remotes other than the local clone paths, and
  never rewrite existing history in the real repo (no rebase, no amend, no
  force). Only add commits on top.

## Tests — `tests/test_pipeline_merger.py`

Build real temporary git repos with `subprocess`; this logic is about git
behavior, so faking git would test nothing.

- Two chunks touching different files → both cherry-pick cleanly, model never
  called (assert the fake client received zero requests). This is the whole
  point of the phase.
- Two chunks touching the same line → conflict detected, model escalation
  invoked with only that chunk's context.
- Verify failing after a clean pick → escalation invoked.
- A chunk that cannot be integrated → aborted, recorded as unmerged, subsequent
  chunks still processed.
- Chunk with no changes → recorded, skipped, not an error.
- Summary names every failed and unmerged chunk.

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
