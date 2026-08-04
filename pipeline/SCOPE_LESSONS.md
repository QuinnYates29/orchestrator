# Chunk scoping lessons from a real run

Source run: `crypto-ratchet` resume `20260731-132547-4f30df`
(`~/tools/crypto-ratchet/.pipeline-runs/20260731-132547-4f30df/`), default
config (`roles.worker: ornith`, `limits.max_agent_turns: 60`). 3 of 6 chunks
merged; the other 2 died on the turn ceiling. This file records what the
workspaces on disk actually contained when they were killed, since the
run's own summary ("exceeded max_agent_turns without finishing") reads like
the work was incomplete, and for one of the two chunks that was false.

## Finding 1: a merge-conflict retry starts the turn counter at zero, with no credit for the first attempt

`chunk-4-market-strategy-risk`'s scope bundled five subsystems into one
chunk: `market/`, `indicators/`, `strategy/`, `risk/`, and
`event_trigger/` (via its shared `tests/` scope), each with its own tests.

- Attempt 1 (base `ff968e30ae`) **completed in 43 turns.**
- Its merge cherry-pick conflicted; escalation could not resolve it, so the
  whole chunk was discarded and rerun from a fresh clone.
- Attempt 2 (base `d5c5d3d1ef`) hit the `turn_budget_warning` at turn 50
  (11 left) and was killed at turn 60, status `failed`.

The run report says attempt 2 "exceeded max_agent_turns without
finishing." Running the actual workspace tells a different story:

```
$ cd chunk-4-market-strategy-risk-attempt2
$ python3.12 -m pytest tests/ -q
186 passed, 1 warning in 0.52s
```

All 186 tests passed. 4,589 lines across five subsystems were fully
implemented and verified in the workspace - the agent simply never reached
`submit_work` before the ceiling, because it was re-deriving from scratch
work it had already produced once. A chunk sized to *barely* fit the turn
budget on a clean run has no slack left for a forced retry, and a retry is
never free - `worker.py`'s own `TURN_WARNING_LEAD` comment already noted
"the useful 80% of the work is sitting in its workspace uncommitted" as a
failure mode; this run is the concrete case of it happening twice in a row
to the same chunk.

**Fix applied:** `planner.py`'s `_system_prompt` gained a fourth splitting
rule: size chunks so a from-scratch *re*-implementation fits the budget,
not just a lucky first pass. Prefer one subsystem per chunk over bundling
several, even inside the same wave.

## Finding 2: the scratch "integration" branch and the final merge are two independent merges - they can disagree about what a completed dependency contains

`chunk-5-decision-portfolio-llm`'s declared scope was `llm/`,
`brokers/paper.py`, `decision/`, `portfolio/`, and its four test files, with
`depends_on: [chunk-1-db-models, chunk-2-config-loader,
chunk-3-broker-interface]` - `config/` was never in its scope, and its
`config/config.py`/`config/env_loader.py` turned out to be **byte-identical**
to `chunk-2-config-loader-attempt1`'s own unmodified output. The worker never
touched them. (First draft of this document assumed it had - that was wrong;
recorded here so the correction isn't lost.)

What actually happened: `chunk-2-config-loader`'s work was merged twice, by
two different code paths that can disagree:

1. `executor.py`'s `_integrate_chunk` does a plain, non-escalated `git merge`
   into a scratch clone (`.pipeline-runs/<run>/integration/`) purely to give
   later-wave chunks something to branch from. This merge succeeded cleanly
   and kept chunk-2's **pre-escalation** design (pydantic `...`-required
   fields) - `chunk-4` and `chunk-5` both cloned from this integration repo
   and so both started with that design, unmodified.
2. `merger.py`'s final pass into the *real* repo cherry-picks the same
   chunk-2 commit and hit a genuine conflict there (the real repo had
   diverged further - see the "already applied" commits on `main` from an
   earlier partial run of this same project). That conflict got escalated to
   a model, which resolved it by keeping a different, better design (defaults
   + a uniform `validate_required()`) - commit `d26b49c` on `main`.

`d26b49c` is not reachable from the integration repo's history at all - it
was created by a separate merge process the integration repo never sees.
Result: chunk-5's workspace, correct on its own terms, disagreed with what
`main` would actually contain once the run finished:

```
$ cd chunk-5-decision-portfolio-llm-attempt2
$ python3.12 -m pytest tests/ -q
251 passed, 5 failed
$ cp ~/tools/crypto-ratchet/config/{config,env_loader}.py config/   # main's real, escalated version
$ python3.12 -m pytest tests/ -q
256 passed
```

All 5 failures were purely a stale-config artifact; every line of chunk-5's
actual assigned work (LLM client, paper broker, decision engine, portfolio
manager) was correct. Dependent-wave workers build against, and are
verified against, a copy of their dependencies that the run's own final
merge may go on to silently replace.

Separately, and smaller: `chunk-4-market-strategy-risk` (scope `market/`,
`indicators/`, `strategy/`, `risk/`, no `config/`) *did* make one real
out-of-scope edit - it added a `try/except ValidationError` wrapper to
`config/env_loader.py` that chunk-2's and chunk-5's copies don't have.
Harmless here (it doesn't touch the fields that caused chunk-5's failures),
but a genuine instance of a worker editing a file `scope` never named.

**Fixes applied:**
- `worker.py`'s `build_initial_messages` now tells the worker explicitly not
  to edit outside its declared scope, and that a file already produced by a
  completed dependency is settled - read and build on it, don't regenerate
  it. This addresses the smaller, real out-of-scope-edit case (chunk-4's
  `env_loader.py` change); it does not fix the integration/final-merge
  divergence, which is a harness gap, not a worker behavior problem - see
  "What this doesn't fix" below.

## Finding 3: merge escalation has unbounded shell access to the real repo and can write far outside the one conflict it was asked to fix

Source run: a later, separately-hand-planned run against this same repo
(`crypto-ratchet/.pipeline-runs/20260803-101911-c0343c`, 4 chunks: `runner`,
`backend`, `dashboard`, `deploy`). Only `backend` completed cleanly;
`runner` and `dashboard` were killed on the turn ceiling (same pattern as
Finding 1) and `deploy` was skipped. The final merge cherry-picked
`backend`'s commit and hit a conflict - untracked files
(`backend/__init__.py`, `backend/main.py`, `tests/test_backend.py`) already
present in the real repo - which escalated to `ds4-full` to resolve.

The model's own summary described exactly that fix (remove the conflicting
untracked files, recreate them from the chunk's diff) and nothing else.
But afterward the real repo's working tree also contained full,
uncommitted copies of `runner/` and `dashboard/` - packages belonging to
the two *killed* chunks, which were never part of this cherry-pick and
whose own commits were never even fetched into the real repo by the normal
merge path (`_integrate_chunk`/`merge()` only fetch a chunk's commit when
its status is `COMPLETED` - `runner` and `dashboard` were not). The
content wasn't a copy of either chunk's actual workspace output either -
`runner/loop.py` here used different variable names and a different fix
for the same underlying bug than the one in `runner`'s own attempt
workspace, meaning the escalation model generated this itself rather than
retrieving it from anywhere. Nothing was committed - `git diff` against
the merge commit shows only the described backend files - so git history
is clean, but the live working tree gained code for two unrelated,
incomplete chunks that no one asked for and nothing verified.

The escalation prompt (`merger.py`'s `_resolve_escalation`) gives the model
full `run_shell` access scoped to "resolve *this* conflict or verify
failure," with no constraint on what else it may touch while doing so, and
no diff review of its actions beyond checking that the specific conflict is
gone. A model with shell access, asked to fix one thing, decided on its own
initiative to also "help" with unrelated, unfinished work.

**Not fixed.** This needs either a diff scope-check after escalation
(reject/warn if changed files fall outside the conflicting chunk's own
scope) or restricting escalation's shell access to the specific conflicted
paths rather than the whole repo. Recorded here rather than patched blind,
since a chunk's own worker got a similar scope reminder in Finding 2 and
that alone isn't a fix for an agent with unrestricted shell access - the
enforcement gap that note names as unfixed is exactly what let this happen
at the merge stage too.

## What this doesn't fix

Four gaps remain, in rough order of how much damage they can do:

1. **Merge escalation's unscoped shell access (Finding 3) is not fixed.**
   The model that resolves a merge conflict can write anywhere in the real
   repo while doing so, and nothing checks its diff against the one
   conflict it was asked to resolve.
2. **The integration/final-merge divergence (Finding 2) is not fixed.**
   `_integrate_chunk`'s scratch merge and `merger.py`'s real-repo merge are
   still two independent code paths with no escalation on the integration
   side - a later-wave chunk can still be built and verified against a
   version of a dependency that the run's own final merge goes on to
   replace. A real fix would need the integration repo to either escalate
   its own conflicts the same way the final merge does, or be rebuilt from
   the real repo's current HEAD immediately before each new wave rather than
   from a cherry-picked replay of this run's own chunk commits.
3. Scope is still not *enforced* - no sandboxing of `write_file`/`edit_file`
   to the declared paths, just a stronger prompt asking nicely.
4. A merge-conflict retry (Finding 1) still reruns the whole chunk from
   scratch rather than just the conflicting file.

All four are reasonable next steps if problems recur after the prompt
changes above; they weren't done here because a prompt-level fix was enough
to address the smaller, real parts of the findings without adding new
mechanism to the harness itself. Finding 3 in particular has no prompt-level
mitigation applied yet - it was only just discovered.
