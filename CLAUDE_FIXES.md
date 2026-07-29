# What Claude fixed

Every implementation phase in this branch was written by a local model —
ds4-full (`deepseek-v4-flash`) or ornith 35B, on the GX10 and on a second
Spark at 10.0.0.17. This file records what *I* changed rather than what they
produced, so the delegated work can be judged on its own.

Phase-by-phase results are in the commit messages. Defects found mid-run were
deliberately left in place and fixed here at the end, except where a defect was
destroying runs faster than the phases could complete — those are marked
**[mid-run]** with the reason.

---

## Harness fixes

### 1. An empty model turn was reported as a finished task — **[mid-run]**

`pipeline/solo.py`, commit `145177d`

Phase 5 reported `ok=True, stop_reason="finished"` after writing no files at
all. The loop said:

```python
if not tool_calls:
    result.ok = True
    result.stop_reason = "finished"
```

Ornith's turn 8 emitted 34,003 characters of `reasoning_content` and then an
empty message — no content, no tool calls — and that branch called it done.
`finish_reason` was only consulted when tool calls were *already* present, so
the one signal that would have exposed it was discarded.

A turn with no tool calls **and** no non-whitespace content is now nudged
toward one concrete action, and after `max_empty_retries` fails with
`stop_reason="empty_response"`. The empty assistant message is popped rather
than appended, since it teaches the next turn nothing.

Fixed mid-run because every remaining phase would otherwise have been able to
report success having done nothing. Phase 5's rerun hit three empty turns and
recovered from all three.

### 2. A backend dropping mid-stream killed the whole run — **[mid-run]**

`pipeline/solo.py`, commit `cd02e05`

Phase 3 died at turn 23 with an unhandled `httpx.RemoteProtocolError` when the
remote llama-server went away. Only `OrchestratorError` was caught, so 23 turns
of work were lost to a few seconds of backend downtime.

Transport errors now retry the turn with exponential backoff (3 attempts, 5 s
base). The conversation so far is still valid, so a retry costs prefill and
nothing else; partial output from the dropped attempt is discarded.

`OrchestratorError` is deliberately **not** retried — a 4xx/5xx means the
backend answered, and hammering a model that is already unhappy is the wrong
move. That path is what kept ds4's OOM 503 correctly terminal.

This fired for real on Phase 4: a `RemoteProtocolError` then a `ConnectError`,
recovered on the third attempt.

### 3. No way to give a heavy reasoner headroom, and `--max-agent-turns` did nothing

`pipeline/cli.py`, `pipeline/solo.py`, commit `aa25aef`

Ornith repeatedly hit the hardcoded 8192-token per-turn cap mid-thought
(35,048 reasoning chars, `finish_reason=length`) and ended turns with no
answer. Added `--max-tokens`.

While wiring it I found `solo` parsed `--max-agent-turns` and then never passed
it to `solo_session` — the flag had been silently inert since Phase 0, with the
config value always winning.

**Calibration note:** more headroom is not free. At `--max-tokens 24576` a
single runaway turn on the remote took 16 minutes and saturated the server —
a 10-token probe timed out at 90 s. 12288-16384 is the usable range; 8192 is
right for ds4, which reasons in short bursts.

### 4. `merger.py` consumed `topological_waves` output as ids

`pipeline/merger.py`, commit after the Phase 4 merge

Phase 4 was written against Phase 3's *partial* executor, where
`topological_waves` returned chunk ids. Phase 3b later corrected it to return
`PlanChunk` objects, per the phase spec. Each phase was internally consistent;
the combination was not, and every merger test failed with `TypeError: cannot
use PlanChunk as a dict key`.

This is the only defect that no single agent could have caught — it existed
solely in the seam between two independently-developed branches.

### 5. 22 tests that had never once executed

`tests/conftest.py` (new)

`tests/test_pipeline_tools.py` was written with bare `async def test_*`. There
is no `pytest-asyncio` in the venv, so pytest collected all 22 and reported
them as failures without running a line. They stayed red for the entire
project and every later phase had to be told to ignore them.

Rather than mechanically rewriting 22 call sites, a `pytest_pyfunc_call` hook
drives coroutine test functions directly. Modules using the repo's existing
`_run(asyncio.run)` helper are unaffected.

Worth recording: once they ran, **all 26 passed**. Phase 1's `edit_file`,
`grep`, `glob` and truncation logic was correct the whole time — only the
tests were broken. I had assumed a defect was hiding behind them; it was not.

### 6. `run_shell`'s timeout did not actually stop anything

`pipeline/_procutil.py`

`test_timeout_becomes_failure_not_exception` runs `sleep 100` with
`timeout_s=0.01` and took the full **100 seconds**. The timeout fired on time,
but `proc.kill()` only kills the direct child — for a shell command that is
`/bin/sh`, and `sleep` survives it. The orphan inherits the stdout/stderr
pipes, so asyncio waits for an EOF that cannot arrive until it finishes
naturally.

So the "dead-man's-switch on a single run_shell call" advertised in the CLI
help was not one: a hung verify command or a wedged agent shell would have held
the pipeline for its full natural duration. Both `run_argv` and `run_shell` now
use `start_new_session=True` and kill the whole process group.

This was the single largest time sink in the project. Every agent paid 100
seconds on every `pytest` invocation, which is most of why Phase 2 took two
hours. Full suite: **100.5s → 0.98s**.

---

## Operational changes (not code)

- **Phase task files given an Environment section.** Phase 1 burned ~40 of its
  60 turns because `run_shell` returned exit 127 on `python` and it never found
  `/home/quinna/tools/orchestrator/.venv/bin/python`. It iterated against a
  pytest that could not run its own tests.
- **That section then caused its own bug.** Naming the venv path led Phase 2 to
  `cd` into `/home/quinna/tools/orchestrator` — the wrong repo — and lose seven
  turns looking for files it had written elsewhere. Rewritten for Phases 3-6 to
  state plainly which directory the agent is already in.
- **Later phase files told agents the 22 failures were not theirs.** Phases 5
  and 3b had both spent turns re-diagnosing them.
- **Local ornith relaunched at `--ctx-size 131072` instead of 262144.** The
  configured context consumed ~114 GB of a 121 GB box; halved, it uses 25 GB
  and leaves real headroom. This is a launch flag, not a config edit.

---

## Pattern worth keeping

Every phase that skipped its required test file shipped broken code:

| Phase | Wrote tests? | Outcome |
|---|---|---|
| 1 | not runnable | implementation was in fact correct — see fix 5 |
| 2 | yes | clean |
| 3 | **no** | `topological_waves` dead on arrival |
| 4 | yes | clean (7 merger tests against real git repos) |
| 5 | yes | clean |
| 6 | **no** | `NameError` on undefined `STATE_PATH`, regressed the suite |

Both continuation runs (3b, 6b) succeeded when the task file named the bug
explicitly and gave a one-line command to reproduce it. That is the cheapest
intervention found in this project.

## Known-unfixable during this run

Neither ds4-server nor ornith's llama-server reports `usage` in streaming
responses, so **every token figure in the event log is `reported: false`**.
`EventLog.emit_usage` records this honestly rather than as zero, but the
original plan's before/after token comparison has no data behind it. Measuring
the token-reduction goal needs a backend that reports usage, or client-side
tokenization.
