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

---

## The end-to-end run, and the seven bugs only it could find

Every fix above came from watching phases run. A final `pipeline run` against a
throwaway repo — three converters plus a `convert()` that needs all of them —
found seven more, **none of which any phase's own test suite could reach**,
because each lived in the seam between two phases:

| Bug | Origin |
|---|---|
| `--max-tokens` parsed by `run`, ignored by planner (4096), worker and merger | 0/3/4 |
| `run_pipeline` took none of the four arguments `_cmd_resume` passes — `resume` was dead on arrival | 6 |
| `run` had no `--no-load`; residency hardcoded `"ds4"` and passed the role's model as a profile | **mine, phase 0** |
| `_init_integration_repo` defined and never called | 3 |
| chunk commits merged without being fetched into the integration repo | 3 |
| my fix for the above gated on `len(waves) > 1`, but integration is per chunk | **mine** |
| agents commit their own `__pycache__`; two chunks importing one module each commit a conflicting binary `.pyc` | 3/4 |
| `_integrate_chunk` never aborted a conflicted merge, so one conflict cascaded into fake ones for every later chunk | 3 |

The lesson is sharper than the test-file one: **six phases with 232 passing
tests still could not tell me whether `pipeline run` worked.** Phase 6 wrote 18
thorough tests for `state.py` and never invoked the CLI that consumes it.
Phase 3's executor tests used a fake client, so no git ever ran and two git
bugs shipped. Unit tests bounded by a phase's own file list cannot see the
integration surface, and that is where every remaining bug lived.

### Final result

Run `20260729-173614-6fa39c`, all five roles on ornith at 10.0.0.17:

```
[merged_after_escalation] chunk-1: Extend registry and add all converter modules
[merged_after_escalation] chunk-2: Add pytest tests for all new modules
[no_changes]              chunk-3: Run tests to verify everything passes
```

3/3 agents completed in isolated clones under supervision, and the merger's
escalation path resolved the `.pyc` collisions on its own rather than failing
the run. The merged library works: 100°C → 212°F, 0°C → 273.15 K, 1 m →
3.2808 ft, 1 kg → 2.2046 lb, and `convert(1, "m", "kg")` raises
`ValueError: Cannot convert between dimensions 'length' and 'mass'`. Its own
25 tests pass.

### Two defects in what the agents produced

Not harness bugs — quality limits of the models, worth recording:

- **`convert.py` never imports the converter modules.** Units register on
  import, so `from units.convert import convert` alone sees an empty registry.
  The 25 tests pass only because pytest imports every test module, and those
  import the converters. A test-order-dependent pass masking a real bug.
- **Unit names drifted from the task.** It asked for "metres, feet, inches";
  the agents registered `m`, `ft`, `in` for length and mass but full names
  (`celsius`, `kelvin`) for temperature. Internally consistent, inconsistent
  with the request.

## ~~Known-unfixable during this run~~ — this was wrong

I recorded here that "neither ds4-server nor ornith's llama-server reports
`usage` in streaming responses", and concluded the token-reduction goal could
not be measured without a different backend.

That was a wrong diagnosis, corrected in the follow-up pass below. Both
backends report usage perfectly well. The client never asked.

---

## Follow-up pass: three fixes from how the end-to-end run behaved

### 7. Token accounting: the flag was never sent

`pipeline/client.py`, `pipeline/events.py`, `pipeline/tokens.py` (new),
`pipeline/worker.py`, `pipeline/planner.py`, `pipeline/merger.py`,
`pipeline/explore.py`

An OpenAI-compatible server sends no usage on a stream unless the request sets
`stream_options: {"include_usage": true}`. `_build_body` never set it. Adding
it turns every `reported: false` into a real number — confirmed against both
backends directly and through the orchestrator proxy, on the streaming and
non-streaming paths.

I should have tested the backends before writing that section. The evidence I
had — an event log full of `reported: false` — was equally consistent with "the
backends can't" and "we didn't ask", and I picked the first without checking
the second, which was one `curl` away.

Two further gaps under the same heading:

- **`emit_usage` was only ever called from `solo.py` and `explore.py`.** The
  whole `run` path — planner, every worker turn, merger escalation — recorded
  nothing at all. Even with usage reported, `pipeline runs` would have shown an
  empty table for exactly the runs worth measuring.
- **`explore.py` read `completion["choices"][0]["usage"]`.** `usage` is a
  sibling of `choices`, not a member of one, so it was always `{}`. Its test
  fixture nested usage in the same wrong place, so the test and the bug agreed
  with each other — the same shape as fix 5, where a test's own scaffolding was
  what hid the defect.

`pipeline/tokens.py` is a fallback estimator for a backend that genuinely
reports nothing. It is a bytes-per-token ratio, not a tokenizer, and events
built from it carry `estimated: true` alongside `reported: false` so the three
states — counted, approximated, unknown — stay distinguishable rather than
collapsing into "zero".

### 8. The planner split by activity instead of by feature

`pipeline/planner.py`

The end-to-end plan was:

```
chunk-1: Extend registry and add all converter modules
chunk-2: Add pytest tests for all new modules      <- no depends_on
chunk-3: Run tests to verify everything passes     <- produced no changes
```

Three separate failures in one plan. `chunk-2` declared no dependency on
`chunk-1`, so it ran in the same wave and wrote tests against an API it could
not see — which is the direct cause of the unit-name drift recorded above.
`chunk-3` had no files to write and duplicated what the harness does after the
merge; it burned an agent to produce an empty diff. And both `chunk-1` and
`chunk-2` touched the registry, which is where the `.pyc` conflicts came from.

The prompt now names all three anti-patterns and says why each one fails.
Prompt-only, deliberately: structural validation that second-guesses a *split*
would reject plans a human would accept.

What is validated is only what cannot run — duplicate ids, dangling edges,
cycles. That used to surface as a `ValueError` out of `topological_waves`
*after* planning finished, taking the run down and discarding every exploration
turn that produced the plan. It is now handed back to the planner, which is the
only thing that can fix it and is still in the loop when it happens.

### 9. Hitting the turn ceiling threw away finished work

`pipeline/worker.py`, `pipeline/solo.py`

Phase 1 hit the 60-turn ceiling after 97 minutes, and everything it had done
was discarded because nothing was submitted. The failure is avoidable: the
agent had no idea it was running out.

Both loops now warn ten turns from the ceiling — land the current edit, run the
tests once, submit, and name whatever is unfinished rather than implying
success. A partial chunk that is submitted is worth more than a complete one
that is not.

---

## Testing the read-only fan-out, and the three bugs that took

`pipeline explore` had never been run. Testing it took three attempts, each
blocked by a different defect, and the last one is the worst thing in this
document.

### 10. `pipeline explore` had never once started

`pipeline/cli.py`

`_cmd_explore` builds a `RunConfig` through `_build_run_config`, which reads
`args.task`. The explore parser deliberately takes `--question` instead and so
never calls `_add_common_args` — meaning it defines neither `--task` nor
`--max-tokens`. The command died on `AttributeError: 'Namespace' object has no
attribute 'task'` before sending a single request.

Note what this means: the workflow had a full test file, 20-odd passing tests,
and no possible way to run. Every test called `_stage_split`, `_run_single_agent`
and `run_explore` directly; none went through the CLI. Same shape as the
`resume` bug — a phase tested its functions and not its entry point.

While fixing it: the explore stages hardcoded `max_tokens=4096`, the identical
defect fix 3 found in the planner, and unreachable from the CLI either way.
Plumbed through with a `--max-tokens` flag.

### 11. Forced `tool_choice` does not force ds4

`pipeline/explore.py`, `pipeline/planner.py`, `pipeline/merger.py`

Second attempt: `RuntimeError: explorer did not submit questions within 10
turns`. The split stage's last-resort safety net is a final turn with
`tool_choice` pinned to `submit_questions`. It does not work.

Reduced to a single call, ds4 answered a pinned `submit_questions` with a
`list_dir` call — and kept doing it when offered *only* the `submit_questions`
schema, inventing a `dir_path` argument that no tool in the request has. It was
not choosing from the tools it was given; it was following the system prompt,
which opens "First, use list_dir to get an overview."

Swapping that system message on the final turn for one that only describes
submitting makes the same model comply immediately. The fix is applied to all
three loops that rely on forced-final — split, planner, merger — because each
one had the same "explore first" / "you have full tool access" instruction
fighting its own escape hatch.

`_stage_split` additionally now degrades to a fan-out of one, using the original
question as the sole sub-question, rather than raising after ten turns of paid
exploration and returning nothing.

### 12. Every tool call in the fan-out had been failing

`pipeline/explore.py`

Third attempt ran end to end and produced a confident, fluent, entirely
unsourced answer — "due to persistent tool errors ... a general description of
typical merger implementations rather than being sourced from the specific
pipeline package's code."

The event log said why. **31 of 31 tool calls returned
`error: 'str' object has no attribute 'get'`.** Tool arguments arrive from the
wire as a JSON string; both explore stages passed that string straight into
`execute_tool`, which does `arguments.get(...)` on it. `read_file`, `list_dir`,
`grep` and `glob` had never returned a byte of the repository. Three agents
spent 10, 11 and 12 turns "exploring" a repo they could not read, and answered
from priors.

The planner and merger both decode this correctly, which is why the defect was
local to explore and why nothing else caught it. Now a shared `_tool_arguments`
helper handles string, dict, empty and malformed input.

This is the most useful failure in the project. The run *succeeded* — exit 0, an
answer produced, three sub-questions "answered", no exception anywhere. Nothing
short of reading the event log would have shown that the entire read-only
workflow was fabricating. An agent pipeline can fail this way silently, and
plausible output is not evidence that the tools ran.

### The run that finally worked

Run `20260729-212753-00f073`, all stages on ds4:

```
tool calls: 20, errored: 0        (was 31 of 31 failing)
agents:     finished in 3, 4, 10 turns   (was 10, 11, and hitting the ceiling)
usage:      28 events, 100% reported     (was 0 events emitted at all)
tokens:     199,303 prompt / 8,430 completion
```

The 6.2k-character answer is correct and file-attributed — `git clone --local`
over worktrees and why, `.git/info/exclude` for build artifacts, the executor's
per-wave `_integrate_chunk` fetch/merge/abort, the merger's two escalation
triggers and `MAX_MERGE_TURNS`. Spot-checked against the source; the claims and
the line references hold.

### Known gap, not fixed

`pipeline runs` reports "no runs found" for explore runs. They write an event
log but no `state.json`, and `find_runs` keys off the state file. The listing is
only documented for `run`/`resume`, so this is a scope question rather than a
defect — recording it rather than widening the change.
