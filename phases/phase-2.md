# Phase 2 — Verify → repair loop

Working in the `orchestrator` repository. This phase adds `pipeline/verify.py`,
modifies `pipeline/worker.py` and `pipeline/models.py`, and adds
`tests/test_pipeline_verify.py`. Do not touch other files.

## Why

Two problems. First, an agent is currently marked COMPLETED merely because it
stopped calling tools (`worker.py`) — nothing checks the work builds or passes.
Second, when an agent does fail, the only recovery is a blind retry from an
empty conversation, which re-pays all the exploration cost at
seconds-per-token. Feeding a failure back into the *same* conversation is
roughly an order of magnitude cheaper and usually enough.

## Read first

- `pipeline/worker.py` — the ReAct loop, the cancellation contract, `finally`
  block that finalizes the git commit.
- `pipeline/models.py` — `AgentStatus`, `AgentState`, `ChunkOutcome`.
- `pipeline/config.py` — `VerifyCfg`, especially the `configured` property.
- `pipeline/tools.py` — `SUBMIT_WORK_TOOL` from Phase 1.
- `pipeline/_procutil.py` — `run_shell`, `ProcessTimeout`.

## What to implement

### 1. `pipeline/verify.py`

```python
@dataclass
class VerifyResult:
    ok: bool
    skipped: bool          # no command configured
    output_tail: str
    duration_s: float
    exit_code: int | None
```

`async def run_verify(cfg: VerifyCfg, cwd: Path) -> VerifyResult`

- If `not cfg.configured`, return `VerifyResult(ok=False, skipped=True, ...)`.
  **`skipped` must never be reported as `ok`.** The absence of a check is not
  evidence the work is good. Callers decide what to do with a skip; this module
  must not pretend.
- Otherwise run the command with `run_shell` in `cwd`, bounded by
  `cfg.timeout_s`. A `ProcessTimeout` is a verify *failure*, not a crash —
  catch it and return `ok=False` with an explanatory tail.
- `output_tail` keeps the last ~2000 chars. Test output puts the failure
  summary at the end; the head is almost always noise.

### 2. Worker integration

In `run_agent`:

- Recognize `submit_work` in the tool-call loop. It is terminal: do not dispatch
  it to `execute_tool`. Record its arguments on the `AgentState` (add a
  `submitted: dict | None` field) and treat it as the agent declaring done.
- After a submit, run verification in the agent's workspace.
  - Verify passes, or is skipped → `AgentStatus.COMPLETED`.
  - Verify fails → append a **user** message to the same conversation
    containing the command, exit code, and output tail, plus an instruction to
    fix the cause and call `submit_work` again. Increment a repair counter and
    continue the loop.
  - When `repair_attempts` exceeds `cfg.max_repair_attempts`, stop with the new
    status `AgentStatus.VERIFY_FAILED`.
- A turn with no tool calls at all is no longer success. Nudge once ("call
  submit_work when you are done, or keep working"); if it happens again, end
  with `AgentStatus.FAILED`. Silence must not be mistaken for completion.
- Keep the existing `max_agent_turns` ceiling, the `ProcessTimeout` →
  `TIMED_OUT` path, the `asyncio.CancelledError` contract, and the `finally`
  block that always finalizes a commit. Do not regress any of them.

### 3. Model changes (`pipeline/models.py`)

- `AgentStatus.VERIFY_FAILED = "verify_failed"`.
- `AgentState`: add `submitted: dict | None = None` and `repair_attempts: int = 0`.
- `ChunkOutcome`: add `verify: VerifyResult | None = None` and
  `submitted: dict | None = None`.
- `RunConfig`: add a `verify` field holding the `VerifyCfg` so the worker can
  reach it. Default to a `VerifyCfg()` (unconfigured) so hand-built RunConfigs
  in tests keep working.

## Constraints

- The outer blind retry in `cli.py` stays as-is and still applies to `KILLED`
  and `TIMED_OUT`. Repair handles *wrong*; retry handles *stuck*. Do not merge
  the two, and do not add `VERIFY_FAILED` to the retry trigger — an agent that
  failed verification three times will not do better with a blank slate.
- Do not change `cli.py` in this phase.

## Tests

`tests/test_pipeline_verify.py`: skipped-is-not-ok; passing command; failing
command captures exit code and tail; timeout becomes a failure not an
exception; tail truncation keeps the end.

Extend `tests/test_pipeline_worker.py` with loop tests using a fake client in
the style of `tests/test_pipeline_solo.py`: submit_work with passing verify →
COMPLETED; failing verify → repair message appended and loop continues; repair
budget exhausted → VERIFY_FAILED; two consecutive no-tool-call turns → FAILED
after one nudge.

## Definition of done

`python -m pytest tests/ -q` fully green, then summarize.
