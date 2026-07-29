# Phase 1 — Token-efficient tool surface

You are working in the `orchestrator` repository. This phase changes ONLY
`pipeline/tools.py` and adds `tests/test_pipeline_tools.py`. Do not modify any
other file.

## Why

Agents in this harness run on models that decode at seconds-per-token. The
single most wasteful thing the current tool set does is `write_file`, which
requires re-emitting a whole file to change three lines. Second is the absence
of any search tool, which forces agents to spend `run_shell` calls on `grep`.
Every change below exists to reduce the number of tokens an agent must generate.

## Read first

- `pipeline/tools.py` — the current four tools and `execute_tool` dispatch.
- `pipeline/_procutil.py` — `run_argv` / `run_shell` / `ProcessTimeout`.
- `tests/test_pipeline_worker.py` — the test style to match (plain pytest, no
  classes, no mocks framework, descriptive test names).

## What to implement

### 1. `edit_file(path, old_string, new_string, replace_all=False)`

Exact string replacement. Rules:

- If `old_string` is not found, return an error string saying so. Do not guess
  or fuzzy-match.
- If `old_string` appears more than once and `replace_all` is false, return an
  error naming how many occurrences were found. Ambiguity must not be resolved
  silently.
- With `replace_all=True`, replace every occurrence and report the count.
- If `old_string == new_string`, that is an error (a no-op edit is a mistake).
- Return a short confirmation, e.g. `edited path (1 replacement)`.

### 2. `grep(pattern, path=".", glob=None, max_results=50)`

- Prefer `rg` (ripgrep, available at `/usr/bin/rg`); fall back to `grep -rn` if
  `rg` is missing. Use `run_argv`, not shell interpolation — the pattern comes
  from a model and must not be word-split or interpreted by a shell.
- Output `file:line:text`, one per line, capped at `max_results` with a
  trailing `...[N more matches]` note when truncated.
- No matches is a normal result, not an error: return `no matches for <pattern>`.

### 3. `glob(pattern)`

- Match paths against the pattern relative to the working directory.
- Sort by modification time, newest first.
- Cap the result count and say so when truncated.

### 4. `read_file` gains `offset` and `limit`

- `offset` is a 1-based line number; `limit` is a line count.
- Output lines prefixed with their line number, `cat -n` style, so the agent can
  construct `edit_file` arguments and reason about locations without re-reading.
- Reading a whole file stays the default when neither is given.

### 5. `submit_work(summary, files_changed, verified, blocked)`

A terminal tool, declared in the schema list but NOT executed by
`execute_tool` — the worker loop interprets it. In this phase only add the
schema and export it as `SUBMIT_WORK_TOOL`; the worker wiring is Phase 2.

- `summary` (string, required) — what was done.
- `files_changed` (array of strings) — paths touched.
- `verified` (string) — what was actually run to check the work, or how it was
  checked. Must not be assumed; if nothing was run, say so.
- `blocked` (string) — anything that could not be completed. Required to be
  honest and explicit rather than omitted.

### 6. Output caps become configurable

`MAX_TOOL_OUTPUT_CHARS = 50_000` is far too large: a single tool result at that
size is roughly 12k input tokens, which crowds context and slows prefill on
every subsequent turn. Replace it with a default of 8000, taken from
`pipeline.config.LimitsCfg.tool_output_chars`, threaded through `execute_tool`
as an optional parameter with that default.

When truncating, keep the HEAD and the TAIL with a marker in the middle, e.g.
`\n...[truncated N chars]...\n`. Tails matter — a test failure summary and a
stack trace's final line are usually the point.

## Constraints

- Keep the existing style: `from __future__ import annotations`, async
  functions, small pure helpers, comments that explain *why* not *what*.
- `execute_tool`'s signature must stay backward compatible for existing callers
  (`worker.py`, `planner.py`, `merger.py`, `solo.py` all call it positionally
  as `execute_tool(name, arguments, cwd, timeout)`).
- `READ_ONLY_TOOL_SCHEMAS` must now include `grep` and `glob` alongside
  `read_file` and `list_dir`. It must NOT include `edit_file` or `write_file`.
- No path sandboxing. That is deliberate and pre-existing; do not add any.
- Every tool returns a string. Errors are returned as strings for the model to
  read, never raised — except `ProcessTimeout` from `run_shell`, which must keep
  propagating (the worker treats it as a distinct terminal condition).

## Tests to write in `tests/test_pipeline_tools.py`

Cover at minimum: edit_file success; not-found; ambiguous-without-replace_all;
replace_all counting; no-op rejection; grep finding and capping matches; grep
with no matches; glob ordering and capping; read_file offset/limit with line
numbers; head+tail truncation preserving both ends; unknown tool dispatch; and
that `SUBMIT_WORK_TOOL` is well-formed with its four properties.

Use `tmp_path`. Do not require network or a running orchestrator.

## Definition of done

`python -m pytest tests/ -q` passes with every pre-existing test still green.
Then stop and summarize what you changed and what you ran.
