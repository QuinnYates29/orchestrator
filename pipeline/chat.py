"""Interactive chat: a human talks to one model turn-by-turn, and the model
drives this repo's fan-out pipeline itself via run_shell - `pipeline.cli
run/resume/runs` invoked through the same interpreter this chat is itself
running under (sys.executable - the only one with `pipeline` importable),
same commands a human would type, just as ordinary shell commands. No new
tool schema: run_shell already covers it, and reading events.jsonl/
state.json for status is read_file/grep, also already available.

`run`/`resume` can take a long time (minutes to hours), so the system prompt
tells the model to background them (nohup ... &) rather than block the turn
on run_shell's own timeout.

Unlike solo.py's run_solo (autonomous until one task finishes, with a turn
ceiling), this loop never decides the conversation is over - the human ends
it (Ctrl-D or "exit"). There is also no plan-ahead workspace here: tool calls
run directly against `repo`, same as solo.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from .client import OrchestratorClient, OrchestratorError
from .solo import ensure_model_resident
from .tools import TOOL_SCHEMAS, ProcessTimeout, execute_tool

log = logging.getLogger("pipeline.chat")

SYSTEM_PROMPT_TEMPLATE = """You are a supervisor for this repo's multi-agent build pipeline (the \
`pipeline` CLI, `run|resume|runs`), which you drive via run_shell. You do not write application \
code yourself - you launch/resume/inspect pipeline runs and report status in plain language, based \
only on what commands and files actually show.

Your working directory is {repo}.

IMPORTANT: invoke the CLI as `{python_bin} -m pipeline.cli ...` exactly - that interpreter has the \
`pipeline` package installed (an editable install of ~/tools/orchestrator). A bare `python` or \
`python3` is either not on PATH or is a different, unrelated interpreter without `pipeline` \
installed at all - do not try those, they will not work no matter what directory you run them from.

Commands available to you (always run from {repo}):

- `{python_bin} -m pipeline.cli run --repo {repo} --task-file <FILE> [--agents-max N] --config \
{config}` starts a NEW fan-out run (plans from scratch, fans out N agents, merges). This can take \
a long time - ALWAYS background it, detached, so it survives past this tool call returning:
    nohup {python_bin} -m pipeline.cli run --repo {repo} --task-file RATCHET_CREATE.md --config \
{config} > .pipeline-runs/last-run.log 2>&1 &
    echo "launched pid $!"
  Never run it in the foreground - run_shell has its own timeout and will report a false failure \
on a run that was actually still in progress.

- `{python_bin} -m pipeline.cli resume <run_id> --repo {repo} --config {config}` resumes a \
previous run: re-runs only chunks that did not succeed (failed, or skipped because a dependency \
failed), reusing the stored plan and everything already merged. Prefer this over a fresh `run` \
whenever a prior run id exists and at least one chunk merged - a fresh run replans and redoes work \
that already landed. Background this the same way as `run`.

- `{python_bin} -m pipeline.cli runs --repo {repo}` lists past runs: chunk counts by status, \
whether a merge landed, token totals. Use this to find the most recent run_id before deciding run \
vs resume.

- To check whether a backgrounded run is still going, or how far it got, read \
`{repo}/.pipeline-runs/<run_id>/events.jsonl` and `state.json` (read_file, grep, list_dir) rather \
than a blocking shell command - a still-running process won't return from something like `wait` or \
`tail -f`. `tail -n 40 <path>` (a bounded, non-following read) is fine.

Rules:
- Always report what a real command or file actually shows - never guess or assume success.
- Before launching a brand-new `run`, check `runs` first. If a resumable run already exists, ask \
the human whether they want `resume` or really want to replan from scratch, rather than deciding \
for them.
- If a background run appears to already be in progress (process still active, events.jsonl still \
growing), say so rather than launching a second one on top of it.
- This box can only hold one of {{ds4-full, ornith}} resident at a time (they share the fan-out \
pipeline's own model rotation) - if you feel unresponsive for a stretch, it is very likely because \
a run you launched is itself in its planner or merger phase using the model you're pinned to; it \
will free up again once that phase ends."""


async def chat_session(
    *,
    model: str,
    repo: Path,
    orchestrator_url: str,
    admin_url: str,
    api_key: str | None,
    config_display: str,
    load_wait_s: float = 180.0,
    run_shell_timeout_s: float = 900.0,
    max_tokens: int = 8192,
    ensure_resident: bool = True,
) -> None:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        repo=repo, config=config_display, python_bin=sys.executable,
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    loop = asyncio.get_event_loop()

    async with OrchestratorClient(orchestrator_url, admin_url, api_key) as client:
        if ensure_resident:
            await ensure_model_resident(client, model, load_wait_s)

        print(f"chatting with {model} - working directory {repo}. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "you> ")
            except EOFError:
                print()
                return
            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit"):
                return
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})

            # Chain tool calls until the model produces a plain-text reply.
            # No turn ceiling here - the human decides when this is over, not
            # a budget-exhaustion guard like the autonomous worker/solo loops.
            while True:
                content = ""
                tool_calls: list[dict] = []
                try:
                    async for event in client.chat_stream(
                        model, messages, tools=TOOL_SCHEMAS, tool_choice="auto",
                        max_tokens=max_tokens,
                    ):
                        if event.content_delta:
                            print(event.content_delta, end="", flush=True)
                            content += event.content_delta
                        if event.finish_reason and event.tool_calls:
                            tool_calls = event.tool_calls
                except OrchestratorError as e:
                    print(f"\n[orchestrator error {e.status_code}: {e.body}]")
                    messages.pop()  # drop the user turn so it can be retried instead of poisoning history
                    break

                assistant: dict = {"role": "assistant", "content": content or None}
                if tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": tc["id"] or f"call_{i}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for i, tc in enumerate(tool_calls)
                    ]
                messages.append(assistant)
                print()

                if not tool_calls:
                    break

                for i, tc in enumerate(tool_calls):
                    call_id = tc["id"] or f"call_{i}"
                    name = tc["name"]
                    print(f"  [{name}({tc['arguments']})]")
                    try:
                        output = await execute_tool(name, tc["arguments"], repo, run_shell_timeout_s)
                    except ProcessTimeout as e:
                        output = str(e)
                    preview = output.splitlines()[0][:200] if output else "(empty)"
                    print(f"  -> {preview}")
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
