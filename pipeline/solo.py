"""Single-agent driver: one model, one working directory, no clones, no
supervisor, no merge.

This is the degenerate case of the fan-out pipeline, and it exists for two
reasons. First, plenty of tasks don't decompose — forcing them through
plan/fan-out/merge just adds three model calls and a git dance around what one
agent was going to do sequentially anyway. Second, and more immediately: the
`ds4-full` launch profile declares `unload_before_load: [ornith, glm, gemma]`,
so ornith *cannot* be resident while ds4-full is. The fan-out pipeline needs
both at once and therefore cannot be driven by ds4-full alone. This can.

Unlike worker.py this doesn't run in a throwaway clone, so it edits whatever
directory it is pointed at. That is the intended behavior — the caller chooses
the workspace — but it is the reason there is no default for `repo`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .client import OrchestratorClient, OrchestratorError
from .config import PipelineCfg
from .events import EventLog, NullEventLog
from .tools import TOOL_SCHEMAS, ProcessTimeout, execute_tool

log = logging.getLogger("pipeline.solo")


@dataclass
class SoloResult:
    ok: bool
    turns: int
    final_message: str = ""
    stop_reason: str = ""          # "finished" | "max_turns" | "tool_timeout" | "error"
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    messages: list[dict] = field(default_factory=list)


EMPTY_TURN_NUDGE = (
    "Your last message was empty - no text and no tool call. Do not think "
    "further about the task in the abstract. Take the single next concrete "
    "action now: call a tool, or state your final answer as plain text."
)

SYSTEM_PROMPT = (
    "You are an autonomous coding agent working directly in a real repository. You have full tool "
    "access (read_file, write_file, list_dir, run_shell) in your working directory.\n\n"
    "Work through the task completely and correctly. Read before you write — inspect the existing "
    "code and match its conventions rather than inventing new ones. Run the project's tests to check "
    "your work as you go.\n\n"
    "When the task is fully done, stop calling tools and write a final summary of what you changed "
    "and what you verified. Do not keep calling tools once the work is finished. If you could not "
    "complete part of the task, say so explicitly in that summary rather than implying success."
)


async def run_solo(
    client: OrchestratorClient,
    *,
    model: str,
    repo: Path,
    task: str,
    max_turns: int = 60,
    run_shell_timeout_s: float = 900.0,
    max_tokens: int = 8192,
    max_empty_retries: int = 2,
    max_transport_retries: int = 3,
    transport_backoff_s: float = 5.0,
    events: EventLog | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> SoloResult:
    """Drive one agent to completion. Returns a SoloResult; never raises for
    ordinary agent failure (turn exhaustion, tool timeout) — those are reported
    as a non-ok result with a stop_reason, since the caller usually wants the
    partial work and the transcript either way."""
    events = events or NullEventLog()
    started = time.monotonic()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    result = SoloResult(ok=False, turns=0, messages=messages)
    empty_turns = 0

    events.emit("solo_start", model=model, repo=str(repo), max_turns=max_turns)

    while result.turns < max_turns:
        result.turns += 1
        turn_started = time.monotonic()
        tool_calls: list[dict] = []
        content = ""
        reasoning_chars = 0
        usage: dict | None = None
        finish_reason = ""
        aborted = False

        # A backend dying mid-stream raises out of httpx, not as an
        # OrchestratorError, and used to kill the run outright - losing every
        # turn of work because a llama-server went away for a few seconds.
        # Retry the turn: the conversation so far is still valid, so a fresh
        # request costs prefill and nothing else.
        for attempt in range(max_transport_retries + 1):
            tool_calls, content, reasoning_chars, usage, finish_reason = [], "", 0, None, ""
            try:
                async for event in client.chat_stream(
                    model, messages, tools=TOOL_SCHEMAS, tool_choice="auto", max_tokens=max_tokens,
                ):
                    if event.reasoning_delta:
                        reasoning_chars += len(event.reasoning_delta)
                    if event.content_delta:
                        content += event.content_delta
                    if event.usage:
                        usage = event.usage
                    if event.finish_reason:
                        finish_reason = event.finish_reason
                        if event.tool_calls:
                            tool_calls = event.tool_calls
                break
            except OrchestratorError as e:
                # The backend answered, it just said no. Retrying a 4xx/5xx
                # here would hammer a model that is already unhappy.
                log.error("turn %d: orchestrator returned %s: %s",
                          result.turns, e.status_code, e.body)
                events.emit("solo_error", turn=result.turns,
                            status_code=e.status_code, body=str(e.body))
                result.stop_reason = "error"
                result.final_message = f"orchestrator error {e.status_code}: {e.body}"
                aborted = True
                break
            except httpx.HTTPError as e:
                detail = f"{type(e).__name__}: {e}"
                if attempt < max_transport_retries:
                    delay = transport_backoff_s * (2 ** attempt)
                    log.warning("turn %d: transport failure (%s), retry %d/%d in %.0fs",
                                result.turns, detail, attempt + 1, max_transport_retries, delay)
                    events.emit("solo_transport_retry", turn=result.turns,
                                attempt=attempt + 1, detail=detail)
                    await asyncio.sleep(delay)
                    continue
                log.error("turn %d: transport failure after %d retries (%s)",
                          result.turns, max_transport_retries, detail)
                events.emit("solo_transport_failed", turn=result.turns, detail=detail)
                result.stop_reason = "transport_error"
                result.final_message = (
                    f"backend connection failed after {max_transport_retries} "
                    f"retries ({detail}); the work so far is on disk"
                )
                aborted = True
                break

        if aborted:
            break

        if usage:
            result.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            result.completion_tokens += int(usage.get("completion_tokens") or 0)
        events.emit_usage("solo", model, usage, turn=result.turns)

        assistant: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc["id"] or f"call_{result.turns}_{i}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for i, tc in enumerate(tool_calls)
            ]
        messages.append(assistant)

        log.info(
            "turn %d: %d tool call(s), %d content chars, %d reasoning chars, %s, %.0fs",
            result.turns, len(tool_calls), len(content), reasoning_chars,
            finish_reason or "no finish_reason", time.monotonic() - turn_started,
        )

        # A turn with no tool calls AND no content is not a finished task, it is
        # a model that reasoned itself into producing nothing. Ornith does this:
        # 34k characters of reasoning_content, then an empty message. Treating
        # it as success reports a phase as complete having written no files.
        if not tool_calls and not content.strip():
            messages.pop()  # an empty assistant turn teaches the next turn nothing
            empty_turns += 1
            log.warning(
                "turn %d: empty response (%d reasoning chars, finish_reason=%s), nudge %d/%d",
                result.turns, reasoning_chars, finish_reason or "none",
                empty_turns, max_empty_retries,
            )
            events.emit("solo_empty_turn", turn=result.turns, empty_turns=empty_turns,
                        finish_reason=finish_reason, reasoning_chars=reasoning_chars)
            if empty_turns > max_empty_retries:
                result.stop_reason = "empty_response"
                result.final_message = (
                    f"model returned {empty_turns} empty responses in a row "
                    f"(last: {reasoning_chars} reasoning chars, finish_reason="
                    f"{finish_reason or 'none'}); giving up rather than reporting success"
                )
                log.error(result.final_message)
                break
            messages.append({"role": "user", "content": EMPTY_TURN_NUDGE})
            continue

        empty_turns = 0

        if not tool_calls:
            result.ok = True
            result.stop_reason = "finished"
            result.final_message = content
            break

        timed_out = False
        for i, tc in enumerate(tool_calls):
            call_id = tc["id"] or f"call_{result.turns}_{i}"
            name = tc["name"]
            try:
                output = await execute_tool(name, tc["arguments"], repo, run_shell_timeout_s)
            except ProcessTimeout as e:
                # Nothing was generating while the subprocess hung, so no
                # stream-watching mechanism could have seen this. Terminal.
                log.error("turn %d: %s", result.turns, e)
                events.emit("solo_tool_timeout", turn=result.turns, tool=name, detail=str(e))
                result.stop_reason = "tool_timeout"
                result.final_message = str(e)
                timed_out = True
                break
            result.tool_calls += 1
            events.emit("tool_call", turn=result.turns, tool=name,
                        arguments=tc["arguments"], result_preview=output[:200])
            log.info("  %s -> %s", name, output.splitlines()[0][:120] if output else "(empty)")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

        if timed_out:
            break
    else:
        result.stop_reason = "max_turns"
        result.final_message = f"hit the {max_turns}-turn ceiling without finishing"
        log.warning(result.final_message)

    result.duration_s = time.monotonic() - started
    events.emit("solo_end", ok=result.ok, turns=result.turns, stop_reason=result.stop_reason,
                tool_calls=result.tool_calls, duration_s=round(result.duration_s, 1),
                prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
    return result


async def solo_session(
    *,
    model: str,
    repo: Path,
    task: str,
    orchestrator_url: str,
    admin_url: str,
    api_key: str | None,
    pipeline_cfg: PipelineCfg,
    events: EventLog | None = None,
    load_wait_s: float = 180.0,
    run_shell_timeout_s: float = 900.0,
    ensure_resident: bool = True,
) -> SoloResult:
    """Wraps run_solo with client setup and residency.

    Residency is guaranteed up front rather than on the first request, because
    a pinned-but-cold backend 503s immediately by design — the right failure
    mode mid-loop, the wrong one to discover after a session has started.
    `model` may name a launch profile (`ds4-full`), in which case the profile is
    requested for its owning model."""
    async with OrchestratorClient(orchestrator_url, admin_url, api_key) as client:
        if ensure_resident:
            await ensure_model_resident(client, model, load_wait_s)
        return await run_solo(
            client, model=model, repo=repo, task=task,
            max_turns=pipeline_cfg.limits.max_agent_turns,
            run_shell_timeout_s=run_shell_timeout_s,
            events=events,
        )


async def ensure_model_resident(client: OrchestratorClient, model: str,
                                 load_wait_s: float = 180.0) -> None:
    """Make `model` resident, whether it names a backend or a launch profile.

    Rather than guessing from the string's shape — `ds4-full` is a profile but
    `ds4-light` and a hypothetical backend called `foo-bar` are indistinguishable
    by dashes alone — this asks the orchestrator and lets it be the authority. A
    profile load is attempted first when the name looks like `<base>-<variant>`
    and a plain load is the fallback, so a wrong guess costs one rejected call
    instead of loading the wrong weights."""
    base = model.split("-", 1)[0] if "-" in model else model
    if base != model:
        try:
            log.info("ensuring %s is resident (profile of %s)", model, base)
            await client.admin_load(base, profile=model, wait_s=load_wait_s)
            return
        except OrchestratorError as e:
            log.info("%s is not a launch profile of %s (%s); loading it as a model",
                     model, base, e.status_code)
    log.info("ensuring %s is resident", model)
    await client.admin_load(model, wait_s=load_wait_s)
