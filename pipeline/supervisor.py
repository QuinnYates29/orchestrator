"""ds4-light supervision: watches all running Ornith agents concurrently for
stuck-thinking loops and kills them (task.cancel()) on a "kill" verdict.

Two independent mechanisms, matching the agreed design:
  - A fast, pure-string-logic repetition check (no LLM, no cost) ticks
    frequently and can trigger an immediate out-of-band review the moment
    something looks suspicious, rather than waiting for the next periodic
    tick.
  - A slower periodic review runs underneath regardless, as the baseline
    safety net for loops that aren't literal text repetition (varied
    wording, circular reasoning) - the kind only judgment can catch.

ds4-light is slow (~3 tok/s per Quinn's own measurement), so its own verdict
is kept short and structured (a forced tool call, capped max_tokens) - the
bottleneck is its output generation, not how much it has to read, so a
compact answer is what actually keeps review latency down.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .client import OrchestratorClient
from .models import AgentState, AgentStatus, ReviewVerdict, RunConfig, ToolCallRecord

log = logging.getLogger("pipeline.supervisor")

REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "review_agent",
        "description": "Submit your verdict on whether this coding agent is stuck - looping in its "
                       "thinking (re-deriving the same reasoning repeatedly without progressing) or "
                       "hallucinating (inventing files, APIs, or facts inconsistent with what it has "
                       "actually observed) - and should be killed, or is making real progress and "
                       "should continue.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["continue", "kill"]},
                "reason": {"type": "string", "description": "One or two sentences. Keep it short."},
            },
            "required": ["verdict"],
        },
    },
}


def detect_repetition(text: str, *, window_chars: int = 400, min_repeats: int = 3) -> bool:
    """Fast, no-LLM pre-filter: does the tail of the reasoning buffer contain
    the same window_chars-sized chunk repeated min_repeats+ times? Natural
    (non-looping) reasoning essentially never repeats a several-hundred-char
    span verbatim multiple times, so this catches the common literal-repeat
    case cheaply. It never kills by itself - only triggers an immediate
    review; the subtler non-literal loop case is left to ds4-light's regular
    periodic judgment."""
    if len(text) < window_chars * min_repeats:
        return False
    tail = text[-(window_chars * (min_repeats + 2)):]
    probe = tail[-window_chars:]
    if not probe.strip():
        return False
    return tail.count(probe) >= min_repeats


def _format_tool_log(records: list[ToolCallRecord]) -> str:
    if not records:
        return "(no tool calls yet)"
    return "\n".join(f"- {r.name}({r.arguments}) -> {r.result_summary}" for r in records)


async def request_verdict(client: OrchestratorClient, state: AgentState,
                           model: str = "ds4-light") -> ReviewVerdict:
    system = (
        "You are reviewing a coding agent's live in-progress work to catch two specific failure "
        "modes: getting stuck in a repetitive thinking loop, or hallucinating facts/files/APIs "
        "inconsistent with what it has actually observed via its tool calls. Respond with a short, "
        "compact verdict - a sentence or two of reasoning is enough, do not write an essay."
    )
    user = (
        f"Assigned task: {state.chunk.title}\n{state.chunk.description}\n\n"
        f"Turn {state.turns}. Current in-progress reasoning for this turn "
        f"(may be partial - it may still be generating):\n{state.reasoning_buffer}\n\n"
        f"Recent tool calls:\n{_format_tool_log(state.tool_call_log[-10:])}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        completion = await client.chat_once(
            model, messages, tools=[REVIEW_TOOL],
            tool_choice={"type": "function", "function": {"name": "review_agent"}},
            max_tokens=200,
        )
    except Exception:
        log.exception("%s review request failed for %s - failing open", model, state.label)
        return ReviewVerdict(verdict="continue")

    message = completion["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        return ReviewVerdict(verdict="continue")
    raw_args = calls[0]["function"]["arguments"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return ReviewVerdict(verdict=args.get("verdict", "continue"), reason=args.get("reason", ""))


async def supervise(client: OrchestratorClient, config: RunConfig,
                     agents: dict[str, tuple[AgentState, asyncio.Task]]) -> None:
    """Runs until cancelled by the caller (once all agent tasks are done).
    `agents` is shared, mutable state owned by the caller - read here, and
    each AgentState/Task pair is only ever cancelled here, never removed."""
    last_reviewed: dict[str, float] = {}
    last_checked_len: dict[str, int] = {}

    supervisor_model = config.model_for("supervisor")

    async def review_and_maybe_kill(label: str, state: AgentState, task: asyncio.Task, prefix: str = ""):
        last_reviewed[label] = time.monotonic()
        verdict = await request_verdict(client, state, supervisor_model)
        if verdict.should_kill and state.status == AgentStatus.RUNNING:
            state.kill_reason = prefix + (
                verdict.reason or f"{supervisor_model} flagged this agent as stuck")
            state.status = AgentStatus.KILLED
            task.cancel()
            log.warning("killed %s: %s", label, state.kill_reason)

    last_fast_tick = time.monotonic()
    last_slow_tick = time.monotonic()
    tick = min(config.fast_repetition_tick_s, config.review_tick_s)

    try:
        while True:
            await asyncio.sleep(tick)
            now = time.monotonic()

            if now - last_fast_tick >= config.fast_repetition_tick_s:
                last_fast_tick = now
                for label, (state, task) in list(agents.items()):
                    if state.status != AgentStatus.RUNNING or task.done():
                        continue
                    buf = state.reasoning_buffer
                    if len(buf) == last_checked_len.get(label, 0):
                        continue  # no new reasoning since the last check
                    last_checked_len[label] = len(buf)
                    if detect_repetition(buf):
                        await review_and_maybe_kill(label, state, task, prefix="[repetition detector] ")

            if now - last_slow_tick >= config.review_tick_s:
                last_slow_tick = now
                for label, (state, task) in list(agents.items()):
                    if state.status != AgentStatus.RUNNING or task.done():
                        continue
                    if now - last_reviewed.get(label, 0.0) < config.review_tick_s * 0.5:
                        continue  # already just reviewed via the fast path
                    await review_and_maybe_kill(label, state, task)
    except asyncio.CancelledError:
        return
