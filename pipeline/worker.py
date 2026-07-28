"""The Ornith worker agent loop: a streaming ReAct tool-calling loop for one
plan chunk, running in its own isolated git clone. Streams (not a single
blocking completion) specifically so the supervisor can observe
reasoning_content while a turn is still generating - a stuck-thinking loop
may never produce a tool call at all, so waiting for turn completion would
never catch it.

Cancellation contract: the supervisor kills an agent via `task.cancel()` on
the asyncio.Task wrapping run_agent(). AgentState is mutated in place and is
the source of truth for the caller - not the task's return value - since a
cancelled task's return value is not directly usable the normal way.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .client import OrchestratorClient
from .models import AgentState, AgentStatus, Plan, PlanChunk, RunConfig, ToolCallRecord
from .tools import TOOL_SCHEMAS, ProcessTimeout, execute_tool
from .workspace import finalize_agent_commit

log = logging.getLogger("pipeline.worker")


def build_initial_messages(chunk: PlanChunk, plan: Plan, retry_context: str = "") -> list[dict]:
    system = (
        "You are an autonomous coding agent implementing one independent chunk of a larger plan. "
        "You have full tool access (read_file, write_file, list_dir, run_shell) in your own isolated "
        "git clone of the repository - this clone is yours alone, work freely within it. Implement "
        "your assigned chunk completely and correctly. When you are done, stop calling tools and give "
        "a final summary of what you did - do not keep calling tools once the work is finished."
    )
    if plan.shared_context:
        system += "\n\nShared context for all agents working on this plan:\n" + plan.shared_context
    if retry_context:
        system += "\n\n" + retry_context

    user = f"# {chunk.title}\n\n{chunk.description}"
    if chunk.scope:
        user += "\n\nExpected scope (files/directories this chunk should touch): " + ", ".join(chunk.scope)
    if chunk.context:
        user += "\n\nAdditional context:\n" + chunk.context
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def run_agent(client: OrchestratorClient, config: RunConfig, plan: Plan, state: AgentState) -> None:
    """Runs the agent loop for `state` to a terminal status. Mutates `state`
    in place; does not return it (the caller already holds the reference)."""
    state.status = AgentStatus.RUNNING
    try:
        while state.turns < config.max_agent_turns:
            state.turns += 1
            state.reasoning_buffer = ""  # this turn's live thinking - reset per turn, not cumulative

            tool_calls: list[dict] = []
            assistant_content = ""

            async for event in client.chat_stream(
                "ornith", state.messages, tools=TOOL_SCHEMAS, tool_choice="auto", max_tokens=8192,
            ):
                if event.reasoning_delta:
                    state.reasoning_buffer += event.reasoning_delta
                if event.content_delta:
                    assistant_content += event.content_delta
                if event.finish_reason and event.tool_calls:
                    tool_calls = event.tool_calls

            assistant_msg: dict = {"role": "assistant", "content": assistant_content or None}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"] or f"call_{state.turns}_{i}",
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            state.messages.append(assistant_msg)

            if not tool_calls:
                state.status = AgentStatus.COMPLETED
                break

            timed_out = False
            for tc in tool_calls:
                call_id = tc["id"] or f"call_{state.turns}"
                try:
                    result = await execute_tool(
                        tc["name"], tc["arguments"], state.workspace, config.run_shell_timeout_s,
                    )
                except ProcessTimeout as e:
                    # Dead-man's-switch: nothing was generating while this was
                    # hung, so the reasoning-stream supervisor could never
                    # have caught it. Distinct terminal status from a kill.
                    state.status = AgentStatus.TIMED_OUT
                    state.kill_reason = str(e)
                    timed_out = True
                    break
                state.tool_call_log.append(ToolCallRecord(
                    id=call_id, name=tc["name"], arguments=tc["arguments"], result_summary=result[:200],
                ))
                state.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

            if timed_out:
                break
        else:
            state.status = AgentStatus.FAILED
            state.kill_reason = f"exceeded max_agent_turns ({config.max_agent_turns}) without finishing"

    except asyncio.CancelledError:
        # The supervisor sets kill_reason/status before calling cancel();
        # this is just a safety default if status wasn't already set.
        if state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.KILLED
        raise
    finally:
        state.finished_at = time.time()
        if state.workspace is not None:
            try:
                suffix = f": {state.kill_reason}" if state.kill_reason else ""
                await finalize_agent_commit(state.workspace, f"[{state.label}] {state.status.value}{suffix}")
            except Exception:
                log.exception("failed to finalize commit for %s", state.label)
