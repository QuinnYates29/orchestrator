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
from .events import EventLog, NullEventLog
from .models import AgentState, AgentStatus, Plan, PlanChunk, RunConfig, ToolCallRecord
from .tokens import estimate_usage
from .tools import SUBMIT_WORK_TOOL, TOOL_SCHEMAS, ProcessTimeout, execute_tool
from .verify import run_verify
from .workspace import finalize_agent_commit

log = logging.getLogger("pipeline.worker")

# How many turns before the ceiling the agent is told to start landing the work.
# An agent that hits max_agent_turns is killed mid-thought and its chunk is a
# total loss, even though the useful 80% of the work is sitting in its
# workspace uncommitted-as-finished. Ten turns is enough to write what it has,
# run the tests once, and call submit_work.
TURN_WARNING_LEAD = 10


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
        user += (
            "\n\nExpected scope (files/directories this chunk should touch): "
            + ", ".join(chunk.scope)
            + ". Do not edit files outside this scope - especially ones owned by a "
              "dependency chunk. If a file like config/config.py already exists because "
              "a dependency chunk completed, its design is settled: read it and build on "
              "it, don't rewrite it to match your own assumptions. Regenerating an "
              "already-correct shared file is how one chunk's own tests pass while it "
              "silently breaks another chunk's."
        )
    if chunk.context:
        user += "\n\nAdditional context:\n" + chunk.context
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_nudge_message() -> dict:
    return {
        "role": "user",
        "content": "You did not call any tool this turn. "
                    "Call `submit_work` when you are done, or keep working."
    }


def _build_turn_warning_message(turns_left: int) -> dict:
    return {
        "role": "user",
        "content": (
            f"Budget warning: you have {turns_left} turns left before this agent is "
            f"stopped, and work that has not been submitted is discarded. Stop "
            f"exploring and start landing what you have: finish the file you are "
            f"editing, run the tests once, and call `submit_work`. If part of the "
            f"chunk is not done, submit anyway and say plainly in the summary what "
            f"is missing - a partial chunk that is submitted is worth far more than "
            f"a complete one that is not."
        ),
    }


def _build_verify_failure_message(command: str, exit_code: int | None, output_tail: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Verification failed.\n\n"
            f"Command: {command}\n"
            f"Exit code: {exit_code}\n"
            f"Output (tail):\n{output_tail}\n\n"
            "Fix the cause of the failure and call `submit_work` again."
        )
    }


def _build_submit_verify_prompt(verify_cfg) -> str:
    """Build a system-prompt addendum explaining the verification contract."""
    cmd = verify_cfg.command or "(none configured)"
    return (
        f"\n\nVerification: after calling submit_work, the system runs "
        f"`{cmd}`. If it fails, you will be asked to fix the problem and "
        f"call submit_work again. You have up to {verify_cfg.max_repair_attempts} "
        f"repair attempts."
    )


async def run_agent(client: OrchestratorClient, config: RunConfig, plan: Plan, state: AgentState,
                    events: EventLog | None = None) -> None:
    """Runs the agent loop for `state` to a terminal status. Mutates `state`
    in place; does not return it (the caller already holds the reference)."""
    events = events or NullEventLog()
    model = config.model_for("worker")
    state.status = AgentStatus.RUNNING
    no_tool_streak = 0
    warned_about_turns = False
    # Warn with TURN_WARNING_LEAD turns to spare; on a budget too small for that
    # lead, warn on the first turn rather than not at all.
    warn_at_turn = max(1, config.max_agent_turns - TURN_WARNING_LEAD)

    try:
        # Append verification instructions to the system message
        # If verify is configured, tell the agent about it
        if config.verify.configured:
            state.messages[0]["content"] += _build_submit_verify_prompt(config.verify)

        while state.turns < config.max_agent_turns:
            state.turns += 1
            state.reasoning_buffer = ""  # this turn's live thinking - reset per turn, not cumulative

            if not warned_about_turns and state.turns >= warn_at_turn:
                warned_about_turns = True
                turns_left = config.max_agent_turns - state.turns + 1
                log.info("%s: %d turns left, telling it to land the work", state.label, turns_left)
                events.emit("turn_budget_warning", chunk=state.chunk.id,
                            turn=state.turns, turns_left=turns_left)
                state.messages.append(_build_turn_warning_message(turns_left))

            tool_calls: list[dict] = []
            assistant_content = ""
            usage: dict | None = None

            async for event in client.chat_stream(
                model, state.messages, tools=TOOL_SCHEMAS + [SUBMIT_WORK_TOOL],
                tool_choice="auto", max_tokens=config.max_tokens,
            ):
                if event.reasoning_delta:
                    state.reasoning_buffer += event.reasoning_delta
                if event.content_delta:
                    assistant_content += event.content_delta
                if event.usage:
                    usage = event.usage
                if event.finish_reason and event.tool_calls:
                    tool_calls = event.tool_calls

            events.emit_usage(
                "worker", model, usage,
                estimate=None if usage else estimate_usage(
                    state.messages, assistant_content, len(state.reasoning_buffer),
                ),
                chunk=state.chunk.id, turn=state.turns,
            )

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
                # No tool calls — nudge once, fail on second occurrence
                if no_tool_streak >= 1:
                    state.status = AgentStatus.FAILED
                    state.kill_reason = "agent stopped producing tool calls and did not call submit_work"
                    break
                no_tool_streak += 1
                state.messages.append(_build_nudge_message())
                continue

            # Reset the no-tool streak since the agent did call something
            no_tool_streak = 0

            timed_out = False
            for tc in tool_calls:
                call_id = tc["id"] or f"call_{state.turns}"

                # --- submit_work is terminal and handled here, not dispatched ---
                if tc["name"] == "submit_work":
                    state.submitted = tc["arguments"]
                    state.tool_call_log.append(ToolCallRecord(
                        id=call_id, name=tc["name"], arguments=tc["arguments"], result_summary="",
                    ))
                    # Run verification
                    verify_result = await run_verify(config.verify, state.workspace)
                    # If verification passes or is skipped, agent is done
                    if verify_result.ok or verify_result.skipped:
                        state.status = AgentStatus.COMPLETED
                        break
                    else:
                        # Verification failed — append a user message to repair
                        state.repair_attempts += 1
                        if state.repair_attempts > config.verify.max_repair_attempts:
                            state.status = AgentStatus.VERIFY_FAILED
                            state.kill_reason = (
                                f"verification failed after {config.verify.max_repair_attempts} "
                                f"repair attempts"
                            )
                            break
                        state.messages.append(_build_verify_failure_message(
                            config.verify.command, verify_result.exit_code, verify_result.output_tail,
                        ))
                        # Stop processing tool calls this turn; the agent will
                        # see the repair message and retry on the next turn.
                        break
                # --- end submit_work handling ---

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

            # If we broke due to submit_work completing or exhausting repairs
            if state.status in (AgentStatus.COMPLETED, AgentStatus.VERIFY_FAILED):
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
