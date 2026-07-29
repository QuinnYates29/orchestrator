"""ds4-full plans the work: explores the repo with read-only tools, then
must call submit_plan with the final structured plan. It decides chunk
count itself based on how independently the work splits - max_agents (if
set) is passed as a soft upper bound, never a forced target."""
from __future__ import annotations

import logging

from .client import OrchestratorClient
from .models import Plan, PlanChunk, RunConfig
from .tools import READ_ONLY_TOOL_SCHEMAS, execute_tool

log = logging.getLogger("pipeline.planner")

MAX_EXPLORATION_TURNS = 20  # safety bound, not a routine restriction - see _run_planning_loop

SUBMIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the final implementation plan, split into independent chunks that can "
                       "be implemented in parallel by separate agents with no coordination between them.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "short unique id, e.g. 'agent-1'"},
                            "title": {"type": "string"},
                            "description": {"type": "string",
                                            "description": "detailed description of what this chunk implements"},
                            "scope": {"type": "array", "items": {"type": "string"},
                                      "description": "files/directories this chunk is expected to touch"},
                            "context": {"type": "string",
                                        "description": "anything this agent needs to know that isn't "
                                                       "obvious from exploring the repo itself"},
                        },
                        "required": ["id", "title", "description"],
                    },
                },
                "shared_context": {
                    "type": "string",
                    "description": "Anything ALL agents need to know - overall architecture decisions, "
                                   "conventions, things every chunk must respect.",
                },
            },
            "required": ["chunks"],
        },
    },
}


def _system_prompt(config: RunConfig) -> str:
    cap_note = f" Produce no more than {config.max_agents} chunks." if config.max_agents else ""
    return (
        "You are planning an implementation task that will be split across multiple independent "
        "agents working in parallel, each in its own isolated clone of the repository with no "
        "visibility into what the others are doing. Explore the repository first using list_dir and "
        "read_file to understand its structure and conventions, then call submit_plan with your final "
        "plan.\n\n"
        "Decide the number of chunks yourself based on how independently the work actually splits - "
        "do not force an arbitrary number of chunks onto work that doesn't decompose that way. Chunks "
        "must be genuinely independent: if two chunks would need to edit the same file or depend on "
        "each other's output, that's a sign the split is wrong. If the task is small enough that it "
        "doesn't benefit from splitting at all, submit a single chunk."
        f"{cap_note}"
    )


async def create_plan(client: OrchestratorClient, config: RunConfig) -> Plan:
    messages = [
        {"role": "system", "content": _system_prompt(config)},
        {"role": "user", "content": config.task},
    ]
    tools = READ_ONLY_TOOL_SCHEMAS + [SUBMIT_PLAN_TOOL]

    for turn in range(MAX_EXPLORATION_TURNS):
        forced_final = turn == MAX_EXPLORATION_TURNS - 1
        tool_choice = (
            {"type": "function", "function": {"name": "submit_plan"}}
            if forced_final else "auto"
        )
        completion = await client.chat_once(
            config.model_for("planner"), messages, tools=tools,
            tool_choice=tool_choice, max_tokens=4096,
        )
        message = completion["choices"][0]["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # ds4-full stopped without submitting - nudge it back on track
            # rather than silently failing the whole run.
            messages.append({
                "role": "user",
                "content": "You must call submit_plan with your final plan, or continue exploring "
                           "with list_dir/read_file first if you're not ready yet.",
            })
            continue

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            if name == "submit_plan":
                return _parse_plan(fn["arguments"])
            import json
            try:
                arguments = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            except json.JSONDecodeError:
                arguments = {}
            result = await execute_tool(name, arguments, config.repo, run_shell_timeout_s=0)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError(f"ds4-full did not submit a plan within {MAX_EXPLORATION_TURNS} turns")


def _parse_plan(raw_arguments) -> Plan:
    import json
    args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    chunks_raw = args.get("chunks") or []
    if not chunks_raw:
        raise RuntimeError("ds4-full submitted a plan with zero chunks")
    chunks = [
        PlanChunk(
            id=c["id"], title=c.get("title", c["id"]), description=c.get("description", ""),
            scope=list(c.get("scope") or []), context=c.get("context", ""),
        )
        for c in chunks_raw
    ]
    log.info("plan: %d chunk(s): %s", len(chunks), ", ".join(c.id for c in chunks))
    return Plan(chunks=chunks, shared_context=args.get("shared_context", ""))
