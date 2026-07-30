"""ds4-full plans the work: explores the repo with read-only tools, then
must call submit_plan with the final structured plan. It decides chunk
count itself based on how independently the work splits - max_agents (if
set) is passed as a soft upper bound, never a forced target."""
from __future__ import annotations

import logging

from .client import OrchestratorClient
from .events import EventLog, NullEventLog
from .models import Plan, PlanChunk, RunConfig
from .tokens import estimate_usage
from .tools import READ_ONLY_TOOL_SCHEMAS, execute_tool

log = logging.getLogger("pipeline.planner")

MAX_EXPLORATION_TURNS = 20  # safety bound, not a routine restriction - see _run_planning_loop

SUBMIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the final implementation plan, split into chunks that can "
                       "be implemented in parallel within each wave. Chunks in the same wave run "
                       "simultaneously from the same base commit. Chunks that genuinely depend "
                       "on the output of another chunk must declare that ordering via the "
                       "depends_on field so they run in a later wave. Each chunk is a vertical "
                       "slice - implementation plus its own tests - never a separate "
                       "test-writing or verification step.",
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
                            "depends_on": {"type": "array", "items": {"type": "string"},
                                           "description": "Chunk ids that must complete before this chunk "
                                                          "can start. Use this only when one chunk genuinely "
                                                          "builds on another's output - do not merge "
                                                          "dependent chunks into one."},
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
        "You are planning an implementation task that will be split across multiple agents. "
        "Each agent runs in its own isolated clone of the repository. Agents in the same parallel "
        "wave do not see each other's output - they all branch from the same base commit. However, "
        "agents in a later wave DO branch from the accumulated result of their dependencies "
        "(the previous wave's completed work is integrated first).\n\n"
        "Explore the repository first using list_dir and read_file to understand its structure and "
        "conventions, then call submit_plan with your final plan.\n\n"
        "Decide the number of chunks yourself based on how independently the work actually splits - "
        "do not force an arbitrary number of chunks onto work that doesn't decompose that way. "
        "Make chunks as parallel as possible - two chunks that can run at the same time should be "
        "in the same wave, even if one conceptually builds on the other. But when one chunk genuinely "
        "needs the output of another (e.g. it imports a function the other defines), declare that "
        "ordering via the `depends_on` field listing the chunk ids it must wait for. Do NOT merge "
        "dependent chunks into a single chunk just to avoid declaring an edge - that throws away "
        "the parallelism you could still get. If the entire task is small enough that it does not "
        "benefit from splitting at all, submit a single chunk with no dependencies.\n\n"
        "Three rules about how NOT to split, each of which produces a plan that looks reasonable "
        "and then fails:\n\n"
        "1. Split by FEATURE, never by ACTIVITY. Every chunk must be a vertical slice that is "
        "complete on its own: its implementation, its tests, and any registration or wiring it "
        "needs. Never create one chunk that writes code and another that writes the tests for that "
        "code. The test-writing agent cannot see the code it is testing (same wave) or can only "
        "guess at it (later wave), so it invents an API that does not match, and neither agent ever "
        "runs a test that fails. The same applies to 'add types', 'add docstrings', 'add error "
        "handling' as separate chunks - those are activities, not features.\n\n"
        "2. Never create a chunk whose job is to verify, test, integrate, review, or 'tie together' "
        "the other chunks. The harness already merges every chunk and runs the project's "
        "verification command afterwards. Such a chunk has no files to write, produces an empty "
        "diff, and simply burns an agent.\n\n"
        "3. If two chunks would both edit the same file, they conflict. Either give one chunk sole "
        "ownership of that file, or make the second `depends_on` the first. A shared registry, "
        "`__init__.py`, or config file that every chunk must append to belongs to exactly one "
        "chunk - usually the first - and the others declare a dependency on it."
        f"{cap_note}"
    )


# The main prompt opens by telling the planner to explore the repo first, which
# on the final turn beats tool_choice outright: with submit_plan forced and no
# other tool even offered, ds4 still answered with a fabricated list_dir call.
# Forcing only works if the instruction to keep exploring is withdrawn too.
FINAL_TURN_SYSTEM_PROMPT = (
    "You are splitting an implementation task into chunks for parallel agents. Call "
    "submit_plan now with the best plan you can form from what you already know. Each chunk "
    "is a vertical slice - implementation plus its own tests - and any chunk that needs "
    "another's output must list it in depends_on. You have no other tools and no further turns."
)


async def create_plan(client: OrchestratorClient, config: RunConfig,
                      events: EventLog | None = None) -> Plan:
    events = events or NullEventLog()
    model = config.model_for("planner")
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
        turn_messages = messages
        turn_tools = tools
        if forced_final:
            turn_messages = [{"role": "system", "content": FINAL_TURN_SYSTEM_PROMPT}] + messages[1:]
            turn_tools = [SUBMIT_PLAN_TOOL]
        completion = await client.chat_once(
            model, turn_messages, tools=turn_tools,
            tool_choice=tool_choice, max_tokens=config.max_tokens,
        )
        message = completion["choices"][0]["message"]
        usage = completion.get("usage")
        events.emit_usage(
            "planner", model, usage,
            estimate=None if usage else estimate_usage(
                messages, message.get("content") or "",
            ),
            turn=turn + 1,
        )
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
                plan = _parse_plan(fn["arguments"])
                problem = _plan_structure_error(plan)
                if problem is None:
                    return plan
                # A structurally impossible plan used to take the whole run down
                # with a ValueError from topological_waves, throwing away every
                # exploration turn that produced it. The planner is the one thing
                # that can fix it, and it is still right here.
                log.warning("rejecting submitted plan: %s", problem)
                events.emit("plan_rejected", turn=turn + 1, reason=problem)
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": f"This plan cannot be executed: {problem}\n\n"
                               f"Fix that and call submit_plan again.",
                })
                break
            import json
            try:
                arguments = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            except json.JSONDecodeError:
                arguments = {}
            result = await execute_tool(name, arguments, config.repo, run_shell_timeout_s=0)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError(f"{model} did not submit a usable plan within {MAX_EXPLORATION_TURNS} turns")


def _plan_structure_error(plan: Plan) -> str | None:
    """Whatever makes this plan unexecutable, phrased for the planner to act on.

    Only checks what genuinely cannot run - a duplicate id, a dangling or
    self-referential edge, a cycle. Judgements about whether the *split* is good
    belong in the prompt, not here: rejecting a plan a human would accept costs
    a whole exploration round-trip."""
    from .executor import topological_waves  # local: executor imports models, not planner

    seen: set[str] = set()
    for chunk in plan.chunks:
        if chunk.id in seen:
            return f"chunk id {chunk.id!r} appears more than once; ids must be unique"
        seen.add(chunk.id)
    for chunk in plan.chunks:
        for dep in chunk.depends_on:
            if dep == chunk.id:
                return f"chunk {chunk.id!r} lists itself in depends_on"
            if dep not in seen:
                return (f"chunk {chunk.id!r} depends on {dep!r}, which is not a chunk in this "
                        f"plan (the ids you submitted are: {', '.join(sorted(seen))})")
    try:
        topological_waves(plan.chunks)
    except ValueError as e:
        return str(e)
    return None


def _parse_plan(raw_arguments) -> Plan:
    import json
    if isinstance(raw_arguments, str):
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError as e:
            # Almost always the generation cap: the model runs out of budget
            # partway through the submit_plan arguments and the JSON ends
            # mid-string. A bare JSONDecodeError points at the parser rather
            # than at the cause, so say what actually happened.
            raise RuntimeError(
                f"the planner's submit_plan arguments were not valid JSON ({e}). "
                f"The arguments were {len(raw_arguments)} characters and most "
                f"likely truncated by the per-turn generation cap - raise "
                f"--max-tokens. First 200 chars: {raw_arguments[:200]!r}"
            ) from e
    else:
        args = raw_arguments
    chunks_raw = args.get("chunks") or []
    if not chunks_raw:
        raise RuntimeError("the planner submitted a plan with zero chunks")
    chunks = [
        PlanChunk(
            id=c["id"], title=c.get("title", c["id"]), description=c.get("description", ""),
            scope=list(c.get("scope") or []), context=c.get("context", ""),
            depends_on=list(c.get("depends_on") or []),
        )
        for c in chunks_raw
    ]
    log.info("plan: %d chunk(s): %s", len(chunks), ", ".join(c.id for c in chunks))
    return Plan(chunks=chunks, shared_context=args.get("shared_context", ""))
