"""ds4-full reviews all chunks' diffs and integrates them directly into the
real target repo (not a clone) using the same full tool access the workers
had. It commits its own work itself via run_shell - this module doesn't
force a commit on its behalf, since that would risk committing something
ds4-full didn't consider final, or losing the specific message it would
have written. It only verifies afterward that a commit actually happened
and warns if not, consistent with never silently pretending success.
"""
from __future__ import annotations

import json
import logging

from .client import OrchestratorClient
from .models import AgentStatus, ChunkOutcome, Plan, RunConfig
from .tools import TOOL_SCHEMAS, execute_tool
from .workspace import capture_base_commit, diff_against_base

log = logging.getLogger("pipeline.merger")

MAX_MERGE_TURNS = 40

FINISH_MERGE_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_merge",
        "description": "Call this once you have finished reviewing, integrating, and committing the "
                       "agents' work into the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string",
                            "description": "What was integrated, from which chunks, and any issues "
                                           "encountered - including failed chunks, do not omit them."},
            },
            "required": ["summary"],
        },
    },
}


def _system_prompt() -> str:
    return (
        "You are reviewing and merging the work of several independent coding agents, each of which "
        "implemented one chunk of a larger plan in its own isolated clone of this repository. You now "
        "have full tool access (read_file, write_file, list_dir, run_shell) to the real repository "
        "itself, not a clone.\n\n"
        "For each successfully completed chunk you will be given its diff. Review each one, decide how "
        "to integrate it (chunks were planned to be independent, but may still need light "
        "reconciliation), and apply the final result directly using your tools. Run any tests or "
        "verification commands you judge appropriate before committing. When satisfied, commit the "
        "integrated result yourself via `git commit` (run_shell) with a clear message, then call "
        "finish_merge with a summary.\n\n"
        "Some chunks may have failed (killed for looping, timed out, or otherwise incomplete) - you "
        "will be told which. Do not silently drop them from your summary; a human needs to know what "
        "still needs to be redone."
    )


def _build_user_prompt(config: RunConfig, plan: Plan, outcomes: list[ChunkOutcome],
                        diffs: dict[str, str]) -> str:
    parts = [f"# Original task\n{config.task}"]
    if plan.shared_context:
        parts.append(f"# Shared plan context\n{plan.shared_context}")
    for outcome in outcomes:
        header = (f"## Chunk {outcome.chunk.id}: {outcome.chunk.title} "
                  f"(status: {outcome.status.value}, attempts: {outcome.attempts})")
        if outcome.status == AgentStatus.COMPLETED:
            diff = diffs.get(outcome.chunk.id, "")
            body = (f"{outcome.chunk.description}\n\nDiff:\n```diff\n{diff}\n```"
                    if diff.strip() else f"{outcome.chunk.description}\n\n(no changes produced)")
        else:
            body = f"{outcome.chunk.description}\n\nFAILED: {outcome.kill_reason or 'no reason recorded'}"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


async def merge(client: OrchestratorClient, config: RunConfig, plan: Plan,
                 outcomes: list[ChunkOutcome], base_commit: str) -> tuple[str | None, str]:
    """Returns (new_commit_or_None, summary)."""
    diffs = {}
    for outcome in outcomes:
        if outcome.status == AgentStatus.COMPLETED and outcome.workspace is not None:
            diffs[outcome.chunk.id] = await diff_against_base(outcome.workspace, base_commit)

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _build_user_prompt(config, plan, outcomes, diffs)},
    ]
    tools = TOOL_SCHEMAS + [FINISH_MERGE_TOOL]
    summary = ""

    for turn in range(MAX_MERGE_TURNS):
        forced_final = turn == MAX_MERGE_TURNS - 1
        tool_choice = (
            {"type": "function", "function": {"name": "finish_merge"}} if forced_final else "auto"
        )
        completion = await client.chat_once(
            config.model_for("merger"), messages, tools=tools,
            tool_choice=tool_choice, max_tokens=8192,
        )
        message = completion["choices"][0]["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user",
                             "content": "Continue working, or call finish_merge when you are done."})
            continue

        done = False
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            raw = fn["arguments"]
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            if name == "finish_merge":
                summary = arguments.get("summary", "")
                done = True
                break
            result = await execute_tool(name, arguments, config.repo, config.run_shell_timeout_s)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        if done:
            break
    else:
        summary = summary or f"ds4-full did not call finish_merge within {MAX_MERGE_TURNS} turns; state as left."
        log.warning("merge pass hit the turn limit without finishing")

    new_head = await capture_base_commit(config.repo)
    merge_commit = new_head if new_head != base_commit else None
    if merge_commit is None:
        log.warning("ds4-full finished the merge pass without creating a new commit in %s", config.repo)
    return merge_commit, summary
