"""Deterministic sequential merge with a model escape hatch.

For each completed chunk (in topological order):
  1. Fetch the chunk's workspace HEAD into the real repo.
  2. Cherry-pick that commit.
  3. If clean → run verify. Pass or skipped → keep, next chunk.
  4. If conflict or verify fails → escalate to the model with *only* that
     chunk's context, tool access, and the FINISH_MERGE_TOOL.
  5. If escalation cannot resolve → abort/reset, record unmerged, continue.

The model is never shown every chunk's diff at once.  Git does the merge;
the model only resolves what git cannot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ._procutil import run_argv
from .client import OrchestratorClient
from .executor import topological_waves
from .models import AgentStatus, ChunkOutcome, Plan, RunConfig
from .tools import TOOL_SCHEMAS, execute_tool
from .verify import run_verify
from .workspace import (
    abort_cherry_pick,
    capture_base_commit,
    changed_files,
    cherry_pick_commit,
    conflicted_files,
    diff_against_base,
    fetch_from_workspace,
    reset_to_commit,
)

log = logging.getLogger("pipeline.merger")

MAX_MERGE_TURNS = 40

FINISH_MERGE_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_merge",
        "description": "Call this once you have resolved the current conflict or verify failure "
                       "and committed your fix. You must have committed before calling this tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What you did to resolve the issue, what files were changed, "
                                   "and any problems you encountered. Be specific.",
                },
            },
            "required": ["summary"],
        },
    },
}


def _escalation_system_prompt() -> str:
    return (
        "You are resolving a merge conflict or verify failure in the real repository. "
        "You have full tool access (read_file, write_file, edit_file, list_dir, run_shell, grep, glob) "
        "and can inspect the conflicted files, run tests, and commit your fix.\n\n"
        "You are given the description of ONE chunk that needs to be integrated. "
        "Do NOT modify other chunks' work — only resolve the problem blocking this chunk.\n\n"
        "Once you have committed your fix, call finish_merge with a summary of what you did."
    )


def _build_escalation_user_prompt(
    outcome: ChunkOutcome,
    conflict_info: str | None,
    verify_info: str | None,
    chunk_diff: str,
) -> str:
    parts = [f"## Chunk {outcome.chunk.id}: {outcome.chunk.title}"]
    parts.append(outcome.chunk.description)

    if chunk_diff.strip():
        parts.append(f"### Changes made by this chunk\n```diff\n{chunk_diff}\n```")
    else:
        parts.append("### Changes made by this chunk\n(no changes produced)")

    if conflict_info:
        parts.append(f"### Merge conflict\n\n{conflict_info}")
    if verify_info:
        parts.append(f"### Verify failure\n\n{verify_info}")

    parts.append(
        "Resolve the issue, commit your fix with a clear message, "
        "then call finish_merge."
    )
    return "\n\n".join(parts)


async def merge(
    client: OrchestratorClient,
    config: RunConfig,
    plan: Plan,
    outcomes: list[ChunkOutcome],
    base_commit: str,
) -> tuple[str | None, str]:
    """Deterministic sequential merge.

    Returns (new_commit_or_None, summary).
    """
    # --- topological order of chunk ids ---
    waves = topological_waves(plan.chunks)
    topo_ids: list[str] = []
    for wave in waves:
        topo_ids.extend(wave)  # wave is already list[str] of chunk ids

    # Build a map of id -> outcome.
    outcome_by_id: dict[str, ChunkOutcome] = {}
    for o in outcomes:
        outcome_by_id[o.chunk.id] = o

    # Per-chunk records for the final summary.
    chunk_records: list[dict] = []

    current_head = base_commit

    for cid in topo_ids:
        outcome = outcome_by_id.get(cid)
        if outcome is None:
            continue

        record: dict[str, object] = {
            "chunk_id": cid,
            "title": outcome.chunk.title,
        }

        if outcome.status != AgentStatus.COMPLETED:
            record["status"] = "skipped"
            record["reason"] = outcome.kill_reason or "chunk did not complete"
            chunk_records.append(record)
            continue

        if outcome.workspace is None:
            record["status"] = "skipped"
            record["reason"] = "no workspace available"
            chunk_records.append(record)
            continue

        # --- Step 1: Fetch the workspace HEAD into the real repo ---
        log.info("merging chunk %s: fetching workspace", cid)
        try:
            # Read the workspace's current branch name.
            _, branch_out, _ = await run_argv(
                ["git", "branch", "--show-current"], cwd=outcome.workspace,
            )
            branch = branch_out.strip()
            if not branch:
                raise RuntimeError("workspace has no current branch")
            await fetch_from_workspace(config.repo, outcome.workspace, branch)
        except RuntimeError as e:
            record["status"] = "failed"
            record["reason"] = f"fetch failed: {e}"
            chunk_records.append(record)
            continue

        # --- Step 2: Determine the commit to cherry-pick ---
        ws_head = await capture_base_commit(outcome.workspace)

        # Check if the workspace made any changes (diff against original base).
        ws_changed = await changed_files(outcome.workspace, base_commit)
        if not ws_changed:
            record["status"] = "no_changes"
            record["reason"] = "chunk produced no changes"
            chunk_records.append(record)
            continue

        # --- Step 3: Cherry-pick ---
        log.info("merging chunk %s: cherry-picking %s", cid, ws_head[:12])
        code, _, err = await cherry_pick_commit(config.repo, ws_head)

        if code == 0:
            # Clean cherry-pick.
            log.info("merging chunk %s: clean cherry-pick", cid)

            # Run verify if configured.
            verify_result = await run_verify(config.verify, config.repo)

            if verify_result.ok or verify_result.skipped:
                current_head = await capture_base_commit(config.repo)
                record["status"] = "merged_cleanly"
                if verify_result.skipped:
                    record["verify"] = "skipped"
                else:
                    record["verify"] = "passed"
                chunk_records.append(record)
                continue
            else:
                # Verify failed — escalation needed.
                log.info("merging chunk %s: verify failed after clean cherry-pick", cid)
                # Revert the cherry-pick commit.
                await reset_to_commit(config.repo, current_head)
                conflict_info = None
                verify_info = verify_result.output_tail
        else:
            # Cherry-pick conflicted.
            log.info("merging chunk %s: cherry-pick conflict: %s", cid, err.strip())
            cfiles = await conflicted_files(config.repo)
            conflict_info = f"Conflicted files: {cfiles}\nCherry-pick stderr: {err.strip()}"
            if cfiles:
                for fname in cfiles[:5]:
                    _, out, _ = await run_argv(
                        ["git", "diff", "--", fname], cwd=config.repo,
                    )
                    conflict_info += f"\n\n--- {fname} ---\n{out}"
            verify_info = None

        # --- Step 4: Escalation — model resolves this one chunk ---
        record["status"] = "escalated"
        resolved = await _resolve_escalation(
            client, config, outcome, conflict_info, verify_info,
            base_commit=base_commit,
        )

        if resolved:
            current_head = await capture_base_commit(config.repo)
            record["resolution"] = resolved
            record["status"] = "merged_after_escalation"
            chunk_records.append(record)
            continue
        else:
            # Escalation failed — abort and move on.
            await _abort_and_reset(config.repo, current_head)
            record["status"] = "unmerged"
            record["reason"] = ("could not resolve conflict/verify failure "
                                "even after model escalation")
            chunk_records.append(record)
            continue

    # --- Assemble final summary from per-chunk records ---
    summary_lines = []
    for rec in chunk_records:
        line = f"[{rec['status']}] {rec['chunk_id']}: {rec['title']}"
        if rec.get("reason"):
            line += f" — {rec['reason']}"
        if rec.get("resolution"):
            line += f"\n  Escalation: {rec['resolution']}"
        summary_lines.append(line)

    summary = "\n".join(summary_lines)

    # Final verification after all merges.
    new_head = await capture_base_commit(config.repo)
    merge_commit = new_head if new_head != base_commit else None
    if merge_commit is None:
        log.warning("merge pass completed without creating any new commits in %s", config.repo)

    return merge_commit, summary


async def _resolve_escalation(
    client: OrchestratorClient,
    config: RunConfig,
    outcome: ChunkOutcome,
    conflict_info: str | None,
    verify_info: str | None,
    base_commit: str = "",
) -> str | None:
    """Give the model a scoped prompt for one chunk's conflict/verify failure.

    Returns the model's finish_merge summary on success, or None if the model
    did not resolve (timed out, didn't call finish_merge, etc.).
    """
    # Build the diff of this chunk from its workspace — diff the workspace HEAD
    # against its parent, which isolates just this chunk's changes.
    diff_str = ""
    if outcome.workspace is not None:
        try:
            _, diff_out, _ = await run_argv(
                ["git", "diff", "HEAD~1", "HEAD"], cwd=outcome.workspace,
            )
            diff_str = diff_out
        except Exception:
            pass

    user_content = _build_escalation_user_prompt(
        outcome, conflict_info, verify_info, diff_str,
    )

    messages = [
        {"role": "system", "content": _escalation_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    tools = TOOL_SCHEMAS + [FINISH_MERGE_TOOL]
    summary = ""
    done = False

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
            messages.append({
                "role": "user",
                "content": "Continue working, or call finish_merge when you are done.",
            })
            continue

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
        if not summary:
            log.warning(
                "escalation for %s hit the turn limit without calling finish_merge",
                outcome.chunk.id,
            )
            return None

    return summary if done else None


async def _abort_and_reset(real_repo: Path, target_commit: str) -> None:
    """Safely abort a cherry-pick or reset to a known good commit."""
    # Check if a cherry-pick is in progress by looking for CHERRY_PICK_HEAD.
    cherry_pick_head = real_repo / ".git" / "CHERRY_PICK_HEAD"
    if cherry_pick_head.exists():
        await abort_cherry_pick(real_repo)
    # Reset to target commit.
    await reset_to_commit(real_repo, target_commit)
