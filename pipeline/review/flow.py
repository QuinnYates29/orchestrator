"""The `pipeline review` workflow: read a diff, report on it, change nothing.

Same idiom as merger.py, where git does the merge and the model only resolves
what git cannot. Here the deterministic checks do what a regex can do - for
free and instantly - and a model is spent only on what a regex cannot see:
whether the logic is right, whether an error path is missing, whether the tests
cover the change.

**This workflow never modifies the repository.** Agents are given
READ_ONLY_TOOL_SCHEMAS, so `write_file`, `edit_file` and `run_shell` are not
merely discouraged, they are absent from the request. That is a structural
guarantee rather than an instruction a model can talk itself out of.

Three stages:

  1. static     - checks.run_checks over the parsed diff. No model calls.
  2. per-file   - one read-only agent per changed file, concurrently, each
                  told what the static pass already found so it spends its
                  turns on what that pass cannot reach.
  3. summarise  - one call over the merged findings for a short overall read.

Findings are merged programmatically, not by a model: they are already
structured, and asking a model to reformat a list it can silently drop items
from would be strictly worse than sorting it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..client import OrchestratorClient, OrchestratorError
from ..config import PipelineCfg
from ..events import EventLog, NullEventLog
from ..tokens import estimate_usage
from ..tools import READ_ONLY_TOOL_SCHEMAS, execute_tool
from .checks import DEFAULT_MAX_LINE_LENGTH, run_checks
from .diff import collect_diff, parse_diff
from .models import FileDiff, Finding, Review, Severity

log = logging.getLogger("pipeline.review")

MAX_AGENT_TURNS = 12
MAX_SUMMARY_TOKENS = 1024


@dataclass
class FileReview:
    path: str
    findings: list[Finding] = field(default_factory=list)
    turns: int = 0
    stop_reason: str = ""
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ReviewRun:
    review: Review
    file_reviews: list[FileReview] = field(default_factory=list)
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    static_findings: int = 0
    model_findings: int = 0

    @property
    def unreviewed(self) -> list[FileReview]:
        """Files the model pass could not report on. Named explicitly because a
        review that silently skipped half the diff must not read as clean."""
        return [f for f in self.file_reviews if f.error or f.stop_reason == "max_turns"]


SUBMIT_FINDINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_findings",
        "description": "Report what you found in this file. Call this exactly once, "
                       "when you are done reading. Submit an empty list if the change "
                       "looks fine - that is a valid and useful answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer",
                                     "description": "Line number in the NEW file that the "
                                                    "finding refers to."},
                            "severity": {"type": "string", "enum": ["info", "warning", "error"],
                                         "description": "error = this is a bug or a security "
                                                        "problem. warning = likely wrong or "
                                                        "risky. info = worth knowing."},
                            "category": {"type": "string",
                                         "description": "Short kebab-case label, e.g. "
                                                        "'logic-error', 'missing-error-handling', "
                                                        "'test-gap', 'api-misuse'."},
                            "message": {"type": "string",
                                        "description": "One or two sentences: what is wrong and "
                                                       "why it matters. Be specific about this "
                                                       "code, not generic advice."},
                        },
                        "required": ["line", "severity", "message"],
                    },
                },
            },
            "required": ["findings"],
        },
    },
}

SYSTEM_PROMPT_FILE = (
    "You are reviewing one file's changes in a git diff. You can read the repository "
    "with read_file, list_dir, grep and glob to understand context around the change.\n\n"
    "You are a reviewer, not an author. You cannot modify anything - you have no tools "
    "that write - and you must not suggest that you have made or will make a change. "
    "Report what you find and stop.\n\n"
    "Look for what a static checker cannot see:\n"
    "  - logic that is wrong, or right only for the cases the author had in mind\n"
    "  - error paths that are missing, swallowed, or wrong\n"
    "  - a function's behaviour disagreeing with its name, docstring or callers\n"
    "  - resource, concurrency or ordering problems\n"
    "  - a change whose tests do not actually exercise it\n\n"
    "Do not report style, formatting, or anything already listed as found by the "
    "static pass. Do not invent problems to look thorough: an empty findings list is "
    "a good answer for a clean change, and a reviewer who cries wolf gets ignored.\n\n"
    "Only report on the lines this diff adds or changes. When you are done, call "
    "submit_findings."
)

SYSTEM_PROMPT_SUMMARY = (
    "You are summarising a completed code review for the person who has to act on it. "
    "Write 2-4 sentences of plain prose: what this change does, and whether anything in "
    "the findings should block it. Do not list the findings again - they are printed "
    "directly above your summary. If nothing is concerning, say so plainly."
)

_SEVERITY_BY_NAME = {s.value: s for s in Severity}
_FILE_HEADER_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$")


def split_diff_sections(text: str) -> dict[str, str]:
    """Raw diff text per file path.

    parse_diff keeps only added lines, which is what the checks want but not
    what a reviewer needs - removed lines and surrounding context are most of
    how you tell whether a change is right.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        match = _FILE_HEADER_RE.match(line)
        if match:
            if current is not None:
                sections[current] = "\n".join(buf)
            current = match.group("b")
            buf = [line]
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    return sections


def _build_user_prompt(path: str, diff_text: str, known: list[Finding],
                       max_diff_chars: int) -> str:
    parts = [f"## File under review: `{path}`"]
    body = diff_text
    if len(body) > max_diff_chars:
        body = body[:max_diff_chars] + "\n… (diff truncated)"
    parts.append(f"### The change\n```diff\n{body}\n```")
    if known:
        listed = "\n".join(f"- line {f.line}: [{f.check}] {f.message}" for f in known)
        parts.append("### Already found by the static pass — do not repeat these\n" + listed)
    else:
        parts.append("### The static pass found nothing in this file.")
    parts.append("Read whatever surrounding context you need, then call submit_findings.")
    return "\n\n".join(parts)


def _parse_submitted(path: str, raw) -> list[Finding]:
    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    out: list[Finding] = []
    for item in args.get("findings") or []:
        if not isinstance(item, dict) or not item.get("message"):
            continue
        try:
            line = int(item.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        category = str(item.get("category") or "review").strip() or "review"
        out.append(Finding(
            path=path,
            line=max(0, line),
            # An unrecognised severity becomes a warning rather than being
            # dropped: losing a real finding to a typo'd enum is the worse error.
            severity=_SEVERITY_BY_NAME.get(str(item.get("severity", "")).lower(), Severity.WARNING),
            check=f"review/{category}",
            message=str(item["message"]).strip(),
            source="model",
        ))
    return out


async def _review_one_file(client: OrchestratorClient, *, model: str, repo: Path,
                           path: str, diff_text: str, known: list[Finding],
                           max_turns: int, max_tokens: int, max_diff_chars: int,
                           events: EventLog) -> FileReview:
    """One read-only agent over one file. Never raises: a file that could not be
    reviewed is recorded as such, since a review that quietly skipped part of
    the diff must not look the same as one that found nothing."""
    result = FileReview(path=path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FILE},
        {"role": "user", "content": _build_user_prompt(path, diff_text, known, max_diff_chars)},
    ]
    tools = READ_ONLY_TOOL_SCHEMAS + [SUBMIT_FINDINGS_TOOL]
    events.emit("review_file_start", path=path, known_findings=len(known))

    for turn in range(max_turns):
        forced_final = turn == max_turns - 1
        turn_messages, turn_tools = messages, tools
        tool_choice = "auto"
        if forced_final:
            # See planner.FINAL_TURN_SYSTEM_PROMPT: a system prompt that tells the
            # model to keep reading beats tool_choice outright, so the last turn
            # gets a prompt that only describes submitting, and nothing else to call.
            tool_choice = {"type": "function", "function": {"name": "submit_findings"}}
            turn_messages = [{"role": "system", "content":
                              "Report what you have found in this file by calling "
                              "submit_findings now. You have no other tools and no "
                              "further turns. An empty list is fine."}] + messages[1:]
            turn_tools = [SUBMIT_FINDINGS_TOOL]

        try:
            completion = await client.chat_once(
                model, turn_messages, tools=turn_tools,
                tool_choice=tool_choice, max_tokens=max_tokens,
            )
        except OrchestratorError as e:
            result.stop_reason, result.error = "error", f"orchestrator {e.status_code}: {e.body}"
            events.emit("review_file_error", path=path, status_code=e.status_code)
            return result

        message = completion["choices"][0]["message"]
        usage = completion.get("usage")
        result.turns = turn + 1
        result.prompt_tokens += int((usage or {}).get("prompt_tokens") or 0)
        result.completion_tokens += int((usage or {}).get("completion_tokens") or 0)
        events.emit_usage(
            "review_file", model, usage,
            estimate=None if usage else estimate_usage(turn_messages, message.get("content") or ""),
            path=path, turn=turn + 1,
        )
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user",
                             "content": "Call submit_findings with what you have, or keep "
                                        "reading with read_file/grep/glob/list_dir first."})
            continue

        for call in tool_calls:
            fn = call["function"]
            if fn["name"] == "submit_findings":
                result.findings = _parse_submitted(path, fn.get("arguments"))
                result.stop_reason = "finished"
                events.emit("review_file_done", path=path, findings=len(result.findings),
                            turns=result.turns)
                return result
            try:
                raw = fn.get("arguments")
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            try:
                output = await execute_tool(fn["name"], args, repo, run_shell_timeout_s=0)
            except Exception as e:                      # noqa: BLE001 - reported, never fatal
                output = f"error: {e}"
            events.emit("tool_call", path=path, turn=turn + 1, tool=fn["name"],
                        result_preview=output[:200])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

    result.stop_reason = "max_turns"
    result.error = f"did not submit findings within {max_turns} turns"
    events.emit("review_file_incomplete", path=path, turns=max_turns)
    return result


def _merge_findings(static: Review, file_reviews: list[FileReview]) -> list[Finding]:
    """Static findings plus model findings, minus duplicates.

    A model finding on a line the static pass already flagged is dropped: the
    static one names the exact rule it matched, so it is the more useful of the
    two, and printing both invites the reader to discount all of it.
    """
    merged = list(static.findings)
    taken = {(f.path, f.line) for f in static.findings}
    for review in file_reviews:
        for finding in review.findings:
            if (finding.path, finding.line) in taken:
                continue
            taken.add((finding.path, finding.line))
            merged.append(finding)
    merged.sort(key=lambda f: (f.path, f.line, f.check))
    return merged


async def _summarise(client: OrchestratorClient, *, model: str, findings: list[Finding],
                     files: int, max_tokens: int, events: EventLog) -> str:
    listing = "\n".join(
        f"- {f.path}:{f.line} [{f.severity.value}] {f.check}: {f.message}" for f in findings[:60]
    ) or "(no findings)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
        {"role": "user", "content": f"{files} file(s) reviewed.\n\nFindings:\n{listing}"},
    ]
    try:
        completion = await client.chat_once(model, messages, max_tokens=max_tokens)
    except OrchestratorError as e:
        log.warning("summary call failed (%s); reporting findings without it", e.status_code)
        return ""
    message = completion["choices"][0]["message"]
    usage = completion.get("usage")
    events.emit_usage(
        "review_summary", model, usage,
        estimate=None if usage else estimate_usage(messages, message.get("content") or ""),
    )
    return (message.get("content") or "").strip()


async def run_review(
    client: OrchestratorClient | None,
    *,
    repo: Path,
    model: str = "",
    rev: str | None = None,
    staged: bool = False,
    static_only: bool = False,
    enabled_checks: set[str] | None = None,
    ignored_checks: set[str] | None = None,
    max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
    max_files: int = 20,
    max_turns: int = MAX_AGENT_TURNS,
    max_tokens: int = 8192,
    max_diff_chars: int = 12000,
    concurrency: int = 2,
    summarise: bool = True,
    events: EventLog | None = None,
) -> ReviewRun:
    """Review a diff. Reads only - nothing here writes to `repo`."""
    events = events or NullEventLog()
    started = time.monotonic()

    diff_text = collect_diff(rev=rev, staged=staged, cwd=repo)
    diffs: list[FileDiff] = parse_diff(diff_text)
    static = run_checks(diffs, enabled=enabled_checks, ignored=ignored_checks,
                        max_line_length=max_line_length)
    events.emit("review_static_done", files=static.files_reviewed,
                findings=len(static.findings))

    run = ReviewRun(review=static, static_findings=len(static.findings))

    if static_only or client is None or not model:
        run.review.findings.sort(key=lambda f: (f.path, f.line, f.check))
        run.duration_s = time.monotonic() - started
        return run

    sections = split_diff_sections(diff_text)
    # Reviewable files, most-changed first, so a cap spends the budget where the
    # risk is rather than on whatever git happened to list alphabetically.
    candidates = [d for d in diffs if not d.is_deleted and d.added_lines]
    candidates.sort(key=lambda d: len(d.added_lines), reverse=True)
    if len(candidates) > max_files:
        log.warning("reviewing the %d largest of %d changed files", max_files, len(candidates))
        events.emit("review_files_capped", total=len(candidates), reviewing=max_files)
        candidates = candidates[:max_files]

    by_path: dict[str, list[Finding]] = {}
    for finding in static.findings:
        by_path.setdefault(finding.path, []).append(finding)

    events.emit("review_agents_start", files=len(candidates), concurrency=concurrency)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(diff: FileDiff) -> FileReview:
        async with semaphore:
            return await _review_one_file(
                client, model=model, repo=repo, path=diff.path,
                diff_text=sections.get(diff.path, ""), known=by_path.get(diff.path, []),
                max_turns=max_turns, max_tokens=max_tokens,
                max_diff_chars=max_diff_chars, events=events,
            )

    run.file_reviews = list(await asyncio.gather(*(one(d) for d in candidates)))
    run.model_findings = sum(len(f.findings) for f in run.file_reviews)
    run.review.findings = _merge_findings(static, run.file_reviews)

    if summarise:
        run.review.summary = await _summarise(
            client, model=model, findings=run.review.findings,
            files=static.files_reviewed, max_tokens=MAX_SUMMARY_TOKENS, events=events,
        )

    for fr in run.file_reviews:
        run.prompt_tokens += fr.prompt_tokens
        run.completion_tokens += fr.completion_tokens
    run.duration_s = time.monotonic() - started
    events.emit("review_end", findings=len(run.review.findings),
                static=run.static_findings, model=run.model_findings,
                unreviewed=len(run.unreviewed), duration_s=round(run.duration_s, 1))
    return run


async def review_session(
    *,
    repo: Path,
    model: str,
    orchestrator_url: str,
    admin_url: str,
    api_key: str | None,
    pipeline_cfg: PipelineCfg,
    load_wait_s: float = 180.0,
    ensure_resident: bool = True,
    static_only: bool = False,
    **kwargs,
) -> ReviewRun:
    """Wraps run_review with client setup and residency."""
    if static_only:
        return await run_review(None, repo=repo, static_only=True, **kwargs)
    async with OrchestratorClient(orchestrator_url, admin_url, api_key) as client:
        if ensure_resident:
            from ..solo import ensure_model_resident
            await ensure_model_resident(client, model, load_wait_s)
        return await run_review(client, repo=repo, model=model, **kwargs)
